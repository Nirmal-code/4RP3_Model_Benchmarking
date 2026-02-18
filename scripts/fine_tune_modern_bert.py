import re
from ast import literal_eval
from pathlib import Path
import csv

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score

CSV_PATH = "data/filtered_emotions.csv"   # put the file next to this script, or change the path
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 256
# Only these emotions will be used as labels (must match columns in the CSV)
TARGET_EMOTIONS = ["anger", "disgust", "fear", "sadness", "joy", "love", "surprise", "optimism"]


def save_learning_curves(log_history, output_dir: Path):
    """Persist training and eval metrics to CSV and PNG (if matplotlib is available)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write compact CSV for later plotting
    csv_path = output_dir / "training_logs.csv"
    fieldnames = ["step", "loss", "eval_loss", "eval_f1_micro", "eval_exact_match"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_history:
            if any(k in row for k in fieldnames):
                writer.writerow({k: row.get(k) for k in fieldnames})

    # Optionally generate PNG curves
    try:
        import matplotlib.pyplot as plt

        train_pts = [(r.get("step"), r.get("loss")) for r in log_history if "loss" in r]
        eval_pts = [(r.get("step"), r.get("eval_loss")) for r in log_history if "eval_loss" in r]
        if train_pts or eval_pts:
            plt.figure(figsize=(6, 4))
            if train_pts:
                plt.plot([p[0] for p in train_pts], [p[1] for p in train_pts], label="train loss")
            if eval_pts:
                plt.plot([p[0] for p in eval_pts], [p[1] for p in eval_pts], label="val loss")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("ModernBERT fine-tuning")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "loss_curves.png", dpi=150)
            plt.close()
    except Exception as exc:  # matplotlib may be missing on the server
        print(f"Skipping curve plot ({exc}); CSV saved to {csv_path}")


def main():
    ds = load_dataset("csv", data_files={"data": CSV_PATH})["data"]
    output_dir = Path("outputs/modernbert_filtered_emotions")

    # Identify label columns: all numeric columns except known metadata
    meta_cols = {
        "text", "id", "author", "subreddit", "link_id", "parent_id",
        "created_utc", "rater_id", "example_very_unclear",
        # optional columns we do not want to treat as labels
        "annotated_emotion", "annotated_emotions", "applicable_emotion", "applicable_emotions",
    }
    label_cols = [c for c in ds.column_names if c not in meta_cols and c != "text"]
    if TARGET_EMOTIONS:
        label_cols = [c for c in label_cols if c in TARGET_EMOTIONS]

    if "text" not in ds.column_names:
        raise ValueError("Expected a 'text' column in the CSV.")

    if len(label_cols) == 0:
        raise ValueError("No label columns found. Check your CSV header.")

    annotated_col = next((c for c in ["annotated_emotion", "annotated_emotions"] if c in ds.column_names), None)
    applicable_col = next((c for c in ["applicable_emotion", "applicable_emotions"] if c in ds.column_names), None)

    def _to_list(val):
        """Normalize strings/lists into a list of emotion names."""
        if val is None:
            return []
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
        if isinstance(val, str):
            txt = val.strip()
            if txt.startswith("[") and txt.endswith("]"):
                try:
                    parsed = literal_eval(txt)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    pass
            return [p.strip() for p in re.split(r"[;,]", txt) if p.strip()]
        return []

    # Keep only rows whose annotated/applicable emotion is one of TARGET_EMOTIONS
    filter_col = annotated_col or applicable_col
    if filter_col:
        before_len = len(ds)

        def keep_example(example):
            emotions = _to_list(example.get(filter_col))
            return any(e in TARGET_EMOTIONS for e in emotions)

        ds = ds.filter(keep_example)
        after_len = len(ds)
        if after_len == 0:
            raise ValueError("Filtering by target emotions removed all rows.")
        print(f"Filtered out {before_len - after_len} rows not in TARGET_EMOTIONS via {filter_col}.")
    else:
        print("No annotated/applicable emotion column found; skipping row-level filtering.")

    def _is_positive(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            return value.strip() not in {"", "0", "false", "False"}
        return False

    # Build label matrix for splitting (multi-label one-hot)
    def build_label_matrix(dataset):
        rows = []
        for ex in dataset:
            rows.append([1 if _is_positive(ex[c]) else 0 for c in label_cols])
        return np.array(rows, dtype=np.int8)

    def iterative_split(y, test_size, seed):
        """
        Lightweight iterative stratification for multilabel data.
        Tries to keep per-emotion prevalence consistent in the test split.
        Returns train_idx, test_idx.
        """
        rng = np.random.default_rng(seed)
        n_samples, n_labels = y.shape
        remaining = list(range(n_samples))
        desired_test = y.sum(axis=0) * float(test_size)
        test_counts = np.zeros(n_labels, dtype=float)
        train_idx, test_idx = [], []

        while remaining:
            remaining_mat = y[remaining]
            label_remaining = remaining_mat.sum(axis=0)
            if label_remaining.max() == 0:
                # No positives left; fallback to size-based split for the rest
                rng.shuffle(remaining)
                cutoff = int(len(remaining) * test_size)
                test_idx.extend(remaining[:cutoff])
                train_idx.extend(remaining[cutoff:])
                break

            target_label = int(np.argmin(np.where(label_remaining > 0, label_remaining, np.inf)))
            candidates = [idx for idx in remaining if y[idx, target_label] == 1]
            rng.shuffle(candidates)

            for idx in candidates:
                label_vec = y[idx]
                deficits = desired_test - test_counts
                # Prefer putting the sample in test if it fills any deficit and we still need test size
                place_in_test = deficits[label_vec == 1].sum() > 0 and (
                    len(test_idx) / max(len(train_idx) + len(test_idx), 1) < test_size
                )
                if place_in_test:
                    test_idx.append(idx)
                    test_counts += label_vec
                else:
                    train_idx.append(idx)
                remaining.remove(idx)

        return np.array(sorted(train_idx)), np.array(sorted(test_idx))

    def three_way_split(dataset, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
        assert abs(train_frac + val_frac + test_frac - 1) < 1e-6
        y_full = build_label_matrix(dataset)

        train_idx, temp_idx = iterative_split(y_full, test_size=1 - train_frac, seed=seed)
        temp_ds = dataset.select(temp_idx.tolist())
        y_temp = build_label_matrix(temp_ds)

        # split remaining into val/test
        val_ratio = test_frac / float(val_frac + test_frac)
        val_idx_rel, test_idx_rel = iterative_split(y_temp, test_size=val_ratio, seed=seed + 1)

        val_idx = temp_idx[val_idx_rel]
        test_idx = temp_idx[test_idx_rel]

        return (
            dataset.select(train_idx.tolist()),
            dataset.select(val_idx.tolist()),
            dataset.select(test_idx.tolist()),
        )

    train_ds, val_ds, test_ds = three_way_split(ds)
    # Save raw test split for later offline evaluation
    output_dir.mkdir(parents=True, exist_ok=True)
    test_csv_path = output_dir / "test.csv"
    test_ds.to_csv(test_csv_path)
    print(f"Saved test split to {test_csv_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Persist label ordering for inference
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.txt"
    labels_path.write_text("\n".join(label_cols), encoding="utf-8")

    id2label = {i: lbl for i, lbl in enumerate(label_cols)}
    label2id = {lbl: i for i, lbl in enumerate(label_cols)}

    def preprocess(batch):
        tok = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
        # Build multi-hot label vectors (float for BCEWithLogitsLoss)
        labels = []
        for i in range(len(batch["text"])):
            labels.append([float(batch[c][i]) for c in label_cols])
        tok["labels"] = labels
        return tok

    train_ds = train_ds.map(preprocess, batched=True)
    val_ds = val_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_cols),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        labels = labels.astype(int)

        # Micro-F1 is standard for multi-label
        f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
        exact_match = (preds == labels).all(axis=1).mean().item()
        return {"f1_micro": f1_micro, "exact_match": exact_match}

    args = TrainingArguments(
        output_dir="outputs/modernbert_filtered_emotions",
        learning_rate=5e-6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        logging_steps=50,
        fp16=True,           # set False if you’re not on CUDA
        bf16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Label columns:", label_cols)
    trainer.train()

    # Save training/validation loss curves for quick inspection
    save_learning_curves(trainer.state.log_history, output_dir)

    print(trainer.evaluate(test_ds))

    trainer.save_model("outputs/modernbert_filtered_emotions/final")
    tokenizer.save_pretrained("outputs/modernbert_filtered_emotions/final")

if __name__ == "__main__":
    main()
