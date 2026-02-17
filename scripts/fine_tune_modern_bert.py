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

def main():
    ds = load_dataset("csv", data_files={"data": CSV_PATH})["data"]

    # Identify label columns: all numeric columns except known metadata
    meta_cols = {
        "text", "id", "author", "subreddit", "link_id", "parent_id",
        "created_utc", "rater_id", "example_very_unclear"
    }
    label_cols = [c for c in ds.column_names if c not in meta_cols and c != "text"]

    if "text" not in ds.column_names:
        raise ValueError("Expected a 'text' column in the CSV.")

    if len(label_cols) == 0:
        raise ValueError("No label columns found. Check your CSV header.")

    # 80/10/10 split
    ds = ds.shuffle(seed=42)
    split1 = ds.train_test_split(test_size=0.2, seed=42)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
    train_ds = split1["train"]
    val_ds = split2["train"]
    test_ds = split2["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

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
        fp16=False,           # set False if you’re not on CUDA
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
    print(trainer.evaluate(test_ds))

    trainer.save_model("outputs/modernbert_filtered_emotions/final")
    tokenizer.save_pretrained("outputs/modernbert_filtered_emotions/final")

if __name__ == "__main__":
    main()
