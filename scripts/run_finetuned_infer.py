import argparse
import csv
import os
from typing import List
from pathlib import Path

# Disable Torch compile/Inductor/Triton to avoid building C/CUDA extensions (Python.h missing on server)
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRITON_DISABLE", "1")

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


META_COLS = {
    "text",
    "id",
    "author",
    "subreddit",
    "link_id",
    "parent_id",
    "created_utc",
    "rater_id",
    "example_very_unclear",
}


def chunk_iter(xs: List, size: int):
    for i in range(0, len(xs), size):
        yield xs[i : i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/modernbert_filtered_emotions/final")
    ap.add_argument("--input_csv", default="data/filtered_emotions.csv")
    ap.add_argument("--output_csv", default="outputs/modernbert_filtered_emotions/pred_sentiment.csv")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.text_field not in df.columns:
        raise ValueError(f"Missing text field '{args.text_field}' in input CSV")

    model_dir = Path(args.model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # Prefer label order saved during training; fallback to config; then CSV
    labels_path = model_dir / "labels.txt"
    if labels_path.exists():
        label_cols = labels_path.read_text(encoding="utf-8").splitlines()
    else:
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        if getattr(cfg, "id2label", None):
            label_cols = [cfg.id2label[i] for i in range(cfg.num_labels)]
        else:
            label_cols = [c for c in df.columns if c not in META_COLS and c != args.text_field]
            if not label_cols:
                raise ValueError("Could not infer label columns.")

    # Load model using saved head shape
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        trust_remote_code=True,
    ).to(device)
    model.config.problem_type = "multi_label_classification"
    if len(label_cols) != model.config.num_labels:
        raise ValueError(
            f"Label count mismatch: labels.txt/config has {model.config.num_labels}, "
            f"but inferred {len(label_cols)} labels ({label_cols}). "
            "Ensure you're using the labels saved with the trained model."
        )
    model.eval()

    # Match deepseek_preds.csv shape: id, text, pred_sentiment (no raw_output).
    fieldnames = [args.id_field, args.text_field, "pred_sentiment"]
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        texts = df[args.text_field].fillna("").tolist()
        ids = df[args.id_field].tolist() if args.id_field in df.columns else [None] * len(df)

        for idx_chunk in chunk_iter(list(range(len(df))), args.batch_size):
            batch_texts = [texts[i] for i in idx_chunk]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)

            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.sigmoid(logits).cpu().numpy()

            for row_idx, prob_row in zip(idx_chunk, probs):
                best_idx = int(prob_row.argmax())
                pred_sentiment = label_cols[best_idx]
                writer.writerow(
                    {
                        args.id_field: ids[row_idx],
                        args.text_field: texts[row_idx],
                        "pred_sentiment": pred_sentiment,
                    }
                )


if __name__ == "__main__":
    main()
