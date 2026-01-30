import argparse
import csv
import json
import re
from typing import List, Dict, Any, Iterable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SENTIMENT_LABELS = ["anger", "disgust", "fear", "sadness", "joy", "love", "surprise", "optimism"]


def build_prompt(text: str) -> str:
    # Keep the output constrained to one of the allowed labels to reduce drift.
    return (
        "Task: Sentiment classification.\n"
        "Labels: anger, disgust, fear, sadness, joy, love, surprise, optimism.\n"
        "Return ONLY valid JSON like {\"sentiment\":\"<label>\"}.\n\n"
        f"Text: {text}\n"
    )


def extract_json_label(generation: str) -> str:
    """
    Best-effort parse:
    - find last {...} block
    - parse sentiment field if possible
    - otherwise fall back to keyword search
    """
    # Try to find a JSON object
    m = re.findall(r"\{.*?\}", generation, flags=re.DOTALL)
    if m:
        candidate = m[-1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "sentiment" in obj:
                s = str(obj["sentiment"]).strip().lower()
                if s in SENTIMENT_LABELS:
                    return s
        except Exception:
            pass

    # Fallback: keyword search
    low = generation.lower()
    for lab in SENTIMENT_LABELS:
        if lab in low:
            return lab
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_input_tokens", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        # Many decoder LLMs don't define a pad token; use EOS for padding.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    def batch_iter(rows: Iterable[Dict[str, Any]], bs: int):
        batch: List[Dict[str, Any]] = []
        for row in rows:
            batch.append(row)
            if len(batch) >= bs:
                yield batch
                batch = []
        if batch:
            yield batch

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in, open(
        args.output_csv, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(
            f_out,
            fieldnames=["id", "text", "pred_sentiment", "raw_output"],
        )
        writer.writeheader()

        for batch in batch_iter(reader, args.batch_size):
            prompts = [
                build_prompt((row.get(args.text_field) or ""))
                for row in batch
            ]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for row, full_text in zip(batch, decoded):
                pred = extract_json_label(full_text)
                writer.writerow(
                    {
                        "id": row.get(args.id_field),
                        "text": row.get(args.text_field),
                        "pred_sentiment": pred,
                        "raw_output": full_text[-2000:],  # keep last chunk for debugging
                    }
                )


if __name__ == "__main__":
    main()
