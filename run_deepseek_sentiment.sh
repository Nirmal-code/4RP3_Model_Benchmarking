#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Put HF cache somewhere with space
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM=false

MODEL_ID="${MODEL_ID:-deepseek-ai/deepseek-coder-6.7b-instruct}"
INPUT_CSV="${1:-data/filtered_emotions.csv}"
OUTPUT_CSV="${2:-outputs/deepseek_preds.csv}"

mkdir -p "$(dirname "$OUTPUT_CSV")"

python3 scripts/deepseek_sentiment_infer.py \
  --model_id "$MODEL_ID" \
  --input_csv "$INPUT_CSV" \
  --output_csv "$OUTPUT_CSV" \
  --batch_size 16 \
  --max_input_tokens 256 \
  --max_new_tokens 16
