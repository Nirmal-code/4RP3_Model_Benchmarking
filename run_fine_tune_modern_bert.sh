#!/usr/bin/env bash
set -euo pipefail

# Default to the 3rd GPU (index 2). Override with GPU_ID if needed.
GPU_ID="${GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Put HF cache somewhere with space
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM=false

# Disable torch.compile/inductor to avoid building C++/Triton extensions (fails without Python headers)
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TRITON_DISABLE=1

python3 scripts/fine_tune_modern_bert.py
