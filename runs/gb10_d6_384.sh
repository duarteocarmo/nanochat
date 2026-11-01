#!/bin/bash
set -euo pipefail

MODEL_TAG="${1:?Usage: $0 EXPERIMENT_NAME}"
WANDB_RUN="$MODEL_TAG"
STEPS="${STEPS:-2400}"
DEVICE_BATCH_SIZE=32
SEQ_LEN=2048
TOTAL_BATCH_SIZE=65536
EVAL_TOKENS=1048576
CORE_MAX_PER_TASK=-1
NPROC_PER_NODE=1

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
# Bagaço3 is already downloaded here; no dataset download is needed.
export NANOCHAT_DATASET=bagaco3
export NANOCHAT_DATA_DIR="$HOME/data/base_data_bagaco3"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$REPO_DIR/dev-ignore}"
mkdir -p "$NANOCHAT_BASE_DIR"

if [ -e "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" ]; then
    echo "Checkpoint directory exists. Choose another MODEL_TAG." >&2
    exit 1
fi

source .venv/bin/activate

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] || [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]; then
    python -m scripts.tok_train --max-chars=20000000 --doc-cap=2000 --vocab-size=32768
fi
python -m scripts.tok_eval

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m -- scripts.base_train \
    --device-type=cuda --model-tag="$MODEL_TAG" \
    --depth=6 --head-dim=128 --window-pattern=L \
    --max-seq-len="$SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$STEPS" --warmup-steps=40 \
    --seed=42 --data-seed=42 \
    --eval-every=400 --eval-tokens="$EVAL_TOKENS" \
    --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
    --wandb-project=ginja --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m -- scripts.base_eval \
    --device-type=cuda --model-tag="$MODEL_TAG" --step="$STEPS" \
    --eval=bpb,core,core-gen,sample \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --split-tokens="$EVAL_TOKENS" --max-per-task="$CORE_MAX_PER_TASK" \
    --run="$WANDB_RUN"
