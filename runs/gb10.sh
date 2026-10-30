#!/bin/bash
set -euo pipefail

MODEL_TAG="${MODEL_TAG:-gb10_d6}"
STEPS="${STEPS:-64}"
DEVICE_BATCH_SIZE=8
SEQ_LEN=512
TOTAL_BATCH_SIZE=16384
EVAL_TOKENS=8192
CORE_MAX_PER_TASK=4
NPROC_PER_NODE=1

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-gb10-bagaco2}"
mkdir -p "$NANOCHAT_BASE_DIR"

if [ -e "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" ]; then
    echo "Checkpoint directory exists. Choose another MODEL_TAG." >&2
    exit 1
fi

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.dataset -n 1

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] || [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]; then
    python -m scripts.tok_train --max-chars=20000000 --doc-cap=2000 --vocab-size=32768
fi
python -m scripts.tok_eval

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train \
    --device-type=cuda --model-tag="$MODEL_TAG" \
    --depth=6 --head-dim=128 --window-pattern=L \
    --max-seq-len="$SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$STEPS" --warmup-steps=4 \
    --seed=42 --data-seed=42 \
    --eval-every="$STEPS" --eval-tokens="$EVAL_TOKENS" \
    --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
    --run=dummy

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval \
    --device-type=cuda --model-tag="$MODEL_TAG" --step="$STEPS" \
    --eval=bpb,core,core-gen,sample \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --split-tokens="$EVAL_TOKENS" --max-per-task="$CORE_MAX_PER_TASK"
