#!/bin/bash

# Train a ~250M parameter Portuguese base model on the Bagaço2 pretraining stack.
# This follows speedrun.sh, but stops after base_eval: tokenizer -> pretrain -> PTCORE/BPB/sample eval.
#
# Run:
#   bash runs/ginjinha_250m_chinchilla.sh
#
# With wandb:
#   WANDB_RUN=ginjinha_250m_chinchilla bash runs/ginjinha_250m_chinchilla.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

MODEL_TAG="${MODEL_TAG:-ginjinha_250m_chinchilla}"
WANDB_RUN="${WANDB_RUN:-dummy}"
if [ -z "${NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NPROC_PER_NODE" = "0" ]; then
        NPROC_PER_NODE=1
    fi
fi
echo "Using $NPROC_PER_NODE GPU process(es)"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B chars for tokenizer training.
python -m nanochat.dataset -n 8

# Download enough Bagaço2 shards for the 250M Chinchilla run while tokenizer trains.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model pretraining

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# ~280M params with default width/head settings. Chinchilla target: 20 tokens per parameter.
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --model-tag="$MODEL_TAG" \
    --depth=11 \
    --target-param-data-ratio=20 \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --fp8 \
    --run="$WANDB_RUN"

# Evaluate: PTCORE, BPB on train/val, and samples.
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE"
