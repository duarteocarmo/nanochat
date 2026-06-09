#!/bin/bash

# Chinchilla-style Portuguese d10 pretraining run.
# Trains a depth-10 base model at ~20 train tokens per scaling parameter:
# 2,675 * 524,288 = 1,402,470,400 tokens (~1,402.5M).
# Run as:
# bash runs/runptd10-chinchilla.sh

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-pt-d10-chinchilla"
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC_PER_NODE=1
DEVICE_BATCH_SIZE=32
TOTAL_BATCH_SIZE=524288
TARGET_PARAM_DATA_RATIO=20
CORE_METRIC_EVERY=500
SAMPLE_EVERY=500
WANDB_RUN=d10_pt_chinchilla
MODEL_TAG=pt-d10-chinchilla
TRAIN_SHARDS=32

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.report reset

# Tokenizer: train on the first 8 Bagaço shards, matching the PT tokenizer setup.
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n "$TRAIN_SHARDS" &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
python -m scripts.tok_eval

wait $DATASET_DOWNLOAD_PID

# Base model pretraining only. PTCORE is evaluated during training and at the end.
# Final base_eval also prints conditioned and unconditioned samples.
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth=10 \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --core-metric-name=ptcore \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --core-metric-max-per-task=-1 \
    --ptcore-split=val \
    --eval-tokens=4194304 \
    --sample-every="$SAMPLE_EVERY" \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --ptcore-split=val \
    --split-tokens=4194304 \
    --eval ptcore,bpb,sample

python -m nanochat.report generate
