#!/bin/bash

# Portuguese d8 pretraining run.
# Trains a depth-8 base model for 3,333 * 524,288 = 1,747,451,904 tokens (~1,747.6M).
# Run as:
# bash runs/runptd8.sh
# Optional overrides:
# NPROC_PER_NODE=4 DEVICE_BATCH_SIZE=16 WANDB_RUN=my_run MODEL_TAG=my_tag bash runs/runptd8.sh
# USE_FP8=1 bash runs/runptd8.sh  # H100+ only

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-pt-d8}"
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
TOTAL_BATCH_SIZE=524288
NUM_ITERATIONS=3333
TARGET_PARAM_DATA_RATIO=41.66231
WANDB_RUN="${WANDB_RUN:-d8_pt_1748m}"
MODEL_TAG="${MODEL_TAG:-pt-d8-1748m}"
TRAIN_SHARDS="${TRAIN_SHARDS:-32}"

TRAIN_EXTRA_ARGS=()
if [ "${USE_FP8:-0}" = "1" ]; then
    TRAIN_EXTRA_ARGS+=(--fp8)
fi

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

# Base model pretraining only. The target ratio is set to the actual d8 token/scaling-param ratio
# so LR/weight-decay scaling sees the intended horizon even though num_iterations fixes it exactly.
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth=8 \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$NUM_ITERATIONS" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --core-metric-name=ptcore \
    --core-metric-every=1000 \
    --eval-tokens=4194304 \
    --sample-every=1000 \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN" \
    "${TRAIN_EXTRA_ARGS[@]}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --ptcore-split=val \
    --split-tokens=4194304 \
    --eval ptcore,bpb,sample

python -m nanochat.report generate
