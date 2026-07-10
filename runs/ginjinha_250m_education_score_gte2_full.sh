#!/bin/bash
set -euo pipefail

# Train the d11 Portuguese base model on enough score >=2 data to cover the full filtered corpus.
# The estimated filtered corpus is 3.3-3.8B tokens; ratio 40 trains on 4.12B tokens.
# Validation remains unfiltered.
#
# Run:
#   bash runs/ginjinha_250m_education_score_gte2_full.sh

# Run identity and storage
RUN_NAME="ginjinha_d11_ratio40_education_score_gte2_full_corpus"
MODEL_TAG="$RUN_NAME"
WANDB_RUN="$RUN_NAME"
HF_BUCKET="duarteocarmo/ginjinha"

# Model and data
DEPTH="11"
TARGET_PARAM_DATA_RATIO="40"
MIN_EDUCATIONAL_SCORE="2"
DEVICE_BATCH_SIZE="32"
TOKENIZER_SHARDS="8"
TRAIN_SHARDS="359"

# Evaluation and checkpoints
EVAL_EVERY="1000"
CORE_METRIC_EVERY="1000"
CORE_MAX_PER_TASK="500"
SAMPLE_EVERY="1000"
SAVE_EVERY="1000"
BASE_EVAL_MODES="core,bpb"

# Runtime and derived paths
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
HF_RUN_URI="hf://buckets/$HF_BUCKET/$WANDB_RUN"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
mkdir -p "$NANOCHAT_BASE_DIR"

NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [ "$NPROC_PER_NODE" = "0" ]; then
    NPROC_PER_NODE=1
fi
echo "Using $NPROC_PER_NODE GPU process(es)"

FP8_ARG=""
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if echo "$GPU_NAMES" | grep -q "H100"; then
    FP8_ARG="--fp8"
    echo "FP8 enabled"
else
    echo "FP8 disabled"
fi

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
uvx hf auth whoami > /dev/null 2>&1 || uvx hf auth login
uvx hf buckets info "$HF_BUCKET" > /dev/null

# Train an unfiltered tokenizer while the full dataset downloads.
python -m nanochat.dataset -n "$TOKENIZER_SHARDS"
python -m nanochat.dataset -n "$TRAIN_SHARDS" &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

wait "$DATASET_DOWNLOAD_PID"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --model-tag="$MODEL_TAG" \
    --depth="$DEPTH" \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --min-educational-score="$MIN_EDUCATIONAL_SCORE" \
    --eval-every="$EVAL_EVERY" \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --core-metric-max-per-task="$CORE_MAX_PER_TASK" \
    --sample-every="$SAMPLE_EVERY" \
    --save-every="$SAVE_EVERY" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    $FP8_ARG \
    --run="$WANDB_RUN"

uvx hf buckets sync "$CHECKPOINT_DIR" "$HF_RUN_URI/checkpoints" --exclude "optim_*.pt"
uvx hf buckets sync "$NANOCHAT_BASE_DIR/tokenizer" "$HF_RUN_URI/tokenizer"
echo "Uploaded model checkpoints and tokenizer to $HF_RUN_URI"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --eval="$BASE_EVAL_MODES" \
    --max-per-task="$CORE_MAX_PER_TASK" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN"

FINAL_MODEL_PATH=$(find "$CHECKPOINT_DIR" -name "model_*.pt" | sort | tail -1)
FINAL_STEP=$(basename "$FINAL_MODEL_PATH" .pt | cut -d_ -f2)
EVAL_CSV="$NANOCHAT_BASE_DIR/base_eval/base_model_${FINAL_STEP}.csv"
uvx hf buckets cp "$EVAL_CSV" "$HF_RUN_URI/eval/"
echo "Uploaded evaluation CSV to $HF_RUN_URI/eval/"
