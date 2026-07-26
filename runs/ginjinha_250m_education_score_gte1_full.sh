#!/bin/bash
set -euo pipefail

# Train the d11 Portuguese base model on enough score >=1 data to cover the full filtered corpus.
# The estimated effective filtered corpus is 9.3B tokens; ratio 91 trains on 9.37B tokens.
# Validation remains unfiltered.
#
# Run:
#   bash runs/ginjinha_250m_education_score_gte1_full.sh

# Run identity and storage
RUN_NAME="ginjinha_d11_ratio91_ptcore5_education_score_gte1"
MODEL_TAG="$RUN_NAME"
WANDB_RUN="$RUN_NAME"
HF_MODEL_REPO="duarteocarmo/ginjinha"

# Model and data
DEPTH="11"
TARGET_PARAM_DATA_RATIO="91"
MIN_EDUCATIONAL_SCORE="1"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
TOKENIZER_SHARDS="8"
TRAIN_SHARDS="359"

# Evaluation and checkpoints
EVAL_EVERY="1000"
CORE_METRIC_EVERY="1000"
CORE_MAX_PER_TASK="-1"
SAMPLE_EVERY="1000"
SAVE_EVERY="1000"
BASE_EVAL_MODES="core,bpb"

# Runtime and derived paths
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
mkdir -p "$NANOCHAT_BASE_DIR"

if [ -z "${NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi
if [ "$NPROC_PER_NODE" = "0" ]; then
    NPROC_PER_NODE=1
fi
echo "Using $NPROC_PER_NODE GPU process(es), device batch size $DEVICE_BATCH_SIZE"

FP8_ARG=""
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if echo "$GPU_NAMES" | grep -qE "H100|H200"; then
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
uvx hf models info "$HF_MODEL_REPO" > /dev/null

# Train an unfiltered tokenizer while the full dataset downloads.
python -m nanochat.dataset -n "$TOKENIZER_SHARDS"
python -m nanochat.dataset -n "$TRAIN_SHARDS" & DATASET_DOWNLOAD_PID=$!
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

uvx hf upload "$HF_MODEL_REPO" "$CHECKPOINT_DIR" "$MODEL_TAG/checkpoints" --repo-type model --commit-message "Add $MODEL_TAG resumable base checkpoint"
uvx hf upload "$HF_MODEL_REPO" "$NANOCHAT_BASE_DIR/tokenizer" "$MODEL_TAG/tokenizer" --repo-type model --commit-message "Add $MODEL_TAG tokenizer"
echo "Uploaded resumable $MODEL_TAG checkpoint and tokenizer to hf://$HF_MODEL_REPO/$MODEL_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --eval="$BASE_EVAL_MODES" \
    --max-per-task="$CORE_MAX_PER_TASK" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN"

FINAL_MODEL_PATH=$(find "$CHECKPOINT_DIR" -name "model_*.pt" | sort | tail -1)
if [ -z "$FINAL_MODEL_PATH" ]; then
    echo "No model checkpoint found in $CHECKPOINT_DIR" >&2
    exit 1
fi
FINAL_STEP=$(basename "$FINAL_MODEL_PATH" .pt | cut -d_ -f2)
EVAL_CSV="$NANOCHAT_BASE_DIR/base_eval/base_model_${FINAL_STEP}.csv"
if [ ! -f "$EVAL_CSV" ]; then
    echo "Evaluation CSV not found at $EVAL_CSV" >&2
    exit 1
fi
for rank in $(seq 0 $((NPROC_PER_NODE - 1))); do
    if [ ! -f "$CHECKPOINT_DIR/optim_${FINAL_STEP}_rank${rank}.pt" ]; then
        echo "Optimizer checkpoint not found for rank $rank at step $FINAL_STEP" >&2
        exit 1
    fi
done
uvx hf upload "$HF_MODEL_REPO" "$EVAL_CSV" "$MODEL_TAG/eval/$(basename "$EVAL_CSV")" --repo-type model --commit-message "Add $MODEL_TAG evaluation"
echo "Uploaded evaluation CSV to hf://$HF_MODEL_REPO/$MODEL_TAG/eval/"
