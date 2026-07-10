#!/bin/bash
set -euo pipefail

# Compare fixed-compute d8 pretraining runs on all data, score >= 1, and score >= 2.
# Validation remains unfiltered for every run.
#
# Run:
#   bash runs/educational_score_ablation.sh
#

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

MODEL_TAG_PREFIX="ginjinha"
WANDB_RUN_PREFIX="ginjinha"
TARGET_PARAM_DATA_RATIO="30"
NUM_TRAIN_SHARDS="170"
DEVICE_BATCH_SIZE="128"
CORE_MAX_PER_TASK="500"

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

# Train one shared, unfiltered tokenizer while the remaining shards download.
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n "$NUM_TRAIN_SHARDS" &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

run_experiment() {
    local education_filter="$1"
    local minimum_educational_score="$2"
    local experiment_name="d8_ratio${TARGET_PARAM_DATA_RATIO}_education_${education_filter}"
    local model_tag="${MODEL_TAG_PREFIX}_${experiment_name}"
    local wandb_run="${WANDB_RUN_PREFIX}_${experiment_name}"

    echo "Running $model_tag with minimum educational score $minimum_educational_score"
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
        --model-tag="$model_tag" \
        --depth=8 \
        --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
        --min-educational-score="$minimum_educational_score" \
        --eval-every=1000 \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        $FP8_ARG \
        --run="$wandb_run"

    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
        --model-tag="$model_tag" \
        --eval=core,bpb \
        --max-per-task="$CORE_MAX_PER_TASK" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --run="$wandb_run"
}

run_experiment "all_scores" -1
run_experiment "score_gte1" 1
run_experiment "score_gte2" 2
