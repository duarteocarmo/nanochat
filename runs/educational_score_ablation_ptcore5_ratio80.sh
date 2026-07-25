#!/bin/bash
set -euo pipefail

# Compare fixed-compute d8 pretraining runs on all data and scores >= 1 and >= 2.
# This doubles the previous data:param ratio from 40 to 80.
# Validation remains unfiltered and the final evaluation uses every row of the five-task PTCORE.
# Final optimizer shards are uploaded with each checkpoint to permit exact resumption.
#
# Run:
#   bash runs/educational_score_ablation_ptcore5_ratio80.sh
#

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

MODEL_TAG_PREFIX="ginjinha"
WANDB_RUN_PREFIX="ginjinha"
HF_MODEL_REPO="${HF_MODEL_REPO:-duarteocarmo/ginjinha}"
TARGET_PARAM_DATA_RATIO="80"
NUM_TRAIN_SHARDS="340"
MAX_SEQ_LEN="2048"
TOTAL_BATCH_SIZE="524288"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-}"
CORE_MAX_PER_TASK="-1"

NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [ "$NPROC_PER_NODE" = "0" ]; then
    NPROC_PER_NODE=1
fi
if [ -z "$DEVICE_BATCH_SIZE" ]; then
    if [ $((256 % NPROC_PER_NODE)) -ne 0 ]; then
        echo "Set DEVICE_BATCH_SIZE explicitly for $NPROC_PER_NODE processes" >&2
        exit 1
    fi
    DEVICE_BATCH_SIZE=$((256 / NPROC_PER_NODE))
    if [ "$DEVICE_BATCH_SIZE" -gt 128 ]; then
        DEVICE_BATCH_SIZE=128
    fi
fi
WORLD_TOKENS_PER_MICRO_BATCH=$((NPROC_PER_NODE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
if [ $((TOTAL_BATCH_SIZE % WORLD_TOKENS_PER_MICRO_BATCH)) -ne 0 ]; then
    echo "TOTAL_BATCH_SIZE must be divisible by NPROC_PER_NODE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN" >&2
    exit 1
fi
echo "Using $NPROC_PER_NODE GPU process(es), device batch size $DEVICE_BATCH_SIZE, gradient accumulation $((TOTAL_BATCH_SIZE / WORLD_TOKENS_PER_MICRO_BATCH))"

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
uvx hf models info "$HF_MODEL_REPO" > /dev/null

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
    local experiment_name="d8_ratio${TARGET_PARAM_DATA_RATIO}_ptcore5_education_${education_filter}"
    local model_tag="${MODEL_TAG_PREFIX}_${experiment_name}"
    local wandb_run="${WANDB_RUN_PREFIX}_${experiment_name}"
    local checkpoint_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$model_tag"

    echo "Running $model_tag with minimum educational score $minimum_educational_score"
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
        --model-tag="$model_tag" \
        --depth=8 \
        --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
        --min-educational-score="$minimum_educational_score" \
        --max-seq-len="$MAX_SEQ_LEN" \
        --total-batch-size="$TOTAL_BATCH_SIZE" \
        --eval-every=1000 \
        --core-metric-every=1000 \
        --core-metric-max-per-task=-1 \
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

    local final_model_path
    local final_step
    local eval_csv
    final_model_path=$(find "$checkpoint_dir" -name "model_*.pt" | sort | tail -1)
    if [ -z "$final_model_path" ]; then
        echo "No model checkpoint found in $checkpoint_dir" >&2
        exit 1
    fi
    final_step=$(basename "$final_model_path" .pt | cut -d_ -f2)
    eval_csv="$NANOCHAT_BASE_DIR/base_eval/base_model_${final_step}.csv"
    if [ ! -f "$eval_csv" ]; then
        echo "Evaluation CSV not found at $eval_csv" >&2
        exit 1
    fi
    for rank in $(seq 0 $((NPROC_PER_NODE - 1))); do
        if [ ! -f "$checkpoint_dir/optim_${final_step}_rank${rank}.pt" ]; then
            echo "Optimizer checkpoint not found for rank $rank at step $final_step" >&2
            exit 1
        fi
    done
    uvx hf upload "$HF_MODEL_REPO" "$checkpoint_dir" "$model_tag/checkpoints" --repo-type model --commit-message "Add $model_tag resumable base checkpoint"
    uvx hf upload "$HF_MODEL_REPO" "$NANOCHAT_BASE_DIR/tokenizer" "$model_tag/tokenizer" --repo-type model --commit-message "Add $model_tag tokenizer"
    uvx hf upload "$HF_MODEL_REPO" "$eval_csv" "$model_tag/eval/$(basename "$eval_csv")" --repo-type model --commit-message "Add $model_tag evaluation"
    echo "Uploaded resumable $model_tag checkpoint, tokenizer, and evaluation to hf://$HF_MODEL_REPO/$model_tag"
}

run_experiment "all_scores" -1
run_experiment "score_gte1" 1
run_experiment "score_gte2" 2
