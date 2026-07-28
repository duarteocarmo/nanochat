#!/bin/bash
set -euo pipefail

# Screen educational-score filters with matched initialization and data-order seeds.
# Train all checkpoints first, then evaluate the frozen six-task PTCORE and BPB once.
#
# Matrix: D6, ratio 40, filters all/>=1/>=2/>=3, seeds 42/1337/2026.
# Hardware: one H100 using FP8.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$NANOCHAT_BASE_DIR"

HF_MODEL_REPO="${HF_MODEL_REPO:-duarteocarmo/ginjinha}"
WANDB_PROJECT="ginjinha-education-ablation"
DEPTH=6
TARGET_PARAM_DATA_RATIO=40
NUM_TRAIN_SHARDS=359
MAX_SEQ_LEN=2048
TOTAL_BATCH_SIZE=262144
DEVICE_BATCH_SIZE=128
NPROC_PER_NODE=1
FINAL_STEP=3540
SEEDS=(42 1337 2026)
FILTER_NAMES=(all_scores score_gte1 score_gte2 score_gte3)
FILTER_SCORES=(-1 1 2 3)

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || true)"
if [[ "$GPU_NAME" != *H100* ]]; then
    echo "Expected one H100, found: ${GPU_NAME:-no GPU}" >&2
    exit 1
fi
WORLD_TOKENS_PER_MICRO_BATCH=$((NPROC_PER_NODE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
if [ "$WORLD_TOKENS_PER_MICRO_BATCH" -ne "$TOTAL_BATCH_SIZE" ]; then
    echo "Expected gradient accumulation 1, got incompatible batch settings" >&2
    exit 1
fi
echo "Using one $GPU_NAME, device batch $DEVICE_BATCH_SIZE, total batch $TOTAL_BATCH_SIZE, gradient accumulation 1"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
uvx hf auth whoami > /dev/null 2>&1 || uvx hf auth login
uvx hf models info "$HF_MODEL_REPO" > /dev/null

# Freeze one tokenizer before downloading shards that must not enter tokenizer training.
python -m nanochat.dataset -n 8
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] || [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]; then
    python -m scripts.tok_train
    python -m scripts.tok_eval
else
    echo "Reusing tokenizer from $NANOCHAT_BASE_DIR/tokenizer"
fi

# All training shards keep strict filters from cycling during the ratio-40 run.
python -m nanochat.dataset -n "$NUM_TRAIN_SHARDS"
SHARD_COUNT=$(find "$NANOCHAT_BASE_DIR/base_data_bagaco2" -name 'shard_*.parquet' -type f | wc -l | tr -d ' ')
if [ "$SHARD_COUNT" -ne 360 ]; then
    echo "Expected 359 training shards plus one validation shard, found $SHARD_COUNT" >&2
    exit 1
fi

model_tag_for() {
    local filter_name="$1"
    local seed="$2"
    echo "ginjinha_d${DEPTH}_ratio${TARGET_PARAM_DATA_RATIO}_education_${filter_name}_seed${seed}"
}

train_experiment() {
    local filter_name="$1"
    local minimum_score="$2"
    local seed="$3"
    local model_tag
    local checkpoint_dir
    model_tag=$(model_tag_for "$filter_name" "$seed")
    checkpoint_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$model_tag"

    if [ -f "$checkpoint_dir/model_$(printf '%06d' "$FINAL_STEP").pt" ] && [ -f "$checkpoint_dir/meta_$(printf '%06d' "$FINAL_STEP").json" ]; then
        echo "Skipping completed training: $model_tag"
    else
        echo "Training $model_tag (minimum score $minimum_score, init/data seed $seed)"
        torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
            --model-tag="$model_tag" \
            --depth="$DEPTH" \
            --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
            --min-educational-score="$minimum_score" \
            --seed="$seed" \
            --data-seed="$seed" \
            --max-seq-len="$MAX_SEQ_LEN" \
            --total-batch-size="$TOTAL_BATCH_SIZE" \
            --device-batch-size="$DEVICE_BATCH_SIZE" \
            --eval-every=-1 \
            --core-metric-every=-1 \
            --sample-every=-1 \
            --save-every=-1 \
            --fp8 \
            --wandb-project="$WANDB_PROJECT" \
            --run="$model_tag"
    fi

    if [ ! -f "$checkpoint_dir/model_$(printf '%06d' "$FINAL_STEP").pt" ]; then
        echo "Final model checkpoint missing for $model_tag" >&2
        exit 1
    fi
    uvx hf upload "$HF_MODEL_REPO" "$checkpoint_dir" "$model_tag/checkpoints" \
        --repo-type model \
        --exclude 'optim_*.pt' \
        --commit-message "Add $model_tag final checkpoint"
    uvx hf upload "$HF_MODEL_REPO" "$NANOCHAT_BASE_DIR/tokenizer" "$model_tag/tokenizer" \
        --repo-type model \
        --commit-message "Add $model_tag tokenizer"
    echo "Uploaded final model and tokenizer for $model_tag"
}

evaluate_experiment() {
    local filter_name="$1"
    local seed="$2"
    local model_tag
    local eval_csv
    model_tag=$(model_tag_for "$filter_name" "$seed")
    eval_csv="$NANOCHAT_BASE_DIR/base_eval/$model_tag.csv"

    if [ -f "$eval_csv" ]; then
        echo "Skipping completed evaluation: $model_tag"
    else
        echo "Evaluating $model_tag with full PTCORE and unfiltered BPB"
        torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
            --model-tag="$model_tag" \
            --step="$FINAL_STEP" \
            --eval=core,bpb \
            --max-per-task=-1 \
            --device-batch-size="$DEVICE_BATCH_SIZE"
    fi

    if [ ! -f "$eval_csv" ]; then
        echo "Evaluation CSV missing for $model_tag" >&2
        exit 1
    fi
    uvx hf upload "$HF_MODEL_REPO" "$eval_csv" "$model_tag/eval/results.csv" \
        --repo-type model \
        --commit-message "Add $model_tag final PTCORE evaluation"
    echo "Uploaded PTCORE evaluation to hf://$HF_MODEL_REPO/$model_tag/eval/results.csv"
}

# Complete one matched-seed block at a time. Evaluation remains deferred until all training ends.
for seed in "${SEEDS[@]}"; do
    for index in "${!FILTER_NAMES[@]}"; do
        train_experiment "${FILTER_NAMES[$index]}" "${FILTER_SCORES[$index]}" "$seed"
    done
done

for seed in "${SEEDS[@]}"; do
    for filter_name in "${FILTER_NAMES[@]}"; do
        evaluate_experiment "$filter_name" "$seed"
    done
done

echo "Completed 12 D6 ratio-40 educational-score runs and final evaluations"
