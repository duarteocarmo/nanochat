#!/bin/bash
set -euo pipefail

# Re-evaluate validation BPB for the completed D6 ratio-40 seed ablation.
# Downloads one train shard, one validation shard, each final checkpoint, and one shared tokenizer.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$NANOCHAT_BASE_DIR"

HF_MODEL_REPO="${HF_MODEL_REPO:-duarteocarmo/ginjinha}"
FINAL_STEP=3540
MAX_SEQ_LEN=2048
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
SPLIT_TOKENS=20971520
NPROC_PER_NODE=1
SEEDS=(42 1337 2026)
FILTER_NAMES=(all_scores score_gte1 score_gte2 score_gte3)

TOKENS_PER_STEP=$((NPROC_PER_NODE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
if [ $((SPLIT_TOKENS % TOKENS_PER_STEP)) -ne 0 ]; then
    echo "Split tokens must be divisible by tokens per step: $TOKENS_PER_STEP" >&2
    exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || true)"
if [ -z "$GPU_NAME" ]; then
    echo "A CUDA GPU is required" >&2
    exit 1
fi
echo "Using $GPU_NAME for validation BPB evaluation"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
uvx hf auth whoami > /dev/null 2>&1 || uvx hf auth login
uvx hf models info "$HF_MODEL_REPO" > /dev/null

# The existing BPB evaluator runs both splits. One train shard is sufficient.
python -m nanochat.dataset -n 1
TRAIN_SHARD="$NANOCHAT_BASE_DIR/base_data_bagaco2/shard_00000.parquet"
VAL_SHARD="$NANOCHAT_BASE_DIR/base_data_bagaco2/shard_00359.parquet"
if [ ! -f "$TRAIN_SHARD" ] || [ ! -f "$VAL_SHARD" ]; then
    echo "Train or validation shard missing" >&2
    exit 1
fi

model_tag_for() {
    local filter_name="$1"
    local seed="$2"
    echo "ginjinha_d6_ratio40_education_${filter_name}_seed${seed}"
}

# Every run used the same frozen tokenizer, so download it once.
TOKENIZER_SOURCE_TAG="$(model_tag_for all_scores 42)"
mkdir -p "$NANOCHAT_BASE_DIR/tokenizer"
uvx hf repos cp "hf://$HF_MODEL_REPO/$TOKENIZER_SOURCE_TAG/tokenizer/tokenizer.pkl" \
    "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
uvx hf repos cp "hf://$HF_MODEL_REPO/$TOKENIZER_SOURCE_TAG/tokenizer/token_bytes.pt" \
    "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt"

for seed in "${SEEDS[@]}"; do
    for filter_name in "${FILTER_NAMES[@]}"; do
        model_tag="$(model_tag_for "$filter_name" "$seed")"
        checkpoint_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$model_tag"
        eval_csv="$NANOCHAT_BASE_DIR/base_eval/${model_tag}_bpb.csv"
        eval_log="$NANOCHAT_BASE_DIR/base_eval/${model_tag}_bpb.log"
        step_padded="$(printf '%06d' "$FINAL_STEP")"
        mkdir -p "$checkpoint_dir" "$NANOCHAT_BASE_DIR/base_eval"

        uvx hf repos cp "hf://$HF_MODEL_REPO/$model_tag/checkpoints/model_${step_padded}.pt" \
            "$checkpoint_dir/model_${step_padded}.pt"
        uvx hf repos cp "hf://$HF_MODEL_REPO/$model_tag/checkpoints/meta_${step_padded}.json" \
            "$checkpoint_dir/meta_${step_padded}.json"

        if [ -f "$eval_csv" ] && grep -q '^val,' "$eval_csv"; then
            echo "Skipping completed validation BPB evaluation: $model_tag"
        else
            torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
                --model-tag="$model_tag" \
                --step="$FINAL_STEP" \
                --eval=bpb \
                --split-tokens="$SPLIT_TOKENS" \
                --device-batch-size="$DEVICE_BATCH_SIZE" \
                --device-type=cuda 2>&1 | tee "$eval_log"

            val_bpb="$(awk '$1 == "val" && $2 == "bpb:" {value=$3} END {print value}' "$eval_log")"
            if ! [[ "$val_bpb" =~ ^[0-9]+\.[0-9]+$ ]]; then
                echo "Could not extract validation BPB for $model_tag" >&2
                exit 1
            fi
            printf 'Split,BPB,Tokens\nval,%s,%s\n' "$val_bpb" "$SPLIT_TOKENS" > "$eval_csv"
        fi

        if [ ! -f "$eval_csv" ] || ! grep -q '^val,' "$eval_csv"; then
            echo "Validation BPB CSV missing or invalid for $model_tag" >&2
            exit 1
        fi
        uvx hf upload "$HF_MODEL_REPO" "$eval_csv" "$model_tag/eval/val_bpb.csv" \
            --repo-type model \
            --commit-message "Add $model_tag validation BPB"
        echo "Uploaded validation BPB to hf://$HF_MODEL_REPO/$model_tag/eval/val_bpb.csv"
    done
done

echo "Completed validation BPB evaluation for all 12 D6 ratio-40 runs"
