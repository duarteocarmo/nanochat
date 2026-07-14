#!/bin/bash
set -euo pipefail

# SFT the saved d11 Portuguese education model on one or more CUDA GPUs.
#
# Run:
#   bash runs/ginjinha_250m_education_score_gte2_full_sft.sh
#
# Hardware-sensitive settings can be overridden without changing this file:
#   NPROC_PER_NODE=2 DEVICE_BATCH_SIZE=16 bash runs/ginjinha_250m_education_score_gte2_full_sft.sh
# Use a smaller DEVICE_BATCH_SIZE if GPU memory is limited. Gradient accumulation
# keeps TOTAL_BATCH_SIZE unchanged. SFT runs for one full dataset epoch.

# Run identity and storage
MODEL_TAG="ginjinha_d11_ratio40_education_score_gte2_full_corpus"
MODEL_STEP="7860"
WANDB_RUN="${MODEL_TAG}_pt_sft"
HF_BUCKET="duarteocarmo/ginjinha"

# Training
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
EVAL_EVERY="100"
EVAL_TOKENS="20971520"
PTCORE_CHAT_EVERY="${PTCORE_CHAT_EVERY:-100}"

# Runtime and derived paths
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
HF_BASE_RUN_URI="hf://buckets/$HF_BUCKET/$MODEL_TAG"
HF_SFT_RUN_URI="hf://buckets/$HF_BUCKET/$WANDB_RUN"
BASE_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
SFT_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"
mkdir -p "$NANOCHAT_BASE_DIR"

if ! command -v nvidia-smi > /dev/null; then
    echo "nvidia-smi is required" >&2
    exit 1
fi
if [ -z "${NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l | tr -d ' ')
fi
for SETTING in NPROC_PER_NODE DEVICE_BATCH_SIZE MAX_SEQ_LEN TOTAL_BATCH_SIZE; do
    if ! [[ "${!SETTING}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$SETTING must be a positive integer" >&2
        exit 1
    fi
done
TOKENS_PER_MICRO_BATCH=$((NPROC_PER_NODE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
if [ $((TOTAL_BATCH_SIZE % TOKENS_PER_MICRO_BATCH)) -ne 0 ]; then
    echo "TOTAL_BATCH_SIZE must be divisible by NPROC_PER_NODE * DEVICE_BATCH_SIZE * MAX_SEQ_LEN" >&2
    exit 1
fi
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd, -)
echo "Using $NPROC_PER_NODE CUDA GPU process(es): $GPU_NAMES"
echo "Device batch size: $DEVICE_BATCH_SIZE; gradient accumulation: $((TOTAL_BATCH_SIZE / TOKENS_PER_MICRO_BATCH))"

command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
uvx hf auth whoami > /dev/null 2>&1 || uvx hf auth login
uvx hf buckets info "$HF_BUCKET" > /dev/null

mkdir -p "$BASE_CHECKPOINT_DIR" "$NANOCHAT_BASE_DIR/tokenizer"
printf -v MODEL_STEP_PADDED "%06d" "$MODEL_STEP"
uvx hf buckets cp "$HF_BASE_RUN_URI/checkpoints/model_$MODEL_STEP_PADDED.pt" "$BASE_CHECKPOINT_DIR/"
uvx hf buckets cp "$HF_BASE_RUN_URI/checkpoints/meta_$MODEL_STEP_PADDED.json" "$BASE_CHECKPOINT_DIR/"
uvx hf buckets sync "$HF_BASE_RUN_URI/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer"
echo "Downloaded base checkpoint step $MODEL_STEP and tokenizer from $HF_BASE_RUN_URI"

# The base bucket excludes optimizer shards, so SFT starts with a fresh optimizer.
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" \
    --model-step="$MODEL_STEP" \
    --load-optimizer=0 \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens="$EVAL_TOKENS" \
    --ptcore-chat-every="$PTCORE_CHAT_EVERY" \
    --run="$WANDB_RUN" \
    "$@"

uvx hf buckets sync "$SFT_CHECKPOINT_DIR" "$HF_SFT_RUN_URI/checkpoints" --exclude "optim_*.pt"
echo "Uploaded SFT checkpoint to $HF_SFT_RUN_URI/checkpoints"
