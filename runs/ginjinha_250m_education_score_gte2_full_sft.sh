#!/bin/bash
set -euo pipefail

# SFT the saved d11 Portuguese education model on one or more H100 GPUs.
#
# Run:
#   bash runs/ginjinha_250m_education_score_gte2_full_sft.sh

# Run identity and storage
MODEL_TAG="ginjinha_d11_ratio40_education_score_gte2_full_corpus"
MODEL_STEP="7860"
WANDB_RUN="${MODEL_TAG}_pt_sft"
HF_BUCKET="duarteocarmo/ginjinha"

# Training
SFT_STEPS="${SFT_STEPS:-500}"
DEVICE_BATCH_SIZE="32"
EVAL_EVERY="100"
EVAL_TOKENS="20971520"
CHATCORE_EVERY="-1"

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
NPROC_PER_NODE=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$NPROC_PER_NODE" = "0" ]; then
    echo "No NVIDIA GPUs found" >&2
    exit 1
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -vq "H100"; then
    echo "This run requires H100 GPUs" >&2
    exit 1
fi
if [ $((8 % NPROC_PER_NODE)) -ne 0 ]; then
    echo "GPU count must divide 8 for the inherited 524,288-token batch size" >&2
    exit 1
fi
echo "Using $NPROC_PER_NODE H100 GPU process(es)"

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
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --num-iterations="$SFT_STEPS" \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens="$EVAL_TOKENS" \
    --chatcore-every="$CHATCORE_EVERY" \
    --run="$WANDB_RUN" \
    "$@"

uvx hf buckets sync "$SFT_CHECKPOINT_DIR" "$HF_SFT_RUN_URI/checkpoints" --exclude "optim_*.pt"
echo "Uploaded SFT checkpoint to $HF_SFT_RUN_URI/checkpoints"
