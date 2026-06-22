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

# Shared run settings.
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MODEL_TAG=pt-d10-chinchilla
PTCORE_SPLIT=val
RUN_TIMESTAMP=$(date +%y%m%d%H%M)
HF_BASE_REPO_ID=${HF_BASE_REPO_ID:-duarteocarmo/variacoes-d10-chinchilla-base}
HF_SFT_REPO_ID=${HF_SFT_REPO_ID:-duarteocarmo/variacoes-d10-chinchilla-sft}

# Dataset shards.
DATASET_TOKENIZER_SHARDS=8
DATASET_TRAIN_SHARDS=32

# Tokenizer training.
TOKENIZER_MAX_CHARS=2000000000
TOKENIZER_VOCAB_SIZE=32768

# Base pretraining.
BASE_TRAIN_DEPTH=10
BASE_TRAIN_DEVICE_BATCH_SIZE=32
BASE_TRAIN_TOTAL_BATCH_SIZE=524288
# BASE_TRAIN_TARGET_PARAM_DATA_RATIO=20
BASE_TRAIN_TARGET_PARAM_DATA_RATIO=30 # overtrain a bit to test
BASE_TRAIN_CORE_METRIC_EVERY=500
BASE_TRAIN_CORE_METRIC_MAX_PER_TASK=-1
BASE_TRAIN_EVAL_TOKENS=4194304
BASE_TRAIN_SAMPLE_EVERY=500
BASE_TRAIN_WANDB_RUN=d10_pt_chinchilla_$RUN_TIMESTAMP

# Final base eval.
BASE_EVAL_DEVICE_BATCH_SIZE=4
BASE_EVAL_SPLIT_TOKENS=4194304
BASE_EVAL_MODES=ptcore,bpb,sample

# Chat SFT.
CHAT_SFT_DEVICE_BATCH_SIZE=16
CHAT_SFT_TOTAL_BATCH_SIZE=524288
CHAT_SFT_EVAL_EVERY=200
CHAT_SFT_EVAL_TOKENS=4194304
CHAT_SFT_CHATCORE_EVERY=200
CHAT_SFT_NUM_ITERATIONS=-1
CHAT_SFT_WANDB_RUN=d10_pt_chinchilla_sft_$RUN_TIMESTAMP

# Chat CLI smoke prompt.
CHAT_CLI_PROMPT="Olá! Qual é a capital de Portugal?"

# Chat eval.
CHAT_EVAL_TASK_NAME=PT-PortugalBasicQA
CHAT_EVAL_BATCH_SIZE=4

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.report reset

# Tokenizer: train on the first Bagaço shards, matching the PT tokenizer setup.
python -m nanochat.dataset -n "$DATASET_TOKENIZER_SHARDS"
python -m nanochat.dataset -n "$DATASET_TRAIN_SHARDS" &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train --max-chars="$TOKENIZER_MAX_CHARS" --vocab-size="$TOKENIZER_VOCAB_SIZE"
python -m scripts.tok_eval

wait $DATASET_DOWNLOAD_PID

# Base model pretraining only. PTCORE is evaluated during training and at the end.
# Final base_eval also prints conditioned and unconditioned samples.
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$BASE_TRAIN_DEPTH" \
    --target-param-data-ratio="$BASE_TRAIN_TARGET_PARAM_DATA_RATIO" \
    --total-batch-size="$BASE_TRAIN_TOTAL_BATCH_SIZE" \
    --device-batch-size="$BASE_TRAIN_DEVICE_BATCH_SIZE" \
    --core-metric-name=ptcore \
    --core-metric-every="$BASE_TRAIN_CORE_METRIC_EVERY" \
    --core-metric-max-per-task="$BASE_TRAIN_CORE_METRIC_MAX_PER_TASK" \
    --ptcore-split="$PTCORE_SPLIT" \
    --eval-tokens="$BASE_TRAIN_EVAL_TOKENS" \
    --sample-every="$BASE_TRAIN_SAMPLE_EVERY" \
    --model-tag="$MODEL_TAG" \
    --run="$BASE_TRAIN_WANDB_RUN"

uvx --from huggingface_hub hf upload "$HF_BASE_REPO_ID" "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" . \
    --repo-type model \
    --commit-message "Upload $MODEL_TAG base checkpoint"

# Use a smaller eval batch than training to avoid BPB OOM during final full-logit eval.
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$BASE_EVAL_DEVICE_BATCH_SIZE" \
    --ptcore-split="$PTCORE_SPLIT" \
    --split-tokens="$BASE_EVAL_SPLIT_TOKENS" \
    --eval "$BASE_EVAL_MODES"

# PT-only SFT. New SFT datasets should be proven in runptlocaltest before landing here.
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$CHAT_SFT_DEVICE_BATCH_SIZE" \
    --total-batch-size="$CHAT_SFT_TOTAL_BATCH_SIZE" \
    --eval-every="$CHAT_SFT_EVAL_EVERY" \
    --eval-tokens="$CHAT_SFT_EVAL_TOKENS" \
    --chatcore-every="$CHAT_SFT_CHATCORE_EVERY" \
    --num-iterations="$CHAT_SFT_NUM_ITERATIONS" \
    --run="$CHAT_SFT_WANDB_RUN"

uvx --from huggingface_hub hf upload "$HF_SFT_REPO_ID" "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG" . \
    --repo-type model \
    --commit-message "Upload $MODEL_TAG SFT checkpoint"

# Quick qualitative chat sample from the SFT checkpoint.
python -m scripts.chat_cli \
    --source=sft \
    --model-tag="$MODEL_TAG" \
    --prompt="$CHAT_CLI_PROMPT"

# First PT chat eval task. No max-problems means full eval.
python -m scripts.chat_eval \
    --source=sft \
    --model-tag="$MODEL_TAG" \
    --task-name="$CHAT_EVAL_TASK_NAME" \
    --batch-size="$CHAT_EVAL_BATCH_SIZE"

python -m nanochat.report generate
python -m scripts.wandb_upload_report \
    --project=nanochat-sft \
    --run="$CHAT_SFT_WANDB_RUN" \
    --path="$NANOCHAT_BASE_DIR/report/report.md"
