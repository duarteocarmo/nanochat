#!/bin/bash

# Tiny Portuguese local integration test.
# Keep this as the first place to add new PT pipeline steps before promoting them
# to the larger Chinchilla GPU scripts.
# Run as:
# bash runs/runptlocaltest.sh

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-pt-localtest"
mkdir -p "$NANOCHAT_BASE_DIR"

WANDB_RUN=dummy
MODEL_TAG=pt-localtest
TRAIN_SHARDS=1
TOKENIZER_MAX_CHARS=200000000
VOCAB_SIZE=4096
DEVICE_BATCH_SIZE=2
TOTAL_BATCH_SIZE=256
MAX_SEQ_LEN=128
EVAL_TOKENS=1024
SPLIT_TOKENS=1024
BASE_STEPS=500
SFT_STEPS=200

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

python -m nanochat.report reset

# Tokenizer/data smoke pass. Keep tiny so this can run on a Mac.
python -m nanochat.dataset -n "$TRAIN_SHARDS"
python -m scripts.tok_train --max-chars="$TOKENIZER_MAX_CHARS" --vocab-size="$VOCAB_SIZE"
python -m scripts.tok_eval

# Tiny base pretraining run with PTCORE and samples wired in like the Chinchilla scripts.
python -m scripts.base_train \
    --depth=2 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every=5 \
    --eval-tokens="$EVAL_TOKENS" \
    --core-metric-name=ptcore \
    --core-metric-every=250 \
    --core-metric-max-per-task=1 \
    --ptcore-split=val \
    --sample-every=5 \
    --save-every=-1 \
    --num-iterations="$BASE_STEPS" \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN"

python -m scripts.base_eval \
    --model-tag="$MODEL_TAG" \
    --device-batch-size=1 \
    --split-tokens="$SPLIT_TOKENS" \
    --max-per-task=1 \
    --ptcore-split=val \
    --eval ptcore,bpb,sample

# Tiny PT-only SFT smoke run. New SFT datasets should be added here first.
python -m scripts.chat_sft \
    --model-tag="$MODEL_TAG" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every=5 \
    --eval-tokens="$EVAL_TOKENS" \
    --chatcore-every=-1 \
    --num-iterations="$SFT_STEPS" \
    --run="$WANDB_RUN"

# Chat CLI smoke check against the SFT checkpoint.
python -m scripts.chat_cli \
    --source=sft \
    --model-tag="$MODEL_TAG" \
    --prompt="Olá! Quem és tu?"

# Tiny PT chat eval smoke check.
python -m scripts.chat_eval \
    --source=sft \
    --model-tag="$MODEL_TAG" \
    --task-name=PT-PortugalBasicQA \
    --batch-size=1 \
    --max-problems=3

# Future speedrun steps. Keep commented until PT-specific versions are wired.
# curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
#     https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# python -m scripts.chat_web --model-tag="$MODEL_TAG"

python -m nanochat.report generate
