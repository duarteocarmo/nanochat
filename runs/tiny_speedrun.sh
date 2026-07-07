#!/bin/bash

# Tiny end-to-end smoke test for the full nanochat flow on a Mac/CPU/MPS.
# It is not meant to train a useful model. It validates that all stages connect:
# dataset -> tokenizer -> tokenizer eval -> base train -> CORE/BPB/sample eval -> SFT -> ChatCORE/chat eval.
#
# Run:
#   bash runs/tiny_speedrun.sh
#
# Useful overrides:
#   DEVICE_TYPE=cpu bash runs/tiny_speedrun.sh
#   SKIP_SETUP=1 bash runs/tiny_speedrun.sh
#   NANOCHAT_BASE_DIR=/tmp/nanochat-tiny bash runs/tiny_speedrun.sh

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-tiny-speedrun}"
export WANDB_RUN="${WANDB_RUN:-dummy}"

DEVICE_TYPE="${DEVICE_TYPE:-}"
MODEL_TAG="${MODEL_TAG:-tiny_speedrun}"
DATASET_SHARDS="${DATASET_SHARDS:-1}"
TOK_MAX_CHARS="${TOK_MAX_CHARS:-9000000}"
TOK_DOC_CAP="${TOK_DOC_CAP:-2000}"
VOCAB_SIZE="${VOCAB_SIZE:-4096}"
BASE_STEPS="${BASE_STEPS:-2}"
SFT_STEPS="${SFT_STEPS:-2}"
SEQ_LEN="${SEQ_LEN:-512}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-512}"
EVAL_TOKENS="${EVAL_TOKENS:-512}"
MAX_CORE_PROBLEMS="-1"
MAX_CHAT_PROBLEMS="${MAX_CHAT_PROBLEMS:-1}"

mkdir -p "$NANOCHAT_BASE_DIR"

DEVICE_ARG=""
if [ -n "$DEVICE_TYPE" ]; then
    DEVICE_ARG="--device-type=$DEVICE_TYPE"
fi

if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra cpu
fi
source .venv/bin/activate

python -m nanochat.dataset -n "$DATASET_SHARDS"

python -m scripts.tok_train \
    --max-chars="$TOK_MAX_CHARS" \
    --doc-cap="$TOK_DOC_CAP" \
    --vocab-size="$VOCAB_SIZE"
python -m scripts.tok_eval

python -m scripts.base_train \
    $DEVICE_ARG \
    --model-tag="$MODEL_TAG" \
    --depth=2 \
    --aspect-ratio=32 \
    --head-dim=32 \
    --window-pattern=L \
    --max-seq-len="$SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every=1 \
    --eval-tokens="$EVAL_TOKENS" \
    --core-metric-every="$BASE_STEPS" \
    --core-metric-max-per-task="$MAX_CORE_PROBLEMS" \
    --sample-every="$BASE_STEPS" \
    --save-every=-1 \
    --num-iterations="$BASE_STEPS" \
    --run="$WANDB_RUN"

python -m scripts.base_eval \
    $DEVICE_ARG \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --split-tokens="$EVAL_TOKENS" \
    --max-per-task="$MAX_CORE_PROBLEMS"

python -m scripts.chat_sft \
    $DEVICE_ARG \
    --model-tag="$MODEL_TAG" \
    --load-optimizer=0 \
    --max-seq-len="$SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every=1 \
    --eval-tokens="$EVAL_TOKENS" \
    --chatcore-every=1 \
    --chatcore-max-cat="$MAX_CHAT_PROBLEMS" \
    --chatcore-max-sample="$MAX_CHAT_PROBLEMS" \
    --mmlu-epochs=0 \
    --gsm8k-epochs=0 \
    --num-iterations="$SFT_STEPS" \
    --run="$WANDB_RUN"

python -m scripts.chat_eval \
    $DEVICE_ARG \
    -i sft \
    -g "$MODEL_TAG" \
    -a ARC-Easy \
    -b "$DEVICE_BATCH_SIZE" \
    -x "$MAX_CHAT_PROBLEMS"
