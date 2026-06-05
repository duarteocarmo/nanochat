#!/bin/bash

# Tiny Portuguese CPU/MPS integration test. This only checks that the PT pipeline runs.
# Run as:
# bash runs/runcpu-pt-mini.sh

set -e

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-pt-mini"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Minimal tokenizer/data pass
python -m nanochat.dataset -n 1
python -m scripts.tok_train --max-chars=2000000 --vocab-size=4096
python -m scripts.tok_eval

# Tiny base model run
python -m scripts.base_train \
    --depth=2 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=128 \
    --device-batch-size=2 \
    --total-batch-size=256 \
    --eval-every=5 \
    --eval-tokens=1024 \
    --core-metric-name=ptcore \
    --core-metric-every=-1 \
    --sample-every=5 \
    --save-every=-1 \
    --num-iterations=10 \
    --model-tag=pt-mini \
    --run=$WANDB_RUN

python -m scripts.base_eval \
    --model-tag=pt-mini \
    --device-batch-size=1 \
    --split-tokens=1024 \
    --max-per-task=1 \
    --ptcore-split=val \
    --eval ptcore,bpb

# Tiny PT-only SFT smoke run
python -m scripts.chat_sft \
    --model-tag=pt-mini \
    --max-seq-len=128 \
    --device-batch-size=2 \
    --total-batch-size=256 \
    --eval-every=5 \
    --eval-tokens=1024 \
    --chatcore-every=-1 \
    --num-iterations=10 \
    --run=$WANDB_RUN

# English chat eval placeholder. Keep commented until PT chat eval is wired.
# python -m scripts.chat_eval \
#     --source=sft \
#     --model-tag=pt-mini \
#     --task-name=ARC-Easy \
#     --batch-size=1 \
#     --max-problems=1

python -m scripts.chat_cli -p "Olá! Quem és tu?" --model-tag=pt-mini
