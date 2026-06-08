#!/bin/bash

# Portuguese single-H100 ~60 minute run.
# Defaults are intentionally conservative for wall-clock time.
# Run as: WANDB_RUN=my-run bash runs/runpth10060m.sh

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
: "${HF_TOKEN:?HF_TOKEN or HUGGING_FACE_HUB_TOKEN must be set}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

RUN_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-pt-h100-60m}"

WANDB_RUN="${WANDB_RUN:-pt-h100-$RUN_TIMESTAMP}"
MODEL_TAG="${MODEL_TAG:-pt-h100-d10-$RUN_TIMESTAMP}"

mkdir -p "$NANOCHAT_BASE_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "PT H100 60m run"
log "base_dir=$NANOCHAT_BASE_DIR"
log "model_tag=$MODEL_TAG depth=10 target_flops=1.5e17 sft_steps=300 eval_every=200 ptcore_every=250 sample_every=200"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.report reset
python -m nanochat.dataset -n 8

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] || [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]; then
    python -m scripts.tok_train
    python -m scripts.tok_eval
else
    log "Tokenizer already exists, skipping tok_train/tok_eval"
fi

log "Starting base pretraining"
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=10 \
    --device-batch-size=64 \
    --target-flops=1.5e17 \
    --fp8 \
    --core-metric-name=ptcore \
    --core-metric-every=250 \
    --eval-every=200 \
    --eval-tokens=4194304 \
    --sample-every=200 \
    --save-every=-1 \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN"

log "Running base evaluation"
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --eval=ptcore,bpb,sample \
    --ptcore-split=val \
    --device-batch-size=16 \
    --split-tokens=4194304

log "Starting PT SFT"
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size=16 \
    --total-batch-size=131072 \
    --eval-every=150 \
    --eval-tokens=1048576 \
    --chatcore-every=-1 \
    --num-iterations=300 \
    --run="${WANDB_RUN}-sft"

python -m nanochat.report generate

log "Done. Report: $NANOCHAT_BASE_DIR/report.md"
