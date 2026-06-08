#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0.0", "torch==2.9.1"]
# ///

"""Suggest a nanochat depth for a Chinchilla-like PT run."""

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from nanochat.gpt import GPT, GPTConfig

GREEN = "\033[1;32m"
RESET = "\033[0m"


def openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab, ff_ratio=4):
    """OpenAI-style forward-pass FLOPs per token for a decoder-only Transformer."""
    d_attn = d_model // n_heads
    d_ff = d_model * ff_ratio

    embeddings = 4 * d_model
    attn_qkv = 2 * n_layers * d_model * 3 * (d_attn * n_heads)
    attn_mask = 2 * n_layers * n_ctx * (d_attn * n_heads)
    attn_project = 2 * n_layers * (d_attn * n_heads) * d_model
    ff = 2 * n_layers * 2 * d_model * d_ff
    logits = 2 * d_model * n_vocab

    return embeddings + attn_qkv + attn_mask + attn_project + ff + logits


def format_row(row, best):
    marker = "*" if row is best else " "
    line = (
        f"{marker} "
        f"d{row['depth']:<4} "
        f"{row['dim']:>5} "
        f"{row['params'] / 1e6:>9.1f}M "
        f"{row['scaling_params'] / 1e6:>9.1f}M "
        f"{row['training_flops_per_token'] / 1e9:>19.3f}G "
        f"{row['tokens'] / 1e6:>12.1f}M "
        f"{row['ratio']:>15.2f}"
    )
    if row is best and "NO_COLOR" not in os.environ:
        return f"{GREEN}{line}{RESET}"
    return line


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tflops", type=float, help="available TFLOPS")
    parser.add_argument("hours", type=float, nargs="?", default=1.0, help="training hours")
    parser.add_argument(
        "--ratio",
        type=float,
        default=float(os.environ.get("CHINCHILLA_RATIO", 20)),
        help="target train tokens per scaling param",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    vocab_size = 32768
    sequence_len = 2048
    aspect_ratio = 64
    head_dim = 128
    window_pattern = "SSSL"
    compute_budget = args.tflops * 1e12 * args.hours * 3600

    rows = []
    for depth in range(2, 27, 2):
        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        num_heads = model_dim // head_dim
        config = GPTConfig(
            sequence_len=sequence_len,
            vocab_size=vocab_size,
            n_layer=depth,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=model_dim,
            window_pattern=window_pattern,
        )
        with torch.device("meta"):
            model = GPT(config=config)
        counts = model.num_scaling_params()
        scaling_params = counts["transformer_matrices"] + counts["lm_head"]
        forward_flops_per_token = openai_flops_per_token(
            n_layers=depth,
            n_heads=num_heads,
            d_model=model_dim,
            n_ctx=sequence_len,
            n_vocab=vocab_size,
        )
        training_flops_per_token = 3 * forward_flops_per_token
        train_tokens = compute_budget / training_flops_per_token
        ratio = train_tokens / scaling_params
        score = abs(math.log(ratio / args.ratio))
        rows.append({
            "depth": depth,
            "dim": model_dim,
            "heads": num_heads,
            "params": counts["total"],
            "scaling_params": scaling_params,
            "forward_flops_per_token": forward_flops_per_token,
            "training_flops_per_token": training_flops_per_token,
            "tokens": train_tokens,
            "ratio": ratio,
            "score": score,
        })

    best = min(rows, key=lambda row: row["score"])
    example = rows[0]

    print(f"Compute budget: {compute_budget:.3e} FLOPs ({args.tflops:g} TFLOPS × {args.hours:g}h)")
    print(f"Target ratio: {args.ratio:g} train tokens / scaling param")
    print("FLOPs method: OpenAI forward-pass count × 3 for training")
    print()
    print("Example calculation for d2:")
    print(
        f"  architecture = {example['depth']} layers, {example['heads']} heads, "
        f"d_model {example['dim']}, ctx {sequence_len}, vocab {vocab_size}"
    )
    print(f"  total params = {example['params'] / 1e6:.1f}M")
    print(f"  scaling params = transformer_matrices + lm_head = {example['scaling_params'] / 1e6:.1f}M")
    print(
        f"  forward FLOPs/token = OpenAI formula = "
        f"{example['forward_flops_per_token']:.3e} FLOPs/token"
    )
    print(
        f"  training FLOPs/token = 3 × forward FLOPs/token = "
        f"3 × {example['forward_flops_per_token']:.3e} = "
        f"{example['training_flops_per_token']:.3e} FLOPs/token "
        f"({example['training_flops_per_token'] / 1e9:.3f}G)"
    )
    print(
        f"  train tokens = compute budget / training FLOPs/token = "
        f"{compute_budget:.3e} / {example['training_flops_per_token']:.3e} = "
        f"{example['tokens'] / 1e6:.1f}M tokens"
    )
    print(
        f"  tokens/scaling = train tokens / scaling params = "
        f"{example['tokens'] / 1e6:.1f}M / {example['scaling_params'] / 1e6:.1f}M = "
        f"{example['ratio']:.2f}"
    )
    print()
    print(
        f"{'':1} {'depth':>5} {'dim':>5} {'params':>10} {'scaling':>10} "
        f"{'OpenAI train FLOPs/tok':>23} {'train tokens':>13} {'tokens/scaling':>15}"
    )
    print(
        f"{'':1} {'-' * 5:>5} {'-' * 5:>5} {'-' * 10:>10} {'-' * 10:>10} "
        f"{'-' * 23:>23} {'-' * 13:>13} {'-' * 15:>15}"
    )
    for row in rows:
        print(format_row(row=row, best=best))
    print()
    print(
        f"Suggested: d{best['depth']} "
        f"({best['params'] / 1e6:.1f}M params, "
        f"{best['tokens'] / 1e6:.1f}M tokens, "
        f"{best['ratio']:.2f} tokens/scaling param)"
    )


if __name__ == "__main__":
    main()
