#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
#     "huggingface_hub",
#     "tiktoken",
#     "tqdm",
# ]
# ///

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hffs
import polars
import tiktoken
from tqdm import tqdm


DATASET_PATH = "datasets/duarteocarmo/fineweb2-bagaco2/fineweb2-ptpt-prototype"
TOTAL_ROWS = 33_137_796


def estimate_file_tokens(parquet: str) -> dict:
    data = polars.read_parquet(source="hf://" + parquet, columns=["text"])
    text = "\n".join(data.get_column(name="text").to_list())
    tokenizer = tiktoken.get_encoding(encoding_name="gpt2")
    tokens = len(tokenizer.encode_ordinary(text=text))
    return {
        "parquet": parquet,
        "rows": data.height,
        "characters": len(text),
        "tokens": tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Bagaço2 GPT-2 tokens from sampled parquet files.")
    parser.add_argument("--n-sample", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-rows", type=int, default=TOTAL_ROWS)
    args = parser.parse_args()

    random.seed(a=args.seed)
    files = hffs.ls(path=DATASET_PATH, detail=False)
    parquets = [file for file in files if file.endswith(".parquet")]
    sampled_parquets = random.sample(population=parquets, k=args.n_sample)

    total_rows = 0
    total_tokens = 0
    total_characters = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(estimate_file_tokens, parquet=parquet) for parquet in sampled_parquets]
        for future in tqdm(as_completed(fs=futures), total=len(futures), desc="Counting files"):
            result = future.result()
            total_rows += result["rows"]
            total_tokens += result["tokens"]
            total_characters += result["characters"]
            print(
                f"{result['parquet']} | rows={result['rows']:,} | "
                f"tokens={result['tokens']:,} | chars/token={result['characters'] / result['tokens']:.2f}"
            )

    tokens_per_row = total_tokens / total_rows
    estimated_total_tokens = tokens_per_row * args.total_rows

    print("\nSample total")
    print(f"Files: {len(sampled_parquets):,}")
    print(f"Rows: {total_rows:,}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Characters: {total_characters:,}")
    print(f"Tokens / row: {tokens_per_row:.2f}")
    print(f"Characters / token: {total_characters / total_tokens:.2f}")
    print("\nEstimated total")
    print(f"Rows: {args.total_rows:,}")
    print(f"GPT-2 tokens: {estimated_total_tokens / 1_000_000_000:.2f}B")


if __name__ == "__main__":
    main()
