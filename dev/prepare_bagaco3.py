"""Shuffle Bagaço3 and write train/validation Parquet shards.

Run:
    uv run python -m dev.prepare_bagaco3

For a local sample:
    uv run python -m dev.prepare_bagaco3 --input-files sample.parquet --validation-documents 8 --output-dir /tmp/bagaco3-sample

Full preparation needs disk space for the source download, Arrow cache, and output.
All columns, educational scores, and document texts are preserved unchanged.
The training loader must use the explicit split directories, not the last shard.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TypedDict

import datasets
import pyarrow
from pyarrow import compute, parquet


class ShardRecord(TypedDict):
    path: str
    documents: int
    characters: int
    row_groups: int


SOURCE_REPO = "duarteocarmo/bagaco3"
SOURCE_REVISION = "b0d142332b2308457f2f3f8ba6f9f5f04460972d"


def write_split(
    dataset: datasets.Dataset,
    output_dir: Path,
    chars_per_shard: int,
    row_group_size: int,
) -> dict:
    output_dir.mkdir(parents=True)
    shards: list[ShardRecord] = []
    sources = Counter()
    scores = Counter()
    writer = None
    shard_documents = shard_characters = documents = 0
    try:
        for batch in dataset.with_format(type="arrow").iter(batch_size=row_group_size):
            if writer is None:
                shard_path = output_dir / f"shard_{len(shards):05d}.parquet"
                temporary_path = shard_path.with_suffix(suffix=".parquet.incomplete")
                writer = parquet.ParquetWriter(
                    where=temporary_path,
                    schema=batch.schema,
                    compression="zstd",
                    compression_level=3,
                    use_dictionary=["source", "language", "license", "category"],
                )
            writer.write_table(table=batch, row_group_size=row_group_size)
            shard_documents += batch.num_rows
            documents += batch.num_rows
            lengths = compute.call_function(
                name="utf8_length", args=[batch.column("text")]
            )
            shard_characters += compute.call_function(
                name="sum", args=[lengths]
            ).as_py()
            for column, counts in [("source", sources), ("educational_score", scores)]:
                for value in compute.call_function(
                    name="value_counts", args=[batch.column(column)]
                ).to_pylist():
                    counts[str(value["values"])] += value["counts"]

            if shard_characters < chars_per_shard and documents < len(dataset):
                continue
            writer.close()
            writer = None
            metadata = parquet.read_metadata(where=temporary_path)
            if metadata.num_rows != shard_documents:
                raise RuntimeError(f"Row count mismatch in {temporary_path}")
            temporary_path.replace(target=shard_path)
            shards.append(
                {
                    "path": f"{output_dir.name}/{shard_path.name}",
                    "documents": shard_documents,
                    "characters": shard_characters,
                    "row_groups": metadata.num_row_groups,
                }
            )
            print(f"Wrote {shard_documents:,} documents to {shard_path}", flush=True)
            shard_documents = shard_characters = 0
    finally:
        if writer is not None:
            writer.close()
    return {
        "documents": sum(shard["documents"] for shard in shards),
        "characters": sum(shard["characters"] for shard in shards),
        "sources": dict(sorted(sources.items())),
        "educational_scores": dict(sorted(scores.items())),
        "shards": shards,
    }


def prepare_dataset(
    dataset: datasets.Dataset,
    output_dir: Path,
    source: dict,
    validation_documents: int = 65_536,
    seed: int = 42,
    chars_per_shard: int = 250_000_000,
    row_group_size: int = 1024,
) -> dict:
    if not 0 < validation_documents < len(dataset):
        raise ValueError(
            "validation_documents must leave at least one document in each split"
        )
    if chars_per_shard < 1 or row_group_size < 1:
        raise ValueError("chars_per_shard and row_group_size must be positive")
    required_columns = {"text", "id", "source", "educational_score"}
    if not required_columns.issubset(dataset.column_names):
        raise ValueError(
            f"Missing columns: {sorted(required_columns - set(dataset.column_names))}"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output_dir}; choose a new output directory"
        )

    shuffled = dataset.shuffle(seed=seed)
    print(f"Shuffled {len(shuffled):,} documents with seed {seed}", flush=True)
    split_datasets = {
        "train": shuffled.select(indices=range(validation_documents, len(shuffled))),
        "validation": shuffled.select(indices=range(validation_documents)),
    }
    output_dir.mkdir(parents=True)
    manifest = {
        "source": source,
        "seed": seed,
        "validation_documents": validation_documents,
        "chars_per_shard": chars_per_shard,
        "row_group_size": row_group_size,
        "versions": {"datasets": datasets.__version__, "pyarrow": pyarrow.__version__},
        "splits": {},
    }
    for split, split_dataset in split_datasets.items():
        manifest["splits"][split] = write_split(
            dataset=split_dataset,
            output_dir=output_dir / split,
            chars_per_shard=chars_per_shard,
            row_group_size=row_group_size,
        )
    if sum(split["documents"] for split in manifest["splits"].values()) != len(dataset):
        raise RuntimeError("Output document count does not match the input")
    (output_dir / "manifest.json").write_text(
        data=json.dumps(obj=manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        data="""---
language:
- pt
license: other
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*.parquet
  - split: validation
    path: validation/*.parquet
---

# Bagaço3 shuffled

Shuffled documents from [Bagaço3](https://huggingface.co/datasets/duarteocarmo/bagaco3).
All source columns, document texts, and educational scores are preserved.
Licenses differ by source. See each document's `license` field and the source dataset card.

The validation documents are selected from the global shuffle and excluded from training.
See `manifest.json` for the source revision, preparation settings, counts, and shard paths.
Use `dev/prepare_bagaco3.py` in the nanochat fork to reproduce preparation.
""",
        encoding="utf-8",
    )
    print(f"Prepared {len(dataset):,} documents in {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dev-ignore/bagaco3-shuffle")
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument(
        "--input-files",
        type=Path,
        nargs="+",
        help="Use local Parquet files instead of downloading Bagaço3",
    )
    parser.add_argument("--validation-documents", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chars-per-shard", type=int, default=250_000_000)
    parser.add_argument("--row-group-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if (
        args.validation_documents < 1
        or args.chars_per_shard < 1
        or args.row_group_size < 1
    ):
        parser.error(
            "Validation documents, characters per shard, and row group size must be positive"
        )
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    if args.input_files:
        files = [str(path.resolve()) for path in args.input_files]
        dataset = datasets.load_dataset(
            path="parquet", data_files=files, split="train", cache_dir=cache_dir
        )
        source = {"files": files}
    else:
        dataset = datasets.load_dataset(
            path=SOURCE_REPO, revision=args.revision, split="train", cache_dir=cache_dir
        )
        source = {"repo_id": SOURCE_REPO, "revision": args.revision}
    prepare_dataset(
        dataset=dataset,
        output_dir=args.output_dir,
        source=source,
        validation_documents=args.validation_documents,
        seed=args.seed,
        chars_per_shard=args.chars_per_shard,
        row_group_size=args.row_group_size,
    )


if __name__ == "__main__":
    main()
