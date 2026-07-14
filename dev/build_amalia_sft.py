"""Build the standardized duarteocarmo/amalia-sft dataset.

The output has one config per source and train/validation Parquet splits.

Run:
    uv run python dev/build_amalia_sft.py

Smoke test:
    uv run python dev/build_amalia_sft.py --max-per-config 100
"""

import argparse
import json
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import numpy
import pyarrow as pyarrow
import pyarrow.json as json_reader
import pyarrow.parquet as parquet


DEFAULT_OUT_DIR = Path("dev-ignore/amalia-sft")
DEFAULT_CACHE_DIR = Path("dev-ignore/amalia-sft-cache")
SOURCES = [
    {
        "config": "pt_persona_instruction",
        "repo_id": "amalia-llm/persona_instruction_following",
        "filename": "if_output_pt_100k_verified_quality5.jsonl",
        "messages_field": "conversations",
        "source_config": "filtered",
        "source_split": "pt",
    },
    {
        "config": "pt_nemotron_instruction",
        "repo_id": "amalia-llm/persona_nemotron",
        "filename": "if_pt_nemo_quality_5_fix.jsonl",
        "messages_field": "conversations",
        "source_config": "instruction_following",
        "source_split": "train",
    },
    {
        "config": "pt_nemotron_general",
        "repo_id": "amalia-llm/persona_nemotron",
        "filename": "persona_general_200k_nemo_quality_5.jsonl",
        "messages_field": "conversations",
        "source_config": "general",
        "source_split": "train",
    },
    {
        "config": "pt_wikipedia",
        "repo_id": "amalia-llm/wikipedia_conversations",
        "filename": "wikipedia_conversations_fix.jsonl",
        "messages_field": "conversation",
        "source_config": "default",
        "source_split": "train",
    },
    {
        "config": "pt_culture",
        "repo_id": "amalia-llm/PT-Culture_Data",
        "filename": "culture_data_sft.jsonl",
        "messages_field": "conversations",
        "source_config": "default",
        "source_split": "train",
    },
    {
        "config": "pt_linguistics",
        "repo_id": "amalia-llm/ptpt-linguistics-if",
        "filename": "200_new_entries.jsonl",
        "messages_field": "conversations",
        "source_config": "default",
        "source_split": "train",
    },
]
SCHEMA = pyarrow.schema([
    ("id", pyarrow.string()),
    ("messages", pyarrow.list_(pyarrow.struct([
        ("role", pyarrow.string()),
        ("content", pyarrow.string()),
    ]))),
    ("source", pyarrow.string()),
    ("source_config", pyarrow.string()),
    ("source_split", pyarrow.string()),
    ("source_index", pyarrow.int64()),
])


def normalize_conversation(messages: list[dict]) -> dict:
    role_map = {
        "system": "system",
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
    }
    normalized = []
    for message in messages:
        source_role = message.get("role", message.get("from"))
        content = message.get("content", message.get("value"))
        normalized.append({"role": role_map.get(source_role, source_role), "content": content})
    alternating_messages = normalized[1:] if normalized and normalized[0]["role"] == "system" else normalized
    if len(alternating_messages) < 2:
        raise ValueError("Conversation must contain at least one user-assistant turn")
    for index, message in enumerate(alternating_messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected_role or not isinstance(message["content"], str):
            raise ValueError(f"Invalid message at position {index}")
    return {"messages": normalized}


def download_source(repo_id: str, filename: str, cache_dir: Path) -> Path:
    local_path = cache_dir / repo_id.replace("/", "--") / filename
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    quoted_filename = urllib.parse.quote(filename, safe="/")
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quoted_filename}"
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url=url, headers=headers)
    temporary_path = local_path.with_suffix(local_path.suffix + ".incomplete")
    with urllib.request.urlopen(request, timeout=120) as response, temporary_path.open("wb") as file:
        shutil.copyfileobj(fsrc=response, fdst=file)
    temporary_path.replace(local_path)
    print(f"Downloaded {repo_id}/{filename}")
    return local_path


def normalized_rows(source: dict, cache_dir: Path, max_per_config: int | None) -> list[dict]:
    source_path = download_source(
        repo_id=source["repo_id"],
        filename=source["filename"],
        cache_dir=cache_dir,
    )
    table = json_reader.read_json(source_path)
    messages_column = table[source["messages_field"]]
    rows = []
    seen = set()
    invalid_count = 0
    duplicate_count = 0
    for source_index in range(len(messages_column)):
        try:
            conversation = normalize_conversation(messages=messages_column[source_index].as_py())
        except ValueError:
            invalid_count += 1
            continue
        key = json.dumps(conversation["messages"], ensure_ascii=False, sort_keys=True)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        rows.append({
            "id": f"{source['config']}_{source_index:06d}",
            "messages": conversation["messages"],
            "source": source["repo_id"],
            "source_config": source["source_config"],
            "source_split": source["source_split"],
            "source_index": source_index,
        })
        if max_per_config is not None and len(rows) >= max_per_config:
            break
    print(f"Prepared {len(rows):,} rows for {source['config']} after filtering {invalid_count:,} invalid and {duplicate_count:,} duplicate rows")
    return rows


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    permutation = numpy.random.default_rng(seed=42).permutation(len(rows))
    shuffled = [rows[int(index)] for index in permutation]
    train_size = int(len(shuffled) * 0.98)
    return shuffled[:train_size], shuffled[train_size:]


def write_dataset(config_rows: dict[str, dict[str, list[dict]]], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for config, splits in config_rows.items():
        config_dir = data_dir / config
        config_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in splits.items():
            path = config_dir / f"{split}-00000-of-00001.parquet"
            table = pyarrow.Table.from_pylist(rows, schema=SCHEMA)
            parquet.write_table(table, path, compression="zstd")
            verified = parquet.read_table(path)
            if verified.schema != SCHEMA or verified.num_rows != len(rows):
                raise RuntimeError(f"Failed verification for {path}")
            print(f"Wrote {len(rows):,} rows to {path}")

    readme = [
        "---",
        "language:",
        "- pt",
        "task_categories:",
        "- text-generation",
        "configs:",
    ]
    for config in config_rows:
        readme.extend([
            f"- config_name: {config}",
            "  data_files:",
            "  - split: train",
            f"    path: data/{config}/train-*.parquet",
            "  - split: validation",
            f"    path: data/{config}/validation-*.parquet",
        ])
    readme.extend([
        "---",
        "",
        "# AMALIA SFT",
        "",
        "Standardized European Portuguese supervised fine-tuning data from AMALIA.",
        "",
        "Each config has deterministic `train` and `validation` splits and a common `messages` schema.",
        "",
        "Exact duplicate and invalid conversations are removed. Source provenance is retained in every row.",
        "",
        "## Counts",
        "",
        "| Config | Train | Validation |",
        "|---|---:|---:|",
    ])
    for config, splits in config_rows.items():
        readme.append(f"| `{config}` | {len(splits['train'])} | {len(splits['validation'])} |")
    readme.append("")
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote AMALIA SFT to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AMALIA SFT")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-per-config", type=int, default=None)
    args = parser.parse_args()

    config_rows = {}
    for source in SOURCES:
        rows = normalized_rows(
            source=source,
            cache_dir=args.cache_dir,
            max_per_config=args.max_per_config,
        )
        train_rows, validation_rows = split_rows(rows=rows)
        config_rows[source["config"]] = {
            "train": train_rows,
            "validation": validation_rows,
        }
    write_dataset(config_rows=config_rows, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
