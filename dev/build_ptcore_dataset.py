"""
Build a standardized PTCORE dataset from Portuguese evaluation sources.

The output is one Hugging Face dataset repo with one config/subset per PTCORE task.
Each config has a validation split with the same schema.

Output schema:
- id: stable row id
- task: normalized task name
- source: original Hugging Face dataset
- source_config: original config/subset
- source_split: original split
- question: prompt/question text
- choices: list[str]
- correct_choice: zero-based index into choices
- metadata: JSON string with source-specific fields

Run:
    uv run python dev/build_ptcore_dataset.py

Smoke test:
    uv run python dev/build_ptcore_dataset.py --max-per-task 5

Optional upload:
    uv run python dev/build_ptcore_dataset.py --push-to-hub --repo-id duarteocarmo/ptcore-eval
"""

import argparse
import json
import re
import shutil
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

import pyarrow as pyarrow
import pyarrow.parquet as parquet


DEFAULT_OUT_DIR = Path("dev-ignore/ptcore")
DEFAULT_CACHE_DIR = Path("dev-ignore/ptcore-cache")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def dataset_parquet_manifest(repo_id: str) -> dict:
    return fetch_json(f"https://huggingface.co/api/datasets/{repo_id}/parquet")


def parquet_urls_for(repo_id: str, config: str, split: str) -> list[str]:
    manifest = dataset_parquet_manifest(repo_id=repo_id)
    try:
        return manifest[config][split]
    except KeyError as error:
        available = {config_name: sorted(splits) for config_name, splits in manifest.items()}
        raise KeyError(f"Missing {repo_id}/{config}/{split}. Available: {available}") from error


def load_rows(repo_id: str, config: str, split: str, cache_dir: Path) -> list[dict]:
    urls = parquet_urls_for(repo_id=repo_id, config=config, split=split)
    shard_dir = cache_dir / repo_id.replace("/", "--") / config / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for shard_index, url in enumerate(urls):
        shard_path = shard_dir / f"{shard_index:05d}.parquet"
        if not shard_path.exists():
            print(f"Downloading {url} ...")
            with urllib.request.urlopen(url, timeout=120) as response:
                shard_path.write_bytes(response.read())
        shard_paths.append(shard_path)
    tables = [parquet.read_table(path) for path in shard_paths]
    table = pyarrow.concat_tables(tables)
    return table.to_pylist()


def clean_task_name(value: str) -> str:
    value = value.lower().replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def normalized_row(
    *,
    row_id: str,
    task: str,
    source: str,
    source_config: str,
    source_split: str,
    question: str,
    choices: list[str],
    correct_choice: int,
    metadata: dict | None = None,
) -> dict:
    choices = [str(choice).strip() for choice in choices]
    if not question or not choices:
        raise ValueError(f"Invalid empty question/choices for {row_id}")
    if not 0 <= correct_choice < len(choices):
        raise ValueError(f"Invalid correct_choice={correct_choice} for {row_id} with {len(choices)} choices")
    return {
        "id": row_id,
        "task": task,
        "source": source,
        "source_config": source_config,
        "source_split": source_split,
        "question": question.strip(),
        "choices": choices,
        "correct_choice": correct_choice,
        "metadata": json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
    }


def adapt_sst2(row: dict, index: int, source: str, config: str, split: str) -> dict:
    label = row["label"].strip().lower()
    label_to_index = {"negative": 0, "positive": 1, "negativo": 0, "positivo": 1}
    return normalized_row(
        row_id=f"sst2_pt_mini_{split}_{index:06d}",
        task="sst2_pt_mini",
        source=source,
        source_config=config,
        source_split=split,
        question=f"Qual é o sentimento expresso no texto?\n\nTexto: {row['text']}",
        choices=["negativo", "positivo"],
        correct_choice=label_to_index[label],
        metadata={"original_label": row["label"]},
    )


def adapt_scala(row: dict, index: int, source: str, config: str, split: str) -> dict:
    label = row["label"].strip().lower()
    return normalized_row(
        row_id=f"scala_pt_{split}_{index:06d}",
        task="scala_pt",
        source=source,
        source_config=config,
        source_split=split,
        question=f"A frase seguinte está escrita corretamente em português?\n\nFrase: {row['text']}",
        choices=["não", "sim"],
        correct_choice=1 if label == "correct" else 0,
        metadata={"original_label": row["label"], "corruption_type": row.get("corruption_type")},
    )


def adapt_portugal_basic_qa(row: dict, index: int, source: str, config: str, split: str) -> dict:
    choices = row["choices"]
    label = row.get("label")
    if label:
        correct_choice = LETTERS.lower().index(label.lower())
    else:
        correct_choice = choices.index(row["answer"])
    return normalized_row(
        row_id=f"portugal_basic_qa_{split}_{index:06d}",
        task="portugal_basic_qa",
        source=source,
        source_config=config,
        source_split=split,
        question=row["question"],
        choices=choices,
        correct_choice=correct_choice,
        metadata={"answer": row.get("answer"), "label": row.get("label")},
    )


def adapt_alba(row: dict, index: int, source: str, config: str, split: str) -> dict:
    return normalized_row(
        row_id=f"alba_mcq_{clean_task_name(config)}_{index:06d}",
        task=f"alba_mcq_{clean_task_name(config)}",
        source=source,
        source_config=config,
        source_split=split,
        question=row["question"],
        choices=row["choices"],
        correct_choice=int(row["correct_choice"]),
        metadata={"subject": row.get("subject"), "scores": row.get("scores"), "source_id": row.get("id")},
    )


def adapt_cultura_viva(row: dict, index: int, source: str, config: str, split: str) -> dict:
    choices = row["choices"]["text"]
    labels = row["choices"]["label"]
    correct_choice = labels.index(row["answerKey"])
    return normalized_row(
        row_id=f"cultura_viva_pt_mcq_{index:06d}",
        task="cultura_viva_pt_mcq",
        source=source,
        source_config=config,
        source_split=split,
        question=row["question"],
        choices=choices,
        correct_choice=correct_choice,
        metadata={
            "source_id": row.get("id"),
            "answerKey": row.get("answerKey"),
            "domain": row.get("domain"),
            "task_type": row.get("_task_type"),
        },
    )


def adapt_pt_exams(row: dict, index: int, source: str, config: str, split: str) -> dict:
    return normalized_row(
        row_id=f"pt_exams_{clean_task_name(config)}_{index:06d}",
        task=f"pt_exams_{clean_task_name(row.get('subject') or config)}",
        source=source,
        source_config=config,
        source_split=split,
        question=row["question"],
        choices=row["choices"],
        correct_choice=int(row["answer"]),
        metadata={
            "year": row.get("year"),
            "phase": row.get("phase"),
            "subject": row.get("subject"),
            "question_group": row.get("question_group"),
            "question_number": row.get("question_number"),
            "is_completion": row.get("is_completion"),
        },
    )


def adapt_piqa(row: dict, index: int, source: str, config: str, split: str) -> dict | None:
    label = int(row["label"])
    if label not in {0, 1}:
        return None
    return normalized_row(
        row_id=f"piqa_mt_pt_{split}_{index:06d}",
        task="piqa_mt_pt",
        source=source,
        source_config=config,
        source_split=split,
        question=f"Qual é a melhor solução para o objetivo seguinte?\n\nObjetivo: {row['goal']}",
        choices=[row["sol1"], row["sol2"]],
        correct_choice=label,
        metadata={"goal": row.get("goal")},
    )


TaskAdapter = Callable[[dict, int, str, str, str], dict | None]


SOURCES: list[dict] = [
    {
        "repo_id": "duarteocarmo/sst2-pt-mini",
        "config": "default",
        "split": "test",
        "adapter": adapt_sst2,
    },
    {
        "repo_id": "duarteocarmo/scala-pt",
        "config": "default",
        "split": "test",
        "adapter": adapt_scala,
    },
    {
        "repo_id": "duarteocarmo/portugal-basic-qa-ptcore",
        "config": "default",
        "split": "val",
        "adapter": adapt_portugal_basic_qa,
    },
    *[
        {
            "repo_id": "amalia-llm/alba_mcq",
            "config": config,
            "split": "test",
            "adapter": adapt_alba,
        }
        for config in [
            "culture_bound_semantics",
            "discourse_analysis",
            "language_variety",
            "morphology",
            "phonetics_phonology",
            "syntax",
            "word_play",
        ]
    ],
    {
        "repo_id": "amalia-llm/cultura-viva-pt-mcq",
        "config": "default",
        "split": "train",
        "adapter": adapt_cultura_viva,
    },
    {
        "repo_id": "amalia-llm/pt_exams",
        "config": "default",
        "split": "test",
        "adapter": adapt_pt_exams,
    },
    {
        "repo_id": "amalia-llm/piqa-mt-pt",
        "config": "default",
        "split": "validation",
        "adapter": adapt_piqa,
    },
]


def iter_normalized_rows(*, cache_dir: Path, max_per_task: int | None = None) -> Iterable[dict]:
    counts: dict[str, int] = {}
    for source_spec in SOURCES:
        repo_id = source_spec["repo_id"]
        config = source_spec["config"]
        split = source_spec["split"]
        adapter: TaskAdapter = source_spec["adapter"]
        print(f"Loading {repo_id} / {config} / {split}")
        source_rows = load_rows(repo_id=repo_id, config=config, split=split, cache_dir=cache_dir)
        for index, source_row in enumerate(source_rows):
            row = adapter(source_row, index, repo_id, config, split)
            if row is None:
                continue
            task = row["task"]
            if max_per_task is not None and counts.get(task, 0) >= max_per_task:
                continue
            counts[task] = counts.get(task, 0) + 1
            yield row


def group_rows_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        task = row["task"]
        grouped.setdefault(task, []).append(row)
    return dict(sorted(grouped.items()))


def write_dataset(rows: list[dict], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    grouped_rows = group_rows_by_task(rows=rows)
    for task, task_rows in grouped_rows.items():
        task_dir = data_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = task_dir / "validation-00000-of-00001.parquet"
        parquet.write_table(pyarrow.Table.from_pylist(task_rows), parquet_path)
        print(f"Wrote {len(task_rows):,} rows to {parquet_path}")

    preview_path = out_dir / "preview.jsonl"
    with preview_path.open("w", encoding="utf-8") as file:
        for task_rows in grouped_rows.values():
            for row in task_rows[:3]:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    readme = [
        "---",
        "configs:",
    ]
    for task in grouped_rows:
        readme.extend([
            f"- config_name: {task}",
            "  data_files:",
            "  - split: validation",
            f"    path: data/{task}/validation-*.parquet",
        ])
    readme.extend([
        "---",
        "",
        "# PTCORE",
        "",
        "Standardized European Portuguese CORE-style multiple-choice evaluation dataset.",
        "",
        "Each config is one PTCORE task and exposes a `validation` split.",
        "",
        "## Schema",
        "",
        "- `id`: stable row id",
        "- `task`: task/subtask name",
        "- `source`: original Hugging Face dataset",
        "- `source_config`: original config/subset",
        "- `source_split`: original split",
        "- `question`: prompt/question text",
        "- `choices`: answer options",
        "- `correct_choice`: zero-based index into `choices`",
        "- `metadata`: source-specific JSON string",
        "",
        "## Counts",
        "",
        "| Config / task | Rows |",
        "|---|---:|",
    ])
    for task, task_rows in grouped_rows.items():
        readme.append(f"| `{task}` | {len(task_rows)} |")
    readme.append("")
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Wrote {len(rows):,} rows across {len(grouped_rows):,} configs to {out_dir}")
    print(f"Wrote preview to {preview_path}")


def push_to_hub(out_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset")
    print(f"Uploaded {out_dir} to https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standardized PTCORE dataset")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-per-task", type=int, default=None, help="Limit rows per task for smoke tests")
    parser.add_argument("--repo-id", type=str, default="duarteocarmo/ptcore-eval")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    rows = list(iter_normalized_rows(cache_dir=args.cache_dir, max_per_task=args.max_per_task))
    if not rows:
        raise RuntimeError("No PTCORE rows were produced")
    write_dataset(rows=rows, out_dir=args.out_dir)
    if args.push_to_hub:
        push_to_hub(out_dir=args.out_dir, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
