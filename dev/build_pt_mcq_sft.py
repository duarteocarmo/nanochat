"""Build and optionally publish the canonical European Portuguese MCQ SFT dataset."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, HfFileSystem


REPO_ID = "duarteocarmo/pt-mcq-sft"
SOURCES = {
    "mmlu": {
        "repo": "LumiOpen/opengpt-x_mmlux",
        "revision": "d4a99431a363e01cf9cba368b6ed0217cde7425e",
        "config": "*_PT-PT",
        "splits": "dev, validation, test",
    },
    "goldenswag": {
        "repo": "LumiOpen/opengpt-x_goldenswagx",
        "revision": "99d3939a3cbf82f523f418d866028f06b28c5fc2",
        "config": "PT-PT",
        "splits": "train",
    },
    "boolq": {
        "repo": "PORTULAN/extraglue",
        "revision": "d3d7ee103cc4961d735ad0e4b403b5718498fd43",
        "config": "boolq_pt-PT",
        "splits": "train",
    },
}


def canonical_row(*, row_id, question, context, choices, answer, category, source, source_config, source_split, source_id):
    return {
        "id": row_id,
        "question": question.strip(),
        "context": context.strip(),
        "choices": [choice.strip() for choice in choices],
        "answer": int(answer),
        "category": category,
        "source_repo": SOURCES[source]["repo"],
        "source_config": source_config,
        "source_split": source_split,
        "source_id": str(source_id),
        "source_revision": SOURCES[source]["revision"],
    }


def clean(*, rows, expected_choice_count):
    valid = []
    for row in rows:
        choices = row["choices"]
        if len(choices) != expected_choice_count:
            continue
        if not 0 <= row["answer"] < len(choices):
            continue
        if not row["question"] or any(not choice for choice in choices):
            continue
        valid.append(row)

    unique = []
    seen = set()
    for row in valid:
        key = json.dumps([row["question"], row["context"], row["choices"]], ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def load_mmlu():
    source = SOURCES["mmlu"]
    filesystem = HfFileSystem()
    pattern = f'datasets/{source["repo"]}@{source["revision"]}/*PT-PT*.jsonl'
    paths = filesystem.glob(pattern)
    rows = []
    for path in paths:
        match = re.fullmatch(r".*/hendrycks_(.+)_PT-PT_(dev|validation|test)\.jsonl", path)
        if match is None:
            raise ValueError(f"Unexpected MMLU path: {path}")
        subject, split = match.groups()
        with filesystem.open(path, "r") as source_file:
            for line in source_file:
                row = json.loads(line)
                rows.append(canonical_row(
                    row_id=f'mmlu:{split}:{row["id"]}',
                    question=row["question"],
                    context="",
                    choices=row["choices"],
                    answer=row["answer"],
                    category=subject,
                    source="mmlu",
                    source_config=f"{subject}_PT-PT",
                    source_split=split,
                    source_id=row["id"],
                ))
    return clean(rows=rows, expected_choice_count=4)


def load_goldenswag():
    source = SOURCES["goldenswag"]
    dataset = load_dataset(
        path=source["repo"],
        name=source["config"],
        split="train",
        revision=source["revision"],
    )
    rows = [
        canonical_row(
            row_id=f'goldenswag:{row["id"]}',
            question="Qual é a continuação mais plausível para o texto?",
            context=row["ctx"],
            choices=row["endings"],
            answer=row["label"],
            category=row["activity_label"],
            source="goldenswag",
            source_config=source["config"],
            source_split="train",
            source_id=row["id"],
        )
        for row in dataset
    ]
    return clean(rows=rows, expected_choice_count=4)


def load_boolq():
    source = SOURCES["boolq"]
    dataset = load_dataset(
        path=source["repo"],
        name=source["config"],
        split="train",
        revision=source["revision"],
    )
    rows = [
        canonical_row(
            row_id=f'boolq:{row["idx"]}',
            question=row["question"],
            context=row["passage"],
            choices=["sim", "não"],
            answer=0 if row["label"] == 1 else 1,
            category="boolq",
            source="boolq",
            source_config=source["config"],
            source_split="train",
            source_id=row["idx"],
        )
        for row in dataset
    ]
    return clean(rows=rows, expected_choice_count=2)


def validate(*, datasets):
    expected_choice_counts = {"mmlu": 4, "goldenswag": 4, "boolq": 2}
    for name, rows in datasets.items():
        if not rows:
            raise ValueError(f"{name} is empty")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} has duplicate IDs")
        for row in rows:
            if len(row["choices"]) != expected_choice_counts[name]:
                raise ValueError(f'{row["id"]} has {len(row["choices"])} choices')
            if not 0 <= row["answer"] < len(row["choices"]):
                raise ValueError(f'{row["id"]} has invalid answer {row["answer"]}')
            if not row["question"] or any(not choice for choice in row["choices"]):
                raise ValueError(f'{row["id"]} has empty text')


def dataset_card(*, datasets):
    counts = {name: len(rows) for name, rows in datasets.items()}
    distributions = {
        name: dict(sorted(Counter(row["answer"] for row in rows).items()))
        for name, rows in datasets.items()
    }
    return f'''---
language:
- pt
task_categories:
- question-answering
task_ids:
- multiple-choice-qa
license: other
pretty_name: European Portuguese MCQ SFT
configs:
- config_name: mmlu
  data_files:
  - split: train
    path: mmlu/train-*
- config_name: goldenswag
  data_files:
  - split: train
    path: goldenswag/train-*
- config_name: boolq
  data_files:
  - split: train
    path: boolq/train-*
---

# European Portuguese MCQ SFT

Canonical European Portuguese multiple-choice data for supervised fine-tuning. The three configs share one schema but remain separate so training code can weight and ablate them independently.

| Config | Rows | Choices | Source splits |
|---|---:|---:|---|
| `mmlu` | {counts["mmlu"]:,} | 4 | dev + validation + test |
| `goldenswag` | {counts["goldenswag"]:,} | 4 | train |
| `boolq` | {counts["boolq"]:,} | 2 | train |

Exact duplicates are removed within each config. Options remain in source order; training code should permute them deterministically and remap `answer`.

## Schema

- `id`: stable ID in this dataset
- `question`: question or task instruction
- `context`: optional passage or sentence prefix
- `choices`: answer options in source order
- `answer`: zero-based index into `choices`
- `category`: source subject or task category
- `source_repo`: immediate Hugging Face source
- `source_config`: source config
- `source_split`: original source split
- `source_id`: original row ID
- `source_revision`: pinned source commit

## Original sources

### MMLU PT-PT

- Immediate source: [LumiOpen/opengpt-x_mmlux](https://huggingface.co/datasets/LumiOpen/opengpt-x_mmlux), a code-free mirror of the OpenGPT-X translations
- Pinned revision: `{SOURCES["mmlu"]["revision"]}`
- Translation source: [Eurolingua/mmlux](https://huggingface.co/datasets/Eurolingua/mmlux), formerly `openGPT-X/mmlux`
- Original English dataset: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)
- Translation paper: [Towards Multilingual LLM Evaluation for European Languages](https://arxiv.org/abs/2410.08928)
- Translation method: DeepL, explicitly targeting PT-PT
- Included source splits: all PT-PT dev, validation, and test files across 57 subjects

**Contamination warning:** this config contains MMLU evaluation splits and overlaps EuroEval MMLU-pt. Do not use it to train a model that will claim clean MMLU or EuroEval MMLU results.

### GoldenSwag PT-PT

- Immediate source: [LumiOpen/opengpt-x_goldenswagx](https://huggingface.co/datasets/LumiOpen/opengpt-x_goldenswagx)
- Pinned revision: `{SOURCES["goldenswag"]["revision"]}`
- Original English dataset: [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)
- Translation paper: [Towards Multilingual LLM Evaluation for European Languages](https://arxiv.org/abs/2410.08928)
- Quality filtering paper: [What the HellaSwag?](https://arxiv.org/abs/2504.07825)
- Translation method: DeepL, explicitly targeting PT-PT
- Included source split: train only

The source validation split duplicates the train split and is intentionally excluded.

### BoolQ PT-PT

- Immediate source: [PORTULAN/extraglue](https://huggingface.co/datasets/PORTULAN/extraglue), config `boolq_pt-PT`
- Pinned revision: `{SOURCES["boolq"]["revision"]}`
- Original English task: [BoolQ](https://arxiv.org/abs/1905.10044)
- Translation paper: [ExtraGLUE](https://arxiv.org/abs/2404.05333)
- Translation method: DeepL, explicitly targeting European Portuguese
- Included source split: train only

BoolQ labels are converted to choices `["sim", "não"]`, with `answer` pointing to the correct choice.

## Transformations

1. Load each source at the pinned revision.
2. Keep only the source splits documented above.
3. Convert all rows to the shared schema.
4. Remove rows with empty questions or choices, invalid answer indices, or unexpected option counts.
5. Remove exact duplicates based on question, context, and choices.
6. Validate option counts and answer indices.

No model-generated content, option permutation, balancing, or oversampling is added here.

## Label distributions

- `mmlu`: `{json.dumps(distributions["mmlu"], sort_keys=True)}`
- `goldenswag`: `{json.dumps(distributions["goldenswag"], sort_keys=True)}`
- `boolq`: `{json.dumps(distributions["boolq"], sort_keys=True)}`

## Licensing and citations

This repository redistributes transformed rows from the sources above. Dataset rights and citation requirements remain with each original source. Review the linked source cards and papers before redistribution or commercial use. The repository uses `license: other` because the collection combines multiple derived datasets.
'''


def push(*, datasets, repo_id):
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    with TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory)
        for name, rows in datasets.items():
            config_dir = output_dir / name
            config_dir.mkdir()
            Dataset.from_list(mapping=rows).to_parquet(
                path_or_buf=config_dir / "train-00000-of-00001.parquet",
            )
        (output_dir / "README.md").write_text(data=dataset_card(datasets=datasets))
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Publish canonical PT-PT MCQ configs",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="publish the private dataset to Hugging Face")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    datasets = {
        "mmlu": load_mmlu(),
        "goldenswag": load_goldenswag(),
        "boolq": load_boolq(),
    }
    validate(datasets=datasets)
    for name, rows in datasets.items():
        distribution = dict(sorted(Counter(row["answer"] for row in rows).items()))
        print(f"{name}: {len(rows):,} rows, labels={distribution}")
    if args.push:
        push(datasets=datasets, repo_id=args.repo_id)
        print(f"Published private dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
