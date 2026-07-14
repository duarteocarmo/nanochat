"""PTCORE evaluation for base models."""

import glob
import os
import random
import time
from pathlib import Path

import pyarrow
import pyarrow.parquet as pyarrow_parquet

from nanochat.common import print0
from nanochat.core_eval import evaluate_task
from tasks.common import load_hub_dataset


PTCORE_REPO_ID = "duarteocarmo/ptcore-eval"
PTCORE_TASKS = [
    {"name": "sst2_pt_mini", "num_fewshot": 0},
    {"name": "scala_pt", "num_fewshot": 0},
    {"name": "portugal_basic_qa", "num_fewshot": 0},
    {"name": "alba_mcq_culture_bound_semantics", "num_fewshot": 0},
    {"name": "alba_mcq_discourse_analysis", "num_fewshot": 0},
    {"name": "alba_mcq_language_variety", "num_fewshot": 0},
    {"name": "alba_mcq_morphology", "num_fewshot": 0},
    {"name": "alba_mcq_phonetics_phonology", "num_fewshot": 0},
    {"name": "alba_mcq_syntax", "num_fewshot": 0},
    {"name": "alba_mcq_word_play", "num_fewshot": 0},
    {"name": "cultura_viva_pt_mcq", "num_fewshot": 0},
    {"name": "pt_exams_bio_geo", "num_fewshot": 0},
    {"name": "pt_exams_geography", "num_fewshot": 0},
    {"name": "pt_exams_history_a", "num_fewshot": 0},
    {"name": "pt_exams_mathematics_a", "num_fewshot": 0},
    {"name": "pt_exams_philosophy", "num_fewshot": 0},
    {"name": "pt_exams_portuguese", "num_fewshot": 0},
    {"name": "piqa_mt_pt", "num_fewshot": 0},
]


def default_local_ptcore_dir() -> Path | None:
    env_path = os.environ.get("NANOCHAT_PTCORE_DIR")
    if env_path:
        return Path(env_path)
    local_path = Path("dev-ignore/ptcore")
    return local_path if local_path.exists() else None


def load_local_task_rows(task_name: str, local_dir: Path) -> list[dict]:
    pattern = local_dir / "data" / task_name / "validation-*.parquet"
    paths = sorted(glob.glob(str(pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No PTCORE parquet files found for {task_name} at {pattern}"
        )
    tables = [pyarrow_parquet.read_table(path) for path in paths]
    return pyarrow.concat_tables(tables).to_pylist()


def load_hub_task_rows(task_name: str, repo_id: str) -> list[dict]:
    dataset = load_hub_dataset(repo_id=repo_id, subset=task_name, split="validation")
    return [dataset[index] for index in range(len(dataset))]


def load_ptcore_task(
    task_name: str, repo_id: str, local_dir: Path | None
) -> list[dict]:
    rows = (
        load_local_task_rows(task_name=task_name, local_dir=local_dir)
        if local_dir
        else load_hub_task_rows(task_name=task_name, repo_id=repo_id)
    )
    return [
        {
            "query": row["question"],
            "choices": row["choices"],
            "gold": int(row["correct_choice"]),
        }
        for row in rows
    ]


def random_baseline_for(data: list[dict]) -> float:
    return sum(1 / len(row["choices"]) for row in data) / len(data)


def evaluate_ptcore(
    model, tokenizer, device, max_per_task=-1, repo_id: str = PTCORE_REPO_ID
) -> dict:
    """
    Evaluate a base model on PTCORE.

    Returns dict with results, centered_results, and core_metric.
    """
    local_dir = default_local_ptcore_dir()
    if local_dir:
        print0(f"Loading PTCORE from local directory: {local_dir}")
    else:
        print0(f"Loading PTCORE from Hugging Face: {repo_id}")

    results = {}
    centered_results = {}

    for task_config in PTCORE_TASKS:
        task_name = task_config["name"]
        task_meta = {
            "task_type": "multiple_choice",
            "num_fewshot": task_config["num_fewshot"],
            "continuation_delimiter": task_config.get("continuation_delimiter", " "),
        }
        start_time = time.time()
        print0(f"Evaluating: {task_name} ({task_meta['num_fewshot']}-shot, type: multiple_choice)... ", end="")

        data = load_ptcore_task(
            task_name=task_name, repo_id=repo_id, local_dir=local_dir
        )
        total_examples = len(data)
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy, stats = evaluate_task(model, tokenizer, data, device, task_meta, return_stats=True)
        stats["num_total"] = total_examples
        baseline = random_baseline_for(data=data)
        centered_result = (accuracy - baseline) / (1.0 - baseline)
        results[task_name] = accuracy
        centered_results[task_name] = centered_result

        elapsed = time.time() - start_time
        print0(
            f"examples selected/available: {stats['num_eval']}/{stats['num_total']} | accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s"
        )

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }
