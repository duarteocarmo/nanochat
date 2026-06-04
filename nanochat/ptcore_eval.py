"""Portuguese CORE-style evaluation using EuroEval PT datasets."""

import random
import time

from datasets import load_dataset

from nanochat.common import print0
from nanochat.core_eval import evaluate_task

PTCORE_TASKS = [
    {
        "label": "sst2-pt",
        "dataset_id": "duarteocarmo/sst2-pt-mini",
        "random_baseline": 0.50,
    },
    {
        "label": "scala-pt",
        "dataset_id": "duarteocarmo/scala-pt",
        "random_baseline": 0.50,
    },
    {
        "label": "mmlu-pt",
        "dataset_id": "duarteocarmo/mmlu-pt-mini",
        "random_baseline": 0.25,
    },
    {
        "label": "goldenswag-pt",
        "dataset_id": "duarteocarmo/goldenswag-pt-mini",
        "random_baseline": 0.25,
    },
]

LABEL_CHOICES = {
    "sst2-pt": (["positivo", "negativo"], {"positive": 0, "negative": 1}),
    "scala-pt": (["correta", "incorreta"], {"correct": 0, "incorrect": 1}),
}
LETTER_CHOICES = ["a", "b", "c", "d"]
LETTER_LABELS = {label: idx for idx, label in enumerate(LETTER_CHOICES)}


def convert_ptcore_row(task_label, row):
    if task_label == "sst2-pt":
        choices, labels = LABEL_CHOICES[task_label]
        return {
            "query": f"Texto: {row['text']}\nSentimento:",
            "choices": choices,
            "gold": labels[row["label"]],
        }

    if task_label == "scala-pt":
        choices, labels = LABEL_CHOICES[task_label]
        return {
            "query": f"Frase: {row['text']}\nA frase está:",
            "choices": choices,
            "gold": labels[row["label"]],
        }

    if task_label in {"mmlu-pt", "goldenswag-pt"}:
        return {
            "query": f"{row['text']}\nResposta:",
            "choices": LETTER_CHOICES,
            "gold": LETTER_LABELS[row["label"]],
        }

    raise ValueError(f"Unsupported PTCORE task: {task_label}")


def load_ptcore_task_data(task, split):
    dataset = load_dataset(path=task["dataset_id"], split=split, token=True)
    return [convert_ptcore_row(task_label=task["label"], row=row) for row in dataset]


def evaluate_ptcore(model, tokenizer, device, max_per_task=-1, split="val"):
    """Evaluate a base model on the Portuguese CORE-style benchmark."""
    results = {}
    centered_results = {}
    task_meta = {
        "task_type": "multiple_choice",
        "num_fewshot": 0,
        "continuation_delimiter": " ",
    }

    for task in PTCORE_TASKS:
        start_time = time.time()
        label = task["label"]
        print0(
            f"Evaluating: {label} (split: {split}, 0-shot, type: multiple_choice)... ",
            end="",
        )

        data = load_ptcore_task_data(task=task, split=split)
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        random_baseline = task["random_baseline"]
        centered_result = (accuracy - random_baseline) / (1.0 - random_baseline)
        results[label] = accuracy
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(
            f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s"
        )

    ptcore_metric = sum(centered_results.values()) / len(centered_results)
    return {
        "results": results,
        "centered_results": centered_results,
        "ptcore_metric": ptcore_metric,
    }
