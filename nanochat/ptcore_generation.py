"""Few-shot, generated-answer PTCORE evaluation for base models."""

import csv
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist

from nanochat.common import is_ddp_initialized, print0
from nanochat.engine import Engine
from nanochat.ptcore_eval import (
    PTCORE_REPO_ID,
    PTCORE_TASKS,
    default_local_ptcore_dir,
    load_ptcore_task,
    random_baseline_for,
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ANSWER_PATTERN = re.compile(
    r"(?:Resposta\s*:\s*)?(?:([A-Z])[.)]?|\(([A-Z])\))",
    flags=re.IGNORECASE | re.ASCII,
)


def parse_ptcore_answer(completion: str, num_choices: int) -> int | None:
    match = ANSWER_PATTERN.fullmatch(string=completion.strip())
    if match is None:
        return None
    letter = (match.group(1) or match.group(2)).upper()
    index = LETTERS.index(letter)
    return index if index < num_choices else None


def select_generation_examples(
    data: list[dict], num_fewshot: int, max_per_task: int
) -> tuple[list[dict], list[dict]]:
    if num_fewshot < 0:
        raise ValueError("num_fewshot must be non-negative")
    shuffled = list(data)
    random.Random(x=1337).shuffle(x=shuffled)
    demonstrations = []
    demo_questions = set()
    for row in reversed(shuffled):
        if len(demonstrations) == num_fewshot:
            break
        if row["query"] not in demo_questions:
            demonstrations.append(row)
            demo_questions.add(row["query"])
    questions = [row for row in shuffled if row["query"] not in demo_questions]
    if len(demonstrations) != num_fewshot or not questions:
        raise ValueError(
            "PTCORE needs enough distinct questions for demonstrations and evaluation"
        )
    if max_per_task > 0:
        questions = questions[:max_per_task]
    return demonstrations, questions


def render_generation_prompt(row: dict, demonstrations: list[dict]) -> str:
    blocks = ["Responde apenas com a letra da opção correta."]
    for index, example in enumerate([*demonstrations, row]):
        if not 2 <= len(example["choices"]) <= len(LETTERS):
            raise ValueError("PTCORE questions must have between 2 and 26 choices")
        choices = "\n".join(
            f"{letter}) {choice}" for letter, choice in zip(LETTERS, example["choices"])
        )
        answer = f" {LETTERS[example['gold']]}" if index < len(demonstrations) else ""
        blocks.append(f"Pergunta: {example['query']}\n{choices}\nResposta:{answer}")
    return "\n\n".join(blocks)


def encode_generation_prompt(
    row: dict, demonstrations: list[dict], tokenizer, max_prompt_tokens: int
) -> tuple[list[int], int]:
    for num_fewshot in range(len(demonstrations), -1, -1):
        prompt = render_generation_prompt(
            row=row, demonstrations=demonstrations[:num_fewshot]
        )
        tokens = tokenizer.encode(text=prompt, prepend=tokenizer.get_bos_token_id())
        if len(tokens) <= max_prompt_tokens:
            return tokens, num_fewshot
    return [], 0


def generate_answer_line(engine: Engine, tokens: list[int], max_new_tokens: int) -> str:
    terminal_ids = {
        engine.tokenizer.get_bos_token_id(),
        engine.tokenizer.encode_special(text="<|assistant_end|>"),
    }
    generated = []
    completion = ""
    for token_column, _ in engine.generate(
        tokens=tokens, num_samples=1, max_tokens=max_new_tokens, temperature=0.0
    ):
        if token_column[0] in terminal_ids:
            break
        generated.append(token_column[0])
        completion = engine.tokenizer.decode(ids=generated).lstrip()
        if "\n" in completion or "\r" in completion:
            return completion.splitlines()[0].strip()
    return completion.strip()


def evaluate_ptcore_generation(
    model,
    tokenizer,
    device,
    max_per_task: int = -1,
    num_fewshot: int = 3,
    max_new_tokens: int = 16,
    repo_id: str = PTCORE_REPO_ID,
) -> dict:
    if num_fewshot < 0:
        raise ValueError("num_fewshot must be non-negative")
    if not 0 < max_new_tokens < model.config.sequence_len:
        raise ValueError(
            "max_new_tokens must be positive and smaller than the model context"
        )
    distributed = is_ddp_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    local_dir = default_local_ptcore_dir()
    engine = Engine(model=model, tokenizer=tokenizer)
    tasks: dict[str, dict] = {}
    samples = []

    for task in PTCORE_TASKS:
        task_name = task["name"]
        start_time = time.monotonic()
        data = load_ptcore_task(
            task_name=task_name, repo_id=repo_id, local_dir=local_dir
        )
        demonstrations, questions = select_generation_examples(
            data=data, num_fewshot=num_fewshot, max_per_task=max_per_task
        )
        task_samples: list[dict] = []
        for index in range(rank, len(questions), world_size):
            row = questions[index]
            tokens, shots = encode_generation_prompt(
                row=row,
                demonstrations=demonstrations,
                tokenizer=tokenizer,
                max_prompt_tokens=model.config.sequence_len - max_new_tokens,
            )
            completion = (
                generate_answer_line(
                    engine=engine, tokens=tokens, max_new_tokens=max_new_tokens
                )
                if tokens
                else ""
            )
            prediction = parse_ptcore_answer(
                completion=completion, num_choices=len(row["choices"])
            )
            task_samples.append(
                {
                    "task": task_name,
                    "index": index,
                    "question": row["query"],
                    "choices": row["choices"],
                    "answer": LETTERS[row["gold"]],
                    "completion": completion,
                    "prediction": LETTERS[prediction]
                    if prediction is not None
                    else None,
                    "correct": prediction == row["gold"],
                    "fewshot": shots,
                    "context_overflow": not tokens,
                }
            )
        samples.extend(task_samples)
        counts = torch.tensor(
            data=[
                sum(sample["correct"] for sample in task_samples),
                sum(sample["prediction"] is None for sample in task_samples),
                len(task_samples),
                sum(sample["fewshot"] for sample in task_samples),
                sum(sample["context_overflow"] for sample in task_samples),
            ],
            dtype=torch.long,
            device=device,
        )
        if distributed:
            dist.all_reduce(tensor=counts, op=dist.ReduceOp.SUM)
        correct, invalid, evaluated, shots_used, context_overflow = counts.tolist()
        accuracy = correct / evaluated
        baseline = random_baseline_for(data=questions)
        tasks[task_name] = {
            "accuracy": accuracy,
            "centered": (accuracy - baseline) / (1 - baseline),
            "invalid_rate": invalid / evaluated,
            "num_eval": evaluated,
            "num_total": len(data),
            "mean_fewshot": shots_used / evaluated,
            "context_overflow": context_overflow,
            "demonstrations": demonstrations,
        }
        print0(
            f"PTCORE generation {task_name}: {evaluated}/{len(data)} examples | "
            f"accuracy: {accuracy:.4f} | invalid: {invalid / evaluated:.4f} | "
            f"context overflow: {context_overflow} | time: {time.monotonic() - start_time:.2f}s"
        )

    if distributed:
        gathered: list | None = [None] * world_size if rank == 0 else None
        dist.gather_object(obj=samples, object_gather_list=gathered, dst=0)
        samples = [sample for group in gathered for sample in group] if gathered else []
    samples.sort(key=lambda sample: (sample["task"], sample["index"]))
    return {
        "tasks": tasks,
        "generation_metric": sum(task["centered"] for task in tasks.values())
        / len(tasks),
        "invalid_rate": sum(task["invalid_rate"] for task in tasks.values())
        / len(tasks),
        "settings": {
            "repo_id": repo_id,
            "fewshot": num_fewshot,
            "max_new_tokens": max_new_tokens,
            "seed": 1337,
        },
        "samples": samples,
    }


def save_generation_results(results: dict, output_prefix: str) -> list[str]:
    json_path = Path(f"{output_prefix}.json")
    csv_path = Path(f"{output_prefix}.csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        data=json.dumps(obj=results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = {
        "Task": "task",
        "Accuracy": "accuracy",
        "Centered": "centered",
        "InvalidRate": "invalid_rate",
        "Examples": "num_eval",
        "MeanFewshot": "mean_fewshot",
        "ContextOverflow": "context_overflow",
    }
    rows = [{"task": name, **result} for name, result in results["tasks"].items()]
    rows.append(
        {
            "task": "PTCORE-Generation",
            "centered": results["generation_metric"],
            "invalid_rate": results["invalid_rate"],
        }
    )
    with csv_path.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(f=file, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(
            rowdicts=(
                {title: row.get(field, "") for title, field in columns.items()}
                for row in rows
            )
        )
    return [str(json_path), str(csv_path)]


def generation_metrics_for_wandb(results: dict) -> dict:
    metrics = {
        "ptcore_generation/metric": results["generation_metric"],
        "ptcore_generation/invalid_rate": results["invalid_rate"],
    }
    for task_name, result in results["tasks"].items():
        for metric in ("accuracy", "invalid_rate"):
            metrics[f"ptcore_generation/{task_name}/{metric}"] = result[metric]
    return metrics
