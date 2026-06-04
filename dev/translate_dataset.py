#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=4.0.0",
#     "httpx>=0.28.0",
#     "polars>=1.0.0",
#     "pyarrow>=18.0.0",
#     "tqdm>=4.66.0",
# ]
# ///

import argparse
import json
import random
import re
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

import httpx
import polars
from datasets import DatasetDict, load_dataset
from tqdm import tqdm


DATASET_NAME = "HuggingFaceTB/smoltalk2"
DATASET_CONFIG = "SFT"
DATASET_SPLIT = "smoltalk_smollm3_everyday_conversations_no_think"
DEFAULT_ENDPOINT = "http://127.0.0.1:18000/v1/chat/completions"
DEFAULT_MODEL = "translategemma-12b-it"
DEFAULT_HUB_OWNER = "duarteocarmo"
DATASETS_SERVER_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def slug_for(*, split: str, target: str) -> str:
    if split == DATASET_SPLIT and target == "pt-PT":
        return "smoltalk2-everyday-conversations-no-think-pt-pt"

    base = split
    for prefix in ("smoltalk_smollm3_", "smoltalk_"):
        if base.startswith(prefix):
            base = base.removeprefix(prefix)
            break

    split_slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    target_slug = re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-")
    return f"smoltalk2-{split_slug}-{target_slug}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate any SmolTalk2 SFT split to pt-PT with local vLLM.",
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--config", default=DATASET_CONFIG)
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument("--output-split", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hub-owner", default=DEFAULT_HUB_OWNER)
    parser.add_argument("--hub-dataset", default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default="en")
    parser.add_argument("--target", default="pt-PT")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-page-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--worker-candidates", default="4,8,16,32,64")
    parser.add_argument("--warmup-rows", type=int, default=16)
    parser.add_argument("--push-every", type=int, default=1_000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--check-examples", type=int, default=20)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--translate-custom-instructions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_slug = slug_for(split=args.split, target=args.target)
    if args.output_split is None:
        args.output_split = args.split
    if args.output_dir is None:
        args.output_dir = f"datasets/{output_slug}"
    if args.hub_dataset is None:
        args.hub_dataset = f"{args.hub_owner}/{output_slug}"
    return args


def count_jsonl_rows(*, path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as file:
        return sum(1 for _ in file)


def prompt_for(*, text: str, source: str, target: str) -> str:
    return f"<<<source>>>{source}<<<target>>>{target}<<<text>>>{text}"


def translate_text(
    *,
    client: httpx.Client,
    endpoint: str,
    model: str,
    text: str,
    source: str,
    target: str,
    temperature: float,
    retries: int,
) -> str:
    if not text.strip():
        return text

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt_for(text=text, source=source, target=target),
            }
        ],
        "temperature": temperature,
    }

    last_error = None
    for attempt in range(retries):
        try:
            response = client.post(url=endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2**attempt)

    raise RuntimeError(f"Translation failed after {retries} retries: {last_error}")


def translate_cached_text(
    *,
    client: httpx.Client,
    endpoint: str,
    model: str,
    text: str,
    source: str,
    target: str,
    temperature: float,
    retries: int,
    cache: dict[tuple[str, str, str], str],
    cache_lock: Lock,
) -> str:
    cache_key = (source, target, text)
    with cache_lock:
        if cache_key in cache:
            return cache[cache_key]

    translated = translate_text(
        client=client,
        endpoint=endpoint,
        model=model,
        text=text,
        source=source,
        target=target,
        temperature=temperature,
        retries=retries,
    )
    with cache_lock:
        cache[cache_key] = translated
    return translated


def translate_chat_template_kwargs(
    *,
    client: httpx.Client,
    endpoint: str,
    model: str,
    row: dict[str, Any],
    source: str,
    target: str,
    temperature: float,
    retries: int,
    cache: dict[tuple[str, str, str], str],
    cache_lock: Lock,
) -> dict[str, Any] | None:
    kwargs = row.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        return None

    custom_instructions = kwargs.get("custom_instructions")
    if not isinstance(custom_instructions, str) or not custom_instructions.strip():
        return kwargs

    translated_kwargs = dict(kwargs)
    translated_kwargs["custom_instructions"] = translate_cached_text(
        client=client,
        endpoint=endpoint,
        model=model,
        text=custom_instructions,
        source=source,
        target=target,
        temperature=temperature,
        retries=retries,
        cache=cache,
        cache_lock=cache_lock,
    )
    return translated_kwargs


def translate_row(
    *,
    client: httpx.Client,
    endpoint: str,
    model: str,
    row: dict[str, Any],
    source: str,
    target: str,
    temperature: float,
    retries: int,
    translate_custom_instructions: bool,
    custom_instruction_cache: dict[tuple[str, str, str], str],
    custom_instruction_cache_lock: Lock,
) -> dict[str, Any]:
    translated_messages = []
    for message in row["messages"]:
        if not isinstance(message["content"], str):
            raise TypeError("Only string message content is supported")
        translated_message = dict(message)
        translated_message["content"] = translate_text(
            client=client,
            endpoint=endpoint,
            model=model,
            text=message["content"],
            source=source,
            target=target,
            temperature=temperature,
            retries=retries,
        )
        translated_messages.append(translated_message)

    translated_row = {
        **row,
        "messages": translated_messages,
    }
    if translate_custom_instructions:
        translated_kwargs = translate_chat_template_kwargs(
            client=client,
            endpoint=endpoint,
            model=model,
            row=row,
            source=source,
            target=target,
            temperature=temperature,
            retries=retries,
            cache=custom_instruction_cache,
            cache_lock=custom_instruction_cache_lock,
        )
        if translated_kwargs is not None:
            translated_row["chat_template_kwargs"] = translated_kwargs
    return translated_row


def translate_rows(
    *,
    rows: list[dict[str, Any]],
    client: httpx.Client,
    endpoint: str,
    model: str,
    source: str,
    target: str,
    temperature: float,
    retries: int,
    workers: int,
    desc: str,
    translate_custom_instructions: bool,
    custom_instruction_cache: dict[tuple[str, str, str], str],
    custom_instruction_cache_lock: Lock,
) -> tuple[list[dict[str, Any]], float]:
    started_at = time.monotonic()
    translated_rows: list[dict[str, Any] | None] = [None] * len(rows)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                translate_row,
                client=client,
                endpoint=endpoint,
                model=model,
                row=row,
                source=source,
                target=target,
                temperature=temperature,
                retries=retries,
                translate_custom_instructions=translate_custom_instructions,
                custom_instruction_cache=custom_instruction_cache,
                custom_instruction_cache_lock=custom_instruction_cache_lock,
            ): index
            for index, row in enumerate(rows)
        }
        for future in tqdm(
            iterable=as_completed(fs=futures),
            total=len(futures),
            desc=desc,
            unit="rows",
        ):
            translated_rows[futures[future]] = future.result()

    elapsed = time.monotonic() - started_at
    return [row for row in translated_rows if row is not None], elapsed


def write_rows(*, output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode="a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def materialize_dataset(
    *,
    output_path: Path,
    output_dir: Path,
    output_split: str,
) -> DatasetDict:
    dataset = load_dataset(
        path="json",
        data_files={output_split: str(output_path)},
    )
    dataset_dict = DatasetDict(dataset)

    hf_dataset_path = output_dir / "hf_dataset"
    if hf_dataset_path.exists():
        shutil.rmtree(path=hf_dataset_path)
    dataset_dict.save_to_disk(dataset_dict_path=str(hf_dataset_path))

    parquet_dir = output_dir / "parquet"
    if parquet_dir.exists():
        shutil.rmtree(path=parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_dataset in dataset_dict.items():
        split_dataset.to_parquet(str(parquet_dir / f"{split_name}.parquet"))

    return dataset_dict


def push_checkpoint(
    *,
    output_path: Path,
    output_dir: Path,
    output_split: str,
    hub_dataset: str,
    private: bool,
    completed_rows: int,
) -> None:
    dataset_dict = materialize_dataset(
        output_path=output_path,
        output_dir=output_dir,
        output_split=output_split,
    )
    dataset_dict.push_to_hub(
        repo_id=hub_dataset,
        private=private,
        commit_message=f"update translated rows to {completed_rows}",
    )
    print(f"pushed {completed_rows:,} rows to {hub_dataset}")


def sample_indices_for(*, total_rows: int, sample_size: int, seed: int) -> list[int]:
    selected_rows = min(sample_size, total_rows)
    return sorted(random.Random(seed).sample(range(total_rows), selected_rows))


def fetch_dataset_rows_page(
    *, dataset: str, config: str, split: str, offset: int, length: int
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{DATASETS_SERVER_ROWS_ENDPOINT}?{query}"
    with urllib.request.urlopen(url=url, timeout=120) as response:
        return json.load(response)


def iter_sampled_rows(*, args: argparse.Namespace, start_index: int) -> Iterator[dict[str, Any]]:
    first_page = fetch_dataset_rows_page(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        offset=0,
        length=1,
    )
    target_rows = requested_row_count(args=args) or args.sample_size
    indices = sample_indices_for(
        total_rows=first_page["num_rows_total"],
        sample_size=target_rows,
        seed=args.sample_seed,
    )
    page_size = args.sample_page_size
    rows_yielded = 0
    current_page_start = None
    current_rows_by_index: dict[int, dict[str, Any]] = {}

    for sample_index, row_index in enumerate(indices):
        if sample_index < start_index:
            continue
        page_start = row_index // page_size * page_size
        if page_start != current_page_start:
            page = fetch_dataset_rows_page(
                dataset=args.dataset,
                config=args.config,
                split=args.split,
                offset=page_start,
                length=page_size,
            )
            current_page_start = page_start
            current_rows_by_index = {
                row["row_idx"]: row["row"] for row in page["rows"]
            }
        yield dict(current_rows_by_index[row_index])
        rows_yielded += 1
        if rows_yielded >= target_rows - start_index:
            break


def worker_candidates_for(*, raw_candidates: str, max_workers: int) -> list[int]:
    candidates = []
    for raw_candidate in raw_candidates.split(","):
        worker_count = int(raw_candidate.strip())
        if 0 < worker_count <= max_workers:
            candidates.append(worker_count)
    if not candidates:
        candidates.append(max_workers)
    return sorted(set(candidates))


def load_streaming_rows(
    *, args: argparse.Namespace, total_rows: int
) -> list[dict[str, Any]]:
    return list(islice(iter_streaming_rows(args=args), total_rows))


def iter_streaming_rows(
    *, args: argparse.Namespace, start_index: int = 0
) -> Iterator[dict[str, Any]]:
    if args.sample_size is not None:
        yield from iter_sampled_rows(args=args, start_index=start_index)
        return

    dataset = load_dataset(
        path=args.dataset,
        name=args.config,
        split=args.split,
        streaming=True,
    )
    for index, row in enumerate(dataset):
        if args.limit is not None and index >= args.limit:
            break
        if index < start_index:
            continue
        yield dict(row)


def check_examples(*, rows: list[dict[str, Any]], n_examples: int) -> None:
    if n_examples <= 0:
        return

    sampled_rows = rows[:n_examples]
    if not sampled_rows:
        print("No rows available for Polars sample check")
        return

    frame = polars.DataFrame(sampled_rows)
    stats_exprs = [polars.col("messages").list.len().alias("message_count")]
    if "source" in frame.columns:
        stats_exprs.append(polars.col("source").alias("source"))
    message_stats = frame.select(*stats_exprs)
    print("\nPolars sample check")
    print(f"Rows checked: {len(sampled_rows):,}")
    print(frame.schema)
    print(message_stats.head(n=10))

    for index, row in enumerate(sampled_rows[:3]):
        first_message = row["messages"][0]
        print(
            f"example {index}: role={first_message['role']} "
            f"content={first_message['content'][:160]!r}"
        )


def maybe_push(
    *,
    args: argparse.Namespace,
    output_path: Path,
    output_dir: Path,
    completed_rows: int,
    last_pushed_at: int,
) -> int:
    if args.no_push or completed_rows - last_pushed_at < args.push_every:
        return last_pushed_at

    push_checkpoint(
        output_path=output_path,
        output_dir=output_dir,
        output_split=args.output_split,
        hub_dataset=args.hub_dataset,
        private=args.private,
        completed_rows=completed_rows,
    )
    return completed_rows


def take_rows(*, rows: Iterator[dict[str, Any]], n_rows: int) -> list[dict[str, Any]]:
    return list(islice(rows, n_rows))


def translate_with_warmup(
    *,
    args: argparse.Namespace,
    rows: Iterator[dict[str, Any]],
    output_path: Path,
    output_dir: Path,
    start_index: int,
    client: httpx.Client,
    custom_instruction_cache: dict[tuple[str, str, str], str],
    custom_instruction_cache_lock: Lock,
) -> tuple[int, int, int]:
    best_workers = args.workers if args.workers else min(args.max_workers, 32)
    last_pushed_at = start_index

    if args.workers:
        return best_workers, last_pushed_at, start_index

    scores = []
    current_index = start_index
    for workers in worker_candidates_for(
        raw_candidates=args.worker_candidates,
        max_workers=args.max_workers,
    ):
        warmup_rows = take_rows(rows=rows, n_rows=args.warmup_rows)
        if not warmup_rows:
            break
        translated_rows, elapsed = translate_rows(
            rows=warmup_rows,
            client=client,
            endpoint=args.endpoint,
            model=args.model,
            source=args.source,
            target=args.target,
            temperature=args.temperature,
            retries=args.retries,
            workers=workers,
            desc=f"warmup workers={workers}",
            translate_custom_instructions=args.translate_custom_instructions,
            custom_instruction_cache=custom_instruction_cache,
            custom_instruction_cache_lock=custom_instruction_cache_lock,
        )
        write_rows(output_path=output_path, rows=translated_rows)
        current_index += len(translated_rows)
        throughput = len(translated_rows) / elapsed if elapsed else 0
        scores.append((throughput, workers))
        print(f"workers={workers} translated {throughput:.2f} rows/sec")
        last_pushed_at = maybe_push(
            args=args,
            output_path=output_path,
            output_dir=output_dir,
            completed_rows=current_index,
            last_pushed_at=last_pushed_at,
        )

    if scores:
        best_workers = max(scores)[1]
    print(f"selected workers={best_workers}")
    return best_workers, last_pushed_at, current_index


def translate_remaining(
    *,
    args: argparse.Namespace,
    rows: Iterator[dict[str, Any]],
    output_path: Path,
    output_dir: Path,
    start_index: int,
    workers: int,
    last_pushed_at: int,
    client: httpx.Client,
    custom_instruction_cache: dict[tuple[str, str, str], str],
    custom_instruction_cache_lock: Lock,
) -> None:
    current_index = start_index
    while True:
        batch_rows = take_rows(rows=rows, n_rows=args.batch_size)
        if not batch_rows:
            break
        batch_end = current_index + len(batch_rows)
        translated_rows, _ = translate_rows(
            rows=batch_rows,
            client=client,
            endpoint=args.endpoint,
            model=args.model,
            source=args.source,
            target=args.target,
            temperature=args.temperature,
            retries=args.retries,
            workers=workers,
            desc=f"translate {current_index:,}-{batch_end:,}",
            translate_custom_instructions=args.translate_custom_instructions,
            custom_instruction_cache=custom_instruction_cache,
            custom_instruction_cache_lock=custom_instruction_cache_lock,
        )
        write_rows(output_path=output_path, rows=translated_rows)
        current_index += len(translated_rows)
        last_pushed_at = maybe_push(
            args=args,
            output_path=output_path,
            output_dir=output_dir,
            completed_rows=current_index,
            last_pushed_at=last_pushed_at,
        )

    if not args.no_push and last_pushed_at != current_index:
        push_checkpoint(
            output_path=output_path,
            output_dir=output_dir,
            output_split=args.output_split,
            hub_dataset=args.hub_dataset,
            private=args.private,
            completed_rows=current_index,
        )
    elif args.no_push:
        materialize_dataset(
            output_path=output_path,
            output_dir=output_dir,
            output_split=args.output_split,
        )


def requested_row_count(*, args: argparse.Namespace) -> int | None:
    row_counts = [
        row_count
        for row_count in [args.limit, args.sample_size]
        if row_count is not None
    ]
    if not row_counts:
        return None
    return min(row_counts)


def prepare_output(*, args: argparse.Namespace, output_path: Path) -> int:
    if args.overwrite and output_path.exists():
        output_path.unlink()
    if not args.resume and output_path.exists():
        raise FileExistsError(f"{output_path} exists. Use --resume or --overwrite.")
    return count_jsonl_rows(path=output_path) if args.resume else 0


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.output_split}.jsonl"

    sample_rows = load_streaming_rows(
        args=args,
        total_rows=max(args.check_examples, 0),
    )
    check_examples(rows=sample_rows, n_examples=args.check_examples)
    if args.check_only:
        return

    target_rows = requested_row_count(args=args)
    start_index = prepare_output(args=args, output_path=output_path)
    if target_rows is not None and start_index > target_rows:
        raise ValueError(
            f"{output_path} has {start_index} rows, but target row count is {target_rows}"
        )
    if target_rows is not None and start_index == target_rows:
        print(f"already translated {target_rows:,} rows")
        return

    rows = iter_streaming_rows(args=args, start_index=start_index)
    limits = httpx.Limits(
        max_connections=max(args.max_workers, args.workers, 1),
        max_keepalive_connections=max(args.max_workers, args.workers, 1),
    )
    timeout = httpx.Timeout(timeout=args.timeout)
    started_at = time.monotonic()
    custom_instruction_cache: dict[tuple[str, str, str], str] = {}
    custom_instruction_cache_lock = Lock()

    with httpx.Client(timeout=timeout, limits=limits) as client:
        workers, last_pushed_at, current_index = translate_with_warmup(
            args=args,
            rows=rows,
            output_path=output_path,
            output_dir=output_dir,
            start_index=start_index,
            client=client,
            custom_instruction_cache=custom_instruction_cache,
            custom_instruction_cache_lock=custom_instruction_cache_lock,
        )
        translate_remaining(
            args=args,
            rows=rows,
            output_path=output_path,
            output_dir=output_dir,
            start_index=current_index,
            workers=workers,
            last_pushed_at=last_pushed_at,
            client=client,
            custom_instruction_cache=custom_instruction_cache,
            custom_instruction_cache_lock=custom_instruction_cache_lock,
        )

    elapsed_minutes = (time.monotonic() - started_at) / 60
    print(f"done in {elapsed_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
