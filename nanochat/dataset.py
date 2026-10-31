"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For Bagaço3 preparation details, see `dev/prepare_bagaco3.py`.
Set NANOCHAT_DATASET=bagaco3 to use the Bagaço3 pretraining corpus.
"""

import argparse
import os
import time
from multiprocessing import Pool

import pyarrow.parquet as pq
import requests

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

DATASET = os.environ.get("NANOCHAT_DATASET", "bagaco2")
if DATASET not in {"bagaco2", "bagaco3"}:
    raise ValueError("NANOCHAT_DATASET must be 'bagaco2' or 'bagaco3'")

DATASET_REPO = (
    "duarteocarmo/bagaco3-shuffle"
    if DATASET == "bagaco3"
    else "duarteocarmo/bagaco2-30b-shuffle"
)
DATASET_REVISION = (
    "f41c9c2fe971836585f07e414f81499a580b032a" if DATASET == "bagaco3" else "main"
)
BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}"
MAX_SHARD = 498 if DATASET == "bagaco3" else 359


def index_to_filename(index):
    filename = f"shard_{index:05d}.parquet"
    if DATASET == "bagaco3":
        return (
            f"train/{filename}"
            if index < MAX_SHARD
            else "validation/shard_00000.parquet"
        )
    return filename


base_dir = get_base_dir()
DATA_DIR = os.environ.get(
    "NANOCHAT_DATA_DIR", os.path.join(base_dir, f"base_data_{DATASET}")
)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported


def list_parquet_files(data_dir=None, warn_on_legacy=False, split=None):
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    if DATASET == "bagaco3":
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        data_dir = os.path.join(data_dir, "train" if split == "train" else "validation")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Could not find {data_dir}. Download {DATASET} shards with: "
            f"NANOCHAT_DATASET={DATASET} python -m nanochat.dataset -n 170"
        )

    parquet_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    if DATASET == "bagaco2" and split is not None:
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    return parquet_paths


def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". Validation uses a dedicated held-out split.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(split=split)
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


# -----------------------------------------------------------------------------
def download_single_file(index):
    """Downloads a single file index, with some backoff"""

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:  # noqa: UP024
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:  # noqa: E722, S110
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="Number of train shards to download (default: -1), -1 = disable",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    args = parser.parse_args()

    # Prepare the output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = (
        MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    )
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)  # always download the validation shard

    # Download the shards
    print(
        f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers..."
    )
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
