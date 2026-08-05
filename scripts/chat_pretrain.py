"""Complete text with the best education ≥3 Ginjinha model on Apple MPS."""

import argparse
import os
import shutil
import urllib.request

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, get_base_dir
from nanochat.engine import Engine

MODEL_REPO = "duarteocarmo/ginjinha"
MODEL_TAG = "ginjinha_d8_ratio40_ptcore5_education_score_gte3"
MODEL_STEP = 3200
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50


def download_model() -> None:
    base_dir = get_base_dir()
    files = {
        f"{MODEL_TAG}/checkpoints/model_{MODEL_STEP:06d}.pt": f"base_checkpoints/{MODEL_TAG}/model_{MODEL_STEP:06d}.pt",
        f"{MODEL_TAG}/checkpoints/meta_{MODEL_STEP:06d}.json": f"base_checkpoints/{MODEL_TAG}/meta_{MODEL_STEP:06d}.json",
        f"{MODEL_TAG}/tokenizer/tokenizer.pkl": "tokenizer/tokenizer.pkl",
    }

    for remote_path, local_path in files.items():
        destination = os.path.join(base_dir, local_path)
        if os.path.exists(destination):
            continue

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        temporary_destination = f"{destination}.part"
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{remote_path}"
        try:
            with urllib.request.urlopen(url=url) as response, open(temporary_destination, "wb") as output_file:
                shutil.copyfileobj(fsrc=response, fdst=output_file)
            os.replace(temporary_destination, destination)
        finally:
            if os.path.exists(temporary_destination):
                os.remove(temporary_destination)
        print(f"Downloaded {local_path}")


def continue_text(engine, tokenizer, prompt: str, max_tokens: int, temperature: float, top_k: int) -> None:
    prompt_tokens = tokenizer.encode(prompt, prepend="<|bos|>")
    samples, _ = engine.generate_batch(
        tokens=prompt_tokens,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    continuation = tokenizer.decode(samples[0][len(prompt_tokens):])
    print(f"{prompt}{continuation}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Text for the model to continue")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum new tokens")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Sample from the top K tokens")
    args = parser.parse_args()

    download_model()
    _, _, _, _, device = compute_init(device_type="mps")
    model, tokenizer, _ = load_model(
        source="base",
        device=device,
        phase="eval",
        model_tag=MODEL_TAG,
        step=MODEL_STEP,
    )
    engine = Engine(model=model, tokenizer=tokenizer)

    if args.prompt:
        continue_text(
            engine=engine,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        return

    print("Type a prompt for the model to continue. Type 'quit' to exit.")
    while True:
        try:
            prompt = input("\nPrompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in {"quit", "exit"}:
            break
        if not prompt:
            continue

        print()
        continue_text(
            engine=engine,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
