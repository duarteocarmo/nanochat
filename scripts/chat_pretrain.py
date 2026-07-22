"""Continue text with the pretrained Ginjinha model on Apple MPS."""

import argparse
import os
import shutil
import urllib.request

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, get_base_dir
from nanochat.engine import Engine

MODEL_REPO = "duarteocarmo/ginjinha"
MODEL_TAG = "ginjinha_d11_ratio40_education_score_gte2_full_corpus"
MODEL_STEP = 7860
MAX_TOKENS = 128


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


def continue_text(engine, tokenizer, prompt: str) -> None:
    prompt_tokens = tokenizer.encode(prompt, prepend="<|bos|>")
    samples, _ = engine.generate_batch(
        tokens=prompt_tokens,
        num_samples=1,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        top_k=None,
    )
    continuation = tokenizer.decode(samples[0][len(prompt_tokens):])
    print(f"{prompt}{continuation}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Text for the model to continue")
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
        continue_text(engine=engine, tokenizer=tokenizer, prompt=args.prompt)
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
        continue_text(engine=engine, tokenizer=tokenizer, prompt=prompt)


if __name__ == "__main__":
    main()
