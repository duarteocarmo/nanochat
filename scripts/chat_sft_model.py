"""Chat with the SFT Ginjinha model on Apple MPS."""

import argparse
import os
import shutil
import urllib.request

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, get_base_dir
from nanochat.engine import Engine

MODEL_REPO = "duarteocarmo/ginjinha"
MODEL_TAG = "ginjinha_d11_ratio40_education_score_gte2_full_corpus"
SFT_TAG = f"{MODEL_TAG}_pt_sft"
MODEL_STEP = 545


def download_model() -> None:
    files = {
        f"{SFT_TAG}/checkpoints/model_{MODEL_STEP:06d}.pt": f"chatsft_checkpoints/{MODEL_TAG}/model_{MODEL_STEP:06d}.pt",
        f"{SFT_TAG}/checkpoints/meta_{MODEL_STEP:06d}.json": f"chatsft_checkpoints/{MODEL_TAG}/meta_{MODEL_STEP:06d}.json",
        f"{MODEL_TAG}/tokenizer/tokenizer.pkl": "tokenizer/tokenizer.pkl",
    }
    base_dir = get_base_dir()
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


def respond(engine, tokenizer, prompt: str) -> None:
    conversation = {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""},
    ]}
    prompt_tokens = tokenizer.render_for_completion(conversation)
    samples, _ = engine.generate_batch(
        tokens=prompt_tokens,
        num_samples=1,
        max_tokens=128,
        temperature=0.0,
        top_k=None,
    )
    print(tokenizer.decode(samples[0][len(prompt_tokens):]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Message for the model")
    args = parser.parse_args()

    download_model()
    _, _, _, _, device = compute_init(device_type="mps")
    model, tokenizer, _ = load_model(
        source="sft",
        device=device,
        phase="eval",
        model_tag=MODEL_TAG,
        step=MODEL_STEP,
    )
    engine = Engine(model=model, tokenizer=tokenizer)

    if args.prompt:
        respond(engine=engine, tokenizer=tokenizer, prompt=args.prompt)
        return

    print("Type a message. Type 'quit' to exit.")
    while True:
        try:
            prompt = input("\nYou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in {"quit", "exit"}:
            break
        if prompt:
            print("\nModel> ", end="")
            respond(engine=engine, tokenizer=tokenizer, prompt=prompt)


if __name__ == "__main__":
    main()
