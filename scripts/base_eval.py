"""
Unified evaluation script for base models.

Supports four evaluation modes (comma-separated):
  --eval core     : PTCORE answer-likelihood metric
  --eval core-gen : Few-shot PTCORE generated-answer metric
  --eval bpb      : Bits per byte on train/val splits
  --eval sample   : Generate samples from the model

Examples:

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""

import argparse
import os

import wandb

from nanochat.checkpoint_manager import load_model
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.engine import Engine
from nanochat.loss_eval import evaluate_bpb
from nanochat.ptcore_eval import evaluate_ptcore
from nanochat.ptcore_generation import (
    evaluate_ptcore_generation,
    generation_metrics_for_wandb,
    save_generation_results,
)
from nanochat.tokenizer import get_token_bytes

# -----------------------------------------------------------------------------
# CORE evaluation


def evaluate_core(model, tokenizer, device, max_per_task=-1) -> dict:
    """Evaluate this fork's CORE metric, backed by PTCORE."""
    return evaluate_ptcore(model, tokenizer, device, max_per_task=max_per_task)


# -----------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument(
        "--eval",
        type=str,
        default="core,bpb,sample",
        help="Comma-separated evaluations: core,core-gen,bpb,sample (default: core,bpb,sample)",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="nanochat model tag to identify the checkpoint directory",
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Model step to load (default = last)"
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=-1,
        help="Max examples per PTCORE task (-1 = all)",
    )
    parser.add_argument(
        "--generation-fewshot",
        type=int,
        default=3,
        help="Fixed demonstration questions reserved per task for core-gen",
    )
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=16,
        help="Maximum generated answer tokens for core-gen",
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=32,
        help="Per-device batch size for BPB evaluation",
    )
    parser.add_argument(
        "--split-tokens",
        type=int,
        default=40 * 524288,
        help="Number of tokens to evaluate per split for BPB",
    )
    parser.add_argument(
        "--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)"
    )
    parser.add_argument(
        "--run",
        type=str,
        default="dummy",
        help="wandb run name ('dummy' disables wandb logging)",
    )
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = {mode.strip() for mode in args.eval.split(",")}
    valid_modes = {"core", "core-gen", "bpb", "sample"}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")
    if args.generation_fewshot < 0 or args.generation_max_new_tokens < 1:
        parser.error(
            "Generation requires non-negative fewshot and positive max-new-tokens"
        )

    # Distributed / precision setup
    device_type = (
        autodetect_device_type() if args.device_type == "" else args.device_type
    )
    _ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # Load model and tokenizer
    model, tokenizer, meta = load_model(
        "base", device, phase="eval", model_tag=args.model_tag, step=args.step
    )
    sequence_len = meta["model_config"]["sequence_len"]
    model_name = f"base_model (step {meta['step']})"
    model_slug = f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    use_dummy_wandb = args.run == "dummy" or ddp_rank != 0
    wandb_run_id = meta.get("wandb_run_id")
    if not use_dummy_wandb and not wandb_run_id:
        raise RuntimeError(
            "Checkpoint has no W&B run ID; cannot attach the evaluation artifact"
        )
    wandb_run = (
        DummyWandb()
        if use_dummy_wandb
        else wandb.init(
            project="ginjinha",
            id=wandb_run_id,
            resume="must",
            settings=wandb.Settings(x_disable_stats=True),
        )
    )

    conditioned_samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if "sample" in eval_modes:
        print0("\n" + "=" * 80)
        print0("Model Samples")
        print0("=" * 80)
        if ddp_rank == 0:
            prompts = [
                "A capital de França é",
                "O símbolo químico do ouro é",
                "Se ontem foi sexta-feira, então amanhã será",
                "O contrário de quente é",
                "Os planetas do sistema solar são:",
                "A minha cor preferida é",
                "O plural de cão é",
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(
                    tokens, num_samples=1, max_tokens=16, temperature=0
                )
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                conditioned_samples.append((prompt, sample_str))

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(
                tokens, num_samples=8, max_tokens=128, temperature=1.0
            )
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)

    # --- BPB evaluation ---
    if "bpb" in eval_modes:
        token_bytes = get_token_bytes(device=device)
        print0("\n" + "=" * 80)
        print0("BPB Evaluation")
        print0("=" * 80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(
                f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})"
            )
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer,
                args.device_batch_size,
                sequence_len,
                split_name,
                device=device,
            )
            bpb = evaluate_bpb(model, loader, steps, token_bytes)
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- PTCORE evaluation ---
    output_paths = []
    if "core" in eval_modes:
        print0("\n" + "=" * 80)
        print0("PTCORE Evaluation")
        print0("=" * 80)
        core_results = evaluate_core(
            model, tokenizer, device, max_per_task=args.max_per_task
        )

        # Write CSV output
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_name = args.model_tag or model_slug
            output_csv_path = os.path.join(base_dir, "base_eval", f"{output_name}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(
                    f"{'PTCORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n"
                )
            output_paths.append(output_csv_path)
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"PTCORE metric: {core_results['core_metric']:.4f}")

    if "core-gen" in eval_modes:
        print0("\n" + "=" * 80)
        print0("PTCORE Generation Evaluation")
        print0("=" * 80)
        generation_results = evaluate_ptcore_generation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_per_task=args.max_per_task,
            num_fewshot=args.generation_fewshot,
            max_new_tokens=args.generation_max_new_tokens,
        )
        if ddp_rank == 0:
            output_name = args.model_tag or model_slug
            output_prefix = os.path.join(
                get_base_dir(), "base_eval", f"{output_name}_generation"
            )
            output_paths.extend(
                save_generation_results(
                    results={
                        "model_tag": args.model_tag,
                        "step": meta["step"],
                        **generation_results,
                    },
                    output_prefix=output_prefix,
                )
            )
            print0(
                f"PTCORE generation metric: {generation_results['generation_metric']:.4f}"
            )
            print0(
                f"PTCORE generation invalid rate: {generation_results['invalid_rate']:.4f}"
            )
            print0(f"Results written to: {output_prefix}.json and {output_prefix}.csv")
            if not use_dummy_wandb:
                wandb_run.log(
                    data={
                        "step": meta["step"],
                        **generation_metrics_for_wandb(results=generation_results),
                    }
                )

    if ddp_rank == 0 and not use_dummy_wandb and conditioned_samples:
        wandb_run.log(
            {
                "step": meta["step"],
                "samples/conditioned": wandb.Table(
                    columns=["prompt", "sample"], data=conditioned_samples
                ),
                "samples/unconditioned": wandb.Table(
                    columns=["index", "sample"],
                    data=list(enumerate(unconditioned_samples)),
                ),
            }
        )

    if ddp_rank == 0 and not use_dummy_wandb and output_paths:
        artifact = wandb.Artifact(f"{model_slug}_base_eval", type="eval")
        for path in output_paths:
            artifact.add_file(local_path=path)
        wandb_run.log_artifact(artifact)  # ty: ignore[unresolved-attribute]

    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
