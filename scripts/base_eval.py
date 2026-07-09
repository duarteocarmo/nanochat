"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : PTCORE metric (Portuguese multiple-choice tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""
import os
import argparse

import wandb

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, DummyWandb
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.ptcore_eval import evaluate_ptcore

# -----------------------------------------------------------------------------
# CORE evaluation

def evaluate_core(model, tokenizer, device, max_per_task=-1) -> dict:
    """Evaluate this fork's CORE metric, backed by PTCORE."""
    return evaluate_ptcore(model, tokenizer, device, max_per_task=max_per_task)

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per PTCORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    parser.add_argument('--run', type=str, default='dummy', help="wandb run name ('dummy' disables wandb logging)")
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # Load model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    sequence_len = meta["model_config"]["sequence_len"]
    token_bytes = get_token_bytes(device=device)
    model_name = f"base_model (step {meta['step']})"
    model_slug = f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    use_dummy_wandb = args.run == "dummy" or ddp_rank != 0
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
        project="ginjinha",
        name=args.run,
        id=meta.get("wandb_run_id"),
        resume="allow",
        config=vars(args),
    )

    # Results to log
    core_results = None
    bpb_results = {}
    samples = []
    sample_rows = []
    unconditioned_samples = []

    # --- Sampling ---
    if 'sample' in eval_modes:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
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
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)
                sample_rows.append((prompt, sample_str))

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)

    # --- BPB evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            bpb = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- PTCORE evaluation ---
    output_csv_path = None
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("PTCORE Evaluation")
        print0("="*80)
        core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)

        # Write CSV output
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'PTCORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"PTCORE metric: {core_results['core_metric']:.4f}")

    if ddp_rank == 0:
        log_data = {"step": meta["step"]}
        if "train" in bpb_results:
            log_data["train/bpb"] = bpb_results["train"]
        if "val" in bpb_results:
            log_data["val/bpb"] = bpb_results["val"]
        if core_results:
            log_data["core/metric"] = core_results["core_metric"]
            log_data.update({f"core/accuracy/{key}": value for key, value in core_results["results"].items()})
            log_data.update({f"core/centered/{key}": value for key, value in core_results["centered_results"].items()})
        wandb_run.log(log_data)

        if not use_dummy_wandb and sample_rows:
            conditioned_table = wandb.Table(columns=["prompt", "sample"], data=sample_rows)
            unconditioned_table = wandb.Table(
                columns=["index", "sample"],
                data=list(enumerate(unconditioned_samples)),
            )
            wandb_run.log({
                "step": meta["step"],
                "samples/conditioned": conditioned_table,
                "samples/unconditioned": unconditioned_table,
            })

        if not use_dummy_wandb and output_csv_path:
            artifact = wandb.Artifact(f"{model_slug}_base_eval", type="eval")
            artifact.add_file(output_csv_path)
            wandb_run.log_artifact(artifact)

    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
