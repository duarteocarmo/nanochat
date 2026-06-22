"""Upload the generated nanochat report to an existing W&B run."""

import argparse
from pathlib import Path

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run", required=True, help="W&B run display name or id")
    parser.add_argument("--path", required=True, help="Path to report.md")
    parser.add_argument(
        "--entity", default=None, help="W&B entity (default: current user)"
    )
    return parser.parse_args()


def find_run(api, entity: str, project: str, run_name: str):
    path = f"{entity}/{project}"
    try:
        return api.run(f"{path}/{run_name}")
    except Exception:
        pass

    matches = list(
        api.runs(
            path,
            filters={"display_name": run_name},
            order="-created_at",
            per_page=10,
            include_sweeps=False,
        )
    )
    if not matches:
        matches = list(
            api.runs(
                path,
                filters={"displayName": run_name},
                order="-created_at",
                per_page=10,
                include_sweeps=False,
            )
        )
    if not matches:
        raise RuntimeError(f"No W&B run found for {path} named/id {run_name!r}")
    return matches[0]


def main():
    args = parse_args()
    if args.run == "dummy":
        print("Skipping W&B report upload for dummy run")
        return

    report_path = Path(args.path).expanduser().resolve()
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    api = wandb.Api(timeout=120)
    entity = args.entity or api.viewer.username
    run = find_run(api=api, entity=entity, project=args.project, run_name=args.run)
    uploaded = run.upload_file(str(report_path), root=str(report_path.parent))
    print(
        f"Uploaded {report_path} to W&B run {entity}/{args.project}/{run.id}: {uploaded.name}"
    )


if __name__ == "__main__":
    main()
