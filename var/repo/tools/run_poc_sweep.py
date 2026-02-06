#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


DEFAULT_BASELINE = (
    "configs/recipes/Qwen2.5-0.5B-Instruct/grpo/config_math_maxent_hparams.yaml"
)
DEFAULT_MAXENT = (
    "configs/recipes/Qwen2.5-0.5B-Instruct/maxent-grpo/config_math_maxent_hparams.yaml"
)


def _parse_seeds(value: Optional[str], count: int, start: int) -> list[int]:
    if value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return list(range(start, start + count))


def _list_wandb_runs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {p for p in root.glob("run-*") if p.is_dir()}


def _pick_new_run(before: set[Path], after: set[Path], root: Path) -> Optional[Path]:
    new_runs = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if new_runs:
        return new_runs[0]
    latest = root / "latest-run"
    if latest.exists():
        try:
            return latest.resolve()
        except OSError:
            return latest
    return None


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metric(run_dir: Path, metric: str) -> Optional[float]:
    summary = run_dir / "files" / "wandb-summary.json"
    if summary.exists():
        data = _read_json(summary)
        if metric in data and data[metric] is not None:
            try:
                return float(data[metric])
            except (TypeError, ValueError):
                pass
    history = run_dir / "files" / "wandb-history.jsonl"
    if history.exists():
        last_val = None
        with history.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if metric in payload and payload[metric] is not None:
                    last_val = payload[metric]
        if last_val is not None:
            try:
                return float(last_val)
            except (TypeError, ValueError):
                return None
    return None


def _format_mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


def _build_accelerate_cmd(
    variant: str,
    recipe: str,
    seed: int,
    run_name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    entrypoint = "src/maxent_grpo/grpo.py" if variant == "baseline" else "src/maxent_grpo/maxent_grpo.py"
    cmd = [
        "accelerate",
        "launch",
    ]
    if args.accelerate_config:
        cmd.extend(["--config_file", args.accelerate_config])
    if args.num_processes:
        cmd.extend(["--num_processes", str(args.num_processes)])
    if args.num_machines:
        cmd.extend(["--num_machines", str(args.num_machines)])
    if args.machine_rank:
        cmd.extend(["--machine_rank", str(args.machine_rank)])
    if args.main_process_ip:
        cmd.extend(["--main_process_ip", args.main_process_ip])
    if args.main_process_port:
        cmd.extend(["--main_process_port", str(args.main_process_port)])
    cmd.extend(
        [
            entrypoint,
            "--config",
            recipe,
            "--seed",
            str(seed),
            "--run_name",
            run_name,
            "--output_dir",
            output_dir.as_posix(),
            "--resume_from_checkpoint",
            "false",
            "--push_to_hub",
            str(args.push_to_hub).lower(),
        ]
    )
    if args.vllm_host:
        cmd.extend(["--vllm_server_host", args.vllm_host])
    if args.vllm_port:
        cmd.extend(["--vllm_server_port", str(args.vllm_port)])
    if args.wandb_project:
        cmd.extend(["--wandb_project", args.wandb_project])
    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])
    if args.wandb_group:
        cmd.extend(["--wandb_run_group", args.wandb_group])
    return cmd


def _build_cli_cmd(
    variant: str,
    recipe: str,
    seed: int,
    run_name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    prefix = variant
    cmd = [
        sys.executable,
        "-m",
        "maxent_grpo.cli",
        f"command=train-{variant}",
        f"{prefix}.recipe={recipe}",
        f"{prefix}.training.seed={seed}",
        f"{prefix}.training.run_name={run_name}",
        f"{prefix}.training.output_dir={output_dir.as_posix()}",
        f"{prefix}.training.push_to_hub={str(args.push_to_hub).lower()}",
        f"{prefix}.training.resume_from_checkpoint=false",
    ]
    if args.wandb_project:
        cmd.append(f"{prefix}.training.wandb_project={args.wandb_project}")
    if args.wandb_entity:
        cmd.append(f"{prefix}.training.wandb_entity={args.wandb_entity}")
    if args.wandb_group:
        cmd.append(f"{prefix}.training.wandb_run_group={args.wandb_group}")
    if args.vllm_host:
        cmd.append(f"{prefix}.training.vllm_server_host={args.vllm_host}")
    if args.vllm_port:
        cmd.append(f"{prefix}.training.vllm_server_port={args.vllm_port}")
    return cmd


def _run_variant(
    variant: str,
    recipe: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[str, Optional[Path]]:
    if variant not in {"baseline", "maxent"}:
        raise ValueError(f"Unsupported variant: {variant}")
    run_name = f"{args.tag}-{variant}-s{seed}"
    output_dir = Path(args.output_root) / variant / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_accelerate:
        cmd = _build_accelerate_cmd(variant, recipe, seed, run_name, output_dir, args)
    else:
        cmd = _build_cli_cmd(variant, recipe, seed, run_name, output_dir, args)

    env = os.environ.copy()
    if args.wandb_mode:
        env["WANDB_MODE"] = args.wandb_mode
    if args.wandb_project:
        env.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        env.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_group:
        env.setdefault("WANDB_RUN_GROUP", args.wandb_group)
    env.setdefault("WANDB_DIR", args.wandb_dir)

    wandb_root = Path(args.wandb_dir)
    before = _list_wandb_runs(wandb_root)
    print(f"[{variant}] seed={seed} run_name={run_name}")
    if args.dry_run:
        print("  DRY RUN:", " ".join(cmd))
        return run_name, None
    subprocess.run(cmd, check=True, env=env)
    after = _list_wandb_runs(wandb_root)
    run_dir = _pick_new_run(before, after, wandb_root)
    return run_name, run_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline/maxent sweeps across seeds and aggregate W&B metrics."
    )
    parser.add_argument("--baseline-recipe", default=DEFAULT_BASELINE)
    parser.add_argument("--maxent-recipe", default=DEFAULT_MAXENT)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--metric", default="eval/mean_reward")
    parser.add_argument("--tag", default="poc-0.5b")
    parser.add_argument("--output-root", default="var/data/poc_runs")
    parser.add_argument("--wandb-dir", default="var/artifacts/wandb")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None, help="online or offline")
    parser.add_argument("--use-accelerate", action="store_true")
    parser.add_argument("--accelerate-config", default=None)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--num-machines", type=int, default=None)
    parser.add_argument("--machine-rank", type=int, default=None)
    parser.add_argument("--main-process-ip", default=None)
    parser.add_argument("--main-process-port", type=int, default=None)
    parser.add_argument("--vllm-host", default=None)
    parser.add_argument("--vllm-port", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds, args.num_seeds, args.seed_start)
    if not seeds:
        raise SystemExit("No seeds provided")
    if not args.wandb_group:
        args.wandb_group = args.tag

    results: dict[str, list[dict]] = {"baseline": [], "maxent": []}
    for seed in seeds:
        for variant, recipe in (("baseline", args.baseline_recipe), ("maxent", args.maxent_recipe)):
            run_name, run_dir = _run_variant(variant, recipe, seed, args)
            metric_val = None
            if run_dir:
                metric_val = _extract_metric(run_dir, args.metric)
            results[variant].append(
                {
                    "seed": seed,
                    "run_name": run_name,
                    "run_dir": str(run_dir) if run_dir else None,
                    "metric": metric_val,
                }
            )

    summary = {
        "metric": args.metric,
        "tag": args.tag,
        "baseline": _format_mean_std(
            [r["metric"] for r in results["baseline"] if r["metric"] is not None]
        ),
        "maxent": _format_mean_std(
            [r["metric"] for r in results["maxent"] if r["metric"] is not None]
        ),
        "runs": results,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path("var") / "artifacts" / "results" / f"poc_sweep_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
