#!/usr/bin/env python3
"""Submit the E72 decoding-frontier evaluation grid.

Each cell loads one frozen terminal checkpoint, evaluates it once through the
ordinary training evaluation path at a chosen (temperature, top_p, K), and
exits. No optimizer step, rollout, or export occurs, so a cell is a pure
function of the checkpoint and its decoding settings.

Every non-decoding argument is inherited from the source run's own recorded
configuration via ``var/artifacts/e72_frontier_source_runs.json``; this script
never re-types a prompt template, split, response length, or draw seed.

Stages
------
a  frontier      T in {0.5,0.7,1.0,1.3,1.6,2.0}, top_p 1.0, K=8 x 4 draws
b  budget        K=32 x 2 draws at T in {1.0,1.3,1.6}; per-prompt draws are
                 retained so distinct@K for smaller K is recovered by
                 subsampling rather than by more generation
c  truncation    top_p 0.95 at T in {1.0,1.6}, K=8 x 4 draws

Stage a alone answers the temperature objection and must pass its reproduction
gate (see aggregate_e72_frontier.py) before b and c are worth submitting.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

STAGE_A_TEMPERATURES = (0.5, 0.7, 1.0, 1.3, 1.6, 2.0)
STAGE_B_TEMPERATURES = (1.0, 1.3, 1.6)
STAGE_C_TEMPERATURES = (1.0, 1.6)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def format_number(value: float) -> str:
    """Stable, filesystem-safe rendering used in both paths and job names."""
    return f"{value:g}".replace(".", "p")


def stage_cells(stage: str) -> tuple[tuple[float, float, int, int], ...]:
    """Return (temperature, top_p, k, draws) for one stage."""
    if stage == "a":
        return tuple((t, 1.0, 8, 4) for t in STAGE_A_TEMPERATURES)
    if stage == "b":
        return tuple((t, 1.0, 32, 2) for t in STAGE_B_TEMPERATURES)
    if stage == "c":
        return tuple((t, 0.95, 8, 4) for t in STAGE_C_TEMPERATURES)
    raise ValueError(f"unknown stage {stage!r}")


def iter_jobs(
    manifest: dict[str, Any],
    stage: str,
    *,
    domains: set[str] | None,
    arms: set[str] | None,
    seeds: set[int] | None,
) -> Iterator[dict[str, Any]]:
    for run in manifest["runs"]:
        if domains and run["domain"] not in domains:
            continue
        if arms and run["arm"] not in arms:
            continue
        if seeds and int(run["seed"]) not in seeds:
            continue
        for temperature, top_p, k, draws in stage_cells(stage):
            yield {
                "stage": stage,
                "domain": run["domain"],
                "arm": run["arm"],
                "seed": int(run["seed"]),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "k": int(k),
                "draws": int(draws),
                "source": run,
            }


def cell_tag(job: dict[str, Any]) -> str:
    return (
        f"T{format_number(job['temperature'])}"
        f"_p{format_number(job['top_p'])}"
        f"_K{job['k']}"
        f"_d{job['draws']}"
    )


# Nodes verified to reproduce each other bit-for-bit on the same GPU model.
# Each entry was admitted by re-measuring one stochastic published cell
# (Graph/xGRPO/seed 43 at T=1, distinct@8 = 2.4062) on that node and requiring
# exact agreement on all four metrics; the probes are retained under
# var/data/e72_frontier_hwcheck/. Reproducibility tracks the GPU model, not the
# node, so widening within a verified pool is safe while crossing models is not.
# node302 is the cluster's only a100 host, so the a100 pool cannot be widened.
VERIFIED_SAME_MODEL_POOL: dict[str, str] = {
    "node105": "node105,node202,node203,node204",
    "node202": "node105,node202,node203,node204",
    "node203": "node105,node202,node203,node204",
    "node204": "node105,node202,node203,node204",
    "node302": "node302",
}


def resolve_nodelist(
    job: dict[str, Any], override: str = "", widen_pool: bool = False
) -> str:
    """Measure each checkpoint on the node that trained it.

    GPU type changes floating-point reduction order and therefore the sampled
    tokens; an unpinned sweep would mix hardware into the accuracy/breadth
    trade-off it exists to isolate. An explicit override is honoured, but a
    missing source node is an error rather than a silent free placement.
    """
    if override:
        return override
    node = job["source"].get("source_node")
    if not node:
        raise SystemExit(
            f"no source node recorded for {job['domain']}/{job['arm']}/"
            f"s{job['seed']}; rebuild the manifest"
        )
    if widen_pool:
        pool = VERIFIED_SAME_MODEL_POOL.get(str(node))
        if not pool:
            raise SystemExit(
                f"no verified same-GPU-model pool for {node}; probe it before widening"
            )
        return pool
    return str(node)


def cell_save_path(root: Path, job: dict[str, Any]) -> Path:
    return (
        root
        / "var"
        / "data"
        / "e72_frontier"
        / job["stage"]
        / job["domain"]
        / f"{job['arm']}_s{job['seed']}"
        / cell_tag(job)
    )


def build_export_vars(root: Path, job: dict[str, Any], save_path: Path) -> dict[str, str]:
    config = job["source"]["inherited_eval_config"]
    checkpoint = job["source"]["export"]["path"]

    env: dict[str, str] = {
        "OAT_ZERO_REPO_ROOT": str(root),
        # Evaluation is arm-independent: the plain variant keeps every
        # auxiliary objective inactive so both arms' checkpoints are measured
        # by identical machinery and only the weights differ.
        "OAT_ZERO_VARIANT": "grpo",
        "OAT_ZERO_MODEL": "custom",
        "OAT_ZERO_PRETRAIN": str(checkpoint),
        "SAVE_PATH": str(save_path),
        # --- the decoding settings under study -------------------------------
        "OAT_ZERO_EVAL_ONLY": "1",
        "OAT_ZERO_EVAL_MODE_COVERAGE_K": str(job["k"]),
        "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE": repr(job["temperature"]),
        "OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P": repr(job["top_p"]),
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": str(job["draws"]),
        # --- inherited verbatim from the source run --------------------------
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED": str(config["eval_mode_coverage_seed"]),
        "OAT_ZERO_PROMPT_DATA": str(config["prompt_data"]),
        "OAT_ZERO_EVAL_DATA": str(config["eval_data"]),
        "OAT_ZERO_PROMPT_TEMPLATE": str(config["prompt_template"]),
        "OAT_ZERO_TEST_SPLIT": str(config["test_split"]),
        "OAT_ZERO_INPUT_KEY": str(config["eval_input_key"]),
        "OAT_ZERO_OUTPUT_KEY": str(config["eval_output_key"]),
        "OAT_ZERO_EVAL_INPUT_KEY": str(config["eval_input_key"]),
        "OAT_ZERO_EVAL_OUTPUT_KEY": str(config["eval_output_key"]),
        "OAT_ZERO_PROMPT_MAX_LENGTH": str(config["prompt_max_length"]),
        "OAT_ZERO_GENERATE_MAX_LENGTH": str(config["generate_max_length"]),
        "OAT_ZERO_EVAL_GENERATE_MAX_LENGTH": str(config["eval_generate_max_length"]),
        "OAT_ZERO_MAX_MODEL_LEN": str(config["max_model_len"]),
        "OAT_ZERO_EVAL_BATCH_SIZE": str(config["eval_batch_size"]),
        "OAT_ZERO_EVAL_TEMPERATURE": repr(float(config["eval_temperature"])),
        "OAT_ZERO_VERIFIER_VERSION": str(config["verifier_version"]),
        "OAT_ZERO_MAX_TRAIN": str(config["max_train"]),
        "OAT_ZERO_NUM_SAMPLES": str(config["num_samples"]),
        "OAT_ZERO_ROLLOUT_BATCH_SIZE": str(config["rollout_batch_size"]),
        "OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE": str(config["rollout_batch_size"]),
        "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE": str(
            config["train_batch_size_per_device"]
        ),
        "OAT_ZERO_ZERO_STAGE": str(config["zero_stage"]),
        "OAT_ZERO_VLLM_GPU_RATIO": repr(float(config["vllm_gpu_ratio"])),
        "OAT_ZERO_COLLOCATE": "1" if config["collocate"] else "0",
        "OAT_ZERO_CANONICAL_ACTION_TASK": str(config["canonical_action_task"]),
        "OAT_ZERO_CANONICAL_GRAPH_ACTIONS": (
            "1" if config["canonical_graph_actions"] else "0"
        ),
        "OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT": str(
            config["canonical_graph_action_count"]
        ),
        "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING": (
            "1" if config["canonical_graph_learner_sampling"] else "0"
        ),
        "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING": (
            "1" if config["canonical_graph_fixed_shape_sampling"] else "0"
        ),
        # --- nothing is trained, stored, or recovered ------------------------
        "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
        "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
        "OAT_ZERO_MAX_QUERIES": "1",
        "OAT_ZERO_SAVE_CKPT": "0",
        "OAT_ZERO_EXPORT_STEPS": "-1",
        "OAT_ZERO_AUTO_RESUME": "0",
        "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        "OAT_ZERO_USE_WB": "0",
        # Inherited for provenance only: evaluation draws are governed by
        # eval_mode_coverage_seed, and no sampling for training occurs.
        "OAT_ZERO_SEED": str(job["seed"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if str(config["canonical_action_task"]) != "none" or config[
        "canonical_graph_actions"
    ]:
        # Canonical-action rollouts require the v0 engine for audited log
        # probabilities; the same constraint governs their evaluation.
        env["VLLM_USE_V1"] = "0"
    return env


def submit(
    job: dict[str, Any],
    env: dict[str, str],
    *,
    root: Path,
    partition: str,
    account: str,
    nodelist: str,
    gres: str,
    cpus: int,
    memory: str,
    time_limit: str,
    hold: bool,
    dry_run: bool,
) -> str | None:
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    name = (
        f"e72f-{job['domain'][:6]}-{job['arm'][:6]}-s{job['seed']}-{cell_tag(job)}"
    )
    sbatch_args = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        f"--gres={gres}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
    ]
    if nodelist:
        sbatch_args.append(f"--nodelist={nodelist}")
    if partition:
        sbatch_args.append(f"--partition={partition}")
    if account:
        sbatch_args.append(f"--account={account}")
    if hold:
        sbatch_args.append("--hold")
    sbatch_args.append(str(root / "ops" / "slurm" / "train_node302.slurm"))

    if dry_run:
        print(" ".join(shlex.quote(part) for part in sbatch_args))
        return None
    result = subprocess.run(
        sbatch_args, cwd=str(root), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"sbatch failed for {name}: {result.stderr.strip()}")
    return result.stdout.strip()


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b", "c"), default="a")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "var" / "artifacts" / "e72_frontier_source_runs.json",
    )
    parser.add_argument("--domains", default="", help="comma-separated subset")
    parser.add_argument("--arms", default="", help="comma-separated subset")
    parser.add_argument("--seeds", default="", help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=0, help="submit at most N cells")
    parser.add_argument("--partition", default="mltheory")
    parser.add_argument("--account", default="mltheory")
    # Empty means "use the node that trained this checkpoint", which is the
    # correct default; an explicit value overrides it for every cell and should
    # only be used deliberately.
    parser.add_argument("--nodelist", default="")
    parser.add_argument(
        "--widen-pool",
        action="store_true",
        help="submit to the verified same-GPU-model pool instead of the single "
        "training node; requires a partition that contains the whole pool",
    )
    parser.add_argument(
        "--source-nodes",
        default="",
        help="comma-separated source nodes to restrict this submission to",
    )
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--time-limit", default="01:30:00")
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="skip cells that already carry an EVAL_ONLY_COMPLETE.json marker",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("problems"):
        raise SystemExit(
            f"{args.manifest} records unresolved problems; rebuild it before launching"
        )

    domains = {part for part in args.domains.split(",") if part}
    arms = {part for part in args.arms.split(",") if part}
    seeds = {int(part) for part in args.seeds.split(",") if part}
    source_nodes = {part for part in args.source_nodes.split(",") if part}

    submitted: list[dict[str, Any]] = []
    skipped = 0
    for job in iter_jobs(manifest, args.stage, domains=domains, arms=arms, seeds=seeds):
        if source_nodes and str(job["source"].get("source_node")) not in source_nodes:
            continue
        save_path = cell_save_path(root, job)
        if args.skip_complete and any(
            save_path.glob("*/EVAL_ONLY_COMPLETE.json")
        ):
            skipped += 1
            continue
        if args.limit and len(submitted) >= args.limit:
            break
        save_path.mkdir(parents=True, exist_ok=True)
        env = build_export_vars(root, job, save_path)
        nodelist = resolve_nodelist(job, args.nodelist, widen_pool=args.widen_pool)
        job_id = submit(
            job,
            env,
            root=root,
            partition=args.partition,
            account=args.account,
            nodelist=nodelist,
            gres=args.gres,
            cpus=args.cpus,
            memory=args.memory,
            time_limit=args.time_limit,
            hold=args.hold,
            dry_run=args.dry_run,
        )
        submitted.append(
            {
                "stage": job["stage"],
                "domain": job["domain"],
                "arm": job["arm"],
                "seed": job["seed"],
                "temperature": job["temperature"],
                "top_p": job["top_p"],
                "k": job["k"],
                "draws": job["draws"],
                "save_path": str(save_path),
                "checkpoint": job["source"]["export"]["path"],
                "nodelist": nodelist,
                "source_node": job["source"].get("source_node"),
                "slurm_job_id": job_id,
            }
        )

    print(
        f"[e72-frontier] stage={args.stage} submitted={len(submitted)} "
        f"skipped_complete={skipped} dry_run={args.dry_run}"
    )
    if args.dry_run or not submitted:
        return 0

    ledger = (
        root / "var" / "artifacts" / f"e72_frontier_stage_{args.stage}_jobs.json"
    )
    existing: list[dict[str, Any]] = []
    if ledger.is_file():
        existing = json.loads(ledger.read_text()).get("cells", [])
    payload = {
        "schema": "e72_frontier_jobs_v1",
        "stage": args.stage,
        "manifest": str(args.manifest),
        "cells": existing + submitted,
    }
    handle, temporary = tempfile.mkstemp(prefix=f".{ledger.name}.", dir=ledger.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, ledger)
    print(f"[e72-frontier] ledger {ledger} now holds {len(payload['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
