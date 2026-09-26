#!/usr/bin/env python3
"""Submit the E125 decoding-frontier grid on the current eight-pass arms.

This is the E72 decoding grid, unchanged, pointed at a different cohort: the
Dr.GRPO control and Re:Dr replay terminal checkpoints the rest of the paper
reports, rather than the earlier twelve-pass cohort whose replay arm predates
the Re:Dr objective contract. Every cell loads one frozen checkpoint, evaluates
it once through the ordinary training evaluation path at a chosen
(temperature, top_p, K), and exits; no optimizer step, rollout, or export
occurs.

The cell construction, the inherited-argument discipline, and the per-cell
pinning all come from ``launch_e72_decoding_frontier`` by import rather than by
copy, so the two sweeps cannot drift apart in anything except which checkpoints
they run on. Only the destinations differ: cells land under
``var/data/e125_frontier`` and the submission ledger is E125's, so neither
sweep can overwrite the other's records.

Pinning is exact here. All fifty checkpoints trained on node302 (a100) or
node105 (a5000), and both arms of every domain/seed pair trained on the same
host, so every cell runs on the GPU model that produced its weights and every
paired contrast is hardware-matched. ``--widen-pool`` is deliberately not
exposed: there is no cell that needs it.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_decoding_frontier as e72  # noqa: E402

CAMPAIGN = "e125_frontier"

#: Reproduce the draw seeding these checkpoints were published under. Live
#: source now strides draw seeds by K so a prompt's draws use disjoint vLLM
#: child streams; the E78 terminal evaluations predate that and seeded draws
#: consecutively, sharing streams. Sweeping under the corrected scheme would
#: change which samples are drawn at every temperature including T=1, so the
#: grid would no longer be anchored to the numbers the paper prints. The fix is
#: not in scope for this sweep, which varies decoding and nothing else.
DISJOINT_DRAWS = "0"
MANIFEST = "var/artifacts/e125_frontier_source_runs.json"
PROTOCOL = "paper/preregistration/e125_decoding_frontier_current_arms_20260914.md"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cell_save_path(root: Path, job: dict[str, Any], campaign: str = CAMPAIGN) -> Path:
    return (
        root
        / "var"
        / "data"
        / campaign
        / job["stage"]
        / job["domain"]
        / f"{job['arm']}_s{job['seed']}"
        / e72.cell_tag(job)
    )


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
    """Submit one cell.

    A near-copy of ``e72.submit`` for one reason only: the job name. Names are
    how a half-submitted sweep is read off ``squeue``, and two campaigns sharing
    a prefix would make the E125 cells indistinguishable from E72's.
    """

    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    name = f"e125f-{job['domain'][:6]}-{job['arm'][:6]}-s{job['seed']}-{e72.cell_tag(job)}"
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


def restorable(job: dict[str, Any]) -> str | None:
    """Why this cell cannot run yet, or None if its weights are on disk.

    Weights are retired to the model archive after a campaign completes. A cell
    submitted against a retired checkpoint does not fail loudly --- it would
    load whatever the path resolves to --- so the check is made here, before
    submission, rather than discovered in a log.
    """

    export = Path(job["source"]["export"]["path"])
    weights = export / "model.safetensors"
    if weights.is_file():
        return None
    archive = job["source"]["export"].get("archive") or {}
    receipt = archive.get("receipt")
    if receipt:
        return f"weights retired; restore with --receipt {receipt}"
    return "weights absent and no archive receipt"


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b", "c"), default="a")
    parser.add_argument("--manifest", type=Path, default=root / MANIFEST)
    parser.add_argument("--domains", default="", help="comma-separated subset")
    parser.add_argument("--arms", default="", help="comma-separated subset")
    parser.add_argument("--seeds", default="", help="comma-separated subset")
    parser.add_argument(
        "--temperatures",
        default="",
        help="comma-separated subset of the stage's temperatures; the gate runs "
        "on T=1.0 alone, so that column is worth submitting and checking before "
        "the rest of the grid is queued",
    )
    parser.add_argument(
        "--source-nodes",
        default="",
        help="comma-separated training nodes to restrict this submission to; "
        "the sweep is pinned per checkpoint, so this selects which pinned "
        "subset goes out now",
    )
    parser.add_argument(
        "--nodelist",
        default="",
        help="override the per-checkpoint pin. Only for the registered hardware "
        "control, which deliberately measures a checkpoint on a GPU model that "
        "did not train it; it requires --campaign so those cells cannot be read "
        "as part of the pinned grid",
    )
    parser.add_argument(
        "--campaign",
        default=CAMPAIGN,
        help="data root under var/data. Anything measured off its training node "
        "must be written somewhere the grid's readers do not look",
    )
    parser.add_argument(
        "--uniform-hardware",
        action="store_true",
        help="permit --nodelist to write into the main campaign, for the case "
        "where every checkpoint is measured on one GPU model on purpose. The "
        "grid is then internally uniform rather than per-checkpoint pinned, and "
        "any cell already measured on other hardware must be moved out first: "
        "one checkpoint's temperature curve must never span two GPU models",
    )
    parser.add_argument("--limit", type=int, default=0, help="submit at most N cells")
    parser.add_argument("--partition", default="mltheory")
    parser.add_argument("--account", default="mltheory")
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

    if args.nodelist and args.campaign == CAMPAIGN and not args.uniform_hardware:
        raise SystemExit(
            "--nodelist unpins the sweep; pass --campaign to keep those cells "
            f"out of {CAMPAIGN}, or --uniform-hardware if the whole grid is "
            "deliberately being measured on one GPU model"
        )
    if args.uniform_hardware and not args.nodelist:
        raise SystemExit("--uniform-hardware only means something with --nodelist")

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("problems"):
        raise SystemExit(
            f"{args.manifest} records unresolved problems; rebuild it before launching"
        )
    if manifest.get("schema") != "e125-frontier-source-runs-v1":
        raise SystemExit(f"{args.manifest} is not an E125 source manifest")

    domains = {part for part in args.domains.split(",") if part}
    arms = {part for part in args.arms.split(",") if part}
    seeds = {int(part) for part in args.seeds.split(",") if part}
    source_nodes = {part for part in args.source_nodes.split(",") if part}
    temperatures = {float(part) for part in args.temperatures.split(",") if part}
    stage_temperatures = {cell[0] for cell in e72.stage_cells(args.stage)}
    unknown = temperatures - stage_temperatures
    if unknown:
        raise SystemExit(
            f"stage {args.stage} has no temperature {sorted(unknown)}; "
            f"it sweeps {sorted(stage_temperatures)}"
        )

    submitted: list[dict[str, Any]] = []
    skipped = 0
    # Keyed by checkpoint, not by cell: one retired checkpoint blocks all six of
    # its temperatures, and listing it six times hides how many are really left.
    unrestored: dict[str, str] = {}
    for job in e72.iter_jobs(manifest, args.stage, domains=domains, arms=arms, seeds=seeds):
        if source_nodes and str(job["source"].get("source_node")) not in source_nodes:
            continue
        if temperatures and float(job["temperature"]) not in temperatures:
            continue
        save_path = cell_save_path(root, job, args.campaign)
        if args.skip_complete and any(save_path.glob("*/EVAL_ONLY_COMPLETE.json")):
            skipped += 1
            continue
        blocked = restorable(job)
        if blocked:
            checkpoint = f"{job['domain']}/{job['arm']}/s{job['seed']}"
            unrestored[checkpoint] = blocked
            continue
        if args.limit and len(submitted) >= args.limit:
            break
        save_path.mkdir(parents=True, exist_ok=True)
        env = e72.build_export_vars(root, job, save_path)
        env["OAT_ZERO_EVAL_MODE_COVERAGE_DISJOINT_DRAWS"] = DISJOINT_DRAWS
        # "any" means: do not name a host at all. Uniform hardware is then
        # enforced by --gres (e.g. gpu:a5000:1), which lets the scheduler use
        # every host of that model instead of queueing behind one named node.
        nodelist = (
            "" if args.nodelist == "any" else e72.resolve_nodelist(job, args.nodelist)
        )
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

    if unrestored:
        # Not a failure: a partial submission by design restores and submits one
        # pinned subset at a time. The blocked cells are named so the next
        # restore is a copy-paste rather than a rediscovery.
        print(
            f"[e125-frontier] {len(unrestored)} checkpoints skipped for "
            "unrestored weights:"
        )
        for checkpoint, reason in sorted(unrestored.items())[:10]:
            print(f"[e125-frontier]   {checkpoint}: {reason}")
        if len(unrestored) > 10:
            print(f"[e125-frontier]   ... and {len(unrestored) - 10} more")

    print(
        f"[e125-frontier] stage={args.stage} submitted={len(submitted)} "
        f"skipped_complete={skipped} unrestored={len(unrestored)} "
        f"dry_run={args.dry_run}"
    )
    if args.dry_run or not submitted:
        return 0

    ledger = (
        root / "var" / "artifacts" / f"{args.campaign}_stage_{args.stage}_jobs.json"
    )
    existing: list[dict[str, Any]] = []
    if ledger.is_file():
        existing = json.loads(ledger.read_text()).get("cells", [])
    payload = {
        "schema": "e125_frontier_jobs_v1",
        "stage": args.stage,
        "campaign": args.campaign,
        "pinned": not args.nodelist,
        "manifest": str(args.manifest),
        "protocol": str(root / PROTOCOL),
        "cells": existing + submitted,
    }
    handle, temporary = tempfile.mkstemp(prefix=f".{ledger.name}.", dir=ledger.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, ledger)
    print(f"[e125-frontier] ledger {ledger} now holds {len(payload['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
