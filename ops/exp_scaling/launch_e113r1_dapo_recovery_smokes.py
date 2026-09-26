#!/usr/bin/env python3
"""Submit the two non-scientific E113-R1 DAPO recovery smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e113_dapo_direct_baseline as e113  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402


COHORT = "e113r1"
SMOKE_DOMAIN = "graph_coloring"
SMOKE_MAX_TRAIN = 32
SMOKE_MAX_QUERIES = (
    SMOKE_MAX_TRAIN * 16 * e113.MAX_GENERATION_BATCHES
)
LEDGER = "var/artifacts/e113r1_dapo_recovery_smoke_jobs.json"
AMENDMENT = (
    "paper/preregistration/e113r1_dapo_recovery_smokes_20260819.md"
)
ORIGINAL_LEDGER = e113.LEDGER


def recovery_run_stamp(family: str) -> str:
    return f"e113r1_{family}_graph_dapo_smoke"


def recovery_job_name(family: str) -> str:
    tag = "q05" if family == "qwen05b" else "f1"
    return f"e113r1-{tag}-graph-dapo-smoke"


def recovery_target(root: Path, family: str) -> Path:
    return root / "var/data" / recovery_run_stamp(family)


def recovery_env(
    root: Path,
    family: str,
    run: dict[str, Any],
    snapshot: Path,
    falcon_root: Path | None,
) -> dict[str, str]:
    env, _ = e113.smoke_env(root, family, run, snapshot, falcon_root)
    env.update(
        {
            "SAVE_PATH": str(recovery_target(root, family)),
            "RUN_STAMP": recovery_run_stamp(family),
            "OAT_ZERO_MAX_QUERIES": str(SMOKE_MAX_QUERIES),
        }
    )
    return env


def recovery_command(
    root: Path,
    family: str,
    run: dict[str, Any],
    env: dict[str, str],
) -> list[str]:
    command = e113.sbatch_command(root, family, run, env, smoke=True)
    return [
        (
            f"--job-name={recovery_job_name(family)}"
            if token.startswith("--job-name=")
            else token
        )
        for token in command
    ]


def audit_held(
    job_id: str,
    family: str,
    run: dict[str, Any],
    snapshot: Path,
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect held E113-R1 job {job_id}")
    required = [
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName={recovery_job_name(family)}",
        f"SAVE_PATH={recovery_target(e113.repo_root(), family)}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        "OAT_ZERO_VARIANT=dapo",
        "OAT_ZERO_CRITIC_TYPE=grpo",
        "OAT_ZERO_DAPO_ENABLED=1",
        "OAT_ZERO_DAPO_CLIP_LOW=0.20",
        "OAT_ZERO_DAPO_CLIP_HIGH=0.28",
        "OAT_ZERO_DAPO_MAX_NUM_GEN_BATCHES=10",
        f"OAT_ZERO_MAX_QUERIES={SMOKE_MAX_QUERIES}",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=0",
        "OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=0",
    ]
    if family == "qwen05b":
        required.append(f"ReqNodeList={run['source_node']}")
    else:
        node, gpu = e79.placement(SMOKE_DOMAIN, int(run["seed"]))
        required.extend((f"ReqNodeList={node}", f"gres/gpu:{gpu}=1"))
    missing = [value for value in required if value not in result.stdout]
    if missing:
        raise RuntimeError(f"held E113-R1 job {job_id} lacks {missing}")
    return result.stdout


def load_original(root: Path) -> dict[str, Any]:
    path = root / ORIGINAL_LEDGER
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "e113_dapo_direct_baseline_jobs_v1":
        raise SystemExit("original E113 ledger schema drifted")
    if payload.get("released") is not True or len(payload.get("runs", [])) != 50:
        raise SystemExit("original E113 ledger is not the released 50-cell record")
    return payload


def smoke_runs(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for family, seeds in e113.FAMILY_SEEDS.items():
        run = next(
            row
            for row in e113.references(root, family)
            if str(row["domain"]) == SMOKE_DOMAIN
            and int(row["seed"]) == int(seeds[0])
        )
        result[family] = run
    return result


def main() -> int:
    root = e113.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    ledger_path = root / LEDGER
    amendment = root / AMENDMENT
    original_path = root / ORIGINAL_LEDGER
    for required in (amendment, original_path):
        if not required.is_file():
            raise SystemExit(f"required E113-R1 input is absent: {required}")
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate E113-R1 submission: {ledger_path}")

    original = load_original(root)
    snapshot = Path(str(original["snapshot_root"])).resolve()
    e113.verify_snapshot(snapshot)
    falcon_root = e79.model_root(root)
    runs = smoke_runs(root)
    prepared: dict[str, dict[str, Any]] = {}
    for family, run in runs.items():
        target = recovery_target(root, family)
        if target.exists():
            raise SystemExit(f"refusing to overwrite E113-R1 smoke: {target}")
        env = recovery_env(root, family, run, snapshot, falcon_root)
        prepared[family] = {
            "run": run,
            "env": env,
            "target": target,
            "command": recovery_command(root, family, run, env),
        }

    if args.dry_run or not args.submit:
        for smoke in prepared.values():
            print(shlex.join(smoke["command"]))
        print(
            f"[e113r1] smokes=2 scientific=0 domain={SMOKE_DOMAIN} "
            f"max_queries={SMOKE_MAX_QUERIES} snapshot={snapshot}"
        )
        return 0

    submitted: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    try:
        for family, smoke in prepared.items():
            job_id = e113.submit_held(smoke["command"])
            submitted.append(job_id)
            held = audit_held(job_id, family, smoke["run"], snapshot)
            records[family] = {
                "scientific": False,
                "family": family,
                "domain": SMOKE_DOMAIN,
                "seed": int(smoke["run"]["seed"]),
                "job_id": int(job_id),
                "run_stamp": recovery_run_stamp(family),
                "run_dir": str(smoke["target"]),
                "max_train": SMOKE_MAX_TRAIN,
                "max_queries": SMOKE_MAX_QUERIES,
                "held_scheduler_record": held,
            }

        payload = {
            "schema": "e113r1_dapo_recovery_smoke_jobs_v1",
            "cohort": COHORT,
            "released": False,
            "scientific_cells": 0,
            "target_steps": SMOKE_MAX_TRAIN,
            "passes": 1,
            "smoke_domain": SMOKE_DOMAIN,
            "max_generation_batches": e113.MAX_GENERATION_BATCHES,
            "max_queries_per_smoke": SMOKE_MAX_QUERIES,
            "amendment": str(amendment),
            "amendment_sha256": e113.e78.digest(amendment),
            "launcher_sha256": e113.e78.digest(Path(__file__)),
            "original_e113_ledger": str(original_path),
            "original_e113_ledger_sha256": e113.e78.digest(original_path),
            "snapshot_root": str(snapshot),
            "snapshot_identity": original["snapshot_identity"],
            "objective": e113.objective(),
            "smokes": records,
            "runs": [],
        }
        e113.e78.atomic_json(ledger_path, payload)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", job_id], check=True)
        payload["released"] = True
        e113.e78.atomic_json(ledger_path, payload)
    except Exception:
        e113.cancel(submitted)
        raise

    print(f"[e113r1] released 2 recovery smokes and 0 scientific cells")
    print(f"[e113r1] ledger {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

