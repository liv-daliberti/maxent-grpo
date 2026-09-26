#!/usr/bin/env python3
"""Submit one fail-closed CPU job that releases E109/E105 after E110."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
E110_JOB_ID = 30647379
GATE = ROOT / "var/artifacts/e106_python_lambda_normalization_combined_gate.json"
E110_LEDGER = ROOT / "var/artifacts/e110_falcon_python_admission_horizon_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e105_post_gate_automatic_release_20260818.md"
SLURM_SCRIPT = ROOT / "ops/slurm/e105_post_gate_release.slurm"
OUT = ROOT / "var/artifacts/e105_post_gate_release_job.json"
DOWNSTREAM_ABSENT = (
    ROOT / "var/artifacts/e105_qwen3_paired_a6000_placement_amendment.json",
    ROOT / "var/artifacts/e109_repaired_python_replay_comparators_jobs.json",
    ROOT / "var/artifacts/e105_group_centered_semantic_repair_full_three_scale_jobs.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def validate_current_state() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (GATE, E110_LEDGER, PROTOCOL, SLURM_SCRIPT):
        if not path.is_file():
            raise RuntimeError(f"post-gate release input is absent: {path}")
    if OUT.exists():
        raise RuntimeError(f"refusing duplicate post-gate release submission: {OUT}")
    present = [str(path) for path in DOWNSTREAM_ABSENT if path.exists()]
    if present:
        raise RuntimeError(f"downstream release artifacts already exist: {present}")
    gate = load(GATE)
    if gate.get("schema") != "e106_python_lambda_normalization_combined_gate_v1":
        raise RuntimeError("combined gate schema drifted")
    if gate.get("complete") is not False or gate.get("passed") is not False:
        raise RuntimeError("post-gate dependency must be submitted from the 14/15 state")
    if gate.get("pointmaze") != "excluded" or gate.get("mechanism_gate_used_outcome_metrics") is not False or gate.get("post_update_outcome_metrics_inspected") is not False:
        raise RuntimeError("combined gate violated outcome blinding or domain scope")
    runs = list(gate.get("runs", []))
    terminal = [run for run in runs if run.get("runtime_complete") is True]
    incomplete = [run for run in runs if run.get("runtime_complete") is not True]
    if len(runs) != 15 or len(terminal) != 14 or len(incomplete) != 1 or int(incomplete[0].get("job_id", -1)) != E110_JOB_ID:
        raise RuntimeError("post-gate dependency is not bound to the sole 14/15 E110 cell")
    ledger = load(E110_LEDGER)
    ledger_jobs = [int(run["job_id"]) for run in ledger.get("runs", [])]
    if ledger.get("released") is not True or ledger_jobs != [E110_JOB_ID]:
        raise RuntimeError("E110 release ledger drifted")
    return gate, ledger


def scheduler_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", str(job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"cannot inspect post-gate job {job_id}")
    return result.stdout.strip()


def command() -> list[str]:
    export = ",".join(
        (
            "ALL",
            f"OAT_ZERO_REPO_ROOT={ROOT}",
            f"OAT_ZERO_RELEASE_SCRIPT_SHA256={digest(SLURM_SCRIPT)}",
            f"OAT_ZERO_RELEASE_PYTHON={Path(sys.executable).resolve()}",
        )
    )
    return [
        "sbatch", "--parsable", f"--dependency=afterany:{E110_JOB_ID}",
        "--job-name=e105-post-gate", f"--export={export}",
        "--partition=all", "--account=allcs", "--cpus-per-task=1",
        "--mem=8G", "--time=01:00:00", "--nice=0", "--no-requeue",
        str(SLURM_SCRIPT),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    gate, ledger = validate_current_state()
    submit = command()
    if not args.submit:
        print(" ".join(shlex.quote(token) for token in submit))
        print("[e105-post-gate] dry_run=True dependency=afterany:30647379")
        return 0
    job_id: int | None = None
    try:
        result = subprocess.run(submit, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"post-gate sbatch failed: {result.stderr.strip()}")
        raw = result.stdout.strip().split(";", 1)[0]
        if not raw.isdigit():
            raise RuntimeError(f"invalid post-gate job id: {result.stdout!r}")
        job_id = int(raw)
        record = scheduler_record(job_id)
        required = (
            "JobState=PENDING", "RunTime=00:00:00",
            f"Dependency=afterany:{E110_JOB_ID}", "Partition=all",
            "Account=allcs", "MinMemoryNode=8G", "TimeLimit=01:00:00",
            f"Command={SLURM_SCRIPT}",
        )
        missing = [needle for needle in required if needle not in record]
        if missing:
            raise RuntimeError(f"post-gate scheduler record lacks {missing}")
        payload = {
            "schema": "e105_post_gate_release_job_v1",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id, "dependency": f"afterany:{E110_JOB_ID}",
            "dependency_job_id": E110_JOB_ID, "dependency_locked": True,
            "fail_closed_on_gate": True, "submitted": True,
            "scientific_configuration_changed": False,
            "outcome_metrics_inspected": False, "pointmaze": "excluded",
            "gate_at_submission": str(GATE),
            "gate_at_submission_sha256": digest(GATE),
            "terminal_cells_at_submission": 14,
            "e110_ledger": str(E110_LEDGER), "e110_ledger_sha256": digest(E110_LEDGER),
            "protocol": str(PROTOCOL), "protocol_sha256": digest(PROTOCOL),
            "slurm_script": str(SLURM_SCRIPT), "slurm_script_sha256": digest(SLURM_SCRIPT),
            "launcher": str(Path(__file__).resolve()), "launcher_sha256": digest(Path(__file__)),
            "python": str(Path(sys.executable).resolve()),
            "downstream_order": ["refresh_gate", "render_mechanism", "paired_placement", "submit_e109", "submit_e105", "campaign_stats"],
            "scheduler_record": record,
            "e110_record_at_submission": ledger["runs"][0].get("held_scheduler_record"),
        }
        e81.atomic_json(OUT, payload)
    except Exception:
        if job_id is not None:
            subprocess.run(["scancel", str(job_id)], check=False)
        raise
    print(f"[e105-post-gate] job={job_id} dependency=afterany:{E110_JOB_ID} artifact={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
