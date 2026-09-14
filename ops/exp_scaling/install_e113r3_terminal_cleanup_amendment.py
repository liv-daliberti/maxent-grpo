#!/usr/bin/env python3
"""Record the exact-job R3 terminal-failure cleanup amendment."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e113_dapo_direct_baseline as e113  # noqa: E402


LEDGER = "var/artifacts/e113r3_dapo_full_relaunch_jobs.json"
PROTOCOL = "paper/preregistration/e113r3s3_terminal_failure_cleanup_20260819.md"
WRAPPER = "ops/slurm/train_node302.slurm"
OUTPUT = "var/artifacts/e113r3_terminal_failure_cleanup_amendment.json"
OBSERVED_JOB = 30790926
FATAL = (
    "DAPO dynamic sampling could not produce a non-constant reward group "
    "within 10 generation batches at learner step 1; rejected=10, "
    "all_zero=10, all_one=0"
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root = e113.repo_root()
    ledger_path = root / LEDGER
    protocol_path = root / PROTOCOL
    wrapper_path = root / WRAPPER
    output_path = root / OUTPUT
    for path in (ledger_path, protocol_path, wrapper_path):
        if not path.is_file():
            raise SystemExit(f"required amendment input is absent: {path}")
    if output_path.exists():
        raise SystemExit(f"refusing duplicate amendment record: {output_path}")
    ledger = load(ledger_path)
    job_ids = sorted(int(run["job_id"]) for run in ledger.get("runs", []))
    if ledger.get("schema") != "e113r3_dapo_full_relaunch_jobs_v1":
        raise SystemExit("R3 ledger schema drifted")
    if ledger.get("released") is not True or job_ids != list(range(30790925, 30790975)):
        raise SystemExit("R3 ledger does not bind the exact released 50 jobs")
    observed = next(run for run in ledger["runs"] if int(run["job_id"]) == OBSERVED_JOB)
    if (Path(str(observed["run_dir"])) / "TRAINING_COMPLETE.json").exists():
        raise SystemExit("observed exhausted job unexpectedly has a completion receipt")
    logs = list((root / "var/artifacts/logs").glob(f"e113r3-*-{OBSERVED_JOB}.out"))
    if len(logs) != 1 or FATAL not in logs[0].read_text(encoding="utf-8", errors="replace"):
        raise SystemExit("observed DAPO exhaustion evidence is absent or ambiguous")
    wrapper = wrapper_path.read_text(encoding="utf-8")
    required_wrapper = (
        "BEGIN E113R3_TERMINAL_FAILURE_CLEANUP_AMENDMENT",
        "30790925",
        "30790974",
        "OAT_ZERO_WATCHDOG_STALE_SECONDS=300",
        "DAPO dynamic sampling could not produce a non-constant reward group",
    )
    missing = [value for value in required_wrapper if value not in wrapper]
    if missing:
        raise SystemExit(f"R3 wrapper amendment lacks {missing}")
    payload = {
        "schema": "e113r3_terminal_failure_cleanup_amendment_v1",
        "installed": True,
        "outcomes_inspected": True,
        "efficacy_outcomes_inspected": False,
        "scientific_cells": 50,
        "same_scientific_cells": True,
        "exact_job_ids": job_ids,
        "wrapper_requeue_after_learner_failure": False,
        "ledger": str(ledger_path),
        "ledger_sha256": e113.e78.digest(ledger_path),
        "protocol": str(protocol_path),
        "protocol_sha256": e113.e78.digest(protocol_path),
        "wrapper": str(wrapper_path),
        "wrapper_sha256": e113.e78.digest(wrapper_path),
        "installer_sha256": e113.e78.digest(Path(__file__)),
        "observed_failure": {
            "job_id": OBSERVED_JOB,
            "model_family": observed["model_family"],
            "domain": observed["domain"],
            "seed": observed["seed"],
            "accepted_updates": 0,
            "fatal_signature": FATAL,
            "log": str(logs[0]),
            "log_sha256": e113.e78.digest(logs[0]),
        },
    }
    e113.e78.atomic_json(output_path, payload)
    print(f"[e113r3s3] installed cleanup amendment for {len(job_ids)} exact jobs")
    print(f"[e113r3s3] record {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
