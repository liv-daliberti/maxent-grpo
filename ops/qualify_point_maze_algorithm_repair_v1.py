#!/usr/bin/env python3
"""Bind PointMaze repair viability to the paired-qualification adapter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viability", type=Path, required=True)
    parser.add_argument("--viability-identity", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--data-identity", type=Path, required=True)
    parser.add_argument("--clarification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-identity", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.viability,
        args.viability_identity,
        args.admission,
        args.data_identity,
        args.clarification,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() or args.output_identity.exists():
        raise FileExistsError("fresh PointMaze repair qualification required")
    viability = json.loads(args.viability.read_text())
    viability_identity = json.loads(args.viability_identity.read_text())
    admission = json.loads(args.admission.read_text())
    summary = viability.get("summary", {})
    attempts = viability.get("attempts", [])
    verified = int(summary.get("verified_completions", -1))
    rate = verified / len(attempts) if attempts else -1.0
    checks = {
        "viability_status_pass": viability.get("status") == "pass",
        "viability_decision": viability.get("decision")
        == "eligible_for_shared_warmstart_design",
        "at_least_one_verified_completion": verified >= 1,
        "non_saturated_verified_rate": 0.0 < rate <= 0.90,
        "at_least_one_multimode_prompt": int(summary.get("multimode_prompts", 0))
        >= 1,
        "at_least_one_prefix_success_prompt": int(
            summary.get("prefix_success_prompts", 0)
        )
        >= 1,
        "admission_pass": admission.get("status") == "pass",
        "viability_job_identity": viability_identity.get("job_id") == 30204570,
        "development_only": viability.get("information_boundary", {}).get(
            "evaluation_split_only"
        )
        in (True, "multi_answer"),
    }
    errors = [f"failed check: {key}" for key, passed in checks.items() if not passed]
    status = "pass" if not errors else "fail"
    identity = {
        "schema": "point-maze-algorithm-repair-v1-qualification-identity",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "viability_job_id": 30204570,
        "viability_receipt_sha256": sha(args.viability),
        "viability_identity_sha256": sha(args.viability_identity),
        "admission_audit_sha256": sha(args.admission),
        "data_identity_sha256": sha(args.data_identity),
        "cardinality_clarification_sha256": sha(args.clarification),
        "development_only": True,
        "final_seed": False,
        "verified_completions": verified,
        "attempt_count": len(attempts),
        "verified_rate": rate,
    }
    atomic(args.output_identity, identity)
    payload = {
        "schema": "point-maze-algorithm-repair-v1-qualification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": (
            "eligible_for_ten_point_maze_stage_b_jobs"
            if status == "pass"
            else "point_maze_algorithm_repair_pair_stopped"
        ),
        "checks": checks,
        "errors": errors,
        "identity_sha256": sha(args.output_identity),
        "summary": {
            "verified_completions": verified,
            "attempt_count": len(attempts),
            "verified_rate": rate,
            "multimode_prompts": summary.get("multimode_prompts"),
            "prefix_success_prompts": summary.get("prefix_success_prompts"),
        },
    }
    atomic(args.output, payload)
    print(json.dumps({"status": status, "verified_rate": rate, "errors": errors}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

