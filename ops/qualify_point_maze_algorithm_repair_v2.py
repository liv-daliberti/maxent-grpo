#!/usr/bin/env python3
"""Qualify PointMaze v2 using its actual sixteen-rollout training prefix."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-identity", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.viability,
        args.viability_identity,
        args.admission,
        args.data_identity,
        args.protocol,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() or args.output_identity.exists():
        raise FileExistsError("fresh PointMaze v2 qualification required")
    viability = json.loads(args.viability.read_text())
    viability_identity = json.loads(args.viability_identity.read_text())
    admission = json.loads(args.admission.read_text())
    attempts = viability.get("attempts", [])
    by_prompt: dict[int, list[dict]] = defaultdict(list)
    for attempt in attempts:
        by_prompt[int(attempt["row_index"])].append(attempt)
    prefix16_success_prompts = sum(
        any(
            bool(attempt.get("verified"))
            and int(attempt.get("sample_index", 10**9)) <= 16
            for attempt in prompt_attempts
        )
        for prompt_attempts in by_prompt.values()
    )
    verified = sum(bool(attempt.get("verified")) for attempt in attempts)
    rate = verified / len(attempts) if attempts else -1.0
    summary = viability.get("summary", {})
    checks = {
        "exact_failed_eight_draw_gate": (
            viability.get("status") == "fail"
            and viability.get("decision")
            == "point_maze_velocity_warmstart_v3_ineligible"
            and int(summary.get("prefix_success_prompts", -1)) == 0
        ),
        "three_training_prefix_success_prompts": (
            prefix16_success_prompts >= 3
        ),
        "two_multimode_prompts": int(summary.get("multimode_prompts", 0)) >= 2,
        "verified_rate_in_0p04_0p25": 0.04 <= rate <= 0.25,
        "admission_pass": (
            admission.get("status") == "pass"
            and admission.get("decision")
            == "admitted_for_0.5b_viability_sampling"
        ),
        "viability_identity": (
            viability_identity.get("job_id") == 30204702
            and viability_identity.get("seed") == 76520
            and viability_identity.get("v2_final_seed") is False
        ),
        "development_only": viability.get("information_boundary", {}).get(
            "development_only"
        )
        is True,
    }
    errors = [
        f"failed check: {key}" for key, passed in checks.items() if not passed
    ]
    status = "pass" if not errors else "fail"
    identity = {
        "schema": "point-maze-algorithm-repair-v2-qualification-identity",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "viability_job_id": 30204702,
        "viability_seed": 76520,
        "viability_receipt_sha256": sha(args.viability),
        "viability_identity_sha256": sha(args.viability_identity),
        "admission_audit_sha256": sha(args.admission),
        "data_identity_sha256": sha(args.data_identity),
        "protocol_sha256": sha(args.protocol),
        "development_only": True,
        "final_seed": False,
        "verified_completions": verified,
        "attempt_count": len(attempts),
        "verified_rate": rate,
        "prefix_size_aligned_to_online_rollouts": 16,
        "prefix16_success_prompts": prefix16_success_prompts,
        "v2_attempts_resampled": False,
    }
    atomic(args.output_identity, identity)
    payload = {
        "schema": "point-maze-algorithm-repair-v2-qualification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": (
            "eligible_for_point_maze_algorithm_repair_v2_pair"
            if status == "pass"
            else "point_maze_algorithm_repair_v2_stopped"
        ),
        "checks": checks,
        "errors": errors,
        "identity_sha256": sha(args.output_identity),
        "summary": {
            "verified_completions": verified,
            "attempt_count": len(attempts),
            "verified_rate": rate,
            "multimode_prompts": summary.get("multimode_prompts"),
            "prefix8_success_prompts": summary.get("prefix_success_prompts"),
            "prefix16_success_prompts": prefix16_success_prompts,
        },
    }
    atomic(args.output, payload)
    print(json.dumps({"status": status, "summary": payload["summary"]}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
