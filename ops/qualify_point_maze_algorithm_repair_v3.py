#!/usr/bin/env python3
"""Fail-closed K=16 qualification for orientation-balanced PointMaze v3."""

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
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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
    if args.output.exists():
        raise FileExistsError("fresh PointMaze v3 qualification required")
    viability = json.loads(args.viability.read_text(encoding="utf-8"))
    identity = json.loads(
        args.viability_identity.read_text(encoding="utf-8")
    )
    admission = json.loads(args.admission.read_text(encoding="utf-8"))
    data = json.loads(args.data_identity.read_text(encoding="utf-8"))
    attempts = viability.get("attempts", [])
    prompt_results = viability.get("prompt_results", [])
    verified = sum(bool(row.get("verified")) for row in attempts)
    verified_rate = verified / len(attempts) if attempts else -1.0
    prefix_success_prompts = sum(
        int(row.get("verified_in_prefix", 0)) > 0
        for row in prompt_results
    )
    multimode_prompts = sum(
        int(row.get("distinct_keys_in_full_sample", 0)) >= 2
        for row in prompt_results
    )
    expected_orientation_counts = {
        "train": {"0": 2, "1": 2, "2": 2, "3": 2},
        "dev": {"0": 1, "1": 1, "2": 1, "3": 1},
        "eval": {"0": 1, "1": 1, "2": 1, "3": 1},
    }
    checks = {
        "exact_256_attempts": len(attempts) == 256,
        "four_prompts_sixty_four_each": (
            len(prompt_results) == 4
            and all(
                int(row.get("verified_in_full_sample", 0))
                + sum(
                    1
                    for attempt in attempts
                    if int(attempt.get("row_index", -1))
                    == int(row.get("row_index", -2))
                    and not bool(attempt.get("verified"))
                )
                == 64
                for row in prompt_results
            )
        ),
        "verified_rate_in_0p02_0p50": 0.02 <= verified_rate <= 0.50,
        "two_k16_prefix_success_prompts": prefix_success_prompts >= 2,
        "two_multimode_prompts": multimode_prompts >= 2,
        "viability_sampling_contract": (
            viability.get("sampling", {}).get("seed") == 76530
            and viability.get("sampling", {}).get("sample_count") == 64
            and viability.get("sampling", {}).get("prefix_count") == 16
        ),
        "development_only": (
            viability.get("information_boundary", {}).get(
                "development_only"
            )
            is True
            and viability.get("data_split_root", "").endswith("/dev")
        ),
        "viability_identity": (
            identity.get("schema_version")
            == "point-maze-algorithm-repair-v3-viability-identity-v1"
            and identity.get("seed") == 76530
            and identity.get("sample_count_per_prompt") == 64
            and identity.get("prefix_count") == 16
            and identity.get("final_seed") is False
        ),
        "admission_pass": (
            admission.get("status") == "pass"
            and admission.get("decision")
            == "admitted_to_point_maze_v3_balanced_viability_gate"
        ),
        "balanced_data_identity": (
            data.get("schema_version")
            == "point-maze-algorithm-repair-data-v3"
            and data.get("orientation_counts")
            == expected_orientation_counts
            and data.get("executable_task_overlap_count") == 0
        ),
        "hash_chain": (
            viability.get("admission_audit_sha256") == sha(args.admission)
            and identity.get("admission_audit_sha256")
            == sha(args.admission)
            and identity.get("data_identity_sha256")
            == sha(args.data_identity)
            and identity.get("protocol_sha256") == sha(args.protocol)
        ),
    }
    errors = [
        f"failed check: {key}"
        for key, passed in checks.items()
        if not passed
    ]
    status = "pass" if not errors else "fail"
    payload = {
        "schema_version": (
            "point-maze-algorithm-repair-v3-qualification-v1"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": (
            "eligible_for_point_maze_algorithm_repair_v3_pair"
            if status == "pass"
            else "point_maze_algorithm_repair_v3_stopped"
        ),
        "checks": checks,
        "errors": errors,
        "summary": {
            "verified_completions": verified,
            "attempt_count": len(attempts),
            "verified_rate": verified_rate,
            "prefix_success_prompts": prefix_success_prompts,
            "multimode_prompts": multimode_prompts,
        },
        "viability_sha256": sha(args.viability),
        "viability_identity_sha256": sha(args.viability_identity),
        "admission_audit_sha256": sha(args.admission),
        "data_identity_sha256": sha(args.data_identity),
        "protocol_sha256": sha(args.protocol),
        "evaluation_rows_loaded": False,
        "final_seed": False,
    }
    atomic(args.output, payload)
    print(json.dumps({"status": status, "summary": payload["summary"]}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
