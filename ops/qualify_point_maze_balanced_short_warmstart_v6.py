#!/usr/bin/env python3
"""Qualify the frozen short balanced PointMaze v6 warm start."""

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
        prefix=f".{path.name}.", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-identity", type=Path, required=True)
    parser.add_argument("--warmstart-identity", type=Path, required=True)
    parser.add_argument("--sft-receipt", type=Path, required=True)
    parser.add_argument("--viability", type=Path, required=True)
    parser.add_argument("--viability-identity", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--v5-qualification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh PointMaze v6 qualification required: {args.output}")

    source = json.loads(args.source_identity.read_text(encoding="utf-8"))
    warmstart = json.loads(args.warmstart_identity.read_text(encoding="utf-8"))
    sft = json.loads(args.sft_receipt.read_text(encoding="utf-8"))
    viability = json.loads(args.viability.read_text(encoding="utf-8"))
    identity = json.loads(args.viability_identity.read_text(encoding="utf-8"))
    admission = json.loads(args.admission.read_text(encoding="utf-8"))
    v5 = json.loads(args.v5_qualification.read_text(encoding="utf-8"))
    summary = viability.get("summary", {})
    attempts = viability.get("attempts", [])
    prompts = viability.get("prompt_results", [])
    verified = int(summary.get("verified_completions", -1))
    rate = verified / 256.0
    checks = {
        "v5_terminal_oversaturation": (
            v5.get("status") == "fail"
            and v5.get("decision") == "point_maze_balanced_warmstart_v5_stopped"
            and v5.get("summary", {}).get("verified_completions") == 243
            and v5.get("checks", {}).get("verified_rate_in_0p02_0p50") is False
        ),
        "balanced_source": (
            source.get("status") == "pass"
            and source.get("orientation_counts")
            == {"0": 2, "1": 2, "2": 2, "3": 2}
        ),
        "warmstart_data": (
            warmstart.get("status") == "pass"
            and warmstart.get("example_count") == 644
            and warmstart.get("episode_count") == 16
            and warmstart.get("policy_interface") == "velocity_state_v3"
        ),
        "short_sft_pass": (
            sft.get("status") == "pass"
            and sft.get("seed") == 76621
            and sft.get("epochs") == 3
            and sft.get("optimizer_steps") == 69
            and sft.get("warmup_steps") == 7
            and sft.get("frozen_short_balanced_v6") is True
        ),
        "identity_contract": (
            identity.get("schema_version")
            == "point-maze-balanced-short-warmstart-v6-identity-v1"
            and identity.get("sft_seed") == 76621
            and identity.get("development_seed") == 76622
            and identity.get("evaluation_rows_loaded") is False
        ),
        "admission_pass": admission.get("status") == "pass",
        "viability_pass": viability.get("status") == "pass",
        "development_only": (
            viability.get("information_boundary", {}).get("development_only")
            is True
            and viability.get("information_boundary", {}).get(
                "evaluation_prompts_loaded"
            )
            is False
            and str(viability.get("data_split_root", "")).endswith("/dev")
        ),
        "sampling_contract": (
            viability.get("sampling", {}).get("seed") == 76622
            and viability.get("sampling", {}).get("sample_count") == 64
            and viability.get("sampling", {}).get("prefix_count") == 16
        ),
        "exact_256_attempts": len(attempts) == 256,
        "four_prompts": len(prompts) == 4,
        "two_prefix_success_prompts": int(
            summary.get("prefix_success_prompts", -1)
        )
        >= 2,
        "two_multimode_prompts": int(summary.get("multimode_prompts", -1)) >= 2,
        "verified_rate_in_0p02_0p50": 0.02 <= rate <= 0.50,
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "point-maze-balanced-short-warmstart-v6-qualification-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            "eligible_for_point_maze_v6_five_seed_pair"
            if passed
            else "point_maze_balanced_short_warmstart_v6_stopped"
        ),
        "checks": checks,
        "errors": [f"failed check: {key}" for key, value in checks.items() if not value],
        "summary": {
            "verified_completions": verified,
            "verified_rate": rate,
            "prefix_success_prompts": summary.get("prefix_success_prompts"),
            "multimode_prompts": summary.get("multimode_prompts"),
        },
        "information_boundary": {
            "v5_development_outcome_loaded_for_duration_repair": True,
            "evaluation_rows_loaded": False,
            "final_seed": False,
            "threshold_changed": False,
        },
        "source_identity_sha256": sha(args.source_identity),
        "warmstart_identity_sha256": sha(args.warmstart_identity),
        "sft_receipt_sha256": sha(args.sft_receipt),
        "viability_sha256": sha(args.viability),
        "viability_identity_sha256": sha(args.viability_identity),
        "admission_sha256": sha(args.admission),
        "v5_qualification_sha256": sha(args.v5_qualification),
        "protocol_sha256": sha(args.protocol),
    }
    atomic(args.output, payload)
    if not passed:
        raise SystemExit("PointMaze short balanced warm-start v6 qualification failed")


if __name__ == "__main__":
    main()
