#!/usr/bin/env python3
"""Qualify the frozen orientation-balanced PointMaze v5 warm start."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import os


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-identity", type=Path, required=True)
    parser.add_argument("--warmstart-identity", type=Path, required=True)
    parser.add_argument("--sft-receipt", type=Path, required=True)
    parser.add_argument("--viability", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh v5 qualification required: {args.output}")

    source = json.loads(args.source_identity.read_text(encoding="utf-8"))
    warmstart = json.loads(args.warmstart_identity.read_text(encoding="utf-8"))
    sft = json.loads(args.sft_receipt.read_text(encoding="utf-8"))
    viability = json.loads(args.viability.read_text(encoding="utf-8"))
    admission = json.loads(args.admission.read_text(encoding="utf-8"))
    summary = viability.get("summary", {})
    attempts = viability.get("attempts", [])
    prompt_results = viability.get("prompt_results", [])
    verified = int(summary.get("verified_completions", -1))
    rate = verified / 256.0
    checks = {
        "balanced_source": (
            source.get("status") == "pass"
            and source.get("orientation_counts")
            == {"0": 2, "1": 2, "2": 2, "3": 2}
            and source.get("expected_replayed_example_count") == 644
            and source.get("development_rows_loaded") is False
            and source.get("evaluation_rows_loaded") is False
        ),
        "warmstart_data": (
            warmstart.get("status") == "pass"
            and warmstart.get("example_count") == 644
            and warmstart.get("episode_count") == 16
            and warmstart.get("policy_interface") == "velocity_state_v3"
            and warmstart.get("information_boundary", {}).get("dev_dataset_loaded")
            is False
            and warmstart.get("information_boundary", {}).get("eval_dataset_loaded")
            is False
        ),
        "sft_pass": (
            sft.get("status") == "pass"
            and sft.get("seed") == 76601
            and sft.get("epochs") == 12
            and sft.get("optimizer_steps") == 276
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
            viability.get("sampling", {}).get("seed") == 76602
            and viability.get("sampling", {}).get("sample_count") == 64
            and viability.get("sampling", {}).get("prefix_count") == 16
        ),
        "exact_256_attempts": len(attempts) == 256,
        "four_prompts_sixty_four_each": (
            len(prompt_results) == 4
            and all(
                int(row.get("verified_in_full_sample", -1)) >= 0
                for row in prompt_results
            )
        ),
        "two_prefix_success_prompts": int(
            summary.get("prefix_success_prompts", -1)
        )
        >= 2,
        "two_multimode_prompts": int(summary.get("multimode_prompts", -1)) >= 2,
        "verified_rate_in_0p02_0p50": 0.02 <= rate <= 0.50,
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "point-maze-balanced-warmstart-v5-qualification-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            "eligible_for_point_maze_v5_five_seed_pair"
            if passed
            else "point_maze_balanced_warmstart_v5_stopped"
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
            "v3_and_v4_development_outcomes_loaded_for_diagnosis": True,
            "evaluation_rows_loaded": False,
            "final_seed": False,
            "threshold_changed": False,
        },
        "source_identity_sha256": sha(args.source_identity),
        "warmstart_identity_sha256": sha(args.warmstart_identity),
        "sft_receipt_sha256": sha(args.sft_receipt),
        "viability_sha256": sha(args.viability),
        "admission_sha256": sha(args.admission),
        "protocol_sha256": sha(args.protocol),
    }
    atomic_json(args.output, payload)
    if not passed:
        raise SystemExit("PointMaze balanced warm-start v5 qualification failed")


if __name__ == "__main__":
    main()
