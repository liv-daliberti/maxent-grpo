#!/usr/bin/env python3
"""Qualify the frozen AntMaze v15/controller-v18 model viability result."""

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
    parser.add_argument("--viability", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh Ant v18 qualification required: {args.output}")

    viability = json.loads(args.viability.read_text(encoding="utf-8"))
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    admission = json.loads(args.admission.read_text(encoding="utf-8"))
    summary = viability.get("summary", {})
    prompts = viability.get("prompt_results", [])
    attempts = viability.get("attempts", [])
    verified = int(summary.get("verified_completions", -1))
    rate = verified / 256.0
    checks = {
        "admission_pass": (
            admission.get("status") == "pass"
            and admission.get("decision")
            == "admitted_to_ant_v15_v18_frozen_model_viability_gate"
        ),
        "identity_contract": (
            identity.get("schema_version")
            == "ant-maze-v15-controller-v18-viability-identity-v1"
            and identity.get("seed") == 76701
            and identity.get("evaluation_rows_loaded") is False
            and identity.get("final_seed") is False
        ),
        "viability_pass": (
            viability.get("status") == "pass"
            and viability.get("schema_version")
            == "ant-maze-interactive-viability-v18"
        ),
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
            viability.get("sampling", {}).get("seed") == 76701
            and viability.get("sampling", {}).get("sample_count") == 64
            and viability.get("sampling", {}).get("prefix_count") == 16
            and viability.get("sampling", {}).get("action_repeat") == 400
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
        "schema_version": "ant-maze-v15-controller-v18-qualification-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            "eligible_for_ant_v15_v18_five_seed_pair"
            if passed
            else "ant_maze_v15_v18_stopped"
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
            "evaluation_rows_loaded": False,
            "final_seed": False,
            "threshold_changed": False,
        },
        "viability_sha256": sha(args.viability),
        "identity_sha256": sha(args.identity),
        "admission_sha256": sha(args.admission),
        "protocol_sha256": sha(args.protocol),
    }
    atomic(args.output, payload)
    if not passed:
        raise SystemExit("AntMaze v15/v18 viability qualification failed")


if __name__ == "__main__":
    main()
