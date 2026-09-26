#!/usr/bin/env python3
"""Extend the Pantry smoke audit with the frozen r2 repair chain."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import audit_pantry_support_mask_drgrpo_smoke_v1 as base


PREFIX = "ppsmoke_support_mask_drgrpo_v1_r2"
FAILED_JOB_ID = 30199460


def _argument(name: str) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"missing {name}") from error


def main() -> None:
    base.PREFIX = PREFIX
    base.SUBMISSION_SCHEMAS = {
        "pantry-support-mask-drgrpo-smoke-submission-r2"
    }
    base.main()

    output = _argument("--output")
    identity_path = _argument("--identity")
    root = _argument("--repo-root").resolve()
    payload = base._load(output)
    identity = base._load(identity_path)
    repair_protocol = root / (
        "paper/preregistration/"
        "pantry_support_mask_drgrpo_smoke_v1_r2_query_budget_repair_20260730.md"
    )
    failed_audit = root / (
        "var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r1_audit.json"
    )
    failed_stdout = root / f"var/artifacts/logs/xdr_train-{FAILED_JOB_ID}.out"
    repair_checks = {
        "repair_identity": (
            identity.get("repair_attempt") == "query_budget_and_audit_contract_r2"
            and identity.get("failed_job_id") == FAILED_JOB_ID
            and identity.get("max_queries") == 496
            and identity.get("train_rows") == 32
            and identity.get("optimizer_updates") == 32
            and identity.get("placement_only") is True
            and identity.get("scientific_change") is False
        ),
        "repair_protocol_matches": (
            repair_protocol.is_file()
            and identity.get("repair_protocol_sha256") == base._sha(repair_protocol)
        ),
        "failed_r1_audit_matches": (
            failed_audit.is_file()
            and identity.get("failed_audit_sha256") == base._sha(failed_audit)
            and base._load(failed_audit).get("status") == "fail"
            and base._load(failed_audit).get("job_id") == FAILED_JOB_ID
            and base._load(failed_audit).get("checks", {}).get(
                "learning_rounds_exact"
            )
            is False
        ),
        "failed_r1_stdout_matches": (
            failed_stdout.is_file()
            and identity.get("failed_stdout_sha256") == base._sha(failed_stdout)
            and "[train] target_optimizer_updates=32"
            in failed_stdout.read_text(errors="replace")
        ),
        "r2_regressions_bound": identity.get("required_regressions")
        == [
            "pantry_learner_sampler_six_steps_and_task_identity",
            "pantry_exact_entropy_64_leaves_63_prefixes",
            "pantry_query_budget_496_for_32_updates",
            "pantry_emitted_nonfinite_telemetry_contract",
        ],
    }
    payload["repair_attempt"] = "query_budget_and_audit_contract_r2"
    payload["failed_job_id"] = FAILED_JOB_ID
    payload["repair_checks"] = repair_checks
    payload.setdefault("checks", {}).update(repair_checks)
    failed = [name for name, passed in repair_checks.items() if not passed]
    if failed:
        payload["status"] = "fail"
        payload["decision"] = "pantry_training_plumbing_ineligible"
        payload.setdefault("errors", []).extend(
            f"failed check: {name}" for name in failed
        )
    base._atomic(output, payload)
    print(json.dumps({"status": payload["status"], "repair_checks": repair_checks}))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
