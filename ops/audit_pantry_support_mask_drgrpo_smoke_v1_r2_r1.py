#!/usr/bin/env python3
"""Audit-only repair for the Pantry r2 submission schema mismatch."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import audit_pantry_support_mask_drgrpo_smoke_v1 as base
import audit_pantry_support_mask_drgrpo_smoke_v1_r2 as r2


def _argument(name: str) -> Path:
    return Path(sys.argv[sys.argv.index(name) + 1])


def main() -> None:
    r2.main()
    output = _argument("--output")
    root = _argument("--repo-root").resolve()
    failed_path = root / (
        "var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r2_audit.json"
    )
    protocol = root / (
        "paper/preregistration/"
        "pantry_support_mask_drgrpo_smoke_v1_r2_r1_audit_schema_repair_20260730.md"
    )
    payload = base._load(output)
    failed = base._load(failed_path)
    failed_checks = failed.get("checks", {})
    expected_other_checks = {
        key: value for key, value in failed_checks.items() if key != "submission_contract"
    }
    repair_checks = {
        "failed_r2_audit_bound": (
            failed.get("status") == "fail"
            and failed.get("job_id") == 30199933
            and failed.get("errors") == ["failed check: submission_contract"]
            and failed_checks.get("submission_contract") is False
            and expected_other_checks
            and all(value is True for value in expected_other_checks.values())
        ),
        "audit_repair_protocol_bound": protocol.is_file(),
        "submission_schema_exact_r2": base._load(
            root / "var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r2_submission.json"
        ).get("schema")
        == "pantry-support-mask-drgrpo-smoke-submission-r2",
    }
    payload["audit_repair"] = "submission_schema_exact_r2_r1"
    payload["failed_r2_audit_sha256"] = base._sha(failed_path)
    payload["audit_repair_protocol_sha256"] = base._sha(protocol)
    payload["audit_repair_checks"] = repair_checks
    payload.setdefault("checks", {}).update(repair_checks)
    failed_repairs = [name for name, passed in repair_checks.items() if not passed]
    if failed_repairs:
        payload["status"] = "fail"
        payload["decision"] = "pantry_training_plumbing_ineligible"
        payload.setdefault("errors", []).extend(
            f"failed check: {name}" for name in failed_repairs
        )
    base._atomic(output, payload)
    print(json.dumps({"status": payload["status"], "audit_repair_checks": repair_checks}))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
