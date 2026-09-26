#!/usr/bin/env python3
"""Extend the v1 Pantry smoke audit with the prospective r1 repair chain."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import audit_pantry_support_mask_drgrpo_smoke_v1 as base


PREFIX = "ppsmoke_support_mask_drgrpo_v1_r1"
FAILED_JOB_ID = 30187473


def _argument(name: str) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"missing {name}") from error


def main() -> None:
    base.PREFIX = PREFIX
    base.main()

    output = _argument("--output")
    identity_path = _argument("--identity")
    root = _argument("--repo-root").resolve()
    payload = base._load(output)
    identity = base._load(identity_path)
    repair_protocol = root / (
        "paper/preregistration/"
        "pantry_support_mask_drgrpo_smoke_v1_r1_horizon_repair_20260730.md"
    )
    failed_audit = root / "var/artifacts/pantry_support_mask_drgrpo_smoke_v1_audit.json"
    failed_stdout = root / f"var/artifacts/logs/xdr_train-{FAILED_JOB_ID}.out"
    placement = root / (
        "var/artifacts/pantry_support_mask_drgrpo_smoke_v1_placement_amendment.json"
    )
    repair_checks = {
        "repair_identity": (
            identity.get("repair_attempt") == "horizon_generic_r1"
            and identity.get("failed_job_id") == FAILED_JOB_ID
            and identity.get("placement_only") is True
            and identity.get("scientific_change") is False
        ),
        "repair_protocol_matches": (
            repair_protocol.is_file()
            and identity.get("repair_protocol_sha256") == base._sha(repair_protocol)
        ),
        "failed_audit_matches": (
            failed_audit.is_file()
            and identity.get("failed_audit_sha256") == base._sha(failed_audit)
            and base._load(failed_audit).get("status") == "fail"
            and base._load(failed_audit).get("job_id") == FAILED_JOB_ID
        ),
        "failed_stdout_matches": (
            failed_stdout.is_file()
            and identity.get("failed_stdout_sha256") == base._sha(failed_stdout)
            and "canonical learner sampler requires three supports"
            in failed_stdout.read_text(errors="replace")
        ),
        "placement_antecedent_matches": (
            placement.is_file()
            and identity.get("placement_amendment_sha256") == base._sha(placement)
            and base._load(placement).get("placement_only") is True
            and base._load(placement).get("scientific_change") is False
        ),
        "six_step_regressions_bound": identity.get("required_regressions")
        == [
            "pantry_learner_sampler_six_steps_and_task_identity",
            "pantry_exact_entropy_64_leaves_63_prefixes",
        ],
    }
    payload["repair_attempt"] = "horizon_generic_r1"
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
