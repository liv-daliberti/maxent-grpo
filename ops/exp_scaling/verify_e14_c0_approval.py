#!/usr/bin/env python3
"""Revalidate E14's C0 approval before authorizing M01 or M05.

The approval JSON is an auditable record, not a signature.  This verifier
therefore hashes every recorded evidence file and replays the complete C0
checker instead of trusting ``approved=true`` or its summary booleans.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # package import in tests
    from .check_e14_c0 import GATE_NAME, check_c0
    from .e14_archival import (
        ArchiveReceiptError,
        replay_archive_authorization_if_needed,
    )
    from .check_e14_preflight import (
        EXPECTED_DATASET_IDENTITY,
        EXPECTED_RUNTIME_IDENTITY,
        GateError,
        _sha256_file,
        _strict_json,
    )
except ImportError:  # direct script execution
    from check_e14_c0 import GATE_NAME, check_c0
    from e14_archival import (
        ArchiveReceiptError,
        replay_archive_authorization_if_needed,
    )
    from check_e14_preflight import (
        EXPECTED_DATASET_IDENTITY,
        EXPECTED_RUNTIME_IDENTITY,
        GateError,
        _sha256_file,
        _strict_json,
    )


REQUIRED_C0_CHECKS = {
    "frozen_identity",
    "immutable_source",
    "terminal_success",
    "preflight_authorization_revalidated",
    "updates_97_128_contiguous_finite",
    "canonical_rollouts_valid",
    "behavior_policy_overlap",
    "final_32_mean_reward_positive",
    "forbidden_treatments_inactive",
    "exact_step_00128_audit",
    "endpoint_mean_p_valid_above_0p05",
}
REQUIRED_EVIDENCE = {
    "identity",
    "preflight_approval",
    "metrics",
    "source_snapshot_marker",
    "stdout",
    "endpoint_audit",
    "checkpoint_config",
}


def _without_approval_time(payload: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(payload)
    comparable.pop("approved_at_utc", None)
    return comparable


def verify_c0_approval_for_source(
    approval_path: Path,
    *,
    expected_source_hash: str,
    logical_repo_root: Path,
) -> dict[str, Any]:
    """Replay a C0 approval and bind it to the prospective treatment source."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_source_hash) is None:
        raise GateError("expected treatment source hash is not a lowercase SHA-256 digest")
    if not approval_path.is_file():
        raise GateError(f"E14 C0 approval is missing: {approval_path}")
    payload = _strict_json(
        approval_path.read_text(encoding="utf-8"), context=str(approval_path)
    )
    if not isinstance(payload, dict):
        raise GateError("E14 C0 approval is not a JSON object")
    if payload.get("approved") is not True:
        raise GateError("E14 C0 approval does not have approved=true")
    if payload.get("gate") != GATE_NAME:
        raise GateError("E14 C0 approval has the wrong gate identifier")
    if payload.get("protocol") != "E14" or payload.get("arm") != "C0":
        raise GateError("E14 C0 approval has the wrong protocol or arm")

    job_id = str(payload.get("job_id", ""))
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise GateError("E14 C0 approval has an invalid Slurm job ID")
    slurm = payload.get("slurm")
    if not isinstance(slurm, dict) or slurm != {
        "state": "COMPLETED",
        "exit_code": "0:0",
    }:
        raise GateError("E14 C0 approval lacks clean Slurm terminal success")
    try:
        approved_time = datetime.fromisoformat(str(payload.get("approved_at_utc")))
    except ValueError as error:
        raise GateError("E14 C0 approval has an invalid UTC timestamp") from error
    if approved_time.tzinfo is None or approved_time.utcoffset() is None:
        raise GateError("E14 C0 approval timestamp is not timezone-aware")

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise GateError("E14 C0 approval has no checks object")
    missing_checks = sorted(REQUIRED_C0_CHECKS - set(checks))
    failed_checks = sorted(
        check for check in REQUIRED_C0_CHECKS if checks.get(check) is not True
    )
    if missing_checks or failed_checks:
        raise GateError(
            "E14 C0 approval checks are incomplete; "
            f"missing={missing_checks} failed={failed_checks}"
        )

    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise GateError("E14 C0 approval has no identity object")
    if identity.get("source_hash") != expected_source_hash:
        raise GateError(
            "treatment Python source differs from the source that passed E14 C0"
        )
    if identity.get("dataset") != EXPECTED_DATASET_IDENTITY:
        raise GateError("E14 C0 approval embeds the wrong dataset identity")
    if identity.get("runtime") != EXPECTED_RUNTIME_IDENTITY:
        raise GateError("E14 C0 approval embeds the wrong runtime identity")
    stamp = identity.get("stamp")
    if not isinstance(stamp, str) or not stamp:
        raise GateError("E14 C0 approval has no stamp")

    run_dir_raw = payload.get("run_dir")
    if not isinstance(run_dir_raw, str) or not Path(run_dir_raw).is_absolute():
        raise GateError("E14 C0 approval run directory is not absolute")
    run_dir = Path(run_dir_raw)

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise GateError("E14 C0 approval has no evidence object")
    missing_evidence = sorted(REQUIRED_EVIDENCE - set(evidence))
    if missing_evidence:
        raise GateError(f"E14 C0 approval lacks evidence {missing_evidence}")
    evidence_paths: dict[str, Path] = {}
    for label, record in evidence.items():
        if not isinstance(label, str) or not isinstance(record, dict):
            raise GateError("E14 C0 evidence records are malformed")
        if set(record) != {"path", "sha256"}:
            raise GateError(f"E14 C0 evidence {label!r} has malformed fields")
        raw_path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise GateError(f"E14 C0 evidence {label!r} path is not absolute")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise GateError(f"E14 C0 evidence {label!r} has an invalid hash")
        path = Path(raw_path)
        if not path.is_file():
            raise GateError(f"E14 C0 evidence {label!r} is missing")
        if _sha256_file(path) != digest:
            raise GateError(f"E14 C0 evidence {label!r} changed after approval")
        evidence_paths[label] = path

    source_marker = evidence_paths["source_snapshot_marker"].resolve()
    if source_marker.name != "__init__.py" or source_marker.parent.name != "oat_drgrpo":
        raise GateError("E14 C0 source snapshot marker has the wrong location")
    source_root = source_marker.parents[1]

    try:
        archive_authorization = replay_archive_authorization_if_needed(approval_path)
    except ArchiveReceiptError as error:
        raise GateError(f"E14 C0 archival replay rejected: {error}") from error

    replayed = check_c0(
        run_dir=run_dir,
        identity_path=evidence_paths["identity"],
        preflight_approval_path=evidence_paths["preflight_approval"],
        endpoint_audit_path=evidence_paths["endpoint_audit"],
        source_root=source_root,
        stdout_path=evidence_paths["stdout"],
        stderr_path=evidence_paths.get("stderr"),
        slurm_state="COMPLETED",
        slurm_exit_code="0:0",
        job_id=job_id,
        logical_repo_root=logical_repo_root,
        archived_removed_steps=(
            archive_authorization["removed_steps"]
            if archive_authorization is not None
            else None
        ),
    )
    if _without_approval_time(replayed) != _without_approval_time(payload):
        raise GateError("E14 C0 approval does not match replayed C0 evidence")

    endpoint = replayed["endpoint_summary"]
    return {
        "approval": str(approval_path.resolve()),
        "approval_sha256": _sha256_file(approval_path),
        "c0_job_id": job_id,
        "c0_stamp": stamp,
        "source_hash": expected_source_hash,
        "endpoint_p_valid_mean": endpoint["p_valid_mean"],
        "endpoint_action_entropy_mean": endpoint["exact_action_entropy_mean"],
        "endpoint_n_eff_valid_mean": endpoint["n_eff_valid_mean"],
        "evidence_files_verified": len(evidence_paths),
        "full_c0_gate_replayed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Revalidate the E14 C0 gate before launching M01 or M05."
    )
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--expected-source-hash", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = verify_c0_approval_for_source(
            args.approval,
            expected_source_hash=args.expected_source_hash,
            logical_repo_root=args.repo_root,
        )
    except GateError as error:
        raise SystemExit(f"E14 C0 approval rejected: {error}") from error
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
