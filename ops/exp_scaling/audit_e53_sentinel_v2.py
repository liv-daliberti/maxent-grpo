#!/usr/bin/env python3
"""Audit E53 with the frozen accumulation-width telemetry interpretation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "ops/exp_scaling/audit_e53_sentinel.py"
REPAIR_PATH = (
    ROOT
    / "paper/preregistration/e53_runtime_audit_scaling_repair_20260726.md"
)
SPEC = importlib.util.spec_from_file_location("audit_e53_sentinel_v1_bound", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import bound E53 auditor from {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

ARMS = BASE.ARMS
REPLAY_ARM = BASE.REPLAY_ARM
DOMAINS = BASE.DOMAINS
SEED = BASE.SEED
MAX_PASSES = BASE.MAX_PASSES
NUM_SAMPLES = BASE.NUM_SAMPLES
DISTINCT_KEY = BASE.DISTINCT_KEY
PASS8_KEY = BASE.PASS8_KEY
MEAN8_KEY = BASE.MEAN8_KEY
IDENTITY_PATH = BASE.IDENTITY_PATH
APPROVAL_PATH = BASE.APPROVAL_PATH
PROTOCOL_PATH = BASE.PROTOCOL_PATH


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_base_check_replay_controller = BASE._check_replay_controller


def _check_replay_controller(
    record: dict[str, Any],
    *,
    step: int,
    reference: float | None,
):
    """Validate logged A directly, then reuse every other bound-v1 check."""

    available_groups = BASE._finite(
        record, "train/canonical_replay_available_groups"
    )
    actual_width = BASE._finite(
        record, "train/canonical_replay_backward_scale"
    )
    local_violations: list[str] = []
    repaired = record
    if available_groups is not None and available_groups > 0:
        if actual_width != float(NUM_SAMPLES):
            local_violations.append(
                f"step {step}: replay accumulation width "
                f"{actual_width!r}, expected {NUM_SAMPLES}"
            )
        alpha = BASE._finite(record, "train/canonical_replay_alpha_used")
        estimator = BASE._finite(
            record, "train/canonical_replay_reward_estimator_scale"
        )
        if alpha is not None and estimator is not None:
            repaired = dict(record)
            # Bound v1 validates the complete mathematical scalar. Supply that
            # derived quantity only to its internal check; the raw logged
            # accumulator width was independently checked immediately above.
            repaired["train/canonical_replay_backward_scale"] = (
                alpha * estimator * NUM_SAMPLES
            )
    result = _base_check_replay_controller(
        repaired,
        step=step,
        reference=reference,
    )
    returned_reference, violations, latest, active = result
    return (
        returned_reference,
        local_violations + violations,
        latest,
        active,
    )


BASE._check_replay_controller = _check_replay_controller


def _load_records(*args, **kwargs):
    return BASE._load_records(*args, **kwargs)


def audit_run(*args, **kwargs):
    return BASE.audit_run(*args, **kwargs)


def checkpoint_gate(*args, **kwargs):
    return BASE.checkpoint_gate(*args, **kwargs)


def behavioral_gate(*args, **kwargs):
    return BASE.behavioral_gate(*args, **kwargs)


def safety_gate(*args, **kwargs):
    return BASE.safety_gate(*args, **kwargs)


def audit(data_root: Path) -> dict[str, Any]:
    payload = BASE.audit(data_root)
    payload["schema"] = "e53_sentinel_audit_v2"
    payload["approval_binding"].update(
        {
            "bound_base_auditor_sha256": _sha256_file(BASE_PATH),
            "runtime_repair_protocol_sha256": _sha256_file(REPAIR_PATH),
            "runtime_repair_auditor_sha256": _sha256_file(
                Path(__file__).resolve()
            ),
        }
    )
    return payload


write_audit_outputs = BASE.write_audit_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "var/data")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "var/artifacts/e53_sentinel_audit_latest.json",
    )
    parser.add_argument("--approval-out", type=Path, default=APPROVAL_PATH)
    args = parser.parse_args()
    payload = audit(args.data_root)
    write_audit_outputs(
        payload=payload,
        audit_out=args.out,
        approval_out=args.approval_out,
    )
    print(
        f"[e53-audit-v2] status={payload['status']} "
        f"violations={len(payload['violations'])} out={args.out}"
    )
    for domain, domain_payload in payload["domains"].items():
        statuses = ", ".join(
            f"{arm}={run['status']}@{run['training_passes']:.2f}"
            for arm, run in domain_payload["runs"].items()
        )
        print(
            f"[e53-audit-v2] {domain}: {statuses}; "
            f"behavior={domain_payload['behavioral_gate']['status']}"
        )
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
