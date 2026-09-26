#!/usr/bin/env python3
"""Verify the immutable E15 outcome used to choose E16's fixed coefficient."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_E15_OUTCOME_SHA256 = (
    "680e739411a5ed9f14754a86c285945e8b20d11a4362c48235de26d9ba9d7de1"
)
EXPECTED_M10 = {
    "exact_action_entropy_mean": 2.7974740052946436,
    "n_eff_valid_mean": 4.437708099723962,
    "p_valid_mean": 0.261672194741692,
    "p_valid_retention_vs_c0": 0.833154545824017,
}


class AntecedentError(ValueError):
    """Raised when E15 evidence is missing, mutable, or scientifically different."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _close(observed: Any, expected: float, label: str) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise AntecedentError(f"E15 {label} is not numeric") from error
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AntecedentError(
            f"E15 {label} drifted: expected={expected!r} observed={value!r}"
        )


def verify_e16_antecedent(path: Path) -> dict[str, Any]:
    digest = _sha256_file(path)
    if digest != EXPECTED_E15_OUTCOME_SHA256:
        raise AntecedentError(
            f"E15 outcome hash drifted: expected={EXPECTED_E15_OUTCOME_SHA256} "
            f"observed={digest}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AntecedentError("E15 outcome is not JSON") from error
    if not isinstance(payload, dict):
        raise AntecedentError("E15 outcome is not an object")
    expected_identity = {
        "schema": "e15_canonical_treatment_comparison_v1",
        "protocol": "E15",
        "status": "selected",
        "selected_arm": "M10",
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise AntecedentError(f"E15 {key} does not equal {expected!r}")
    if payload.get("single_seed_engineering_calibration") is not True:
        raise AntecedentError("E15 lost its single-seed calibration scope")
    if payload.get("does_not_authorize_scale_or_domain_expansion") is not True:
        raise AntecedentError("E15 lost its scope firewall")
    arms = payload.get("arms")
    if not isinstance(arms, list):
        raise AntecedentError("E15 arms are malformed")
    by_name = {
        str(arm.get("arm")): arm for arm in arms if isinstance(arm, dict)
    }
    if set(by_name) != {"M075", "M10"}:
        raise AntecedentError("E15 must contain exactly M075 and M10")
    m10 = by_name["M10"]
    if (
        m10.get("runtime_valid") is not True
        or m10.get("behaviorally_safe") is not True
        or m10.get("diversity_effective") is not True
        or m10.get("viable") is not True
        or m10.get("failures") != []
    ):
        raise AntecedentError("E15 M10 is no longer the selected viable arm")
    for label, expected in EXPECTED_M10.items():
        _close(m10.get(label), expected, f"M10 {label}")

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "c0_approval",
        "e14_outcome",
        "m075_result",
        "m10_result",
    }:
        raise AntecedentError("E15 evidence manifest is incomplete")
    evidence_summary: dict[str, dict[str, str]] = {}
    for label, record in evidence.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise AntecedentError(f"E15 evidence {label} is malformed")
        evidence_path = Path(str(record["path"]))
        if not evidence_path.is_absolute() or not evidence_path.is_file():
            raise AntecedentError(f"E15 evidence {label} is missing")
        evidence_hash = _sha256_file(evidence_path)
        if evidence_hash != record["sha256"]:
            raise AntecedentError(f"E15 evidence {label} changed")
        evidence_summary[label] = {
            "path": str(evidence_path.resolve()),
            "sha256": evidence_hash,
        }
    return {
        "e15_evidence": evidence_summary,
        "e15_outcome": str(path.resolve()),
        "e15_outcome_sha256": digest,
        "fixed_alpha": 0.10,
        "selected_arm": "M10",
        "selected_exact_action_entropy": EXPECTED_M10[
            "exact_action_entropy_mean"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e15-outcome", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = verify_e16_antecedent(args.e15_outcome)
    except (AntecedentError, FileNotFoundError) as error:
        raise SystemExit(f"E16 E15 antecedent rejected: {error}") from error
    print(json.dumps(summary, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
