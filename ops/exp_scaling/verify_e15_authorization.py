#!/usr/bin/env python3
"""Fail-closed authorization replay for the prospective E15 dose calibration.

E15 is a user-authorized follow-up to E14's negative fixed-dose gate.  The
launcher must not trust a hand-edited status field: it replays both the exact
E14 C0 approval and the final M01/M05 comparison, including their archived
training evidence, before allowing either E15 arm to be configured or run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:  # package import in tests
    from .check_e14_preflight import GateError, _sha256_file, _strict_json
    from .check_e14_treatments import (
        COMPARISON_SCHEMA,
        compare_result_payloads,
    )
    from .verify_e14_c0_approval import verify_c0_approval_for_source
except ImportError:  # direct script execution
    from check_e14_preflight import GateError, _sha256_file, _strict_json
    from check_e14_treatments import COMPARISON_SCHEMA, compare_result_payloads
    from verify_e14_c0_approval import verify_c0_approval_for_source


EXPECTED_SOURCE_HASH = (
    "8f0ee26fe1a4482efcf55a96e3f3de0a689aa2f94813f39df66887f9d6bf6329"
)
EXPECTED_C0_APPROVAL_SHA256 = (
    "4ab8b0173034931f287b489d042f9aad735fc7986e789de7e31c9363d1972d97"
)
EXPECTED_E14_OUTCOME_SHA256 = (
    "50456aee52b0ab9ad3fe32deaf8da33caabe7d3a99c90eafb44cf493faa794de"
)
EXPECTED_THRESHOLDS = {
    "exact_entropy_gain_min_nats": math.log(1.25),
    "n_eff_valid_ratio_min": 1.25,
    "p_valid_absolute_strict_min": 0.05,
    "p_valid_retention_min": 0.8,
    "tie_relative_width": 0.05,
}


def _without_comparison_time(payload: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(payload)
    comparable.pop("compared_at_utc", None)
    return comparable


def _require_close(observed: Any, expected: float, *, label: str) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise GateError(f"E14 outcome {label} is not numeric") from error
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise GateError(
            f"E14 outcome {label}={value!r}; expected frozen value {expected!r}"
        )


def verify_e15_authorization(
    *,
    c0_approval_path: Path,
    e14_outcome_path: Path,
    expected_source_hash: str,
    logical_repo_root: Path,
) -> dict[str, Any]:
    """Replay the frozen evidence that prospectively authorizes E15 only."""

    if expected_source_hash != EXPECTED_SOURCE_HASH:
        raise GateError(
            "E15 Python source differs from the exact source used by E14 C0/M01/M05"
        )
    if _sha256_file(c0_approval_path) != EXPECTED_C0_APPROVAL_SHA256:
        raise GateError("E15 received a different C0 approval artifact")
    if _sha256_file(e14_outcome_path) != EXPECTED_E14_OUTCOME_SHA256:
        raise GateError("E15 received a different E14 comparison artifact")

    c0_summary = verify_c0_approval_for_source(
        c0_approval_path,
        expected_source_hash=expected_source_hash,
        logical_repo_root=logical_repo_root,
    )

    outcome = _strict_json(
        e14_outcome_path.read_text(encoding="utf-8"), context=str(e14_outcome_path)
    )
    if not isinstance(outcome, dict):
        raise GateError("E14 outcome is not a JSON object")
    if outcome.get("schema") != COMPARISON_SCHEMA or outcome.get("protocol") != "E14":
        raise GateError("E14 outcome has the wrong schema or protocol")
    if outcome.get("status") != "no_viable_dose" or outcome.get("selected_arm") is not None:
        raise GateError("E15 is authorized only by E14's frozen no-viable-dose outcome")
    if outcome.get("single_seed_engineering_calibration") is not True:
        raise GateError("E14 outcome lost its single-seed engineering scope")
    if outcome.get("does_not_authorize_scale_or_domain_expansion") is not True:
        raise GateError("E14 outcome lost its scale/domain firewall")
    thresholds = outcome.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(EXPECTED_THRESHOLDS):
        raise GateError("E14 outcome thresholds drifted")
    for label, expected in EXPECTED_THRESHOLDS.items():
        _require_close(thresholds[label], expected, label=f"threshold {label}")

    evidence = outcome.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "c0_approval",
        "m01_result",
        "m05_result",
    }:
        raise GateError("E14 outcome evidence is incomplete")
    for label, record in evidence.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise GateError(f"E14 outcome evidence {label!r} is malformed")
        path = Path(str(record["path"]))
        if not path.is_absolute() or not path.is_file():
            raise GateError(f"E14 outcome evidence {label!r} is missing")
        if _sha256_file(path) != record["sha256"]:
            raise GateError(f"E14 outcome evidence {label!r} changed")
    if Path(str(evidence["c0_approval"]["path"])).resolve() != c0_approval_path.resolve():
        raise GateError("E14 outcome references a different C0 approval")

    replayed = compare_result_payloads(
        c0_approval_path=c0_approval_path,
        m01_path=Path(str(evidence["m01_result"]["path"])),
        m05_path=Path(str(evidence["m05_result"]["path"])),
    )
    if _without_comparison_time(replayed) != _without_comparison_time(outcome):
        raise GateError("E14 outcome does not match replayed M01/M05 evidence")

    arms = {str(arm.get("arm")): arm for arm in replayed.get("arms", [])}
    if set(arms) != {"M01", "M05"}:
        raise GateError("E14 outcome does not contain exactly M01 and M05")
    m05 = arms["M05"]
    if (
        m05.get("runtime_valid") is not True
        or m05.get("behaviorally_safe") is not True
        or m05.get("diversity_effective") is not False
        or m05.get("viable") is not False
        or m05.get("failures")
        != ["exact valid-mode effective support gain < 25%"]
    ):
        raise GateError("E14 M05 is not the frozen safe valid-support near miss")
    if not (
        float(m05["exact_action_entropy_gain_vs_c0"]) >= math.log(1.25)
        and 1.0 < float(m05["n_eff_valid_ratio_vs_c0"]) < 1.25
        and float(m05["p_valid_retention_vs_c0"]) >= 0.8
    ):
        raise GateError("E14 M05 near-miss values do not justify the E15 calibration")

    return {
        "authorized_protocol": "E15",
        "authorized_arms": ["M075", "M10"],
        "c0_approval": str(c0_approval_path.resolve()),
        "c0_approval_sha256": _sha256_file(c0_approval_path),
        "e14_outcome": str(e14_outcome_path.resolve()),
        "e14_outcome_sha256": _sha256_file(e14_outcome_path),
        "source_hash": expected_source_hash,
        "c0_gate_replayed": c0_summary["full_c0_gate_replayed"],
        "e14_comparison_replayed": True,
        "m05_n_eff_valid_ratio_vs_c0": m05["n_eff_valid_ratio_vs_c0"],
        "m05_p_valid_retention_vs_c0": m05["p_valid_retention_vs_c0"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay E14 C0 and M01/M05 before configuring E15."
    )
    parser.add_argument("--c0-approval", type=Path, required=True)
    parser.add_argument("--e14-outcome", type=Path, required=True)
    parser.add_argument("--expected-source-hash", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = verify_e15_authorization(
            c0_approval_path=args.c0_approval,
            e14_outcome_path=args.e14_outcome,
            expected_source_hash=args.expected_source_hash,
            logical_repo_root=args.repo_root,
        )
    except (GateError, FileNotFoundError) as error:
        raise SystemExit(f"E15 authorization rejected: {error}") from error
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
