#!/usr/bin/env python3
"""Corrected implementation of the Pantry final-v1 audit amendment.

The v2 draft stopped because it included the terminal summary row and required
diagnostics from a separate controller on bank-ineligible updates.  This
version narrows validation to the exact disputed quantity: the 384 effective
mass-controller coefficients.  It also retains v1's already-satisfied 5%
tolerance check for the separate replay-balance coefficient.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import audit_pantry_support_retention_final_v1_amendment_v2 as base


def controller_check(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    # This is the exact boundary used by the immutable v1 auditor: initial
    # evaluation at row zero, 384 optimizer rows, then a terminal summary.
    rows = metric_rows[1 : base.UPDATES + 1]
    errors: list[str] = []
    if len(rows) != base.UPDATES:
        errors.append(
            f"expected {base.UPDATES} controller rows, found {len(rows)}"
        )
    if any(
        "train/canonical_replay_mass_alpha_used" not in row for row in rows
    ):
        errors.append("one or more optimizer rows lack mass-controller telemetry")

    alpha_used: list[float] = []
    mass_used: list[float] = []
    for index, row in enumerate(rows):
        observation = index + 1
        try:
            replay_used = base.value(
                row, "canonical_replay_alpha_used"
            )
            used = base.value(row, "canonical_replay_mass_alpha_used")
            before = base.value(
                row, "canonical_replay_mass_alpha_before"
            )
            next_alpha = base.value(
                row, "canonical_replay_mass_next_alpha"
            )
            observations = base.value(
                row, "canonical_replay_mass_observations"
            )
            ema = base.value(
                row, "canonical_replay_mass_surprisal_ema"
            )
            warm = base.value(
                row, "canonical_replay_mass_warmup_complete"
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(
                f"row {observation}: incomplete controller telemetry: {exc}"
            )
            continue

        scalars = (
            replay_used,
            used,
            before,
            next_alpha,
            observations,
            ema,
            warm,
        )
        if not all(math.isfinite(item) for item in scalars):
            errors.append(f"row {observation}: non-finite controller telemetry")
            continue
        if replay_used <= 0 or used <= 0 or ema <= 0:
            errors.append(f"row {observation}: non-positive controller state")

        alpha_used.append(replay_used)
        mass_used.append(used)
        # Preserve the already-satisfied replay-alpha criterion from v1.
        if not math.isclose(
            replay_used,
            base.BASE_ALPHA,
            rel_tol=0.05,
            abs_tol=0.005,
        ):
            errors.append(
                f"row {observation}: replay alpha violates v1 tolerance"
            )
        if not base.close(observations, observation):
            errors.append(
                f"row {observation}: mass observation index mismatch"
            )
        if not base.close(used, before):
            errors.append(
                f"row {observation}: mass used/before discontinuity"
            )
        if index == 0:
            if not base.close(used, base.BASE_ALPHA):
                errors.append(
                    "row 1: mass controller did not start at base alpha"
                )
        else:
            previous_next = base.value(
                rows[index - 1],
                "canonical_replay_mass_next_alpha",
            )
            if not base.close(used, previous_next):
                errors.append(
                    f"row {observation}: mass next-to-used "
                    "trajectory discontinuity"
                )

        expected_warm = 1.0 if observation >= base.WARMUP else 0.0
        if not base.close(warm, expected_warm):
            errors.append(f"row {observation}: mass warmup flag mismatch")

        if observation <= base.WARMUP:
            expected_next = base.BASE_ALPHA
        else:
            try:
                reference = base.value(
                    row,
                    "canonical_replay_mass_surprisal_reference",
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(
                    f"row {observation}: post-warmup reference absent: {exc}"
                )
                continue
            if reference <= 0:
                errors.append(
                    f"row {observation}: post-warmup reference is non-positive"
                )
                continue
            expected_next = base.BASE_ALPHA * ema / reference
        if not base.close(next_alpha, expected_next):
            errors.append(
                f"row {observation}: mass controller law mismatch"
            )

    return {
        "passed": not errors and len(rows) == base.UPDATES,
        "row_count": len(rows),
        "warmup_observations": base.WARMUP,
        "registered_base_alpha": base.BASE_ALPHA,
        "minimum_replay_alpha_used": min(alpha_used) if alpha_used else None,
        "maximum_replay_alpha_used": max(alpha_used) if alpha_used else None,
        "minimum_mass_alpha_used": min(mass_used) if mass_used else None,
        "maximum_mass_alpha_used": max(mass_used) if mass_used else None,
        "errors": errors,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    v2_draft = (
        root
        / "var/artifacts/"
        "pantry_support_retention_final_v1_audit_amendment_v2.json"
    )
    output = (
        root
        / "var/artifacts/"
        "pantry_support_retention_final_v1_audit_amendment_v3.json"
    )
    base.controller_check = controller_check
    original_argv = sys.argv
    try:
        sys.argv = [
            original_argv[0],
            "--repo-root",
            str(root),
            "--output",
            str(output),
        ]
        status = base.main()
    finally:
        sys.argv = original_argv

    payload = json.loads(output.read_text())
    payload["schema"] = (
        "pantry-support-retention-final-audit-amendment-v3"
    )
    payload["implementation_note"] = (
        "Supersedes a stopped v2 draft that over-scoped trajectory "
        "validation; v1's immutable experimental receipt remains unchanged."
    )
    if v2_draft.is_file():
        payload["supersedes_stopped_draft"] = {
            "path": str(v2_draft),
            "sha256": base.sha256(v2_draft),
            "status": json.loads(v2_draft.read_text()).get("status"),
        }
    base.atomic(output, payload)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
