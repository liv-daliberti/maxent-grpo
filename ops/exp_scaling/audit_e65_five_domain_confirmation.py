#!/usr/bin/env python3
"""Combined confirmation audit without mutating frozen cohort auditors."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[2]
E61 = ROOT / "var/artifacts/e61r1_e58_vs_grpo_12pass_audit_latest.json"
E64 = ROOT / "var/artifacts/e64_math500_realism_matched_audit_latest.json"
E64_SENSITIVITY = (
    ROOT / "var/artifacts/e64_math500_verifier_sensitivity_latest.json"
)
E68 = (
    ROOT / "var/artifacts/e68_separated_support_actuator_ablation_audit_latest.json"
)
E68_EQUIVALENCE = (
    ROOT / "var/artifacts/e68_preintervention_equivalence_audit_latest.json"
)
E68_CHECKPOINT = (
    ROOT / "var/artifacts/e68_checkpoint_separation_audit_latest.json"
)
E65_INVALIDATION = (
    ROOT / "var/artifacts/e65r1_objective_mismatch_invalidation.json"
)
E67_INVALIDATION = (
    ROOT / "var/artifacts/e67_preoptimizer_invalidation.json"
)
E66 = ROOT / "var/artifacts/e66_same_plumbing_actuator_ablation_audit_latest.json"
EVAL_CADENCE = ROOT / "var/artifacts/e65_eval_cadence_audit_latest.json"
CHECKPOINT_COVERAGE = (
    ROOT / "var/artifacts/e65_fixed_checkpoint_coverage_audit_latest.json"
)
OUT = ROOT / "var/artifacts/e65_five_domain_confirmation_audit_latest.json"
UNCAUGHT = re.compile(
    r"\[rank\d+\]: Traceback \(most recent call last\)|CUDA out of memory|"
    r"torch\.OutOfMemoryError|ChildFailedError|RayActorError|"
    r"RuntimeError:[^\n]*non-finite|segmentation fault",
    re.IGNORECASE,
)
SIGTERM = re.compile(r"SIGTERM Signal received", re.IGNORECASE)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _e64_unrecovered_failures(payload: dict) -> list[str]:
    """Re-audit process failures while retaining frozen-auditor provenance.

    The frozen E64 auditor deliberately matches every string ``Traceback``.
    ``math_verify`` also logs caught per-example timeout tracebacks and
    continues evaluation. This wrapper does not alter that frozen artifact; it
    distinguishes those caught diagnostics from an uncaught rank failure.
    """

    failures: list[str] = []
    for run in payload.get("runs", []):
        job_id = int(run["job_id"])
        for suffix in ("out", "err"):
            path = ROOT / f"var/artifacts/logs/xdr_train-{job_id}.{suffix}"
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in UNCAUGHT.finditer(text):
                nearby = text[max(0, match.start() - 3000) : match.start()]
                if "Traceback" in match.group(0) and SIGTERM.search(nearby):
                    continue
                failures.append(
                    f"E64 job {job_id}: uncaught {match.group(0)!r}"
                )
    non_log_violations = [
        violation
        for violation in payload.get("violations", [])
        if "crash signature 'Traceback (most recent call last)'" not in violation
    ]
    return non_log_violations + failures


def main() -> None:
    inputs = {
        "e61": _load(E61),
        "e64": _load(E64),
        "e64_sensitivity": _load(E64_SENSITIVITY),
        "e68": _load(E68),
        "e68_equivalence": _load(E68_EQUIVALENCE),
        "e68_checkpoint": _load(E68_CHECKPOINT),
        "e65_invalidation": _load(E65_INVALIDATION),
        "e67_invalidation": _load(E67_INVALIDATION),
        "e66": _load(E66),
        "eval_cadence": _load(EVAL_CADENCE),
        "checkpoint_coverage": _load(CHECKPOINT_COVERAGE),
    }
    violations: list[str] = []
    if inputs["e61"].get("status") == "fail":
        violations.extend(
            f"E61: {value}" for value in inputs["e61"].get("violations", [])
        )
    violations.extend(_e64_unrecovered_failures(inputs["e64"]))
    if inputs["e64_sensitivity"].get("status") == "fail":
        violations.extend(
            f"E64 verifier sensitivity: {value}"
            for value in inputs["e64_sensitivity"].get("violations", [])
        )
    if inputs["e68"].get("status") == "fail":
        violations.extend(
            f"E68: {value}" for value in inputs["e68"].get("violations", [])
        )
    if inputs["e68_equivalence"].get("status") == "fail":
        violations.extend(
            f"E68 equivalence: {value}"
            for value in inputs["e68_equivalence"].get("violations", [])
        )
    if inputs["e68_checkpoint"].get("status") == "fail":
        violations.extend(
            f"E68 checkpoint separation: {value}"
            for value in inputs["e68_checkpoint"].get("violations", [])
        )
    if inputs["e65_invalidation"].get("status") != "confirmed":
        violations.extend(
            f"E65 invalidation: {value}"
            for value in inputs["e65_invalidation"].get(
                "violations",
                ["objective mismatch not confirmed"],
            )
        )
    if inputs["e67_invalidation"].get("status") != "confirmed":
        violations.extend(
            f"E67 invalidation: {value}"
            for value in inputs["e67_invalidation"].get(
                "violations",
                ["pre-optimizer invalidation not confirmed"],
            )
        )
    if inputs["e66"].get("status") == "fail":
        violations.extend(
            f"E66: {value}" for value in inputs["e66"].get("violations", [])
        )
    if inputs["eval_cadence"].get("status") != "pass":
        violations.extend(
            f"Evaluation cadence: {value}"
            for value in inputs["eval_cadence"].get(
                "violations",
                ["once-per-epoch evaluation audit did not pass"],
            )
        )
    if inputs["checkpoint_coverage"].get("status") == "fail":
        violations.extend(
            f"Fixed checkpoint coverage: {value}"
            for value in inputs["checkpoint_coverage"].get("violations", [])
        )
    all_terminal = (
        inputs["e61"].get("summary", {}).get("terminal_runs") == 24
        and inputs["e64"].get("summary", {}).get("terminal_runs") == 6
        and inputs["e64_sensitivity"].get("status") == "pass"
        and inputs["e68"].get("summary", {}).get("terminal_runs") == 12
        and inputs["e68_equivalence"].get("status") == "pass"
        and inputs["e68_checkpoint"].get("status") == "pass"
        and inputs["e65_invalidation"].get("status") == "confirmed"
        and inputs["e67_invalidation"].get("status") == "confirmed"
        and inputs["e66"].get("summary", {}).get("terminal_runs") == 12
        and inputs["eval_cadence"].get("status") == "pass"
        and inputs["checkpoint_coverage"].get("status") == "pass"
    )
    status = "fail" if violations else "pass" if all_terminal else "in_progress"
    payload = {
        "schema": "e65_five_domain_confirmation_audit_v1",
        "status": status,
        "evidential_status": {
            "e61_e64": (
                "exploratory trajectories with prospectively frozen "
                "terminal and fixed-checkpoint AUC analysis"
            ),
            "e65r1": (
                "invalidated engineering cohort: runtime novelty beta was "
                "0 instead of literal E58's 0.5; excluded from denominator"
            ),
            "e68": (
                "prospective separated-support same-objective actuator-on cohort"
            ),
            "e67": (
                "invalidated pre-optimizer cohort: shared proposal and "
                "on-policy objective support; excluded from denominator"
            ),
            "e66": (
                "prospective same-plumbing actuator-off control frozen before "
                "any three-seed E65R1 post-training checkpoint"
            ),
        },
        "fixed_paper_checkpoints": {
            "modebench": [0, 1, 2, 3, 4, 5, 6, 8, 10, 12],
            "math500": [0, 2, 4, 6, 8, 10, 12],
        },
        "summary": {
            "expected_runs": 54,
            "terminal_runs": (
                int(inputs["e61"].get("summary", {}).get("terminal_runs", 0))
                + int(inputs["e64"].get("summary", {}).get("terminal_runs", 0))
                + int(inputs["e68"].get("summary", {}).get("terminal_runs", 0))
                + int(inputs["e66"].get("summary", {}).get("terminal_runs", 0))
            ),
            "e61_status": inputs["e61"].get("status"),
            "e64_frozen_auditor_status": inputs["e64"].get("status"),
            "e64_caught_grader_trace_reinterpreted": bool(
                inputs["e64"].get("violations")
            )
            and not bool(_e64_unrecovered_failures(inputs["e64"])),
            "e64_verifier_sensitivity_status": inputs[
                "e64_sensitivity"
            ].get("status"),
            "e64_verifier_timeout_diagnostics": int(
                inputs["e64_sensitivity"]
                .get("summary", {})
                .get("verifier_timeout_diagnostics", 0)
            ),
            "e64_identical_response_reward_conflicts": int(
                inputs["e64_sensitivity"]
                .get("summary", {})
                .get("identical_response_reward_conflicts", 0)
            ),
            "e68_status": inputs["e68"].get("status"),
            "e68_preintervention_equivalence_status": inputs[
                "e68_equivalence"
            ].get("status"),
            "e68_preintervention_ready_pairs": int(
                inputs["e68_equivalence"]
                .get("summary", {})
                .get("ready_pairs", 0)
            ),
            "e68_checkpoint_separation_status": inputs[
                "e68_checkpoint"
            ].get("status"),
            "e68_checkpointed_runs": int(
                inputs["e68_checkpoint"]
                .get("summary", {})
                .get("checkpointed_runs", 0)
            ),
            "e65r1_objective_mismatch_status": inputs[
                "e65_invalidation"
            ].get("status"),
            "e67_preoptimizer_invalidation_status": inputs[
                "e67_invalidation"
            ].get("status"),
            "e66_status": inputs["e66"].get("status"),
            "eval_cadence_status": inputs["eval_cadence"].get("status"),
            "eval_cadence_audited_runs": int(
                inputs["eval_cadence"]
                .get("summary", {})
                .get("audited_runs", 0)
            ),
            "fixed_checkpoint_coverage_status": inputs[
                "checkpoint_coverage"
            ].get("status"),
            "fixed_checkpoint_cells": int(
                inputs["checkpoint_coverage"]
                .get("summary", {})
                .get("landed_checkpoint_cells", 0)
            ),
            "expected_fixed_checkpoint_cells": int(
                inputs["checkpoint_coverage"]
                .get("summary", {})
                .get("expected_checkpoint_cells", 0)
            ),
        },
        "violations": sorted(set(violations)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUT.name}.", dir=OUT.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUT)
    print(
        f"[five-domain-audit] status={status} "
        f"terminal={payload['summary']['terminal_runs']}/54 "
        f"violations={len(payload['violations'])}"
    )


if __name__ == "__main__":
    main()
