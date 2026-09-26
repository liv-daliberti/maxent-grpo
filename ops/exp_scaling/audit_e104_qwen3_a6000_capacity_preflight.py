#!/usr/bin/env python3
"""Outcome-blind runtime audit for the E104 Qwen-3B A6000 preflight."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e104_group_centered_semantic_repair as e104_audit  # noqa: E402
import launch_e104_group_centered_semantic_repair_three_scale as e104  # noqa: E402
import launch_e104_qwen3_a6000_capacity_preflight as launch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / launch.LEDGER
OUT = ROOT / launch.AUDIT


def parse_run(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    paths = e104_audit.shared.metric_paths(run_dir)
    violations: list[str] = []
    last_step = -1
    rows = 0
    active_min = 1.0
    legacy_max = 0.0
    controller_max = 0.0
    mean_abs_max = 0.0
    advantage_abs_max = 0.0
    replay_gradient_max = 0.0
    for path, line_number, row in e104_audit.shared.rows(paths):
        if row.get("__invalid_json__"):
            violations.append(f"invalid JSON at {path}:{line_number}")
            continue
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    violations.append(f"non-finite {key} at {path}:{line_number}")
        step = row.get(
            "misc/global_step",
            row.get("trainer/global_step", row.get("trainer/step")),
        )
        if e104_audit.shared.finite(step):
            last_step = max(last_step, int(step))
        active = e104_audit.shared.metric(
            row,
            "semantic_shannon_success_conditioned_group_centered_advantage_active",
        )
        if active is None:
            continue
        rows += 1
        active_min = min(active_min, active)
        legacy_max = max(
            legacy_max,
            abs(
                e104_audit.shared.metric(
                    row,
                    "semantic_shannon_success_conditioned_signed_advantage_active",
                )
                or 0.0
            ),
        )
        controller_max = max(
            controller_max,
            abs(e104_audit.shared.metric(row, "semantic_rms_controller_active") or 0.0),
        )
        mean_abs_max = max(
            mean_abs_max,
            abs(
                e104_audit.shared.metric(
                    row,
                    "semantic_shannon_success_conditioned_group_centered_effective_advantage_mean",
                )
                or 0.0
            ),
        )
        for key in (
            "semantic_shannon_success_conditioned_group_centered_effective_advantage_min",
            "semantic_shannon_success_conditioned_group_centered_effective_advantage_max",
        ):
            advantage_abs_max = max(
                advantage_abs_max,
                abs(e104_audit.shared.metric(row, key) or 0.0),
            )
        replay_gradient_max = max(
            replay_gradient_max,
            e104_audit.shared.metric(
                row, "canonical_replay_applied_score_gradient_l2"
            )
            or 0.0,
        )
    if not paths:
        violations.append("no train_metrics.jsonl exists")
    if last_step < launch.TRAIN_ROWS:
        violations.append(f"only {last_step}/{launch.TRAIN_ROWS} optimizer steps")
    if rows == 0 or active_min < 1.0:
        violations.append("v6 group-centered estimator was not active")
    if legacy_max > 0.0:
        violations.append("legacy semantic estimator was active")
    if controller_max > 0.0:
        violations.append("semantic RMS controller was active")
    if mean_abs_max > e104_audit.MEAN_TOLERANCE:
        violations.append("group-centered mean exceeded tolerance")
    if advantage_abs_max > e104_audit.ADVANTAGE_BOUND:
        violations.append("semantic advantage exceeded eta bound")
    if replay_gradient_max <= 0.0:
        violations.append("verified replay was not applied")
    return (
        {
            "metric_paths": [str(path) for path in paths],
            "last_step": last_step,
            "train_rows": rows,
            "group_centered_active_min": active_min if rows else 0.0,
            "legacy_active_max": legacy_max,
            "controller_active_max": controller_max,
            "semantic_effective_mean_abs_max": mean_abs_max,
            "semantic_advantage_abs_max": advantage_abs_max,
            "replay_gradient_l2_max": replay_gradient_max,
        },
        violations,
    )


def main() -> int:
    if not LEDGER.is_file():
        raise SystemExit(f"preflight ledger is absent: {LEDGER}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    violations: list[str] = []
    protocol = Path(str(ledger.get("protocol", "")))
    launcher_path = Path(str(ledger.get("launcher", "")))
    if not protocol.is_file() or e104.digest(protocol) != ledger.get(
        "protocol_sha256"
    ):
        violations.append("preflight protocol digest mismatch")
    if not launcher_path.is_file() or e104.digest(
        launcher_path
    ) != ledger.get("launcher_sha256"):
        violations.append("preflight launcher digest mismatch")
    snapshot = Path(str(ledger.get("snapshot_root", "")))
    try:
        e104.verify_snapshot(snapshot)
    except SystemExit as exc:
        violations.append(str(exc))
    report, run_violations = parse_run(Path(str(ledger["run_dir"])))
    violations.extend(run_violations)
    job_id = int(ledger["job_id"])
    state = e104_audit.scheduler_states([job_id]).get(job_id, "UNKNOWN")
    complete = state.startswith("COMPLETED") and report["last_step"] >= 1
    if state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")):
        violations.append(f"terminal scheduler state {state}")
    for stream in ("stdout", "stderr"):
        for marker in e104_audit.log_failures(Path(str(ledger[stream]))):
            violations.append(f"{stream} contains {marker!r}")
    payload = {
        "schema": "e104_qwen3_a6000_capacity_preflight_gate_v1",
        "ledger": str(LEDGER),
        "snapshot_root": str(snapshot),
        "job_id": job_id,
        "scheduler_state": state,
        "complete": complete,
        "passed": complete and not violations,
        "outcome_metrics_inspected": False,
        "report": report,
        "violations": violations,
    }
    e104_audit.shared.atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
