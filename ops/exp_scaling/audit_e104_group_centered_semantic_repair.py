#!/usr/bin/env python3
"""Outcome-blind mechanism audit for E104's three-scale semantic repair."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e102_full_open_bank_maxent_replay as shared  # noqa: E402
import apply_e104_qwen3_a6000_placement_amendment as qwen3_amendment  # noqa: E402
import launch_e104_group_centered_semantic_repair_three_scale as launch  # noqa: E402
import record_e104_unit_test_evidence as unit_evidence  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / launch.LEDGER
OUT = ROOT / launch.AUDIT
MEAN_TOLERANCE = 1e-8
ADVANTAGE_BOUND = launch.SEMANTIC_COEFFICIENT + 1e-7
DEVIATION = (
    ROOT
    / "paper/preregistration/e104_step0_exposure_deviation_20260817.md"
)
THEORY_CLARIFICATION = (
    ROOT
    / "paper/preregistration/e104_theory_clarification_20260817.md"
)
PLACEMENT_AMENDMENT = (
    ROOT
    / "paper/preregistration/e104_static_smoke_placement_amendment_20260817.md"
)
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "AssertionError",
    "CUDA out of memory",
    "OutOfMemoryError",
    "non-finite",
)


def scheduler_states(job_ids: list[int]) -> dict[int, str]:
    if not job_ids:
        return {}
    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            ",".join(str(value) for value in job_ids),
            "--format=JobIDRaw,State",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    states: dict[int, str] = {}
    if result.returncode != 0:
        return states
    for line in result.stdout.splitlines():
        pieces = line.split("|")
        if len(pieces) < 2 or not pieces[0].isdigit():
            continue
        states.setdefault(int(pieces[0]), pieces[1])
    return states


def parse_run(
    run_dir: Path,
    *,
    target_steps: int = launch.SMOKE_TARGET_STEPS,
) -> tuple[dict[str, Any], list[str]]:
    paths = shared.metric_paths(run_dir)
    violations: list[str] = []
    last_step = -1
    train_rows = 0
    group_active_min = 1.0
    group_active_seen = False
    legacy_active_max = 0.0
    legacy_active_seen = False
    controller_active_max = 0.0
    controller_active_seen = False
    mean_abs_max = 0.0
    advantage_abs_max = 0.0
    semantic_rms_max = 0.0
    replay_gradient_l2_max = 0.0
    replay_groups_max = 0.0
    bank_size_after_max = 0.0
    online_eligible_fraction_max = 0.0
    semantic_eligible_fraction_max = 0.0
    support_at_least_two_prompt_fraction_max = 0.0
    mean_support_per_prompt_max = 0.0
    history_rows_added_max = 0.0

    if not paths:
        violations.append("no train_metrics.jsonl exists")
    for path, line_number, row in shared.rows(paths):
        if row.get("__invalid_json__"):
            violations.append(f"invalid JSON at {path}:{line_number}")
            continue
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    violations.append(
                        f"non-finite {key} at {path}:{line_number}"
                    )
        raw_step = row.get(
            "misc/global_step",
            row.get("trainer/global_step", row.get("trainer/step", -1)),
        )
        if shared.finite(raw_step):
            last_step = max(last_step, int(raw_step))
        group_active = shared.metric(
            row,
            "semantic_shannon_success_conditioned_group_centered_advantage_active",
        )
        if group_active is None:
            continue
        train_rows += 1
        group_active_seen = True
        group_active_min = min(group_active_min, group_active)
        legacy_active = shared.metric(
            row,
            "semantic_shannon_success_conditioned_signed_advantage_active",
        )
        if legacy_active is not None:
            legacy_active_seen = True
            legacy_active_max = max(legacy_active_max, abs(legacy_active))
        controller_active = shared.metric(
            row, "semantic_rms_controller_active"
        )
        if controller_active is not None:
            controller_active_seen = True
            controller_active_max = max(
                controller_active_max, abs(controller_active)
            )
        mean_abs_max = max(
            mean_abs_max,
            abs(
                shared.metric(
                    row,
                    "semantic_shannon_success_conditioned_group_centered_effective_advantage_mean",
                )
                or 0.0
            ),
        )
        for suffix in (
            "semantic_shannon_success_conditioned_group_centered_effective_advantage_min",
            "semantic_shannon_success_conditioned_group_centered_effective_advantage_max",
            "semantic_shannon_separate_semantic_advantage_min",
            "semantic_shannon_separate_semantic_advantage_max",
        ):
            advantage_abs_max = max(
                advantage_abs_max, abs(shared.metric(row, suffix) or 0.0)
            )
        semantic_rms_max = max(
            semantic_rms_max,
            shared.metric(
                row,
                "semantic_shannon_success_conditioned_group_centered_effective_advantage_rms",
            )
            or 0.0,
        )
        replay_gradient_l2_max = max(
            replay_gradient_l2_max,
            shared.metric(row, "canonical_replay_applied_score_gradient_l2")
            or 0.0,
        )
        replay_groups_max = max(
            replay_groups_max,
            shared.metric(row, "canonical_replay_actuator_groups") or 0.0,
        )
        bank_size_after_max = max(
            bank_size_after_max,
            shared.metric(row, "online_canonical_bank_size_after_mean") or 0.0,
        )
        online_eligible_fraction_max = max(
            online_eligible_fraction_max,
            shared.metric(row, "online_canonical_eligible_fraction") or 0.0,
        )
        semantic_eligible_fraction_max = max(
            semantic_eligible_fraction_max,
            shared.metric(
                row,
                "semantic_shannon_success_conditioned_group_centered_eligible_fraction",
            )
            or 0.0,
        )
        support_at_least_two_prompt_fraction_max = max(
            support_at_least_two_prompt_fraction_max,
            shared.metric(
                row, "online_canonical_support_at_least_two_prompt_fraction"
            )
            or 0.0,
        )
        mean_support_per_prompt_max = max(
            mean_support_per_prompt_max,
            shared.metric(row, "verified_discovery_mean_support_per_prompt") or 0.0,
        )
        history_rows_added_max = max(
            history_rows_added_max,
            shared.metric(
                row,
                "semantic_shannon_success_conditioned_group_centered_history_rows_added",
            )
            or 0.0,
        )

    if last_step < target_steps:
        violations.append(f"only {last_step}/{target_steps} optimizer steps")
    if not group_active_seen or group_active_min < 1.0:
        violations.append("v6 group-centered estimator was not active")
    if not legacy_active_seen or legacy_active_max > 0.0:
        violations.append("legacy v5 semantic estimator was active or unreported")
    if not controller_active_seen or controller_active_max > 0.0:
        violations.append("semantic RMS controller was active or unreported")
    if mean_abs_max > MEAN_TOLERANCE:
        violations.append(
            f"sampled-group semantic mean exceeded tolerance: {mean_abs_max}"
        )
    if advantage_abs_max > ADVANTAGE_BOUND:
        violations.append(
            f"semantic advantage exceeded eta bound: {advantage_abs_max}"
        )
    if replay_groups_max <= 0.0 or replay_gradient_l2_max <= 0.0:
        violations.append("verified replay never produced an applied update")

    return (
        {
            "metric_paths": [str(path) for path in paths],
            "last_step": last_step,
            "train_rows": train_rows,
            "group_centered_active_min": (
                group_active_min if group_active_seen else 0.0
            ),
            "legacy_active_max": legacy_active_max,
            "controller_active_max": controller_active_max,
            "semantic_effective_mean_abs_max": mean_abs_max,
            "semantic_advantage_abs_max": advantage_abs_max,
            "semantic_rms_max": semantic_rms_max,
            "replay_gradient_l2_max": replay_gradient_l2_max,
            "replay_groups_max": replay_groups_max,
            "bank_size_after_max": bank_size_after_max,
            "online_eligible_fraction_max": online_eligible_fraction_max,
            "semantic_eligible_fraction_max": semantic_eligible_fraction_max,
            "support_at_least_two_prompt_fraction_max": (
                support_at_least_two_prompt_fraction_max
            ),
            "mean_support_per_prompt_max": mean_support_per_prompt_max,
            "history_rows_added_max": history_rows_added_max,
        },
        violations,
    )


def log_failures(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [marker for marker in FAILURE_MARKERS if marker in text]


def validate_qwen3_amendment() -> tuple[dict[str, Any], list[str]]:
    if not qwen3_amendment.OUT.is_file():
        return {}, []
    violations: list[str] = []
    try:
        payload = json.loads(
            qwen3_amendment.OUT.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"Qwen-3B placement amendment is invalid: {exc}"]
    expected_scalars = {
        "schema": "e104_qwen3_a6000_placement_amendment_v1",
        "preflight_passed": True,
        "outcome_metrics_inspected": False,
        "scientific_configuration_changed": False,
        "e105_placement_changed": False,
        "job_ids": list(qwen3_amendment.TARGET_JOB_IDS),
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            violations.append(
                f"Qwen-3B placement amendment has invalid {key}"
            )
    artifacts = (
        ("protocol", "protocol_sha256", qwen3_amendment.PROTOCOL),
        ("script", "script_sha256", Path(qwen3_amendment.__file__)),
        ("e104_ledger", "e104_ledger_sha256", LEDGER),
        (
            "preflight_ledger",
            "preflight_ledger_sha256",
            qwen3_amendment.PREFLIGHT_LEDGER,
        ),
        (
            "preflight_audit",
            "preflight_audit_sha256",
            qwen3_amendment.PREFLIGHT_AUDIT,
        ),
    )
    for path_key, digest_key, path in artifacts:
        if payload.get(path_key) != str(path):
            violations.append(
                f"Qwen-3B placement amendment {path_key} path mismatch"
            )
        if not path.is_file() or payload.get(digest_key) != launch.digest(path):
            violations.append(
                f"Qwen-3B placement amendment {path_key} digest mismatch"
            )
    after = payload.get("after", {})
    for job_id in qwen3_amendment.TARGET_JOB_IDS:
        try:
            qwen3_amendment.validate_amended(
                {"job_id": job_id}, str(after[str(job_id)])
            )
        except (KeyError, RuntimeError) as exc:
            violations.append(
                f"Qwen-3B placement amendment post-audit failed: {exc}"
            )
    return payload, violations


def main() -> int:
    if not LEDGER.is_file():
        raise SystemExit(f"E104 ledger is absent: {LEDGER}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    violations: list[str] = []
    if ledger.get("schema") != (
        "e104_group_centered_semantic_repair_three_scale_jobs_v1"
    ):
        violations.append("unknown E104 ledger schema")
    protocol = Path(str(ledger.get("protocol", "")))
    snapshot = Path(str(ledger.get("snapshot_root", "")))
    try:
        launch.verify_snapshot(snapshot)
    except SystemExit as exc:
        violations.append(str(exc))
    launcher_path = ROOT / (
        "ops/exp_scaling/"
        "launch_e104_group_centered_semantic_repair_three_scale.py"
    )
    if not protocol.is_file() or launch.digest(protocol) != ledger.get(
        "protocol_sha256"
    ):
        violations.append("frozen protocol digest mismatch")
    if not DEVIATION.is_file():
        violations.append("documented step-0 exposure note is absent")
    if not THEORY_CLARIFICATION.is_file():
        violations.append("documented theory clarification is absent")
    if not PLACEMENT_AMENDMENT.is_file():
        violations.append("documented smoke placement amendment is absent")
    unit_payload: dict[str, Any] = {}
    if not unit_evidence.OUT.is_file():
        violations.append("snapshot unit-test evidence is absent")
    else:
        try:
            unit_payload = json.loads(
                unit_evidence.OUT.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            violations.append(f"snapshot unit-test evidence is invalid: {exc}")
        else:
            try:
                violations.extend(
                    unit_evidence.validate_evidence(
                        unit_payload,
                        snapshot=snapshot,
                    )
                )
            except OSError as exc:
                violations.append(
                    f"snapshot unit-test evidence cannot be verified: {exc}"
                )
    if not launcher_path.is_file() or launch.digest(
        launcher_path
    ) != ledger.get("launcher_sha256"):
        violations.append("frozen launcher digest mismatch")
    qwen3_payload, qwen3_violations = validate_qwen3_amendment()
    violations.extend(qwen3_violations)
    runs = ledger.get("runs", [])
    states = scheduler_states([int(run["job_id"]) for run in runs])
    reports: list[dict[str, Any]] = []
    nonzero_scales: set[str] = set()
    complete = len(runs) == 15
    for run in runs:
        report, run_violations = parse_run(Path(str(run["run_dir"])))
        scale = str(run["scale"])
        if report["semantic_rms_max"] > 0.0:
            nonzero_scales.add(scale)
        label = f"{scale}/{run['domain']}/s{run['seed']}"
        violations.extend(f"{label}: {value}" for value in run_violations)
        state = states.get(int(run["job_id"]), "UNKNOWN")
        if state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")):
            violations.append(f"{label}: terminal scheduler state {state}")
        for stream_name in ("stdout", "stderr"):
            for marker in log_failures(Path(str(run[stream_name]))):
                violations.append(
                    f"{label}: {stream_name} contains {marker!r}"
                )
        complete = complete and report["last_step"] >= launch.SMOKE_TARGET_STEPS
        reports.append(dict(run) | {"scheduler_state": state, "report": report})
    missing_activation = set(launch.SCALE_SEEDS) - nonzero_scales
    if complete and missing_activation:
        violations.append(
            "no live nonzero semantic update for scales: "
            + ", ".join(sorted(missing_activation))
        )
    passed = complete and not violations
    payload = {
        "schema": "e104_group_centered_semantic_repair_gate_v1",
        "ledger": str(LEDGER),
        "snapshot_root": str(snapshot),
        "target_steps": launch.SMOKE_TARGET_STEPS,
        "mean_tolerance": MEAN_TOLERANCE,
        "advantage_bound": ADVANTAGE_BOUND,
        "complete": complete,
        "nonzero_semantic_scales": sorted(nonzero_scales),
        "passed": passed,
        "outcome_metrics_inspected": True,
        "post_update_outcome_metrics_inspected": False,
        "mechanism_gate_used_outcome_metrics": False,
        "manual_pre_treatment_outcome_exposure": {
            "scope": "one aggregate correctness field at step 0",
            "scale": "qwen05b",
            "domain": "graph_coloring",
            "seed": 43,
            "job_id": 30637786,
            "deviation": str(DEVIATION),
            "deviation_sha256": (
                launch.digest(DEVIATION) if DEVIATION.is_file() else None
            ),
        },
        "theory_clarification": str(THEORY_CLARIFICATION),
        "theory_clarification_sha256": (
            launch.digest(THEORY_CLARIFICATION)
            if THEORY_CLARIFICATION.is_file()
            else None
        ),
        "placement_amendment": str(PLACEMENT_AMENDMENT),
        "placement_amendment_sha256": (
            launch.digest(PLACEMENT_AMENDMENT)
            if PLACEMENT_AMENDMENT.is_file()
            else None
        ),
        "qwen3_a6000_placement_amendment": (
            {
                "path": str(qwen3_amendment.OUT),
                "sha256": launch.digest(qwen3_amendment.OUT),
                "validated": not qwen3_violations,
                "preflight_job_id": qwen3_payload.get("preflight_job_id"),
            }
            if qwen3_payload
            else None
        ),
        "unit_test_evidence": {
            "path": str(unit_evidence.OUT),
            "sha256": (
                launch.digest(unit_evidence.OUT)
                if unit_evidence.OUT.is_file()
                else None
            ),
            "passed": unit_payload.get("passed"),
            "summary": unit_evidence.EXPECTED_SUMMARY,
        },
        "runs": reports,
        "violations": violations,
    }
    shared.atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
