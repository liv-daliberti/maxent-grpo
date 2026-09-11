#!/usr/bin/env python3
"""Live fail-closed audit for E67's prospective 12-job repair cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE_HASH = (
    "f6147daacbfdde22e0e9d5fab6fc45b41017d4dbf5e2848f827923ddf7828a7f"
)
IDENTITY = (
    ROOT / "var/artifacts/e67_corrected_same_objective_actuator_ablation_identity.json"
)
PROTOCOL = (
    ROOT / "paper/preregistration/e67_corrected_same_objective_actuator_ablation_05b.md"
)
LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_e67_corrected_same_objective_actuator_ablation.sh"
)
OUT = (
    ROOT / "var/artifacts/e67_corrected_same_objective_actuator_ablation_audit_latest.json"
)
PLACEMENT_AMENDMENT = (
    ROOT
    / "paper/preregistration/e67_zero_step_placement_amendment_20260727.md"
)
PLACEMENT_SCRIPT = (
    ROOT / "ops/exp_scaling/amend_e67_zero_step_placement.sh"
)
PLACEMENT_RECORD = (
    ROOT / "var/artifacts/e67_zero_step_placement_amendment.json"
)
EXPECTED_STEPS = {
    "graph_coloring": 192 * 12,
    "countdown": 384 * 12,
    "python_factor": 384 * 12,
    "mathir": 384 * 12,
}
UNCAUGHT = re.compile(
    r"\[rank\d+\]: Traceback \(most recent call last\)|CUDA out of memory|"
    r"torch\.OutOfMemoryError|ChildFailedError|RayActorError|"
    r"RuntimeError:[^\n]*non-finite|segmentation fault",
    re.IGNORECASE,
)
SIGTERM = re.compile(r"SIGTERM Signal received", re.IGNORECASE)
def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _zero(value: Any) -> bool:
    return _finite(value) and math.isclose(
        float(value),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _metric(row: dict[str, Any], name: str) -> Any:
    """Read raw actor telemetry or its parser-normalized train alias."""

    for prefix in ("train/", "actor/"):
        key = f"{prefix}{name}"
        if key in row:
            return row[key]
    return None


def _run_dir(run_stamp: str, job_id: int) -> Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _scan_log(path: Path) -> tuple[list[str], int]:
    if not path.is_file():
        return [], 0
    text = path.read_text(encoding="utf-8", errors="replace")
    failures: list[str] = []
    interruptions = 0
    for match in UNCAUGHT.finditer(text):
        nearby = text[max(0, match.start() - 16000) : match.start()]
        if "Traceback" in match.group(0) and SIGTERM.search(nearby):
            interruptions += 1
        else:
            failures.append(match.group(0))
    return failures, interruptions


def _scan_metrics(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    latest_step = -1
    records = 0
    interventions = 0
    maximum_admitted = 0
    objective_contract_records = 0
    intervention_events: list[dict[str, Any]] = []
    violations: list[str] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            violations.append(f"{label}: invalid JSON line {line_number}")
            continue
        step = row.get("trainer/global_step", row.get("trainer/step", -1))
        if _finite(step):
            latest_step = int(step)
        if not any(key.startswith("train/") for key in row):
            continue
        records += 1
        if "train/online_canonical_new_outcome_count" in row:
            objective_contract_records += 1
            applied = row.get(
                "train/"
                "online_canonical_advantage_applied_after_task_centering"
            )
            if not (
                _finite(applied)
                and math.isclose(float(applied), 1.0)
            ):
                violations.append(
                    f"{label}: E58 novelty objective not applied at "
                    f"line {line_number}"
                )
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    violations.append(
                        f"{label}: non-finite {key} at line {line_number}"
                    )
        required_zero_suffixes = (
            "gold_support_feedback",
            "desired_mode_count_feedback",
            "eval_feedback",
            "rows_sent_to_ppo",
            "projection_active",
        )
        for key, value in row.items():
            if any(key.endswith(suffix) for suffix in required_zero_suffixes):
                if _finite(value) and not _zero(value):
                    violations.append(
                        f"{label}: forbidden nonzero {key} at line {line_number}"
                    )
        enabled = _metric(
            row,
            "counterfactual_proposal_singleton_entropy_gate_enabled",
        )
        available = _metric(
            row,
            "counterfactual_proposal_singleton_entropy_gate_available",
        )
        anchor_available = _metric(
            row,
            "counterfactual_proposal_anchor_available",
        )
        proposal_path_reached = (
            (_finite(available) and float(available) > 0)
            or (_finite(anchor_available) and float(anchor_available) > 0)
        )
        if proposal_path_reached and not (
            _finite(enabled) and math.isclose(float(enabled), 1.0)
        ):
            violations.append(
                f"{label}: proposal path reached with singleton entropy gate "
                f"disabled at line {line_number}"
            )
        admitted = _metric(
            row,
            "counterfactual_proposal_admitted_new_outcomes",
        )
        if _finite(admitted):
            admitted_count = int(admitted)
            maximum_admitted = max(maximum_admitted, admitted_count)
            if admitted_count > 1:
                violations.append(
                    f"{label}: admitted {admitted_count} outcomes in one group"
                )
            if admitted_count > 0:
                interventions += 1
                support = _metric(
                    row,
                    "counterfactual_proposal_"
                    "singleton_entropy_gate_known_support",
                )
                active = _metric(
                    row,
                    "counterfactual_proposal_"
                    "singleton_entropy_gate_active",
                )
                warmup = _metric(
                    row,
                    "counterfactual_proposal_"
                    "singleton_entropy_gate_warmup_complete",
                )
                below_reference = _metric(
                    row,
                    "counterfactual_proposal_"
                    "singleton_entropy_gate_below_reference",
                )
                inverse_multiplier = _metric(
                    row,
                    "counterfactual_proposal_"
                    "singleton_entropy_gate_inverse_multiplier",
                )
                event_available = _metric(
                    row,
                    "counterfactual_proposal_"
                    "singleton_entropy_gate_available",
                )
                if not (_finite(support) and int(support) == 1):
                    violations.append(
                        f"{label}: admission without singleton support"
                    )
                if not (_finite(active) and math.isclose(float(active), 1.0)):
                    violations.append(
                        f"{label}: admission without entropy-collapse gate"
                    )
                for gate_name, value in (
                    ("gate enabled", enabled),
                    ("gate available", event_available),
                    ("entropy warmup", warmup),
                    ("entropy below reference", below_reference),
                ):
                    if not (
                        _finite(value)
                        and math.isclose(float(value), 1.0)
                    ):
                        violations.append(
                            f"{label}: admission without {gate_name}"
                        )
                if not (
                    _finite(inverse_multiplier)
                    and float(inverse_multiplier) > 1.0
                ):
                    violations.append(
                        f"{label}: admission without inverse multiplier > 1"
                    )
                intervention_events.append(
                    {
                        "step": int(step) if _finite(step) else None,
                        "line_number": line_number,
                        "admitted_new_outcomes": admitted_count,
                        "known_support_before_admission": (
                            int(support) if _finite(support) else None
                        ),
                        "gate_enabled": (
                            float(enabled) if _finite(enabled) else None
                        ),
                        "gate_available": (
                            float(event_available)
                            if _finite(event_available)
                            else None
                        ),
                        "warmup_complete": (
                            float(warmup) if _finite(warmup) else None
                        ),
                        "entropy_below_reference": (
                            float(below_reference)
                            if _finite(below_reference)
                            else None
                        ),
                        "inverse_multiplier": (
                            float(inverse_multiplier)
                            if _finite(inverse_multiplier)
                            else None
                        ),
                        "gate_active": (
                            float(active) if _finite(active) else None
                        ),
                        "conditioned_rows_sent_to_ppo": _metric(
                            row,
                            "counterfactual_proposal_"
                            "conditioned_rows_sent_to_ppo",
                        ),
                        "transform_rows_sent_to_ppo": _metric(
                            row,
                            "counterfactual_proposal_"
                            "transform_rows_sent_to_ppo",
                        ),
                        "coefficient_projection_active": _metric(
                            row,
                            "semantic_shannon_success_conditioned_signed_"
                            "open_set_projection_active",
                        ),
                    }
                )
    return (
        {
            "latest_step": latest_step,
            "metric_records": records,
            "interventions": interventions,
            "maximum_admitted_per_group": maximum_admitted,
            "objective_contract_records": objective_contract_records,
            "intervention_events": intervention_events,
        },
        violations,
    )


def main() -> None:
    violations: list[str] = []
    if not IDENTITY.is_file():
        payload = {
            "schema": "e67_corrected_same_objective_actuator_ablation_audit_v1",
            "status": "not_submitted",
            "summary": {"expected_runs": 12, "materialized_runs": 0},
            "domains": {},
            "violations": [],
        }
    else:
        identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
        if (
            identity.get("schema")
            != "e67_corrected_same_objective_actuator_ablation_v1"
        ):
            violations.append("identity schema mismatch")
        if identity.get("source_hash") != EXPECTED_SOURCE_HASH:
            violations.append("frozen source hash mismatch")
        if identity.get("objective_contract") != {
            "online_canonical_bank_alpha": 0.0,
            "online_canonical_novelty_beta": 0.5,
            "runtime_assertion": True,
            "only_algorithmic_delta_from_e66": [
                "counterfactual_proposals",
                "singleton_entropy_gate",
            ],
        }:
            violations.append("same-objective runtime contract mismatch")
        if not PLACEMENT_RECORD.is_file():
            violations.append("zero-step placement amendment record missing")
        else:
            placement = json.loads(
                PLACEMENT_RECORD.read_text(encoding="utf-8")
            )
            if (
                placement.get("schema")
                != "e67_zero_step_placement_amendment_v1"
            ):
                violations.append("placement amendment schema mismatch")
            if placement.get("amendment_sha256") != _digest(
                PLACEMENT_AMENDMENT
            ):
                violations.append("placement amendment hash mismatch")
            if placement.get("script_sha256") != _digest(
                PLACEMENT_SCRIPT
            ):
                violations.append("placement amendment script hash mismatch")
            if placement.get("scientific_settings_changed") is not False:
                violations.append(
                    "placement amendment changed scientific settings"
                )
        for key, path in (
            ("protocol_sha256", PROTOCOL),
            ("launcher_sha256", LAUNCHER),
        ):
            if identity.get(key) != _digest(path):
                violations.append(f"{key} mismatch")
        domains: dict[str, Any] = {}
        terminal_runs = 0
        materialized_runs = 0
        metric_runs = 0
        total_interventions = 0
        for domain, expected_step in EXPECTED_STEPS.items():
            expected_rows = identity["jobs"].get(domain, [])
            manifest_path = ROOT / (
                "var/artifacts/"
                f"{expected_rows[0]['run_stamp'].rsplit('_' + identity['arm'] + '_s', 1)[0]}"
                "_comparative_jobs.tsv"
            )
            if _digest(manifest_path) != identity["manifest_sha256"][domain]:
                violations.append(f"{domain}: manifest hash mismatch")
            rows = list(csv.DictReader(manifest_path.open(), delimiter="\t"))
            runs = []
            for row, expected in zip(rows, expected_rows, strict=True):
                normalized = {
                    "arm": row["arm"],
                    "seed": int(row["seed"]),
                    "job_id": int(row["job_id"]),
                    "run_stamp": row["run_stamp"],
                }
                if normalized != expected:
                    violations.append(f"{domain}: manifest identity mismatch")
                label = f"{domain}/s{normalized['seed']}/j{normalized['job_id']}"
                run_dir = _run_dir(
                    normalized["run_stamp"],
                    normalized["job_id"],
                )
                metrics = {
                    "latest_step": -1,
                    "metric_records": 0,
                    "interventions": 0,
                    "maximum_admitted_per_group": 0,
                    "objective_contract_records": 0,
                    "intervention_events": [],
                }
                if run_dir is not None:
                    materialized_runs += 1
                    metrics_path = run_dir / "train_metrics.jsonl"
                    if metrics_path.is_file():
                        metric_runs += 1
                        metrics, metric_violations = _scan_metrics(
                            metrics_path,
                            label=label,
                        )
                        violations.extend(metric_violations)
                total_interventions += metrics["interventions"]
                interruptions = 0
                for suffix in ("out", "err"):
                    failures, count = _scan_log(
                        ROOT
                        / (
                            "var/artifacts/logs/"
                            f"xdr_train-{normalized['job_id']}.{suffix}"
                        )
                    )
                    interruptions += count
                    violations.extend(
                        f"{label}: uncaught {failure!r}"
                        for failure in failures
                    )
                runtime_contract_verified = False
                stdout_path = (
                    ROOT
                    / "var/artifacts/logs/"
                    f"xdr_train-{normalized['job_id']}.out"
                )
                if stdout_path.is_file():
                    stdout_text = stdout_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                    runtime_contract_verified = (
                        "online_canonical_bank_alpha=0 "
                        "novelty_beta=0.50" in stdout_text
                        and "online canonical bank enabled: alpha=0 "
                        "beta=0.5" in stdout_text
                    )
                if (
                    metrics["metric_records"] > 0
                    and not runtime_contract_verified
                ):
                    violations.append(
                        f"{label}: runtime novelty-beta contract unverified"
                    )
                terminal = metrics["latest_step"] >= expected_step
                terminal_runs += int(terminal)
                runs.append(
                    {
                        **normalized,
                        **metrics,
                        "expected_step": expected_step,
                        "training_passes": (
                            metrics["latest_step"] / (expected_step / 12)
                            if metrics["latest_step"] >= 0
                            else None
                        ),
                        "terminal": terminal,
                        "runtime_objective_contract_verified": (
                            runtime_contract_verified
                        ),
                        "infrastructure_interruptions": interruptions,
                    }
                )
            domains[domain] = {"runs": runs}
        status = (
            "fail"
            if violations
            else "pass"
            if terminal_runs == 12
            else "in_progress"
        )
        payload = {
            "schema": "e67_corrected_same_objective_actuator_ablation_audit_v1",
            "status": status,
            "summary": {
                "expected_runs": 12,
                "materialized_runs": materialized_runs,
                "metric_runs": metric_runs,
                "terminal_runs": terminal_runs,
                "entropy_gated_interventions": total_interventions,
            },
            "domains": domains,
            "violations": sorted(set(violations)),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUT.name}.", dir=OUT.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUT)
    print(
        f"[e67-audit] status={payload['status']} "
        f"summary={payload['summary']} "
        f"violations={len(payload['violations'])}"
    )


if __name__ == "__main__":
    main()
