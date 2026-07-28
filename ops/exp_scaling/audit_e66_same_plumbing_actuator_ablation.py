#!/usr/bin/env python3
"""Fail-closed audit for E66's same-plumbing actuator-off controls."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


EXP_SCALING_DIR = Path(__file__).resolve().parent
if str(EXP_SCALING_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_SCALING_DIR))

from e66_e68_mathir_seed_overflow_contract import (  # noqa: E402
    is_registered_seed_overflow,
    load_recovery_contract,
)


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e66_same_plumbing_actuator_ablation_identity.json"
PROTOCOL = (
    ROOT / "paper/preregistration/e66_same_plumbing_actuator_ablation_05b.md"
)
LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_e66_same_plumbing_actuator_ablation.sh"
)
RESUME_AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e66_same_family_resume_placement_amendment_20260727.md"
)
RESUME_SCRIPT = (
    ROOT / "ops/exp_scaling/amend_e66_same_family_resume_placement.sh"
)
RESUME_RECORD = (
    ROOT
    / "var/artifacts/e66_same_family_resume_placement_amendment.json"
)
GRAPH_DRAIN_AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e66_e68_graph_a6000_drain_recovery_amendment_20260727.md"
)
GRAPH_DRAIN_SCRIPT = (
    ROOT
    / "ops/exp_scaling/"
    "amend_e66_e68_graph_a6000_drain_recovery.sh"
)
GRAPH_DRAIN_RECORD = (
    ROOT
    / "var/artifacts/"
    "e66_e68_graph_a6000_drain_recovery_amendment.json"
)
GRAPH_CONTAMINATION_AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e66_e68_graph_a6000_contamination_recovery_amendment_20260727.md"
)
GRAPH_CONTAMINATION_SCRIPT = (
    ROOT
    / "ops/exp_scaling/"
    "amend_e66_e68_graph_a6000_contamination_recovery.sh"
)
GRAPH_CONTAMINATION_RECORD = (
    ROOT
    / "var/artifacts/"
    "e66_e68_graph_a6000_contamination_recovery_amendment.json"
)
OUT = ROOT / "var/artifacts/e66_same_plumbing_actuator_ablation_audit_latest.json"
SOURCE_HASH = "f6147daacbfdde22e0e9d5fab6fc45b41017d4dbf5e2848f827923ddf7828a7f"
EXECUTION_HASH = "ff2a4f37d8653ba1a5888538d22743269c73d541b7bc6c8db214cedaec9c5d8f"
EXPECTED_STEPS = {
    "graph_coloring": 192 * 12,
    "countdown": 384 * 12,
    "python_factor": 384 * 12,
    "mathir": 384 * 12,
}
MANIFESTS = {
    "graph_coloring": ROOT
    / "var/artifacts/gce66_same_plumbing_control_05b_12ep_comparative_jobs.tsv",
    "countdown": ROOT
    / "var/artifacts/cde66_same_plumbing_control_05b_12ep_comparative_jobs.tsv",
    "python_factor": ROOT
    / "var/artifacts/pye66_same_plumbing_control_05b_12ep_comparative_jobs.tsv",
    "mathir": ROOT
    / "var/artifacts/mie66_same_plumbing_control_05b_12ep_comparative_jobs.tsv",
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


def _run_dir(run_stamp: str, job_id: int) -> Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _scan_log(
    path: Path,
    *,
    allow_registered_seed_overflow: bool = False,
) -> tuple[list[str], int]:
    if not path.is_file():
        return [], 0
    text = path.read_text(encoding="utf-8", errors="replace")
    failures: list[str] = []
    interruptions = 0
    for match in UNCAUGHT.finditer(text):
        nearby = text[max(0, match.start() - 16000) : match.start()]
        if (
            allow_registered_seed_overflow
            and "Traceback" in match.group(0)
            and is_registered_seed_overflow(text, match.end())
        ):
            interruptions += 1
        elif "Traceback" in match.group(0) and SIGTERM.search(nearby):
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
    proposal_activity_records = 0
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
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    violations.append(
                        f"{label}: non-finite {key} at line {line_number}"
                    )
            if (
                key.endswith(("_nan", "_inf"))
                and _finite(value)
                and float(value) > 0
            ):
                violations.append(
                    f"{label}: nonfinite diagnostic {key} at line {line_number}"
                )
        for key, value in row.items():
            if (
                key.endswith(
                    (
                        "gold_support_feedback",
                        "desired_mode_count_feedback",
                        "eval_feedback",
                        "rows_sent_to_ppo",
                        "projection_active",
                    )
                )
                and _finite(value)
                and not _zero(value)
            ):
                violations.append(
                    f"{label}: forbidden nonzero {key} at line {line_number}"
                )
        proposal_values = [
            value
            for key, value in row.items()
            if "counterfactual_proposal" in key
            and _finite(value)
            and not _zero(value)
        ]
        if proposal_values:
            proposal_activity_records += 1
            violations.append(
                f"{label}: actuator-off control has proposal activity at "
                f"line {line_number}"
            )
    return (
        {
            "latest_step": latest_step,
            "metric_records": records,
            "proposal_activity_records": proposal_activity_records,
        },
        violations,
    )


def main() -> None:
    if not IDENTITY.is_file():
        payload = {
            "schema": "e66_same_plumbing_actuator_ablation_audit_v1",
            "status": "not_submitted",
            "summary": {
                "expected_runs": 12,
                "materialized_runs": 0,
                "metric_runs": 0,
                "terminal_runs": 0,
            },
            "domains": {},
            "violations": [],
        }
    else:
        identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
        violations: list[str] = []
        if identity.get("schema") != "e66_same_plumbing_actuator_ablation_v1":
            violations.append("identity schema mismatch")
        if identity.get("source_hash") != SOURCE_HASH:
            violations.append("frozen E65R1 source hash mismatch")
        if identity.get("execution_surface_hash") != EXECUTION_HASH:
            violations.append("frozen E65R1 execution hash mismatch")
        if identity.get("protocol_sha256") != _digest(PROTOCOL):
            violations.append("protocol hash mismatch")
        if identity.get("launcher_sha256") != _digest(LAUNCHER):
            violations.append("launcher hash mismatch")
        if identity.get("arm") != "verified_first_global_replay_canonical":
            violations.append("control is not literal E58")
        expected_contract = {
            "replicated_freeform_sampling": True,
            "local_actor_weight_sync": True,
            "counterfactual_proposals": False,
            "singleton_entropy_gate": False,
        }
        if identity.get("execution_contract") != expected_contract:
            violations.append("same-plumbing execution contract mismatch")
        recovery_mappings, recovery_violations = load_recovery_contract()
        violations.extend(recovery_violations)
        mathir_recoveries = recovery_mappings["e66"]
        pre_amendment_latest_steps: dict[str, int] = {}
        if not RESUME_RECORD.is_file():
            violations.append("same-family resume amendment record missing")
        else:
            resume_amendment = json.loads(
                RESUME_RECORD.read_text(encoding="utf-8")
            )
            pre_amendment_latest_steps = resume_amendment.get(
                "pre_amendment_latest_steps", {}
            )
            if (
                resume_amendment.get("schema")
                != "e66_same_family_resume_placement_amendment_v1"
                or resume_amendment.get("amendment_sha256")
                != _digest(RESUME_AMENDMENT)
                or resume_amendment.get("script_sha256")
                != _digest(RESUME_SCRIPT)
                or resume_amendment.get("scientific_settings_changed")
                is not False
                or resume_amendment.get("unaffected_running_jobs")
                != [30128403, 30128404, 30128405]
                or sorted(
                    job_id
                    for values in resume_amendment.get(
                        "affected_jobs", {}
                    ).values()
                    for job_id in values
                )
                != list(range(30128394, 30128403))
                or set(pre_amendment_latest_steps)
                != {
                    str(job_id)
                    for job_id in range(30128394, 30128403)
                }
            ):
                violations.append(
                    "same-family resume amendment contract mismatch"
                )
        graph_drain_latest_steps: dict[str, int] = {}
        if not GRAPH_DRAIN_RECORD.is_file():
            violations.append("Graph A6000 drain-recovery record missing")
        else:
            graph_drain = json.loads(
                GRAPH_DRAIN_RECORD.read_text(encoding="utf-8")
            )
            graph_drain_latest_steps = graph_drain.get(
                "pre_amendment_latest_steps",
                {},
            )
            if (
                graph_drain.get("schema")
                != "e66_e68_graph_a6000_drain_recovery_amendment_v1"
                or graph_drain.get("amendment_sha256")
                != _digest(GRAPH_DRAIN_AMENDMENT)
                or graph_drain.get("script_sha256")
                != _digest(GRAPH_DRAIN_SCRIPT)
                or graph_drain.get("scientific_settings_changed") is not False
                or graph_drain.get("paired_arms_moved_together") is not True
                or graph_drain.get("affected_jobs")
                != {
                    "e66_graph_coloring": [
                        30128394,
                        30128395,
                        30128396,
                    ],
                    "e68_graph_coloring": [
                        30130469,
                        30130470,
                        30130471,
                    ],
                }
                or graph_drain.get("mutation")
                != {
                    "account": "allcs",
                    "partition": "lowprio",
                    "nodes": ["node205", "node206", "node207"],
                    "gres": "gpu:a6000:1",
                }
                or set(graph_drain_latest_steps)
                != {
                    str(job_id)
                    for job_id in (
                        30128394,
                        30128395,
                        30128396,
                        30130469,
                        30130470,
                        30130471,
                    )
                }
            ):
                violations.append(
                    "Graph A6000 drain-recovery contract mismatch"
                )
        graph_contamination_latest_steps: dict[str, int] = {}
        if not GRAPH_CONTAMINATION_RECORD.is_file():
            violations.append(
                "Graph A6000 contamination-recovery record missing"
            )
        else:
            graph_contamination = json.loads(
                GRAPH_CONTAMINATION_RECORD.read_text(encoding="utf-8")
            )
            graph_contamination_latest_steps = graph_contamination.get(
                "pre_amendment_latest_steps",
                {},
            )
            registered = graph_contamination.get(
                "registered_contamination",
                {},
            )
            if (
                graph_contamination.get("schema")
                != (
                    "e66_e68_graph_a6000_contamination_recovery_"
                    "amendment_v1"
                )
                or graph_contamination.get("amendment_sha256")
                != _digest(GRAPH_CONTAMINATION_AMENDMENT)
                or graph_contamination.get("script_sha256")
                != _digest(GRAPH_CONTAMINATION_SCRIPT)
                or graph_contamination.get("scientific_settings_changed")
                is not False
                or graph_contamination.get("paired_arms_moved_together")
                is not True
                or graph_contamination.get("affected_jobs")
                != {
                    "e66_graph_coloring": [
                        30128394,
                        30128395,
                        30128396,
                    ],
                    "e68_graph_coloring": [
                        30130469,
                        30130470,
                        30130471,
                    ],
                }
                or graph_contamination.get("mutation")
                != {
                    "account": "mltheory",
                    "partition": "pvl-lowprio",
                    "nodes": ["node103", "node104", "node805"],
                    "gres": "gpu:a6000:1",
                }
                or set(graph_contamination_latest_steps)
                != {
                    str(job_id)
                    for job_id in (
                        30128394,
                        30128395,
                        30128396,
                        30130469,
                        30130470,
                        30130471,
                    )
                }
                or registered.get("job_ids") != [30130469, 30130470]
                or registered.get("node") != "node206"
                or registered.get("optimizer_metric_records")
                != {"30130469": 0, "30130470": 0}
            ):
                violations.append(
                    "Graph A6000 contamination-recovery contract mismatch"
                )
        domains: dict[str, Any] = {}
        materialized_runs = 0
        metric_runs = 0
        terminal_runs = 0
        recovery_materialized_runs = 0
        for domain, expected_step in EXPECTED_STEPS.items():
            manifest = MANIFESTS[domain]
            if identity.get("manifest_sha256", {}).get(domain) != _digest(
                manifest
            ):
                violations.append(f"{domain}: manifest hash mismatch")
            with manifest.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            expected_rows = identity.get("jobs", {}).get(domain, [])
            if len(rows) != 3 or len(expected_rows) != 3:
                violations.append(f"{domain}: expected exactly three seeds")
            runs: list[dict[str, Any]] = []
            for row, expected in zip(rows, expected_rows, strict=True):
                normalized = {
                    "arm": row["arm"],
                    "seed": int(row["seed"]),
                    "job_id": int(row["job_id"]),
                    "run_stamp": row["run_stamp"],
                }
                if normalized != expected:
                    violations.append(f"{domain}: manifest identity mismatch")
                label = (
                    f"{domain}/s{normalized['seed']}/j{normalized['job_id']}"
                )
                metrics = {
                    "latest_step": -1,
                    "metric_records": 0,
                    "proposal_activity_records": 0,
                }
                run_dir = _run_dir(
                    normalized["run_stamp"],
                    normalized["job_id"],
                )
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
                recovery = (
                    mathir_recoveries.get(normalized["job_id"])
                    if domain == "mathir"
                    else None
                )
                recovery_run_dir = None
                recovery_metrics = {
                    "latest_step": -1,
                    "metric_records": 0,
                    "proposal_activity_records": 0,
                }
                if recovery is not None:
                    recovery_run_dir = _run_dir(
                        normalized["run_stamp"],
                        int(recovery["recovery_job_id"]),
                    )
                    if recovery_run_dir is not None:
                        recovery_materialized_runs += 1
                        recovery_metrics_path = (
                            recovery_run_dir / "train_metrics.jsonl"
                        )
                        if recovery_metrics_path.is_file():
                            recovery_metrics, recovery_metric_violations = (
                                _scan_metrics(
                                    recovery_metrics_path,
                                    label=(
                                        f"{label}/recovery"
                                        f"j{recovery['recovery_job_id']}"
                                    ),
                                )
                            )
                            violations.extend(recovery_metric_violations)
                    metrics = {
                        "latest_step": max(
                            metrics["latest_step"],
                            recovery_metrics["latest_step"],
                        ),
                        "metric_records": (
                            metrics["metric_records"]
                            + recovery_metrics["metric_records"]
                        ),
                        "proposal_activity_records": (
                            metrics["proposal_activity_records"]
                            + recovery_metrics["proposal_activity_records"]
                        ),
                    }
                pre_amendment_step = int(
                    pre_amendment_latest_steps.get(
                        str(normalized["job_id"]),
                        -1,
                    )
                )
                pre_amendment_step = max(
                    pre_amendment_step,
                    int(
                        graph_drain_latest_steps.get(
                            str(normalized["job_id"]),
                            -1,
                        )
                    ),
                )
                pre_amendment_step = max(
                    pre_amendment_step,
                    int(
                        graph_contamination_latest_steps.get(
                            str(normalized["job_id"]),
                            -1,
                        )
                    ),
                )
                if metrics["latest_step"] < pre_amendment_step:
                    violations.append(
                        f"{label}: trace regressed below pre-amendment step "
                        f"{pre_amendment_step}"
                    )
                interruptions = 0
                for suffix in ("out", "err"):
                    failures, count = _scan_log(
                        ROOT
                        / (
                            "var/artifacts/logs/"
                            f"xdr_train-{normalized['job_id']}.{suffix}"
                        ),
                        allow_registered_seed_overflow=(
                            recovery is not None
                        ),
                    )
                    interruptions += count
                    violations.extend(
                        f"{label}: uncaught {failure!r}"
                        for failure in failures
                    )
                    if recovery is not None:
                        recovery_failures, recovery_count = _scan_log(
                            ROOT
                            / (
                                "var/artifacts/logs/"
                                f"xdr_train-{recovery['recovery_job_id']}."
                                f"{suffix}"
                            )
                        )
                        interruptions += recovery_count
                        violations.extend(
                            f"{label}/recovery"
                            f"j{recovery['recovery_job_id']}: "
                            f"uncaught {failure!r}"
                            for failure in recovery_failures
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
                        "infrastructure_interruptions": interruptions,
                        "recovery_job_id": (
                            int(recovery["recovery_job_id"])
                            if recovery is not None
                            else None
                        ),
                        "recovery_run_dir": (
                            str(recovery_run_dir.relative_to(ROOT))
                            if recovery_run_dir is not None
                            else None
                        ),
                        "recovery_latest_step": (
                            recovery_metrics["latest_step"]
                            if recovery is not None
                            else None
                        ),
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
            "schema": "e66_same_plumbing_actuator_ablation_audit_v1",
            "status": status,
            "summary": {
                "expected_runs": 12,
                "materialized_runs": materialized_runs,
                "metric_runs": metric_runs,
                "terminal_runs": terminal_runs,
                "recovery_materialized_runs": recovery_materialized_runs,
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
        f"[e66-audit] status={payload['status']} "
        f"summary={payload['summary']} "
        f"violations={len(payload['violations'])}"
    )


if __name__ == "__main__":
    main()
