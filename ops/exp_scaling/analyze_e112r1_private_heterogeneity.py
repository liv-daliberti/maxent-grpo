#!/usr/bin/env python3
"""Diagnose the frozen E112-R1 private endpoint subset without new unblinding.

This analysis joins the already-frozen 2026-08-23 private endpoint payload to
training-only mechanism telemetry.  It deliberately does not open any
additional evaluation result or sampled-draw file, compute pooled effects,
fit a cross-domain association, or authorize a change to the live campaign.

The output stays under ``var/artifacts/private_interim`` and makes two issues
that are easy to hide in an endpoint-only plot explicit:

* ``distinct@8 - pass@8`` can fall when an arm rescues previously unsolved
  prompts with one correct mode, even while raw ``distinct@8`` rises; and
* the non-Pantry E112-R1 treatment uses replicated free-form sampling and
  local actor synchronization whereas its historical ReplayDr comparators use
  the ordinary collector path.  The repository's E66 protocol already treats
  that request-stream difference as a separate intervention.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / (
    "var/artifacts/e112r1_private_interim_unblinding_freeze_20260823.json"
)
ENDPOINTS = ROOT / (
    "var/artifacts/private_interim/"
    "e112r1_subset_endpoint_effects_20260823.json"
)
TREATMENT_LEDGER = ROOT / (
    "var/artifacts/e112r1_verified_support_discovery_full_three_scale_jobs.json"
)
COMPARATOR_LEDGERS = (
    ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json",
    ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json",
    ROOT / "var/artifacts/e109_repaired_python_replay_comparators_jobs.json",
)
DEFAULT_OUTPUT = ROOT / (
    "var/artifacts/private_interim/"
    "e112r1_heterogeneity_diagnostic_20260823.json"
)
PRIVATE_LABEL = "PRIVATE EXPLORATORY INTERIM — NOT FOR PAPER OR SELECTION"
TARGET_STEPS = 3072

ENV_DEFAULTS = {
    "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING": "0",
    "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC": "0",
    "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING": "0",
    "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING": "0",
    "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "0",
    "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
}
PAIRING_FIELDS = (
    "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING",
    "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC",
    "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING",
    "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING",
    "OAT_ZERO_TRAIN_BATCH_SIZE",
    "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE",
    "OAT_ZERO_ROLLOUT_BATCH_SIZE",
    "OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE",
    "OAT_ZERO_NUM_SAMPLES",
    "OAT_ZERO_SOURCE_ROOT",
    "OAT_ZERO_OPS_SNAPSHOT_ROOT",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def finite(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def metric(row: dict[str, Any], suffix: str) -> float | None:
    for prefix in ("train/", "actor/"):
        value = finite(row.get(prefix + suffix))
        if value is not None:
            return value
    return None


def metric_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("debug_job*/train_metrics.jsonl"))


def unique_train_rows(paths: Iterable[Path]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Read the last complete training row for each optimizer step."""

    by_step: dict[int, dict[str, Any]] = {}
    invalid: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    invalid.append(f"{relative(path)}:{line_number}")
                    continue
                if not isinstance(row, dict) or not any(
                    key.startswith("train/") for key in row
                ):
                    continue
                step_value = finite(
                    row.get(
                        "misc/global_step",
                        row.get("trainer/global_step", row.get("trainer/step")),
                    )
                )
                if step_value is None:
                    continue
                step = int(step_value)
                if step > 0:
                    by_step[step] = row
    return by_step, invalid


def mean_present(rows: Iterable[dict[str, Any]], suffix: str) -> float | None:
    values = [value for row in rows if (value := metric(row, suffix)) is not None]
    return sum(values) / len(values) if values else None


def fraction(rows: list[dict[str, Any]], predicate: Any) -> float | None:
    if not rows:
        return None
    return sum(bool(predicate(row)) for row in rows) / len(rows)


def latest_present(rows: list[dict[str, Any]], suffix: str) -> float | None:
    for row in reversed(rows):
        value = metric(row, suffix)
        if value is not None:
            return value
    return None


def mechanism_summary(run_dir: Path) -> dict[str, Any]:
    paths = metric_paths(run_dir)
    by_step, invalid = unique_train_rows(paths)
    rows = [by_step[step] for step in sorted(by_step)]
    prefix = "semantic_shannon_success_conditioned_verified_support_"
    semantic_rms = mean_present(rows, prefix + "effective_advantage_rms")
    task_rms = mean_present(rows, "semantic_shannon_separate_base_advantage_rms")
    realized_ratio = None
    if semantic_rms is not None and task_rms is not None and task_rms > 0.0:
        realized_ratio = semantic_rms / task_rms

    available_modes = [
        value
        for row in rows
        if (value := metric(row, "canonical_replay_available_modes")) is not None
    ]
    proposal_groups = sum(
        metric(row, "counterfactual_proposal_groups_generated") or 0.0
        for row in rows
    )
    proposal_rows = sum(
        metric(row, "counterfactual_proposal_rows_generated") or 0.0
        for row in rows
    )
    admissions = latest_present(rows, "counterfactual_proposal_cumulative_new_outcomes")
    admission_rate = None
    if admissions is not None and proposal_groups > 0.0:
        admission_rate = admissions / proposal_groups

    all_one = fraction(
        rows, lambda row: (metric(row, "all_one_rewards_count") or 0.0) > 0.0
    )
    all_zero = fraction(
        rows, lambda row: (metric(row, "all_zero_rewards_count") or 0.0) > 0.0
    )
    mixed = None
    if all_one is not None and all_zero is not None:
        mixed = max(0.0, 1.0 - all_one - all_zero)

    return {
        "metric_paths": [relative(path) for path in paths],
        "unique_optimizer_steps": len(rows),
        "last_optimizer_step": max(by_step, default=0),
        "missing_optimizer_steps": [
            step for step in range(1, TARGET_STEPS + 1) if step not in by_step
        ],
        "invalid_json_rows": invalid,
        "reward_regime": {
            "all_one_group_fraction": all_one,
            "all_zero_group_fraction": all_zero,
            "mixed_reward_group_fraction": mixed,
        },
        "semantic": {
            "mean_effective_advantage_rms": semantic_rms,
            "mean_task_advantage_rms": task_rms,
            "realized_semantic_to_task_rms_ratio": realized_ratio,
            "mean_eligible_fraction": mean_present(rows, prefix + "eligible_fraction"),
            "active_update_fraction": fraction(
                rows,
                lambda row: (metric(row, prefix + "effective_advantage_rms") or 0.0)
                > 0.0,
            ),
            "both_sign_update_fraction": fraction(
                rows,
                lambda row: (
                    (metric(row, prefix + "effective_advantage_min") or 0.0) < 0.0
                    < (metric(row, prefix + "effective_advantage_max") or 0.0)
                ),
            ),
            "mean_verified_support_size": mean_present(
                rows, prefix + "verified_support_size_mean"
            ),
            "mean_external_verified_support_size": mean_present(
                rows, prefix + "external_verified_support_size_mean"
            ),
            "mean_external_support_nonempty_group_fraction": mean_present(
                rows, prefix + "external_verified_support_nonempty_group_fraction"
            ),
            "mean_support_at_least_two_eligible_fraction": mean_present(
                rows,
                prefix + "verified_support_at_least_two_eligible_fraction",
            ),
        },
        "proposal": {
            "groups_generated": proposal_groups,
            "rows_generated": proposal_rows,
            "cumulative_admissions": admissions,
            "admissions_per_generated_group": admission_rate,
        },
        "replay": {
            "mean_available_modes": (
                sum(available_modes) / len(available_modes)
                if available_modes
                else None
            ),
            "capacity_hit_fraction": (
                sum(value >= 16.0 for value in available_modes)
                / len(available_modes)
                if available_modes
                else None
            ),
            "mean_normalized_model_entropy": mean_present(
                rows, "canonical_replay_normalized_model_entropy"
            ),
            "mean_cross_entropy_excess": mean_present(
                rows, "canonical_replay_cross_entropy_excess"
            ),
        },
        "retention": {
            "tracked_admissions": latest_present(
                rows, "canonical_replay_proposal_retention_tracked_admissions"
            ),
            "rollout_conversion_fraction": latest_present(
                rows,
                "canonical_replay_proposal_retention_rollout_conversion_fraction",
            ),
            "score_retained_fraction": latest_present(
                rows,
                "canonical_replay_proposal_retention_score_retained_fraction",
            ),
            "joint_retained_fraction": latest_present(
                rows,
                "canonical_replay_proposal_retention_joint_retained_fraction",
            ),
            "mean_token_logprob_drop": latest_present(
                rows,
                "canonical_replay_proposal_retention_mean_logprob_drop_mean",
            ),
            "sequence_logprob_drop": latest_present(
                rows,
                "canonical_replay_proposal_retention_sequence_logprob_drop_mean",
            ),
        },
    }


def parse_env(record: str) -> dict[str, str]:
    result = dict(ENV_DEFAULTS)
    for match in re.finditer(r"(?:^|,)(OAT_ZERO_[A-Z0-9_]+)=([^,\s]+)", record):
        result[match.group(1)] = match.group(2)
    return result


def pairing_summary(treatment: dict[str, Any], comparator: dict[str, Any]) -> dict[str, Any]:
    treatment_env = parse_env(str(treatment.get("held_scheduler_record", "")))
    comparator_env = parse_env(str(comparator.get("held_scheduler_record", "")))
    values = {
        field: {
            "treatment": treatment_env.get(field),
            "comparator": comparator_env.get(field),
            "matched": treatment_env.get(field) == comparator_env.get(field),
        }
        for field in PAIRING_FIELDS
    }
    mismatches = [field for field, record in values.items() if not record["matched"]]
    sampler_fields = (
        "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING",
        "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC",
        "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING",
        "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING",
    )
    return {
        "fields": values,
        "mismatched_fields": mismatches,
        "sampler_path_matched": all(values[field]["matched"] for field in sampler_fields),
        "source_snapshot_matched": values["OAT_ZERO_SOURCE_ROOT"]["matched"],
        "ops_snapshot_matched": values["OAT_ZERO_OPS_SNAPSHOT_ROOT"]["matched"],
    }


def sign(value: float, *, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def endpoint_decomposition(row: dict[str, Any]) -> dict[str, Any]:
    pass_effect = float(row["effect"]["pass8"])
    adjusted_effect = float(row["effect"]["adjusted_breadth8"])
    distinct_effect = pass_effect + adjusted_effect
    treatment = row["treatment"]
    comparator = row["comparator"]
    return {
        "pass8_effect": pass_effect,
        "adjusted_breadth8_effect": adjusted_effect,
        "distinct8_effect": distinct_effect,
        "effect_signs": {
            "pass8": sign(pass_effect),
            "adjusted_breadth8": sign(adjusted_effect),
            "distinct8": sign(distinct_effect),
        },
        "treatment": {
            "pass8": float(treatment["pass8"]),
            "distinct8": float(treatment["distinct8"]),
            "adjusted_breadth8": float(treatment["distinct8"])
            - float(treatment["pass8"]),
        },
        "comparator": {
            "pass8": float(comparator["pass8"]),
            "distinct8": float(comparator["distinct8"]),
            "adjusted_breadth8": float(comparator["distinct8"])
            - float(comparator["pass8"]),
        },
        "interpretation": (
            "accuracy_rescue_with_raw_breadth_gain"
            if pass_effect > 0.0 and adjusted_effect < 0.0 and distinct_effect > 0.0
            else "breadth_gain_with_accuracy_tradeoff"
            if pass_effect < 0.0 and adjusted_effect > 0.0
            else "pareto_positive"
            if pass_effect > 0.0 and adjusted_effect > 0.0
            else "breadth_only_or_neutral_accuracy"
            if adjusted_effect > 0.0 and math.isclose(pass_effect, 0.0, abs_tol=1e-12)
            else "mixed_or_null"
        ),
    }


def run_index(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in payload.get("runs", []):
        job_id = int(row["job_id"])
        if job_id in result:
            raise RuntimeError(f"duplicate job ID {job_id}")
        result[job_id] = row
    return result


def build() -> dict[str, Any]:
    freeze = load_object(FREEZE)
    endpoints = load_object(ENDPOINTS)
    treatment_ledger = load_object(TREATMENT_LEDGER)
    if freeze.get("terminal_cells") != 33:
        raise RuntimeError("expected the frozen 33-cell private membership")
    if endpoints.get("frozen_terminal_cells") != 33:
        raise RuntimeError("endpoint payload is not the frozen 33-cell subset")
    if endpoints.get("freeze_sha256") != sha256(FREEZE):
        raise RuntimeError("endpoint payload does not bind the selected freeze")
    if endpoints.get("label") != PRIVATE_LABEL:
        raise RuntimeError("endpoint payload lost its private label")
    if endpoints.get("campaign_mutation_allowed") is not False:
        raise RuntimeError("endpoint payload permits campaign mutation")
    if endpoints.get("paper_efficacy_output_allowed") is not False:
        raise RuntimeError("endpoint payload permits paper efficacy output")

    treatments = run_index(treatment_ledger)
    comparators: dict[int, dict[str, Any]] = {}
    for path in COMPARATOR_LEDGERS:
        for job_id, row in run_index(load_object(path)).items():
            if job_id in comparators:
                raise RuntimeError(f"duplicate comparator job ID {job_id}")
            comparators[job_id] = row

    cells: list[dict[str, Any]] = []
    seen: set[int] = set()
    for family in endpoints["cells"]:
        for endpoint_row in family["per_seed"]:
            treatment_job_id = int(endpoint_row["treatment_job_id"])
            comparator_job_id = int(endpoint_row["comparator_job_id"])
            treatment = treatments.get(treatment_job_id)
            comparator = comparators.get(comparator_job_id)
            if treatment is None or comparator is None:
                raise RuntimeError(
                    f"missing ledger binding for pair {treatment_job_id}/{comparator_job_id}"
                )
            if treatment_job_id in seen:
                raise RuntimeError(f"duplicate endpoint treatment {treatment_job_id}")
            seen.add(treatment_job_id)
            cells.append(
                {
                    "scale": str(family["scale"]),
                    "domain": str(family["domain"]),
                    "seed": int(endpoint_row["seed"]),
                    "treatment_job_id": treatment_job_id,
                    "comparator_job_id": comparator_job_id,
                    "endpoint": endpoint_decomposition(endpoint_row),
                    "pairing": pairing_summary(treatment, comparator),
                    "mechanism": mechanism_summary(Path(str(treatment["run_dir"]))),
                }
            )
    if len(cells) != 33:
        raise RuntimeError(f"expected 33 diagnostic cells, got {len(cells)}")

    return {
        "schema": "e112r1-private-heterogeneity-diagnostic-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": PRIVATE_LABEL,
        "confirmatory": False,
        "campaign_mutation_allowed": False,
        "paper_efficacy_output_allowed": False,
        "new_evaluation_outcomes_read": False,
        "pointmaze": "excluded",
        "analysis_boundary": (
            "joins the frozen 33 exact endpoint pairs to treatment-only training "
            "telemetry; no mean, interval, hypothesis test, fit, pooling, or new "
            "evaluation checkpoint is read"
        ),
        "known_design_limit": (
            "non-Pantry treatment and historical comparator sampler/request "
            "paths are not matched; E66 previously defined this plumbing "
            "difference as a separate intervention"
        ),
        "inputs": {
            relative(FREEZE): sha256(FREEZE),
            relative(ENDPOINTS): sha256(ENDPOINTS),
            relative(TREATMENT_LEDGER): sha256(TREATMENT_LEDGER),
            **{relative(path): sha256(path) for path in COMPARATOR_LEDGERS},
        },
        "cells": cells,
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    private_root = (ROOT / "var/artifacts/private_interim").resolve()
    try:
        path.resolve().relative_to(private_root)
    except ValueError as exc:
        raise RuntimeError(f"private output must stay under {private_root}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    payload = build()
    atomic_write(args.output, payload)
    if args.stdout:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"[e112r1-private-diagnostic] cells={len(payload['cells'])} "
            f"output={args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
