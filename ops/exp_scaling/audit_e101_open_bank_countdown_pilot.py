#!/usr/bin/env python3
"""Fail-closed live audit for the E101 open-bank Countdown pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e101_open_bank_countdown_pilot_jobs.json"
OUT = ROOT / "var/artifacts/e101_open_bank_countdown_pilot_audit_latest.json"
TARGET_STEPS = 128
WALLTIME_CAP_SECONDS = 55 * 60
FAILURE = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"torch\.OutOfMemoryError|ChildFailedError|RayActorError|"
    r"RuntimeError:[^\n]*non-finite|segmentation fault",
    re.IGNORECASE,
)
EVAL_FIELDS = {
    "greedy": "eval/multi_answer/accuracy",
    "pass8": "eval/multi_answer/sampled_any_correct_at_8",
    "mean8": "eval/multi_answer/sampled_mean_at_8",
    "distinct8": "eval/multi_answer/sampled_distinct_correct_at_8",
    "coverage8": "eval/multi_answer/sampled_mode_coverage_at_8",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    value = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        value.update(str(item.relative_to(path)).encode())
        value.update(b"\0")
        value.update(item.read_bytes())
        value.update(b"\0")
    return value.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def metric(row: dict[str, Any], suffix: str) -> float | None:
    for prefix in ("train/", "actor/"):
        key = prefix + suffix
        value = row.get(key)
        if finite(value):
            return float(value)
    return None


def scheduler_rows(job_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not job_ids:
        return {}
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-j",
            ",".join(str(value) for value in job_ids),
            "-n",
            "-P",
            "-o",
            "JobIDRaw,State,ElapsedRaw,NodeList,ExitCode",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for raw in result.stdout.splitlines():
        parts = raw.split("|")
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        job_id = int(parts[0])
        if job_id not in job_ids:
            continue
        elapsed = int(parts[2]) if parts[2].isdigit() else None
        rows[job_id] = {
            "state": parts[1],
            "elapsed_seconds": elapsed,
            "node": parts[3],
            "exit_code": parts[4],
        }
    return rows


def log_failures(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return sorted({match.group(0) for match in FAILURE.finditer(text)})


def parse_metrics(
    path: Path,
    *,
    allow_exact_grammar_transforms: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    train_records = 0
    last_step = -1
    evaluations: dict[int, dict[str, float]] = {}
    actuator_updates = 0
    balance_updates = 0
    realized_replay_tokens = 0.0
    maximum_actuator_modes = 0.0
    maximum_balance_modes = 0.0
    proposal_active_updates = 0
    proposal_transform_enabled_max = 0.0
    proposal_exact_grammar_enabled_max = 0.0
    proposal_transform_candidates = 0.0
    proposal_transform_validator_positive = 0.0
    proposal_transform_novel_outcomes = 0.0
    proposal_transform_success_updates = 0
    proposal_original_prompt_groups = 0.0
    proposal_groups_generated = 0.0
    proposal_rows_generated = 0.0
    proposal_task_positive_rows = 0.0
    proposal_validator_positive_rows = 0.0
    proposal_validator_disagreement_rows = 0.0
    proposal_same_anchor_rows = 0.0
    proposal_known_alternate_rows = 0.0
    proposal_novel_candidate_rows = 0.0
    proposal_novel_unique_outcomes = 0.0
    proposal_attempts_exhausted_updates = 0
    proposal_admissions = 0.0
    proposal_cumulative = 0.0
    proposal_to_ppo = 0.0
    proposal_objective_delta = 0.0
    forbidden_feedback = 0.0
    first_admission_step: int | None = None
    actuation_at_or_after_admission = 0

    if not path.is_file():
        return {
            "materialized": False,
            "last_step": last_step,
            "evaluations": [],
        }, violations

    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            violations.append(f"invalid metrics JSON at line {line_number}")
            continue
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    violations.append(f"non-finite {key} at line {line_number}")
        raw_step = row.get("trainer/global_step", row.get("trainer/step", -1))
        step = int(raw_step) if finite(raw_step) else -1
        last_step = max(last_step, step)
        if any(key.startswith("eval/multi_answer/") for key in row):
            values = {
                name: float(row[key])
                for name, key in EVAL_FIELDS.items()
                if finite(row.get(key))
            }
            if values:
                evaluations[step] = values
        if not any(key.startswith("train/") for key in row):
            continue
        train_records += 1
        groups = metric(row, "canonical_replay_actuator_groups") or 0.0
        modes = metric(row, "canonical_replay_actuator_modes") or 0.0
        eligible = metric(row, "canonical_replay_eligible_groups") or 0.0
        retained = metric(row, "canonical_replay_retained_modes") or 0.0
        tokens = metric(row, "canonical_replay_realized_response_tokens") or 0.0
        if groups > 0:
            actuator_updates += 1
        if eligible > 0:
            balance_updates += 1
        realized_replay_tokens += tokens
        maximum_actuator_modes = max(maximum_actuator_modes, modes)
        maximum_balance_modes = max(maximum_balance_modes, retained)

        active = metric(row, "counterfactual_proposal_singleton_only_active") or 0.0
        transform_enabled = metric(row, "counterfactual_proposal_transform_enabled") or 0.0
        exact_grammar_enabled = metric(
            row,
            "counterfactual_proposal_exact_grammar_transform_enabled",
        ) or 0.0
        transform_candidates = metric(row, "counterfactual_proposal_transform_candidate_surfaces") or 0.0
        transform_validator_positive = metric(
            row,
            "counterfactual_proposal_transform_validator_positive",
        ) or 0.0
        transform_novel = metric(
            row,
            "counterfactual_proposal_transform_novel_unique_outcomes",
        ) or 0.0
        transform_success = metric(
            row,
            "counterfactual_proposal_transform_success",
        ) or 0.0
        original_groups = metric(row, "counterfactual_proposal_original_prompt_groups") or 0.0
        generated_groups = metric(row, "counterfactual_proposal_groups_generated") or 0.0
        generated_rows = metric(row, "counterfactual_proposal_rows_generated") or 0.0
        task_positive = metric(row, "counterfactual_proposal_task_reward_positive_rows") or 0.0
        validator_positive = metric(row, "counterfactual_proposal_validator_positive_rows") or 0.0
        disagreement = metric(row, "counterfactual_proposal_validator_task_disagreement_rows") or 0.0
        same_anchor = metric(row, "counterfactual_proposal_same_anchor_rows") or 0.0
        known_alternate = metric(row, "counterfactual_proposal_known_alternate_rows") or 0.0
        novel_candidates = metric(row, "counterfactual_proposal_novel_candidate_rows") or 0.0
        novel_unique = metric(row, "counterfactual_proposal_novel_unique_outcomes") or 0.0
        attempts_exhausted = metric(row, "counterfactual_proposal_attempts_exhausted") or 0.0
        admitted = metric(row, "counterfactual_proposal_admitted_new_outcomes") or 0.0
        cumulative = metric(row, "counterfactual_proposal_cumulative_new_outcomes") or 0.0
        if active > 0:
            proposal_active_updates += 1
        proposal_transform_enabled_max = max(
            proposal_transform_enabled_max,
            transform_enabled,
        )
        proposal_exact_grammar_enabled_max = max(
            proposal_exact_grammar_enabled_max,
            exact_grammar_enabled,
        )
        proposal_transform_candidates += transform_candidates
        proposal_transform_validator_positive += transform_validator_positive
        proposal_transform_novel_outcomes += transform_novel
        proposal_transform_success_updates += int(transform_success > 0)
        proposal_original_prompt_groups += original_groups
        proposal_groups_generated += generated_groups
        proposal_rows_generated += generated_rows
        proposal_task_positive_rows += task_positive
        proposal_validator_positive_rows += validator_positive
        proposal_validator_disagreement_rows += disagreement
        proposal_same_anchor_rows += same_anchor
        proposal_known_alternate_rows += known_alternate
        proposal_novel_candidate_rows += novel_candidates
        proposal_novel_unique_outcomes += novel_unique
        proposal_attempts_exhausted_updates += int(attempts_exhausted > 0)
        proposal_admissions += admitted
        proposal_cumulative = max(proposal_cumulative, cumulative)
        if admitted > 0 and first_admission_step is None:
            first_admission_step = step
        if first_admission_step is not None and step >= first_admission_step and groups > 0:
            actuation_at_or_after_admission += 1

        for key, value in row.items():
            if not finite(value):
                continue
            if key.endswith("rows_sent_to_ppo") or key.endswith("rows_to_ppo"):
                proposal_to_ppo = max(proposal_to_ppo, abs(float(value)))
            if key.endswith("gold_support_feedback") or key.endswith("desired_mode_count_feedback") or key.endswith("eval_feedback"):
                forbidden_feedback = max(forbidden_feedback, abs(float(value)))
        objective_delta = metric(row, "counterfactual_proposal_objective_outcome_delta")
        if objective_delta is not None:
            proposal_objective_delta = max(proposal_objective_delta, abs(objective_delta))

    ordered_evals = [
        {"step": step, **values} for step, values in sorted(evaluations.items())
    ]
    changes: dict[str, float] = {}
    if len(ordered_evals) >= 2:
        start, finish = ordered_evals[0], ordered_evals[-1]
        for name in EVAL_FIELDS:
            if name in start and name in finish:
                changes[name] = float(finish[name]) - float(start[name])
    if proposal_transform_enabled_max != 0.0:
        violations.append(
            "legacy transform-derived proposals were active: "
            f"enabled={proposal_transform_enabled_max}"
        )
    if not allow_exact_grammar_transforms and (
        proposal_exact_grammar_enabled_max != 0.0
        or proposal_transform_candidates != 0.0
    ):
        violations.append(
            "exact-grammar transform-derived proposals were active: "
            f"enabled={proposal_exact_grammar_enabled_max}, "
            f"candidates={proposal_transform_candidates}"
        )
    if proposal_to_ppo != 0.0:
        violations.append(f"proposal rows reached PPO: {proposal_to_ppo}")
    if proposal_objective_delta != 0.0:
        violations.append(f"proposal changed neutral objective support: {proposal_objective_delta}")
    if forbidden_feedback != 0.0:
        violations.append(f"forbidden training feedback was nonzero: {forbidden_feedback}")

    return {
        "materialized": True,
        "last_step": last_step,
        "train_records": train_records,
        "evaluations": ordered_evals,
        "endpoint_change": changes,
        "mechanism": {
            "replay_actuator_updates": actuator_updates,
            "replay_balance_updates": balance_updates,
            "realized_replay_response_tokens": realized_replay_tokens,
            "maximum_actuator_modes": maximum_actuator_modes,
            "maximum_balance_modes": maximum_balance_modes,
            "proposal_singleton_active_updates": proposal_active_updates,
            "proposal_transform_enabled_max": proposal_transform_enabled_max,
            "proposal_exact_grammar_enabled_max": proposal_exact_grammar_enabled_max,
            "proposal_transform_candidate_surfaces": proposal_transform_candidates,
            "proposal_transform_validator_positive": proposal_transform_validator_positive,
            "proposal_transform_novel_outcomes": proposal_transform_novel_outcomes,
            "proposal_transform_success_updates": proposal_transform_success_updates,
            "proposal_original_prompt_groups": proposal_original_prompt_groups,
            "proposal_groups_generated": proposal_groups_generated,
            "proposal_rows_generated": proposal_rows_generated,
            "proposal_task_positive_rows": proposal_task_positive_rows,
            "proposal_validator_positive_rows": proposal_validator_positive_rows,
            "proposal_validator_disagreement_rows": proposal_validator_disagreement_rows,
            "proposal_same_anchor_rows": proposal_same_anchor_rows,
            "proposal_known_alternate_rows": proposal_known_alternate_rows,
            "proposal_novel_candidate_rows": proposal_novel_candidate_rows,
            "proposal_novel_unique_outcomes": proposal_novel_unique_outcomes,
            "proposal_attempts_exhausted_updates": proposal_attempts_exhausted_updates,
            "proposal_admissions": proposal_admissions,
            "proposal_cumulative_new_outcomes": proposal_cumulative,
            "proposal_rows_to_ppo_max_abs": proposal_to_ppo,
            "proposal_objective_outcome_delta_max_abs": proposal_objective_delta,
            "forbidden_feedback_max_abs": forbidden_feedback,
            "first_admission_step": first_admission_step,
            "replay_actuation_updates_at_or_after_first_admission": actuation_at_or_after_admission,
        },
    }, violations


def main() -> int:
    if not LEDGER.is_file():
        payload = {
            "schema": "e101_open_bank_countdown_pilot_audit_v1",
            "status": "not_submitted",
            "runs": [],
            "violations": [],
        }
        atomic_json(OUT, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    global_violations: list[str] = []
    for key in ("protocol", "source_manifest"):
        path = Path(ledger[key])
        recorded = ledger[f"{key}_sha256"]
        if not path.is_file() or digest(path) != recorded:
            global_violations.append(f"frozen {key} digest mismatch")
    launcher = ROOT / "ops/exp_scaling/launch_e101_open_bank_countdown_pilot.py"
    if not launcher.is_file() or digest(launcher) != ledger["launcher_sha256"]:
        global_violations.append("frozen launcher digest mismatch")
    data_root = Path(ledger["data_root"])
    if not data_root.is_dir() or tree_digest(data_root) != ledger["data_tree_sha256"]:
        global_violations.append("frozen data-tree digest mismatch")
    for index, amendment in enumerate(ledger.get("placement_amendments", [])):
        path = Path(amendment["amendment"])
        if not path.is_file() or digest(path) != amendment["amendment_sha256"]:
            global_violations.append(
                f"placement amendment {index} digest mismatch"
            )
    job_ids = [int(run["job_id"]) for run in ledger["runs"]]
    scheduler = scheduler_rows(job_ids)
    runs: list[dict[str, Any]] = []
    for registered in ledger["runs"]:
        job_id = int(registered["job_id"])
        attempt = Path(registered["run_dir"]) / f"debug_job{job_id}"
        metrics, violations = parse_metrics(attempt / "train_metrics.jsonl")
        state = scheduler.get(job_id, {})
        elapsed = state.get("elapsed_seconds")
        if elapsed is not None and int(elapsed) > WALLTIME_CAP_SECONDS:
            violations.append(f"elapsed time exceeded cap: {elapsed}s")
        failures = log_failures(Path(registered["stdout"])) + log_failures(Path(registered["stderr"]))
        if failures:
            violations.extend(f"runtime failure: {value}" for value in failures)
        scheduler_state = str(state.get("state", ""))
        if scheduler_state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")):
            violations.append(f"terminal scheduler state: {scheduler_state}")
        if scheduler_state.startswith("COMPLETED"):
            if int(metrics.get("last_step", -1)) < TARGET_STEPS:
                violations.append("completed before registered terminal step")
            evaluation_steps = {int(row["step"]) for row in metrics.get("evaluations", [])}
            if not {0, 64, 128}.issubset(evaluation_steps):
                violations.append(f"missing registered evaluations: have {sorted(evaluation_steps)}")
        runs.append(
            {
                "arm": registered["arm"],
                "variant": registered["variant"],
                "job_id": job_id,
                "run_dir": registered["run_dir"],
                "scheduler": state,
                "metrics": metrics,
                "violations": violations,
            }
        )

    states = [str(run["scheduler"].get("state", "UNKNOWN")) for run in runs]
    complete = all(state.startswith("COMPLETED") for state in states)
    violations = global_violations + [
        f"{run['arm']}: {value}" for run in runs for value in run["violations"]
    ]
    if violations:
        status = "invalid"
    elif not complete:
        status = "running"
    else:
        opened = next(run for run in runs if run["arm"] == "open")
        mechanism = opened["metrics"]["mechanism"]
        if mechanism["proposal_admissions"] <= 0:
            status = "complete_no_admission"
        elif mechanism["replay_actuation_updates_at_or_after_first_admission"] <= 0:
            status = "complete_admission_not_actuated"
        else:
            status = "complete"
    payload = {
        "schema": "e101_open_bank_countdown_pilot_audit_v1",
        "status": status,
        "target_steps": TARGET_STEPS,
        "walltime_cap_seconds": WALLTIME_CAP_SECONDS,
        "runs": runs,
        "violations": violations,
    }
    atomic_json(OUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
