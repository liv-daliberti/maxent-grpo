#!/usr/bin/env python3
"""Fail-closed live audit for E63's cross-domain transform pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e63_cross_domain_transform_pilot_identity.json"
OUT = ROOT / "var/artifacts/e63_cross_domain_transform_pilot_audit_latest.json"
PROTOCOL = ROOT / "paper/preregistration/e63_cross_domain_transform_pilot.md"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e63_cross_domain_transform_pilot.sh"
PLOTTER = ROOT / "ops/exp_scaling/plot_e63_cross_domain_transform_pilot.py"
WATCHER = ROOT / "ops/exp_scaling/watch_e63_cross_domain_transform_pilot.sh"
DOMAINS = ("graph_coloring", "countdown", "mathir")
PREFIXES = {
    "graph_coloring": "gce63_transform_pilot",
    "countdown": "cde63_transform_pilot",
    "mathir": "mie63_transform_pilot",
}
TERMINAL_STEPS = {
    "graph_coloring": 192,
    "countdown": 384,
    "mathir": 384,
}
DATA_ROOTS = {
    "graph_coloring": ROOT / "var/data/exact_answer_mode_probe",
    "countdown": ROOT / "var/data/exact_countdown_easy3_probe",
    "mathir": ROOT / "var/data/mathir_action_menu_v1",
}
CONTROL = "verified_first_bootstrap_local_canonical"
TREATMENT = "verified_counterfactual_canonical"
SEED = 9011
NUM_SAMPLES = 16
MAX_PROPOSAL_ATTEMPTS = 3
PROPOSAL_TEMPERATURE = 1.0
PROPOSAL_TEMPERATURE_STEP = 0.2
FLOAT32_TELEMETRY_TOLERANCE = 1e-6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tree(root: Path) -> str:
    lines = [
        f"{_sha256(path)}  ./{path.relative_to(root).as_posix()}\n"
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        step = int(value.get("trainer/global_step", -1))
        previous = records.get(step)
        if previous is not None:
            conflicts = [
                key
                for key in set(previous) | set(value)
                if key.startswith(("train/", "actor/"))
                and previous.get(key) != value.get(key)
            ]
            if conflicts:
                raise ValueError(
                    f"{path}:{line_number} conflicts at step {step}"
                )
            if len(value) > len(previous):
                records[step] = value
        else:
            records[step] = value
    return [records[step] for step in sorted(records)]


def _evaluations(path: Path) -> list[dict[str, float | int]]:
    by_step: dict[int, dict[int, dict[str, Any]]] = {}
    if not path.is_file():
        return []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or value.get("evaluation_kind")
            != "fixed_seed_sampled_k_neutral"
            or not isinstance(value.get("draw_index"), int)
            or not isinstance(value.get("metrics"), dict)
        ):
            continue
        draw_index = int(value["draw_index"])
        if not 0 <= draw_index < 4:
            continue
        by_step.setdefault(int(value["step"]), {})[draw_index] = value[
            "metrics"
        ]
    result: list[dict[str, float | int]] = []
    for step in sorted(by_step):
        draws = by_step[step]
        if set(draws) != {0, 1, 2, 3}:
            continue
        rows = [draws[index] for index in range(4)]
        result.append(
            {
                "step": step,
                "distinct8": sum(
                    float(row["distinct_correct_modes_at_k"])
                    for row in rows
                )
                / 4,
                "pass8": sum(
                    float(row["any_correct_at_k"]) for row in rows
                )
                / 4,
                "mean8": sum(float(row["mean_at_k"]) for row in rows)
                / 4,
            }
        )
    return result


def _run_root(*, prefix: str, arm: str, job_id: int) -> Path | None:
    candidates = sorted(
        (ROOT / "var/data").glob(
            f"*{prefix}_{arm}_s{SEED}/debug_job{job_id}"
        )
    )
    return candidates[0] if len(candidates) == 1 else None


def _crash_signatures(job_id: int) -> list[str]:
    signatures = (
        "Traceback (most recent call last):",
        "RuntimeError:",
        "CUDA out of memory",
        "Segmentation fault",
        "slurmstepd: error:",
    )
    hits: list[str] = []
    for path in sorted((ROOT / "var/artifacts/logs").glob(f"*-{job_id}.*")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for signature in signatures:
            if signature in text:
                hits.append(f"{path.name}: {signature}")
    return hits


def _audit_arm(
    *,
    domain: str,
    prefix: str,
    terminal_step: int,
    arm: str,
    job_id: int,
    violations: list[str],
) -> dict[str, Any]:
    label = f"{domain}/{arm}"
    run_root = _run_root(prefix=prefix, arm=arm, job_id=job_id)
    records: list[dict[str, Any]] = []
    if run_root is not None:
        metrics_path = run_root / "train_metrics.jsonl"
        if metrics_path.is_file():
            try:
                records = _load_records(metrics_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                violations.append(f"{label}: cannot inspect metrics: {error}")
    crash_hits = _crash_signatures(job_id)
    for hit in crash_hits:
        violations.append(f"{label}: crash signature {hit}")

    latest_step = -1
    generated_groups = 0
    generated_rows = 0
    original_prompt_groups = 0
    conditioned_prompt_groups = 0
    admitted_new = 0
    cumulative_new = 0
    proposal_active_records = 0
    anchor_records = 0
    current_neutral_anchor_records = 0
    prior_bank_anchor_records = 0
    disagreement_rows = 0
    transform_candidate_surfaces = 0
    transform_validator_positive = 0
    transform_tokenization_rejections = 0
    transform_novel_outcomes = 0
    transform_success_records = 0
    maximum_support = 0.0
    neutral_request_seeds: dict[str, int] = {}
    for record in records:
        step = int(record.get("trainer/global_step", -1))
        latest_step = max(latest_step, step)
        for key, value in record.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not math.isfinite(float(value))
            ):
                violations.append(f"{arm} step {step}: nonfinite {key}")
            if key.startswith("train/maxent_"):
                violations.append(
                    f"{arm} step {step}: direct MaxEnt telemetry"
                )
        for key in (
            "train/canonical_replay_projection_active",
            "train/canonical_replay_mass_projection_active",
            "train/canonical_replay_alpha_projection_active",
            "train/semantic_shannon_success_conditioned_signed_"
            "open_set_projection_active",
            "train/canonical_replay_gold_support_feedback",
        ):
            value = record.get(key, 0)
            if _finite(value) and not math.isclose(float(value), 0.0):
                violations.append(
                    f"{arm} step {step}: forbidden nonzero {key}"
                )
        support = record.get(
            "train/verified_discovery_mean_support_per_prompt"
        )
        if _finite(support):
            maximum_support = max(maximum_support, float(support))
        if step > 0:
            neutral_request_seed = record.get("actor/sampling_request_seed")
            if not _finite(neutral_request_seed):
                violations.append(
                    f"{arm} step {step}: missing neutral request seed"
                )
            else:
                neutral_request_seeds[str(step)] = int(
                    float(neutral_request_seed)
                )

        if arm != TREATMENT:
            continue
        enabled = record.get("actor/counterfactual_proposal_enabled")
        if _finite(enabled):
            proposal_active_records += 1
            if float(enabled) != 1.0:
                violations.append(
                    f"{arm} step {step}: proposal enabled flag is not one"
                )
        conditioned_ppo = record.get(
            "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo"
        )
        if _finite(conditioned_ppo) and float(conditioned_ppo) != 0.0:
            violations.append(
                f"{arm} step {step}: conditioned proposal entered PPO"
            )
        neutral_rows = record.get(
            "actor/counterfactual_proposal_neutral_ppo_rows"
        )
        if _finite(neutral_rows) and int(neutral_rows) != NUM_SAMPLES:
            violations.append(
                f"{arm} step {step}: neutral PPO width changed"
            )
        for key in (
            "actor/counterfactual_proposal_gold_support_feedback",
            "actor/counterfactual_proposal_desired_mode_count_feedback",
            "actor/counterfactual_proposal_eval_feedback",
            "actor/counterfactual_proposal_transform_gold_support_feedback",
            "actor/counterfactual_proposal_transform_desired_mode_count_feedback",
            "actor/counterfactual_proposal_transform_eval_feedback",
            "actor/counterfactual_proposal_transform_rows_sent_to_ppo",
        ):
            value = record.get(key, 0)
            if _finite(value) and float(value) != 0.0:
                violations.append(
                    f"{arm} step {step}: forbidden feedback {key}"
                )
        anchor = record.get(
            "actor/counterfactual_proposal_anchor_available"
        )
        anchor_records += int(_finite(anchor) and float(anchor) == 1.0)
        current_anchor = record.get(
            "actor/counterfactual_proposal_anchor_from_current_neutral"
        )
        current_neutral_anchor_records += int(
            _finite(current_anchor) and float(current_anchor) == 1.0
        )
        prior_anchor = record.get(
            "actor/counterfactual_proposal_anchor_from_prior_bank"
        )
        prior_bank_anchor_records += int(
            _finite(prior_anchor) and float(prior_anchor) == 1.0
        )
        record_groups = int(
            float(
                record.get(
                    "actor/counterfactual_proposal_groups_generated",
                    0,
                )
            )
            if _finite(
                record.get(
                    "actor/counterfactual_proposal_groups_generated",
                    0,
                )
            )
            else 0
        )
        record_rows = int(
            float(
                record.get(
                    "actor/counterfactual_proposal_rows_generated",
                    0,
                )
            )
            if _finite(
                record.get(
                    "actor/counterfactual_proposal_rows_generated",
                    0,
                )
            )
            else 0
        )
        generated_groups += record_groups
        generated_rows += record_rows
        original_prompt_value = record.get(
            "actor/counterfactual_proposal_original_prompt_groups",
            0,
        )
        conditioned_prompt_value = record.get(
            "actor/counterfactual_proposal_conditioned_prompt_groups",
            0,
        )
        if _finite(original_prompt_value):
            original_prompt_groups += int(float(original_prompt_value))
        if _finite(conditioned_prompt_value):
            conditioned_prompt_groups += int(
                float(conditioned_prompt_value)
            )
        transform_success = record.get(
            "actor/counterfactual_proposal_transform_success",
            0,
        )
        record_transform_success = bool(
            _finite(transform_success) and float(transform_success) == 1.0
        )
        transform_success_records += int(record_transform_success)
        record_transform_candidates = record.get(
            "actor/counterfactual_proposal_transform_candidate_surfaces",
            0,
        )
        record_transform_validator_positive = record.get(
            "actor/counterfactual_proposal_transform_validator_positive",
            0,
        )
        record_transform_rejections = record.get(
            "actor/counterfactual_proposal_"
            "transform_tokenization_rejections",
            0,
        )
        record_transform_novel = record.get(
            "actor/counterfactual_proposal_"
            "transform_novel_unique_outcomes",
            0,
        )
        if _finite(record_transform_candidates):
            transform_candidate_surfaces += int(
                float(record_transform_candidates)
            )
        if _finite(record_transform_validator_positive):
            transform_validator_positive += int(
                float(record_transform_validator_positive)
            )
        if _finite(record_transform_rejections):
            transform_tokenization_rejections += int(
                float(record_transform_rejections)
            )
        if _finite(record_transform_novel):
            transform_novel_outcomes += int(
                float(record_transform_novel)
            )
        if _finite(anchor) and float(anchor) == 1.0:
            if record_transform_success:
                if record_groups != 0 or record_rows != 0:
                    violations.append(
                        f"{arm} step {step}: successful transform also sampled"
                    )
                if (
                    not _finite(record_transform_candidates)
                    or float(record_transform_candidates) < 1
                    or not _finite(record_transform_validator_positive)
                    or float(record_transform_validator_positive) < 1
                    or not _finite(record_transform_novel)
                    or float(record_transform_novel) < 1
                ):
                    violations.append(
                        f"{arm} step {step}: transform success lacks "
                        "validator-positive novel evidence"
                    )
            else:
                if not 1 <= record_groups <= MAX_PROPOSAL_ATTEMPTS:
                    violations.append(
                        f"{arm} step {step}: anchored fallback group count "
                        "is invalid"
                    )
                if record_rows != record_groups * NUM_SAMPLES:
                    violations.append(
                        f"{arm} step {step}: fallback rows do not match "
                        "group width"
                    )
                max_attempts = record.get(
                    "actor/counterfactual_proposal_max_attempts"
                )
                if (
                    not _finite(max_attempts)
                    or int(float(max_attempts)) != MAX_PROPOSAL_ATTEMPTS
                ):
                    violations.append(
                        f"{arm} step {step}: fallback budget telemetry mismatch"
                    )
                proposal_temperature = record.get(
                    "actor/counterfactual_proposal_sampling_temperature"
                )
                if (
                    not _finite(proposal_temperature)
                    or not math.isclose(
                        float(proposal_temperature),
                        PROPOSAL_TEMPERATURE,
                    )
                ):
                    violations.append(
                        f"{arm} step {step}: proposal temperature mismatch"
                    )
                temperature_step = record.get(
                    "actor/counterfactual_proposal_temperature_step"
                )
                if (
                    not _finite(temperature_step)
                    or not math.isclose(
                        float(temperature_step),
                        PROPOSAL_TEMPERATURE_STEP,
                        rel_tol=0.0,
                        abs_tol=FLOAT32_TELEMETRY_TOLERANCE,
                    )
                ):
                    violations.append(
                        f"{arm} step {step}: proposal temperature step mismatch"
                    )
                last_temperature = record.get(
                    "actor/counterfactual_proposal_last_temperature"
                )
                expected_last_temperature = (
                    PROPOSAL_TEMPERATURE
                    + PROPOSAL_TEMPERATURE_STEP * (record_groups - 1)
                )
                if (
                    not _finite(last_temperature)
                    or not math.isclose(
                        float(last_temperature),
                        expected_last_temperature,
                        rel_tol=0.0,
                        abs_tol=FLOAT32_TELEMETRY_TOLERANCE,
                    )
                ):
                    violations.append(
                        f"{arm} step {step}: last proposal temperature mismatch"
                    )
            record_original_prompt_groups = record.get(
                "actor/counterfactual_proposal_original_prompt_groups"
            )
            if (
                not _finite(record_original_prompt_groups)
                or int(float(record_original_prompt_groups)) != record_groups
            ):
                violations.append(
                    f"{arm} step {step}: original-prompt group count mismatch"
                )
            record_conditioned_prompt_groups = record.get(
                "actor/counterfactual_proposal_conditioned_prompt_groups"
            )
            if (
                not _finite(record_conditioned_prompt_groups)
                or float(record_conditioned_prompt_groups) != 0.0
            ):
                violations.append(
                    f"{arm} step {step}: conditioned proposal prompt used"
                )
            if record_groups:
                seed_isolation = record.get(
                    "actor/counterfactual_proposal_seed_isolation_active"
                )
                proposal_seed_min = record.get(
                    "actor/counterfactual_proposal_request_seed_min"
                )
                proposal_seed_max = record.get(
                    "actor/counterfactual_proposal_request_seed_max"
                )
                if (
                    not _finite(seed_isolation)
                    or float(seed_isolation) != 1.0
                    or not _finite(proposal_seed_min)
                    or not _finite(proposal_seed_max)
                ):
                    violations.append(
                        f"{arm} step {step}: proposal request seeds are not "
                        "isolated"
                    )
        admitted = record.get(
            "actor/counterfactual_proposal_admitted_new_outcomes",
            0,
        )
        admitted_new += int(float(admitted)) if _finite(admitted) else 0
        cumulative = record.get(
            "actor/counterfactual_proposal_cumulative_new_outcomes",
            0,
        )
        if _finite(cumulative):
            if float(cumulative) < cumulative_new:
                violations.append(
                    f"{arm} step {step}: cumulative proposals decreased"
                )
            cumulative_new = max(cumulative_new, int(float(cumulative)))
        disagreement = record.get(
            "actor/counterfactual_proposal_"
            "validator_task_disagreement_rows",
            0,
        )
        disagreement_rows += (
            int(float(disagreement)) if _finite(disagreement) else 0
        )

    evaluations = (
        _evaluations(run_root / "eval_mode_coverage_draws.jsonl")
        if run_root is not None
        else []
    )
    post_initial_positive_excess = sum(
        int(
            int(row["step"]) > 0
            and float(row["distinct8"]) > float(row["pass8"]) + 1e-12
        )
        for row in evaluations
    )
    terminal = latest_step >= terminal_step
    if terminal and arm == TREATMENT:
        if proposal_active_records == 0:
            violations.append("treatment: proposal telemetry never appeared")
        if anchor_records == 0:
            violations.append(
                "treatment: no verified anchor was available"
            )
        if current_neutral_anchor_records == 0:
            violations.append(
                "treatment: current neutral group never bootstrapped an anchor"
            )
        if admitted_new == 0 or cumulative_new == 0:
            violations.append(
                "treatment: no novel verified proposal was admitted"
            )
        if transform_success_records == 0 or transform_novel_outcomes == 0:
            violations.append(
                "treatment: validator-preserving transform never escaped "
                "singleton support"
            )
        if post_initial_positive_excess < 1:
            violations.append(
                f"{domain}: no post-initial neutral multiplicity point"
            )

    return {
        "arm": arm,
        "domain": domain,
        "job_id": job_id,
        "run_root": str(run_root) if run_root is not None else None,
        "latest_step": latest_step,
        "terminal": terminal,
        "evaluation_points": len(evaluations),
        "evaluations": evaluations,
        "post_initial_positive_excess_points": (
            post_initial_positive_excess
        ),
        "maximum_discovered_mean_support": maximum_support,
        "neutral_request_seeds": neutral_request_seeds,
        "proposal": {
            "active_records": proposal_active_records,
            "anchor_records": anchor_records,
            "current_neutral_anchor_records": current_neutral_anchor_records,
            "prior_bank_anchor_records": prior_bank_anchor_records,
            "generated_groups": generated_groups,
            "generated_rows": generated_rows,
            "original_prompt_groups": original_prompt_groups,
            "conditioned_prompt_groups": conditioned_prompt_groups,
            "admitted_new_outcomes": admitted_new,
            "cumulative_new_outcomes": cumulative_new,
            "validator_task_disagreement_rows": disagreement_rows,
            "transform_candidate_surfaces": transform_candidate_surfaces,
            "transform_validator_positive": transform_validator_positive,
            "transform_tokenization_rejections": (
                transform_tokenization_rejections
            ),
            "transform_novel_outcomes": transform_novel_outcomes,
            "transform_success_records": transform_success_records,
        },
        "crash_signatures": crash_hits,
    }


def audit(identity_path: Path) -> dict[str, Any]:
    violations: list[str] = []
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if not isinstance(identity, dict):
            raise ValueError("identity is not an object")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "schema": "e63_cross_domain_transform_pilot_audit_v1",
            "status": "in_progress",
            "violations": [f"identity unavailable: {error}"],
            "domains": {},
            "latest_step": -1,
        }

    if identity.get("schema") != "e63_cross_domain_transform_pilot_v1":
        violations.append("identity schema mismatch")
    for key, path in {
        "protocol_sha256": PROTOCOL,
        "launcher_sha256": LAUNCHER,
        "auditor_sha256": Path(__file__).resolve(),
        "plotter_sha256": PLOTTER,
        "watcher_sha256": WATCHER,
    }.items():
        if not path.is_file() or identity.get(key) != _sha256(path):
            violations.append(f"{key} binding mismatch")

    source_hash = str(identity.get("source_hash", ""))
    ops_hash = str(identity.get("execution_surface_hash", ""))
    source_root = (
        ROOT
        / f"var/artifacts/source_snapshots/e63_transform_pilot_{source_hash}/src"
    )
    ops_root = (
        ROOT
        / f"var/artifacts/source_snapshots/e63_transform_pilot_ops_{ops_hash}/ops"
    )
    try:
        if _hash_tree(source_root) != source_hash:
            violations.append("source snapshot binding mismatch")
        if _hash_tree(ops_root) != ops_hash:
            violations.append("execution snapshot binding mismatch")
    except OSError as error:
        violations.append(f"cannot inspect snapshots: {error}")

    jobs = identity.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != set(DOMAINS):
        violations.append("identity job map mismatch")
        jobs = {}
    data_hashes = identity.get("data_tree_sha256")
    if not isinstance(data_hashes, dict):
        data_hashes = {}
        violations.append("identity data hashes missing")
    for domain, data_root in DATA_ROOTS.items():
        try:
            if data_hashes.get(domain) != _hash_tree(data_root):
                violations.append(f"{domain}: data binding mismatch")
        except OSError as error:
            violations.append(f"{domain}: cannot inspect data: {error}")
    manifest_hashes = identity.get("manifest_sha256")
    if not isinstance(manifest_hashes, dict):
        manifest_hashes = {}
        violations.append("identity manifest hashes missing")

    domain_results: dict[str, Any] = {}
    for domain in DOMAINS:
        prefix = PREFIXES[domain]
        manifest = ROOT / f"var/artifacts/{prefix}_comparative_jobs.tsv"
        if (
            not manifest.is_file()
            or manifest_hashes.get(domain) != _sha256(manifest)
        ):
            violations.append(f"{domain}: manifest binding mismatch")
        try:
            with manifest.open(encoding="utf-8", newline="") as handle:
                manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
        except (OSError, csv.Error) as error:
            manifest_rows = []
            violations.append(f"{domain}: cannot inspect manifest: {error}")
        raw_jobs = jobs.get(domain)
        if not isinstance(raw_jobs, list) or len(raw_jobs) != 2:
            violations.append(f"{domain}: identity jobs mismatch")
            raw_jobs = []
        job_map: dict[str, int] = {}
        for record in raw_jobs:
            if not isinstance(record, dict):
                continue
            arm = str(record.get("arm", ""))
            try:
                job_map[arm] = int(record["job_id"])
            except (KeyError, TypeError, ValueError):
                continue
        if set(job_map) != {CONTROL, TREATMENT}:
            violations.append(f"{domain}: identity arm map mismatch")
        for arm in (CONTROL, TREATMENT):
            job_id = int(job_map.get(arm, -1))
            exact = [
                row
                for row in manifest_rows
                if row.get("arm") == arm
                and row.get("seed") == str(SEED)
                and row.get("job_id") == str(job_id)
                and row.get("run_stamp") == f"{prefix}_{arm}_s{SEED}"
            ]
            if len(exact) != 1:
                violations.append(
                    f"{domain}/{arm}: manifest exact-job binding mismatch"
                )
        arm_results = {
            arm: _audit_arm(
                domain=domain,
                prefix=prefix,
                terminal_step=TERMINAL_STEPS[domain],
                arm=arm,
                job_id=int(job_map.get(arm, -1)),
                violations=violations,
            )
            for arm in (CONTROL, TREATMENT)
            if int(job_map.get(arm, -1)) >= 0
        }
        domain_terminal = (
            len(arm_results) == 2
            and all(
                bool(result["terminal"])
                for result in arm_results.values()
            )
        )
        if CONTROL in arm_results and TREATMENT in arm_results:
            control_seeds = arm_results[CONTROL]["neutral_request_seeds"]
            treatment_seeds = arm_results[TREATMENT][
                "neutral_request_seeds"
            ]
            for step in sorted(
                set(control_seeds) & set(treatment_seeds),
                key=int,
            ):
                if control_seeds[step] != treatment_seeds[step]:
                    violations.append(
                        f"{domain}: matched neutral request seed differs "
                        f"at step {step}"
                    )
        if domain_terminal:
            control_evals = arm_results[CONTROL]["evaluations"]
            treatment_evals = arm_results[TREATMENT]["evaluations"]
            if not control_evals or not treatment_evals:
                violations.append(
                    f"{domain}: terminal comparison lacks evaluations"
                )
            else:
                control_final = control_evals[-1]
                treatment_final = treatment_evals[-1]
                if int(control_final["step"]) != int(
                    treatment_final["step"]
                ):
                    violations.append(
                        f"{domain}: unmatched terminal evaluation steps"
                    )
                elif float(treatment_final["distinct8"]) + 1e-12 < float(
                    control_final["distinct8"]
                ):
                    violations.append(
                        f"{domain}: terminal treatment distinct@8 is "
                        "below control"
                    )
        domain_results[domain] = {
            "terminal": domain_terminal,
            "terminal_step": TERMINAL_STEPS[domain],
            "arms": arm_results,
        }

    all_terminal = (
        len(domain_results) == len(DOMAINS)
        and all(
            bool(result["terminal"])
            for result in domain_results.values()
        )
    )

    status = (
        "fail"
        if violations
        else "pass"
        if all_terminal
        else "in_progress"
    )
    latest_step = max(
        (
            int(arm_result["latest_step"])
            for domain_result in domain_results.values()
            for arm_result in domain_result["arms"].values()
        ),
        default=-1,
    )
    return {
        "schema": "e63_cross_domain_transform_pilot_audit_v1",
        "status": status,
        "violations": violations,
        "identity_path": str(identity_path),
        "latest_step": latest_step,
        "all_terminal": all_terminal,
        "domains": domain_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, default=IDENTITY)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    payload = audit(args.identity)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(
        f"[e63-cross-domain-audit] status={payload['status']} "
        f"step={payload['latest_step']} "
        f"violations={len(payload['violations'])} out={args.out}"
    )
    if payload["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
