#!/usr/bin/env python3
"""Audit E117 mechanism identity without reading terminal efficacy contrasts."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e117_same_plumbing_component_preflight as e117  # noqa: E402


AUDIT = "var/artifacts/e117_same_plumbing_component_preflight_audit.json"


def finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return True


def rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not finite_json(row):
            raise ValueError(f"non-object or nonfinite JSON row: {path}")
        result.append(row)
    return result


def scheduler_status(job_ids: list[int]) -> dict[int, dict[str, Any]]:
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(str(job_id) for job_id in job_ids),
            "--format=JobIDRaw,State,ExitCode,Restarts,NodeList",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"could not query E117 accounting: {result.stderr.strip()}")
    status: dict[int, dict[str, Any]] = {}
    wanted = set(job_ids)
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        job_id = int(parts[0])
        if job_id not in wanted:
            continue
        status[job_id] = {
            "state": parts[1].split("+", 1)[0],
            "exit_code": parts[2],
            "restarts": int(parts[3] or 0),
            "node_list": parts[4],
        }
    return status


def metric(row: dict[str, Any], key: str) -> float:
    if key not in row:
        raise ValueError(f"required metric is absent: {key}")
    value = row[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"non-numeric metric {key}: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {key}: {value!r}")
    return value


RETENTION_PREFIXES = (
    "train/online_canonical_proposal_retention_",
    "train/canonical_replay_proposal_retention_",
)
RETENTION_REQUIRED_FIELDS = (
    "tracking_enabled",
    "adaptive_priority_enabled",
    "tracked_admissions",
    "rollout_eligible_admissions",
    "rollout_converted_admissions",
    "rollout_conversion_fraction",
    "rollout_row_frequency",
    "score_observed_admissions",
    "score_followup_admissions",
    "score_retained_admissions",
    "score_retained_fraction",
    "joint_eligible_admissions",
    "joint_retained_admissions",
    "joint_retained_fraction",
    "mean_logprob_drop_mean",
    "mean_logprob_drop_max",
    "sequence_logprob_drop_mean",
    "sequence_logprob_drop_max",
    "rollout_refresh_requests_cumulative",
    "score_refresh_requests_cumulative",
    "refresh_requests_cumulative",
    "priority_visits_added_cumulative",
    "gold_support_feedback",
    "desired_mode_count_feedback",
    "eval_feedback",
)
RETENTION_ZERO_FIELDS = (
    "adaptive_priority_enabled",
    "rollout_refresh_requests_cumulative",
    "score_refresh_requests_cumulative",
    "gold_support_feedback",
    "desired_mode_count_feedback",
    "eval_feedback",
    "refresh_requests_cumulative",
    "priority_visits_added_cumulative",
)
RETENTION_C_ZERO_FIELDS = (
    "tracked_admissions",
    "rollout_eligible_admissions",
    "rollout_converted_admissions",
    "rollout_conversion_fraction",
    "rollout_row_frequency",
    "score_observed_admissions",
    "score_followup_admissions",
    "score_retained_admissions",
    "score_retained_fraction",
    "joint_eligible_admissions",
    "joint_retained_admissions",
    "joint_retained_fraction",
    "mean_logprob_drop_mean",
    "mean_logprob_drop_max",
    "sequence_logprob_drop_mean",
    "sequence_logprob_drop_max",
)
RETENTION_MONOTONE_FIELDS = (
    "tracked_admissions",
    "rollout_eligible_admissions",
    "rollout_converted_admissions",
    "score_observed_admissions",
    "score_followup_admissions",
    "joint_eligible_admissions",
    "rollout_refresh_requests_cumulative",
    "score_refresh_requests_cumulative",
    "refresh_requests_cumulative",
    "priority_visits_added_cumulative",
)
RETENTION_INTEGER_FIELDS = (
    "tracked_admissions",
    "rollout_eligible_admissions",
    "rollout_converted_admissions",
    "score_observed_admissions",
    "score_followup_admissions",
    "score_retained_admissions",
    "joint_eligible_admissions",
    "joint_retained_admissions",
    "rollout_refresh_requests_cumulative",
    "score_refresh_requests_cumulative",
    "refresh_requests_cumulative",
    "priority_visits_added_cumulative",
)


def exported_environment_text(record: str) -> str:
    try:
        tokens = shlex.split(record, posix=True)
    except ValueError as error:
        raise ValueError(f"scheduler record tokenization failed: {error}") from error
    submit_indexes = [
        index for index, token in enumerate(tokens) if token.startswith("SubmitLine=")
    ]
    if len(submit_indexes) != 1:
        raise ValueError("scheduler record must contain exactly one SubmitLine")

    exports: list[str] = []
    index = submit_indexes[0]
    while index < len(tokens):
        token = tokens[index]
        if token == "--export":
            index += 1
            if index >= len(tokens) or tokens[index].startswith("--"):
                raise ValueError("scheduler --export argument lacks a value")
            exports.append(tokens[index])
        elif token.startswith("--export="):
            exports.append(token.split("=", 1)[1])
        index += 1
    if len(exports) != 1:
        raise ValueError("scheduler record must contain exactly one --export argument")
    if not exports[0]:
        raise ValueError("scheduler --export argument is empty")
    return exports[0]


def exported_environment(record: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in exported_environment_text(record).split(","):
        if item == "ALL":
            continue
        if "=" not in item:
            raise ValueError(f"malformed scheduler export: {item!r}")
        key, value = item.split("=", 1)
        if not key or key in result:
            raise ValueError(f"duplicate/empty scheduler export: {key!r}")
        result[key] = value
    return result


def expected_nodes(ledger: dict[str, Any]) -> dict[tuple[str, str], str]:
    result = {
        (str(row["scale"]), str(row["domain"])): str(row["node"])
        for row in ledger.get("sentinels", [])
    }
    for identity, node in (ledger.get("effective_nodes") or {}).items():
        try:
            scale, domain = str(identity).split("/", 1)
        except ValueError as error:
            raise ValueError(
                f"malformed effective node identity: {identity}"
            ) from error
        result[(scale, domain)] = str(node)
    if set(result) != set(e117.SENTINELS):
        raise ValueError("effective E117 node map does not cover four sentinels")
    return result


def validate_launch_blocks(
    ledger: dict[str, Any],
    accounting: dict[int, dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    nodes = expected_nodes(ledger)
    varying = {
        "SAVE_PATH",
        "RUN_STAMP",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY",
    }
    by_block: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    snapshot = Path(str(ledger.get("snapshot_root", "")))
    for run in ledger.get("runs", []):
        scale = str(run["scale"])
        domain = str(run["domain"])
        arm = str(run["arm"])
        job_id = int(run["job_id"])
        try:
            record = str(run["held_scheduler_record"])
            environment_text = exported_environment_text(record)
            environment = exported_environment(record)
        except ValueError as error:
            failures.append(f"{scale}/{domain}/{arm}: {error}")
            continue
        by_block[(scale, domain)][arm] = environment
        environment_sha256 = hashlib.sha256(environment_text.encode()).hexdigest()
        if environment_sha256 != run.get("scientific_environment_sha256"):
            failures.append(f"{scale}/{domain}/{arm}: frozen environment hash drifted")
        actual_node = str(accounting.get(job_id, {}).get("node_list", ""))
        if actual_node != nodes[(scale, domain)]:
            failures.append(
                f"{scale}/{domain}/{arm}: actual node {actual_node!r} != "
                f"registered {nodes[(scale, domain)]!r}"
            )
        expected_compute = "1" if arm == "c" else "0"
        expected_eta = "0.1" if arm == "f" else "0.0"
        if (
            environment.get(
                "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY"
            )
            != expected_compute
        ):
            failures.append(f"{scale}/{domain}/{arm}: admission flag drifted")
        if environment.get("OAT_ZERO_SEMANTIC_SHANNON_COEF") != expected_eta:
            failures.append(f"{scale}/{domain}/{arm}: semantic coefficient drifted")
        expected_fixed = {
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_SEED": str(e117.SEED),
            "OAT_ZERO_MAX_TRAIN": str(e117.TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(e117.PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(e117.PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(e117.TARGET_STEPS),
            "OAT_ZERO_SAVE_STEPS": str(e117.CHECKPOINT_INTERVAL),
            "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": str(e117.EVAL_DRAWS),
        }
        expected_fixed.update(e117.objective_for_arm(arm))
        drifted = {
            key: (environment.get(key), value)
            for key, value in expected_fixed.items()
            if environment.get(key) != value
        }
        if drifted:
            failures.append(
                f"{scale}/{domain}/{arm}: registered launch values drifted: {drifted}"
            )
    for (scale, domain), environments in by_block.items():
        if set(environments) != set(e117.ARMS):
            failures.append(f"{scale}/{domain}: launch block lacks C/P/F")
            continue
        common = {
            arm: {key: value for key, value in env.items() if key not in varying}
            for arm, env in environments.items()
        }
        if common["c"] != common["p"] or common["p"] != common["f"]:
            failures.append(f"{scale}/{domain}: scientific exports differ across C/P/F")
    return failures


def validate_frozen_provenance(ledger: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    checks = (
        (
            Path(str(ledger.get("snapshot_root", ""))) / "SNAPSHOT_IDENTITY.json",
            ledger.get("snapshot_identity_sha256"),
            "training snapshot identity",
        ),
        (
            Path(str(ledger.get("original_ledger", ""))),
            ledger.get("original_ledger_sha256"),
            "original E117 ledger",
        ),
    )
    for path, expected, label in checks:
        if not path.is_file():
            failures.append(f"{label} is absent: {path}")
        elif not isinstance(expected, str) or e117.digest(path) != expected:
            failures.append(f"{label} digest drifted")
    return failures


def validate_registered_design(ledger: dict[str, Any]) -> list[str]:
    expected = {
        "schema": "e117_same_plumbing_component_preflight_jobs_v1",
        "released": True,
        "seed": e117.SEED,
        "arms": list(e117.ARMS),
        "train_rows": e117.TRAIN_ROWS,
        "passes": e117.PASSES,
        "target_steps": e117.TARGET_STEPS,
        "checkpoint_interval_steps": e117.CHECKPOINT_INTERVAL,
        "evaluation_draws": e117.EVAL_DRAWS,
        "proposal_fixed_control_groups": e117.PROPOSAL_GROUPS,
        "proposal_max_attempts": e117.e111.PROPOSAL_MAX_ATTEMPTS,
        "proposal_temperature": e117.e111.PROPOSAL_TEMPERATURE,
        "replay_weight": e117.e111.REPLAY_WEIGHT,
        "semantic_coefficient_f": e117.SEMANTIC_COEFFICIENT,
        "efficacy_gate": False,
        "outcomes_inspected_for_release": False,
        "pointmaze": "excluded",
    }
    return [
        f"registered design drifted: {key}={ledger.get(key)!r} expected={value!r}"
        for key, value in expected.items()
        if ledger.get(key) != value
    ]


def _monotone(values: list[float]) -> bool:
    return all(right >= left for left, right in zip(values, values[1:]))


def audit_arm_rows(
    arm: str, arm_rows: dict[int, dict[str, Any]]
) -> tuple[dict[str, float], list[str]]:
    failures: list[str] = []
    eligible_updates = 0
    actuable_updates = 0
    active_updates = 0
    admitted = 0.0
    discarded = 0.0
    cumulative: list[float] = []
    retention_counters: dict[tuple[str, str], list[float]] = defaultdict(list)
    request_seeds: list[float] = []
    fixed_request_seeds: list[float] = []
    for step, row in sorted(arm_rows.items()):
        for leak_key in (
            "actor/counterfactual_fixed_control_rows_sent_to_ppo",
            "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo",
            "actor/counterfactual_proposal_transform_rows_sent_to_ppo",
            "actor/counterfactual_proposal_objective_outcome_delta",
            "actor/counterfactual_proposal_gold_support_feedback",
            "actor/counterfactual_proposal_eval_feedback",
            "actor/counterfactual_proposal_desired_mode_count_feedback",
        ):
            if metric(row, leak_key) != 0.0:
                failures.append(f"step={step} leakage {leak_key}")
        fixed_groups = metric(
            row, "actor/counterfactual_fixed_control_groups_generated"
        )
        fixed_rows = metric(row, "actor/counterfactual_fixed_control_rows_generated")
        fixed_budget = metric(
            row,
            "actor/counterfactual_fixed_control_charged_response_token_budget",
        )
        fixed_seed_min = metric(
            row, "actor/counterfactual_fixed_control_request_seed_min"
        )
        fixed_seed_max = metric(
            row, "actor/counterfactual_fixed_control_request_seed_max"
        )
        fixed_request_seeds.append(fixed_seed_min)
        consumed = metric(
            row,
            "actor/counterfactual_fixed_control_groups_consumed_by_explorer",
        )
        unused = metric(row, "actor/counterfactual_fixed_control_groups_discarded")
        proposal_groups = metric(row, "actor/counterfactual_proposal_groups_generated")
        proposal_rows = metric(row, "actor/counterfactual_proposal_rows_generated")
        neutral_ppo_rows = metric(row, "actor/counterfactual_proposal_neutral_ppo_rows")
        neutral_positive = metric(
            row,
            "actor/counterfactual_proposal_neutral_task_reward_positive_rows",
        )
        neutral_reward_mean = metric(row, "actor/rewards")
        for name, value in (
            ("fixed-control groups", fixed_groups),
            ("fixed-control rows", fixed_rows),
            ("fixed-control charged budget", fixed_budget),
            ("consumed fixed-control groups", consumed),
            ("unused fixed-control groups", unused),
            ("proposal groups", proposal_groups),
            ("proposal rows", proposal_rows),
            ("neutral PPO rows", neutral_ppo_rows),
            ("neutral task-positive rows", neutral_positive),
        ):
            if value < 0.0 or not value.is_integer():
                failures.append(f"step={step} invalid {name}: {value}")
        if fixed_groups != float(e117.PROPOSAL_GROUPS) or fixed_rows != 16.0:
            failures.append(f"step={step} fixed-control request shape drifted")
        if fixed_budget <= 0.0 or fixed_seed_min != fixed_seed_max:
            failures.append(f"step={step} fixed-control request identity drifted")
        if consumed + unused != fixed_groups or proposal_groups != consumed:
            failures.append(f"step={step} fixed-control consumption was inconsistent")
        if proposal_rows != proposal_groups * 16.0:
            failures.append(f"step={step} proposal row accounting drifted")
        if neutral_ppo_rows != 16.0 or not math.isclose(
            neutral_positive,
            neutral_reward_mean * neutral_ppo_rows,
            rel_tol=0.0,
            abs_tol=2e-6,
        ):
            failures.append(f"step={step} neutral PPO/task-reward count drifted")
        proposal_seed_keys = (
            "actor/counterfactual_proposal_request_seed_min",
            "actor/counterfactual_proposal_request_seed_max",
        )
        proposal_seed_present = [key in row for key in proposal_seed_keys]
        if proposal_groups > 0.0:
            if not all(proposal_seed_present):
                failures.append(f"step={step} proposal request seed was absent")
            else:
                proposal_seed_min = metric(row, proposal_seed_keys[0])
                proposal_seed_max = metric(row, proposal_seed_keys[1])
                if (
                    proposal_seed_min != fixed_seed_min
                    or proposal_seed_max != fixed_seed_max
                ):
                    failures.append(
                        f"step={step} proposal/fixed-control seed identity drifted"
                    )
        elif any(proposal_seed_present):
            failures.append(f"step={step} inactive proposal exposed request seeds")
        compute_only = metric(
            row, "actor/counterfactual_proposal_admission_compute_only"
        )
        if compute_only != (1.0 if arm == "c" else 0.0):
            failures.append(f"step={step} compute-only flag drift")
        candidates = metric(row, "actor/counterfactual_proposal_novel_unique_outcomes")
        row_admitted = metric(
            row, "actor/counterfactual_proposal_admitted_new_outcomes"
        )
        stored = metric(row, "actor/counterfactual_proposal_stored_exemplars")
        row_discarded = metric(
            row, "actor/counterfactual_proposal_candidates_discarded"
        )
        for name, value in (
            ("novel candidates", candidates),
            ("admissions", row_admitted),
            ("stored exemplars", stored),
            ("discarded candidates", row_discarded),
        ):
            if value < 0.0 or not value.is_integer():
                failures.append(f"step={step} invalid {name}: {value}")
        admitted += row_admitted
        discarded += row_discarded
        cumulative.append(
            metric(row, "actor/counterfactual_proposal_cumulative_new_outcomes")
        )
        if cumulative[-1] < 0.0 or not cumulative[-1].is_integer():
            failures.append(f"step={step} invalid cumulative admissions")
        request_seeds.append(metric(row, "actor/sampling_request_seed"))
        if arm == "c":
            if row_admitted != 0.0 or stored != 0.0 or row_discarded != candidates:
                failures.append(f"step={step} C did not discard exactly at admission")
            if cumulative[-1] != 0.0:
                failures.append(f"step={step} C proposal state mutated")
        else:
            if row_discarded != 0.0:
                failures.append(f"step={step} {arm} discarded admissions")
            if row_admitted != candidates or stored != candidates:
                failures.append(
                    f"step={step} {arm} did not admit/store every novel candidate"
                )
        if cumulative[-1] != admitted:
            failures.append(
                f"step={step} cumulative proposal counter disagrees with admissions"
            )

        for prefix_index, prefix in enumerate(RETENTION_PREFIXES):
            present = any(key.startswith(prefix) for key in row)
            if not present and prefix_index == 0:
                raise ValueError(f"required retention view is absent: {prefix}")
            if not present:
                continue
            view = {
                field: metric(row, prefix + field)
                for field in RETENTION_REQUIRED_FIELDS
            }
            if view["tracking_enabled"] != 1.0:
                failures.append(f"step={step} retention tracking disabled: {prefix}")
            for field in RETENTION_ZERO_FIELDS:
                if view[field] != 0.0:
                    failures.append(f"step={step} retention leakage {prefix + field}")
            for field in RETENTION_INTEGER_FIELDS:
                if view[field] < 0.0 or not view[field].is_integer():
                    failures.append(
                        f"step={step} invalid retention counter: {prefix + field}"
                    )
            for field in (
                "rollout_conversion_fraction",
                "rollout_row_frequency",
                "score_retained_fraction",
                "joint_retained_fraction",
            ):
                if not 0.0 <= view[field] <= 1.0:
                    failures.append(
                        f"step={step} invalid retention fraction: {prefix + field}"
                    )
            subset_pairs = (
                ("rollout_converted_admissions", "rollout_eligible_admissions"),
                ("rollout_eligible_admissions", "tracked_admissions"),
                ("score_followup_admissions", "score_observed_admissions"),
                ("score_observed_admissions", "tracked_admissions"),
                ("score_retained_admissions", "score_followup_admissions"),
                ("joint_eligible_admissions", "rollout_eligible_admissions"),
                ("joint_eligible_admissions", "score_followup_admissions"),
                ("joint_retained_admissions", "joint_eligible_admissions"),
            )
            for subset, superset in subset_pairs:
                if view[subset] > view[superset]:
                    failures.append(
                        f"step={step} inconsistent retention subsets: "
                        f"{prefix + subset}>{prefix + superset}"
                    )
            fraction_triples = (
                (
                    "rollout_conversion_fraction",
                    "rollout_converted_admissions",
                    "rollout_eligible_admissions",
                ),
                (
                    "score_retained_fraction",
                    "score_retained_admissions",
                    "score_followup_admissions",
                ),
                (
                    "joint_retained_fraction",
                    "joint_retained_admissions",
                    "joint_eligible_admissions",
                ),
            )
            for fraction, numerator, denominator in fraction_triples:
                expected_fraction = (
                    view[numerator] / view[denominator] if view[denominator] else 0.0
                )
                if not math.isclose(
                    view[fraction],
                    expected_fraction,
                    rel_tol=0.0,
                    abs_tol=2e-7,
                ):
                    failures.append(
                        f"step={step} inconsistent retention fraction: "
                        f"{prefix + fraction}"
                    )
            if view["refresh_requests_cumulative"] != (
                view["rollout_refresh_requests_cumulative"]
                + view["score_refresh_requests_cumulative"]
            ):
                failures.append(
                    f"step={step} inconsistent retention refresh total: {prefix}"
                )
            if arm == "c":
                for field in RETENTION_C_ZERO_FIELDS:
                    if view[field] != 0.0:
                        failures.append(
                            f"step={step} C retention state mutated: {prefix + field}"
                        )
            for field in RETENTION_MONOTONE_FIELDS:
                retention_counters[(prefix, field)].append(view[field])
        online_tracked = retention_counters[
            (RETENTION_PREFIXES[0], "tracked_admissions")
        ][-1]
        if arm == "c" and online_tracked != 0.0:
            failures.append(f"step={step} C tracked a proposal admission")
        if arm in ("p", "f") and online_tracked != cumulative[-1]:
            failures.append(f"step={step} {arm} retention/admission state disagreed")
        eligible = metric(
            row,
            "train/semantic_shannon_success_conditioned_verified_support_"
            "verified_support_at_least_two_eligible_fraction",
        )
        raw_rms = metric(
            row,
            "train/semantic_shannon_success_conditioned_verified_support_"
            "raw_eligible_advantage_rms",
        )
        rms = metric(
            row,
            "train/semantic_shannon_success_conditioned_verified_support_"
            "effective_advantage_rms",
        )
        if not 0.0 <= eligible <= 1.0:
            failures.append(f"step={step} invalid semantic eligible fraction")
        if raw_rms < 0.0 or rms < 0.0:
            failures.append(f"step={step} semantic RMS was negative")
        coefficient = metric(
            row,
            "train/semantic_shannon_success_conditioned_verified_support_"
            "open_set_coefficient_used",
        )
        expected_coefficient = e117.SEMANTIC_COEFFICIENT if arm == "f" else 0.0
        if not math.isclose(
            coefficient, expected_coefficient, rel_tol=0.0, abs_tol=2e-7
        ):
            failures.append(f"step={step} semantic coefficient drifted")
        if raw_rms > coefficient + 2e-7:
            failures.append(f"step={step} raw semantic RMS exceeded coefficient")
        if rms > raw_rms + 2e-7:
            failures.append(f"step={step} effective semantic RMS exceeded raw RMS")
        if arm in ("c", "p") and raw_rms != 0.0:
            failures.append(f"step={step} raw semantic pressure was nonzero")
        if eligible > 0:
            eligible_updates += 1
        if raw_rms > 0:
            actuable_updates += 1
        if rms > 0:
            active_updates += 1
        if arm in ("c", "p") and rms != 0.0:
            failures.append(f"step={step} semantic pressure was nonzero")
        if arm == "f" and raw_rms > 0 and rms <= 0:
            failures.append(f"step={step} actuable F update had zero semantic pressure")
    if len(set(request_seeds)) != len(request_seeds):
        failures.append("neutral request seeds repeated across optimizer steps")
    if len(set(fixed_request_seeds)) != len(fixed_request_seeds):
        failures.append("fixed-control request seeds repeated across optimizer steps")
    if not _monotone(cumulative):
        failures.append("proposal admission counter decreased across a resume")
    for (prefix, field), values in retention_counters.items():
        if values and not _monotone(values):
            failures.append(
                f"retention counter decreased across a resume: {prefix + field}"
            )
    return {
        "proposal_admissions": admitted,
        "proposal_candidates_discarded": discarded,
        "semantic_eligible_updates": float(eligible_updates),
        "semantic_actuable_updates": float(actuable_updates),
        "semantic_active_updates": float(active_updates),
        "terminal_proposal_admissions": cumulative[-1],
        "terminal_tracked_admissions": retention_counters[
            (RETENTION_PREFIXES[0], "tracked_admissions")
        ][-1],
    }, failures


def _integer_training_step(row: dict[str, Any], key: str) -> int:
    if key not in row:
        raise ValueError(f"required E117 step field is absent: {key}")
    value = row[key]
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or int(value) != value
    ):
        raise ValueError(f"invalid E117 step field {key}: {value!r}")
    return int(value)


def _validate_terminal_training_sentinel(
    canonical: dict[str, Any],
    sentinel: dict[str, Any],
) -> None:
    for key in (
        "trainer/global_step",
        "misc/policy_sgd_step",
        "misc/query_step",
        "misc/prompt_consumed",
        "misc/pi_beta_version",
    ):
        if key not in canonical or key not in sentinel:
            raise ValueError(f"terminal E117 sentinel lacks state field: {key}")
        if canonical[key] != sentinel[key]:
            raise ValueError(f"terminal E117 sentinel mutated state field: {key}")
    mechanism_keys = sorted(
        key
        for key in set(canonical) | set(sentinel)
        if key.startswith(("actor/", "train/"))
    )
    for key in mechanism_keys:
        if key not in canonical or key not in sentinel:
            raise ValueError(
                f"terminal E117 sentinel changed mechanism field presence: {key}"
            )
        if canonical[key] != sentinel[key]:
            raise ValueError(f"terminal E117 sentinel mutated mechanism field: {key}")


def training_rows(debug: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    terminal_sentinels: list[dict[str, Any]] = []
    for row in rows(debug / "train_metrics.jsonl"):
        if row.get("misc/global_step") is None:
            continue
        step = _integer_training_step(row, "misc/global_step")
        if step <= 0 or "actor/sampling_request_seed" not in row:
            continue
        trainer_step = _integer_training_step(row, "trainer/step")
        trainer_global_step = _integer_training_step(row, "trainer/global_step")
        policy_sgd_step = _integer_training_step(row, "misc/policy_sgd_step")
        if trainer_global_step != step or policy_sgd_step != step:
            raise ValueError(
                "E117 optimizer-step identity drifted: "
                f"misc={step} trainer_global={trainer_global_step} "
                f"policy_sgd={policy_sgd_step}"
            )
        if trainer_step == step:
            if step in result:
                raise ValueError(
                    f"duplicate canonical E117 optimizer row at step {step}"
                )
            result[step] = row
            continue
        if step == e117.TARGET_STEPS and trainer_step == step + 1:
            terminal_sentinels.append(row)
            continue
        raise ValueError(
            "unrecognized E117 training row: "
            f"global_step={step} trainer_step={trainer_step}"
        )
    expected = set(range(1, e117.TARGET_STEPS + 1))
    if set(result) != expected:
        raise ValueError(
            "E117 training-step coverage drifted: missing={} extra={}".format(
                sorted(expected - set(result)), sorted(set(result) - expected)
            )
        )
    if len(terminal_sentinels) != 1:
        raise ValueError(
            "E117 expected exactly one validated terminal bookkeeping row, "
            f"found {len(terminal_sentinels)}"
        )
    _validate_terminal_training_sentinel(
        result[e117.TARGET_STEPS], terminal_sentinels[0]
    )
    return result


def coverage_draws(debug: Path) -> dict[int, list[dict[str, Any]]]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows(debug / "eval_mode_coverage_draws.jsonl"):
        raw_step = row.get("step")
        if (
            not isinstance(raw_step, (int, float))
            or isinstance(raw_step, bool)
            or not math.isfinite(float(raw_step))
            or int(raw_step) != raw_step
        ):
            raise ValueError(f"E117 evaluation row has invalid step: {raw_step!r}")
        by_step[int(raw_step)].append(row)
    expected_steps = {0, e117.TARGET_STEPS}
    if set(by_step) != expected_steps:
        raise ValueError(f"E117 evaluation-step coverage drifted: {sorted(by_step)}")
    for step, selected in by_step.items():
        sampled = [
            row
            for row in selected
            if row.get("evaluation_kind") == "fixed_seed_sampled_k_neutral"
        ]
        greedy = [
            row
            for row in selected
            if row.get("evaluation_kind") == "deterministic_greedy_trace_neutral"
        ]
        if len(sampled) != e117.EVAL_DRAWS:
            raise ValueError(
                f"E117 expected {e117.EVAL_DRAWS} sampled draws at step {step}, "
                f"found {len(sampled)}"
            )
        indices = [row.get("draw_index") for row in sampled]
        if any(type(value) is not int for value in indices):
            raise ValueError(
                f"E117 draw indices are not integers at step {step}: {indices!r}"
            )
        if indices != list(range(e117.EVAL_DRAWS)):
            raise ValueError(f"E117 draw indices drifted at step {step}: {indices!r}")
        if len(greedy) != 1 or greedy[0].get("draw_index") is not None:
            raise ValueError(f"E117 greedy companion is incomplete at step {step}")
        if len(selected) != len(sampled) + len(greedy):
            raise ValueError(f"E117 evaluation kind drifted at step {step}")
        for row in selected:
            if not isinstance(row.get("metrics"), dict) or not row["metrics"]:
                raise ValueError(
                    f"E117 evaluation metrics are incomplete at step {step}"
                )
            if (
                not isinstance(row.get("prompts"), list)
                or not row["prompts"]
                or any(not isinstance(prompt, dict) for prompt in row["prompts"])
            ):
                raise ValueError(f"E117 evaluation output is incomplete at step {step}")
    return dict(by_step)


def step_zero_draws(debug: Path) -> list[dict[str, Any]]:
    return coverage_draws(debug)[0]


def evaluation_request_metadata(
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key not in ("metrics", "prompts")}
        for row in selected
    ]


def exact_values(
    arm_rows: dict[str, dict[int, dict[str, Any]]],
    *,
    step: int,
    keys: tuple[str, ...],
) -> list[str]:
    failures: list[str] = []
    for key in keys:
        values: dict[str, float] = {}
        invalid = False
        for arm in e117.ARMS:
            try:
                values[arm] = metric(arm_rows[arm][step], key)
            except ValueError as error:
                failures.append(f"step={step} arm={arm} {error}")
                invalid = True
        if not invalid and len(set(values.values())) != 1:
            failures.append(f"step={step} key={key} values={values}")
    return failures


def conditional_exact_values(
    arm_rows: dict[str, dict[int, dict[str, Any]]],
    *,
    step: int,
    keys: tuple[str, ...],
) -> list[str]:
    failures: list[str] = []
    for key in keys:
        present = {arm: key in arm_rows[arm][step] for arm in e117.ARMS}
        if not any(present.values()):
            continue
        if not all(present.values()):
            failures.append(f"step={step} conditional key={key} presence={present}")
            continue
        failures.extend(exact_values(arm_rows, step=step, keys=(key,)))
    return failures


def stage1_execution_readiness(
    block_summaries: list[dict[str, Any]], *, identity_passed: bool
) -> dict[str, Any]:
    proposal_blocks = [
        f"{block['scale']}/{block['domain']}"
        for block in block_summaries
        if block["arms"].get("p", {}).get("proposal_admissions", 0.0) > 0.0
    ]
    semantic_blocks = [
        f"{block['scale']}/{block['domain']}"
        for block in block_summaries
        if block["arms"].get("f", {}).get("semantic_active_updates", 0.0) > 0.0
    ]
    blockers: list[str] = []
    if not identity_passed:
        blockers.append("mechanism identity/leakage audit failed")
    if not proposal_blocks:
        blockers.append("P admission was not exercised in any sentinel")
    if not semantic_blocks:
        blockers.append("F semantic pressure was not exercised in any sentinel")
    return {
        "proposal_replay_exercised_blocks": proposal_blocks,
        "semantic_pressure_exercised_blocks": semantic_blocks,
        "ready": not blockers,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = e117.repo_root()
    ledger_path = (args.ledger or (root / e117.LEDGER)).resolve()
    output_path = (args.output or (root / AUDIT)).resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    runs = list(ledger.get("runs", []))
    if len(runs) != 12:
        raise SystemExit("E117 release ledger is incomplete or malformed")
    design_failures = validate_registered_design(ledger)
    identities = {
        (str(run["scale"]), str(run["domain"]), str(run["arm"])) for run in runs
    }
    expected_identities = {
        (scale, domain, arm) for scale, domain in e117.SENTINELS for arm in e117.ARMS
    }
    if identities != expected_identities:
        raise SystemExit("E117 ledger cell identity drifted")

    accounting = scheduler_status([int(run["job_id"]) for run in runs])
    nonterminal = {
        int(run["job_id"]): accounting.get(int(run["job_id"]), {"state": "UNKNOWN"})
        for run in runs
        if accounting.get(int(run["job_id"]), {}).get("state") != "COMPLETED"
        or accounting.get(int(run["job_id"]), {}).get("exit_code") != "0:0"
    }
    if nonterminal:
        payload = {
            "schema": "e117_same_plumbing_component_preflight_audit_v3",
            "ledger": str(ledger_path),
            "ledger_sha256": e117.digest(ledger_path),
            "terminal": False,
            "efficacy_outcomes_used": False,
            "endpoint_contrasts_computed": False,
            "pointmaze": "excluded",
            "mechanism_blocks": [],
            "resume_exercised": any(
                int(row.get("restarts", 0)) > 0 for row in accounting.values()
            ),
            "resume_continuity": "not_auditable",
            "resumed_jobs": {
                str(job_id): row["restarts"]
                for job_id, row in accounting.items()
                if int(row.get("restarts", 0)) > 0
            },
            "stage1_execution_readiness": {
                "proposal_replay_exercised_blocks": [],
                "semantic_pressure_exercised_blocks": [],
                "ready": False,
                "blockers": ["all 12 training jobs did not complete successfully"],
            },
            "noncompleted_jobs": nonterminal,
            "failures": ["all 12 training jobs did not complete successfully"],
            "passed": False,
        }
        e117.e111.e81.atomic_json(output_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    artifacts: dict[tuple[str, str, str], dict[str, Any]] = {}
    failures: list[str] = list(design_failures)
    failures.extend(validate_frozen_provenance(ledger))
    try:
        failures.extend(validate_launch_blocks(ledger, accounting))
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"launch-block audit could not be completed: {error}")
    for run in runs:
        identity = (str(run["scale"]), str(run["domain"]), str(run["arm"]))
        run_dir = Path(str(run["run_dir"]))
        if not (run_dir / "TRAINING_COMPLETE.json").is_file():
            failures.append(f"{identity}: TRAINING_COMPLETE.json absent")
            continue
        debug_matches = list(run_dir.glob(f"debug_job{int(run['job_id'])}"))
        if len(debug_matches) != 1:
            failures.append(f"{identity}: expected one debug directory")
            continue
        debug = debug_matches[0]
        try:
            evaluation = coverage_draws(debug)
            artifacts[identity] = {
                "debug": str(debug),
                "train": training_rows(debug),
                "evaluation": evaluation,
                "step_zero": evaluation[0],
            }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"{identity}: {error}")

    all_step_identity_keys = (
        "actor/sampling_request_seed",
        "actor/counterfactual_fixed_control_groups_generated",
        "actor/counterfactual_fixed_control_rows_generated",
        "actor/counterfactual_fixed_control_charged_response_token_budget",
        "actor/counterfactual_fixed_control_rows_sent_to_ppo",
        "actor/counterfactual_fixed_control_request_seed_min",
        "actor/counterfactual_fixed_control_request_seed_max",
        "actor/counterfactual_fixed_control_sampling_temperature_min",
        "actor/counterfactual_fixed_control_sampling_temperature_max",
    )
    preintervention_keys = all_step_identity_keys + (
        "actor/rewards",
        "actor/counterfactual_fixed_control_groups_consumed_by_explorer",
        "actor/counterfactual_fixed_control_groups_discarded",
        "actor/counterfactual_proposal_groups_generated",
        "actor/counterfactual_proposal_rows_generated",
        "actor/counterfactual_proposal_task_reward_positive_rows",
        "actor/counterfactual_proposal_validator_positive_rows",
        "actor/counterfactual_proposal_novel_candidate_rows",
        "actor/counterfactual_proposal_novel_unique_outcomes",
    )
    block_summaries: list[dict[str, Any]] = []
    for scale, domain in e117.SENTINELS:
        if any((scale, domain, arm) not in artifacts for arm in e117.ARMS):
            continue
        by_arm = {arm: artifacts[(scale, domain, arm)] for arm in e117.ARMS}
        step_zero = {arm: by_arm[arm]["step_zero"] for arm in e117.ARMS}
        if (
            len({json.dumps(value, sort_keys=True) for value in step_zero.values()})
            != 1
        ):
            failures.append(f"{scale}/{domain}: step-zero coverage draws differ")
        terminal_metadata = {
            arm: evaluation_request_metadata(
                by_arm[arm]["evaluation"][e117.TARGET_STEPS]
            )
            for arm in e117.ARMS
        }
        if (
            len(
                {
                    json.dumps(value, sort_keys=True)
                    for value in terminal_metadata.values()
                }
            )
            != 1
        ):
            failures.append(
                f"{scale}/{domain}: terminal evaluation request metadata differ"
            )
        arm_rows = {arm: by_arm[arm]["train"] for arm in e117.ARMS}
        failures.extend(
            f"{scale}/{domain}: {value}"
            for value in exact_values(arm_rows, step=1, keys=preintervention_keys)
        )
        failures.extend(
            f"{scale}/{domain}: {value}"
            for value in conditional_exact_values(
                arm_rows,
                step=1,
                keys=(
                    "actor/counterfactual_proposal_request_seed_min",
                    "actor/counterfactual_proposal_request_seed_max",
                ),
            )
        )
        for step in range(1, e117.TARGET_STEPS + 1):
            failures.extend(
                f"{scale}/{domain}: {value}"
                for value in exact_values(
                    arm_rows, step=step, keys=all_step_identity_keys
                )
            )

        activation: dict[str, dict[str, float]] = {}
        for arm in e117.ARMS:
            try:
                summary, arm_failures = audit_arm_rows(arm, arm_rows[arm])
            except (IndexError, KeyError, TypeError, ValueError) as error:
                summary = {}
                arm_failures = [str(error)]
            failures.extend(
                f"{scale}/{domain}/{arm}: {value}" for value in arm_failures
            )
            activation[arm] = summary
        block_summaries.append(
            {"scale": scale, "domain": domain, "arms": dict(activation)}
        )

    resumed = {
        str(job_id): row["restarts"]
        for job_id, row in accounting.items()
        if int(row["restarts"]) > 0
    }
    identity_passed = not failures
    readiness = stage1_execution_readiness(
        block_summaries, identity_passed=identity_passed
    )
    payload = {
        "schema": "e117_same_plumbing_component_preflight_audit_v3",
        "ledger": str(ledger_path),
        "ledger_sha256": e117.digest(ledger_path),
        "terminal": True,
        "efficacy_outcomes_used": False,
        "endpoint_contrasts_computed": False,
        "pointmaze": "excluded",
        "mechanism_blocks": block_summaries,
        "resume_exercised": bool(resumed),
        "resume_continuity": (
            "verified"
            if resumed and identity_passed
            else "failed"
            if resumed
            else "unexercised"
        ),
        "resumed_jobs": resumed,
        "stage1_execution_readiness": readiness,
        "failures": failures,
        "passed": identity_passed,
    }
    e117.e111.e81.atomic_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
