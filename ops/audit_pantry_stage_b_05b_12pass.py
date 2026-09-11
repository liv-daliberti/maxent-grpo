#!/usr/bin/env python3
"""Fail-closed terminal audit for PantryPlan's ten Stage-B jobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEEDS = (43, 44, 45, 46, 47)
UPDATES = 384
EVAL_INTERVAL = 8
EVAL_DRAWS = 4
PREFIX = "ppe70_clean_stage_b_05b_12pass"
QUALIFICATION = "pantry_support_mask_paired_integration_v3_audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(f"{sha(path)}  ./{path.relative_to(root).as_posix()}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_dir(root: Path, stamp: str, job_id: int) -> Path | None:
    found = list((root / "var/data").glob(f"*_{stamp}/debug_job{job_id}"))
    return found[0] if len(found) == 1 else None


def scheduler_terminal(job_id: int) -> bool:
    output = subprocess.run(
        ["sacct", "-X", "-j", str(job_id), "--format=State,ExitCode", "-n", "-P"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return "COMPLETED|0:0" in output


def main() -> None:
    parsed = parse_args()
    if parsed.output.exists():
        raise FileExistsError("fresh Pantry Stage-B audit output required")
    root = parsed.repo_root.resolve()
    identity = json.loads(parsed.identity.read_text())
    submission = json.loads(parsed.submission.read_text())
    rows = list(csv.DictReader(parsed.manifest.open(), delimiter="\t"))
    errors: list[str] = []
    checks: dict[str, bool] = {}
    expected = {(arm, str(seed)) for arm in ARMS for seed in SEEDS}
    observed = {(row["arm"], row["seed"]) for row in rows}
    checks["manifest_exact_ten_cells"] = observed == expected and len(rows) == 10
    jobs = {
        f"{row['arm']}/s{row['seed']}": int(row["job_id"])
        for row in rows
    }
    checks["identity_contract"] = (
        identity.get("schema") == "pantry-stage-b-05b-12pass-v1"
        and identity.get("arms") == list(ARMS)
        and identity.get("seeds") == list(SEEDS)
        and identity.get("optimizer_updates") == UPDATES
        and identity.get("prompt_passes") == 12
        and identity.get("evaluation_interval_updates") == EVAL_INTERVAL
        and identity.get("evaluation_draws") == EVAL_DRAWS
        and identity.get("jobs") == jobs
        and identity.get("final_seed_cohort") is True
    )
    checks["submission_contract"] = (
        submission.get("schema") == "pantry-stage-b-05b-12pass-submission-v1"
        and submission.get("identity_sha256") == sha(parsed.identity)
        and submission.get("manifest_sha256") == sha(parsed.manifest)
        and submission.get("jobs") == jobs
        and submission.get("released") is True
    )
    checks["source_snapshot"] = (
        tree_hash(Path(identity["source_root"])) == identity.get("source_hash")
    )
    checks["ops_snapshot"] = (
        tree_hash(Path(identity["ops_root"])) == identity.get("execution_hash")
    )
    checks["data_snapshot"] = (
        tree_hash(root / "var/data/pantry_plan_modebench_v2")
        == identity.get("data_tree_sha256")
    )
    qualification = root / "var/artifacts" / QUALIFICATION
    qualification_payload = json.loads(qualification.read_text())
    checks["qualification_pass"] = (
        qualification_payload.get("status") == "pass"
        and qualification_payload.get("decision") == "eligible_for_ten_stage_b_jobs"
        and sha(qualification) == identity.get("qualification_audit_sha256")
    )

    cell_results: dict[str, Any] = {}
    traversal: dict[int, dict[str, list[tuple[int, int]]]] = {
        seed: {} for seed in SEEDS
    }
    expected_eval_rounds = {0, *range(EVAL_INTERVAL, UPDATES + 1, EVAL_INTERVAL)}
    for row in rows:
        arm = row["arm"]
        seed = int(row["seed"])
        job_id = int(row["job_id"])
        label = f"{arm}/s{seed}/j{job_id}"
        if not scheduler_terminal(job_id):
            errors.append(f"{label}: scheduler not complete")
        directory = run_dir(root, row["run_stamp"], job_id)
        if directory is None:
            errors.append(f"{label}: unique run directory absent")
            continue
        metrics = directory / "train_metrics.jsonl"
        if not metrics.is_file():
            errors.append(f"{label}: metrics absent")
            continue
        metric_rows = [
            json.loads(line)
            for line in metrics.read_text().splitlines()
            if line.strip()
        ]
        update_rows = metric_rows[1 : UPDATES + 1]
        rounds = [int(item.get("train/learning_round", -1)) for item in update_rows]
        if len(metric_rows) != UPDATES + 2 or rounds != list(range(1, UPDATES + 1)):
            errors.append(f"{label}: exact {UPDATES}-update stream absent")
        evaluation_rounds: set[int] = set()
        positive = multimode = semantic_nonzero = novelty_nonzero = 0
        replay_eligible = applied_replay = raw_replay = 0
        traversal[seed][arm] = []
        for index, item in enumerate(metric_rows[:-1]):
            learning_round = int(item.get("train/learning_round", 0))
            if "eval/multi_answer/sampled_any_correct_at_8" in item:
                evaluation_rounds.add(learning_round)
                if item.get("eval/multi_answer/sampled_any_correct_at_8_draw_count") != EVAL_DRAWS:
                    errors.append(f"{label}/eval{learning_round}: draw count mismatch")
                for draw in range(EVAL_DRAWS):
                    for stem in (
                        "sampled_any_correct_at_8",
                        "sampled_mean_at_8",
                        "sampled_distinct_correct_at_8",
                    ):
                        key = f"eval/multi_answer/{stem}_draw_{draw}"
                        if not finite(item.get(key)):
                            errors.append(f"{label}/eval{learning_round}: missing {key}")
            if index == 0:
                continue
            step = index
            for key, expected_value in {
                "actor/canonical_pantry_support_mask_actions": 1.0,
                "actor/canonical_action_count": 6.0,
                "actor/canonical_action_support_size": 2.0,
                "actor/canonical_sequence_support_size": 64.0,
                "train/canonical_exact_leaf_count": 64.0,
                "train/canonical_exact_prefix_row_count": 63.0,
                "train/canonical_replay_gold_support_feedback": 0.0,
                "train/canonical_replay_alpha_projection_active": 0.0,
                "train/canonical_replay_global_scheduler_active": 1.0,
                "train/canonical_replay_global_groups_per_step": 1.0,
                "train/online_canonical_actor_positive_validator_negative_rows": 0.0,
                "train/online_canonical_validator_positive_actor_negative_rows": 0.0,
                "train/online_canonical_validator_task_disagreement_rows": 0.0,
            }.items():
                if not finite(item.get(key)) or not math.isclose(
                    float(item[key]), expected_value, abs_tol=1e-7
                ):
                    errors.append(f"{label}/step{step}: {key} mismatch")
            for key, value in item.items():
                if key.startswith("train/") and isinstance(value, (int, float)) and not finite(value):
                    errors.append(f"{label}/step{step}: nonfinite {key}")
                if key.startswith("train/") and key.endswith(("_nan", "_inf")) and finite(value) and float(value) != 0:
                    errors.append(f"{label}/step{step}: nonzero {key}")
            positive += float(item.get("actor/rewards", 0.0)) > 0
            multimode += float(item.get("train/online_canonical_support_at_least_two_prompt_fraction", 0.0)) > 0
            semantic_nonzero += float(item.get("train/semantic_shannon_separate_semantic_advantage_rms", 0.0)) > 0
            novelty_nonzero += float(item.get("train/online_canonical_novelty_advantage_rms", 0.0)) > 0
            eligible = float(item.get("train/canonical_replay_eligible_groups", 0.0))
            applied = float(item.get("train/canonical_replay_applied_score_gradient_l2", 0.0))
            raw = max(
                float(item.get("train/canonical_replay_mass_score_gradient_l2", 0.0)),
                float(item.get("train/canonical_replay_balance_score_gradient_l2", 0.0)),
                abs(float(item.get("train/canonical_replay_raw_weighted_loss", 0.0))),
            )
            replay_eligible += eligible > 0
            applied_replay += applied > 0
            raw_replay += raw > 0
            traversal[seed][arm].append((
                int(item.get("train/canonical_replay_score_passes", 0)),
                int(item.get("train/canonical_replay_charged_response_token_budget", 0)),
            ))
            compute_only = float(item.get(
                "train/canonical_replay_compute_only_configured",
                item.get("train/canonical_replay_compute_only", -1),
            ))
            if arm == CONTROL and eligible > 0 and (compute_only != 1 or applied != 0):
                errors.append(f"{label}/step{step}: control replay derivative not zero")
            if arm == TREATMENT and eligible > 0 and compute_only != 0:
                errors.append(f"{label}/step{step}: treatment unexpectedly compute-only")
        if evaluation_rounds != expected_eval_rounds:
            errors.append(f"{label}: exact quarter-pass evaluation cadence absent")
        if positive == 0:
            errors.append(f"{label}: no positive verified reward")
        if multimode == 0:
            errors.append(f"{label}: no online two-mode prompt")
        if arm == TREATMENT and semantic_nonzero + novelty_nonzero == 0:
            errors.append(f"{label}: no applied exploration advantage")
        if arm == TREATMENT and replay_eligible > 0 and applied_replay == 0:
            errors.append(f"{label}: eligible replay never applied")
        if arm == CONTROL and replay_eligible > 0 and raw_replay == 0:
            errors.append(f"{label}: compute control lacks raw replay telemetry")
        stdout = root / f"var/artifacts/logs/xdr_train-{job_id}.out"
        text = stdout.read_text(errors="replace") if stdout.is_file() else ""
        if "Traceback (most recent call last):" in text or f"[slurm] job_id={job_id}" not in text:
            errors.append(f"{label}: terminal stdout contract failed")
        cell_results[f"{arm}/s{seed}"] = {
            "job_id": job_id,
            "metrics_sha256": sha(metrics),
            "stdout_sha256": sha(stdout),
            "positive_reward_updates": positive,
            "multimode_updates": multimode,
            "semantic_nonzero_updates": semantic_nonzero,
            "novelty_nonzero_updates": novelty_nonzero,
            "replay_eligible_updates": replay_eligible,
            "applied_replay_nonzero_updates": applied_replay,
            "raw_replay_nonzero_updates": raw_replay,
            "evaluation_coordinates": len(evaluation_rounds),
        }
    checks["compute_traversal_match_by_seed"] = all(
        set(traversal[seed]) == set(ARMS)
        and traversal[seed][CONTROL] == traversal[seed][TREATMENT]
        for seed in SEEDS
    )
    for key, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {key}")
    payload = {
        "schema": "pantry-stage-b-05b-12pass-audit-v1",
        "status": "pass" if not errors else "fail",
        "decision": "pantry_terminal_eligible" if not errors else "pantry_stage_b_stopped",
        "checks": checks,
        "cells": cell_results,
        "errors": sorted(set(errors)),
        "evidence": {
            "identity_sha256": sha(parsed.identity),
            "submission_sha256": sha(parsed.submission),
            "manifest_sha256": sha(parsed.manifest),
        },
    }
    atomic(parsed.output, payload)
    print(json.dumps({"status": payload["status"], "errors": len(payload["errors"])}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
