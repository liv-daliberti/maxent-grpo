#!/usr/bin/env python3
"""Fail-closed audit for the two-cell PointMaze algorithm-repair pair."""

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
SEED = 76501
UPDATES = 96
EVAL_ROUNDS = tuple(range(0, 97, 2))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scheduler_complete(job_id: int) -> bool:
    output = subprocess.run(
        ["sacct", "-X", "-j", str(job_id), "--format=State,ExitCode", "-n", "-P"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return "COMPLETED|0:0" in output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("fresh PointMaze repair-pair audit required")
    root = args.repo_root.resolve()
    identity = json.loads(args.identity.read_text())
    submission = json.loads(args.submission.read_text())
    rows = list(csv.DictReader(args.manifest.open(), delimiter="\t"))
    errors: list[str] = []
    checks: dict[str, bool] = {}
    observed = {(row["arm"], row["seed"]) for row in rows}
    checks["manifest_exact_pair"] = observed == {
        (CONTROL, str(SEED)),
        (TREATMENT, str(SEED)),
    } and len(rows) == 2
    jobs = {f"{row['arm']}/s{row['seed']}": int(row["job_id"]) for row in rows}
    checks["identity_contract"] = (
        identity.get("schema") == "point-maze-stage-b-05b-12pass-identity-v1"
        and identity.get("repair_schema")
        == "point-maze-algorithm-repair-pair-identity-v1"
        and identity.get("arms") == list(ARMS)
        and identity.get("seeds") == [SEED]
        and identity.get("jobs") == jobs
        and identity.get("optimizer_updates") == UPDATES
        and identity.get("prompt_passes") == 12
        and identity.get("development_only") is True
        and identity.get("final_seed_cohort") is False
        and identity.get("evaluation_common_random_numbers") is True
    )
    checks["submission_contract"] = (
        submission.get("schema") == "point-maze-stage-b-05b-12pass-submission-v1"
        and submission.get("repair_schema")
        == "point-maze-algorithm-repair-pair-submission-v1"
        and submission.get("identity_sha256") == sha(args.identity)
        and submission.get("manifest_sha256") == sha(args.manifest)
        and submission.get("jobs") == jobs
        and submission.get("released") is True
    )
    checks["source_snapshot"] = (
        tree_hash(Path(identity["source_root"])) == identity.get("source_hash")
    )
    checks["ops_snapshot"] = (
        tree_hash(Path(identity["execution_root"])) == identity.get("execution_hash")
    )
    checks["model_snapshot"] = (
        tree_hash(root / "var/models/point_maze_interactive_warmstart_v3")
        == identity.get("model_tree_sha256")
    )
    checks["data_snapshot"] = (
        tree_hash(root / "var/data/point_maze_algorithm_repair_v1")
        == identity.get("data_tree_sha256")
    )
    qualification = (
        root / "var/artifacts/point_maze_algorithm_repair_v1_qualification.json"
    )
    checks["qualification_pass"] = (
        qualification.is_file()
        and sha(qualification) == identity.get("qualification_audit_sha256")
        and json.loads(qualification.read_text()).get("status") == "pass"
    )

    cell_results: dict[str, Any] = {}
    traversal: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        arm = row["arm"]
        seed = int(row["seed"])
        job_id = int(row["job_id"])
        label = f"{arm}/s{seed}/j{job_id}"
        if not scheduler_complete(job_id):
            errors.append(f"{label}: scheduler not complete")
        receipt_path = root / row["receipt"]
        metrics_path = root / row["metrics"]
        replay_path = root / row["state_replay"]
        if not all(path.is_file() for path in (receipt_path, metrics_path, replay_path)):
            errors.append(f"{label}: terminal artifact absent")
            continue
        receipt = json.loads(receipt_path.read_text())
        metrics = read_jsonl(metrics_path)
        replay = read_jsonl(replay_path)
        training = [
            item
            for item in metrics
            if item.get("schema") == "point-maze-stage-b-training-metric-v1"
        ]
        evaluations = [
            item
            for item in metrics
            if item.get("schema") == "point-maze-algorithm-repair-evaluation-v1"
        ]
        if [item.get("learning_round") for item in training] != list(
            range(1, UPDATES + 1)
        ):
            errors.append(f"{label}: exact 96-round training stream absent")
        if [item.get("learning_round") for item in evaluations] != list(EVAL_ROUNDS):
            errors.append(f"{label}: exact 49-coordinate evaluation stream absent")
        if len(replay) != UPDATES:
            errors.append(f"{label}: exact 96-row replay stream absent")
        verified = sum(float(item.get("verified_episodes", 0)) for item in training)
        verified_rate = verified / (UPDATES * 16)
        task_updates = sum(
            float(item.get("task_advantage_rms", 0)) > 0 for item in training
        )
        if not (0.10 <= verified_rate <= 0.90):
            errors.append(f"{label}: verified rate outside [0.10, 0.90]")
        if task_updates < 24:
            errors.append(f"{label}: fewer than 24 task-gradient updates")
        traversal[arm] = []
        raw_exploration = applied_exploration = raw_replay = applied_replay = 0
        for update, item in enumerate(training, start=1):
            for key, expected in {
                "fixed_policy_slots": 1536,
                "replay_decision_forward_slots": 1536,
                "policy_microbatch_size": 16,
                "action_support_escapes": 0,
                "optimizer_step": update,
            }.items():
                if item.get(key) != expected:
                    errors.append(f"{label}/u{update}: {key} mismatch")
            if any(
                isinstance(value, (int, float)) and not finite(value)
                for value in item.values()
            ):
                errors.append(f"{label}/u{update}: nonfinite training metric")
            raw_e = float(item.get("raw_exploration_advantage_rms", 0))
            applied_e = float(item.get("applied_exploration_advantage_rms", 0))
            raw_r = float(item.get("replay_raw_score_gradient_l2", 0))
            applied_r = float(item.get("replay_applied_score_gradient_l2", 0))
            raw_exploration += raw_e > 0
            applied_exploration += applied_e > 0
            raw_replay += raw_r > 0
            applied_replay += applied_r > 0
            if arm == CONTROL and (
                applied_e != 0
                or applied_r != 0
                or float(item.get("replay_compute_only", -1)) != 1
            ):
                errors.append(f"{label}/u{update}: control derivative mismatch")
            if arm == TREATMENT and float(item.get("replay_compute_only", -1)) != 0:
                errors.append(f"{label}/u{update}: treatment is compute-only")
            traversal[arm].append(
                (
                    int(item.get("fixed_policy_slots", -1)),
                    int(item.get("replay_decision_forward_slots", -1)),
                )
            )
        if arm == TREATMENT and (applied_exploration == 0 or applied_replay == 0):
            errors.append(f"{label}: treatment mechanism never applied")
        if arm == CONTROL and (applied_exploration != 0 or applied_replay != 0):
            errors.append(f"{label}: control mechanism derivative nonzero")
        for evaluation in evaluations:
            if (
                evaluation.get("evaluation_request_seed_schedule")
                != "arm_seed_row_draw_sample_decision__checkpoint_invariant"
                or evaluation.get("evaluation_draw_count") != 4
                or evaluation.get("evaluation_trajectory_count") != 132
            ):
                errors.append(f"{label}: common-random-number evaluation mismatch")
            for metric in ("greedy", "mean8", "pass8", "distinct8"):
                if not finite(evaluation.get(metric)):
                    errors.append(f"{label}: missing evaluation {metric}")
        if (
            receipt.get("schema") != "point-maze-algorithm-repair-receipt-v1"
            or receipt.get("status") != "complete"
            or receipt.get("arm") != arm
            or receipt.get("seed") != seed
            or receipt.get("job_id") != job_id
            or receipt.get("metrics_sha256") != sha(metrics_path)
            or receipt.get("state_replay_sha256") != sha(replay_path)
            or receipt.get("counts", {}).get("optimizer_updates") != UPDATES
            or receipt.get("counts", {}).get("evaluation_coordinates") != len(
                EVAL_ROUNDS
            )
            or receipt.get("evaluation_request_seed_schedule")
            != "arm_seed_row_draw_sample_decision__checkpoint_invariant"
        ):
            errors.append(f"{label}: terminal receipt mismatch")
        cell_results[arm] = {
            "job_id": job_id,
            "verified_rate": verified_rate,
            "task_advantage_nonzero_updates": task_updates,
            "raw_exploration_updates": raw_exploration,
            "applied_exploration_updates": applied_exploration,
            "raw_replay_updates": raw_replay,
            "applied_replay_updates": applied_replay,
            "receipt_sha256": sha(receipt_path),
            "metrics_sha256": sha(metrics_path),
            "state_replay_sha256": sha(replay_path),
        }
    checks["compute_traversal_match"] = (
        set(traversal) == set(ARMS) and traversal[CONTROL] == traversal[TREATMENT]
    )
    for key, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {key}")
    payload = {
        "schema": "point-maze-algorithm-repair-pair-audit-v1",
        "status": "pass" if not errors else "fail",
        "decision": (
            "eligible_for_point_maze_algorithm_repair_five_seed_final"
            if not errors
            else "point_maze_algorithm_repair_stopped"
        ),
        "checks": checks,
        "cells": cell_results,
        "errors": sorted(set(errors)),
        "evidence": {
            "identity_sha256": sha(args.identity),
            "submission_sha256": sha(args.submission),
            "manifest_sha256": sha(args.manifest),
        },
    }
    atomic(args.output, payload)
    print(json.dumps({"status": payload["status"], "errors": len(payload["errors"])}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

