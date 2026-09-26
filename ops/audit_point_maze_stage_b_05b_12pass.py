#!/usr/bin/env python3
"""Fail-closed terminal audit for the ten PointMaze Stage-B paper cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.interactive_episode_replay import (  # noqa: E402
    InteractiveEpisodeRecord,
    interactive_transition_sha256,
)
from oat_drgrpo.maze_modebench import parse_maze_action_spec  # noqa: E402
from oat_drgrpo.point_maze_interactive_policy import (  # noqa: E402
    POINT_POLICY_LABELS,
    render_point_policy_prompt_v3,
)
from oat_drgrpo.point_maze_interactive_process import (  # noqa: E402
    PointMazeInteractiveProcess,
)


CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEEDS = (43, 44, 45, 46, 47)
UPDATES = 96
EVAL_ROUNDS = tuple(range(0, UPDATES + 1, 2))
FAMILIES = ("bar7", "block9", "bar9", "asymmetric_block9")
GEOMETRY_SHIFT_FAMILIES = (
    "wide_block9_shift",
    "cross9_shift",
    "upper_offset9_shift",
    "lower_offset9_shift",
)
BALANCED_V5_EVALUATION_FAMILIES = (
    "medium_wide_block9_v3",
    "medium_lower_offset9_v3",
    "hard_wide_block11_v3",
    "hard_diamond11_v3",
)


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
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def atomic(path: Path, payload: Mapping[str, Any]) -> None:
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scheduler_complete(job_id: int) -> bool:
    output = subprocess.run(
        ["sacct", "-X", "-j", str(job_id), "--format=State,ExitCode", "-n", "-P"],
        check=True, text=True, capture_output=True,
    ).stdout
    return "COMPLETED|0:0" in output


def replay_cell(
    *,
    arm: str,
    seed: int,
    replay_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    worker_python: Path,
) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    counts = {"updates": 0, "episodes": 0, "decisions": 0, "transition_matches": 0}
    if len(replay_rows) != UPDATES:
        return [f"{arm}/s{seed}: expected {UPDATES} state-replay rows"], counts
    action_ids = tuple(
        int(tokenizer.encode(label, add_special_tokens=False)[0])
        for label in POINT_POLICY_LABELS
    )
    with PointMazeInteractiveProcess(worker_python=worker_python) as worker:
        for expected_update, payload in enumerate(replay_rows, start=1):
            label = f"{arm}/s{seed}/update{expected_update}"
            row_index = (expected_update - 1) % len(train_rows)
            row = train_rows[row_index]
            if (
                payload.get("schema") != "point-maze-stage-b-state-replay-v1"
                or payload.get("arm") != arm
                or payload.get("seed") != seed
                or payload.get("update") != expected_update
                or payload.get("source_row_index") != row_index
                or payload.get("family") != row.get("answer_mode_family")
                or payload.get("instance_fingerprint") != row.get("instance_fingerprint")
            ):
                errors.append(f"{label}: replay identity mismatch")
                continue
            raw_spec = row["answer"]
            if isinstance(raw_spec, str):
                raw_spec = json.loads(raw_spec)
            spec = parse_maze_action_spec(raw_spec)
            raw_episodes = payload.get("episodes")
            if not isinstance(raw_episodes, list) or len(raw_episodes) != 16:
                errors.append(f"{label}: replay episode count mismatch")
                continue
            try:
                episodes = [InteractiveEpisodeRecord.from_state_dict(value) for value in raw_episodes]
            except Exception as error:
                errors.append(f"{label}: invalid episode: {type(error).__name__}: {error}")
                continue
            group_prompt = tuple(tokenizer.encode(str(row["problem"]), add_special_tokens=False))
            if any(episode.group_prompt_token_ids != group_prompt for episode in episodes):
                errors.append(f"{label}: static group prompt changed")
            sessions = [
                {"session_id": f"audit-{arm}-s{seed}-u{expected_update}-e{index}", "spec": raw_spec}
                for index in range(16)
            ]
            reset = worker.reset_batch(sessions)
            observations = {item["session_id"]: item for item in reset}
            finals: list[dict[str, Any] | None] = [None] * 16
            maximum = max(len(episode.decisions) for episode in episodes)
            for decision_index in range(maximum):
                requests = []
                metadata = []
                for episode_index, episode in enumerate(episodes):
                    if decision_index >= len(episode.decisions):
                        continue
                    decision = episode.decisions[decision_index]
                    session_id = sessions[episode_index]["session_id"]
                    before = observations[session_id]
                    rendered = render_point_policy_prompt_v3(
                        str(row["problem"]), before, tuple(spec.action_tokens), (),
                    )
                    if decision.prompt_token_ids != tuple(tokenizer.encode(rendered, add_special_tokens=False)):
                        errors.append(f"{label}/e{episode_index}/d{decision_index}: prompt mismatch")
                    if tuple(decision.allowed_token_ids) != action_ids:
                        errors.append(f"{label}/e{episode_index}/d{decision_index}: action support mismatch")
                        continue
                    try:
                        position = decision.allowed_token_ids.index(decision.selected_token_id)
                    except ValueError:
                        errors.append(f"{label}/e{episode_index}/d{decision_index}: selected action escaped support")
                        continue
                    action = spec.action_tokens[position]
                    requests.append({"session_id": session_id, "action": action})
                    metadata.append((episode_index, action, decision, before))
                transitions = worker.step_batch(requests) if requests else []
                if len(transitions) != len(metadata):
                    errors.append(f"{label}/d{decision_index}: worker batch mismatch")
                    break
                for (episode_index, action, decision, before), transition in zip(metadata, transitions):
                    session_id = sessions[episode_index]["session_id"]
                    if transition["session_id"] != session_id:
                        errors.append(f"{label}/e{episode_index}: worker reordered session")
                        continue
                    transition_hash = interactive_transition_sha256(before=before, action=action, after=transition)
                    counts["decisions"] += 1
                    if transition_hash != decision.transition_sha256:
                        errors.append(f"{label}/e{episode_index}/d{decision_index}: transition hash mismatch")
                    else:
                        counts["transition_matches"] += 1
                    observations[session_id] = transition
                    if transition["done"]:
                        finals[episode_index] = transition
            for episode_index, (episode, transition) in enumerate(zip(episodes, finals)):
                if transition is None:
                    errors.append(f"{label}/e{episode_index}: no terminal replay")
                    continue
                key = transition.get("canonical_key")
                key = None if key is None else str(key)
                if key != episode.outcome_key or float(key is not None) != episode.task_reward:
                    errors.append(f"{label}/e{episode_index}: canonical terminal mismatch")
            counts["updates"] += 1
            counts["episodes"] += len(episodes)
    return errors, counts


def main() -> None:
    parsed = parse_args()
    if parsed.output.exists():
        raise FileExistsError("fresh PointMaze Stage-B audit output required")
    root = parsed.repo_root.resolve()
    identity = json.loads(parsed.identity.read_text())
    balanced_v5 = (
        identity.get("schema")
        == "point-maze-balanced-v5-stage-b-05b-12pass-identity-v1"
    )
    geometry_shift = (
        identity.get("schema")
        == "point-maze-geometry-shift-stage-b-05b-12pass-identity-v2"
    )
    identity_schema = (
        "point-maze-balanced-v5-stage-b-05b-12pass-identity-v1"
        if balanced_v5
        else (
            "point-maze-geometry-shift-stage-b-05b-12pass-identity-v2"
            if geometry_shift
            else "point-maze-stage-b-05b-12pass-identity-v1"
        )
    )
    submission_schema = (
        "point-maze-balanced-v5-stage-b-05b-12pass-submission-v1"
        if balanced_v5
        else (
            "point-maze-geometry-shift-stage-b-05b-12pass-submission-v2"
            if geometry_shift
            else "point-maze-stage-b-05b-12pass-submission-v1"
        )
    )
    data_stem = (
        "point_maze_algorithm_repair_v3"
        if balanced_v5
        else (
            "point_maze_geometry_shift_v1"
            if geometry_shift
            else "point_maze_modebench_v1"
        )
    )
    qualification_stem = (
        "point_maze_balanced_warmstart_v5_qualification.json"
        if balanced_v5
        else (
            "point_maze_geometry_shift_paired_smoke_v1_audit.json"
            if geometry_shift
            else "point_maze_interactive_paired_smoke_v3_audit.json"
        )
    )
    qualification_schema = (
        "point-maze-balanced-warmstart-v5-qualification-v1"
        if balanced_v5
        else (
            "point-maze-interactive-paired-smoke-audit-v4"
            if geometry_shift
            else "point-maze-interactive-paired-smoke-audit-v3"
        )
    )
    qualification_decision = (
        "eligible_for_point_maze_v5_five_seed_pair"
        if balanced_v5
        else (
            "eligible_for_ten_point_maze_geometry_shift_replacement_jobs"
            if geometry_shift
            else "eligible_for_ten_point_maze_stage_b_jobs"
        )
    )
    families = (
        BALANCED_V5_EVALUATION_FAMILIES
        if balanced_v5
        else (GEOMETRY_SHIFT_FAMILIES if geometry_shift else FAMILIES)
    )
    submission = json.loads(parsed.submission.read_text())
    manifest_rows = list(csv.DictReader(parsed.manifest.open(), delimiter="\t"))
    errors: list[str] = []
    checks: dict[str, bool] = {}
    expected_cells = {(arm, str(seed)) for arm in ARMS for seed in SEEDS}
    observed_cells = {(row["arm"], row["seed"]) for row in manifest_rows}
    checks["manifest_exact_ten_cells"] = len(manifest_rows) == 10 and observed_cells == expected_cells
    jobs = {f"{row['arm']}/s{row['seed']}": int(row["job_id"]) for row in manifest_rows}
    checks["identity_contract"] = (
        identity.get("schema") == identity_schema
        and identity.get("arms") == list(ARMS) and identity.get("seeds") == list(SEEDS)
        and identity.get("optimizer_updates") == UPDATES and identity.get("prompt_passes") == 12
        and identity.get("evaluation_rounds") == list(EVAL_ROUNDS)
        and identity.get("jobs") == jobs and identity.get("final_seed_cohort") is True
    )
    checks["submission_contract"] = (
        submission.get("schema") == submission_schema
        and submission.get("identity_sha256") == sha(parsed.identity)
        and submission.get("manifest_sha256") == sha(parsed.manifest)
        and submission.get("jobs") == jobs and submission.get("released") is True
    )
    checks["source_snapshot"] = tree_hash(Path(identity["source_root"])) == identity.get("source_hash")
    checks["ops_snapshot"] = tree_hash(Path(identity["execution_root"])) == identity.get("execution_hash")
    model_stem = (
        "point_maze_interactive_warmstart_v5_balanced"
        if balanced_v5
        else "point_maze_interactive_warmstart_v3"
    )
    checks["model_snapshot"] = tree_hash(root / f"var/models/{model_stem}") == identity.get("model_tree_sha256")
    checks["data_snapshot"] = tree_hash(root / f"var/data/{data_stem}") == identity.get("data_tree_sha256")
    qualification = root / f"var/artifacts/{qualification_stem}"
    qualification_payload = json.loads(qualification.read_text())
    checks["qualification_pass"] = (
        qualification_payload.get("schema") == qualification_schema
        and qualification_payload.get("status") == "pass"
        and qualification_payload.get("decision") == qualification_decision
        and sha(qualification) == identity.get("qualification_audit_sha256")
    )

    from datasets import load_from_disk
    from transformers import AutoTokenizer
    train_rows = load_from_disk(str(root / f"var/data/{data_stem}/train"))["train"].to_list()
    model = root / f"var/models/{model_stem}"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    worker_python = root / "var/maze_runtime/venv/bin/python"

    cell_results: dict[str, Any] = {}
    fixed_traversal: dict[int, dict[str, list[tuple[int, int]]]] = {seed: {} for seed in SEEDS}
    for row in manifest_rows:
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
        replay_rows = read_jsonl(replay_path)
        training = [item for item in metrics if item.get("schema") == "point-maze-stage-b-training-metric-v1"]
        evaluations = [item for item in metrics if item.get("schema") == "point-maze-stage-b-evaluation-v1"]
        if [item.get("learning_round") for item in training] != list(range(1, UPDATES + 1)):
            errors.append(f"{label}: exact training-round sequence absent")
        if [item.get("learning_round") for item in evaluations] != list(EVAL_ROUNDS):
            errors.append(f"{label}: exact evaluation-round sequence absent")
        fixed_traversal[seed][arm] = []
        raw_exploration = applied_exploration = raw_replay = applied_replay = 0
        for expected_update, item in enumerate(training, start=1):
            for key, expected in {
                "fixed_policy_slots": 1536, "replay_decision_forward_slots": 1536,
                "policy_microbatch_size": 16, "action_support_escapes": 0,
                "optimizer_step": expected_update,
            }.items():
                if item.get(key) != expected:
                    errors.append(f"{label}/update{expected_update}: {key} mismatch")
            for key, value in item.items():
                if isinstance(value, (int, float)) and not finite(value):
                    errors.append(f"{label}/update{expected_update}: nonfinite {key}")
            if float(item.get("behavior_live_logprob_abs_diff_max", 1.0)) > 1e-4:
                errors.append(f"{label}/update{expected_update}: behavior/live drift")
            raw_e = float(item.get("raw_exploration_advantage_rms", 0.0))
            applied_e = float(item.get("applied_exploration_advantage_rms", 0.0))
            raw_r = float(item.get("replay_raw_score_gradient_l2", 0.0))
            applied_r = float(item.get("replay_applied_score_gradient_l2", 0.0))
            raw_exploration += raw_e > 0
            applied_exploration += applied_e > 0
            raw_replay += raw_r > 0
            applied_replay += applied_r > 0
            compute_only = float(item.get("replay_compute_only", -1))
            if arm == CONTROL and (applied_e != 0 or applied_r != 0 or compute_only != 1):
                errors.append(f"{label}/update{expected_update}: control derivative mismatch")
            if arm == TREATMENT and compute_only != 0:
                errors.append(f"{label}/update{expected_update}: treatment is compute-only")
            if arm == TREATMENT and raw_e > 0 and applied_e <= 0:
                errors.append(f"{label}/update{expected_update}: exploration derivative not applied")
            if arm == TREATMENT and raw_r > 0 and applied_r <= 0:
                errors.append(f"{label}/update{expected_update}: replay derivative not applied")
            fixed_traversal[seed][arm].append((
                int(item.get("fixed_policy_slots", -1)),
                int(item.get("replay_decision_forward_slots", -1)),
            ))
        for evaluation in evaluations:
            round_index = int(evaluation.get("learning_round", -1))
            if evaluation.get("evaluation_draw_count") != 4 or evaluation.get("evaluation_trajectory_count") != 132:
                errors.append(f"{label}/eval{round_index}: evaluation count mismatch")
            for metric_name in ("greedy", "mean8", "pass8", "distinct8"):
                if not finite(evaluation.get(metric_name)):
                    errors.append(f"{label}/eval{round_index}: missing {metric_name}")
            for family in families:
                if not finite(evaluation.get(f"eval/{family}/greedy")):
                    errors.append(f"{label}/eval{round_index}: missing {family} greedy")
                for draw in range(4):
                    for stem in ("mean8", "pass8", "distinct8"):
                        if not finite(evaluation.get(f"eval/{family}/{stem}_draw_{draw}")):
                            errors.append(f"{label}/eval{round_index}: missing {family}/{stem}/{draw}")
        if (
            receipt.get("schema") != "point-maze-stage-b-05b-12pass-receipt-v1"
            or receipt.get("status") != "complete" or receipt.get("arm") != arm
            or receipt.get("seed") != seed or receipt.get("job_id") != job_id
            or receipt.get("metrics_sha256") != sha(metrics_path)
            or receipt.get("state_replay_sha256") != sha(replay_path)
            or receipt.get("counts", {}).get("optimizer_updates") != UPDATES
            or receipt.get("counts", {}).get("evaluation_coordinates") != len(EVAL_ROUNDS)
            or receipt.get("information_boundary", {}).get("evaluation_feedback_to_training") is not False
        ):
            errors.append(f"{label}: terminal receipt mismatch")
        replay_errors, replay_counts = replay_cell(
            arm=arm, seed=seed, replay_rows=replay_rows, train_rows=train_rows,
            tokenizer=tokenizer, worker_python=worker_python,
        )
        errors.extend(replay_errors)
        if replay_counts["decisions"] != replay_counts["transition_matches"]:
            errors.append(f"{label}: not every transition replay matched")
        cell_results[f"{arm}/s{seed}"] = {
            "job_id": job_id, "receipt_sha256": sha(receipt_path),
            "metrics_sha256": sha(metrics_path), "state_replay_sha256": sha(replay_path),
            "raw_exploration_updates": raw_exploration,
            "applied_exploration_updates": applied_exploration,
            "raw_replay_updates": raw_replay, "applied_replay_updates": applied_replay,
            "state_replay": replay_counts,
        }
    checks["compute_traversal_match_by_seed"] = all(
        set(fixed_traversal[seed]) == set(ARMS)
        and fixed_traversal[seed][CONTROL] == fixed_traversal[seed][TREATMENT]
        for seed in SEEDS
    )
    for key, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {key}")
    payload = {
        "schema": (
            "point-maze-balanced-v5-stage-b-05b-12pass-audit-v1"
            if balanced_v5
            else (
                "point-maze-geometry-shift-stage-b-05b-12pass-audit-v2"
                if geometry_shift
                else "point-maze-stage-b-05b-12pass-audit-v1"
            )
        ),
        "status": "pass" if not errors else "fail",
        "decision": (
            "point_maze_balanced_v5_terminal_eligible"
            if balanced_v5 and not errors
            else (
                "point_maze_geometry_shift_terminal_eligible"
                if geometry_shift and not errors
                else (
                    "point_maze_terminal_eligible"
                    if not errors
                    else (
                        "point_maze_balanced_v5_stage_b_stopped"
                        if balanced_v5
                        else (
                            "point_maze_geometry_shift_stage_b_stopped"
                            if geometry_shift
                            else "point_maze_stage_b_stopped"
                        )
                    )
                )
            )
        ),
        "checks": checks, "cells": cell_results, "errors": sorted(set(errors)),
        "evidence": {
            "identity_sha256": sha(parsed.identity), "submission_sha256": sha(parsed.submission),
            "manifest_sha256": sha(parsed.manifest),
        },
    }
    atomic(parsed.output, payload)
    print(json.dumps({"status": payload["status"], "errors": len(payload["errors"])}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
