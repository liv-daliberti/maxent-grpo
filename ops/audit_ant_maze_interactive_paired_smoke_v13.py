#!/usr/bin/env python3
"""Independently audit the frozen AntMaze v13 paired online smoke."""

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
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.ant_maze_interactive_policy import (  # noqa: E402
    ANT_POLICY_ACTIONS,
    render_ant_policy_prompt,
)
from oat_drgrpo.ant_maze_interactive_process import AntMazeInteractiveProcess  # noqa: E402
from oat_drgrpo.interactive_episode_replay import (  # noqa: E402
    InteractiveEpisodeRecord,
    interactive_transition_sha256,
)
from oat_drgrpo.maze_modebench import parse_maze_action_spec  # noqa: E402


CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
SEED = 76313


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def atomic(path: Path, payload: Mapping[str, Any]) -> None:
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
        check=True, capture_output=True, text=True,
    ).stdout
    return "COMPLETED|0:0" in output


def replay_arm(
    *, arm: str, replay_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]],
    tokenizer: Any, worker_python: Path,
) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    counts = {"episodes": 0, "decisions": 0, "transition_matches": 0}
    if len(replay_rows) != 4:
        return [f"{arm}: expected four state-replay rows"], counts
    action_ids = tuple(
        int(tokenizer.encode(action, add_special_tokens=False)[0])
        for action in ANT_POLICY_ACTIONS
    )
    with AntMazeInteractiveProcess(worker_python=worker_python) as worker:
        for expected_update, payload in enumerate(replay_rows, start=1):
            label = f"{arm}/update{expected_update}"
            row = source_rows[expected_update - 1]
            if (
                payload.get("schema") != "ant-maze-interactive-state-replay-v13"
                or payload.get("arm") != arm or payload.get("seed") != SEED
                or payload.get("update") != expected_update
                or payload.get("source_row_index") != expected_update - 1
                or payload.get("instance_fingerprint") != row.get("instance_fingerprint")
            ):
                errors.append(f"{label}: replay identity mismatch")
                continue
            raw_spec = row["answer"]
            if isinstance(raw_spec, str):
                raw_spec = json.loads(raw_spec)
            spec = parse_maze_action_spec(raw_spec)
            try:
                episodes = [
                    InteractiveEpisodeRecord.from_state_dict(value)
                    for value in payload.get("episodes", [])
                ]
            except Exception as error:
                errors.append(f"{label}: invalid episode: {type(error).__name__}: {error}")
                continue
            if len(episodes) != 16:
                errors.append(f"{label}: episode count mismatch")
                continue
            static_prompt = tuple(tokenizer.encode(str(row["problem"]), add_special_tokens=False))
            if any(episode.group_prompt_token_ids != static_prompt for episode in episodes):
                errors.append(f"{label}: group prompt mismatch")
            sessions = [
                {"session_id": f"audit-{arm}-u{expected_update}-e{index}", "spec": raw_spec}
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
                    rendered = render_ant_policy_prompt(
                        str(row["problem"]), before, ANT_POLICY_ACTIONS, (),
                    )
                    if decision.prompt_token_ids != tuple(tokenizer.encode(rendered, add_special_tokens=False)):
                        errors.append(f"{label}/e{episode_index}/d{decision_index}: prompt mismatch")
                    if tuple(decision.allowed_token_ids) != action_ids:
                        errors.append(f"{label}/e{episode_index}/d{decision_index}: support mismatch")
                        continue
                    try:
                        position = decision.allowed_token_ids.index(decision.selected_token_id)
                    except ValueError:
                        errors.append(f"{label}/e{episode_index}/d{decision_index}: selected action escaped")
                        continue
                    action = spec.action_tokens[position]
                    requests.append({"session_id": session_id, "action": action})
                    metadata.append((episode_index, decision, before, action))
                transitions = worker.step_batch(requests) if requests else []
                if len(transitions) != len(metadata):
                    errors.append(f"{label}/d{decision_index}: worker batch mismatch")
                    break
                for (episode_index, decision, before, action), after in zip(metadata, transitions):
                    session_id = sessions[episode_index]["session_id"]
                    if after["session_id"] != session_id:
                        errors.append(f"{label}/e{episode_index}: session reorder")
                        continue
                    observed_hash = interactive_transition_sha256(before=before, action=action, after=after)
                    counts["decisions"] += 1
                    if observed_hash != decision.transition_sha256:
                        errors.append(f"{label}/e{episode_index}/d{decision_index}: transition mismatch")
                    else:
                        counts["transition_matches"] += 1
                    observations[session_id] = after
                    if after["done"]:
                        finals[episode_index] = after
            for episode_index, (episode, after) in enumerate(zip(episodes, finals)):
                if after is None:
                    errors.append(f"{label}/e{episode_index}: terminal replay absent")
                    continue
                key = after.get("canonical_key")
                key = None if key is None else str(key)
                if key != episode.outcome_key or float(key is not None) != episode.task_reward:
                    errors.append(f"{label}/e{episode_index}: canonical key mismatch")
            counts["episodes"] += len(episodes)
    return errors, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("fresh AntMaze paired audit output required")
    root = args.repo_root.resolve()
    identity = json.loads(args.identity.read_text())
    cohort = identity.get("artifact_cohort")
    expected_microbatch = 16 if cohort == "v13r2" else 4
    rows = list(csv.DictReader(args.manifest.open(), delimiter="\t"))
    errors: list[str] = []
    checks = {
        "manifest_pair": len(rows) == 2 and {row["arm"] for row in rows} == set(ARMS),
        "identity_pair": (
            identity.get("schema") == "ant-maze-interactive-paired-smoke-identity-v13"
            and cohort in ("v13r1", "v13r2")
            and identity.get("seed") == SEED and identity.get("development_only") is True
            and identity.get("policy_microbatch_size") == expected_microbatch
            and (
                (cohort == "v13r1" and identity.get("cancelled_zero_runtime_predecessor_jobs") == [30202665, 30202666])
                or (cohort == "v13r2" and identity.get("failed_predecessor_jobs") == [30202916, 30202917, 30202918])
            )
            and identity.get("optimizer_updates_per_arm") == 4
            and identity.get("fixed_policy_slots_per_arm") == 1024
            and identity.get("fixed_replay_decision_slots_per_arm") == 1024
        ),
    }
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    source_rows = load_from_disk(str(args.train_root))["train"].to_list()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    traversal = {}
    results = {}
    for row in rows:
        arm = row["arm"]
        job_id = int(row["job_id"])
        label = f"{arm}/j{job_id}"
        if not scheduler_complete(job_id):
            errors.append(f"{label}: scheduler not complete")
        receipt_path = root / row["receipt"]
        metrics_path = root / row["metrics"]
        replay_path = root / row["state_replay"]
        if not all(path.is_file() for path in (receipt_path, metrics_path, replay_path)):
            errors.append(f"{label}: output absent")
            continue
        receipt = json.loads(receipt_path.read_text())
        metrics = read_jsonl(metrics_path)
        replay_rows = read_jsonl(replay_path)
        if [item.get("update") for item in metrics] != [1, 2, 3, 4]:
            errors.append(f"{label}: exact update sequence absent")
        for update, metric in enumerate(metrics, start=1):
            if any(isinstance(value, (int, float)) and not finite(value) for value in metric.values()):
                errors.append(f"{label}/update{update}: nonfinite metric")
            if (
                metric.get("fixed_policy_slots") != 256
                or metric.get("replay_mode_slots") != 16.0
                or metric.get("replay_decision_forward_slots") != 256.0
                or metric.get("policy_microbatch_size") != expected_microbatch
                or metric.get("action_support_escapes") != 0
                or float(metric.get("behavior_live_logprob_abs_diff_max", 1.0)) > 1e-4
            ):
                errors.append(f"{label}/update{update}: traversal mismatch")
            if arm == CONTROL and (
                metric.get("applied_exploration_advantage_rms") != 0.0
                or metric.get("replay_applied_score_gradient_l2") != 0.0
                or metric.get("replay_compute_only") != 1.0
            ):
                errors.append(f"{label}/update{update}: control derivative nonzero")
            if arm == TREATMENT and metric.get("replay_compute_only") != 0.0:
                errors.append(f"{label}/update{update}: treatment compute-only")
        verified = sum(int(item.get("verified_episodes", 0)) for item in metrics)
        multimode = sum(int(item.get("distinct_verified_keys", 0) >= 2) for item in metrics)
        raw_replay = sum(float(item.get("replay_raw_score_gradient_l2", 0)) > 0 for item in metrics)
        applied_replay = sum(float(item.get("replay_applied_score_gradient_l2", 0)) > 0 for item in metrics)
        applied_exploration = sum(float(item.get("applied_exploration_advantage_rms", 0)) > 0 for item in metrics)
        if verified == 0:
            errors.append(f"{label}: no verified rollout")
        if multimode == 0:
            errors.append(f"{label}: no multimode training map")
        if arm == CONTROL and raw_replay == 0:
            errors.append(f"{label}: no raw control replay telemetry")
        if arm == CONTROL and applied_replay != 0:
            errors.append(f"{label}: control replay derivative escaped zero")
        if arm == TREATMENT and applied_exploration == 0:
            errors.append(f"{label}: treatment exploration never applied")
        if arm == TREATMENT and raw_replay > 0 and applied_replay == 0:
            errors.append(f"{label}: eligible replay never applied")
        replay_errors, replay_counts = replay_arm(
            arm=arm, replay_rows=replay_rows, source_rows=source_rows,
            tokenizer=tokenizer, worker_python=args.worker_python,
        )
        errors.extend(replay_errors)
        if replay_counts["decisions"] != replay_counts["transition_matches"]:
            errors.append(f"{label}: transition replay incomplete")
        if (
            receipt.get("schema") != "ant-maze-interactive-paired-smoke-receipt-v13"
            or receipt.get("status") != "complete" or receipt.get("job_id") != job_id
            or receipt.get("arm") != arm or receipt.get("seed") != SEED
            or receipt.get("metrics_sha256") != sha(metrics_path)
            or receipt.get("state_replay_sha256") != sha(replay_path)
        ):
            errors.append(f"{label}: receipt mismatch")
        traversal[arm] = (
            [item.get("fixed_policy_slots") for item in metrics],
            [item.get("replay_decision_forward_slots") for item in metrics],
        )
        results[arm] = {
            "job_id": job_id, "receipt_sha256": sha(receipt_path),
            "metrics_sha256": sha(metrics_path), "state_replay_sha256": sha(replay_path),
            "verified_episodes": verified, "multimode_updates": multimode,
            "raw_replay_updates": raw_replay, "applied_replay_updates": applied_replay,
            "applied_exploration_updates": applied_exploration,
            "replay_counts": replay_counts,
            "initial_model_tree_sha256": receipt.get("initial_model_tree_sha256"),
        }
    checks["initial_model_match"] = (
        set(results) == set(ARMS)
        and len({results[arm]["initial_model_tree_sha256"] for arm in ARMS}) == 1
    )
    checks["fixed_traversal_match"] = set(traversal) == set(ARMS) and traversal[CONTROL] == traversal[TREATMENT]
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    payload = {
        "schema": "ant-maze-interactive-paired-smoke-audit-v13",
        "status": "pass" if not errors else "fail",
        "decision": "eligible_for_ten_ant_maze_stage_b_jobs" if not errors else "ant_maze_stage_b_stopped",
        "checks": checks, "arms": results, "errors": sorted(set(errors)),
        "evidence": {"identity_sha256": sha(args.identity), "manifest_sha256": sha(args.manifest)},
    }
    atomic(args.output, payload)
    print(f"[ant-paired-audit] status={payload['status']} errors={len(payload['errors'])}", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
