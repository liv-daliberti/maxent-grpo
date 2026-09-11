#!/usr/bin/env python3
"""Independently audit the frozen PointMaze interactive paired smoke."""

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
FAMILIES = ("bar7", "block9", "bar9", "asymmetric_block9")
GEOMETRY_SHIFT_FAMILIES = (
    "wide_block9_shift",
    "cross9_shift",
    "upper_offset9_shift",
    "lower_offset9_shift",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-split-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _schedule_complete(job_id: int) -> bool:
    result = subprocess.run(
        ["sacct", "-X", "-j", str(job_id), "--format=State,ExitCode", "-n", "-P"],
        check=True,
        capture_output=True,
        text=True,
    )
    return "COMPLETED|0:0" in result.stdout


def _replay_arm(
    *,
    arm: str,
    replay_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    tokenizer: Any,
    worker_python: Path,
    seed: int,
    row_indices: tuple[int, ...],
    replay_schema: str,
    families: tuple[str, ...],
) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    counts = {"episodes": 0, "decisions": 0, "transition_matches": 0}
    if len(replay_rows) != 4:
        return [f"{arm}: expected four state-replay rows"], counts
    with PointMazeInteractiveProcess(worker_python=worker_python) as worker:
        for expected_update, payload in enumerate(replay_rows, start=1):
            label = f"{arm}/update{expected_update}"
            if (
                payload.get("schema") != replay_schema
                or payload.get("arm") != arm
                or payload.get("seed") != seed
                or payload.get("update") != expected_update
                or payload.get("source_row_index") != row_indices[expected_update - 1]
                or payload.get("family") != families[expected_update - 1]
            ):
                errors.append(f"{label}: replay identity mismatch")
                continue
            row = source_rows[expected_update - 1]
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
                errors.append(f"{label}: invalid episode record: {type(error).__name__}: {error}")
                continue
            expected_group_prompt = tuple(
                tokenizer.encode(str(row["problem"]), add_special_tokens=False)
            )
            if any(episode.group_prompt_token_ids != expected_group_prompt for episode in episodes):
                errors.append(f"{label}: static group prompt tokens changed")
            sessions = [
                {"session_id": f"audit-{arm}-u{expected_update}-e{index}", "spec": raw_spec}
                for index in range(16)
            ]
            reset = worker.reset_batch(sessions)
            observations = {item["session_id"]: item for item in reset}
            counts["episodes"] += len(episodes)
            maximum = max(len(episode.decisions) for episode in episodes)
            finals: list[dict[str, Any] | None] = [None] * 16
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
                        str(row["problem"]),
                        before,
                        tuple(spec.action_tokens),
                        (),
                    )
                    expected_prompt = tuple(
                        tokenizer.encode(rendered, add_special_tokens=False)
                    )
                    if decision.prompt_token_ids != expected_prompt:
                        errors.append(
                            f"{label}/episode{episode_index}/decision{decision_index}: prompt token mismatch"
                        )
                    if tuple(decision.allowed_token_ids) != tuple(
                        int(tokenizer.encode(option, add_special_tokens=False)[0])
                        for option in POINT_POLICY_LABELS
                    ):
                        errors.append(
                            f"{label}/episode{episode_index}/decision{decision_index}: action support mismatch"
                        )
                    support_position = decision.allowed_token_ids.index(
                        decision.selected_token_id
                    )
                    action = spec.action_tokens[support_position]
                    requests.append({"session_id": session_id, "action": action})
                    metadata.append((episode_index, decision, before, action))
                transitions = worker.step_batch(requests) if requests else []
                if len(transitions) != len(metadata):
                    errors.append(f"{label}: worker replay batch size changed")
                    break
                for (episode_index, decision, before, action), after in zip(metadata, transitions):
                    session_id = sessions[episode_index]["session_id"]
                    if after.get("session_id") != session_id:
                        errors.append(f"{label}: worker replay order changed")
                        continue
                    observed_hash = interactive_transition_sha256(
                        before=before,
                        action=action,
                        after=after,
                    )
                    counts["decisions"] += 1
                    if observed_hash != decision.transition_sha256:
                        errors.append(
                            f"{label}/episode{episode_index}/decision{decision_index}: transition hash mismatch"
                        )
                    else:
                        counts["transition_matches"] += 1
                    observations[session_id] = after
                    if after["done"]:
                        finals[episode_index] = after
            for episode_index, (episode, final) in enumerate(zip(episodes, finals)):
                if final is None:
                    errors.append(f"{label}/episode{episode_index}: replay did not terminate")
                    continue
                observed_key = final.get("canonical_key")
                if observed_key != episode.outcome_key:
                    errors.append(f"{label}/episode{episode_index}: canonical key mismatch")
                if float(observed_key is not None) != episode.task_reward:
                    errors.append(f"{label}/episode{episode_index}: task reward mismatch")
    return errors, counts


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("fresh PointMaze paired audit output required")
    root = args.repo_root.resolve()
    identity = json.loads(args.identity.read_text())
    identity_schema = identity.get("schema")
    if identity_schema == "point-maze-interactive-paired-smoke-identity-v1":
        version = 1
        seed = 75301
        row_indices = (0, 2, 4, 6)
    elif identity_schema == "point-maze-interactive-paired-smoke-identity-v2":
        version = 2
        seed = 75302
        row_indices = (1, 3, 5, 7)
    elif identity_schema == "point-maze-interactive-paired-smoke-identity-v3":
        version = 3
        seed = 75303
        row_indices = (1, 3, 5, 7)
        families = FAMILIES
    elif identity_schema == "point-maze-interactive-paired-smoke-identity-v4":
        version = 4
        seed = 75304
        row_indices = (0, 2, 4, 6)
        families = GEOMETRY_SHIFT_FAMILIES
    else:
        raise ValueError("unknown PointMaze paired-smoke identity schema")
    if version in {1, 2}:
        families = FAMILIES
    metric_schema = f"point-maze-interactive-paired-smoke-metric-v{version}"
    receipt_schema = f"point-maze-interactive-paired-smoke-receipt-v{version}"
    replay_schema = f"point-maze-interactive-state-replay-v{version}"
    manifest = list(csv.DictReader(args.manifest.open(), delimiter="\t"))
    errors: list[str] = []
    checks: dict[str, bool] = {}
    jobs = {row["arm"]: int(row["job_id"]) for row in manifest}
    checks["identity"] = (
        identity.get("schema") == f"point-maze-interactive-paired-smoke-identity-v{version}"
        and identity.get("seed") == seed
        and identity.get("row_indices") == list(row_indices)
        and identity.get("arms") == list(ARMS)
        and identity.get("jobs") == jobs
        and identity.get("optimizer_updates_per_arm") == 4
    )
    checks["manifest"] = (
        len(manifest) == 2
        and {(row["arm"], int(row["seed"])) for row in manifest}
        == {(CONTROL, seed), (TREATMENT, seed)}
    )

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    dataset = load_from_disk(str(args.data_split_root))
    source_rows = [dataset["train"][index] for index in row_indices]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    arm_results: dict[str, Any] = {}
    traversal: dict[str, tuple[list[int], list[int]]] = {}
    for row in manifest:
        arm = row["arm"]
        job_id = int(row["job_id"])
        label = f"{arm}/j{job_id}"
        receipt_path = root / row["receipt"]
        metrics_path = root / row["metrics"]
        replay_path = root / row["state_replay"]
        if not _schedule_complete(job_id):
            errors.append(f"{label}: scheduler not complete")
        if not all(path.is_file() for path in (receipt_path, metrics_path, replay_path)):
            errors.append(f"{label}: terminal artifacts missing")
            continue
        receipt = json.loads(receipt_path.read_text())
        metrics = _read_jsonl(metrics_path)
        replay_rows = _read_jsonl(replay_path)
        if (
            receipt.get("schema") != receipt_schema
            or receipt.get("status") != "complete"
            or receipt.get("arm") != arm
            or receipt.get("seed") != seed
            or receipt.get("job_id") != job_id
            or receipt.get("identity_sha256") != sha(args.identity)
            or receipt.get("metrics_sha256") != sha(metrics_path)
            or receipt.get("state_replay_sha256") != sha(replay_path)
        ):
            errors.append(f"{label}: receipt identity mismatch")
        expected_counts = {
            "prompts": 4,
            "rollouts": 64,
            "fixed_policy_slots": 6144,
            "replay_group_slots": 4,
            "replay_mode_slots": 64,
            "optimizer_steps": 4,
        }
        if any(receipt.get("counts", {}).get(key) != value for key, value in expected_counts.items()):
            errors.append(f"{label}: fixed count contract mismatch")
        if (
            len(metrics) != 4
            or any(item.get("schema") != metric_schema for item in metrics)
            or [item.get("update") for item in metrics] != [1, 2, 3, 4]
        ):
            errors.append(f"{label}: exact four-update metrics absent")
        for update, metric in enumerate(metrics, start=1):
            if any(
                isinstance(value, float) and not math.isfinite(value)
                for value in metric.values()
            ):
                errors.append(f"{label}/update{update}: nonfinite metric")
            if (
                metric.get("fixed_policy_slots") != 1536
                or metric.get("replay_mode_slots") != 16.0
                or metric.get("replay_decision_forward_slots") != 1536.0
                or metric.get("policy_microbatch_size") != (16 if version in {3, 4} else 4)
                or metric.get("action_support_escapes") != 0
                or metric.get("behavior_live_logprob_abs_diff_max", 1.0) > 1e-4
            ):
                errors.append(f"{label}/update{update}: policy traversal contract mismatch")
            if arm == CONTROL and (
                metric.get("applied_exploration_advantage_rms") != 0.0
                or metric.get("replay_applied_score_gradient_l2") != 0.0
                or metric.get("replay_compute_only") != 1.0
            ):
                errors.append(f"{label}/update{update}: control derivative is nonzero")
            if arm == TREATMENT and metric.get("replay_compute_only") != 0.0:
                errors.append(f"{label}/update{update}: treatment is compute-only")
        verified = sum(int(item.get("verified_episodes", 0)) for item in metrics)
        multimode = sum(int(item.get("distinct_verified_keys", 0) >= 2) for item in metrics)
        raw_exploration = sum(float(item.get("raw_exploration_advantage_rms", 0)) > 0 for item in metrics)
        applied_exploration = sum(float(item.get("applied_exploration_advantage_rms", 0)) > 0 for item in metrics)
        raw_replay = sum(float(item.get("replay_raw_score_gradient_l2", 0)) > 0 for item in metrics)
        applied_replay = sum(float(item.get("replay_applied_score_gradient_l2", 0)) > 0 for item in metrics)
        if verified == 0:
            errors.append(f"{label}: no verified rollout")
        if multimode == 0:
            errors.append(f"{label}: no verifier-distinct multimode prompt")
        if raw_replay > 0 and arm == CONTROL and applied_replay != 0:
            errors.append(f"{label}: control replay derivative escaped zero")
        if arm == CONTROL and raw_replay == 0:
            errors.append(f"{label}: control lacks raw replay telemetry")
        if arm == TREATMENT and applied_exploration == 0:
            errors.append(f"{label}: treatment lacks applied exploration advantage")
        if arm == TREATMENT and raw_replay > 0 and applied_replay == 0:
            errors.append(f"{label}: eligible treatment replay was not applied")
        replay_errors, replay_counts = _replay_arm(
            arm=arm,
            replay_rows=replay_rows,
            source_rows=source_rows,
            tokenizer=tokenizer,
            worker_python=args.worker_python,
            seed=seed,
            row_indices=row_indices,
            replay_schema=replay_schema,
            families=families,
        )
        errors.extend(replay_errors)
        traversal[arm] = (
            [int(item.get("fixed_policy_slots", -1)) for item in metrics],
            [int(item.get("replay_decision_forward_slots", -1)) for item in metrics],
        )
        arm_results[arm] = {
            "job_id": job_id,
            "receipt_sha256": sha(receipt_path),
            "metrics_sha256": sha(metrics_path),
            "state_replay_sha256": sha(replay_path),
            "verified_episodes": verified,
            "multimode_updates": multimode,
            "raw_exploration_updates": raw_exploration,
            "applied_exploration_updates": applied_exploration,
            "raw_replay_updates": raw_replay,
            "applied_replay_updates": applied_replay,
            "replay_counts": replay_counts,
            "initial_model_tree_sha256": receipt.get("initial_model_tree_sha256"),
        }
    checks["initial_model_match"] = (
        set(arm_results) == set(ARMS)
        and len({arm_results[arm]["initial_model_tree_sha256"] for arm in ARMS}) == 1
    )
    checks["fixed_traversal_match"] = (
        set(traversal) == set(ARMS) and traversal[CONTROL] == traversal[TREATMENT]
    )
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    payload = {
        "schema": f"point-maze-interactive-paired-smoke-audit-v{version}",
        "status": "pass" if not errors else "fail",
        "decision": (
            (
                "eligible_for_ten_point_maze_geometry_shift_replacement_jobs"
                if version == 4
                else "eligible_for_ten_point_maze_stage_b_jobs"
            )
            if not errors
            else (
                "point_maze_geometry_shift_replacement_stopped"
                if version == 4
                else "point_maze_stage_b_stopped"
            )
        ),
        "checks": checks,
        "arms": arm_results,
        "errors": sorted(set(errors)),
        "evidence": {
            "identity_sha256": sha(args.identity),
            "manifest_sha256": sha(args.manifest),
        },
    }
    atomic(args.output, payload)
    print(
        f"[point-paired-audit] status={payload['status']} errors={len(payload['errors'])}",
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
