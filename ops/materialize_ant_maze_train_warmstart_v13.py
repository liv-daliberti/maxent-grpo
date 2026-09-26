#!/usr/bin/env python3
"""Replay only certified AntMaze train routes into one-token policy examples."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.ant_maze_interactive_policy import (  # noqa: E402
    ANT_POLICY_ACTIONS,
    render_ant_policy_prompt,
)
from oat_drgrpo.ant_maze_interactive_process import (  # noqa: E402
    AntMazeInteractiveProcess,
)
from oat_drgrpo.maze_modebench import parse_maze_action_spec  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def raw_spec(row: dict[str, Any]) -> dict[str, Any]:
    value = row["answer"]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("AntMaze answer must contain a JSON specification")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-split-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"fresh AntMaze warm-start root required: {args.output_root}")
    from datasets import load_from_disk

    dataset = load_from_disk(str(args.train_split_root))
    if set(dataset) != {"train"}:
        raise ValueError("AntMaze warm start requires the train split only")
    rows = dataset["train"].to_list()
    if len(rows) != 4:
        raise ValueError("AntMaze v13 requires exactly four train maps")
    specs = [raw_spec(row) for row in rows]
    parsed = [parse_maze_action_spec(spec) for spec in specs]
    if any(tuple(spec.action_tokens) != ANT_POLICY_ACTIONS for spec in parsed):
        raise ValueError("AntMaze v13 train alphabet drift")
    map_ids = {spec.map_id for spec in parsed}
    if len(map_ids) != len(rows):
        raise ValueError("AntMaze train map IDs are not unique")

    identity_path = args.data_root / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    certifications = {
        str(record["map_id"]): record
        for record in identity["certification"]
        if record.get("split") == "train"
    }
    if set(certifications) != map_ids:
        raise ValueError("AntMaze train certification set differs from rows")
    if any(len(record.get("routes", [])) != 2 for record in certifications.values()):
        raise ValueError("every AntMaze train map must have two certified routes")

    episodes = []
    for row_index, (row, raw, spec) in enumerate(zip(rows, specs, parsed)):
        certification = certifications[spec.map_id]
        if certification["spec_sha256"] != spec.spec_sha256:
            raise ValueError("AntMaze train certification/spec hash mismatch")
        for route_index, route in enumerate(certification["routes"]):
            episodes.append({
                "session_id": f"train-{row_index:02d}-route-{route_index}",
                "row_index": row_index,
                "row": row,
                "raw_spec": raw,
                "spec": spec,
                "route": route,
                "actions": str(route["program"]).split(),
                "history": [],
                "examples": [],
            })

    worker = AntMazeInteractiveProcess(worker_python=args.worker_python)
    try:
        observations = worker.reset_batch([
            {"session_id": episode["session_id"], "spec": episode["raw_spec"]}
            for episode in episodes
        ])
        by_session = {row["session_id"]: row for row in observations}
        active = list(episodes)
        decision_round = 0
        while active:
            decision_round += 1
            choices = []
            for episode in active:
                if not episode["actions"]:
                    raise RuntimeError("certified Ant route ended before success")
                action = episode["actions"].pop(0)
                if action not in ANT_POLICY_ACTIONS:
                    raise RuntimeError("certified Ant action left public support")
                observation = by_session[episode["session_id"]]
                episode["examples"].append({
                    "prompt": render_ant_policy_prompt(
                        str(episode["row"]["problem"]),
                        observation,
                        ANT_POLICY_ACTIONS,
                        episode["history"],
                    ),
                    "label": action,
                    "action": action,
                    "step_index": len(episode["history"]),
                    "achieved_goal": observation["achieved_goal"],
                    "desired_goal": observation["desired_goal"],
                    "velocity_xy": observation["velocity_xy"],
                    "remaining_actions": observation["remaining_actions"],
                })
                choices.append((episode, action))
            transitions = worker.step_batch([
                {"session_id": episode["session_id"], "action": action}
                for episode, action in choices
            ])
            next_active = []
            for (episode, action), transition in zip(choices, transitions):
                if transition["session_id"] != episode["session_id"]:
                    raise RuntimeError("AntMaze worker reordered train sessions")
                episode["history"].append(action)
                by_session[episode["session_id"]] = transition
                if transition["done"]:
                    if episode["actions"]:
                        raise RuntimeError("certified Ant route terminated before program end")
                    if transition["canonical_key"] != episode["route"]["canonical_key"]:
                        raise RuntimeError("Ant train replay canonical key changed")
                    episode["terminal"] = {
                        "canonical_key": transition["canonical_key"],
                        "directed_gates": transition["directed_gates"],
                        "final_goal_distance": transition["final_goal_distance"],
                        "simulator_steps": transition["simulator_steps"],
                    }
                else:
                    next_active.append(episode)
            active = next_active
            if decision_round > 16:
                raise RuntimeError("Ant train replay exceeded frozen horizon")
    finally:
        worker.close()

    examples = []
    episode_records = []
    for episode in episodes:
        for example in episode["examples"]:
            examples.append({
                "episode_id": episode["session_id"],
                "map_id": episode["spec"].map_id,
                "family": str(episode["row"].get("answer_mode_family", "")),
                "route_key": episode["route"]["canonical_key"],
                **example,
            })
        episode_records.append({
            "episode_id": episode["session_id"],
            "map_id": episode["spec"].map_id,
            "route_key": episode["route"]["canonical_key"],
            "program_sha256": episode["route"]["program_sha256"],
            "decision_count": len(episode["examples"]),
            "terminal": episode["terminal"],
        })
    if not examples:
        raise RuntimeError("Ant warm start produced no examples")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent))
    try:
        examples_path = temporary / "examples.jsonl"
        with examples_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, allow_nan=False, sort_keys=True) + "\n")
        payload = {
            "schema_version": "ant-maze-interactive-warmstart-data-v13",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "split": "train_only",
            "train_map_ids": sorted(map_ids),
            "map_count": len(map_ids),
            "episode_count": len(episode_records),
            "example_count": len(examples),
            "policy_interface": "closed_loop_public_markov_compass_token_v13",
            "action_support": list(ANT_POLICY_ACTIONS),
            "route_keys": sorted(record["route_key"] for record in episode_records),
            "episodes": episode_records,
            "hashes": {
                "examples_sha256": sha(examples_path),
                "train_rows_sha256": canonical_sha(rows),
                "dataset_identity_sha256": sha(identity_path),
                "protocol_sha256": sha(args.protocol),
                "materializer_sha256": sha(Path(__file__).resolve()),
            },
            "information_boundary": {
                "train_dataset_rows_loaded": len(rows),
                "dev_dataset_loaded": False,
                "eval_dataset_loaded": False,
                "certification_records_selected_by_split": "train",
                "model_sampled": False,
                "online_reward_used": False,
            },
        }
        (temporary / "identity.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output_root)
    except BaseException:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise
    print(
        f"[ant-v13-warmstart-data] maps={len(map_ids)} episodes={len(episodes)} "
        f"examples={len(examples)} output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
