#!/usr/bin/env python3
"""Replay certified train-only PointMaze routes into state-action examples."""

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

from oat_drgrpo.maze_modebench import parse_maze_action_spec  # noqa: E402
from oat_drgrpo.point_maze_interactive_policy import (  # noqa: E402
    POINT_POLICY_LABELS,
    render_point_policy_prompt,
    render_point_policy_prompt_v2,
    render_point_policy_prompt_v3,
)
from oat_drgrpo.point_maze_interactive_process import (  # noqa: E402
    PointMazeInteractiveProcess,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _spec(row: dict[str, Any]) -> dict[str, Any]:
    value = row["answer"]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("PointMaze answer must contain a JSON specification")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-split-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--policy-interface",
        choices=("history_v1", "compact_state_v2", "velocity_state_v3"),
        default="history_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"fresh warm-start root required: {args.output_root}")

    from datasets import load_from_disk

    dataset = load_from_disk(str(args.train_split_root))
    if set(dataset) != {"train"}:
        raise ValueError("warm-start builder requires the train split only")
    rows = dataset["train"].to_list()
    if len(rows) != 8:
        raise ValueError("PointMaze warm start requires exactly eight train maps")
    specs = [_spec(row) for row in rows]
    parsed_specs = [parse_maze_action_spec(spec) for spec in specs]
    prompt_renderer = {
        "history_v1": render_point_policy_prompt,
        "compact_state_v2": render_point_policy_prompt_v2,
        "velocity_state_v3": render_point_policy_prompt_v3,
    }[args.policy_interface]
    train_map_ids = {spec.map_id for spec in parsed_specs}
    if len(train_map_ids) != len(rows):
        raise ValueError("train map IDs are not unique")

    dataset_identity_path = args.data_root / "identity.json"
    dataset_identity = json.loads(dataset_identity_path.read_text(encoding="utf-8"))
    certifications = {
        str(record["map_id"]): record
        for record in dataset_identity["certification"]
        if record.get("split") == "train"
    }
    if set(certifications) != train_map_ids:
        raise ValueError("train-only certification set differs from train rows")
    if any(
        len(record.get("routes", [])) != 2
        for record in certifications.values()
    ):
        raise ValueError("every train map must have exactly two certified routes")

    episodes = []
    for row_index, (row, raw_spec, spec) in enumerate(
        zip(rows, specs, parsed_specs)
    ):
        certification = certifications[spec.map_id]
        if certification["spec_sha256"] != spec.spec_sha256:
            raise ValueError("train certification/spec hash mismatch")
        for route_index, route in enumerate(certification["routes"]):
            episodes.append(
                {
                    "session_id": f"train-{row_index:02d}-route-{route_index}",
                    "row_index": row_index,
                    "row": row,
                    "raw_spec": raw_spec,
                    "spec": spec,
                    "route": route,
                    "actions": str(route["program"]).split(),
                    "history": [],
                    "examples": [],
                }
            )

    worker = PointMazeInteractiveProcess(worker_python=args.worker_python)
    try:
        observations = worker.reset_batch(
            [
                {
                    "session_id": episode["session_id"],
                    "spec": episode["raw_spec"],
                }
                for episode in episodes
            ]
        )
        by_session = {
            observation["session_id"]: observation
            for observation in observations
        }
        active = list(episodes)
        step_index = 0
        while active:
            step_index += 1
            choices = []
            for episode in active:
                if not episode["actions"]:
                    raise RuntimeError("certified route ended before terminal success")
                action = episode["actions"].pop(0)
                action_tuple = tuple(episode["spec"].action_tokens)
                if action not in action_tuple:
                    raise RuntimeError("certified train action left the public alphabet")
                option_index = action_tuple.index(action)
                episode["examples"].append(
                    {
                        "prompt": prompt_renderer(
                            str(episode["row"]["problem"]),
                            by_session[episode["session_id"]],
                            action_tuple,
                            episode["history"],
                        ),
                        "label": POINT_POLICY_LABELS[option_index],
                        "action": action,
                        "step_index": len(episode["history"]),
                        "achieved_goal": by_session[episode["session_id"]][
                            "achieved_goal"
                        ],
                        "desired_goal": by_session[episode["session_id"]][
                            "desired_goal"
                        ],
                        "velocity_xy": by_session[episode["session_id"]].get(
                            "velocity_xy"
                        ),
                        "remaining_actions": by_session[episode["session_id"]][
                            "remaining_actions"
                        ],
                    }
                )
                choices.append((episode, action))

            transitions = worker.step_batch(
                [
                    {
                        "session_id": episode["session_id"],
                        "action": action,
                    }
                    for episode, action in choices
                ]
            )
            next_active = []
            for (episode, action), transition in zip(choices, transitions):
                if transition["session_id"] != episode["session_id"]:
                    raise RuntimeError("worker reordered train-only sessions")
                episode["history"].append(action)
                by_session[episode["session_id"]] = transition
                if transition["done"]:
                    if episode["actions"]:
                        raise RuntimeError(
                            "certified route reached terminal before consuming program"
                        )
                    if transition["canonical_key"] != episode["route"]["canonical_key"]:
                        raise RuntimeError("train-only replay canonical key changed")
                    episode["terminal"] = {
                        "canonical_key": transition["canonical_key"],
                        "directed_gates": transition["directed_gates"],
                        "final_goal_distance": transition["final_goal_distance"],
                        "simulator_steps": transition["simulator_steps"],
                    }
                else:
                    next_active.append(episode)
            active = next_active
            if step_index > 96:
                raise RuntimeError("train-only replay exceeded the public horizon")
    finally:
        worker.close()

    examples = []
    episode_records = []
    for episode in episodes:
        route = episode["route"]
        episode_id = episode["session_id"]
        for example in episode["examples"]:
            examples.append(
                {
                    "episode_id": episode_id,
                    "map_id": episode["spec"].map_id,
                    "family": str(episode["row"].get("answer_mode_family", "")),
                    "route_key": route["canonical_key"],
                    **example,
                }
            )
        episode_records.append(
            {
                "episode_id": episode_id,
                "map_id": episode["spec"].map_id,
                "family": str(episode["row"].get("answer_mode_family", "")),
                "route_key": route["canonical_key"],
                "program_sha256": route["program_sha256"],
                "decision_count": len(episode["examples"]),
                "terminal": episode["terminal"],
            }
        )
    if not examples:
        raise RuntimeError("warm-start materialization produced no examples")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.",
            dir=args.output_root.parent,
        )
    )
    try:
        examples_path = temporary_root / "examples.jsonl"
        with examples_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(
                    json.dumps(example, allow_nan=False, sort_keys=True) + "\n"
                )
        identity = {
            "schema_version": (
                "point-maze-interactive-warmstart-data-v1"
                if args.policy_interface == "history_v1"
                else (
                    "point-maze-interactive-warmstart-data-v2"
                    if args.policy_interface == "compact_state_v2"
                    else "point-maze-interactive-warmstart-data-v3"
                )
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "split": "train_only",
            "train_map_ids": sorted(train_map_ids),
            "map_count": len(train_map_ids),
            "episode_count": len(episode_records),
            "example_count": len(examples),
            "policy_interface": args.policy_interface,
            "route_keys": sorted(
                record["route_key"] for record in episode_records
            ),
            "episodes": episode_records,
            "hashes": {
                "examples_sha256": _sha256(examples_path),
                "train_rows_sha256": _canonical_sha256(rows),
                "dataset_identity_sha256": _sha256(dataset_identity_path),
                "protocol_sha256": _sha256(args.protocol),
                "materializer_sha256": _sha256(Path(__file__).resolve()),
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
        identity_path = temporary_root / "identity.json"
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_root.replace(args.output_root)
    except BaseException:
        for path in temporary_root.iterdir():
            path.unlink()
        temporary_root.rmdir()
        raise
    print(
        "[point-warmstart-data] "
        f"maps={len(train_map_ids)} episodes={len(episodes)} "
        f"examples={len(examples)} output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
