#!/usr/bin/env python3
"""Replay certified train-only waypoint routes into public SFT examples."""

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


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.point_maze_waypoint import (  # noqa: E402
    parse_point_waypoint_spec,
)
from oat_drgrpo.point_maze_waypoint_policy import (  # noqa: E402
    POINT_WAYPOINT_ACTION_TO_LABEL,
    point_waypoint_allowed_labels,
    render_point_waypoint_prompt,
)
from oat_drgrpo.point_maze_waypoint_process import (  # noqa: E402
    PointMazeWaypointProcess,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"fresh waypoint SFT root required: {args.output_root}")
    identity_path = args.data_root / "identity.json"
    if not identity_path.is_file():
        raise FileNotFoundError(identity_path)

    from datasets import load_from_disk

    dataset = load_from_disk(str(args.data_root))
    if set(dataset) != {"train", "dev", "eval"}:
        raise ValueError("waypoint pilot requires exact train/dev/eval splits")
    rows = dataset["train"].to_list()
    if not rows:
        raise ValueError("waypoint SFT requires nonempty training rows")
    raw_specs = []
    parsed_specs = []
    for row in rows:
        raw = row["answer"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        raw_specs.append(raw)
        parsed_specs.append(parse_point_waypoint_spec(raw))
    map_ids = {spec.base_spec.map_id for spec in parsed_specs}
    if len(map_ids) != len(rows):
        raise ValueError("waypoint training map IDs are not unique")

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    certifications = {
        str(record["map_id"]): record
        for record in identity["certification"]
        if record.get("split") == "train"
    }
    if set(certifications) != map_ids:
        raise ValueError("train certification set differs from waypoint rows")

    episodes = []
    for row_index, (row, raw, spec) in enumerate(zip(rows, raw_specs, parsed_specs)):
        certification = certifications[spec.base_spec.map_id]
        if certification.get("spec_sha256") != spec.spec_sha256:
            raise ValueError("waypoint certification/spec hash mismatch")
        routes = certification.get("routes", [])
        if not 3 <= len(routes) <= 5:
            raise ValueError("waypoint SFT map must carry three to five routes")
        for route_index, route in enumerate(routes):
            episodes.append(
                {
                    "session_id": f"train-{row_index:03d}-route-{route_index}",
                    "row_index": row_index,
                    "row": row,
                    "raw_spec": raw,
                    "spec": spec,
                    "route": route,
                    "actions": str(route["program"]).split(),
                    "examples": [],
                }
            )

    with PointMazeWaypointProcess(worker_python=args.worker_python) as worker:
        by_map: dict[int, list[dict[str, Any]]] = {}
        for episode in episodes:
            by_map.setdefault(int(episode["row_index"]), []).append(episode)
        for done_maps, (_row_index, chunk) in enumerate(sorted(by_map.items())):
            if done_maps % 16 == 0:
                print(
                    f"[waypoint-sft-data] replaying map {done_maps}/{len(by_map)}",
                    flush=True,
                )
            reset = worker.reset_batch(
                [
                    {
                        "session_id": episode["session_id"],
                        "spec": episode["raw_spec"],
                    }
                    for episode in chunk
                ]
            )
            observations = {item["session_id"]: item for item in reset}
            active = list(chunk)
            while active:
                requests = []
                for episode in active:
                    if not episode["actions"]:
                        raise RuntimeError("certified waypoint program ended early")
                    state = observations[episode["session_id"]]
                    action = episode["actions"].pop(0)
                    allowed_actions = tuple(state["allowed_actions"])
                    if action not in allowed_actions:
                        raise RuntimeError(
                            "certified waypoint action is not locally legal"
                        )
                    allowed_labels = point_waypoint_allowed_labels(allowed_actions)
                    episode["examples"].append(
                        {
                            "prompt": render_point_waypoint_prompt(
                                str(episode["row"]["problem"]),
                                state,
                            ),
                            "label": POINT_WAYPOINT_ACTION_TO_LABEL[action],
                            "allowed_labels": list(allowed_labels),
                            "action": action,
                            "step_index": len(episode["examples"]),
                            "current_cell": state["current_cell"],
                            "previous_cell": state["previous_cell"],
                            "goal_cell": state["goal_cell"],
                            "achieved_goal": state["achieved_goal"],
                            "velocity_xy": state["velocity_xy"],
                            "remaining_actions": state["remaining_actions"],
                        }
                    )
                    requests.append(
                        {
                            "session_id": episode["session_id"],
                            "action": action,
                        }
                    )
                transitions = worker.step_batch(requests)
                if len(transitions) != len(active):
                    raise RuntimeError("waypoint SFT worker batch size changed")
                next_active = []
                for episode, transition in zip(active, transitions):
                    if transition["session_id"] != episode["session_id"]:
                        raise RuntimeError("waypoint SFT worker reordered sessions")
                    observations[episode["session_id"]] = transition
                    if transition["done"]:
                        if episode["actions"]:
                            raise RuntimeError(
                                "waypoint route terminated before program end"
                            )
                        if (
                            transition["canonical_key"]
                            != episode["route"]["canonical_key"]
                        ):
                            raise RuntimeError("waypoint SFT canonical route changed")
                        episode["terminal"] = {
                            "canonical_key": transition["canonical_key"],
                            "directed_gates": transition["directed_gates"],
                            "final_goal_distance": transition["final_goal_distance"],
                            "simulator_steps": transition["simulator_steps"],
                        }
                    else:
                        next_active.append(episode)
                active = next_active

    examples = []
    episode_records = []
    for episode in episodes:
        for example in episode["examples"]:
            examples.append(
                {
                    "episode_id": episode["session_id"],
                    "map_id": episode["spec"].base_spec.map_id,
                    "family": str(episode["row"].get("answer_mode_family", "")),
                    "route_key": episode["route"]["canonical_key"],
                    **example,
                }
            )
        episode_records.append(
            {
                "episode_id": episode["session_id"],
                "map_id": episode["spec"].base_spec.map_id,
                "route_key": episode["route"]["canonical_key"],
                "program_sha256": episode["route"]["program_sha256"],
                "decision_count": len(episode["examples"]),
                "terminal": episode["terminal"],
            }
        )
    if not examples:
        raise RuntimeError("waypoint SFT materialization produced no examples")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.",
            dir=args.output_root.parent,
        )
    )
    try:
        examples_path = temporary / "examples.jsonl"
        with examples_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(
                    json.dumps(example, allow_nan=False, sort_keys=True) + "\n"
                )
        output_identity = {
            "schema": "point-maze-waypoint-sft-data-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "split": "train_only",
            "policy_interface": "point_waypoint_v1",
            "map_count": len(map_ids),
            "episode_count": len(episode_records),
            "example_count": len(examples),
            "train_map_ids": sorted(map_ids),
            "episodes": episode_records,
            "hashes": {
                "examples_sha256": _sha256(examples_path),
                "dataset_identity_sha256": _sha256(identity_path),
                "materializer_sha256": _sha256(Path(__file__).resolve()),
            },
            "information_boundary": {
                "train_dataset_rows_loaded": len(rows),
                "dev_rows_materialized": False,
                "eval_rows_materialized": False,
                "certification_records_selected_by_split": "train",
                "model_sampled": False,
                "online_reward_used": False,
                "per_state_legal_support_recorded": True,
            },
        }
        (temporary / "identity.json").write_text(
            json.dumps(output_identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output_root)
    except BaseException:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise
    print(
        f"[waypoint-sft-data] maps={len(map_ids)} episodes={len(episodes)} "
        f"examples={len(examples)} output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
