#!/usr/bin/env python3
"""Materialize fresh AntMaze v5 route specifications and real fixtures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from datasets import Dataset, DatasetDict


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1]))
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.ant_maze_worker_v5 import RECEIPT_SHA256  # noqa: E402
from oat_drgrpo.maze_modebench import (  # noqa: E402
    ANT_ACTIONS,
    ANT_MAZE_ACTION_VERSION,
    ANT_MAZE_VERIFIER,
    parse_maze_action_spec,
)
from oat_drgrpo.maze_modebench_process import MazeVerifierProcess  # noqa: E402


UPPER = tuple(["N"] * 5 + ["E"] * 6 + ["S"] * 5 + ["SE"] * 2 + ["NE"])
LOWER = tuple(["S"] * 6 + ["E"] * 7 + ["N"] * 10 + ["NE"])
PERIPHERAL_CELLS = ((1, 5), (2, 5), (4, 5), (5, 5))
MAP_SIZE = 7
STATIC_WALLS = ((3, 3),)
RESET_CELL = (3, 2)
GOAL_CELL = (3, 4)
BOUNDS_XY = ((-12.0, 12.0), (-12.0, 12.0))
CONTROLLER_RECEIPT_SHA256 = RECEIPT_SHA256
ANT_WORKER_SOURCE = "ant_maze_worker_v5.py"
DEFAULT_OUTPUT = ROOT / "var/data/ant_maze_modebench_v1"
RESET_SEED_BASE = 76_300
MAP_ID_PREFIX = "ant_admission"
MIN_ACTIONS = 8
MAX_ACTIONS = 32
ACTION_REPEAT = 75
DATA_SCHEMA = "ant-maze-modebench-data-v1"
VERSION_LABEL = "ant-maze-data"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity(worker_python: Path) -> dict[str, Any]:
    source = (
        "import json,sys;"
        f"sys.path.insert(0,{str(SRC)!r});"
        "from oat_drgrpo.maze_runtime_identity import maze_runtime_identity;"
        "print(json.dumps(maze_runtime_identity(),sort_keys=True))"
    )
    completed = subprocess.run(
        [str(worker_python), "-c", source],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.splitlines()[-1])


def _maps() -> list[list[list[int]]]:
    subsets = [
        subset
        for size in range(1, len(PERIPHERAL_CELLS) + 1)
        for subset in itertools.combinations(PERIPHERAL_CELLS, size)
    ][:12]
    maps = []
    for subset in subsets:
        maze = [[1] * MAP_SIZE]
        maze.extend(
            [[1] + [0] * (MAP_SIZE - 2) + [1] for _ in range(MAP_SIZE - 2)]
        )
        maze.append([1] * MAP_SIZE)
        for row, column in (*STATIC_WALLS, *subset):
            maze[row][column] = 1
        maps.append(maze)
    if len({_canonical_sha256(maze) for maze in maps}) != 12:
        raise RuntimeError("AntMaze admission maps are not unique")
    return maps


def _prompt(spec: dict[str, Any]) -> str:
    rows = ["".join("#" if cell else "." for cell in row) for row in spec["maze_map"]]
    return "\n".join(
        [
            "Navigate the Ant from S to G in the frozen maze.",
            "The map rows are:",
            *rows,
            f"S={spec['reset_cell']}; G={spec['goal_cell']}.",
            "Use only these commands: " + ", ".join(spec["action_tokens"]) + ".",
            (
                f"Return {spec['min_actions']} to {spec['max_actions']} commands "
                "inside \\boxed{}, separated by spaces. Do not explain."
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--worker-python",
        type=Path,
        default=ROOT / "var/maze_runtime/venv/bin/python",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh AntMaze data root required: {output_root}")
    runtime = _runtime_identity(args.worker_python)
    maps = _maps()
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "eval": [],
    }
    certification = []
    verifier = MazeVerifierProcess(
        timeout_seconds=25.0,
        worker_python=args.worker_python,
    )
    try:
        for index, maze_map in enumerate(maps):
            split = ("train", "dev", "eval")[index // 4]
            spec: dict[str, Any] = {
                "verifier": ANT_MAZE_VERIFIER,
                "maze_action_version": ANT_MAZE_ACTION_VERSION,
                "environment_id": "AntMaze_UMaze-v5",
                "environment_sha256": runtime["ant_environment_sha256"],
                "controller_sha256": CONTROLLER_RECEIPT_SHA256,
                "map_id": f"{MAP_ID_PREFIX}_{split}_{index:02d}",
                "maze_map": maze_map,
                "reset_cell": list(RESET_CELL),
                "goal_cell": list(GOAL_CELL),
                "reset_seed": RESET_SEED_BASE + index,
                "min_actions": MIN_ACTIONS,
                "max_actions": MAX_ACTIONS,
                "action_repeat": ACTION_REPEAT,
                "action_tokens": list(ANT_ACTIONS),
                "success_threshold": 0.5,
                "max_segment_length": 1.0,
                "bounds_xy": [list(axis) for axis in BOUNDS_XY],
                "route_gates": [
                    {
                        "id": "upper",
                        "axis": "x",
                        "coordinate": 0.0,
                        "span": [1.0, 10.0],
                        "hysteresis": 0.1,
                    },
                    {
                        "id": "lower",
                        "axis": "x",
                        "coordinate": 0.0,
                        "span": [-10.0, -1.0],
                        "hysteresis": 0.1,
                    },
                ],
            }
            spec["spec_sha256"] = parse_maze_action_spec(spec).spec_sha256
            route_records = []
            for route_name, tokens in (("upper", UPPER), ("lower", LOWER)):
                candidate = " ".join(tokens)
                result = verifier.validate_with_execution(candidate, spec)
                if result is None:
                    raise RuntimeError(
                        f"{spec['map_id']} {route_name} fixture failed"
                    )
                validation, execution = result
                route_records.append(
                    {
                        "route_name": route_name,
                        "program": candidate,
                        "program_sha256": hashlib.sha256(
                            candidate.encode("ascii")
                        ).hexdigest(),
                        "canonical_key": validation.canonical_key,
                        "directed_gates": list(validation.directed_gates),
                        "simulator_steps": validation.simulator_steps,
                        "execution": execution,
                    }
                )
            if len({record["canonical_key"] for record in route_records}) != 2:
                raise RuntimeError(f"{spec['map_id']} route keys collided")
            certification.append(
                {
                    "map_id": spec["map_id"],
                    "split": split,
                    "spec_sha256": spec["spec_sha256"],
                    "routes": route_records,
                }
            )
            rows_by_split[split].append(
                {
                    "problem": _prompt(spec),
                    "answer": json.dumps(
                        spec, sort_keys=True, separators=(",", ":")
                    ),
                    "modebench_task": ANT_MAZE_VERIFIER,
                    "answer_mode_family": "upper_lower_obstacle",
                    "answer_mode_split": split,
                    "answer_mode_count": 2,
                    "certified_simple_route_count": 2,
                    "instance_fingerprint": spec["spec_sha256"],
                }
            )
    finally:
        verifier.close()

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        DatasetDict({"train": Dataset.from_list(rows_by_split["train"])}).save_to_disk(
            str(staging / "train")
        )
        DatasetDict({"multi_answer": Dataset.from_list(rows_by_split["dev"])}).save_to_disk(
            str(staging / "dev")
        )
        DatasetDict({"multi_answer": Dataset.from_list(rows_by_split["eval"])}).save_to_disk(
            str(staging / "eval")
        )
        split_hashes = {
            split: _canonical_sha256(rows)
            for split, rows in rows_by_split.items()
        }
        identity = {
            "schema_version": DATA_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "real_routes_certified_perturbation_audit_pending",
            "runtime_identity": runtime,
            "controller_receipt_sha256": CONTROLLER_RECEIPT_SHA256,
            "worker_python": str(args.worker_python.resolve()),
            "worker_source_sha256": _sha256_file(
                SRC / "oat_drgrpo/maze_modebench_worker.py"
            ),
            "ant_worker_source_sha256": _sha256_file(
                SRC / f"oat_drgrpo/{ANT_WORKER_SOURCE}"
            ),
            "verifier_source_sha256": _sha256_file(
                SRC / "oat_drgrpo/maze_modebench.py"
            ),
            "split_rows": {
                split: len(rows) for split, rows in rows_by_split.items()
            },
            "split_rows_sha256": split_hashes,
            "split_overlap_count": 0,
            "map_count": 12,
            "action_repeat": ACTION_REPEAT,
            "program_sha256": {
                "upper": hashlib.sha256(" ".join(UPPER).encode("ascii")).hexdigest(),
                "lower": hashlib.sha256(" ".join(LOWER).encode("ascii")).hexdigest(),
            },
            "certification": certification,
        }
        (staging / "identity.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"[{VERSION_LABEL}] train=4 dev=4 eval=4 real_routes=24 "
        f"output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
