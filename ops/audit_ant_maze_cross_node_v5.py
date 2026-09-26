#!/usr/bin/env python3
"""Prospective cross-node reproducibility audit for AntMaze controller v5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
from typing import Any


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1]))
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.ant_maze_worker_v5 import (  # noqa: E402
    RECEIPT_SHA256,
    execute_ant_v5_raw,
)
from oat_drgrpo.maze_modebench import (  # noqa: E402
    ANT_ACTIONS,
    ANT_MAZE_ACTION_VERSION,
    ANT_MAZE_VERIFIER,
    parse_maze_action_spec,
    validate_maze_execution,
)
from oat_drgrpo.maze_runtime_identity import maze_runtime_identity  # noqa: E402


UPPER = tuple(["N"] * 5 + ["E"] * 6 + ["S"] * 5 + ["SE"] * 2 + ["NE"])
LOWER = tuple(["S"] * 6 + ["E"] * 7 + ["N"] * 10 + ["NE"])
PERIPHERAL_CELLS = ((1, 5), (2, 5), (4, 5), (5, 5))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _maps() -> list[list[list[int]]]:
    subsets = [
        subset
        for size in range(1, len(PERIPHERAL_CELLS) + 1)
        for subset in itertools.combinations(PERIPHERAL_CELLS, size)
    ][:12]
    maps = []
    for subset in subsets:
        maze = [
            [1] * 7,
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1] * 7,
        ]
        for row, column in subset:
            maze[row][column] = 1
        maps.append(maze)
    if len({_canonical_sha256(maze) for maze in maps}) != 12:
        raise RuntimeError("AntMaze audit maps are not unique")
    return maps


def _spec(index: int, maze_map: list[list[int]]) -> dict[str, Any]:
    runtime = maze_runtime_identity()
    split = ("train", "dev", "eval")[index // 4]
    spec: dict[str, Any] = {
        "verifier": ANT_MAZE_VERIFIER,
        "maze_action_version": ANT_MAZE_ACTION_VERSION,
        "environment_id": "AntMaze_UMaze-v5",
        "environment_sha256": runtime["ant_environment_sha256"],
        "controller_sha256": RECEIPT_SHA256,
        "map_id": f"ant_admission_{split}_{index:02d}",
        "maze_map": maze_map,
        "reset_cell": [3, 2],
        "goal_cell": [3, 4],
        "reset_seed": 76_300 + index,
        "min_actions": 8,
        "max_actions": 32,
        "action_repeat": 75,
        "action_tokens": list(ANT_ACTIONS),
        "success_threshold": 0.5,
        "max_segment_length": 1.0,
        "bounds_xy": [[-12.0, 12.0], [-12.0, 12.0]],
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
    return spec


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--maximum-final-distance", type=float, default=0.45)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh Ant cross-node receipt required: {args.output}")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0 < args.maximum_final_distance < 0.5:
        raise ValueError("robustness margin must be inside the success threshold")

    records = []
    for map_index, maze_map in enumerate(_maps()):
        spec = _spec(map_index, maze_map)
        for route_name, tokens in (("upper", UPPER), ("lower", LOWER)):
            candidate = " ".join(tokens)
            for repetition in range(args.repetitions):
                execution = None
                validation = None
                error = None
                try:
                    execution = execute_ant_v5_raw(candidate, spec)
                    validation = validate_maze_execution(
                        candidate,
                        spec,
                        execution,
                    )
                except Exception as caught:
                    error = f"{type(caught).__name__}: {caught}"
                distance = (
                    float(execution["final_goal_distance"])
                    if execution is not None
                    else None
                )
                margin_pass = bool(
                    validation is not None
                    and distance is not None
                    and distance <= args.maximum_final_distance
                )
                records.append(
                    {
                        "map_index": map_index,
                        "map_id": spec["map_id"],
                        "spec_sha256": spec["spec_sha256"],
                        "route_name": route_name,
                        "program_sha256": hashlib.sha256(
                            candidate.encode("ascii")
                        ).hexdigest(),
                        "repetition": repetition,
                        "validated": validation is not None,
                        "margin_pass": margin_pass,
                        "canonical_key": (
                            validation.canonical_key
                            if validation is not None
                            else None
                        ),
                        "directed_gates": (
                            list(validation.directed_gates)
                            if validation is not None
                            else []
                        ),
                        "error": error,
                        "execution": execution,
                    }
                )

    passed = all(record["margin_pass"] for record in records)
    payload = {
        "schema_version": "ant-maze-v5-cross-node-replica-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "job_id": int(args.job_id),
        "replica": args.replica,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "controller_receipt_sha256": RECEIPT_SHA256,
        "runtime_identity": maze_runtime_identity(),
        "map_count": 12,
        "route_count": 2,
        "repetitions": args.repetitions,
        "maximum_final_distance": args.maximum_final_distance,
        "summary": {
            "execution_count": len(records),
            "validated_count": sum(row["validated"] for row in records),
            "margin_pass_count": sum(row["margin_pass"] for row in records),
        },
        "records": records,
    }
    _atomic_json(args.output, payload)
    print(
        "[ant-cross-node] "
        f"replica={args.replica} host={payload['hostname']} "
        f"status={payload['status']} "
        f"validated={payload['summary']['validated_count']}/{len(records)} "
        f"margin={payload['summary']['margin_pass_count']}/{len(records)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
