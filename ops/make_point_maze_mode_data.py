#!/usr/bin/env python3
"""Materialize source-bound PointMaze language-action benchmark splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from datasets import Dataset, DatasetDict


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.maze_modebench import (  # noqa: E402
    MAZE_ACTION_VERSION,
    POINT_ACTIONS,
    POINT_MAZE_VERIFIER,
)
from oat_drgrpo.maze_modebench_process import MazeVerifierProcess  # noqa: E402


DEFAULT_WORKER_PYTHON = ROOT / "var/maze_runtime/venv/bin/python"
DEFAULT_OUTPUT = ROOT / "var/data/point_maze_modebench_v1"
ROTATE_TOKEN_CW = {
    "N": "E",
    "NE": "SE",
    "E": "S",
    "SE": "SW",
    "S": "W",
    "SW": "NW",
    "W": "N",
    "NW": "NE",
    "COAST": "COAST",
}


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
        "sys.path.insert(0,'src');"
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


def _rotate_map_cw(maze_map: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*maze_map[::-1])]


def _rotate_cell_cw(cell: tuple[int, int], size: int) -> tuple[int, int]:
    row, column = cell
    return column, size - 1 - row


def _rotate_point_cw(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return y, -x


def _rotate_gate_cw(gate: dict[str, Any]) -> dict[str, Any]:
    axis = gate["axis"]
    coordinate = float(gate["coordinate"])
    low, high = (float(value) for value in gate["span"])
    if axis == "x":
        endpoints = ((coordinate, low), (coordinate, high))
    else:
        endpoints = ((low, coordinate), (high, coordinate))
    first, second = (_rotate_point_cw(point) for point in endpoints)
    if abs(first[0] - second[0]) < 1e-12:
        new_axis = "x"
        new_coordinate = first[0]
        new_span = sorted((first[1], second[1]))
    else:
        new_axis = "y"
        new_coordinate = first[1]
        new_span = sorted((first[0], second[0]))
    return {
        "id": gate["id"],
        "axis": new_axis,
        "coordinate": new_coordinate,
        "span": new_span,
        "hysteresis": gate["hysteresis"],
    }


def _rotate_program_cw(program: list[str]) -> list[str]:
    return [ROTATE_TOKEN_CW[token] for token in program]


def _base_families() -> list[dict[str, Any]]:
    border9 = [1] * 9
    return [
        {
            "family": "bar7",
            "maze_map": [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            "reset": (3, 1),
            "goal": (3, 5),
            "bounds": [[-3.5, 3.5], [-3.5, 3.5]],
            "spans": ((0.4, 3.0), (-3.0, -0.4)),
            "counts": (7, 18, 7),
        },
        {
            "family": "block9",
            "maze_map": [
                border9,
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                border9,
            ],
            "reset": (4, 1),
            "goal": (4, 7),
            "bounds": [[-4.5, 4.5], [-4.5, 4.5]],
            "spans": ((1.2, 3.4), (-3.4, -1.2)),
            "counts": (8, 24, 11),
        },
        {
            "family": "bar9",
            "maze_map": [
                border9,
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                border9,
            ],
            "reset": (4, 1),
            "goal": (4, 7),
            "bounds": [[-4.5, 4.5], [-4.5, 4.5]],
            "spans": ((0.55, 3.4), (-3.4, -0.55)),
            "counts": (8, 24, 11),
        },
        {
            "family": "asymmetric_block9",
            "maze_map": [
                border9,
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                border9,
            ],
            "reset": (4, 1),
            "goal": (4, 7),
            "bounds": [[-4.5, 4.5], [-4.5, 4.5]],
            "spans": ((1.2, 3.4), (-3.4, -1.2)),
            "counts": (8, 24, 11),
        },
    ]


def _rotated_family(base: dict[str, Any], rotations: int) -> dict[str, Any]:
    maze_map = [list(row) for row in base["maze_map"]]
    reset = tuple(base["reset"])
    goal = tuple(base["goal"])
    first, east, last = base["counts"]
    programs = [
        ["N"] * first + ["E"] * east + ["S"] * last,
        ["S"] * first + ["E"] * east + ["N"] * last,
    ]
    gates = [
        {
            "id": "route_a",
            "axis": "x",
            "coordinate": 0.0,
            "span": list(base["spans"][0]),
            "hysteresis": 0.1,
        },
        {
            "id": "route_b",
            "axis": "x",
            "coordinate": 0.0,
            "span": list(base["spans"][1]),
            "hysteresis": 0.1,
        },
    ]
    for _ in range(rotations):
        size = len(maze_map)
        maze_map = _rotate_map_cw(maze_map)
        reset = _rotate_cell_cw(reset, size)
        goal = _rotate_cell_cw(goal, size)
        programs = [_rotate_program_cw(program) for program in programs]
        gates = [_rotate_gate_cw(gate) for gate in gates]
    return {
        **base,
        "maze_map": maze_map,
        "reset": reset,
        "goal": goal,
        "programs": programs,
        "gates": gates,
    }


def _prompt(spec: dict[str, Any]) -> str:
    rows = []
    for row_index, row in enumerate(spec["maze_map"]):
        chars = []
        for column_index, cell in enumerate(row):
            coordinate = [row_index, column_index]
            if coordinate == spec["reset_cell"]:
                chars.append("S")
            elif coordinate == spec["goal_cell"]:
                chars.append("G")
            else:
                chars.append("#" if cell else ".")
        rows.append("".join(chars))
    return "\n".join(
        [
            "Navigate the point mass from S to G in this fixed maze:",
            *rows,
            "",
            "Output only a whitespace-separated action program inside \\boxed{}.",
            "Allowed actions: N NE E SE S SW W NW COAST.",
            f"Use between {spec['min_actions']} and {spec['max_actions']} tokens.",
            f"Each token is held for {spec['action_repeat']} simulator steps.",
        ]
    )


def _make_spec(
    family: dict[str, Any],
    *,
    split: str,
    rotation: int,
    environment_sha256: str,
    seed: int,
) -> dict[str, Any]:
    spec = {
        "verifier": POINT_MAZE_VERIFIER,
        "maze_action_version": MAZE_ACTION_VERSION,
        "environment_id": "PointMaze_UMaze-v3",
        "environment_sha256": environment_sha256,
        "controller_sha256": None,
        "map_id": f"{family['family']}_{split}_r{rotation}",
        "maze_map": family["maze_map"],
        "reset_cell": list(family["reset"]),
        "goal_cell": list(family["goal"]),
        "reset_seed": seed,
        "min_actions": 2,
        "max_actions": 96,
        "action_repeat": 5,
        "action_tokens": list(POINT_ACTIONS),
        "success_threshold": 0.45,
        "max_segment_length": 0.2,
        "bounds_xy": family["bounds"],
        "route_gates": family["gates"],
    }
    spec["spec_sha256"] = _canonical_sha256(spec)
    return spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists; pass --overwrite")
        if output_root.name != "point_maze_modebench_v1":
            raise SystemExit("refusing overwrite outside point_maze_modebench_v1")
        shutil.rmtree(output_root)
    runtime = _runtime_identity(args.worker_python)
    rotations_by_split = {"train": (0, 2), "dev": (1,), "eval": (3,)}
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in rotations_by_split
    }
    certification = []
    verifier = MazeVerifierProcess(worker_python=args.worker_python)
    try:
        for family_index, base in enumerate(_base_families()):
            for split, rotations in rotations_by_split.items():
                for rotation in rotations:
                    rotated = _rotated_family(base, rotation)
                    spec = _make_spec(
                        rotated,
                        split=split,
                        rotation=rotation,
                        environment_sha256=runtime["point_environment_sha256"],
                        seed=43_001 + 100 * family_index + rotation,
                    )
                    route_records = []
                    for program in rotated["programs"]:
                        candidate = " ".join(program)
                        validation = verifier.validate(candidate, spec)
                        if validation is None:
                            raise RuntimeError(
                                f"{spec['map_id']} known route program failed"
                            )
                        route_records.append(
                            {
                                "program": candidate,
                                "program_sha256": hashlib.sha256(
                                    candidate.encode("ascii")
                                ).hexdigest(),
                                "canonical_key": validation.canonical_key,
                                "directed_gates": list(validation.directed_gates),
                                "simulator_steps": validation.simulator_steps,
                            }
                        )
                    keys = {record["canonical_key"] for record in route_records}
                    if len(keys) != 2:
                        raise RuntimeError(
                            f"{spec['map_id']} does not certify two route keys"
                        )
                    certification.append(
                        {
                            "map_id": spec["map_id"],
                            "split": split,
                            "family": base["family"],
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
                            "modebench_task": POINT_MAZE_VERIFIER,
                            "answer_mode_family": base["family"],
                            "answer_mode_split": split,
                            "certified_simple_route_count": 2,
                            "instance_fingerprint": spec["spec_sha256"],
                        }
                    )
    finally:
        verifier.close()

    DatasetDict({"train": Dataset.from_list(rows_by_split["train"])}).save_to_disk(
        str(output_root / "train")
    )
    DatasetDict({"multi_answer": Dataset.from_list(rows_by_split["dev"])}).save_to_disk(
        str(output_root / "dev")
    )
    DatasetDict({"multi_answer": Dataset.from_list(rows_by_split["eval"])}).save_to_disk(
        str(output_root / "eval")
    )
    split_hashes = {
        split: _canonical_sha256(rows) for split, rows in rows_by_split.items()
    }
    fingerprints = {
        split: {row["instance_fingerprint"] for row in rows}
        for split, rows in rows_by_split.items()
    }
    if any(
        fingerprints[left] & fingerprints[right]
        for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval"))
    ):
        raise RuntimeError("PointMaze split fingerprints overlap")
    identity = {
        "schema_version": "point-maze-modebench-data-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "two_route_programs_certified_perturbation_audit_pending",
        "runtime_identity": runtime,
        "worker_python": str(args.worker_python.resolve()),
        "worker_source_sha256": _sha256_file(
            ROOT / "src/oat_drgrpo/maze_modebench_worker.py"
        ),
        "verifier_source_sha256": _sha256_file(
            ROOT / "src/oat_drgrpo/maze_modebench.py"
        ),
        "split_rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "split_rows_sha256": split_hashes,
        "split_overlap_count": 0,
        "families": [family["family"] for family in _base_families()],
        "certification": certification,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    print(
        "[point-maze-data] "
        f"train={len(rows_by_split['train'])} dev={len(rows_by_split['dev'])} "
        f"eval={len(rows_by_split['eval'])} routes={len(certification) * 2}"
    )


if __name__ == "__main__":
    main()
