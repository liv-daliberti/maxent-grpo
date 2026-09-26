#!/usr/bin/env python3
"""Materialize the certified 384/64/128 PointMaze full-scale v7 split."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

import make_point_maze_mode_data as base


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])
)
SOURCE_ROOT = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
base.ROOT = ROOT
base.SRC = SOURCE_ROOT
DEFAULT_OUTPUT = ROOT / "var/data/point_maze_fullscale_v7"

FAMILIES = (
    ("medium_block9_v7", 9, "block"),
    ("medium_wide9_v7", 9, "wide"),
    ("medium_cross9_v7", 9, "cross"),
    ("medium_diamond9_v7", 9, "diamond"),
    ("hard_block11_v7", 11, "block"),
    ("hard_wide11_v7", 11, "wide"),
    ("hard_cross11_v7", 11, "cross"),
    ("hard_diamond11_v7", 11, "diamond"),
)
SPLIT_ROUNDS = {"train": 12, "dev": 2, "eval": 4}
SPLIT_ORDINAL_OFFSET = {"train": 0, "dev": 48, "eval": 56}
SEED_BASE = {"train": 87_000, "dev": 88_000, "eval": 89_000}


def _task_fingerprint(spec: dict[str, Any]) -> str:
    return base._canonical_sha256(
        {
            "maze_map": spec["maze_map"],
            "reset_cell": spec["reset_cell"],
            "goal_cell": spec["goal_cell"],
            "route_gates": spec["route_gates"],
        }
    )


def _map_with(size: int, obstacles: set[tuple[int, int]]) -> list[list[int]]:
    return [
        [
            int(
                row in {0, size - 1}
                or column in {0, size - 1}
                or (row, column) in obstacles
            )
            for column in range(size)
        ]
        for row in range(size)
    ]


def _base_obstacles(size: int, shape: str) -> set[tuple[int, int]]:
    center = size // 2
    if shape == "block":
        return {
            (row, column)
            for row in range(center - 1, center + 2)
            for column in range(center - 1, center + 2)
        }
    if shape == "wide":
        half_width = 2 if size == 9 else 3
        return {
            (row, column)
            for row in range(center - 1, center + 2)
            for column in range(center - half_width, center + half_width + 1)
        }
    if shape == "cross":
        radius = 2 if size == 9 else 3
        return {
            (row, column)
            for row, column in (
                *((center, value) for value in range(center - radius, center + radius + 1)),
                *((value, center) for value in range(center - radius, center + radius + 1)),
            )
        }
    if shape == "diamond":
        radius = 2 if size == 9 else 3
        return {
            (row, column)
            for row in range(center - radius, center + radius + 1)
            for column in range(center - radius, center + radius + 1)
            if abs(row - center) + abs(column - center) <= radius
        }
    raise ValueError(f"unknown PointMaze v7 shape: {shape}")


def _variant_family(
    *,
    label: str,
    size: int,
    shape: str,
    ordinal: int,
    nonce: int,
) -> dict[str, Any]:
    center = size // 2
    radius = 2 if size == 9 else 3
    base_obstacles = _base_obstacles(size, shape)
    optional = sorted(
        {
            (row, column)
            for row in range(center - radius, center + radius + 1)
            for column in range(center - radius, center + radius + 1)
        }
        - base_obstacles
    )
    digest = hashlib.sha256(
        f"point-maze-fullscale-v7:{label}:{ordinal}:{nonce}".encode("ascii")
    ).digest()
    bits = int.from_bytes(digest, "big")
    obstacles = set(base_obstacles)
    obstacles.update(
        cell for index, cell in enumerate(optional) if bits & (1 << index)
    )
    if size == 9:
        counts = (8, 24, 11)
        spans = ((1.2, 3.4), (-3.4, -1.2))
    else:
        counts = (12, 32, 15)
        spans = ((2.0, 5.0), (-5.0, -2.0))
    return {
        "family": f"{label}_m{ordinal:02d}",
        "maze_map": _map_with(size, obstacles),
        "reset": (center, 1),
        "goal": (center, size - 2),
        "bounds": [[-size / 2, size / 2], [-size / 2, size / 2]],
        "spans": spans,
        "counts": counts,
        "variant_nonce": nonce,
        "obstacle_count": len(obstacles),
    }


def _materialize_row(
    *,
    verifier: Any,
    runtime: dict[str, Any],
    split: str,
    row_index: int,
    label: str,
    size: int,
    shape: str,
    ordinal: int,
    rotation: int,
    observed_fingerprints: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for nonce in range(10_000):
        family = _variant_family(
            label=label,
            size=size,
            shape=shape,
            ordinal=ordinal,
            nonce=nonce,
        )
        rotated = base._rotated_family(family, rotation)
        spec = base._make_spec(
            rotated,
            split=split,
            rotation=rotation,
            environment_sha256=runtime["point_environment_sha256"],
            seed=SEED_BASE[split] + row_index,
        )
        fingerprint = _task_fingerprint(spec)
        if fingerprint not in observed_fingerprints:
            break
    else:
        raise RuntimeError("could not construct a unique PointMaze v7 task")

    routes = []
    for program in rotated["programs"]:
        candidate = " ".join(program)
        validation = verifier.validate(candidate, spec)
        if validation is None:
            raise RuntimeError(f"{spec['map_id']} known route program failed")
        routes.append(
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
    if len({route["canonical_key"] for route in routes}) != 2:
        raise RuntimeError(f"{spec['map_id']} does not certify two route keys")

    observed_fingerprints.add(fingerprint)
    row = {
        "problem": base._prompt(spec),
        "answer": json.dumps(spec, sort_keys=True, separators=(",", ":")),
        "modebench_task": base.POINT_MAZE_VERIFIER,
        "answer_mode_family": label,
        "answer_mode_split": split,
        "certified_simple_route_count": 2,
        "instance_fingerprint": spec["spec_sha256"],
    }
    certification = {
        "map_id": spec["map_id"],
        "split": split,
        "row_index": row_index,
        "family": label,
        "shape": shape,
        "size": size,
        "ordinal": ordinal,
        "rotation": rotation,
        "variant_nonce": family["variant_nonce"],
        "obstacle_count": family["obstacle_count"],
        "task_fingerprint": fingerprint,
        "spec_sha256": spec["spec_sha256"],
        "routes": routes,
    }
    return row, certification


def main() -> None:
    parser = base.argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-python", type=Path, default=base.DEFAULT_WORKER_PYTHON
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh PointMaze v7 data required: {output_root}")

    runtime = base._runtime_identity(args.worker_python)
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLIT_ROUNDS
    }
    certifications: list[dict[str, Any]] = []
    observed_fingerprints: set[str] = set()
    verifier = base.MazeVerifierProcess(worker_python=args.worker_python)
    try:
        for split, rounds in SPLIT_ROUNDS.items():
            for round_index in range(rounds):
                for family_index, (label, size, shape) in enumerate(FAMILIES):
                    for rotation in range(4):
                        row_index = len(rows_by_split[split])
                        ordinal = (
                            SPLIT_ORDINAL_OFFSET[split]
                            + 4 * round_index
                            + rotation
                        )
                        row, certification = _materialize_row(
                            verifier=verifier,
                            runtime=runtime,
                            split=split,
                            row_index=row_index,
                            label=label,
                            size=size,
                            shape=shape,
                            ordinal=ordinal,
                            rotation=rotation,
                            observed_fingerprints=observed_fingerprints,
                        )
                        rows_by_split[split].append(row)
                        certifications.append(certification)
    finally:
        verifier.close()

    expected_rows = {"train": 384, "dev": 64, "eval": 128}
    if {
        split: len(rows) for split, rows in rows_by_split.items()
    } != expected_rows:
        raise RuntimeError("PointMaze v7 split cardinality drift")
    split_fingerprints = {
        split: {row["instance_fingerprint"] for row in rows}
        for split, rows in rows_by_split.items()
    }
    if len(observed_fingerprints) != sum(expected_rows.values()) or any(
        split_fingerprints[left] & split_fingerprints[right]
        for left, right in (
            ("train", "dev"),
            ("train", "eval"),
            ("dev", "eval"),
        )
    ):
        raise RuntimeError("PointMaze v7 executable task overlap")

    orientation_counts = {
        split: {
            str(rotation): sum(
                certification["split"] == split
                and certification["rotation"] == rotation
                for certification in certifications
            )
            for rotation in range(4)
        }
        for split in SPLIT_ROUNDS
    }
    family_counts = {
        split: {
            label: sum(
                row["answer_mode_family"] == label
                for row in rows_by_split[split]
            )
            for label, _size, _shape in FAMILIES
        }
        for split in SPLIT_ROUNDS
    }
    if orientation_counts != {
        "train": {"0": 96, "1": 96, "2": 96, "3": 96},
        "dev": {"0": 16, "1": 16, "2": 16, "3": 16},
        "eval": {"0": 32, "1": 32, "2": 32, "3": 32},
    }:
        raise RuntimeError("PointMaze v7 orientation balance drift")
    if any(
        set(counts.values()) != {expected_rows[split] // len(FAMILIES)}
        for split, counts in family_counts.items()
    ):
        raise RuntimeError("PointMaze v7 family balance drift")

    DatasetDict(
        {"train": Dataset.from_list(rows_by_split["train"])}
    ).save_to_disk(str(output_root / "train"))
    DatasetDict(
        {"multi_answer": Dataset.from_list(rows_by_split["dev"])}
    ).save_to_disk(str(output_root / "dev"))
    DatasetDict(
        {"multi_answer": Dataset.from_list(rows_by_split["eval"])}
    ).save_to_disk(str(output_root / "eval"))
    split_hashes = {
        split: base._canonical_sha256(rows)
        for split, rows in rows_by_split.items()
    }
    identity = {
        "schema_version": "point-maze-fullscale-v7-data-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "decision": "admitted_for_point_maze_fullscale_v7_online_comparison",
        "runtime_identity": runtime,
        "worker_python": str(args.worker_python.resolve()),
        "worker_source_sha256": base._sha256_file(
            SOURCE_ROOT / "oat_drgrpo/maze_modebench_worker.py"
        ),
        "verifier_source_sha256": base._sha256_file(
            SOURCE_ROOT / "oat_drgrpo/maze_modebench.py"
        ),
        "split_rows": expected_rows,
        "split_rows_sha256": split_hashes,
        "split_overlap_count": 0,
        "executable_task_overlap_count": 0,
        "orientation_counts": orientation_counts,
        "family_counts": family_counts,
        "orientation_balanced_within_each_split": True,
        "family_balanced_within_each_split": True,
        "families": [label for label, _size, _shape in FAMILIES],
        "certified_route_count": 2 * len(certifications),
        "certification": certifications,
        "information_boundary": {
            "development_rows_available_for_gating_only": True,
            "evaluation_rows_loaded_by_online_training": False,
            "certified_routes_in_model_context": False,
            "v6_terminal_outcome_loaded_before_data_freeze": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[point-fullscale-v7-data] "
        f"train={expected_rows['train']} dev={expected_rows['dev']} "
        f"eval={expected_rows['eval']} certified_routes="
        f"{identity['certified_route_count']}"
    )


if __name__ == "__main__":
    main()
