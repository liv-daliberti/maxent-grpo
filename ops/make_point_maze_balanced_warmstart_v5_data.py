#!/usr/bin/env python3
"""Materialize an exactly compute-matched, orientation-balanced PointMaze SFT slate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from datasets import Dataset, DatasetDict


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])
)
SOURCE_ROOT = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(ROOT / "ops"))

import make_point_maze_mode_data as base  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var/data/point_maze_balanced_warmstart_source_v5"
ASSIGNMENTS = (
    ("bar7", 0),
    ("bar7", 1),
    ("block9", 2),
    ("block9", 3),
    ("bar9", 0),
    ("bar9", 1),
    ("asymmetric_block9", 2),
    ("asymmetric_block9", 3),
)


def _task_fingerprint(spec: dict[str, Any]) -> str:
    return base._canonical_sha256(
        {
            "maze_map": spec["maze_map"],
            "reset_cell": spec["reset_cell"],
            "goal_cell": spec["goal_cell"],
            "route_gates": spec["route_gates"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-python",
        type=Path,
        default=ROOT / "var/maze_runtime/venv/bin/python",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh balanced warm-start source required: {output_root}")

    runtime = base._runtime_identity(args.worker_python)
    families = {str(item["family"]): item for item in base._base_families()}
    rows: list[dict[str, Any]] = []
    certification: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    verifier = base.MazeVerifierProcess(worker_python=args.worker_python)
    try:
        for row_index, (family_name, rotation) in enumerate(ASSIGNMENTS):
            rotated = base._rotated_family(families[family_name], rotation)
            spec = base._make_spec(
                rotated,
                split="train",
                rotation=rotation,
                environment_sha256=runtime["point_environment_sha256"],
                seed=84_000 + row_index,
            )
            route_records = []
            for program in rotated["programs"]:
                candidate = " ".join(program)
                validation = verifier.validate(candidate, spec)
                if validation is None:
                    raise RuntimeError(f"{spec['map_id']} certified route failed")
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
            if len({item["canonical_key"] for item in route_records}) != 2:
                raise RuntimeError(f"{spec['map_id']} lacks two executable modes")
            fingerprint = _task_fingerprint(spec)
            if fingerprint in fingerprints:
                raise RuntimeError("balanced warm-start contains duplicate executable tasks")
            fingerprints.add(fingerprint)
            certification.append(
                {
                    "map_id": spec["map_id"],
                    "split": "train",
                    "family": family_name,
                    "rotation": rotation,
                    "task_fingerprint": fingerprint,
                    "spec_sha256": spec["spec_sha256"],
                    "routes": route_records,
                }
            )
            rows.append(
                {
                    "problem": base._prompt(spec),
                    "answer": json.dumps(spec, sort_keys=True, separators=(",", ":")),
                    "modebench_task": base.POINT_MAZE_VERIFIER,
                    "answer_mode_family": family_name,
                    "answer_mode_split": "train",
                    "certified_simple_route_count": 2,
                    "instance_fingerprint": spec["spec_sha256"],
                }
            )
    finally:
        verifier.close()

    orientation_counts = {
        str(rotation): sum(assigned == rotation for _family, assigned in ASSIGNMENTS)
        for rotation in range(4)
    }
    if orientation_counts != {"0": 2, "1": 2, "2": 2, "3": 2}:
        raise RuntimeError("balanced warm-start orientation contract drift")
    if len(rows) != 8 or len(certification) != 8:
        raise RuntimeError("balanced warm-start requires exactly eight train maps")

    output_root.mkdir(parents=True)
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(
        str(output_root / "train")
    )
    identity = {
        "schema_version": "point-maze-balanced-warmstart-source-v5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "split_rows": {"train": 8},
        "runtime_identity": runtime,
        "worker_python": str(args.worker_python.resolve()),
        "orientation_counts": orientation_counts,
        "orientation_balanced": True,
        "same_family_multiplicity_as_warmstart_v3": True,
        "expected_replayed_example_count": 644,
        "evaluation_rows_loaded": False,
        "development_rows_loaded": False,
        "certification": certification,
        "rows_sha256": base._canonical_sha256(rows),
    }
    (output_root / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[point-balanced-warmstart-v5-data] "
        "maps=8 routes=16 orientations=2/2/2/2"
    )


if __name__ == "__main__":
    main()
