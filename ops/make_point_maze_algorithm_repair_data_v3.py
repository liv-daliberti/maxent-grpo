#!/usr/bin/env python3
"""Materialize the split-balanced PointMaze orientation-repair slate."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

import make_point_maze_mode_data as base
from make_point_maze_algorithm_repair_data_v1 import repair_families
from make_point_maze_geometry_shift_data import geometry_shift_families


ROOT = Path(
    os.environ.get(
        "OAT_ZERO_REPO_ROOT",
        Path(__file__).resolve().parents[1],
    )
)
SOURCE_ROOT = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
base.ROOT = ROOT
base.SRC = SOURCE_ROOT
DEFAULT_OUTPUT = ROOT / "var/data/point_maze_algorithm_repair_v3"
SPLIT_ASSIGNMENTS = {
    "train": (
        ("medium_wide_block9_v3", "wide_block9_shift", 0),
        ("medium_cross9_v3", "cross9_shift", 1),
        ("medium_upper_offset9_v3", "upper_offset9_shift", 2),
        ("medium_lower_offset9_v3", "lower_offset9_shift", 3),
        ("hard_block11_v3", "block11_repair", 0),
        ("hard_wide_block11_v3", "wide_block11_repair", 1),
        ("hard_bar11_v3", "bar11_repair", 2),
        ("hard_diamond11_v3", "diamond11_repair", 3),
    ),
    "dev": (
        ("medium_cross9_v3", "cross9_shift", 0),
        ("medium_upper_offset9_v3", "upper_offset9_shift", 1),
        ("hard_block11_v3", "block11_repair", 2),
        ("hard_bar11_v3", "bar11_repair", 3),
    ),
    "eval": (
        ("medium_wide_block9_v3", "wide_block9_shift", 2),
        ("medium_lower_offset9_v3", "lower_offset9_shift", 0),
        ("hard_wide_block11_v3", "wide_block11_repair", 3),
        ("hard_diamond11_v3", "diamond11_repair", 1),
    ),
}
SEED_BASE = {"train": 78_000, "dev": 79_000, "eval": 80_000}


def _source_families() -> dict[str, dict[str, Any]]:
    families = geometry_shift_families() + repair_families()
    return {str(family["family"]): family for family in families}


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
    parser = base.argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-python",
        type=Path,
        default=base.DEFAULT_WORKER_PYTHON,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"fresh PointMaze orientation-repair data required: {output_root}"
        )
    runtime = base._runtime_identity(args.worker_python)
    sources = _source_families()
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLIT_ASSIGNMENTS
    }
    certification: list[dict[str, Any]] = []
    task_fingerprints: dict[str, set[str]] = {
        split: set() for split in SPLIT_ASSIGNMENTS
    }
    verifier = base.MazeVerifierProcess(worker_python=args.worker_python)
    try:
        for split, assignments in SPLIT_ASSIGNMENTS.items():
            for row_index, (name, source_name, rotation) in enumerate(
                assignments
            ):
                family = dict(sources[source_name])
                family["family"] = name
                rotated = base._rotated_family(family, rotation)
                spec = base._make_spec(
                    rotated,
                    split=split,
                    rotation=rotation,
                    environment_sha256=runtime["point_environment_sha256"],
                    seed=SEED_BASE[split] + row_index,
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
                            "directed_gates": list(
                                validation.directed_gates
                            ),
                            "simulator_steps": validation.simulator_steps,
                        }
                    )
                if len(
                    {record["canonical_key"] for record in route_records}
                ) != 2:
                    raise RuntimeError(
                        f"{spec['map_id']} does not certify two route keys"
                    )
                task_fingerprint = _task_fingerprint(spec)
                if task_fingerprint in task_fingerprints[split]:
                    raise RuntimeError(
                        f"{split} contains a duplicate executable task"
                    )
                task_fingerprints[split].add(task_fingerprint)
                certification.append(
                    {
                        "map_id": spec["map_id"],
                        "split": split,
                        "family": name,
                        "source_family": source_name,
                        "rotation": rotation,
                        "task_fingerprint": task_fingerprint,
                        "spec_sha256": spec["spec_sha256"],
                        "routes": route_records,
                    }
                )
                rows_by_split[split].append(
                    {
                        "problem": base._prompt(spec),
                        "answer": json.dumps(
                            spec,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "modebench_task": base.POINT_MAZE_VERIFIER,
                        "answer_mode_family": name,
                        "answer_mode_split": split,
                        "certified_simple_route_count": 2,
                        "instance_fingerprint": spec["spec_sha256"],
                    }
                )
    finally:
        verifier.close()

    if any(
        task_fingerprints[left] & task_fingerprints[right]
        for left, right in (
            ("train", "dev"),
            ("train", "eval"),
            ("dev", "eval"),
        )
    ):
        raise RuntimeError(
            "PointMaze orientation-repair executable tasks overlap splits"
        )
    orientation_counts = {
        split: {
            str(rotation): sum(
                assigned_rotation == rotation
                for _name, _source, assigned_rotation in assignments
            )
            for rotation in range(4)
        }
        for split, assignments in SPLIT_ASSIGNMENTS.items()
    }
    if orientation_counts != {
        "train": {"0": 2, "1": 2, "2": 2, "3": 2},
        "dev": {"0": 1, "1": 1, "2": 1, "3": 1},
        "eval": {"0": 1, "1": 1, "2": 1, "3": 1},
    }:
        raise RuntimeError("PointMaze v3 orientation balance drift")

    output_root.mkdir(parents=True)
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
        "schema_version": "point-maze-algorithm-repair-data-v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "two_route_programs_certified_perturbation_audit_pending",
        "runtime_identity": runtime,
        "worker_python": str(args.worker_python.resolve()),
        "worker_source_sha256": base._sha256_file(
            SOURCE_ROOT / "oat_drgrpo/maze_modebench_worker.py"
        ),
        "verifier_source_sha256": base._sha256_file(
            SOURCE_ROOT / "oat_drgrpo/maze_modebench.py"
        ),
        "split_rows": {
            split: len(rows) for split, rows in rows_by_split.items()
        },
        "split_rows_sha256": split_hashes,
        "split_overlap_count": 0,
        "executable_task_overlap_count": 0,
        "orientation_counts": orientation_counts,
        "orientation_balanced_within_each_split": True,
        "families": sorted(
            {
                name
                for assignments in SPLIT_ASSIGNMENTS.values()
                for name, _source, _rotation in assignments
            }
        ),
        "split_assignments": {
            split: [
                {
                    "family": name,
                    "source_family": source,
                    "rotation": rotation,
                }
                for name, source, rotation in assignments
            ]
            for split, assignments in SPLIT_ASSIGNMENTS.items()
        },
        "certification": certification,
        "information_boundary": {
            "v2_partial_training_rates_loaded": True,
            "v2_terminal_pair_outcome_loaded": False,
            "v3_route_outcomes_loaded_before_freeze": False,
            "v3_model_sampled_before_freeze": False,
            "coefficient_changed": False,
        },
    }
    (output_root / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[point-repair-v3-data] "
        f"train={len(rows_by_split['train'])} "
        f"dev={len(rows_by_split['dev'])} "
        f"eval={len(rows_by_split['eval'])} "
        f"routes={2 * len(certification)}"
    )


if __name__ == "__main__":
    main()
