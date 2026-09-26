#!/usr/bin/env python3
"""Materialize and MuJoCo-certify the PointMaze waypoint pilot dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
from typing import Any


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.point_maze_waypoint import (  # noqa: E402
    POINT_WAYPOINT_ACTION_VERSION,
    POINT_WAYPOINT_CONTROLLER_SHA256,
)
from oat_drgrpo.point_maze_waypoint_data import (  # noqa: E402
    DEFAULT_POINT_WAYPOINT_SPLIT_COUNTS,
    POINT_WAYPOINT_DATA_SEED,
    generate_point_waypoint_tasks,
    point_waypoint_task_rows,
)
from oat_drgrpo.point_maze_waypoint_process import (  # noqa: E402
    PointMazeWaypointProcess,
)


DEFAULT_WORKER_PYTHON = ROOT / "var/maze_runtime/venv/bin/python"
DEFAULT_OUTPUT = ROOT / "var/data/point_maze_waypoint_pilot_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity(worker_python: Path) -> dict[str, Any]:
    source = (
        "import json,os,sys;"
        "sys.path.insert(0,os.environ.get('OAT_ZERO_SOURCE_ROOT','src'));"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=POINT_WAYPOINT_DATA_SEED)
    parser.add_argument(
        "--train-count",
        type=int,
        default=DEFAULT_POINT_WAYPOINT_SPLIT_COUNTS["train"],
    )
    parser.add_argument(
        "--dev-count",
        type=int,
        default=DEFAULT_POINT_WAYPOINT_SPLIT_COUNTS["dev"],
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=DEFAULT_POINT_WAYPOINT_SPLIT_COUNTS["eval"],
    )
    parser.add_argument(
        "--exclude-identity",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Development smoke only; materialize this many rows per split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"fresh waypoint output required: {output}")
    if not args.worker_python.is_file():
        raise FileNotFoundError(f"missing pinned maze runtime: {args.worker_python}")
    counts = {
        "train": int(args.train_count),
        "dev": int(args.dev_count),
        "eval": int(args.eval_count),
    }
    if any(count <= 0 for count in counts.values()):
        raise ValueError("waypoint split counts must be positive")
    if args.limit_per_split is not None:
        if not 1 <= args.limit_per_split <= min(counts.values()):
            raise ValueError("limit-per-split is outside the pilot split bounds")
        counts = {split: int(args.limit_per_split) for split in counts}

    excluded_fingerprints: set[str] = set()
    excluded_identities = []
    for path in args.exclude_identity:
        prior = json.loads(path.read_text(encoding="utf-8"))
        fingerprints = {
            str(record["instance_fingerprint"])
            for record in prior.get("certification", [])
        }
        if not fingerprints:
            raise ValueError(f"excluded waypoint identity has no fingerprints: {path}")
        excluded_fingerprints.update(fingerprints)
        excluded_identities.append(
            {"path": str(path.resolve()), "sha256": _sha256_file(path), "fingerprint_count": len(fingerprints)}
        )

    runtime = _runtime_identity(args.worker_python)
    tasks = generate_point_waypoint_tasks(
        environment_sha256=runtime["point_environment_sha256"],
        split_counts=counts,
        seed=args.seed,
        excluded_fingerprints=tuple(sorted(excluded_fingerprints)),
    )
    certification = []
    with PointMazeWaypointProcess(worker_python=args.worker_python) as worker:
        for split in ("train", "dev", "eval"):
            for row_index, task in enumerate(tasks[split]):
                routes = []
                for route_index, (expected_route, program) in enumerate(
                    sorted(task.route_programs.items())
                ):
                    result = worker.execute_program(
                        session_id=f"cert-{split}-{row_index}-{route_index}",
                        spec=task.spec,
                        actions=program.split(),
                    )
                    canonical_key = result.get("canonical_key")
                    if (
                        result.get("success") is not True
                        or result.get("validation_error") is not None
                        or not isinstance(canonical_key, str)
                        or not canonical_key.endswith(":" + expected_route)
                    ):
                        raise RuntimeError(
                            f"{task.spec['map_id']} route {expected_route} failed "
                            f"certification: {result}"
                        )
                    routes.append(
                        {
                            "route_key": expected_route,
                            "program": program,
                            "program_sha256": hashlib.sha256(
                                program.encode("ascii")
                            ).hexdigest(),
                            "canonical_key": canonical_key,
                            "simulator_steps": int(result["simulator_steps"]),
                        }
                    )
                certification.append(
                    {
                        "split": split,
                        "row_index": row_index,
                        "map_id": task.spec["map_id"],
                        "spec_sha256": task.spec["spec_sha256"],
                        "instance_fingerprint": task.instance_fingerprint,
                        "routes": routes,
                    }
                )

    from datasets import Dataset, DatasetDict

    rows = point_waypoint_task_rows(tasks)
    dataset = DatasetDict(
        {split: Dataset.from_list(split_rows) for split, split_rows in rows.items()}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output))
    identity = {
        "schema": "point-maze-waypoint-pilot-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_seed": args.seed,
        "excluded_prior_identities": excluded_identities,
        "split_counts": counts,
        "environment_identity": runtime,
        "action_version": POINT_WAYPOINT_ACTION_VERSION,
        "controller_sha256": POINT_WAYPOINT_CONTROLLER_SHA256,
        "source_sha256": {
            "point_maze_waypoint.py": _sha256_file(
                SRC / "oat_drgrpo/point_maze_waypoint.py"
            ),
            "point_maze_waypoint_data.py": _sha256_file(
                SRC / "oat_drgrpo/point_maze_waypoint_data.py"
            ),
            "point_maze_waypoint_worker.py": _sha256_file(
                SRC / "oat_drgrpo/point_maze_waypoint_worker.py"
            ),
        },
        "information_boundary": {
            "certified_programs_in_model_rows": False,
            "evaluation_rows_used_for_training": False,
            "legal_mask": "adjacent free cells only",
            "goal_filtering": False,
            "shortest_path_filtering": False,
            "revisit_filtering": False,
        },
        "certification": certification,
    }
    (output / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "split_counts": counts,
                "certified_routes": sum(len(row["routes"]) for row in certification),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
