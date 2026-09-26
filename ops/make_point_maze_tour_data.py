#!/usr/bin/env python
"""Materialize PointMaze Tour maps by executing every landmark order.

Stage 0 of the v2 admission ladder.  For each candidate map this executes all
``K!`` orders through the pinned runtime, calibrates the map's step budget so
that the number of orders finishing inside it lands in the target band, and
admits the map only if such a budget exists.  A map's certified tour set is
therefore measured, never declared.

Certified orders are written to ``identity.json`` as audit witnesses.  They
never enter a model row, a warm start, online training, or a replay bank.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oat_drgrpo.maze_runtime_identity import maze_runtime_identity  # noqa: E402
from oat_drgrpo.point_maze_tour import (  # noqa: E402
    iter_tour_orders,
    parse_point_tour_spec,
    render_point_tour_problem,
    tour_grid_cost,
    with_tour_step_budget,
)
from oat_drgrpo.point_maze_tour_data import (  # noqa: E402
    DEFAULT_TOUR_SPLIT_COUNTS,
    POINT_TOUR_DATA_SEED,
    TOUR_LANDMARKS,
    choose_tour_budget,
    point_tour_candidate,
    point_tour_task_row,
)

_WORKER: Any = None


def _worker():
    global _WORKER
    if _WORKER is None:
        from oat_drgrpo.point_maze_tour_worker import InteractivePointTourWorker

        _WORKER = InteractivePointTourWorker()
    return _WORKER


def measure_candidate(job: tuple[str, int, str, int, int, int]) -> dict[str, Any]:
    """Execute every order of one candidate map and calibrate its budget."""

    split, attempt, environment_sha256, seed, minimum_modes, maximum_modes = job
    task = point_tour_candidate(
        environment_sha256=environment_sha256,
        split=split,
        attempt=attempt,
        seed=seed,
    )
    spec = parse_point_tour_spec(task.spec)
    worker = _worker()
    measured: list[dict[str, Any]] = []
    for index, order in enumerate(iter_tour_orders(spec)):
        session_id = f"{split}-{attempt}-{index}"
        worker.reset({"session_id": session_id, "spec": task.spec})
        response = None
        for landmark in order:
            response = worker.step_leg(
                {"session_id": session_id, "landmark": landmark}
            )
            if response["done"]:
                break
        if response is None:
            continue
        if response["success"] and response["validation_error"] is None:
            measured.append(
                {
                    "order": list(order),
                    "executed_order": list(response["landmark_order"]),
                    "canonical_key": response["canonical_key"],
                    "steps": int(response["simulator_steps"]),
                    "grid_cells": tour_grid_cost(spec, order),
                }
            )
    if not measured:
        return {"split": split, "attempt": attempt, "admitted": False, "reason": "no_order_completed"}
    if any(row["order"] != row["executed_order"] for row in measured):
        return {
            "split": split,
            "attempt": attempt,
            "admitted": False,
            "reason": "executed_order_disagrees_with_choice",
        }
    if len({row["canonical_key"] for row in measured}) != len(measured):
        return {
            "split": split,
            "attempt": attempt,
            "admitted": False,
            "reason": "canonical_keys_collide",
        }
    chosen = choose_tour_budget(
        [row["steps"] for row in measured],
        minimum_modes=minimum_modes,
        maximum_modes=maximum_modes,
    )
    if chosen is None:
        return {
            "split": split,
            "attempt": attempt,
            "admitted": False,
            "reason": "no_budget_in_band",
            "completed": len(measured),
        }
    budget, admitted_count = chosen
    final_spec = with_tour_step_budget(task.spec, budget)
    certified = sorted(
        (row for row in measured if row["steps"] <= budget),
        key=lambda row: (row["steps"], row["order"]),
    )
    return {
        "split": split,
        "attempt": attempt,
        "admitted": True,
        "family": task.family,
        "instance_fingerprint": task.instance_fingerprint,
        "spec": final_spec,
        "problem": render_point_tour_problem(final_spec),
        "tour_step_budget": budget,
        "certified_tour_count": admitted_count,
        "completed_order_count": len(measured),
        "certified_orders": certified,
        "step_cost_min": min(row["steps"] for row in measured),
        "step_cost_max": max(row["steps"] for row in measured),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=POINT_TOUR_DATA_SEED)
    parser.add_argument("--landmarks", type=int, default=TOUR_LANDMARKS)
    parser.add_argument("--minimum-modes", type=int, default=8)
    parser.add_argument("--maximum-modes", type=int, default=16)
    parser.add_argument("--train", type=int, default=DEFAULT_TOUR_SPLIT_COUNTS["train"])
    parser.add_argument("--dev", type=int, default=DEFAULT_TOUR_SPLIT_COUNTS["dev"])
    parser.add_argument("--eval", type=int, default=DEFAULT_TOUR_SPLIT_COUNTS["eval"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-attempts-per-map", type=int, default=40)
    parser.add_argument("--probe", type=int, default=0, help="measure N candidates and exit")
    parser.add_argument(
        "--stage-output",
        type=Path,
        help=(
            "write measured rows here and stop. The maze runtime has no "
            "datasets package, so enumeration and packing run in two phases."
        ),
    )
    parser.add_argument(
        "--pack-from",
        type=Path,
        help="pack a staged file into a DatasetDict; requires the paper venv",
    )
    return parser.parse_args()


def pack(stage_path: Path, output_root: Path) -> None:
    """Phase two: turn staged rows into the DatasetDict the trainer loads."""

    from datasets import Dataset, DatasetDict

    staged = json.loads(stage_path.read_text(encoding="ascii"))
    split_rows = staged["split_rows"]
    identity = staged["identity"]
    dataset = DatasetDict(
        {split: Dataset.from_list(rows) for split, rows in split_rows.items()}
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_root))
    identity["row_sha256"] = {
        split: hashlib.sha256(
            json.dumps(rows, allow_nan=False, sort_keys=True).encode("ascii")
        ).hexdigest()
        for split, rows in split_rows.items()
    }
    identity_path = output_root / "identity.json"
    identity_path.write_text(
        json.dumps(identity, allow_nan=False, indent=1, sort_keys=True) + "\n",
        encoding="ascii",
    )
    for split, rows in split_rows.items():
        modes = [row["certified_tour_count"] for row in rows]
        print(
            f"{split}: {len(rows)} maps, mean certified tours "
            f"{sum(modes) / len(modes):.2f} (min {min(modes)}, max {max(modes)})"
        )
    print(f"wrote {identity_path}")


def main() -> None:
    args = parse_args()
    if args.pack_from is not None:
        pack(args.pack_from, args.output_root)
        return
    runtime = maze_runtime_identity()
    environment_sha256 = runtime["point_environment_sha256"]
    counts = {"train": args.train, "dev": args.dev, "eval": args.eval}

    if args.probe:
        jobs = [
            ("train", attempt, environment_sha256, args.seed, args.minimum_modes, args.maximum_modes)
            for attempt in range(args.probe)
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(measure_candidate, jobs))
        admitted = [row for row in results if row["admitted"]]
        print(f"probe: {len(admitted)}/{len(results)} candidates admitted")
        reasons: dict[str, int] = {}
        for row in results:
            if not row["admitted"]:
                reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
            print(f"  rejected {count:3d}  {reason}")
        if admitted:
            modes = [row["certified_tour_count"] for row in admitted]
            budgets = [row["tour_step_budget"] for row in admitted]
            print(f"  certified tours per map: min {min(modes)} mean {sum(modes)/len(modes):.2f} max {max(modes)}")
            print(f"  step budget: min {min(budgets)} max {max(budgets)}")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    accepted: dict[str, list[dict[str, Any]]] = {split: [] for split in counts}
    attempts_used: dict[str, int] = {split: 0 for split in counts}
    rejections: dict[str, int] = {}
    # Geometry fingerprints are deduplicated across every split, so the
    # evaluation maps are disjoint from training and development by
    # construction rather than by audit.
    seen_fingerprints: set[str] = set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split, needed in counts.items():
            attempt = 0
            limit = needed * args.max_attempts_per_map
            while len(accepted[split]) < needed:
                if attempt >= limit:
                    raise RuntimeError(
                        f"{split}: only {len(accepted[split])}/{needed} maps admitted "
                        f"within {limit} attempts"
                    )
                jobs = [
                    (
                        split,
                        attempt + offset,
                        environment_sha256,
                        args.seed,
                        args.minimum_modes,
                        args.maximum_modes,
                    )
                    for offset in range(args.batch)
                ]
                attempt += args.batch
                for row in pool.map(measure_candidate, jobs):
                    if not row["admitted"]:
                        rejections[row["reason"]] = rejections.get(row["reason"], 0) + 1
                        continue
                    if row["instance_fingerprint"] in seen_fingerprints:
                        rejections["duplicate_geometry"] = (
                            rejections.get("duplicate_geometry", 0) + 1
                        )
                        continue
                    if len(accepted[split]) < needed:
                        seen_fingerprints.add(row["instance_fingerprint"])
                        accepted[split].append(row)
                attempts_used[split] = attempt
                print(
                    f"  {split}: {len(accepted[split])}/{needed} admitted "
                    f"after {attempt} attempts",
                    flush=True,
                )

    fingerprints = {
        row["instance_fingerprint"] for rows in accepted.values() for row in rows
    }
    total = sum(len(rows) for rows in accepted.values())
    if len(fingerprints) != total:
        raise RuntimeError("tour maps contain duplicate geometry fingerprints")

    identity: dict[str, Any] = {
        "schema": "point-maze-tour-identity-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "landmarks": args.landmarks,
        "mode_band": [args.minimum_modes, args.maximum_modes],
        "runtime": runtime,
        "attempts_used": attempts_used,
        "rejections": rejections,
        "splits": {},
    }
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split, rows in accepted.items():
        split_rows[split] = [
            point_tour_task_row(
                problem=row["problem"],
                spec=row["spec"],
                family=row["family"],
                split=split,
                certified_tour_count=row["certified_tour_count"],
                instance_fingerprint=row["instance_fingerprint"],
            )
            for row in rows
        ]
        modes = [row["certified_tour_count"] for row in rows]
        identity["splits"][split] = {
            "count": len(rows),
            "mean_certified_tours": sum(modes) / len(modes),
            "min_certified_tours": min(modes),
            "max_certified_tours": max(modes),
            "maps": [
                {
                    "map_id": row["spec"]["map_id"],
                    "family": row["family"],
                    "instance_fingerprint": row["instance_fingerprint"],
                    "tour_step_budget": row["tour_step_budget"],
                    "certified_tour_count": row["certified_tour_count"],
                    "completed_order_count": row["completed_order_count"],
                    "step_cost_min": row["step_cost_min"],
                    "step_cost_max": row["step_cost_max"],
                    "certified_orders": row["certified_orders"],
                }
                for row in rows
            ],
        }
        print(
            f"{split}: {len(rows)} maps, mean certified tours "
            f"{identity['splits'][split]['mean_certified_tours']:.2f}"
        )

    stage_path = args.stage_output or (args.output_root / "staged_rows.json")
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        json.dumps(
            {"identity": identity, "split_rows": split_rows},
            allow_nan=False,
            sort_keys=True,
        ),
        encoding="ascii",
    )
    print(f"staged {stage_path}; pack it with --pack-from under the paper venv")


if __name__ == "__main__":
    main()
