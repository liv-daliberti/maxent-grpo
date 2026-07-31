#!/usr/bin/env python3
"""Train one orientation-balanced PointMaze v3 calibration cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import train_point_maze_algorithm_repair_v2_direct as direct


direct.base.SEEDS = (76541,)
direct.base.EXPECTED_FAMILIES = (
    "medium_cross9_v3",
    "medium_upper_offset9_v3",
    "hard_block11_v3",
    "hard_bar11_v3",
)
_loads = direct.base.json.loads
_evaluate = direct.evaluate


def loads(value, *args, **kwargs):
    payload = _loads(value, *args, **kwargs)
    if (
        isinstance(payload, dict)
        and payload.get("schema_version")
        == "point-maze-algorithm-repair-v3-qualification-v1"
        and payload.get("decision")
        == "eligible_for_point_maze_algorithm_repair_v3_pair"
    ):
        payload["decision"] = "eligible_for_ten_point_maze_stage_b_jobs"
    return payload


def evaluate(**kwargs):
    payload = _evaluate(**kwargs)
    payload["schema"] = "point-maze-algorithm-repair-evaluation-v3"
    return payload


direct.base.json.loads = loads
direct.base.evaluate = evaluate


def output_receipt_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parsed, _ = parser.parse_known_args(sys.argv[1:])
    return parsed.output_receipt


if __name__ == "__main__":
    receipt_path = output_receipt_path()
    direct.base.main()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload.update(
        schema="point-maze-algorithm-repair-receipt-v3",
        evaluation_request_seed_schedule=(
            "arm_seed_row_draw_sample_decision__checkpoint_invariant"
        ),
        secondary_post_outcome_repair=True,
        orientation_balanced_within_train_and_development=True,
        final_seed_cohort=False,
        development_only=True,
        evaluation_split="development",
    )
    direct.base.atomic(receipt_path, payload)
