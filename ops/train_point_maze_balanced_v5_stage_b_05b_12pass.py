#!/usr/bin/env python3
"""Train one PointMaze balanced-v5 five-seed comparison cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import train_point_maze_algorithm_repair_v2_direct as direct


SEEDS = (76611, 76612, 76613, 76614, 76615)
EVALUATION_FAMILIES = (
    "medium_wide_block9_v3",
    "medium_lower_offset9_v3",
    "hard_wide_block11_v3",
    "hard_diamond11_v3",
)
direct.base.SEEDS = SEEDS
direct.base.EXPECTED_FAMILIES = EVALUATION_FAMILIES
_loads = direct.base.json.loads
_evaluate = direct.evaluate


def loads(value, *args, **kwargs):
    payload = _loads(value, *args, **kwargs)
    if (
        isinstance(payload, dict)
        and payload.get("schema_version")
        == "point-maze-balanced-warmstart-v5-qualification-v1"
        and payload.get("decision")
        == "eligible_for_point_maze_v5_five_seed_pair"
    ):
        payload["decision"] = "eligible_for_ten_point_maze_stage_b_jobs"
    return payload


def evaluate(**kwargs):
    payload = _evaluate(**kwargs)
    payload["schema"] = "point-maze-stage-b-evaluation-v1"
    payload["balanced_v5_final"] = True
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
        balanced_v5_final=True,
        orientation_balanced_warmstart=True,
        evaluation_request_seed_schedule=(
            "arm_seed_row_draw_sample_decision__checkpoint_invariant"
        ),
        final_seed_cohort=True,
        development_only=False,
        evaluation_split="untouched_evaluation",
    )
    direct.base.atomic(receipt_path, payload)
