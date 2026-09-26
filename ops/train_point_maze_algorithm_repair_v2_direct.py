#!/usr/bin/env python3
"""Direct PointMaze repair-v2 trainer with checkpoint-invariant evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import train_point_maze_stage_b_05b_12pass as base


base.SEEDS = (76521,)
base.EXPECTED_FAMILIES = (
    "cross9_balanced",
    "upper_offset9_balanced",
    "block11_balanced",
    "bar11_balanced",
)
_evaluate = base.evaluate


def evaluate(**kwargs):
    actual_update = int(kwargs["update"])
    kwargs["update"] = 0
    result = _evaluate(**kwargs)
    result.update(
        schema="point-maze-algorithm-repair-evaluation-v2",
        learning_round=actual_update,
        training_passes=actual_update / base.TRAIN_PROMPTS,
        evaluation_request_seed_schedule=(
            "arm_seed_row_draw_sample_decision__checkpoint_invariant"
        ),
    )
    return result


base.evaluate = evaluate


def output_receipt_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parsed, _ = parser.parse_known_args(sys.argv[1:])
    return parsed.output_receipt


if __name__ == "__main__":
    receipt_path = output_receipt_path()
    base.main()
    payload = json.loads(receipt_path.read_text())
    payload.update(
        schema="point-maze-algorithm-repair-receipt-v2",
        evaluation_request_seed_schedule=(
            "arm_seed_row_draw_sample_decision__checkpoint_invariant"
        ),
        secondary_post_outcome_repair=True,
        k16_aligned_algorithmic_gate=True,
    )
    base.atomic(receipt_path, payload)
