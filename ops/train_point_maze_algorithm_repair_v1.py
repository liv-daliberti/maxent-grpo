#!/usr/bin/env python3
"""Train one PointMaze repair arm with checkpoint-invariant evaluation draws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import train_point_maze_stage_b_05b_12pass as base


base.SEEDS = (76501,)
base.EXPECTED_FAMILIES = (
    "block11_repair",
    "wide_block11_repair",
    "bar11_repair",
    "diamond11_repair",
)
_evaluate = base.evaluate


def evaluate(**kwargs):
    actual_update = int(kwargs["update"])
    # The original evaluator varied Monte Carlo request seeds by checkpoint.
    # Calling it at coordinate zero gives every checkpoint common random
    # numbers; only the model parameters vary.
    kwargs["update"] = 0
    result = _evaluate(**kwargs)
    result["schema"] = "point-maze-algorithm-repair-evaluation-v1"
    result["learning_round"] = actual_update
    result["training_passes"] = actual_update / base.TRAIN_PROMPTS
    result["evaluation_request_seed_schedule"] = (
        "arm_seed_row_draw_sample_decision__checkpoint_invariant"
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
    payload["schema"] = "point-maze-algorithm-repair-receipt-v1"
    payload["evaluation_request_seed_schedule"] = (
        "arm_seed_row_draw_sample_decision__checkpoint_invariant"
    )
    payload["secondary_post_outcome_repair"] = True
    base.atomic(receipt_path, payload)

