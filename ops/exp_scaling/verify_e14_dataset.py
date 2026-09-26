#!/usr/bin/env python3
"""Validate and hash E14's frozen three-hidden-node graph data."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from oat_drgrpo.math_grader import boxed_reward_fn


EXPECTED_COMBINED_CONTENT_HASH = (
    "8e8bd8d4986920784cb067f99f4dc1b401f553b0d40a378dd4ada5dab29b48d6"
)


def validate_e14_rows(
    rows: list[dict[str, Any]], *, split_tag: str
) -> list[dict[str, Any]]:
    """Return canonical row records after enforcing E14's data contract."""

    canonical_rows = []
    for index, row in enumerate(rows):
        try:
            reference = json.loads(str(row["answer"]))
            problem = str(row["problem"])
            partial = list(reference["partial_colors"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{split_tag}[{index}] is not a graph-coloring reference"
            ) from error
        if reference.get("verifier") != "graph_coloring":
            raise ValueError(f"{split_tag}[{index}] has the wrong verifier")
        if sum(color is None for color in partial) != 3:
            raise ValueError(
                f"{split_tag}[{index}] does not have exactly three hidden nodes"
            )
        for required_text in (
            "There are exactly 3 question marks.",
            "exactly 3 digits",
            "for the missing positions from left to right",
            "inside \\boxed{}.",
        ):
            if required_text not in problem:
                raise ValueError(
                    f"{split_tag}[{index}] is missing prompt contract {required_text!r}"
                )
        valid_action_count = 0
        for action_tuple in itertools.product("123", repeat=3):
            _, reward = boxed_reward_fn("".join(action_tuple), str(row["answer"]))
            valid_action_count += int(float(reward) == 1.0)
        if valid_action_count <= 0:
            raise ValueError(f"{split_tag}[{index}] has no grader-valid action")
        declared_mode_count = int(row.get("answer_mode_count", 0))
        if declared_mode_count != valid_action_count:
            raise ValueError(
                f"{split_tag}[{index}] answer_mode_count mismatch: "
                f"declared={declared_mode_count} grader={valid_action_count}"
            )
        if valid_action_count < 2:
            raise ValueError(f"{split_tag}[{index}] is not a multi-answer row")
        canonical_rows.append(
            {
                "answer": str(row["answer"]),
                "answer_mode_count": declared_mode_count,
                "answer_mode_split": str(row["answer_mode_split"]),
                "modebench_task": str(row["modebench_task"]),
                "problem": problem,
            }
        )
    return canonical_rows


def _content_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    train = load_from_disk(str(args.data_root / "train"))["train"]
    evaluation = load_from_disk(str(args.data_root / "eval"))["multi_answer"]
    train_rows = validate_e14_rows(list(train), split_tag="train")
    eval_rows = validate_e14_rows(list(evaluation), split_tag="multi_answer")
    if len(train_rows) != 192 or len(eval_rows) != 96:
        raise SystemExit(
            "E14 requires the frozen 192/96 train/eval pools; "
            f"got {len(train_rows)}/{len(eval_rows)}"
        )
    combined_content_hash = _content_hash({"eval": eval_rows, "train": train_rows})
    if combined_content_hash != EXPECTED_COMBINED_CONTENT_HASH:
        raise SystemExit(
            "E14 data do not match the frozen content hash: "
            f"expected={EXPECTED_COMBINED_CONTENT_HASH} "
            f"observed={combined_content_hash}"
        )
    print(
        json.dumps(
            {
                "combined_content_hash": combined_content_hash,
                "eval_content_hash": _content_hash(eval_rows),
                "eval_rows": len(eval_rows),
                "hidden_nodes_per_row": 3,
                "train_content_hash": _content_hash(train_rows),
                "train_rows": len(train_rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
