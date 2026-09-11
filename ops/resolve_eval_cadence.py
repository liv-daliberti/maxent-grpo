#!/usr/bin/env python3
"""Resolve a hardware-independent evaluation cadence.

Quarter-epoch evaluation remains the fail-closed default.  A frozen campaign
may explicitly allow a sparser requested interval when full-benchmark
evaluation would otherwise dominate the training budget.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCadence:
    effective_pool_size: int
    quarter_prompt_interval: int
    prompt_interval: int
    eval_steps: int


def resolve_quarter_epoch_cadence(
    *,
    prompt_pool_size: int,
    max_train: int,
    rollout_batch_size: int,
    requested_prompt_interval: int | None = None,
    allow_sparse_requested_interval: bool = False,
) -> EvalCadence:
    """Return the default quarter-epoch or an explicitly allowed sparse cadence."""
    for name, value in (
        ("prompt_pool_size", prompt_pool_size),
        ("max_train", max_train),
        ("rollout_batch_size", rollout_batch_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if requested_prompt_interval is not None and requested_prompt_interval <= 0:
        raise ValueError(
            "requested_prompt_interval must be positive, "
            f"got {requested_prompt_interval}"
        )

    effective_pool_size = min(prompt_pool_size, max_train)
    quarter_prompt_interval = math.ceil(effective_pool_size / 4)
    prompt_interval = quarter_prompt_interval
    if requested_prompt_interval is not None:
        prompt_interval = (
            requested_prompt_interval
            if allow_sparse_requested_interval
            else min(requested_prompt_interval, quarter_prompt_interval)
        )
    eval_steps = math.ceil(prompt_interval / rollout_batch_size)
    return EvalCadence(
        effective_pool_size=effective_pool_size,
        quarter_prompt_interval=quarter_prompt_interval,
        prompt_interval=prompt_interval,
        eval_steps=eval_steps,
    )


def load_prompt_pool_size(prompt_data: str, train_split: str) -> int:
    from datasets import DatasetDict, load_from_disk

    dataset = load_from_disk(prompt_data)
    if isinstance(dataset, DatasetDict):
        if train_split not in dataset:
            available = ", ".join(sorted(dataset))
            raise ValueError(
                f"train split {train_split!r} not found; available: {available}"
            )
        dataset = dataset[train_split]
    return len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--max-train", type=int, required=True)
    parser.add_argument("--rollout-batch-size", type=int, required=True)
    parser.add_argument("--requested-prompt-interval", type=int)
    parser.add_argument(
        "--allow-sparse-requested-interval",
        action="store_true",
        help=(
            "honor a requested interval looser than quarter-epoch; this must "
            "be frozen and audited by the calling campaign"
        ),
    )
    args = parser.parse_args()

    cadence = resolve_quarter_epoch_cadence(
        prompt_pool_size=load_prompt_pool_size(args.prompt_data, args.train_split),
        max_train=args.max_train,
        rollout_batch_size=args.rollout_batch_size,
        requested_prompt_interval=args.requested_prompt_interval,
        allow_sparse_requested_interval=args.allow_sparse_requested_interval,
    )
    print(
        cadence.effective_pool_size,
        cadence.quarter_prompt_interval,
        cadence.prompt_interval,
        cadence.eval_steps,
        sep="\t",
    )


if __name__ == "__main__":
    main()
