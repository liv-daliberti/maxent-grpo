#!/usr/bin/env python3
"""Translate E14 optimizer updates into OAT's trajectory-query stop budget."""

from __future__ import annotations

import argparse


def max_queries_for_exact_updates(*, updates: int, trajectories_per_update: int) -> int:
    """Return a budget that stops after exactly ``updates`` under strict ``>``.

    OAT increments ``query_step`` by the number of trajectories and checks
    ``query_step > max_queries`` after each update. Any budget in
    ``[(updates-1)G, updates*G-1]`` therefore stops after ``updates``. We use
    the lower boundary, except that OAT treats nonpositive budgets as unset.
    """

    if updates <= 0 or trajectories_per_update <= 0:
        raise ValueError("updates and trajectories_per_update must be positive")
    return max(1, (updates - 1) * trajectories_per_update)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--trajectories-per-update", type=int, required=True)
    args = parser.parse_args()
    print(
        max_queries_for_exact_updates(
            updates=args.updates,
            trajectories_per_update=args.trajectories_per_update,
        )
    )


if __name__ == "__main__":
    main()
