#!/usr/bin/env python3
"""Report read-only progress for E81 beside its inherited E78 comparators.

E81 trains one new arm. Its scientific pair is the completed E78 cohort, so
this report shows the new arm's depth next to the `replay` and `control` depth
already on disk for the same domain and seed. Progress is read from realized
optimizer steps in each run's metrics log; the scheduler query only explains
which cells are running or waiting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parent))
import status_e78 as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = base.E81_LEDGER


def render(snapshot: dict[str, object], pairs: dict[tuple[str, int, str], int]) -> str:
    rows = snapshot["rows"]
    target = int(snapshot["target"])
    steps_per_pass = int(snapshot["steps_per_pass"])
    interval = int(snapshot["checkpoint_interval"])
    total = len(rows) * target
    realized = sum(int(row["step"]) for row in rows)

    lines = [
        time.strftime("E81 status  %Y-%m-%d %H:%M:%S %Z"),
        "arm: verified replay + fixed semantic MaxEnt (eta = 0.10)",
        (
            f"scheduler: {base.state_summary(base.state_counts(rows))}"
            if rows
            else "scheduler: no registered runs"
        ),
        (
            f"training:  {realized:,}/{total:,} optimizer steps "
            f"({100 * realized / total:.2f}%) | "
            f"started {sum(int(row['step']) > 0 for row in rows)}/{len(rows)} | "
            f"terminal {sum(int(row['step']) >= target for row in rows)}/{len(rows)}"
        ),
        (
            f"equivalent full-cohort depth: "
            f"{realized / (len(rows) * steps_per_pass):.2f}/"
            f"{snapshot['passes']} passes"
        ),
        "",
        "pass depth by cell, new arm beside its inherited E78 pair",
        "domain     seed  semantic    replay   control   paired complete",
        "---------- ---- --------- --------- --------- ----------------",
    ]
    paired_ready = 0
    for domain in snapshot["domains"]:
        for seed in sorted({int(row["seed"]) for row in rows}):
            group = [
                row
                for row in rows
                if row["domain"] == domain and int(row["seed"]) == seed
            ]
            if not group:
                continue
            semantic = max(int(row["step"]) for row in group)
            replay = pairs.get((str(domain), seed, "replay"), 0)
            control = pairs.get((str(domain), seed, "control"), 0)
            complete = min(semantic, replay, control) >= target
            paired_ready += complete
            lines.append(
                f"{base.DOMAIN_LABELS.get(str(domain), str(domain)):<10} "
                f"{seed:>4} {semantic / steps_per_pass:>9.1f} "
                f"{replay / steps_per_pass:>9.1f} "
                f"{control / steps_per_pass:>9.1f} "
                f"{'yes' if complete else 'no':>16}"
            )

    reach = []
    for step in range(interval, target + 1, interval):
        count = sum(int(row["checkpoint"]) >= step for row in rows)
        reach.append(f"{step / steps_per_pass:g}: {count}/{len(rows)}")
        if count == 0:
            break
    lines.extend(
        [
            "",
            f"analysable paired cells (all three arms at pass 8): "
            f"{paired_ready}/{len(rows)}",
            "registered checkpoint reach: " + " | ".join(reach),
            "depth is live metrics; E78 columns are inherited, never re-run.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        metavar="SECONDS",
        help="refresh continuously at this interval; Ctrl-C exits",
    )
    args = parser.parse_args()
    while True:
        snapshot = base.load_semantic_snapshot(args.ledger.resolve())
        print(
            render(snapshot, base.e78_pair_depth(int(snapshot["target"]))),
            flush=True,
        )
        if args.watch <= 0:
            return
        print(flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
