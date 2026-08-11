#!/usr/bin/env python3
"""Report cells, terminal counts, and realized depth for every active cohort.

Read-only. Progress comes from realized optimizer steps in each run's metrics
log; the scheduler query only explains which cells are running or waiting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parent))
import status_e78 as shared  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"

import cohorts as registry  # noqa: E402

# Derived from the single cohort registry, so a launched cohort cannot be
# missing from this table without also failing test_cohort_registry.py.
COHORTS = tuple(
    (c.label, c.ledger, c.kind == "point_maze") for c in registry.REGISTRY
)


def cohort_row(label: str, ledger: Path, point: bool) -> dict[str, object] | None:
    if not ledger.is_file():
        return None
    snapshot = (
        shared.load_point_snapshot(ledger) if point else shared.load_snapshot(ledger)
    )
    rows = snapshot["rows"]
    target = int(snapshot["target"])
    steps_per_pass = int(snapshot["steps_per_pass"])
    counts = shared.state_counts(rows)
    realized = sum(int(row["step"]) for row in rows)
    return {
        "label": label,
        "cells": len(rows),
        "terminal": sum(int(row["step"]) >= target for row in rows),
        "running": counts["RUNNING"],
        "pending": counts["PENDING"],
        "realized": realized,
        "total": len(rows) * target,
        "depth": realized / (len(rows) * steps_per_pass) if rows else 0.0,
        "passes": int(snapshot["passes"]),
    }


def render(rows: list[dict[str, object]], *, markdown: bool) -> str:
    cells = sum(int(r["cells"]) for r in rows)
    terminal = sum(int(r["terminal"]) for r in rows)
    running = sum(int(r["running"]) for r in rows)
    pending = sum(int(r["pending"]) for r in rows)
    realized = sum(int(r["realized"]) for r in rows)
    total = sum(int(r["total"]) for r in rows)

    if markdown:
        out = [
            "| cohort | cells | terminal | running | pending | steps | depth |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            out.append(
                f"| {r['label']} | {r['cells']} | {r['terminal']} | {r['running']} "
                f"| {r['pending']} | {r['realized']:,}/{r['total']:,} "
                f"| {r['depth']:.2f}p |"
            )
        out.append(
            f"| **total** | **{cells}** | **{terminal}** | {running} | {pending} "
            f"| {realized:,}/{total:,} | **{100 * realized / total:.1f}%** |"
        )
        return "\n".join(out)

    # Width follows the labels rather than a constant, so naming a cohort
    # accurately can never be discouraged by the table losing its alignment.
    width = max([len("cohort"), len("TOTAL"), *(len(str(r["label"])) for r in rows)])
    rule = "-" * (width + 50)
    out = [
        time.strftime("campaign status  %Y-%m-%d %H:%M:%S %Z"),
        "",
        f"{'cohort':<{width}} {'cells':>5} {'term':>5} {'run':>4} {'pend':>5} "
        f"{'steps':>19} {'depth':>7}",
        rule,
    ]
    for r in rows:
        out.append(
            f"{r['label']:<{width}} {r['cells']:>5} {r['terminal']:>5} "
            f"{r['running']:>4} {r['pending']:>5} "
            f"{r['realized']:>9,}/{r['total']:<9,} "
            f"{r['depth']:>5.2f}/{r['passes']}p"
        )
    out.append(rule)
    out.append(
        f"{'TOTAL':<{width}} {cells:>5} {terminal:>5} {running:>4} {pending:>5} "
        f"{realized:>9,}/{total:<9,} {100 * realized / total:>6.1f}%"
    )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="emit a markdown table")
    parser.add_argument("--json", action="store_true", help="emit the raw rows")
    args = parser.parse_args()

    rows = [
        row
        for label, name, point in COHORTS
        if (row := cohort_row(label, ARTIFACTS / name, point)) is not None
    ]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(render(rows, markdown=args.markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
