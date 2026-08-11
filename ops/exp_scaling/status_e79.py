#!/usr/bin/env python3
"""Report combined progress for E79 Falcon3-1B and E78-PM PointMaze."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import status_e78 as shared  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json"
FALCON_POINT_LEDGER = (
    ROOT / "var/artifacts/e79pm_falcon_point_maze_verified_replay_jobs.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        metavar="SECONDS",
        help="refresh continuously at this interval; Ctrl-C exits",
    )
    args = parser.parse_args()
    while True:
        report = shared.render(shared.load_snapshot(args.ledger.resolve()))
        report = report.replace("E78 status", "E79 Falcon status", 1)
        report = report.replace("optimizer steps", "prompt updates")
        if shared.POINT_LEDGER.is_file():
            point = shared.render_point(
                shared.load_point_snapshot(shared.POINT_LEDGER)
            ).replace("optimizer steps", "prompt updates")
            report += "\n\n" + point
        if FALCON_POINT_LEDGER.is_file():
            point = shared.render_point(
                shared.load_point_snapshot(FALCON_POINT_LEDGER)
            ).replace("optimizer steps", "prompt updates")
            report += "\n\n" + point
        print(report, flush=True)
        if args.watch <= 0:
            return
        print(flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
