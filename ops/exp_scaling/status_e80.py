#!/usr/bin/env python3
"""Report E80 Qwen2.5-3B progress, optionally with all active paper cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import status_e78 as shared  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
E80_LEDGER = ROOT / "var/artifacts/e80_qwen3b_aligned_verified_replay_jobs.json"
E79_LEDGER = ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json"
E78_LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
E79PM_LEDGER = ROOT / "var/artifacts/e79pm_falcon_point_maze_verified_replay_jobs.json"


def cohort(path: Path, label: str) -> str:
    report = shared.render(shared.load_snapshot(path.resolve()))
    report = report.replace("E78 status", label, 1)
    return report.replace("optimizer steps", "prompt updates")


def render_all(qwen3b_only: bool = False) -> str:
    reports = [cohort(E80_LEDGER, "E80 Qwen2.5-3B status")]
    if qwen3b_only:
        return "\n\n".join(reports)
    for path, label in (
        (E78_LEDGER, "E78 Qwen2.5-0.5B status"),
        (E79_LEDGER, "E79 Falcon3-1B status"),
    ):
        if path.is_file():
            reports.append(cohort(path, label))
    if shared.POINT_LEDGER.is_file():
        reports.append(
            shared.render_point(shared.load_point_snapshot(shared.POINT_LEDGER))
            .replace("E78-PM", "E78-PM Qwen2.5-0.5B")
            .replace("optimizer steps", "prompt updates")
        )
    if E79PM_LEDGER.is_file():
        reports.append(
            shared.render_point(shared.load_point_snapshot(E79PM_LEDGER))
            .replace("E78-PM", "E79-PM Falcon3-1B")
            .replace("optimizer steps", "prompt updates")
        )
    return "\n\n".join(reports)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen3b-only",
        action="store_true",
        help="hide E78, E79, and PointMaze sections",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        metavar="SECONDS",
        help="refresh continuously at this interval; Ctrl-C exits",
    )
    args = parser.parse_args()
    while True:
        print(render_all(args.qwen3b_only), flush=True)
        if args.watch <= 0:
            return
        print(flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()

