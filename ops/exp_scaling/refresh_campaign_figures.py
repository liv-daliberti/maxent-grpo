#!/usr/bin/env python3
"""Refresh all live campaign curves and their combined/split figures once."""

from __future__ import annotations

import fcntl
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "var/artifacts"
LOCK_PATH = ARTIFACTS / "refresh_campaign_figures.lock"
STEPS = (
    ROOT / "ops/exp_scaling/refresh_campaign_curves.py",
    ROOT / "ops/plot_canonical_maxent_paper.py",
    ROOT / "ops/exp_scaling/plot_divergence.py",
    ROOT / "ops/plot_e21_math_token_maxent_live.py",
)


def run_step(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(f"[figure-refresh] {script.name} failed with status {result.returncode}")
        print(result.stdout.rstrip())
        raise SystemExit(result.returncode)


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[figure-refresh] skipped: another refresh is still running")
            return 0
        started = datetime.now().astimezone()
        for script in STEPS:
            run_step(script)
        elapsed = (datetime.now().astimezone() - started).total_seconds()
        print(
            "[figure-refresh] completed "
            f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z} "
            f"in {elapsed:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
