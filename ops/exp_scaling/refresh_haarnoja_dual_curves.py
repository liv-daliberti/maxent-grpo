#!/usr/bin/env python3
"""Refresh the six E5 Haarnoja-dual artifacts, including queued empty rows."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
PARSER = ROOT / "ops/exp_scaling/parse_scaling_curve.py"
ARTIFACTS = ROOT / "var/artifacts"
RUN_DATA = ROOT / "var/data"

CELLS = (
    ("cde5_haarnoja_05b_v1", 384, 16),
    ("gce5_haarnoja_05b_v1", 192, 16),
    ("cde5_haarnoja_3b_2xa5000_v1", 384, 32),
    ("gce5_haarnoja_3b_2xa5000_v1", 1024, 32),
    ("cde5_haarnoja_7b_v1", 384, 32),
    ("gce5_haarnoja_7b_v1", 1024, 32),
)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for stamp, prompt_pool_size, num_samples in CELLS:
        subprocess.run(
            [
                sys.executable,
                str(PARSER),
                "--stamp-prefix",
                stamp,
                "--run-data-root",
                str(RUN_DATA),
                "--out",
                str(ARTIFACTS / f"{stamp}_scaling_curve.json"),
                "--prompt-pool-size",
                str(prompt_pool_size),
                "--num-samples",
                str(num_samples),
                "--max-training-passes",
                "5",
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
