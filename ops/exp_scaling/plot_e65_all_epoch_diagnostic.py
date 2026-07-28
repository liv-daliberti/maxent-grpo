#!/usr/bin/env python3
"""Render the full five-domain diagnostic with every integer MATH epoch."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_e61r1_e58_vs_grpo_12pass as primary


primary.OUT = (
    primary.ROOT
    / "paper/figures/e61r1_e58_vs_grpo_05b_12ep_all_epoch_diagnostic_live"
)
primary.MATH500_DISPLAY_PASSES = tuple(range(13))
primary.MATH500_DISPLAY_DESCRIPTION = (
    "every complete integer epoch on the supplemental diagnostic surface"
)


if __name__ == "__main__":
    primary.main()
