#!/usr/bin/env python3
"""Build the three-page report with one consistent live Ant v17 binding."""

from __future__ import annotations

import build_e70_historical_multipage_v2_20260730 as v2


v2.v1.DIAGNOSTIC_RENDERER = (
    v2.v1.ROOT
    / "ops/exp_scaling/"
    "plot_e70_current_diagnostics_square_v2_20260730.py"
)


if __name__ == "__main__":
    v2.main()
