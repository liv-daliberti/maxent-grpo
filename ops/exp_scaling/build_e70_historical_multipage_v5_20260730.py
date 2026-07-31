#!/usr/bin/env python3
"""Build the three-page report with the live Ant v18 repair binding."""

from __future__ import annotations

import build_e70_historical_multipage_v4_20260730 as v4


v4.v3.v2.v1.PRIMARY_RENDERER = (
    v4.v3.v2.v1.ROOT
    / "ops/exp_scaling/"
    "plot_e70_primary_square_repairs_v5_20260730.py"
)
v4.v3.v2.v1.DIAGNOSTIC_RENDERER = (
    v4.v3.v2.v1.ROOT
    / "ops/exp_scaling/"
    "plot_e70_current_diagnostics_square_v4_20260730.py"
)


if __name__ == "__main__":
    v4.v3.v2.main()
