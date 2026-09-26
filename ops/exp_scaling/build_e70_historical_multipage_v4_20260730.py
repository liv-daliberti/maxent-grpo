#!/usr/bin/env python3
"""Build the three-page report with corrected Pantry and Ant bindings."""

from __future__ import annotations

import build_e70_historical_multipage_v3_20260730 as v3


v3.v2.v1.PRIMARY_RENDERER = (
    v3.v2.v1.ROOT
    / "ops/exp_scaling/"
    "plot_e70_primary_square_repairs_v4_20260730.py"
)
v3.v2.v1.DIAGNOSTIC_RENDERER = (
    v3.v2.v1.ROOT
    / "ops/exp_scaling/"
    "plot_e70_current_diagnostics_square_v3_20260730.py"
)


if __name__ == "__main__":
    v3.v2.main()
