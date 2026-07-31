#!/usr/bin/env python3
"""Bind diagnostics to Pantry amendment and the live Ant v18 repair."""

from __future__ import annotations

import plot_e70_current_diagnostics_square_v3_20260730 as v3
import plot_e70_primary_square_repairs_v5_20260730 as v5


v3.v2.diagnostics.primary = v5.primary


if __name__ == "__main__":
    v3.v2.diagnostics.render()
