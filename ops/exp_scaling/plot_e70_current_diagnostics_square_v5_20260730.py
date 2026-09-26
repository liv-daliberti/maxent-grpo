#!/usr/bin/env python3
"""Bind diagnostics to the render-protected live Ant v18 state."""

from __future__ import annotations

import plot_e70_current_diagnostics_square_v3_20260730 as v3
import plot_e70_primary_square_repairs_v6_20260730 as v6


v3.v2.diagnostics.primary = v6.primary


if __name__ == "__main__":
    v3.v2.diagnostics.render()
