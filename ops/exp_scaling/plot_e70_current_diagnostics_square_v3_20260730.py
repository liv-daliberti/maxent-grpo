#!/usr/bin/env python3
"""Bind diagnostics to live Ant v17 and Pantry's passing amendment."""

from __future__ import annotations

import plot_e70_current_diagnostics_square_v2_20260730 as v2


v2.diagnostics.primary.repair.REPAIR_INPUTS["pantry_audit"] = (
    v2.diagnostics.ROOT
    / "var/artifacts/"
    "pantry_support_retention_final_v1_audit_amendment_v3.json"
)


if __name__ == "__main__":
    v2.diagnostics.render()
