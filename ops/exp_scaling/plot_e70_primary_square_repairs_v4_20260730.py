#!/usr/bin/env python3
"""Bind the square primary surface to Pantry's passing audit amendment."""

from __future__ import annotations

import plot_e70_primary_square_repairs_v3_20260730 as v3


v3.primary.repair.REPAIR_INPUTS["pantry_audit"] = (
    v3.primary.ROOT
    / "var/artifacts/"
    "pantry_support_retention_final_v1_audit_amendment_v3.json"
)


if __name__ == "__main__":
    v3.primary.render()
