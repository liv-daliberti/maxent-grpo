#!/usr/bin/env python3
"""Run the unchanged paired Pantry audit against the prospective r1 paths."""

import audit_pantry_support_mask_paired_smoke_v1 as base

base.PREFIX = "ppsmoke_support_mask_paired_v1_r1"

if __name__ == "__main__":
    base.main()
