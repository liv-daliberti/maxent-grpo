#!/usr/bin/env python3
"""Audit the 96-update Pantry v3 query-budget repair pair."""

import audit_pantry_support_mask_paired_smoke_v1 as base

base.PREFIX = "ppsmoke_support_mask_paired_integration_v3"
base.UPDATES = 96

if __name__ == "__main__":
    base.main()
