#!/usr/bin/env python3
"""Apply the strengthened paired Pantry audit to the 96-update v2 bridge run."""

import audit_pantry_support_mask_paired_smoke_v1 as base

base.PREFIX = "ppsmoke_support_mask_paired_integration_v2"
base.UPDATES = 96

if __name__ == "__main__":
    base.main()
