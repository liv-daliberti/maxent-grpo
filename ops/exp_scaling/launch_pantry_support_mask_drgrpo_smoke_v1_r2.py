#!/usr/bin/env python3
"""Launch the frozen Pantry r2 query-budget and audit-contract repair."""

from __future__ import annotations

import os
from pathlib import Path


os.environ["PANTRY_REPAIR_SUFFIX"] = "r2"
os.environ["PANTRY_REPAIR_ENTRYPOINT"] = str(Path(__file__).resolve())

from launch_pantry_support_mask_drgrpo_smoke_v1_r1 import main  # noqa: E402


if __name__ == "__main__":
    main()
