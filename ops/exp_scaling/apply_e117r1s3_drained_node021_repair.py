#!/usr/bin/env python3
"""Install the E117-R1-S3 zero-step node021 thermal-drain repair."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_e117r1s2_drained_node_repair as repair  # noqa: E402


repair.PROTOCOL = "paper/preregistration/e117r1s3_drained_node021_repair_20260825.md"
repair.ARTIFACT = "var/artifacts/e117r1s3_drained_node021_repair.json"
repair.APPLICATION = Path(__file__).resolve()
repair.AMENDMENT_NAME = "E117-R1-S3"
repair.SCHEMA = "e117r1s3_drained_node021_repair_v1"
repair.SOURCE_NODE = "node021"
repair.TARGET_NODES = {
    "countdown": "node103",
    "graph_coloring": "node104",
}
repair.EXPECTED_JOB_IDS = tuple(range(30873695, 30873701))
repair.LOG_LABEL = "e117r1s3"


if __name__ == "__main__":
    raise SystemExit(repair.main())
