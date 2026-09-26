#!/usr/bin/env python3
"""Continue timed-out E118 Qwen-3B Graph/Re:Max seed 73."""

from pathlib import Path

import repair_e118q3_graph_s72_timeout as repair


repair.OLD_JOB_ID = 31010889
repair.CHECKPOINT_STEP = "01536"
repair.REPAIR = (
    Path(__file__).resolve().parents[2]
    / "var/artifacts/e118q3_graph_s73_replay_timeout_repair_20260902.json"
)


if __name__ == "__main__":
    raise SystemExit(repair.main())
