#!/usr/bin/env python3
"""Record E111's outcome-blind checkpoint ZIP-validation runtime amendment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "paper/preregistration/e111_checkpoint_zip_validation_runtime_amendment_20260818.md"
RECORD = ROOT / "var/artifacts/e111_checkpoint_zip_validation_runtime_amendment.json"
LIVE_OPS = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops"
ROOT_HELPER = ROOT / "ops/validate_deepspeed_checkpoint.py"
ROOT_RUNNER = ROOT / "ops/run_experiment.sh"
LIVE_HELPER = LIVE_OPS / "validate_deepspeed_checkpoint.py"
LIVE_RUNNER = LIVE_OPS / "run_experiment.sh"
SELECTOR = '"$PYTHON_BIN" "$checkpoint_validator" --select-under "$SAVE_PATH"'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if RECORD.exists():
        raise SystemExit(f"refusing duplicate amendment record: {RECORD}")
    if ROOT_HELPER.read_bytes() != LIVE_HELPER.read_bytes():
        raise RuntimeError("root and live E111 checkpoint validators differ")
    for runner in (ROOT_RUNNER, LIVE_RUNNER):
        text = runner.read_text(encoding="utf-8")
        if text.count(SELECTOR) != 1:
            raise RuntimeError(f"runtime selector is absent or duplicated: {runner}")
    payload = {
        "schema": "e111_checkpoint_zip_validation_runtime_amendment_v1",
        "recorded_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": digest(PROTOCOL),
        "root_helper": str(ROOT_HELPER),
        "root_helper_sha256": digest(ROOT_HELPER),
        "root_runner": str(ROOT_RUNNER),
        "root_runner_sha256": digest(ROOT_RUNNER),
        "live_helper": str(LIVE_HELPER),
        "live_helper_sha256": digest(LIVE_HELPER),
        "live_runner": str(LIVE_RUNNER),
        "live_runner_sha256": digest(LIVE_RUNNER),
        "root_live_helper_identical": True,
        "selection_rule": "highest-step/newest-tie among complete model+optimizer ZIP checkpoints",
        "checkpoint_storage_only": True,
        "python_training_source_changed": False,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "running_jobs_signaled_or_reset": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "installed": True,
    }
    RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[e111-checkpoint-validator] installed=True record={RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
