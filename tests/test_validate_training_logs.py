"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for the training log validator CLI.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path


def _load_validator():
    tools_dir = (
        Path(__file__).resolve().parents[1] / "tools" / "validate_training_logs.py"
    )
    return runpy.run_path(str(tools_dir))


def test_validate_logs_passes_on_finite_metrics(tmp_path):
    mod = _load_validator()
    log_path = tmp_path / "logs" / "train_demo.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train/loss": 0.1,
        "train/learning_rate": 1e-4,
        "train/global_step": 1,
        "train/custom_metric": 2.0,
    }
    log_path.write_text(f"INFO metrics {json.dumps(payload)}\n")
    errors = mod["validate_logs"](
        [log_path], required_keys=mod["DEFAULT_REQUIRED_KEYS"]
    )
    assert errors == []


def test_validate_logs_flags_nan_and_missing_keys(tmp_path):
    mod = _load_validator()
    log_path = tmp_path / "logs" / "train_bad.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    bad_payload = {
        "train/loss": float("nan"),
        "train/learning_rate": 1e-4,
    }
    log_path.write_text(f"{json.dumps(bad_payload, allow_nan=True)}\n")
    errors = mod["validate_logs"](
        [log_path], required_keys=mod["DEFAULT_REQUIRED_KEYS"]
    )
    assert any("non-finite metric" in err for err in errors)
    assert any("missing keys" in err for err in errors)