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

Tests for the training log validator.
"""

from __future__ import annotations

from pathlib import Path

from tools.validate_logs import run_validation, validate_file


def test_validate_file_passes_clean_metrics(tmp_path: Path):
    log = tmp_path / "train_good.log"
    log.write_text(
        "step 1 train/loss=0.5 train/learning_rate=1e-4 train/kl_per_completion_token: 0.01\n",
        encoding="utf-8",
    )
    errors, hits = validate_file(
        log, require_keys=["train/loss", "train/learning_rate"]
    )
    assert not errors
    assert hits["train/loss"] == 1
    assert hits["train/learning_rate"] == 1


def test_validate_file_flags_nan(tmp_path: Path):
    log = tmp_path / "train_bad.log"
    log.write_text("train/loss: nan\n", encoding="utf-8")
    errors, _hits = validate_file(log, require_keys=[])
    assert errors
    assert errors[0].key == "train/loss"
    assert "NaN" in errors[0].value or errors[0].message


def test_run_validation_handles_multiple_files(tmp_path: Path):
    good = tmp_path / "good.log"
    bad = tmp_path / "bad.log"
    good.write_text("train/loss=0.1 train/learning_rate=0.01\n", encoding="utf-8")
    bad.write_text("train/learning_rate=0.02 train/loss=inf\n", encoding="utf-8")
    errors = run_validation([good, bad], require_keys=["train/loss"])
    assert any(err.path == bad for err in errors)
    assert all(err.path != good for err in errors)
