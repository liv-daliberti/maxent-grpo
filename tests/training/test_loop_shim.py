"""Tests for the legacy loop compatibility shim."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from maxent_grpo.training.loop import run_training_loop


def test_run_training_loop_fails_fast_with_clear_message() -> None:
    with pytest.raises(RuntimeError, match="Custom training loop removed"):
        run_training_loop(SimpleNamespace())
