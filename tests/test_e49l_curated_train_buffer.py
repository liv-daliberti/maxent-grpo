from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CERTIFIER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "certify_e49h_curated_routes.py"
)
CONTRACTS = (
    ROOT
    / "ops/math_strategy_calibration/"
    "e49l_curated_distinct_routes_train_buffer.json"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49l_curated_train_buffer_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49l_curated_train_buffer.sh"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49l_curated_buffer_test",
        CERTIFIER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_buffer_is_frozen_closed_and_nonleaking():
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    assert len(contracts) == 5
    assert all(row_id.startswith("train:") for row_id in contracts)
    module = _load()
    _, _, menus = module._load_and_validate(
        contracts_path=CONTRACTS,
        source=ROOT / "var/data/e49b_math_strategy_toy",
        expected_train=5,
        expected_eval=0,
    )
    assert len(menus) == 5


def test_protocol_and_launcher_preserve_strict_gates():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "FROZEN BEFORE ANY E49L 72B REQUEST" in protocol
    assert "four-way unanimity" in protocol
    assert "manual false-new\nis exactly zero" in protocol
    assert "E49H_EXPERIMENT=e49l" in launcher
    assert "E49H_EXPECTED_TRAIN=5" in launcher
    assert "E49H_EXPECTED_EVAL=0" in launcher
