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
    "e49k_curated_distinct_routes_eval_buffer.json"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49k_curated_eval_buffer_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49k_curated_eval_buffer.sh"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49k_curated_buffer_test",
        CERTIFIER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_buffer_is_frozen_closed_and_nonleaking():
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    assert len(contracts) == 10
    assert all(row_id.startswith("eval:") for row_id in contracts)
    module = _load()
    _, _, menus = module._load_and_validate(
        contracts_path=CONTRACTS,
        source=ROOT / "var/data/e49b_math_strategy_toy",
        expected_train=0,
        expected_eval=10,
    )
    assert len(menus) == 10


def test_protocol_preserves_unanimous_mathir_and_manual_gate():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "FROZEN BEFORE ANY E49K 72B REQUEST" in text
    assert "four-way unanimity" in text
    assert "manual false-new is exactly zero" in text
    assert "at least ten train and ten evaluation" in text


def test_launcher_uses_fresh_eval_only_identity():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "E49H_EXPERIMENT=e49k" in text
    assert "E49H_EXPECTED_TRAIN=0" in text
    assert "E49H_EXPECTED_EVAL=10" in text
    assert "e49k_curated_eval_buffer_v1" in text
