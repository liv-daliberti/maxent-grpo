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
    "e49p_curated_distinct_routes_eval_reserve.json"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49p_curated_eval_reserve_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49p_curated_eval_reserve.sh"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49p_curated_reserve_test",
        CERTIFIER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reserve_is_closed_nonleaking_and_eval_only():
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


def test_reserve_preserves_full_execution_mathir_and_manual_gates():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "FROZEN BEFORE ANY E49P 72B REQUEST" in protocol
    assert "four-way unanimity" in protocol
    assert "manual false-new must be exactly zero" in protocol.casefold()
    assert "E49H_EXPERIMENT=e49p" in launcher
    assert "E49H_EXPECTED_EVAL=10" in launcher
