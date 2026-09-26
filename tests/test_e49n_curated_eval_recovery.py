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
    "e49n_curated_distinct_routes_eval_recovery.json"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49n_curated_eval_recovery_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49n_curated_eval_recovery.sh"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49n_curated_recovery_test",
        CERTIFIER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_cohort_is_closed_and_nonleaking():
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    assert len(contracts) == 6
    assert all(row_id.startswith("eval:") for row_id in contracts)
    module = _load()
    _, _, menus = module._load_and_validate(
        contracts_path=CONTRACTS,
        source=ROOT / "var/data/e49b_math_strategy_toy",
        expected_train=0,
        expected_eval=6,
    )
    assert len(menus) == 6


def test_recovery_is_new_full_audit_not_selective_vote_reuse():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "FROZEN BEFORE ANY E49N 72B REQUEST" in protocol
    assert "four entirely\nnew audits" in protocol
    assert "four-way unanimity" in protocol
    assert "manual false-new must be exactly zero" in protocol.casefold()
    assert "E49H_EXPERIMENT=e49n" in launcher
    assert "E49H_EXPECTED_EVAL=6" in launcher
