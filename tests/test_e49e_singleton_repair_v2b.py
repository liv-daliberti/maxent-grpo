from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPAIR = (
    ROOT
    / "ops/math_strategy_calibration/repair_e49e_singleton_gaps_v2b.py"
)
ABORT = (
    ROOT
    / "paper/preregistration/"
    "e49e_v2_node302_inflight_abort_amendment_20260724.md"
)
RELAUNCH = (
    ROOT
    / "paper/preregistration/"
    "e49e_answer_blind_v2b_node915_relaunch_amendment_20260724.md"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49e_singleton_repair_v2b_test",
        REPAIR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2b_has_fresh_version_and_disjoint_fixed_seeds():
    module = _load()
    assert module.REPAIR_VERSION.endswith("_v2b")
    assert module.PROPOSAL_SEEDS == (492251, 492252, 492253, 492254)
    assert set(module.PROPOSAL_SEEDS).isdisjoint(module.v2.PROPOSAL_SEEDS)
    assert module.PROPOSAL_ROLES == module.v2.PROPOSAL_ROLES


def test_v2_is_explicitly_abandoned_and_v2b_is_prefrozen():
    abort = ABORT.read_text(encoding="utf-8")
    relaunch = RELAUNCH.read_text(encoding="utf-8")
    assert "TERMINAL ABANDONMENT" in abort
    assert "must never be resumed or resampled" in abort
    assert "FROZEN BEFORE ANY V2B" in relaunch
    assert "node915" in relaunch
