from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPAIR = (
    ROOT
    / "ops/math_strategy_calibration/"
    "repair_e49e_singleton_gaps_v3_curated.py"
)
CONTRACTS = (
    ROOT
    / "ops/math_strategy_calibration/"
    "e49e_curated_singleton_contracts_toy.json"
)
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e49e_curated_singleton_contingency_amendment_20260724.md"
)
PREFLIGHT = (
    ROOT
    / "var/artifacts/e49e_trace_bank_math_toy_repair_v1/preflight.json"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49e_singleton_repair_v3_curated_job.sh"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49e_singleton_repair_v3_test",
        REPAIR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_curated_manifest_covers_exact_frozen_gap_set():
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    assert set(contracts) == set(preflight["singleton_repair_rows"])
    assert len(contracts) == 11
    assert (
        "FROZEN BEFORE ANY ANSWER-BLIND V2 PROPOSAL"
        in AMENDMENT.read_text(encoding="utf-8")
    )


def test_every_curated_contract_is_closed_singleton_and_nonleaking():
    module = _load()
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    for payload in contracts.values():
        menu = module.base.pipeline._menu_from_payload(payload)
        closed = module.base.pipeline._extract_closed_candidate(menu, "S1")
        assert closed is not None
        assert closed.sha256 == menu.sha256
        assert len(menu.strategies) == 1


def test_v3_contract_provenance_cannot_masquerade_as_generated():
    source = REPAIR.read_text(encoding="utf-8")
    assert '"precommitted_curated_singleton_contract"' in source
    assert '"frozen_problem_specific_contract"' in source
    assert '"contracts_manifest_sha256"' in source
    assert "failed double-sound audit" in source
    assert "_objective_double_sound" in source
    assert "_audit_affirms_route_integrity" in source


def test_v3_launcher_requires_pre_v2_snapshot_and_terminal_v2():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "terminal V2b repair evidence is incomplete" in launcher
    assert "V3 contingency differs from the pre-V2b frozen version" in launcher
    assert "curated_audit_requests_at_freeze" in launcher
    assert "e49e_repair_singletons_v3_curated_node915.slurm" in launcher
    assert "objective_dual_audit_amendment_sha256" in launcher
