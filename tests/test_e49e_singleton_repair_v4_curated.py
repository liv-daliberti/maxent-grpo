from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPAIR = (
    ROOT
    / "ops/math_strategy_calibration/"
    "repair_e49e_singleton_gaps_v4_curated.py"
)
CONTRACTS = (
    ROOT
    / "ops/math_strategy_calibration/"
    "e49e_curated_singleton_contracts_toy_v4.json"
)
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e49e_curated_v4_two_row_correction_amendment_20260724.md"
)
V3_RECORDS = (
    ROOT
    / "var/artifacts/e49e_trace_bank_math_toy_repair_v3/"
    "repair_records.jsonl"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49e_singleton_repair_v4_curated_job.sh"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49e_singleton_repair_v4_test",
        REPAIR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_contracts_cover_exact_v3_failures():
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in V3_RECORDS.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert set(contracts) == {
        row["row_id"] for row in rows if row.get("pass") is not True
    }
    assert len(contracts) == 2
    assert "FROZEN BEFORE ANY V4" in AMENDMENT.read_text(encoding="utf-8")


def test_v4_contracts_are_closed_singletons():
    module = _load()
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    for payload in contracts.values():
        menu = module.impl.base.pipeline._menu_from_payload(payload)
        closed = module.impl.base.pipeline._extract_closed_candidate(
            menu,
            "S1",
        )
        assert closed is not None
        assert closed.sha256 == menu.sha256
        assert len(menu.strategies) == 1


def test_v4_carries_only_validated_v3_successes():
    source = REPAIR.read_text(encoding="utf-8")
    assert "prior V3 singleton repair changed" in source
    assert "prior_repair_record_sha256" in source
    assert "E49E_V4_CURATED_CONTRACTS" in source


def test_v4_launcher_binds_terminal_v3_and_both_contract_hashes():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "terminal V3 evidence or V4 correction set changed" in launcher
    assert "v3_contracts_sha256" in launcher
    assert "v4_contracts_sha256" in launcher
    assert "v4_audit_requests_at_freeze" in launcher
    assert "e49e_repair_singletons_v4_curated_node915.slurm" in launcher
