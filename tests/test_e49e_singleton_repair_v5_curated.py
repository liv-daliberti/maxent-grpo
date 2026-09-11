from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPAIR = (
    ROOT
    / "ops/math_strategy_calibration/"
    "repair_e49e_singleton_gaps_v5_curated.py"
)
CONTRACTS = (
    ROOT
    / "ops/math_strategy_calibration/"
    "e49e_curated_singleton_contracts_toy_v5.json"
)
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e49e_curated_v5_odd_index_correction_amendment_20260724.md"
)
V4_RECORDS = (
    ROOT
    / "var/artifacts/e49e_trace_bank_math_toy_repair_v4/"
    "repair_records.jsonl"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49e_singleton_repair_v5_curated_job.sh"
)
SLURM = (
    ROOT
    / "ops/slurm/"
    "e49e_repair_singletons_v5_curated_node915.slurm"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49e_singleton_repair_v5_test",
        REPAIR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_contract_covers_exact_v4_failure():
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in V4_RECORDS.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert set(contracts) == {
        row["row_id"] for row in rows if row.get("pass") is not True
    }
    assert len(contracts) == 1
    assert "FROZEN BEFORE ANY V5" in AMENDMENT.read_text(encoding="utf-8")


def test_v5_contract_is_closed_singleton():
    module = _load()
    payload = next(
        iter(json.loads(CONTRACTS.read_text(encoding="utf-8")).values())
    )
    menu = module.impl.base.pipeline._menu_from_payload(payload)
    closed = module.impl.base.pipeline._extract_closed_candidate(menu, "S1")
    assert closed is not None
    assert closed.sha256 == menu.sha256
    assert len(menu.strategies) == 1


def test_v5_carries_only_validated_v4_successes():
    source = REPAIR.read_text(encoding="utf-8")
    assert "prior V4 singleton repair changed" in source
    assert "E49E_PRIOR_REPAIR_V4_RECORDS" in source
    assert "E49E_V5_CURATED_CONTRACTS" in source


def test_v5_launcher_binds_terminal_v4_and_all_contract_hashes():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "terminal V4 evidence or V5 correction set changed" in launcher
    assert "prior_v4_repair_records_sha256" in launcher
    assert "embedded_v3_repair_records_sha256" in launcher
    assert "v3_contracts_sha256" in launcher
    assert "v4_contracts_sha256" in launcher
    assert "v5_contracts_sha256" in launcher
    assert "v5_audit_requests_at_freeze" in launcher
    assert "e49e_repair_singletons_v5_curated_node915.slurm" in launcher


def test_v5_slurm_uses_one_worker_and_rechecks_frozen_inputs():
    source = SLURM.read_text(encoding="utf-8")
    assert "--nodelist=node915" in source
    assert "--repair-workers 1" in source
    assert "frozen V5 file binding failed" in source
    assert "frozen V5 tree binding failed" in source
