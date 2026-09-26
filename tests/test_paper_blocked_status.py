"""Contracts for scientific gate failures in the paper registry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"


def load_matrix():
    sys.path.insert(0, str(EXP))
    path = EXP / "paper_matrix.py"
    spec = importlib.util.spec_from_file_location("paper_matrix_blocked_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_cell(matrix, *, state="PENDING", step=0, target=3072, blocked=""):
    return matrix.Cell(
        key=matrix.CellKey("rlep_dr", "qwen05b", "graph", 43),
        source="e98",
        ledger="e98_rlep_dr_05b_jobs.json",
        arm="rlep_dr",
        job_id=30508711,
        state=state,
        step=step,
        target=target,
        blocked_because=blocked,
    )


def test_e98_source_records_the_failed_scientific_gate():
    matrix = load_matrix()
    source = matrix.E98_FAILED_FEASIBILITY

    assert "253 ineligible prompts" in source.blocked_because
    assert "30508710" in source.blocked_because
    assert "blocked" in matrix.STATUS_ORDER


def test_e98r1_is_the_live_unblocked_rlep_source():
    matrix = load_matrix()
    source = next(source for source in matrix.SOURCES if source.tag == "e98r1")

    assert source.ledger == "e98r1_sparse_rlep_dr_05b_jobs.json"
    assert source.method_for("rlep_dr_sparse") == "rlep_dr"
    assert source.blocked_because == ""


def test_frozen_gate_failure_is_blocked_not_scheduler_pending():
    matrix = load_matrix()
    cell = make_cell(matrix, blocked="frozen replay-pool gate failed")

    assert cell.status == "blocked"


def test_realized_horizon_remains_terminal_despite_old_gate_metadata():
    matrix = load_matrix()
    cell = make_cell(
        matrix,
        state="COMPLETED",
        step=3072,
        target=3072,
        blocked="historical gate metadata",
    )

    assert cell.status == "terminal"


def test_unblocked_scheduler_record_keeps_its_scheduler_status():
    matrix = load_matrix()

    assert make_cell(matrix).status == "pending"
