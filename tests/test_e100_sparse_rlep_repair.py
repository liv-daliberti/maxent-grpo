from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"


def load(name: str):
    sys.path.insert(0, str(EXP))
    path = EXP / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_e100_execution_repair_scope_is_fixed():
    repair = load("repair_e100_sparse_rlep_gates")

    assert set(repair.RECOVERED) == {
        ("graph_coloring", 58),
        ("countdown", 57),
        ("mathir", 55),
        ("mathir", 58),
    }
    assert repair.RETRY == ("python_factors", 57)
    assert set(repair.BLOCKED) == {
        ("python_factors", 56),
        ("python_factors", 59),
    }
    assert repair.SMOKE_AUDIT_ID == 30572806
    assert repair.SMOKE_ROOT.name == "e100_sparse_rlep_smoke_graph_s55"


def test_e100_blocked_runs_are_explicit_and_not_operational():
    matrix = load("paper_matrix")
    ledger = json.loads(
        (ROOT / "var/artifacts/e100_sparse_rlep_dr_falcon1b_jobs.json").read_text()
    )
    if "blocked_runs" not in ledger:
        return

    assert len(ledger["runs"]) == 23
    assert {
        (row["domain"], row["seed"]) for row in ledger["blocked_runs"]
    } == {("python_factors", 56), ("python_factors", 59)}
    assert all(row["blocked_because"] for row in ledger["blocked_runs"])

    source = next(item for item in matrix.SOURCES if item.tag == "e100")
    cells = list(matrix._source_cells(source))
    blocked = [cell for cell in cells if cell.status == "blocked"]
    assert len(cells) == 25
    assert {(cell.key.domain, cell.key.seed) for cell in blocked} == {
        ("python_factors", 56),
        ("python_factors", 59),
    }
