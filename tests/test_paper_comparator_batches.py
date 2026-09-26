"""Contracts for planning missing direct-comparator batches."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"


def load(name: str, path: Path):
    sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def current_registration_shape(matrix):
    """Reproduce the immutable E95/E97/E98 coverage without live status reads."""

    static = {domain.key for domain in matrix.DOMAINS if domain.stratum == "static"}
    present = set()
    for key in matrix.desired_keys():
        if key.method == "grpo":
            if key.scale in {"qwen05b", "falcon1b"} and key.domain in static:
                present.add(key)
            if key.scale == "qwen3b" and key.domain in static and key.seed == 70:
                present.add(key)
        if (
            key.method in {"ucpo", "rlep_dr"}
            and key.scale == "qwen05b"
            and key.domain in {"graph_coloring", "python_factors", "pantry_plan"}
        ):
            present.add(key)
    return {key: object() for key in present}


def test_empty_registry_partitions_all_225_comparator_cells_once():
    batches = load(
        "paper_comparator_batches_empty_test",
        EXP / "paper_comparator_batches.py",
    )
    plan = batches.build_plan({})
    records = [
        tuple(sorted(record.items()))
        for batch in plan["records"]
        for record in batch["records"]
    ]

    assert plan["target_cells"] == plan["missing_cells"] == 225
    assert plan["registered_cells"] == 0
    assert plan["batches"] == 9
    assert len(records) == len(set(records)) == 225
    assert all(not batch["submission_authorized"] for batch in plan["records"])


def test_current_source_coverage_becomes_140_cells_in_7_batches():
    batches = load(
        "paper_comparator_batches_shape_test",
        EXP / "paper_comparator_batches.py",
    )
    plan = batches.build_plan(current_registration_shape(batches.matrix))
    by_key = {batch["key"]: batch for batch in plan["records"]}

    assert plan["registered_cells"] == 85
    assert plan["missing_cells"] == 140
    assert plan["batches"] == 7
    assert by_key["grpo_qwen3b_static"]["cells"] == 20
    assert by_key["ucpo_qwen05b_static"]["cells"] == 10
    assert by_key["rlep_dr_qwen05b_static"]["cells"] == 10
    assert sum(
        batch["cells"]
        for batch in plan["records"]
        if batch["method"] == "rlep_dr"
    ) == 60
    assert {
        batch["protocol_state"]
        for batch in plan["records"]
        if batch["method"] == "rlep_dr"
    } == {"blocked_feasibility_amendment"}


def test_registered_cell_is_never_emitted_as_new_work():
    batches = load(
        "paper_comparator_batches_single_test",
        EXP / "paper_comparator_batches.py",
    )
    registered = batches.matrix.CellKey("grpo", "qwen05b", "graph_coloring", 43)
    plan = batches.build_plan({registered: object()})
    records = {
        (record["method"], record["scale"], record["domain"], record["seed"])
        for batch in plan["records"]
        for record in batch["records"]
    }

    assert plan["registered_cells"] == 1
    assert plan["missing_cells"] == 224
    assert (registered.method, registered.scale, registered.domain, registered.seed) not in records
