"""Contracts for E69's sealed MATH12K route-development population."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
MATERIALIZER_PATH = (
    ROOT / "ops/route_successor/materialize_e69_math12k_route_dev.py"
)


def load_materializer():
    spec = importlib.util.spec_from_file_location(
        "materialize_e69_math12k_route_dev",
        MATERIALIZER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_largest_remainder_allocation_is_exact_and_deterministic():
    module = load_materializer()
    counts = {
        ("Algebra", "1"): 10,
        ("Algebra", "2"): 20,
        ("Geometry", "3"): 30,
    }

    quotas = module.largest_remainder_quotas(counts, 13)

    assert quotas == {
        ("Algebra", "1"): 2,
        ("Algebra", "2"): 4,
        ("Geometry", "3"): 7,
    }
    assert sum(quotas.values()) == 13
    with pytest.raises(module.MaterializationError):
        module.largest_remainder_quotas(counts, 61)


def test_selection_excludes_train_sealed_duplicates_and_long_prompts():
    module = load_materializer()
    source = []
    for index in range(12_000):
        source.append(
            {
                "problem": f"problem {index}",
                "answer": str(index),
                "subject": "Algebra" if index % 2 else "Geometry",
                "level": str(index % 5 + 1),
            }
        )
    source[384]["problem"] = source[0]["problem"]
    source[385]["problem"] = "sealed overlap"
    source[386]["problem"] = source[387]["problem"]
    sealed = [{"problem": "sealed overlap"}]
    lengths = [100] * len(source)
    lengths[388] = module.PROMPT_MAX_LENGTH + 1

    selected, audit = module.select_dev_indices(
        source,
        sealed,
        prompt_lengths=lengths,
    )

    assert len(selected) == 128
    assert min(selected) >= 384
    assert not {384, 385, 388}.intersection(selected)
    assert audit["excluded_counts"] == {
        "blank_answer": 0,
        "blank_problem": 0,
        "duplicate_normalized_problem": 1,
        "prompt_too_long": 1,
        "sealed_math500_overlap": 1,
        "train_overlap": 1,
    }
    assert sum(audit["quotas"].values()) == 128


def test_materializer_contains_no_math500_and_is_atomically_auditable(
    tmp_path: Path,
):
    module = load_materializer()
    output = tmp_path / "math12k_384_route_dev128_v1"

    manifest = module.materialize(output)

    observed = json.loads(
        (output / module.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert observed == manifest
    assert observed["output"]["train_splits"] == ["train"]
    assert observed["output"]["train_rows"] == 384
    assert observed["output"]["dev_splits"] == ["math_dev"]
    assert observed["output"]["dev_rows"] == 128
    assert observed["sealed_math500_firewall"]["normalized_problem_overlap"] == 0
    assert not (output / "eval/math").exists()
    assert module.audit_materialized(output) == observed["audit"]
    with pytest.raises(FileExistsError, match="--overwrite"):
        module.materialize(output)
