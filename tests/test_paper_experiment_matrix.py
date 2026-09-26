"""Contracts for the final method x scale x domain x seed paper grid."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"


def load_matrix():
    sys.path.insert(0, str(EXP))
    path = EXP / "paper_matrix.py"
    spec = importlib.util.spec_from_file_location("paper_matrix_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_promised_matrix_has_750_unique_cells():
    matrix = load_matrix()
    matrix.validate_spec()
    keys = matrix.desired_keys()

    assert len(matrix.METHODS) == 10
    assert len(matrix.SCALES) == 3
    assert len(matrix.DOMAINS) == 5
    assert all(len(scale.seeds) == 5 for scale in matrix.SCALES)
    assert len(keys) == len(set(keys)) == 750


def test_method_names_match_the_paper_program():
    matrix = load_matrix()
    assert {method.key for method in matrix.METHODS} == {
        "grpo",
        "drgrpo",
        "ucpo",
        "rlep_dr",
        "replay_grpo",
        "adaptive_replay_grpo",
        "semantic_maxent",
        "adaptive_semantic_maxent",
        "replay_semantic_maxent",
        "adaptive_semantic_replay",
    }


def test_active_paper_matrix_contains_only_five_static_domains():
    matrix = load_matrix()
    domains = {domain.key for domain in matrix.DOMAINS}
    assert domains == {
        "graph_coloring", "countdown", "python_factors", "mathir",
        "pantry_plan",
    }
    assert all("maze" not in domain for domain in domains)


def test_every_source_maps_only_to_declared_methods():
    matrix = load_matrix()
    declared = {method.key for method in matrix.METHODS}
    for source in matrix.SOURCES:
        assert source.arm_methods
        assert {method for _, method in source.arm_methods} <= declared


def test_adaptive_semantic_without_replay_is_visible_as_a_real_gap():
    matrix = load_matrix()
    mapped = {
        method
        for source in matrix.SOURCES
        for _, method in source.arm_methods
    }
    assert "adaptive_semantic_maxent" not in mapped
    assert "adaptive_semantic_replay" in mapped
