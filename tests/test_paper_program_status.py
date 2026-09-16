"""Contracts for the generated ten-method manuscript coverage registry."""

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


def test_program_status_covers_the_canonical_matrix_exactly_once():
    builder = load(
        "paper_program_status_under_test",
        EXP / "build_paper_program_status.py",
    )
    payload = builder.build_payload(builder.matrix.load_cells())

    assert payload["shape"] == {
        "methods": 10,
        "scales": 3,
        "domains": 5,
        "seeds_per_scale": 5,
        "cells": 750,
    }
    assert len(payload["methods"]) == 10
    assert sum(row["target"] for row in payload["methods"]) == 750
    assert payload["registered"] + payload["missing"] == 750


def test_program_status_preserves_missing_and_live_repaired_scientific_states():
    builder = load(
        "paper_program_status_boundary_test",
        EXP / "build_paper_program_status.py",
    )
    payload = builder.build_payload(builder.matrix.load_cells())
    rows = {row["key"]: row for row in payload["methods"]}

    assert rows["adaptive_semantic_maxent"]["registered"] == 0
    assert rows["adaptive_semantic_maxent"]["missing"] == 75
    assert rows["ucpo"]["registered"] == 75
    assert rows["grpo"]["terminal"] >= 69
    assert rows["grpo"]["terminal"] + rows["grpo"]["active"] == 75
    assert rows["ucpo"]["terminal"] == 50
    assert rows["ucpo"]["active"] == 25
    assert rows["rlep_dr"]["registered"] == 75
    assert rows["rlep_dr"]["blocked"] == 3
    assert rows["rlep_dr"]["terminal"] >= 43
    assert sum(
        rows["rlep_dr"][field]
        for field in (
            "terminal",
            "active",
            "blocked",
            "partial",
            "failed",
            "inactive",
        )
    ) == 75
    assert rows["replay_grpo"]["terminal"] > 0


def test_latex_body_contains_every_method_and_explicit_open_state():
    builder = load(
        "paper_program_status_tex_test",
        EXP / "build_paper_program_status.py",
    )
    payload = builder.build_payload(builder.matrix.load_cells())
    body = builder.render_tex(payload)

    assert body.count(" \\\\") == 10
    for method in builder.matrix.METHODS:
        assert method.label in body
    assert "unregistered" in body
