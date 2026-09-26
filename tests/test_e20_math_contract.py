"""Prospective E20 data/protocol contract tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
IMPORTER_PATH = ROOT / "ops/math500/import_oat_math.py"
PROTOCOL_PATH = ROOT / "paper/preregistration/e20_math_canonical_maxent.md"


def load_importer():
    spec = importlib.util.spec_from_file_location("import_oat_math", IMPORTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e20_pins_the_initial_oat_paper_artifacts():
    importer = load_importer()

    assert importer.UPSTREAM_URL == (
        "https://github.com/sail-sg/understand-r1-zero.git"
    )
    assert importer.UPSTREAM_COMMIT == (
        "559bcfd7a50727e7ed97f06a586da2c97236f496"
    )
    assert importer.EXPECTED_FILES[
        "datasets/evaluation_suite/math/data-00000-of-00001.arrow"
    ] == "d383d13c807e2904d0db6f8d98496a0574c0a1d51f40331b920c849cf226ef5a"
    assert importer.EXPECTED_FILES[
        "datasets/train/math_lvl3to5_8k/train/data-00000-of-00001.arrow"
    ] == "bf8e6fbf72b17c88b35e2c60eec8a677ac31377f278bac033792a765ad5ec1e1"


def test_e20_protocol_keeps_math500_held_out_and_excludes_rescaling():
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")

    assert "NO TRAINING JOBS AUTHORIZED" in protocol
    assert "MATH-500 is never sampled for an optimizer update" in protocol
    assert "exact train/evaluation problem overlap: zero" in protocol
    assert "No aggregation-rescaling arm may be added" in protocol
    assert "constrained Dr.GRPO, `alpha=0`" in protocol
    assert "Standard MaxEnt with fixed `alpha`" in protocol
    assert "Standard MaxEnt with proportional entropy control" in protocol
    assert "Standard MaxEnt with Haarnoja-style dual entropy control" in protocol


def test_landed_e20_data_still_matches_the_import_manifest():
    importer = load_importer()
    root = importer.DEFAULT_OUTPUT_ROOT
    if not root.is_dir():
        pytest.skip("paper MATH artifacts have not been imported in this checkout")

    audit = importer.audit_materialized(root)
    manifest = json.loads((root / "IMPORT_MANIFEST.json").read_text(encoding="utf-8"))
    assert audit == manifest["audit"]
    assert audit == {
        "train_rows": 8523,
        "train_unique_problems": 8522,
        "train_blank_answers": 2,
        "train_duplicate_problems": 1,
        "math500_rows": 500,
        "math500_unique_problems": 500,
        "exact_problem_overlap": 0,
    }
