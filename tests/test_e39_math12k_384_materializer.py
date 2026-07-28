"""Focused contracts for E39's deterministic MATH12K-384 materializer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
MATERIALIZER_PATH = ROOT / "ops/math500/materialize_e39_math12k_384.py"


def load_materializer():
    spec = importlib.util.spec_from_file_location(
        "materialize_e39_math12k_384", MATERIALIZER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e39_pins_math12k_math500_and_tokenizer_sources():
    module = load_materializer()

    assert module.TRAIN_SOURCE_ROOT == (
        ROOT
        / "var/seed_paper_eval/external/SEED-GRPO/datasets/train/math_12k"
    )
    assert module.EVAL_SOURCE_ROOT == (
        ROOT / "var/data/oat_drgrpo_math_paper/eval"
    )
    assert module.DEFAULT_OUTPUT_ROOT == ROOT / "var/data/math12k_384_math500"
    assert module.MANIFEST_NAME == "MATERIALIZATION_MANIFEST.json"
    assert module.SOURCE_CHECKOUT_COMMIT == (
        "325cb1a20bb60f8efd4cdc77a1565491c29fd289"
    )
    assert module.SOURCE_DATASET_COMMIT == (
        "ffa64bfdeaeda20029e831a749714f68079d7f9c"
    )
    assert module.TRAIN_SOURCE_FILES[
        "train/data-00000-of-00001.arrow"
    ] == "125db2efb27057f37b383d44110f3b7d49a1f55636b01f737ac5fd8cc27cf829"
    assert module.EVAL_SOURCE_FILES[
        "math/data-00000-of-00001.arrow"
    ] == "d383d13c807e2904d0db6f8d98496a0574c0a1d51f40331b920c849cf226ef5a"
    assert module.TOKENIZER_REVISION == (
        "7ae557604adf67be50417f59c2c2f167def9a775"
    )


def test_e39_source_audit_freezes_rows_order_admission_and_leakage():
    module = load_materializer()
    result = module.audit_sources()
    audit = result["audit"]

    assert audit["train_source_rows"] == 12_000
    assert audit["train_rows"] == 384
    assert audit["train_unique_normalized_problems"] == 384
    assert audit["train_blank_answers"] == 0
    assert audit["train_identity"]["ordered_row_sha256"] == (
        module.TRAIN_ORDERED_ROW_HASH
    )
    assert audit["train_identity"]["ordered_problem_sha256"] == (
        module.TRAIN_ORDERED_PROBLEM_HASH
    )
    assert audit["train_subject_counts"] == module.EXPECTED_SUBJECT_COUNTS
    assert audit["train_level_counts"] == module.EXPECTED_LEVEL_COUNTS
    assert audit["eval_split"] == "math"
    assert audit["eval_rows"] == 500
    assert audit["eval_unique_normalized_problems"] == 500
    assert audit["normalized_problem_overlap"] == 0
    assert audit["prompt_admission"]["template"] == "qwen_math"
    assert audit["prompt_admission"]["maximum_tokens"] == 900
    assert audit["prompt_admission"]["maximum_tokens"] <= 1_024
    assert audit["prompt_admission"]["ordered_token_lengths_sha256"] == (
        module.TRAIN_PROMPT_LENGTH_HASH
    )


def test_e39_materializes_exact_datasetdicts_and_atomic_manifest(tmp_path: Path):
    module = load_materializer()
    output = tmp_path / "math12k_384_math500"

    manifest = module.materialize(output)

    assert (output / "train/dataset_dict.json").is_file()
    assert (output / "eval/dataset_dict.json").is_file()
    manifest_path = output / module.MANIFEST_NAME
    assert manifest_path.is_file()
    assert not list(output.glob(f".{module.MANIFEST_NAME}.*"))
    observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert observed_manifest == manifest
    assert observed_manifest["selection"] == {
        "rule": "first_384_source_rows_in_source_order",
        "source_indices": [0, 383],
        "train_split": "train",
        "eval_split": "math",
    }
    assert observed_manifest["output"]["train_splits"] == ["train"]
    assert observed_manifest["output"]["train_rows"] == 384
    assert observed_manifest["output"]["eval_splits"] == ["math"]
    assert observed_manifest["output"]["eval_rows"] == 500
    assert module.audit_materialized(output) == observed_manifest["audit"]

    with pytest.raises(FileExistsError, match="--overwrite"):
        module.materialize(output)
