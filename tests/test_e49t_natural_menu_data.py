from __future__ import annotations

import importlib.util
import pathlib

from datasets import load_from_disk

from oat_drgrpo.math_strategy_menu import parse_strategy_menu


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/materialize_e49t_natural_menu_data.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("materialize_e49t", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e49t_materialization_preserves_menus_and_removes_xml_contract(
    tmp_path,
):
    module = _load()
    output = tmp_path / "e49t"
    manifest = module.materialize(output)
    assert manifest["row_counts"] == {"train": 50, "eval": 50}
    assert manifest["multi_support_counts"] == {"train": 10, "eval": 10}

    for split, name in (("train", "train"), ("eval", "math")):
        dataset = load_from_disk(str(output / split))[name]
        for row in dataset:
            problem = row["problem"]
            menu = parse_strategy_menu(problem)
            assert menu is not None
            assert menu.sha256 == row["strategy_menu_sha256"]
            assert "<action_trace>" not in problem
            assert "<action_step" not in problem
            assert "MUST begin at its first character" not in problem
            assert "declaration by itself proves nothing" in problem


def test_e49t_launcher_is_matched_three_epoch_e46_and_calibration_gated():
    source = (
        ROOT
        / "ops/exp_scaling/launch_e49t_natural_menu_math_toy_05b.sh"
    ).read_text(encoding="utf-8")
    assert "OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja" in source
    assert "OAT_ZERO_MAX_PROMPT_EPOCHS=3" in source
    assert "OAT_ZERO_NUM_PROMPT_EPOCH=3" in source
    assert "OAT_ZERO_NUM_SAMPLES=16" in source
    assert "OAT_ZERO_TRAIN_SEEDS=45" in source
    assert "OAT_ZERO_TRAIN_GRES=gpu:a100:1" in source
    assert "OAT_ZERO_MATH_STRATEGY_ALLOW_UNSTRUCTURED_INFERENCE=1" in source
    assert "OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD=1" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0" in source
    assert "calibration.get(\"pass\") is not True" in source
    assert "endpoint_record_sha256" in source
