import importlib.util
import pathlib

from datasets import load_from_disk


ROOT = pathlib.Path(__file__).resolve().parents[1]
MATERIALIZER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "materialize_e49v_exact_oat_natural_menu.py"
)
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/"
    "launch_e49v_exact_oat_natural_menu_math_05b.sh"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49v_exact_oat_natural_menu_full_20260726.md"
)


def _module():
    spec = importlib.util.spec_from_file_location("e49v_materializer", MATERIALIZER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_e49v_overlay_is_exactly_the_certified_toy_subset():
    module = _module()
    overlays = module._toy_overlays()
    assert {split: len(rows) for split, rows in overlays.items()} == {
        "train": 50,
        "eval": 50,
    }
    for split, expected_rows in (("train", 384), ("eval", 500)):
        dataset_dict = load_from_disk(
            str(ROOT / "var/data/math12k_384_math500" / split)
        )
        dataset = dataset_dict[next(iter(dataset_dict))]
        assert len(dataset) == expected_rows
        source_keys = {
            module._source_key(split, row) for row in dataset
        }
        assert len(source_keys) == expected_rows
        assert len(source_keys & set(overlays[split])) == 50


def test_e49v_generated_menu_is_one_complete_ordered_combo():
    module = _module()
    payload = {
        "schema": "math_strategy_action_menu_v1",
        "actions": [
            {"action_id": "A1", "operation": "Form the exact equation."},
            {"action_id": "A2", "operation": "Solve it on the stated domain."},
        ],
        "strategies": [
            {
                "strategy_id": "S1",
                "action_ids": ["A1", "A2"],
                "plan": "Form and solve the exact equation.",
            }
        ],
    }
    menu = module._parse_menu(payload)
    assert len(menu.strategies) == 1
    assert menu.strategies[0].action_combo == "A1>A2"


def test_e49v_full_launcher_is_exact_current_e46_matched_contract():
    text = LAUNCHER.read_text(encoding="utf-8")
    required = [
        "OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_LEARNING_RATE=0.0000002",
        "OAT_ZERO_MAX_TRAIN=384",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=3",
        "OAT_ZERO_NUM_PROMPT_EPOCH=3",
        "OAT_ZERO_EVAL_STEPS=384",
        "OAT_ZERO_SAVE_STEPS=384",
        "OAT_ZERO_TRAIN_GRES=gpu:a100:1",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=0.50",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=0.003",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=0.90",
        "OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0",
        "OAT_ZERO_MATH_STRATEGY_ALLOW_UNSTRUCTURED_INFERENCE=1",
        "OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD=1",
        '"optimizer_updates": 1152',
    ]
    for value in required:
        assert value in text
    assert "e49t_natural_menu_math_toy_advancement_v1" in text
    assert "e49t_route_confusion_calibration_result_v1" in text
    assert "e49t_declaration_mismatch_result_v1" in text


def test_e49v_protocol_preserves_exact_oat_and_full_eval():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "exactly the 384 rows" in text
    assert "exactly all 500 MATH-500 rows" in text
    assert "exactly three" in text
    assert "current E46 normalized canonical-bank Haarnoja" in text
    assert "Policy-entropy adaptation" in text
