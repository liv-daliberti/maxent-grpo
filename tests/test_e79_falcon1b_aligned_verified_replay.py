from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_e79_falcon1b_aligned_verified_replay.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/e79_falcon1b_aligned_verified_replay_20260804.md"
)


def load_launcher():
    spec = importlib.util.spec_from_file_location("e79_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e79_is_five_domains_two_arms_fresh_seeds_and_eight_passes():
    e79 = load_launcher()
    assert len(e79.DOMAINS) == 5
    assert e79.ARMS == ("control", "replay")
    assert e79.SEEDS == (55, 56, 57, 58, 59)
    assert e79.PASSES == 8
    assert e79.TRAIN_ROWS == 384
    assert e79.TARGET_STEPS == 3072
    assert e79.CHECKPOINT_INTERVAL == 192


def test_e79_optimizer_recipe_is_common_across_arms_and_explicit():
    e79 = load_launcher()
    assert e79.optimizer_env() == {
        "OAT_ZERO_LEARNING_RATE": "2e-07",
        "OAT_ZERO_LR_SCHEDULER": "constant",
        "OAT_ZERO_LR_WARMUP_RATIO": "0.0",
        "OAT_ZERO_ADAM_BETA_1": "0.9",
        "OAT_ZERO_ADAM_BETA_2": "0.999",
        "OAT_ZERO_L2": "0.0",
        "OAT_ZERO_BETA": "0.0",
        "OAT_ZERO_NUM_PPO_EPOCHS": "1",
        "OAT_ZERO_MAX_NORM": "1.0",
    }


def test_e79_replay_is_the_only_applied_auxiliary_derivative():
    e79 = load_launcher()
    control = e79.fixed_objective("control")
    replay = e79.fixed_objective("replay")
    ignored = {
        "OAT_ZERO_VARIANT",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY",
    }
    assert {k: v for k, v in control.items() if k not in ignored} == {
        k: v for k, v in replay.items() if k not in ignored
    }
    assert control["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "1"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
    assert replay["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA"] == "0.0"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "0"


def test_e79_uses_falcon_surface_and_amended_response_budgets():
    e79 = load_launcher()
    expected = {
        "graph_coloring": ("falcon_boxed", "192", "512"),
        "countdown": ("falcon_boxed", "192", "512"),
        "python_factors": ("falcon_boxed", "512", "768"),
        "mathir": ("falcon_boxed", "128", "384"),
        "pantry_plan": ("falcon_pantry_support_mask", "8", "704"),
    }
    for domain, (prompt, generate, context) in expected.items():
        qwen = "qwen_pantry_support_mask" if domain == "pantry_plan" else "qwen_boxed"
        surface = e79.task_surface(domain, qwen)
        assert surface["OAT_ZERO_PROMPT_TEMPLATE"] == prompt
        assert surface["OAT_ZERO_GENERATE_MAX_LENGTH"] == generate
        assert surface["OAT_ZERO_EVAL_GENERATE_MAX_LENGTH"] == generate
        assert surface["OAT_ZERO_MAX_MODEL_LEN"] == context


def test_e79_pairs_both_arms_on_the_same_physical_node():
    e79 = load_launcher()
    for domain in e79.DOMAINS:
        for seed in e79.SEEDS:
            node, gpu = e79.placement(domain, seed)
            assert node.startswith("node")
            assert gpu in {"a5000", "a6000"}


def test_e79_protocol_freezes_independent_recipe_and_no_selection():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "Frozen before submission",
        "without consulting any replay-arm outcome",
        "exactly eight passes",
        "passes 0, 0.5, 1.0, ..., 8.0",
        "5 domains x 2 arms x 5 seeds = 50 runs",
        "do not pool domains",
        "do not select a best checkpoint",
    ):
        assert literal in text


def test_train_entrypoint_exposes_optimizer_controls_with_safe_defaults():
    text = (ROOT / "ops/train.sh").read_text(encoding="utf-8")
    for literal in (
        '${OAT_ZERO_LR_SCHEDULER:-constant}',
        '${OAT_ZERO_LR_WARMUP_RATIO:-0.0}',
        '${OAT_ZERO_ADAM_BETA_1:-0.9}',
        '${OAT_ZERO_ADAM_BETA_2:-0.95}',
        '${OAT_ZERO_L2:-0.0}',
    ):
        assert literal in text

