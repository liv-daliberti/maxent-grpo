from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e80_qwen3b_aligned_verified_replay.py"
PROTOCOL = ROOT / "paper/preregistration/e80_qwen3b_aligned_verified_replay_20260805.md"


def load_launcher():
    spec = importlib.util.spec_from_file_location("e80_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e80_is_five_domains_two_arms_fresh_seeds_and_eight_passes():
    e80 = load_launcher()
    assert len(e80.DOMAINS) == 5
    assert e80.ARMS == ("control", "replay")
    assert e80.SEEDS == (65, 66, 67, 68, 69)
    assert e80.PASSES == 8
    assert e80.TRAIN_ROWS == 384
    assert e80.TARGET_STEPS == 3072
    assert e80.CHECKPOINT_INTERVAL == 192


def test_e80_optimizer_recipe_is_common_explicit_and_scale_aware():
    e80 = load_launcher()
    assert e80.optimizer_env() == {
        "OAT_ZERO_LEARNING_RATE": "1e-07",
        "OAT_ZERO_LR_SCHEDULER": "cosine_with_min_lr",
        "OAT_ZERO_LR_WARMUP_RATIO": "0.1",
        "OAT_ZERO_ADAM_BETA_1": "0.9",
        "OAT_ZERO_ADAM_BETA_2": "0.999",
        "OAT_ZERO_L2": "0.0",
        "OAT_ZERO_BETA": "0.0",
        "OAT_ZERO_NUM_PPO_EPOCHS": "1",
        "OAT_ZERO_MAX_NORM": "1.0",
        "OAT_ZERO_TEMPERATURE": "1.0",
        "OAT_ZERO_TOP_P": "1.0",
        "OAT_ZERO_NUM_SAMPLES": "16",
    }
    assert e80.LEARNING_RATE * e80.MIN_LR_RATIO == 1e-8


def test_e80_replay_is_the_only_applied_auxiliary_derivative():
    e80 = load_launcher()
    control = e80.fixed_objective("control")
    replay = e80.fixed_objective("replay")
    ignored = {
        "OAT_ZERO_VARIANT",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY",
    }
    assert {key: value for key, value in control.items() if key not in ignored} == {
        key: value for key, value in replay.items() if key not in ignored
    }
    assert control["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "1"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
    assert replay["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA"] == "0.0"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "0"


def test_e80_uses_proven_one_a100_3b_memory_layout():
    e80 = load_launcher()
    assert e80.memory_env() == {
        "OAT_ZERO_N_GPU": "1",
        "OAT_ZERO_NUM_GPUS_PER_ACTOR": "1",
        "OAT_ZERO_ADAM_OFFLOAD": "1",
        "OAT_ZERO_ACTIVATION_OFFLOADING": "1",
        "OAT_ZERO_ZERO_STAGE": "2",
        "OAT_ZERO_VLLM_GPU_RATIO": "0.25",
        "OAT_ZERO_EVAL_BATCH_SIZE": "32",
    }
    for domain in e80.DOMAINS:
        for seed in e80.SEEDS:
            assert e80.placement(domain, seed) == ("node302", "a100")


def test_e80_model_and_native_qwen_surface_are_pinned():
    e80 = load_launcher()
    assert e80.model_root(ROOT).name == e80.MODEL_REVISION
    runs = e80.references(ROOT)
    assert len(runs) == 25
    for run in runs:
        assert run["inherited_eval_config"]["prompt_template"].startswith("qwen_")


def test_e80_protocol_freezes_no_selection_and_half_pass_grid():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "Frozen before submission",
        "exactly eight passes",
        "passes 0, 0.5, 1.0, ..., 8.0",
        "5 domains x 2 arms x 5 seeds = 50 runs",
        "do not pool domains",
        "do not select a best checkpoint",
        "peak learning rate 1e-7",
        "cosine",
    ):
        assert literal in text


def test_e80_status_is_a_single_monitor_for_current_cohorts():
    text = (ROOT / "ops/exp_scaling/status_e80.py").read_text(encoding="utf-8")
    for literal in (
        "e80_qwen3b_aligned_verified_replay_jobs.json",
        "e78_verified_replay_only_05b_jobs.json",
        "e79_falcon1b_aligned_verified_replay_jobs.json",
        "e79pm_falcon_point_maze_verified_replay_jobs.json",
    ):
        assert literal in text
