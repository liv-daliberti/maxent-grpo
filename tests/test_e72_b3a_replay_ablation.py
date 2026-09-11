"""Tests for E72 B3a, the replay-gradient remove-one ablation.

The arm's whole claim is that it differs from the treatment in exactly one
place. These tests check the two ways that claim can silently break: a
coefficient drifting away from the reference run, and hardware placement
drifting away from the seed it is paired with.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


b3a = _load("e72_b3a", "ops/exp_scaling/launch_e72_b3a_replay_ablation.py")


EVAL_CONFIG = {
    "prompt_template": "qwen_boxed",
    "test_split": "multi_answer",
    "prompt_data": "/data/graph/train",
    "eval_data": "/data/graph/eval",
    "eval_input_key": "problem",
    "eval_output_key": "answer",
    "prompt_max_length": 256,
    "generate_max_length": 192,
    "eval_generate_max_length": 192,
    "max_model_len": 512,
    "eval_batch_size": 64,
    "eval_temperature": 0.0,
    "eval_mode_coverage_k": 8,
    "eval_mode_coverage_temperature": 1.0,
    "eval_mode_coverage_draws": 4,
    "eval_mode_coverage_seed": 610100,
    "canonical_action_task": "none",
    "canonical_graph_actions": False,
    "canonical_graph_action_count": 3,
    "canonical_graph_learner_sampling": False,
    "canonical_graph_fixed_shape_sampling": False,
    "verifier_version": "fast",
    "num_samples": 16,
    "rollout_batch_size": 1,
    "train_batch_size_per_device": 1,
    "max_train": 384,
    "zero_stage": 2,
    "vllm_gpu_ratio": 0.25,
    "collocate": True,
    "pretrain": "/models/base",
}

TRAIN_CONFIG = {
    "learning_rate": 2e-07,
    "num_prompt_epoch": 12,
    "num_ppo_epochs": 1,
    "max_queries": 100000000,
    "max_norm": 1.0,
    "beta": 0.0,
    "temperature": 1.0,
    "top_p": 1.0,
    "train_batch_size": 16,
    "pi_buffer_maxlen_per_device": 16,
    "sync_params_every": 1,
    "eval_steps": 96,
    "save_steps": 384,
    "save_from": 384,
    "resume_steps": 384,
    "export_steps": 0,
    "seed": 43,
}


def _run(**overrides):
    run = {
        "domain": "graph_coloring",
        "arm": "xgrpo",
        "curve_arm_label": "verified_first_global_replay_canonical",
        "seed": 43,
        "run_stamp": (
            "gce71_scale384_05b_12pass_verified_first_global_replay_canonical_s43"
        ),
        "run_dir": "/runs/xgrpo_s43",
        "source_node": "node105",
        "inherited_eval_config": dict(EVAL_CONFIG),
        "inherited_train_config": dict(TRAIN_CONFIG),
        "inherited_objective_config": dict(b3a.EXPECTED_OBJECTIVE),
    }
    for key, value in overrides.items():
        if key in ("objective", "training", "evaluation"):
            target = {
                "objective": "inherited_objective_config",
                "training": "inherited_train_config",
                "evaluation": "inherited_eval_config",
            }[key]
            run[target].update(value)
        else:
            run[key] = value
    return run


def test_a_matching_reference_run_passes_inheritance():
    assert b3a.check_inheritance(_run()) == []


def test_a_drifted_dose_blocks_the_launch():
    # If the reference run was not trained at the frozen dose, this arm is not
    # a remove-one ablation of it.
    problems = b3a.check_inheritance(_run(objective={"semantic_shannon_coef": 0.2}))
    assert any("semantic_shannon_coef" in problem for problem in problems)

    problems = b3a.check_inheritance(_run(objective={"online_canonical_novelty_beta": 0.0}))
    assert any("online_canonical_novelty_beta" in problem for problem in problems)

    problems = b3a.check_inheritance(_run(training={"learning_rate": 1e-06}))
    assert any("learning_rate" in problem for problem in problems)


def test_reference_must_have_a_live_replay_gradient():
    # Ablating replay against a reference that already had it disabled would
    # compare the arm to itself.
    problems = b3a.check_inheritance(
        _run(objective={"online_canonical_replay_compute_only": True})
    )
    assert any("replay_compute_only" in problem for problem in problems)


def test_seed_and_placement_must_be_recorded():
    assert any(
        "training seed" in problem
        for problem in b3a.check_inheritance(_run(training={"seed": 44}))
    )
    assert any(
        "source node" in problem
        for problem in b3a.check_inheritance(_run(source_node=None))
    )


def test_run_identity_is_distinct_from_the_reference_cohort(tmp_path):
    run = _run()
    stamp = b3a.run_stamp(run)
    assert stamp == "gce71_scale384_05b_12pass_b3a_s43"
    # The ablation must not write into the reference run's directory.
    assert b3a.VARIANT in str(b3a.save_path(tmp_path, run))
    assert stamp != run["run_stamp"]


def test_environment_trains_from_initialization_with_the_inherited_objective(tmp_path):
    env = b3a.build_export_vars(ROOT, _run(), tmp_path)

    # The ablation trains from the base model, not from a terminal checkpoint.
    assert env["OAT_ZERO_PRETRAIN"] == "/models/base"
    assert env["OAT_ZERO_VARIANT"] == b3a.VARIANT
    assert "OAT_ZERO_EVAL_ONLY" not in env

    # Discovery credit is inherited unchanged from the reference arm.
    assert float(env["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) == 0.10
    assert float(env["OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA"]) == 0.50
    assert float(env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) == 0.10
    assert float(env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA"]) == 0.10

    # Schedule and optimizer match the cohort exactly.
    assert float(env["OAT_ZERO_LEARNING_RATE"]) == 2e-07
    assert env["OAT_ZERO_NUM_PROMPT_EPOCH"] == "12"
    assert env["OAT_ZERO_SEED"] == "43"
    assert env["OAT_ZERO_EVAL_PROMPT_INTERVAL"] == "96"
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_SEED"] == "610100"


def test_canonical_action_domains_keep_the_audited_engine(tmp_path):
    env = b3a.build_export_vars(
        ROOT,
        _run(
            evaluation={
                "canonical_action_task": "pantry_support_mask",
                "canonical_graph_learner_sampling": True,
                "canonical_graph_action_count": 6,
            }
        ),
        tmp_path,
    )
    assert env["VLLM_USE_V1"] == "0"
    assert env["OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT"] == "6"


def test_b1a_removes_discovery_credit_and_keeps_replay(tmp_path):
    """B1a is the complement of B3a: replay acts, discovery credit does not."""
    env = b3a.build_export_vars(ROOT, _run(), tmp_path, "b1a")

    # Discovery channels off.
    assert float(env["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) == 0.0
    assert float(env["OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA"]) == 0.0

    # Replay untouched: the mass and balance losses, their controllers, and the
    # global scheduler carry the treatment's own doses.
    assert env["OAT_ZERO_VARIANT"] == "verified_first_replay_only_ablation"
    assert float(env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) == 0.10
    assert float(env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA"]) == 0.10


def test_arms_are_complementary_and_write_to_separate_run_directories(tmp_path):
    run = _run()
    b3a_env = b3a.build_export_vars(ROOT, run, tmp_path, "b3a")
    b1a_env = b3a.build_export_vars(ROOT, run, tmp_path, "b1a")

    # B3a keeps discovery credit; B1a keeps replay. Neither keeps both.
    assert float(b3a_env["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) == 0.10
    assert float(b1a_env["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) == 0.0
    assert b3a_env["OAT_ZERO_VARIANT"] != b1a_env["OAT_ZERO_VARIANT"]

    assert b3a.run_stamp(run, "b3a") != b3a.run_stamp(run, "b1a")
    assert b3a.save_path(tmp_path, run, "b3a") != b3a.save_path(tmp_path, run, "b1a")


def test_b1a_disables_the_switches_that_require_a_positive_coefficient(tmp_path):
    """A zero coefficient with these switches on fails argument validation.

    The treatment's variant hardcodes all three to on, so B1a must use its own
    variant and must not leave any of them enabled; the first B1a launch died
    at startup on exactly this and was requeued by the watchdog.
    """
    env = b3a.build_export_vars(ROOT, _run(), tmp_path, "b1a")
    assert env["OAT_ZERO_VARIANT"] == "verified_first_replay_only_ablation"
    assert env["OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE"] == "0"
    assert env["OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE"] == "0"
    assert env["OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION"] == "0"
    assert float(env["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) == 0.0


def test_b1b_is_rehearsal_only_and_distinct_from_b1a(tmp_path):
    """B1b removes balancing; B1a keeps it. Confusing them would answer the
    reviewer's question with an arm that still balances."""
    b1b = b3a.build_export_vars(ROOT, _run(), tmp_path, "b1b")
    b1a = b3a.build_export_vars(ROOT, _run(), tmp_path, "b1a")

    # Mass-only objective: the learner never forms the balance loss under it.
    assert b1b["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == "verified_likelihood_per_rollout"
    assert b1b["OAT_ZERO_VARIANT"] == "verified_first_replay_rehearsal_only"

    # No rarity weighting either.
    assert float(b1b["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) == 0.0
    assert float(b1b["OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA"]) == 0.0

    # The two arms must not collapse onto one another.
    assert b1b["OAT_ZERO_VARIANT"] != b1a["OAT_ZERO_VARIANT"]
    assert b3a.run_stamp(_run(), "b1b") != b3a.run_stamp(_run(), "b1a")

    # Same replay budget as the treatment: coefficient unchanged at .10.
    assert float(b1b["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) == 0.10
