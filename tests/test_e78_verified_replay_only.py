from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e78_verified_replay_only_05b.py"
PROTOCOL = ROOT / "paper/preregistration/e78_verified_replay_only_05b_20260804.md"


def load_launcher():
    spec = importlib.util.spec_from_file_location("e78_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e78_is_exactly_five_domains_two_arms_five_seeds_and_eight_passes():
    e78 = load_launcher()
    assert len(e78.DOMAINS) == 5
    assert e78.ARMS == ("control", "replay")
    assert e78.SEEDS == (43, 44, 45, 46, 47)
    assert e78.PASSES == 8
    assert e78.TRAIN_ROWS == 384
    assert e78.TARGET_STEPS == 3072
    assert e78.CHECKPOINT_INTERVAL == 192


def test_e78_replay_is_the_only_applied_auxiliary_derivative():
    e78 = load_launcher()
    control = e78.fixed_objective("control")
    replay = e78.fixed_objective("replay")

    shared = {
        key: value
        for key, value in control.items()
        if key
        not in {
            "OAT_ZERO_VARIANT",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY",
        }
    }
    assert shared == {
        key: value
        for key, value in replay.items()
        if key
        not in {
            "OAT_ZERO_VARIANT",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY",
        }
    }
    assert control["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "1"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "verified_likelihood_per_rollout"
    )
    assert replay["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA"] == "0.0"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "0"
    assert replay["OAT_ZERO_MAXENT_ALPHA"] == "0.0"


def test_e78_protocol_freezes_half_pass_reporting_and_no_selection():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "Frozen before submission",
        "exactly eight passes",
        "3,072 optimizer updates",
        "every 192 updates",
        "passes 0, 0.5, 1.0, ..., 8.0",
        "5 domains x 2 arms x 5 seeds = 50 runs",
        "do not pool domains",
        "do not select a best checkpoint",
    ):
        assert literal in text

