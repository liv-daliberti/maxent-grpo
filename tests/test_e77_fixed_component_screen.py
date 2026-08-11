"""Contracts for the E77 fixed-component necessity screen."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    path = ROOT / "ops/exp_scaling/launch_e77_fixed_component_screen.py"
    spec = importlib.util.spec_from_file_location("e77_launcher", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_monitor():
    path = ROOT / "ops/exp_scaling/status_e72.py"
    spec = importlib.util.spec_from_file_location("e72_status_for_e77", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_registered_exact_zero_factorial():
    e77 = load_launcher()
    observed = {
        arm: (
            float(config["eta"]),
            float(config["mu"]),
            float(config["alpha"]),
        )
        for arm, config in e77.ARM_CONFIG.items()
    }
    assert observed == {
        "none": (0.0, 0.0, 0.0),
        "mass_only": (0.0, 0.10, 0.0),
        "no_semantic": (0.0, 0.10, 0.10),
        "no_mass": (0.10, 0.0, 0.10),
        "no_balance": (0.10, 0.10, 0.0),
        "full_fixed": (0.10, 0.10, 0.10),
    }


def test_split_remove_one_arms_use_exact_zero_coefficients():
    e77 = load_launcher()

    no_mass = e77.arm_overrides("no_mass")
    assert no_mass["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "split_mass_balance_per_rollout"
    )
    assert float(no_mass["OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA"]) == 0.0
    assert float(no_mass["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) == 0.10

    no_balance = e77.arm_overrides("no_balance")
    assert float(no_balance["OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA"]) == 0.10
    assert float(no_balance["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) == 0.0


def test_semantic_switches_and_compute_control_are_fail_closed():
    e77 = load_launcher()
    for arm in ("none", "mass_only", "no_semantic"):
        env = e77.arm_overrides(arm)
        assert float(env["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) == 0.0
        assert env["OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE"] == "0"
        assert env[
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE"
        ] == "0"

    for arm in ("no_mass", "no_balance", "full_fixed"):
        env = e77.arm_overrides(arm)
        assert float(env["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) == 0.10
        assert env["OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE"] == "1"

    assert e77.arm_overrides("none")[
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"
    ] == "1"
    assert all(
        e77.arm_overrides(arm)[
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"
        ]
        == "0"
        for arm in e77.ARMS
    )


def test_protocol_uses_only_sealed_validation_and_one_screen_seed():
    e77 = load_launcher()
    protocol = (
        ROOT / "paper/preregistration/e77_fixed_component_screen_05b_20260804.md"
    ).read_text()
    assert e77.SEED == 58
    assert e77.PASSES == 4
    assert e77.TARGET_STEPS == 1280
    assert "reported ModeBench test sets are never passed" in protocol
    assert "One seed cannot establish" in protocol


def test_e77_is_registered_in_campaign_monitor(tmp_path):
    monitor = load_monitor()
    run_dir = tmp_path / "run"
    metrics = run_dir / "debug_job1" / "train_metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"misc/global_step": 80}) + "\n")
    ledger = tmp_path / monitor.E77_LEDGER
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "domain": "graph_coloring",
                        "run_dir": str(run_dir),
                        "target_steps": 1280,
                    }
                ]
            }
        )
    )

    progress = monitor.e77_progress(tmp_path)

    assert progress["released"] is True
    assert progress["runs_registered"] == 1
    assert progress["runs_progressing"] == 1
    assert progress["steps"] == 80
    assert progress["total_steps"] == 1280
    assert progress["domains"]["graph_coloring"]["runs_registered"] == 1
