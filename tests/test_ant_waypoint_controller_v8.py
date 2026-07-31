from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ops/train_ant_waypoint_controller_v7.py"
V8 = ROOT / "ops/train_ant_waypoint_controller_v8.py"
SLURM = ROOT / "ops/slurm/train_ant_waypoint_controller_v8.slurm"
PROTOCOL = ROOT / "paper/preregistration/ant_waypoint_controller_v8_20260729.md"


def test_v7_base_remains_behaviorally_frozen_under_version_parameters():
    text = BASE.read_text(encoding="utf-8")
    assert 'CONTROLLER_VERSION = "v7"' in text
    assert "DEFAULT_TIMESTEPS = 5_000_000" in text
    assert "DEFAULT_SEED = 73007" in text
    assert "DEFAULT_LEARNING_RATE = 1e-5" in text
    assert "EVALUATION_SEED_OFFSET = 2_000_000" in text
    assert "success_rate_at_least_0p90" in text
    assert "minimum_heading_success_rate_at_least_0p75" in text


def test_v8_is_only_a_fresh_seed_conservative_continuation():
    text = V8.read_text(encoding="utf-8")
    assert 'controller.CONTROLLER_VERSION = "v8"' in text
    assert "ant_waypoint_v7.zip" in text
    assert "2bb5d192698639a294ac8266ae9be68484c9d493f4f7ba4317ee06d2d4b04442" in text
    assert "controller.DEFAULT_TIMESTEPS = 3_000_000" in text
    assert "controller.DEFAULT_SEED = 73008" in text
    assert "controller.DEFAULT_LEARNING_RATE = 5e-6" in text
    assert "controller.EVALUATION_SEED_OFFSET = 3_000_000" in text
    assert "v7 evaluation episodes are not training inputs" in text


def test_v8_execution_and_protocol_freeze_matching_settings():
    slurm = SLURM.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    for value in ("--timesteps 3000000", "--seed 73008", "--learning-rate 0.000005"):
        assert value in slurm
    assert "seed base `3073008`" in protocol
    assert "overall success at least 0.90" in protocol
    assert "every heading at least 0.75" in protocol
    assert "No threshold may be relaxed" in protocol
