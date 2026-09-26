from __future__ import annotations

import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from oat_drgrpo.learner.init import build_maxent_controllers
from oat_drgrpo.learner.run import ZeroMathRunMixin
from oat_drgrpo.maxent_controllers import (
    MaxEntDualController,
    MaxEntInverseController,
    MaxEntProportionalController,
)
from oat_drgrpo.maxent_length_controller import MaxEntLengthController
from oat_drgrpo.on_policy_maxent import (
    standard_maxent_length_penalty_loss,
    standard_maxent_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def _args(entropy_control: str) -> SimpleNamespace:
    values = {
        "maxent_alpha": 0.05,
        "maxent_control_target_ratio": 0.0,
        "maxent_control_max_alpha": 0.5,
        "maxent_control_warmup_steps": 64,
        "maxent_control_ema_decay": 0.0,
        "maxent_control_gain": 2.0,
        "maxent_control_target_entropy": 1.0,
        "maxent_dual_target_ratio": 0.0,
        "maxent_dual_min_alpha": 0.005,
        "maxent_dual_max_alpha": 0.5,
        "maxent_dual_warmup_steps": 64,
        "maxent_dual_alpha_lr": 0.1,
        "maxent_dual_ema_decay": 0.7,
        "maxent_dual_target_entropy": 1.0,
        "maxent_inverse_adaptation": False,
        "maxent_inverse_warmup_steps": 64,
        "maxent_inverse_ema_decay": 0.9,
        "maxent_length_target": 16.0,
        "maxent_length_lambda_init": 0.001,
        "maxent_length_lambda_max": 0.02,
        "maxent_length_dual_lr": 0.002,
        "maxent_length_ema_decay": 0.0,
    }
    if entropy_control == "proportional":
        values["maxent_control_target_ratio"] = 0.8
    elif entropy_control == "dual":
        values["maxent_dual_target_ratio"] = 0.8
    elif entropy_control == "inverse":
        values["maxent_inverse_adaptation"] = True
    elif entropy_control != "fixed":
        raise ValueError(entropy_control)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("entropy_control", "expected_type"),
    [
        ("fixed", type(None)),
        ("proportional", MaxEntProportionalController),
        ("dual", MaxEntDualController),
        ("inverse", MaxEntInverseController),
    ],
)
def test_controller_factory_combines_each_alpha_rule_with_length_dual(
    entropy_control, expected_type
):
    alpha_controller, length_controller = build_maxent_controllers(
        _args(entropy_control)
    )

    assert isinstance(alpha_controller, expected_type)
    assert isinstance(length_controller, MaxEntLengthController)


def test_inverse_factory_binds_to_conditional_content_entropy_sensor():
    args = _args("inverse")
    args.maxent_objective = "conditional_token_mean"

    alpha_controller, _ = build_maxent_controllers(args)

    assert isinstance(alpha_controller, MaxEntInverseController)
    assert (
        alpha_controller.entropy_units
        == "conditional_content_token_nats_mean_v1"
    )
    assert (
        alpha_controller.observation_metric_key
        == "maxent_conditional_token_entropy"
    )


class _Harness(ZeroMathRunMixin):
    pass


class _MetricStrategy:
    def __init__(self, entropy: float, length: float):
        self.values = {
            "maxent_sequence_entropy": entropy,
            "maxent_expected_length": length,
        }
        self.calls: list[str] = []

    def all_reduce(self, values):
        key = next(iter(values))
        self.calls.append(key)
        return {key: self.values[key]}


@pytest.mark.parametrize("entropy_control", ["proportional", "dual", "inverse"])
def test_entropy_and_length_controllers_advance_independently_in_one_step(
    entropy_control,
):
    learner = _Harness()
    (
        learner._maxent_alpha_controller,
        learner._maxent_length_controller,
    ) = build_maxent_controllers(_args(entropy_control))
    learner.strategy = _MetricStrategy(entropy=0.5, length=24.0)
    alpha_before = learner._maxent_alpha_controller.current_alpha
    lambda_before = learner._maxent_length_controller.current_lambda
    train_info = {
        "maxent_sequence_entropy": 0.5,
        "maxent_expected_length": 24.0,
    }

    learner._update_maxent_alpha_controller(train_info)
    learner._update_maxent_length_controller(train_info)

    assert learner.strategy.calls == [
        "maxent_sequence_entropy",
        "maxent_expected_length",
    ]
    if entropy_control == "inverse":
        assert learner._maxent_alpha_controller.current_alpha == alpha_before
        assert "maxent_inverse_next_alpha" in train_info
    else:
        assert learner._maxent_alpha_controller.current_alpha > alpha_before
    assert learner._maxent_length_controller.current_lambda > lambda_before
    assert "maxent_length_next_alpha" not in train_info
    assert (
        "maxent_control_next_alpha" in train_info
        or "maxent_dual_next_alpha" in train_info
        or "maxent_inverse_next_alpha" in train_info
    )
    assert train_info["maxent_length_lambda_next"] == pytest.approx(0.002)


def test_joint_actor_loss_contains_entropy_reward_and_length_cost():
    raw_entropy = torch.tensor(12.0, requires_grad=True)
    raw_length = torch.tensor(20.0, requires_grad=True)
    entropy_loss = standard_maxent_loss(
        raw_entropy,
        alpha=0.05,
        reward_estimator_scale=15 / 16,
        update_normalizer=192,
    )
    length_loss = standard_maxent_length_penalty_loss(
        raw_length,
        length_lambda=0.002,
        reward_estimator_scale=15 / 16,
        update_normalizer=192,
    )
    joint_loss = entropy_loss + length_loss
    joint_loss.backward()

    assert entropy_loss.item() < 0
    assert length_loss.item() > 0
    assert joint_loss.item() == pytest.approx(
        entropy_loss.item() + length_loss.item()
    )
    assert raw_entropy.grad.item() < 0
    assert raw_length.grad.item() > 0


def _set_checkpoint_fields(learner: _Harness) -> None:
    learner.global_step = 4
    learner.policy_sgd_step = 4.0
    learner.query_step = 4
    learner.prompt_consumed = 4
    learner.prompt_epoch = 0
    learner.steps = 4
    learner._prompt_batches_consumed_total = 4
    learner.update_interval = 1
    learner._xdr_tau_controller = None
    learner._wandb_run_id = None
    learner._wandb_run_name = None


@pytest.mark.parametrize("entropy_control", ["proportional", "dual", "inverse"])
def test_checkpoint_round_trip_restores_both_controller_states(entropy_control):
    source = _Harness()
    source._maxent_alpha_controller, source._maxent_length_controller = (
        build_maxent_controllers(_args(entropy_control))
    )
    _set_checkpoint_fields(source)
    source._maxent_alpha_controller.observe(0.5)
    source._maxent_length_controller.observe(24.0)

    checkpoint = source._checkpoint_client_state()

    assert "maxent_alpha_controller_state" in checkpoint
    assert "maxent_length_controller_state" in checkpoint
    restored = _Harness()
    restored._maxent_alpha_controller, restored._maxent_length_controller = (
        build_maxent_controllers(_args(entropy_control))
    )
    restored._xdr_tau_controller = None
    restored._restore_training_progress_state(checkpoint)

    assert restored._maxent_alpha_controller.state_dict() == (
        source._maxent_alpha_controller.state_dict()
    )
    assert restored._maxent_length_controller.state_dict() == (
        source._maxent_length_controller.state_dict()
    )


@pytest.mark.parametrize(
    ("variant", "control_ratio", "dual_ratio"),
    [
        ("maxent", "0.0", "0.0"),
        ("maxent_control", "0.8", "0.0"),
        ("maxent_dual", "0.0", "0.8"),
    ],
)
def test_runtime_variants_preserve_common_length_settings(
    tmp_path, variant, control_ratio, dual_ratio
):
    fake_root = tmp_path / "root"
    (fake_root / "ops").mkdir(parents=True)
    (fake_root / "var/data/easy3/train").mkdir(parents=True)
    (fake_root / "var/data/easy3/eval").mkdir(parents=True)
    (fake_root / "var/data/easy3/train/dataset_dict.json").write_text("{}")
    (fake_root / "var/data/easy3/eval/dataset_dict.json").write_text("{}")
    (fake_root / "ops/repo_env.sh").write_text("#!/usr/bin/env bash\n")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/usr/bin/env bash\nprintf '384\\t96\\t96\\t96\\n'\n")
    fake_python.chmod(0o755)
    capture = tmp_path / "capture"
    capture.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$OAT_ZERO_MAXENT_LENGTH_TARGET\" "
        "\"$OAT_ZERO_MAXENT_LENGTH_LAMBDA_INIT\" "
        "\"$OAT_ZERO_MAXENT_LENGTH_LAMBDA_MAX\" "
        "\"$OAT_ZERO_MAXENT_LENGTH_EMA_DECAY\" "
        "\"$OAT_ZERO_MAXENT_LENGTH_DUAL_LR\" "
        "\"$OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO\" "
        "\"$OAT_ZERO_MAXENT_DUAL_TARGET_RATIO\" "
        "\"$OAT_ZERO_MAXENT_DUAL_EMA_DECAY\"\n"
    )
    capture.chmod(0o755)
    env = {
        **os.environ,
        "OAT_ZERO_REPO_ROOT": str(fake_root),
        "OAT_ZERO_PYTHON": str(fake_python),
        "OAT_ZERO_TRAIN_SCRIPT": str(capture),
        "OAT_ZERO_DATA_ROOT": str(fake_root / "var/data/easy3"),
        "OAT_ZERO_VARIANT": variant,
        "OAT_ZERO_MAXENT_ALPHA": "0.002",
        "OAT_ZERO_MAXENT_CONTROL_RATIO": "0.8",
        "OAT_ZERO_MAXENT_DUAL_RATIO": "0.8",
        "OAT_ZERO_MAXENT_LENGTH_TARGET": "16",
        "OAT_ZERO_MAXENT_LENGTH_LAMBDA_INIT": "0.001",
        "OAT_ZERO_MAXENT_LENGTH_LAMBDA_MAX": "0.02",
        "OAT_ZERO_MAXENT_LENGTH_EMA_DECAY": "0.9",
        "OAT_ZERO_MAXENT_LENGTH_DUAL_LR": "0.00005",
        "SAVE_PATH": str(tmp_path / "save"),
    }

    completed = subprocess.run(
        ["bash", str(ROOT / "ops/run_experiment.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.stdout.splitlines()[-8:] == [
        "16",
        "0.001",
        "0.02",
        "0.9",
        "0.00005",
        control_ratio,
        dual_ratio,
        "0.7",
    ]


def test_shared_submitter_forwards_length_settings_to_all_maxent_variants():
    submitter = (ROOT / "ops/submit_countdown_comparative.sh").read_text()

    condition = (
        'if [[ "$variant" == "maxent" || "$variant" == "maxent_control" || '
        '"$variant" == "maxent_dual" || "$variant" == "maxent_length_dual" ]]; then'
    )
    assert condition in submitter
    for name in (
        "TARGET",
        "LAMBDA_INIT",
        "LAMBDA_MAX",
        "EMA_DECAY",
        "DUAL_LR",
    ):
        assert f"OAT_ZERO_MAXENT_LENGTH_{name}=${{MAXENT_LENGTH_{name}}}" in submitter


def test_shared_launchers_default_and_pin_responsive_dual_entropy_ema():
    train = (ROOT / "ops/train.sh").read_text()
    runtime = (ROOT / "ops/run_experiment.sh").read_text()
    submitter = (ROOT / "ops/submit_countdown_comparative.sh").read_text()

    assert 'OAT_ZERO_MAXENT_DUAL_EMA_DECAY:-0.7' in train
    assert '--maxent-dual-ema-decay "$MAXENT_DUAL_EMA_DECAY"' in train
    assert 'OAT_ZERO_MAXENT_DUAL_EMA_DECAY:-0.7' in runtime
    assert 'OAT_ZERO_MAXENT_DUAL_EMA_DECAY:-0.7' in submitter
    assert (
        "OAT_ZERO_MAXENT_DUAL_EMA_DECAY=${MAXENT_DUAL_EMA_DECAY}"
        in submitter
    )
