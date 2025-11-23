"""Tests covering the MaxEnt setup helpers that prepare training inputs."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Tuple

import pytest

from test_run_setup_reference import _load_run_setup

class _FakeLoader:
    """Minimal loader exposing __len__ for step count inference."""

    def __init__(self, length: int):
        self._length = length

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._length


@pytest.fixture
def run_setup(monkeypatch):
    """Load run_setup with torch/accelerate stubs applied."""
    return _load_run_setup(monkeypatch)


@pytest.fixture
def bundle_classes(run_setup) -> Tuple[type, type, type]:
    """Return TrainDataBundle, PromptIOConfig, and PromptCacheEntry."""
    from maxent_helpers.run_types import PromptIOConfig, TrainDataBundle
    from maxent_helpers.run_training_types import PromptCacheEntry

    return TrainDataBundle, PromptIOConfig, PromptCacheEntry


def _build_train_data(
    bundle_cls,
    prompt_io_cls,
    cache_entry,
    loader: _FakeLoader,
    steps_per_epoch: int,
):
    prompt_io = prompt_io_cls(
        prompt_column="prompt",
        solution_column="answer",
        prompt_length_cache_get=lambda _prompt: cache_entry,
    )
    return bundle_cls(
        train_dataset=object(),
        train_loader=loader,
        train_sampler=None,
        prompt_io=prompt_io,
        steps_per_epoch=steps_per_epoch,
        batch_size=4,
    )


def test_prepare_with_accelerator_updates_sharded_step_count(run_setup, bundle_classes):
    """steps_per_epoch should track the prepared (per-rank) dataloader."""

    TrainDataBundle, PromptIOConfig, PromptCacheEntry = bundle_classes
    bundle = SimpleNamespace(model="initial_model")
    optimizer = SimpleNamespace()
    original_loader = _FakeLoader(length=128)
    prepared_loader = _FakeLoader(length=32)
    cache_entry = PromptCacheEntry(input_ids=[0], attention_mask=[1])
    train_data = _build_train_data(
        TrainDataBundle,
        PromptIOConfig,
        cache_entry,
        original_loader,
        steps_per_epoch=128,
    )
    prepared_model = SimpleNamespace(name="prepared")

    class _Accel:
        def prepare(self, model, opt, loader):
            assert loader is original_loader
            return prepared_model, opt, prepared_loader

    accelerator = _Accel()
    updated_bundle, updated_opt, updated_data = run_setup.prepare_with_accelerator(
        accelerator, bundle, optimizer, train_data
    )

    assert updated_bundle.model is prepared_model
    assert updated_opt is optimizer
    assert updated_data.train_loader is prepared_loader
    assert updated_data.steps_per_epoch == len(prepared_loader)


def _default_args(**overrides):
    base = {
        "per_device_train_batch_size": 1,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
        "max_grad_norm": 1.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_resolve_training_hyperparams_validates_inputs(run_setup):
    with pytest.raises(ValueError):
        run_setup.resolve_training_hyperparams(
            _default_args(per_device_train_batch_size=0)
        )
    with pytest.raises(ValueError):
        run_setup.resolve_training_hyperparams(_default_args(num_train_epochs=0))
    with pytest.raises(ValueError):
        run_setup.resolve_training_hyperparams(
            _default_args(gradient_accumulation_steps=0)
        )
    with pytest.raises(ValueError):
        run_setup.resolve_training_hyperparams(_default_args(learning_rate=0.0))
    with pytest.raises(ValueError):
        run_setup.resolve_training_hyperparams(_default_args(max_grad_norm=-1.0))


def test_resolve_training_hyperparams_returns_expected_values(run_setup):
    args = _default_args(
        per_device_train_batch_size=8,
        num_train_epochs=3,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        max_grad_norm=0.5,
    )
    params = run_setup.resolve_training_hyperparams(args)
    assert params.batch_size == 8
    assert params.num_epochs == 3
    assert params.grad_accum_steps == 2
    assert params.learning_rate == pytest.approx(2e-5)
    assert params.max_grad_norm == pytest.approx(0.5)


def _default_weight_args(**overrides):
    base = {
        "init_kl_coeff": 3.0,
        "beta": 0.04,
        "maxent_tau": 0.0,
        "maxent_q_temperature": 1.0,
        "maxent_q_epsilon": 1e-6,
        "maxent_length_normalize_ref": True,
        "kl_target": 0.07,
        "kl_horizon": 100,
        "kl_ctl_step_size": 0.15,
        "maxent_tau_lr": 0.0,
        "maxent_tau_min": 0.0,
        "maxent_tau_max": 0.0,
        "maxent_tau_warmup_steps": -1,
        "warmup_steps": 0,
        "maxent_target_weight_entropy": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_weighting_settings_use_init_beta_and_tau(run_setup):
    args = _default_weight_args()
    weighting = run_setup._resolve_weighting_settings(args)
    assert weighting.beta == pytest.approx(3.0)
    assert weighting.tau == pytest.approx(0.0)
    assert weighting.denom == pytest.approx(3.0)


def test_weighting_settings_respect_tau_override(run_setup):
    args = _default_weight_args(maxent_tau=0.25)
    weighting = run_setup._resolve_weighting_settings(args)
    assert weighting.tau == pytest.approx(0.25)
    assert weighting.denom == pytest.approx(weighting.beta + 0.25)


def test_weighting_settings_accepts_single_f_init_kl_coef(run_setup):
    """Fallback to init_kl_coef to match TRL's parameter name."""

    args = _default_weight_args(init_kl_coeff=None, init_kl_coef=1.75)
    weighting = run_setup._resolve_weighting_settings(args)
    assert weighting.beta == pytest.approx(1.75)


def test_weighting_settings_fallback_to_beta_attr(run_setup):
    """Use training_args.beta when init_kl_* aliases are absent."""

    args = _default_weight_args(
        init_kl_coeff=None,
        init_kl_coef=None,
        beta=2.5,
    )
    weighting = run_setup._resolve_weighting_settings(args)
    assert weighting.beta == pytest.approx(2.5)


def test_weighting_settings_propagate_kl_controller_fields(run_setup):
    args = _default_weight_args(
        kl_target=0.07,
        kl_horizon=50000,
        kl_ctl_step_size=0.15,
    )
    weighting = run_setup._resolve_weighting_settings(args)
    assert weighting.kl_target == pytest.approx(0.07)
    assert weighting.kl_horizon == 50000
    assert weighting.kl_ctl_step_size == pytest.approx(0.15)


def test_build_checkpoint_config_handles_enum_like_strategy(run_setup, tmp_path):
    """Enum-like save_strategy inputs should normalize to plain strings."""

    class _EnumLike:
        def __init__(self, value: str):
            self.value = value

        def __str__(self) -> str:  # pragma: no cover - string fallback guard
            return f"SaveStrategy.{self.value}"

    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        save_strategy=_EnumLike("steps"),
        save_steps=5,
        save_total_limit=2,
        push_to_hub=False,
        hub_model_id=None,
        hub_token=None,
    )
    cfg = run_setup.build_checkpoint_config(args)
    assert cfg.save_strategy == "steps"
    assert cfg.save_steps == 5
    assert cfg.output_dir.endswith("out")


def test_q_distribution_neutralized_for_grpo(run_setup):
    """maxent q knobs should be forced to neutral values for the GRPO objective."""

    args = SimpleNamespace(
        maxent_q_temperature=0.3,
        maxent_q_epsilon=0.42,
        train_grpo_objective=True,
    )
    q_dist = run_setup._resolve_q_distribution(args)
    assert q_dist.temperature == 1.0
    assert q_dist.epsilon == pytest.approx(1e-8)


def test_weighting_denom_uses_beta_for_grpo(run_setup):
    """GRPO objective should keep the beta ref term (no tau)."""

    args = SimpleNamespace(
        init_kl_coeff=3.0,
        maxent_tau=0.5,
        train_grpo_objective=True,
        maxent_q_temperature=1.0,
        maxent_q_epsilon=0.0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        maxent_tau_lr=0.0,
        maxent_tau_min=0.0,
        maxent_tau_max=0.0,
        maxent_tau_warmup_steps=-1,
        warmup_steps=0,
        maxent_target_weight_entropy=None,
        maxent_length_normalize_ref=True,
    )
    weighting = run_setup._resolve_weighting_settings(args)
    assert weighting.beta == pytest.approx(3.0)
    assert weighting.denom == pytest.approx(3.0)


def test_trl_prepare_deepspeed_receives_model_first(run_setup, monkeypatch):
    """The TRL hook should receive (model, accelerator, plugin) in that order."""

    calls = []

    def _fake_prepare(model, accelerator, plugin):
        calls.append((model, accelerator, plugin))

    fake_utils = SimpleNamespace(prepare_deepspeed=_fake_prepare)
    fake_trainer = SimpleNamespace(utils=fake_utils)
    fake_trl = SimpleNamespace(trainer=fake_trainer)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setitem(sys.modules, "trl.trainer", fake_trainer)
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", fake_utils)

    plugin = object()
    accelerator = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=plugin))
    model = object()

    run_setup._maybe_prepare_with_trl(accelerator, model)

    assert calls == [(model, accelerator, plugin)]


def test_prepare_training_data_respects_dataloader_settings(run_setup, monkeypatch):
    recorded = {}

    class _RecordingLoader:
        def __init__(self, dataset, **kwargs):
            recorded["dataset"] = dataset
            recorded["kwargs"] = kwargs
            self._steps = 4

        def __len__(self):
            return self._steps

    dataset_obj = object()
    monkeypatch.setattr(
        run_setup,
        "_map_training_dataset",
        lambda _config, _columns: dataset_obj,
    )

    class _Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": [0], "attention_mask": [1]}

    script_args = SimpleNamespace(
        dataset_prompt_column="prompt",
        dataset_solution_column="answer",
        dataset_train_split="train",
    )
    training_args = SimpleNamespace(system_prompt=None)
    context = run_setup.DatasetContext(script_args, training_args, _Tokenizer())
    runtime = run_setup.DataLoaderRuntimeOptions(
        num_workers=3,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
    )
    config = run_setup.DataPrepConfig(
        context=context,
        batch_size=2,
        max_prompt_len=16,
        data_loader_cls=_RecordingLoader,
        runtime=runtime,
    )

    bundle = run_setup.prepare_training_data(config)
    assert bundle.train_loader is not None
    assert bundle.train_sampler is None
    assert recorded["dataset"] is dataset_obj
    kwargs = recorded["kwargs"]
    assert kwargs["num_workers"] == 3
    assert kwargs["pin_memory"] is False
    assert kwargs["drop_last"] is True
    assert kwargs["persistent_workers"] is True


def test_prompt_cache_respects_large_prompt_limit(run_setup):
    tokenizer_calls = {}

    class _Tokenizer:
        def __call__(self, text, **_kwargs):
            tokenizer_calls["text"] = text
            return {"input_ids": [0] * len(text), "attention_mask": [1] * len(text)}

    cache_get = run_setup._build_prompt_length_cache(_Tokenizer(), max_prompt_len=4000)
    long_prompt = "x" * 3000
    entry = cache_get(long_prompt)
    assert len(tokenizer_calls["text"]) == len(long_prompt)
    assert len(entry.input_ids) == len(long_prompt)
