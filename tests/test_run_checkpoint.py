"""Tests for checkpoint synchronization and barriers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace
from importlib import import_module, reload

import pytest

from test_run_setup_reference import _load_run_setup


class _DummyModel:
    def __init__(self) -> None:
        self.saved = []

    def save_pretrained(self, path: str) -> None:
        self.saved.append(path)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.saved = []

    def save_pretrained(self, path: str) -> None:
        self.saved.append(path)


class _DummyAccelerator:
    def __init__(self, is_main: bool, output_root: str) -> None:
        self.is_main_process = is_main
        self.events = []
        self._output_root = output_root

    def wait_for_everyone(self) -> None:
        self.events.append("wait")

    def save_state(self, path: str) -> None:
        # Record the path we were asked to save to.
        self.events.append(f"save_state:{os.path.relpath(path, self._output_root)}")

    def unwrap_model(self, model: _DummyModel) -> _DummyModel:
        self.events.append("unwrap")
        return model


@pytest.fixture
def checkpoint_modules(monkeypatch, tmp_path):
    _load_run_setup(monkeypatch)
    ckpt_mod = reload(import_module("maxent_helpers.run_checkpoint"))
    types_mod = reload(import_module("maxent_helpers.run_types"))
    return ckpt_mod, types_mod, tmp_path


def _checkpoint_cfg(types_mod, tmp_path):
    return types_mod.CheckpointConfig(
        output_dir=str(tmp_path),
        save_strategy="steps",
        save_steps=1,
        save_total_limit=0,
        hub=types_mod.HubPushConfig(enabled=False, model_id=None, token=None),
    )


def _checkpoint_manager(ckpt_mod, accel, model, tok, cfg):
    handles = ckpt_mod.CheckpointHandles(accel, model, tok)
    return ckpt_mod.CheckpointManager(handles, cfg)


def test_checkpoint_manager_save_applies_barriers_main_rank(checkpoint_modules):
    ckpt_mod, types_mod, tmp_path = checkpoint_modules
    accel = _DummyAccelerator(is_main=True, output_root=str(tmp_path))
    model = _DummyModel()
    tok = _DummyTokenizer()
    mgr = _checkpoint_manager(ckpt_mod, accel, model, tok, _checkpoint_cfg(types_mod, tmp_path))
    mgr.save("checkpoint-1")

    assert accel.events[:4] == [
        "wait",
        "save_state:checkpoint-1",
        "wait",
        "unwrap",
    ]
    assert model.saved == [os.path.join(str(tmp_path), "checkpoint-1")]
    assert tok.saved == [os.path.join(str(tmp_path), "checkpoint-1")]


def test_checkpoint_manager_save_applies_barriers_non_main(checkpoint_modules):
    ckpt_mod, types_mod, tmp_path = checkpoint_modules
    accel = _DummyAccelerator(is_main=False, output_root=str(tmp_path))
    model = _DummyModel()
    tok = _DummyTokenizer()
    mgr = _checkpoint_manager(ckpt_mod, accel, model, tok, _checkpoint_cfg(types_mod, tmp_path))
    mgr.save("checkpoint-2")

    assert accel.events == [
        "wait",
        "save_state:checkpoint-2",
        "wait",
        "unwrap",
    ]
    assert model.saved == []
    assert tok.saved == []


def test_checkpoint_manager_shallow_save_skips_model_and_tokenizer(monkeypatch, checkpoint_modules):
    ckpt_mod, types_mod, tmp_path = checkpoint_modules
    # Prepare a base directory with static metadata to copy from.
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    config_path = base_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    # Point output_dir at the base_dir so _copy_static_checkpoint_files can see it.
    cfg = types_mod.CheckpointConfig(
        output_dir=str(base_dir),
        save_strategy="steps",
        save_steps=1,
        save_total_limit=0,
        hub=types_mod.HubPushConfig(enabled=False, model_id=None, token=None),
    )
    monkeypatch.setenv("MAXENT_CHECKPOINT_METADATA_MODE", "shallow")
    accel = _DummyAccelerator(is_main=True, output_root=str(base_dir))
    model = _DummyModel()
    tok = _DummyTokenizer()
    mgr = _checkpoint_manager(ckpt_mod, accel, model, tok, cfg)
    mgr.save("checkpoint-shallow")

    ckpt_dir = base_dir / "checkpoint-shallow"
    # We should still apply barriers and deepspeed/accelerator state save.
    assert accel.events and accel.events[0] == "wait"
    # But we should not have invoked save_pretrained on the model/tokenizer.
    assert model.saved == []
    assert tok.saved == []
    # And the static config.json should have been copied into the checkpoint dir.
    assert (ckpt_dir / "config.json").is_file()


def test_copy_initial_model_snapshot_uses_metadata_source(monkeypatch, checkpoint_modules):
    ckpt_mod, _, tmp_path = checkpoint_modules
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    # Create a couple of representative files in the base dir.
    (base_dir / "config.json").write_text("{}", encoding="utf-8")
    (base_dir / "model.safetensors").write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("MAXENT_CHECKPOINT_METADATA_SOURCE", str(base_dir))

    ckpt_mod._copy_initial_model_snapshot(str(target_dir))

    assert (target_dir / "config.json").is_file()
    assert (target_dir / "model.safetensors").is_file()


def test_copy_initial_model_snapshot_infers_grpo_base_dir(monkeypatch, checkpoint_modules):
    ckpt_mod, _, tmp_path = checkpoint_modules
    # Simulate MaxEnt output and sibling GRPO directory under the same root.
    root = tmp_path / "data"
    root.mkdir()
    grpo_dir = root / "Qwen2.5-1.5B-Open-R1-GRPO-math-v1"
    grpo_dir.mkdir()
    maxent_dir = root / "Qwen2.5-1.5B-Open-R1-MaxEnt-GRPO-math-v1"
    maxent_dir.mkdir()
    (grpo_dir / "config.json").write_text("{}", encoding="utf-8")
    (grpo_dir / "model.safetensors").write_text("dummy", encoding="utf-8")

    # No MAXENT_CHECKPOINT_METADATA_SOURCE set; inference should kick in.
    monkeypatch.delenv("MAXENT_CHECKPOINT_METADATA_SOURCE", raising=False)

    ckpt_mod._copy_initial_model_snapshot(str(maxent_dir))

    assert (maxent_dir / "config.json").is_file()
    assert (maxent_dir / "model.safetensors").is_file()


def test_finalize_training_applies_barriers_main_rank(checkpoint_modules):
    ckpt_mod, types_mod, tmp_path = checkpoint_modules
    accel = _DummyAccelerator(is_main=True, output_root=str(tmp_path))
    model = _DummyModel()
    tok = _DummyTokenizer()
    cfg = _checkpoint_cfg(types_mod, tmp_path)
    ckpt_mod.finalize_training(accel, model, tok, cfg)

    assert accel.events[:4] == [
        "wait",
        "save_state:.",
        "wait",
        "unwrap",
    ]
    assert model.saved == [str(tmp_path)]
    assert tok.saved == [str(tmp_path)]


def test_finalize_training_applies_barriers_non_main(checkpoint_modules):
    ckpt_mod, types_mod, tmp_path = checkpoint_modules
    accel = _DummyAccelerator(is_main=False, output_root=str(tmp_path))
    model = _DummyModel()
    tok = _DummyTokenizer()
    cfg = _checkpoint_cfg(types_mod, tmp_path)
    ckpt_mod.finalize_training(accel, model, tok, cfg)

    assert accel.events == [
        "wait",
        "save_state:.",
        "wait",
        "unwrap",
    ]
    assert model.saved == []
    assert tok.saved == []


def test_skip_deepspeed_state_save_honored_with_deepspeed(monkeypatch, checkpoint_modules):
    ckpt_mod, _, _ = checkpoint_modules
    monkeypatch.setenv("MAXENT_SKIP_DEEPSPEED_STATE_SAVE", "true")
    accel = SimpleNamespace(
        is_main_process=True,
        state=SimpleNamespace(distributed_type="DEEPSPEED", deepspeed_plugin=None),
    )
    assert ckpt_mod._should_skip_accelerator_state(accel) is True


def test_force_skip_deepspeed_state_save(monkeypatch, checkpoint_modules):
    ckpt_mod, _, _ = checkpoint_modules
    monkeypatch.setenv("MAXENT_SKIP_DEEPSPEED_STATE_SAVE", "true")
    monkeypatch.setenv("MAXENT_FORCE_SKIP_DEEPSPEED_STATE_SAVE", "true")
    accel = SimpleNamespace(
        is_main_process=True,
        state=SimpleNamespace(distributed_type="DEEPSPEED", deepspeed_plugin=None),
    )
    assert ckpt_mod._should_skip_accelerator_state(accel) is True


def test_allow_deepspeed_state_save(monkeypatch, checkpoint_modules):
    ckpt_mod, _, _ = checkpoint_modules
    monkeypatch.setenv("MAXENT_SKIP_DEEPSPEED_STATE_SAVE", "true")
    monkeypatch.setenv("MAXENT_ALLOW_DEEPSPEED_STATE_SAVE", "true")
    accel = SimpleNamespace(
        is_main_process=True,
        state=SimpleNamespace(distributed_type="DEEPSPEED", deepspeed_plugin=None),
    )
    assert ckpt_mod._should_skip_accelerator_state(accel) is False


def test_save_wraps_zero_gather_params(monkeypatch, checkpoint_modules):
    ckpt_mod, types_mod, tmp_path = checkpoint_modules
    calls = []

    @contextmanager
    def _recorder(model, enabled):
        calls.append((model, enabled))
        yield

    monkeypatch.setattr(ckpt_mod, "_maybe_zero_gather_params", _recorder)
    accel = _DummyAccelerator(is_main=True, output_root=str(tmp_path))
    model = _DummyModel()
    tok = _DummyTokenizer()
    mgr = _checkpoint_manager(ckpt_mod, accel, model, tok, _checkpoint_cfg(types_mod, tmp_path))
    mgr.save("checkpoint-ctx")
    assert calls and calls[0][0] is model and calls[0][1] is True


def test_finalize_wraps_zero_gather_params(monkeypatch, checkpoint_modules):
    ckpt_mod, types_mod, tmp_path = checkpoint_modules
    calls = []

    @contextmanager
    def _recorder(model, enabled):
        calls.append((model, enabled))
        yield

    monkeypatch.setattr(ckpt_mod, "_maybe_zero_gather_params", _recorder)
    accel = _DummyAccelerator(is_main=True, output_root=str(tmp_path))
    model = _DummyModel()
    tok = _DummyTokenizer()
    cfg = _checkpoint_cfg(types_mod, tmp_path)
    ckpt_mod.finalize_training(accel, model, tok, cfg)
    assert calls and calls[0][0] is model and calls[0][1] is True


def test_save_prefers_deepspeed_checkpoint(monkeypatch, checkpoint_modules):
    ckpt_mod, types_mod, tmp_path = checkpoint_modules
    engine_calls = []

    class _DummyDSEngine:
        def save_checkpoint(self, path: str) -> None:
            engine_calls.append(path)

    engine = _DummyDSEngine()
    accel = _DummyAccelerator(is_main=True, output_root=str(tmp_path))
    accel.state = SimpleNamespace(
        distributed_type="DEEPSPEED",
        deepspeed_plugin=SimpleNamespace(zero_stage=3),
        deepspeed_engine=engine,
    )
    accel.deepspeed_engine = engine
    model = _DummyModel()
    tok = _DummyTokenizer()
    mgr = _checkpoint_manager(ckpt_mod, accel, model, tok, _checkpoint_cfg(types_mod, tmp_path))
    mgr.save("checkpoint-ds")

    assert engine_calls == [os.path.join(str(tmp_path), "checkpoint-ds")]
    assert all(not evt.startswith("save_state:") for evt in accel.events)


def test_skip_env_disables_deepspeed_checkpoint(monkeypatch, checkpoint_modules):
    ckpt_mod, types_mod, tmp_path = checkpoint_modules
    engine_calls = []

    class _DummyDSEngine:
        def save_checkpoint(self, path: str) -> None:
            engine_calls.append(path)

    engine = _DummyDSEngine()
    accel = _DummyAccelerator(is_main=True, output_root=str(tmp_path))
    accel.state = SimpleNamespace(
        distributed_type="DEEPSPEED",
        deepspeed_plugin=SimpleNamespace(zero_stage=3),
        deepspeed_engine=engine,
    )
    accel.deepspeed_engine = engine
    model = _DummyModel()
    tok = _DummyTokenizer()
    monkeypatch.setenv("MAXENT_SKIP_DEEPSPEED_STATE_SAVE", "true")
    mgr = _checkpoint_manager(ckpt_mod, accel, model, tok, _checkpoint_cfg(types_mod, tmp_path))
    mgr.save("checkpoint-ds-skip")

    assert engine_calls == []
    assert all(not evt.startswith("save_state:") for evt in accel.events)


def test_prefer_accelerate_state_save_with_deepspeed(monkeypatch, checkpoint_modules):
    ckpt_mod, types_mod, tmp_path = checkpoint_modules
    engine_calls = []

    class _DummyDSEngine:
        def save_checkpoint(self, path: str) -> None:
            engine_calls.append(path)

    engine = _DummyDSEngine()
    accel = _DummyAccelerator(is_main=True, output_root=str(tmp_path))
    accel.state = SimpleNamespace(
        distributed_type="DEEPSPEED",
        deepspeed_plugin=SimpleNamespace(zero_stage=3),
        deepspeed_engine=engine,
    )
    accel.deepspeed_engine = engine
    model = _DummyModel()
    tok = _DummyTokenizer()
    monkeypatch.setenv("MAXENT_PREFER_ACCELERATE_STATE_SAVE", "true")
    mgr = _checkpoint_manager(ckpt_mod, accel, model, tok, _checkpoint_cfg(types_mod, tmp_path))
    mgr.save("checkpoint-ds-prefer")

    assert engine_calls == []
    assert any(evt == "save_state:checkpoint-ds-prefer" for evt in accel.events)
