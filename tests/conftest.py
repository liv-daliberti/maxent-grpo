"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from pathlib import Path
from fnmatch import fnmatch
from types import ModuleType, SimpleNamespace
import os
import random
import sys
import importlib.util

import pytest

from .helpers.run_setup_stubs import install_training_stubs

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    # Make the src-layout packages importable without installing the project.
    sys.path.insert(0, str(_SRC_ROOT))
_PKG_ROOT = _SRC_ROOT / "maxent_grpo"
if _PKG_ROOT.exists() and str(_PKG_ROOT) not in sys.path:
    # Enable legacy imports like ``pipelines`` to resolve to the current package layout.
    sys.path.append(str(_PKG_ROOT))
# Alias legacy ``training`` imports to the current package layout.
try:
    import importlib

    training_pkg = importlib.import_module("maxent_grpo.training")
    sys.modules.setdefault("training", training_pkg)
except Exception:
    pass
try:
    from ops.sitecustomize import _install_torch_stub

    _install_torch_stub()
except Exception:
    pass
try:
    import types as _types
    import torch as _torch_mod

    try:
        from maxent_grpo.training.runtime.torch_stub import (
            _build_torch_stub as _torch_builder,
        )

        sym_cls = getattr(_torch_builder(), "SymBool", None)
    except Exception:
        sym_cls = None
    _torch_mod.SymBool = sym_cls or type(
        "SymBool", (), {"__bool__": lambda self: bool(getattr(self, "value", False))}
    )
    if not hasattr(_torch_mod, "_dynamo"):
        _torch_mod._dynamo = _types.SimpleNamespace(
            disable=lambda fn=None, recursive=False: fn, graph_break=lambda: None
        )
    else:
        dyn = _torch_mod._dynamo
        if not hasattr(dyn, "disable"):
            dyn.disable = lambda fn=None, recursive=False: fn
        if not hasattr(dyn, "graph_break"):
            dyn.graph_break = lambda: None
    if not hasattr(_torch_mod, "manual_seed"):
        _torch_mod.manual_seed = lambda *_a, **_k: None
except Exception:
    pass
try:
    import tests.test_scoring as _ts  # type: ignore

    if "__getattr__" in _ts.__dict__:
        delattr(_ts, "__getattr__")  # type: ignore[arg-type]
except Exception:
    pass

_DEFAULT_TEST_MARKER = "core"
_GROUP_MARKER_PATTERNS = {
    "training": [
        "test_run_training*.py",
        "test_run_helpers.py",
        "test_run_logging.py",
    ],
    "generation": [
        "test_run_generation*.py",
        "test_generate.py",
        "test_grpo_prompt.py",
    ],
    "setup": [],
    "logging": [
        "test_run_logging.py",
        "test_wandb_logging.py",
    ],
    "rewards": [
        "test_rewards.py",
        "test_run_training_rewards.py",
        "test_run_training_weighting.py",
    ],
    "vllm": [
        "test_run_generation_vllm.py",
        "test_vllm_patch.py",
    ],
}


if not hasattr(importlib.util, "_MAXENT_GRPO_ORIG_FIND_SPEC"):
    # Keep a stable reference to the real importlib implementation so test
    # wrappers and sitecustomize can stack without recursing into each other.
    importlib.util._MAXENT_GRPO_ORIG_FIND_SPEC = importlib.util.find_spec

_ORIG_FIND_SPEC = importlib.util.find_spec


def _patched_find_spec(name: str, package: str | None = None):
    """Shield importlib when torch stubs lack __spec__."""
    if name == "torch":
        return None
    try:
        return _ORIG_FIND_SPEC(name, package)
    except ValueError:
        if name == "torch":
            return None
        raise


importlib.util.find_spec = _patched_find_spec


def _install_transformers_stub() -> None:
    """Provide a minimal transformers stub for tests lacking the package."""
    if "transformers" in sys.modules:
        return
    tf_module = ModuleType("transformers")
    tf_module.__spec__ = SimpleNamespace()
    tf_module.__path__ = []
    trainer_utils = ModuleType("transformers.trainer_utils")
    trainer_utils.get_last_checkpoint = lambda *_args, **_kwargs: None
    utils_module = ModuleType("transformers.utils")
    utils_module.logging = SimpleNamespace(
        set_verbosity=lambda *args, **kwargs: None,
        enable_default_handler=lambda *args, **kwargs: None,
        enable_explicit_format=lambda *args, **kwargs: None,
    )

    class _AutoConfig:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return SimpleNamespace(num_attention_heads=8)

    tf_module.trainer_utils = trainer_utils
    tf_module.utils = utils_module
    tf_module.set_seed = lambda *_args, **_kwargs: None
    tf_module.PreTrainedModel = type("PreTrainedModel", (), {})
    tf_module.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    tf_module.AutoConfig = _AutoConfig
    sys.modules["transformers"] = tf_module
    sys.modules["transformers.trainer_utils"] = trainer_utils
    sys.modules["transformers.utils"] = utils_module


_install_transformers_stub()


def _install_trl_stub() -> None:
    """Provide a lightweight TRL stub so optional imports don't explode."""
    if "trl" in sys.modules:
        # Ensure nested utils module exists when a stub was already injected.
        trl_mod = sys.modules["trl"]
        if getattr(trl_mod, "__spec__", None) is None:
            trl_mod.__spec__ = SimpleNamespace()
        trainer_mod = sys.modules.get("trl.trainer")
        utils_mod = sys.modules.get("trl.trainer.utils")
        if trainer_mod is None:
            trainer_mod = ModuleType("trl.trainer")
            trainer_mod.__spec__ = SimpleNamespace()
            sys.modules["trl.trainer"] = trainer_mod
        elif getattr(trainer_mod, "__spec__", None) is None:
            trainer_mod.__spec__ = SimpleNamespace()
        if utils_mod is None:
            utils_mod = ModuleType("trl.trainer.utils")
            utils_mod.__spec__ = SimpleNamespace()
            utils_mod.prepare_deepspeed = lambda *a, **k: None
            sys.modules["trl.trainer.utils"] = utils_mod
        elif getattr(utils_mod, "__spec__", None) is None:
            utils_mod.__spec__ = SimpleNamespace()
            if not hasattr(utils_mod, "prepare_deepspeed"):
                utils_mod.prepare_deepspeed = lambda *a, **k: None
        setattr(trainer_mod, "utils", utils_mod)
        setattr(trl_mod, "trainer", trainer_mod)
        return
    trl_mod = ModuleType("trl")
    trl_mod.__spec__ = SimpleNamespace()
    trl_mod.ScriptArguments = type("ScriptArguments", (), {})
    trl_mod.GRPOConfig = type("GRPOConfig", (), {})
    trl_mod.ModelConfig = type("ModelConfig", (), {})
    trl_mod.get_kbit_device_map = lambda *a, **k: {}
    trl_mod.get_quantization_config = lambda *a, **k: None
    trainer_mod = ModuleType("trl.trainer")
    trainer_mod.__spec__ = SimpleNamespace()
    utils_mod = ModuleType("trl.trainer.utils")
    utils_mod.__spec__ = SimpleNamespace()
    utils_mod.prepare_deepspeed = lambda *a, **k: None
    trainer_mod.utils = utils_mod
    trl_mod.trainer = trainer_mod
    sys.modules["trl"] = trl_mod
    sys.modules["trl.trainer"] = trainer_mod
    sys.modules["trl.trainer.utils"] = utils_mod


_install_trl_stub()


def _install_wandb_stub() -> None:
    """Register a minimal wandb stub with a __spec__ to satisfy importlib."""
    if "wandb" in sys.modules:
        wandb_mod = sys.modules["wandb"]
        if getattr(wandb_mod, "__spec__", None) is None:
            wandb_mod.__spec__ = SimpleNamespace()
        if not hasattr(wandb_mod, "errors"):
            wandb_mod.errors = SimpleNamespace(Error=RuntimeError)
    else:
        wandb_mod = ModuleType("wandb")
        wandb_mod.__spec__ = SimpleNamespace()
        wandb_mod.errors = SimpleNamespace(Error=RuntimeError)
        sys.modules["wandb"] = wandb_mod
    wandb_errors = sys.modules.get("wandb.errors")
    if wandb_errors is None:
        wandb_errors = ModuleType("wandb.errors")
        wandb_errors.__spec__ = SimpleNamespace()
        wandb_errors.Error = RuntimeError
        sys.modules["wandb.errors"] = wandb_errors


_install_wandb_stub()


def _markers_for_path(path: Path) -> set[str]:
    filename = path.name
    matched = {
        marker
        for marker, patterns in _GROUP_MARKER_PATTERNS.items()
        if any(fnmatch(filename, pattern) for pattern in patterns)
    }
    if not matched:
        matched.add(_DEFAULT_TEST_MARKER)
    return matched


def pytest_collection_modifyitems(config, items):
    """Tag every test with unit+group markers so pytest -m selection works."""
    for item in items:
        item.add_marker("unit")
        for marker in _markers_for_path(Path(item.fspath)):
            item.add_marker(marker)
    seed = getattr(config, "_maxent_random_seed", None)
    if seed is None:
        seed_env = os.environ.get("PYTEST_RANDOM_SEED")
        try:
            seed = int(seed_env) if seed_env is not None else None
        except ValueError:
            seed = None
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        config._maxent_random_seed = seed
    random.Random(seed).shuffle(items)


@pytest.fixture
def training_stubs(monkeypatch):
    """Install lightweight stubs for torch/accelerate/transformers before imports."""
    return install_training_stubs(monkeypatch)


@pytest.fixture(autouse=True)
def _ensure_torch_stub():
    """Reinstall the torch stub before each test to avoid leaked monkeypatches."""
    try:
        from ops.sitecustomize import _install_torch_stub

        _install_torch_stub()
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _allow_stubbed_dependencies(monkeypatch):
    """Ensure dependency guards permit lightweight stubs in tests."""
    monkeypatch.setenv("ALLOW_STUBS", "1")
    yield


def pytest_report_header(config):
    seed = getattr(config, "_maxent_random_seed", None)
    if seed is None:
        return None
    return f"test order randomized with PYTEST_RANDOM_SEED={seed}"
