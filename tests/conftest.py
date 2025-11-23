"""Shared pytest fixtures for maxent helpers."""

from __future__ import annotations

from pathlib import Path
from fnmatch import fnmatch
from types import ModuleType, SimpleNamespace
import sys
import importlib.util

import pytest

from .helpers.run_setup_stubs import install_training_stubs

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    # Make the src-layout packages importable without installing the project.
    sys.path.insert(0, str(_SRC_ROOT))
try:
    from ops.sitecustomize import _install_torch_stub

    _install_torch_stub()
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
