"""Test/bootstrap helper to expose the src/ layout as a top-level package."""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
import os
from types import ModuleType, SimpleNamespace

_ROOT_DIR = Path(__file__).resolve().parent.parent
_VAR_ROOT = _ROOT_DIR / "var"
_VAR_ROOT.mkdir(parents=True, exist_ok=True)
_PYCACHE_DIR = _VAR_ROOT / "pycache"
_PYCACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(_PYCACHE_DIR))

_SRC_ROOT = _ROOT_DIR / "src"
if _SRC_ROOT.exists():
    src_str = str(_SRC_ROOT)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_ORIG_FIND_SPEC = importlib.util.find_spec


def _maxent_find_spec(name: str, package: str | None = None):
    """Treat lightweight stubs (missing files) as absent for optional deps."""
    if name == "torch":
        module = sys.modules.get("torch")
        if isinstance(module, ModuleType) and getattr(module, "__file__", None) is None:
            return None
        if isinstance(module, ModuleType) and getattr(module, "__spec__", None) is None:
            return None
    try:
        return _ORIG_FIND_SPEC(name, package)
    except ValueError:
        if name == "torch":
            return None
        raise


importlib.util.find_spec = _maxent_find_spec


def _install_transformers_stub() -> None:
    """Register a lightweight transformers stub for tests when missing."""
    if "transformers" in sys.modules:
        return
    if importlib.util.find_spec("transformers") is not None:
        return
    tf_stub = ModuleType("transformers")
    tf_stub.__spec__ = None
    tf_stub.__path__ = []
    tf_stub.set_seed = lambda *_args, **_kwargs: None
    trainer_utils = ModuleType("transformers.trainer_utils")
    trainer_utils.get_last_checkpoint = lambda *_args, **_kwargs: None
    utils_mod = ModuleType("transformers.utils")
    utils_mod.logging = SimpleNamespace(
        set_verbosity=lambda *args, **kwargs: None,
        enable_default_handler=lambda *args, **kwargs: None,
        enable_explicit_format=lambda *args, **kwargs: None,
    )
    sys.modules["transformers"] = tf_stub
    sys.modules["transformers.trainer_utils"] = trainer_utils
    sys.modules["transformers.utils"] = utils_mod
    tf_stub.trainer_utils = trainer_utils
    tf_stub.utils = utils_mod


_install_transformers_stub()


def _install_accelerate_stub() -> None:
    """Provide a tiny accelerate stub for environments without the package."""
    if "accelerate" in sys.modules:
        return
    spec = None
    try:
        spec = importlib.util.find_spec("accelerate")
    except ValueError:
        spec = None
    if spec is not None:
        return
    accel_mod = ModuleType("accelerate")

    class _Accelerator:
        def __init__(self, **_kwargs):
            self.is_main_process = True
            self.process_index = 0
            self.num_processes = 1
            self.device = "cpu"
            self.gradient_accumulation_steps = 1
            self.sync_gradients = True
            self.gradient_state = SimpleNamespace(
                set_gradient_accumulation_steps=lambda _steps: None
            )

        def clip_grad_norm_(self, *_args, **_kwargs):
            return 0.0

        def gather(self, value):
            return value

        def gather_object(self, value):
            return [value]

        def broadcast_object_list(self, *_args, **_kwargs):
            return None

        def wait_for_everyone(self):
            return None

        def accumulate(self, _model):
            @contextmanager
            def _ctx():
                yield

            return _ctx()

        def backward(self, _loss):
            return None

        def load_state(self, _path):
            return None

        def save_state(self, _path):
            return None

        def unwrap_model(self, model):
            return model

        def prepare(self, *objects):
            if len(objects) == 1:
                return objects[0]
            return objects

        def set_gradient_accumulation_steps(self, steps):
            self.gradient_accumulation_steps = steps
            setter = getattr(
                self.gradient_state, "set_gradient_accumulation_steps", None
            )
            if callable(setter):
                setter(steps)

    accel_mod.Accelerator = _Accelerator
    accel_state = ModuleType("accelerate.state")
    accel_state.DistributedType = SimpleNamespace(DEEPSPEED="deepspeed")
    accel_mod.state = accel_state
    sys.modules.setdefault("accelerate", accel_mod)
    sys.modules.setdefault("accelerate.state", accel_state)


_install_accelerate_stub()
