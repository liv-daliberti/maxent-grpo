"""Test package for MaxEnt helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import importlib.abc
import importlib.machinery

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    # Ensure src-layout packages are discoverable whenever tests.* is imported.
    sys.path.insert(0, str(_SRC_ROOT))


def _clear_torch_stub() -> None:
    module = sys.modules.get("torch")
    if module is None:
        return
    if getattr(module, "__file__", None):
        return
    for key in list(sys.modules):
        if key == "torch" or key.startswith("torch."):
            sys.modules.pop(key, None)


_TORCH_SENSITIVE_MODULES = {
    "tests.test_run_training_loss",
    "training.weighting.loss",
}


class _TorchAwareFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that clears torch stubs before importing critical modules."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _TORCH_SENSITIVE_MODULES:
            _clear_torch_stub()
        return importlib.machinery.PathFinder.find_spec(fullname, path, target)


if not any(isinstance(finder, _TorchAwareFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _TorchAwareFinder())
