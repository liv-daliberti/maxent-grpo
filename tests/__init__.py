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
    "tests.training.test_run_training_loss",
    "maxent_grpo.training.weighting.loss",
}


class _TorchAwareFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that clears torch stubs before importing critical modules."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _TORCH_SENSITIVE_MODULES:
            _clear_torch_stub()
        return importlib.machinery.PathFinder.find_spec(fullname, path, target)


if not any(isinstance(finder, _TorchAwareFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _TorchAwareFinder())
