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

import os
import warnings
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except Exception:  # pragma: no cover - pydantic missing
    pass

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.utils\.hub",
)
warnings.filterwarnings(
    "ignore",
    message=r"A NumPy version >=1\.21\.6 and <1\.28\.0 is required for this version of SciPy.*",
    category=UserWarning,
)


def _patch_vllm_guided_decoding_params() -> None:
    """Backfill ``vllm.sampling_params.GuidedDecodingParams`` when absent.

    Some cluster images expose a vLLM build where TRL's expected symbol is not
    present. TRL imports this symbol at module import time (even if guided
    decoding is unused), so define a tiny compatibility placeholder to allow
    server-mode training to boot.
    """

    try:
        import vllm.sampling_params as _sp  # type: ignore[import-not-found]
    except Exception:
        return

    if hasattr(_sp, "GuidedDecodingParams"):
        return

    class GuidedDecodingParams:  # pragma: no cover - runtime compatibility shim
        """Compatibility placeholder for TRL imports on older vLLM builds."""

        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    setattr(_sp, "GuidedDecodingParams", GuidedDecodingParams)


_patch_vllm_guided_decoding_params()


def _patch_vllm_get_open_port() -> None:
    """Backfill ``vllm.utils.get_open_port`` when absent."""

    try:
        import socket
        import vllm.utils as _vu  # type: ignore[import-not-found]
    except Exception:
        return

    if hasattr(_vu, "get_open_port"):
        return

    def get_open_port() -> int:  # pragma: no cover - runtime compatibility shim
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    setattr(_vu, "get_open_port", get_open_port)


_patch_vllm_get_open_port()

_ROOT_DIR = Path(__file__).resolve().parent
_VAR_ROOT = _ROOT_DIR / "var"
_VAR_ROOT.mkdir(parents=True, exist_ok=True)
_PYCACHE_DIR = _VAR_ROOT / "pycache"
_PYCACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(_PYCACHE_DIR))

_CACHE_ROOT = _VAR_ROOT / "cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _setdefault_dir(env_name: str, path: Path) -> Path:
    """Populate filesystem-backed env vars with repo-local defaults."""

    if not os.environ.get(env_name):
        os.environ[env_name] = str(path)
    resolved = Path(os.environ[env_name]).expanduser()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Re-create PIP cache may fail when pointed at a file; tolerate silently.
        pass
    return resolved


_ENV_PATH_DEFAULTS = {
    "XDG_CACHE_HOME": _CACHE_ROOT / "xdg",
    "XDG_CONFIG_HOME": _VAR_ROOT / "config",
    "PIP_CACHE_DIR": _CACHE_ROOT / "pip",
    "TMPDIR": _VAR_ROOT / "tmp",
    "HF_HOME": _CACHE_ROOT / "huggingface",
    "HUGGINGFACE_HUB_CACHE": _CACHE_ROOT / "huggingface" / "hub",
    "HF_HUB_CACHE": _CACHE_ROOT / "huggingface" / "hub",
    "HF_DATASETS_CACHE": _CACHE_ROOT / "huggingface" / "datasets",
    "TRANSFORMERS_CACHE": _CACHE_ROOT / "huggingface" / "transformers",
}
for _env, _path in _ENV_PATH_DEFAULTS.items():
    _setdefault_dir(_env, _path)
