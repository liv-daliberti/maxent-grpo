from __future__ import annotations

import importlib
import os
from pathlib import Path


def test_sitecustomize_sets_repo_local_hf_dirs(monkeypatch):
    """sitecustomize should default HF caches to var/cache inside the repo."""

    cache_keys = [
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "TMPDIR",
        "PIP_CACHE_DIR",
    ]
    for key in cache_keys:
        monkeypatch.delenv(key, raising=False)

    import sitecustomize as bootstrap

    module = importlib.reload(bootstrap)
    var_root = module._VAR_ROOT
    hf_cache = var_root / "cache" / "huggingface"
    assert Path(os.environ["HF_HOME"]) == hf_cache
    assert (hf_cache).is_dir()
    assert Path(os.environ["HUGGINGFACE_HUB_CACHE"]) == hf_cache / "hub"
    assert Path(os.environ["HF_HUB_CACHE"]) == hf_cache / "hub"
    assert Path(os.environ["HF_DATASETS_CACHE"]) == hf_cache / "datasets"
    assert Path(os.environ["TRANSFORMERS_CACHE"]) == hf_cache / "transformers"
    assert Path(os.environ["XDG_CACHE_HOME"]) == var_root / "cache" / "xdg"
    assert Path(os.environ["XDG_CONFIG_HOME"]) == var_root / "config"
    assert Path(os.environ["TMPDIR"]) == var_root / "tmp"
    assert Path(os.environ["PIP_CACHE_DIR"]) == var_root / "cache" / "pip"
