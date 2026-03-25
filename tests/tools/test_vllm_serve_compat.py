from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "vllm_serve_compat.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "maxent_test_vllm_serve_compat",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_blocked_tail_to_allowed_token_ids_contiguous_tail():
    allowed = _MODULE._blocked_tail_to_allowed_token_ids([5, 6, 7, 8, 9])
    assert allowed == [0, 1, 2, 3, 4]


def test_blocked_tail_to_allowed_token_ids_rejects_non_contiguous_tail():
    with pytest.raises(ValueError, match="contiguous tail range"):
        _MODULE._blocked_tail_to_allowed_token_ids([5, 7, 8, 9])
