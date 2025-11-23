"""Smoke tests for the legacy utils namespace."""

from __future__ import annotations


def test_utils_init_exports_empty_all():
    import utils

    assert utils.__all__ == []
