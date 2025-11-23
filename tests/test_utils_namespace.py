"""Smoke test for utils namespace placeholder."""


def test_utils_all_empty():
    import utils

    assert utils.__all__ == []
