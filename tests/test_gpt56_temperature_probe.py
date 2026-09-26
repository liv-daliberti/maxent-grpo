"""Cached capability checks must never masquerade as a different grid."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('gpt_probe', Path(__file__).resolve().parents[1] / 'ops/probe_gpt56_temperature.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def result(pairs):
    return {'results': [{'reasoning_effort': e, 'requested_temperature': t} for e, t in pairs]}


def test_cached_original_grid_allows_recorded_fallback():
    m.validate_cached_grid(result([('medium', 0.5), ('medium', 1.0), ('none', 1.5)]), [0.5, 1.0], ['medium'])


@pytest.mark.parametrize('pairs', [[('none', 0.5)], [('none', 0.0), ('none', 0.0)], [('none', 0.0), ('medium', 0.0)]])
def test_rejects_missing_duplicate_or_extra_cached_conditions(pairs):
    with pytest.raises(ValueError, match='different grid'):
        m.validate_cached_grid(result(pairs), [0.0], ['none'])


def test_zero_is_preserved_as_a_requested_temperature():
    m.validate_cached_grid(result([('none', 0.0), ('medium', 0.0)]), [0.0], ['none', 'medium'])
