"""The new Python route cannot contaminate other sealed calibration domains."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import fit_modebench_level3_python_v5_independent as revised
import finalize_modebench_level3_independent as finalizer


def test_generator_route_is_restored_when_authenticated_fitting_fails():
    original = revised.mixture.generator
    with pytest.raises(RuntimeError, match='invalid saved receipt'):
        with revised._registered_generator_route():
            assert revised.mixture.generator is revised.generator
            raise RuntimeError('invalid saved receipt')
    assert revised.mixture.generator is original


def test_unregistered_domain_cannot_use_python_revision(monkeypatch):
    monkeypatch.setattr(revised, 'revision_identity', lambda: {'name': 'python_v5'})
    with pytest.raises(ValueError, match='unknown or changed'):
        finalizer.candidate_route({'candidate_revision': {'name': 'python_v5'}}, 'mathir')


@pytest.mark.parametrize('changed', [None, {}, {'name': 'python_v4'}, {'name': 'python_v5', 'adapter_sha256': 'changed'}])
def test_unknown_or_mutated_recipe_revision_fails_closed(monkeypatch, changed):
    monkeypatch.setattr(revised, 'revision_identity', lambda: {'name': 'python_v5', 'adapter_sha256': 'registered'})
    with pytest.raises(ValueError, match='unknown or changed'):
        finalizer.candidate_route({'candidate_revision': changed}, 'python_factors')


def test_old_recipes_keep_the_original_generator_and_source_identity(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(finalizer, 'generator', lambda domain: sentinel)
    monkeypatch.setattr(finalizer, 'generator_sources', lambda domain: {'original.py': 'sealed'})
    assert finalizer.recipe_generator('python_factors', {}) is sentinel
    assert finalizer.recipe_generator_sources('python_factors', {}) == {'original.py': 'sealed'}
