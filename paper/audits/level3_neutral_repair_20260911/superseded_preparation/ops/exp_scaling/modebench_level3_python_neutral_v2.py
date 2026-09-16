"""Prospective numeric difficulty presets for neutral Python Level 3.

Reuse the exact uniform conditional sampler in Python v7 in an isolated module.
Only numeric case windows/minimum bands and the versioned RNG namespace differ.
"""
import importlib.util
from pathlib import Path

BASE = Path(__file__).with_name('modebench_level3_python_v7.py')
_spec = importlib.util.spec_from_file_location('_neutral_python_candidate_sampler', BASE)
_sampler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sampler)
PROFILE = 'python_neutral_numeric_bands_v1'
GENERATOR = 'modebench_level3_python_neutral_candidate_v2'
CASE_WINDOWS = ((12, 240), (30, 256), (50, 320), (80, 512))
MINIMUM_BANDS = ((12, 29), (30, 49), (50, 79), (80, 119))
PRESETS = {i: f'neutral_minimum_{a}_{b}_cases_{lo}_{hi}'
           for i, ((a, b), (lo, hi)) in enumerate(zip(MINIMUM_BANDS, CASE_WINDOWS))}
for _name in ('PROFILE', 'GENERATOR', 'CASE_WINDOWS', 'MINIMUM_BANDS', 'PRESETS'):
    setattr(_sampler, _name, globals()[_name])
build_pool = _sampler.build_pool
available_capacity = _sampler.available_capacity
eligible_cases = _sampler.eligible_cases
catalog = _sampler.catalog
