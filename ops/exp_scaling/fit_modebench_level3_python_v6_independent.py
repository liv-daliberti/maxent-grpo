#!/usr/bin/env python3
"""Fit registered Python v6 development pools with the unchanged mixture rule.

The sealed v2 evaluator, independent seed validator, weight objective, grid,
selection seed and both gates are reused. Only the explicitly registered Python
numeric case law changes. Older failed recipes and their sources remain intact.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling'):
    sys.path.insert(0, str(ROOT / directory))

import fit_modebench_level3 as mixture
import fit_modebench_level3_independent as independent

REVISION = 'python_v6'
REGISTRATION = ROOT / 'var/artifacts/modebench_level3_v2/python_v6/candidate_protocol.json'
REGISTRATION_SHA = '15885c03ec9cdedb0624fb80aded112d21edaa9a09134f5e52c7315fd81a131f'
POOL_ROOT = ROOT / 'var/data/modebench_level3_calibration_python_v6'
BASELINE = ROOT / 'var/results/modebench_level3_v2/calibration_05b_python_factors.json'
BASELINE_SHA = 'd72b6c76ea0d3ccf7dafb5bdb0520031e0670780437e823da31f8bf1f385ac7b'
GENERATOR = ROOT / 'ops/exp_scaling/modebench_level3_python_v6.py'
_ROUTE_LOCK = threading.RLock()


def registration():
    if mixture.file_sha(REGISTRATION) != REGISTRATION_SHA:
        raise ValueError('Python v6 prospective registration changed')
    record = json.loads(REGISTRATION.read_text())
    expected = {
        'schema': 'modebench_level3_python_candidate_protocol_v6',
        'candidate_revision': REVISION,
        'development_only': True,
        'generator_path': str(GENERATOR),
        'generator_source_sha256': mixture.file_sha(GENERATOR),
        'materializer_path': str(ROOT / 'ops/exp_scaling/materialize_modebench_level3_python_v6.py'),
        'materializer_source_sha256': mixture.file_sha(ROOT / 'ops/exp_scaling/materialize_modebench_level3_python_v6.py'),
        'pool_root': str(POOL_ROOT),
        'development_draw_labels': list(independent.DEVELOPMENT_DRAW_LABELS),
        'baseline_receipt_path': str(BASELINE),
        'baseline_receipt_sha256': BASELINE_SHA,
    }
    if mixture.file_sha(BASELINE) != BASELINE_SHA:
        raise ValueError('registered corrected Python baseline receipt changed')
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Python v6 candidate registration differs from the fixed development route')
    # Authenticate the recorded historical inputs; do not require the global
    # history inventory to stay empty of the future, legitimately published v2.
    snapshot = record['source_snapshot']
    for path, expected_hash in snapshot['files_sha256'].items():
        if mixture.file_sha(path) != expected_hash:
            raise ValueError(f'Python v6 registered source input changed: {path}')
    for directory, expected_files in snapshot['directory_files'].items():
        actual = sorted(str(path.resolve()) for path in Path(directory).rglob('*') if path.is_file())
        if actual != expected_files:
            raise ValueError(f'Python v6 registered source inventory changed: {directory}')
    return record


def generator(domain):
    if domain != 'python_factors':
        raise ValueError('Python v6 route requires python_factors')
    registration()
    from modebench_level3_python_v6 import build_pool
    return build_pool


def generator_sources(domain):
    generator(domain)
    sources = mixture.local_dependency_sources([
        GENERATOR, Path(__file__),
        ROOT / 'ops/exp_scaling/materialize_modebench_level3_python_v6.py',
        ROOT / 'ops/exp_scaling/materialize_modebench_level3.py',
        ROOT / 'ops/exp_scaling/modebench_level3_allocation.py',
    ])
    sources[str(REGISTRATION.relative_to(ROOT))] = mixture.file_sha(REGISTRATION)
    return dict(sorted(sources.items()))


def revision_identity():
    registration()
    return {
        'name': REVISION,
        'registration_path': str(REGISTRATION),
        'registration_sha256': mixture.file_sha(REGISTRATION),
        'adapter_path': str(Path(__file__).resolve()),
        'adapter_sha256': mixture.file_sha(Path(__file__)),
    }


def validate_revision(recipe, domain):
    if domain != 'python_factors' or recipe.get('candidate_revision') != revision_identity():
        raise ValueError('unknown or changed independent candidate revision')


@contextmanager
def _registered_generator_route():
    # The legacy fitter intentionally exposes one generator lookup. Restrict
    # its temporary replacement to this synchronous call and restore it even
    # when receipt authentication or fitting raises. No files are modified.
    with _ROUTE_LOCK:
        previous = mixture.generator
        mixture.generator = generator
        try:
            yield
        finally:
            mixture.generator = previous


def fit_recipe(baseline_path, score_paths, domain, output=None):
    registration()
    if domain != 'python_factors' or Path(baseline_path).resolve() != BASELINE:
        raise ValueError('Python v6 requires the registered corrected development baseline')
    expected_receipts = {ROOT / f'var/results/modebench_level3_v2/calibration_3b_python_v6_d{tier}.json': tier for tier in range(4)}
    if len(score_paths) != 4 or {Path(path).resolve() for path in score_paths} != set(expected_receipts):
        raise ValueError('Python v6 requires the four canonical registered candidate receipts')
    for path in score_paths:
        receipt = json.loads(Path(path).read_text())
        source = receipt.get('identity', {}).get('source', {})
        if source.get('kind') != 'jsonl':
            raise ValueError('Python v6 candidate must use a registered JSONL pool')
        pool = Path(source.get('path', '')).resolve()
        expected_pool = POOL_ROOT / 'pools/python_factors' / f'difficulty_{expected_receipts[Path(path).resolve()]}.jsonl'
        if pool != expected_pool:
            raise ValueError('candidate receipt source differs from its registered Python v6 tier')
        certificate = json.loads(pool.with_suffix('.identity.json').read_text())
        if (certificate.get('candidate_revision') != REVISION
                or certificate.get('candidate_protocol_path') != str(REGISTRATION)
                or certificate.get('candidate_protocol_sha256') != REGISTRATION_SHA
                or not certificate.get('checks')
                or any(value is not True for value in certificate['checks'].values())):
            raise ValueError('Python v6 pool certificate differs from the registered structural audit')
    initial_sources = generator_sources(domain)
    initial_revision = revision_identity()
    with _registered_generator_route():
        result = independent.fit_recipe(baseline_path, score_paths, domain)
    result['candidate_revision'] = initial_revision
    result['provenance']['generator_sources_sha256'] = initial_sources
    result['information_boundary']['candidate_revision_basis'] = (
        'The corrected Python v4 and v5 development fits failed. This registered '
        'numeric minimum-band range uses new development prompts; no confirmation outcome '
        'enters fitting, and the original full-pool objective and both gates remain fixed.'
    )
    if initial_sources != generator_sources(domain) or initial_revision != revision_identity():
        raise ValueError('Python v6 generator registration or adapter changed during fitting')
    if output is not None:
        mixture.atomic_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--scores', type=Path, required=True, nargs=4)
    parser.add_argument('--domain', choices=['python_factors'], default='python_factors')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = fit_recipe(args.baseline, args.scores, args.domain, args.output)
    print(json.dumps({key: result[key] for key in ('decision', 'weights', 'development')}))


if __name__ == '__main__':
    main()
