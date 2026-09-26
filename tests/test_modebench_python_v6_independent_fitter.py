"""Prospective Python v6 keeps the registered inputs and frozen fit rule intact."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import fit_modebench_level3_python_v6_independent as revised


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def registration_files(tmp_path, monkeypatch):
    monkeypatch.setattr(revised, 'ROOT', tmp_path)
    monkeypatch.setattr(revised, 'REGISTRATION', tmp_path / 'candidate_protocol.json')
    monkeypatch.setattr(revised, 'POOL_ROOT', tmp_path / 'pools')
    generator = tmp_path / 'ops/exp_scaling/modebench_level3_python_v6.py'
    materializer = tmp_path / 'ops/exp_scaling/materialize_modebench_level3_python_v6.py'
    for path in (generator, materializer):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixed source\n')
    monkeypatch.setattr(revised, 'GENERATOR', generator)
    baseline = write_json(tmp_path / 'baseline.json', {'fixed': True})
    monkeypatch.setattr(revised, 'BASELINE', baseline)
    monkeypatch.setattr(revised, 'BASELINE_SHA', revised.mixture.file_sha(baseline))
    historical = tmp_path / 'history'
    historical.mkdir()
    item = historical / 'fixed.txt'
    item.write_text('fixed')
    record = {
        'schema': 'modebench_level3_python_candidate_protocol_v6',
        'candidate_revision': 'python_v6',
        'development_only': True,
        'generator_path': str(generator),
        'generator_source_sha256': revised.mixture.file_sha(generator),
        'materializer_path': str(materializer),
        'materializer_source_sha256': revised.mixture.file_sha(materializer),
        'pool_root': str(revised.POOL_ROOT),
        'development_draw_labels': list(revised.independent.DEVELOPMENT_DRAW_LABELS),
        'baseline_receipt_path': str(baseline),
        'baseline_receipt_sha256': revised.BASELINE_SHA,
        'source_snapshot': {
            'files_sha256': {str(item): revised.mixture.file_sha(item)},
            'directory_files': {str(historical): [str(item)]},
        },
    }
    write_json(revised.REGISTRATION, record)
    monkeypatch.setattr(revised, 'REGISTRATION_SHA', revised.mixture.file_sha(revised.REGISTRATION))
    return record, historical, item


def test_registration_authenticates_fixed_source_and_boundary(registration_files):
    record, _, _ = registration_files
    assert revised.registration() == record


@pytest.mark.parametrize('mutation', ['registration', 'baseline', 'generator', 'history_file', 'history_inventory'])
def test_registration_rejects_input_or_inventory_mutation(registration_files, mutation):
    _, historical, item = registration_files
    if mutation == 'registration':
        revised.REGISTRATION.write_text('{}')
    elif mutation == 'baseline':
        revised.BASELINE.write_text('{}')
    elif mutation == 'generator':
        revised.GENERATOR.write_text('# changed\n')
    elif mutation == 'history_file':
        item.write_text('changed')
    else:
        (historical / 'added.txt').write_text('extra')
    with pytest.raises(ValueError):
        revised.registration()


@pytest.fixture
def authenticated_fit_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(revised, 'ROOT', tmp_path)
    monkeypatch.setattr(revised, 'POOL_ROOT', tmp_path / 'candidate_pool')
    monkeypatch.setattr(revised, 'BASELINE', tmp_path / 'fixed_baseline.json')
    monkeypatch.setattr(revised, 'REGISTRATION', tmp_path / 'protocol.json')
    monkeypatch.setattr(revised, 'registration', lambda: {'registered': True})
    scores = []
    pools = []
    for tier in range(4):
        pool = revised.POOL_ROOT / 'pools/python_factors' / f'difficulty_{tier}.jsonl'
        pool.parent.mkdir(parents=True, exist_ok=True)
        pool.write_text('{}\n')
        write_json(pool.with_suffix('.identity.json'), {
            'candidate_revision': 'python_v6',
            'candidate_protocol_path': str(revised.REGISTRATION),
            'candidate_protocol_sha256': revised.REGISTRATION_SHA,
            'checks': {'support': True, 'fresh': True},
        })
        scores.append(write_json(tmp_path / 'var/results/modebench_level3_v2' / f'calibration_3b_python_v6_d{tier}.json',
                                 {'identity': {'source': {'kind': 'jsonl', 'path': str(pool)}}}))
        pools.append(pool)
    monkeypatch.setattr(revised, 'generator_sources', lambda domain: {'unchanged.py': 'source_hash'})
    monkeypatch.setattr(revised, 'revision_identity', lambda: {'name': 'python_v6', 'adapter_sha256': 'sealed'})
    return scores, pools


@pytest.mark.parametrize('wrong', ['domain', 'baseline', 'duplicate_score', 'old_score', 'missing_score'])
def test_fit_rejects_unregistered_domain_baseline_or_receipt_set(authenticated_fit_inputs, monkeypatch, wrong):
    scores, _ = authenticated_fit_inputs
    monkeypatch.setattr(revised.independent, 'fit_recipe', lambda *args: pytest.fail('fit must not run'))
    domain, baseline = 'python_factors', revised.BASELINE
    if wrong == 'domain':
        domain = 'mathir'
    elif wrong == 'baseline':
        baseline = revised.BASELINE.with_name('other.json')
    elif wrong == 'duplicate_score':
        scores[-1] = scores[0]
    elif wrong == 'old_score':
        scores[-1] = scores[-1].with_name('calibration_3b_python_v5_d3.json')
    else:
        scores = scores[:-1]
    with pytest.raises(ValueError):
        revised.fit_recipe(baseline, scores, domain)


@pytest.mark.parametrize('wrong', ['tier_swapped', 'kind', 'certificate_revision', 'certificate_hash', 'certificate_checks_empty', 'certificate_check_false'])
def test_fit_rejects_source_or_certificate_changes(authenticated_fit_inputs, monkeypatch, wrong):
    scores, pools = authenticated_fit_inputs
    monkeypatch.setattr(revised.independent, 'fit_recipe', lambda *args: pytest.fail('fit must not run'))
    receipt = json.loads(scores[0].read_text())
    certificate_path = pools[0].with_suffix('.identity.json')
    certificate = json.loads(certificate_path.read_text())
    if wrong == 'tier_swapped':
        receipt['identity']['source']['path'] = str(pools[1])
    elif wrong == 'kind':
        receipt['identity']['source']['kind'] = 'saved_dataset'
    elif wrong == 'certificate_revision':
        certificate['candidate_revision'] = 'python_v5'
    elif wrong == 'certificate_hash':
        certificate['candidate_protocol_sha256'] = 'other'
    elif wrong == 'certificate_checks_empty':
        certificate['checks'] = {}
    else:
        certificate['checks']['fresh'] = False
    write_json(scores[0], receipt)
    write_json(certificate_path, certificate)
    with pytest.raises(ValueError):
        revised.fit_recipe(revised.BASELINE, scores, 'python_factors')


def test_fit_calls_unchanged_rule_once_and_preserves_failed_result(authenticated_fit_inputs, monkeypatch):
    scores, _ = authenticated_fit_inputs
    original = revised.mixture.generator
    calls = []
    def frozen_fit(*args):
        assert revised.mixture.generator is revised.generator
        calls.append(args)
        return {'development_fit_pass': False, 'decision': 'failed', 'weights': [0.05, 0.65, 0.0, 0.3],
                'development': {'forecast': 'unaltered'}, 'provenance': {}, 'information_boundary': {}}
    monkeypatch.setattr(revised.independent, 'fit_recipe', frozen_fit)
    result = revised.fit_recipe(revised.BASELINE, scores, 'python_factors')
    assert calls == [(revised.BASELINE, scores, 'python_factors')]
    assert revised.mixture.generator is original
    assert result['development_fit_pass'] is False
    assert result['decision'] == 'failed'
    assert result['weights'] == [0.05, 0.65, 0.0, 0.3]
    assert result['development'] == {'forecast': 'unaltered'}
    assert result['candidate_revision']['name'] == 'python_v6'


def test_fitting_exception_restores_other_domain_generator(authenticated_fit_inputs, monkeypatch):
    scores, _ = authenticated_fit_inputs
    original = revised.mixture.generator
    def failed_fit(*args):
        raise RuntimeError('invalid saved receipt')
    monkeypatch.setattr(revised.independent, 'fit_recipe', failed_fit)
    with pytest.raises(RuntimeError, match='invalid saved receipt'):
        revised.fit_recipe(revised.BASELINE, scores, 'python_factors')
    assert revised.mixture.generator is original


def test_source_mutation_during_fit_rejects_publication(authenticated_fit_inputs, monkeypatch, tmp_path):
    scores, _ = authenticated_fit_inputs
    sources = iter([{'source.py': 'before'}, {'source.py': 'after'}])
    monkeypatch.setattr(revised, 'generator_sources', lambda domain: next(sources))
    monkeypatch.setattr(revised.independent, 'fit_recipe', lambda *args: {'provenance': {}, 'information_boundary': {}})
    output = tmp_path / 'must_not_exist.json'
    with pytest.raises(ValueError, match='changed during fitting'):
        revised.fit_recipe(revised.BASELINE, scores, 'python_factors', output)
    assert not output.exists()


@pytest.mark.parametrize('candidate,domain', [
    (None, 'python_factors'), ({'name': 'python_v5'}, 'python_factors'),
    ({'name': 'python_v6', 'adapter_sha256': 'other'}, 'python_factors'),
    ({'name': 'python_v6', 'adapter_sha256': 'sealed'}, 'mathir'),
])
def test_unknown_or_mutated_recipe_revision_fails_closed(monkeypatch, candidate, domain):
    monkeypatch.setattr(revised, 'revision_identity', lambda: {'name': 'python_v6', 'adapter_sha256': 'sealed'})
    with pytest.raises(ValueError, match='unknown or changed'):
        revised.validate_revision({'candidate_revision': candidate}, domain)
