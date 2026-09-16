"""Synthetic-only final reporting and immutable publication checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'ops/exp_scaling/summarize_modebench_level3_v3.py'
spec = importlib.util.spec_from_file_location('v3_summary_tests', SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


@pytest.fixture
def env(tmp_path, monkeypatch):
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m.common, 'CAMPAIGN', campaign)
    monkeypatch.setattr(m.common, 'DATASET', tmp_path / 'data/v3')
    monkeypatch.setattr(m.common, 'OLD_DATASET', tmp_path / 'data/v2')
    monkeypatch.setattr(m.common, 'OLD_REPLAY', tmp_path / 'old_replay.json')
    for key, path in {'SOURCE': tmp_path / 'helper.py', 'TEST': tmp_path / 'tests.py',
                      'REPORT': campaign / 'confirmation/confirmation_report.json',
                      'OUTPUT': tmp_path / 'artifacts/summary.md', 'HERE': campaign / 'summary',
                      'INTENT': campaign / 'summary/publication_intent.json',
                      'PUBLICATION': campaign / 'summary/publication.json'}.items():
        monkeypatch.setattr(m, key, path)
    m.SOURCE.write_text('reviewed helper'); m.TEST.write_text('reviewed tests')
    directory = tmp_path / 'evidence'; directory.mkdir()
    evidence_file = directory / 'original.json'; evidence_file.write_text('sealed evidence')
    return SimpleNamespace(root=tmp_path, evidence_file=evidence_file, directory=directory)


def report_fixture(env, monkeypatch, passed=True):
    domains = {}
    for domain in m.common.DOMAINS:
        baseline = {'pass1': .25, 'pass8': .5}
        candidate = {'pass1': .265625, 'pass8': .515625}
        if not passed and domain == 'python_factors':
            candidate = {'pass1': .3125, 'pass8': .625}
        differences = {metric: candidate[metric] - baseline[metric] for metric in m.METRICS}
        gates = {metric: abs(differences[metric]) <= m.common.TOLERANCES[metric] for metric in m.METRICS}
        fresh = domain in m.common.REVISED
        domains[domain] = {'means': {'baseline': baseline, 'candidate': candidate}, 'differences': differences,
            'within_tolerance': gates, 'observed_approximate_match': all(gates.values()),
            'fresh_candidate_confirmation': fresh, 'candidate_confirmation_round': 2 if fresh else 1,
            'fixed_reference': {'receipt_path': str(env.root / ('historical_l1_' + domain + '.json'))},
            'candidate_receipt': {'path': str(env.root / ('candidate_' + domain + '.json'))}}
    report = {'status': 'matched_fixed_reference' if passed else 'outside_fixed_reference_tolerance',
        'confirmation_match_verified': passed, 'all_five_domains_complete': True, 'errors': {}, 'missing_domains': [],
        'tolerances': m.common.TOLERANCES, 'domains': domains,
        'information_boundary': {'reference_semantics': 'fixed_measured_level1_benchmark', 'adaptive_confirmation_round': 2,
            'historical_level1_confirmation_used_as_fixed_reference': True,
            'historical_level3_confirmation_used_for_mixture_fitting': False,
            'new_candidate_fitting_uses_development_outcomes_only': True,
            'all_five_fresh_same_round': False, 'statistical_equivalence_claimed': False, 'treatment_training_started': False},
        'dataset': {'dataset_root': str(m.common.DATASET), 'split_sizes': m.common.SPLITS,
            'retained_domains': list(m.common.RETAINED), 'fresh_candidate_domains': list(m.common.REVISED),
            'path': str(m.common.DATASET / 'identity.json'), 'sha256': 'i' * 64,
            'frozen_recipe_bundle_path': str(m.common.DATASET / 'frozen_recipes.json')},
        'attempts': 8192, 'new_confirmation_jobs': 2, 'all_attempts_regraded_with_original_grader': True,
        'every_attempt_including_failures_and_full_canonical_keys_compared': True,
        'completed_development_audit': {'path': str(m.common.CAMPAIGN / 'development/completed_development_audit.json')},
        'registration_path': str(m.common.CAMPAIGN / 'registration.json')}
    m.common.atomic_new(m.REPORT, report)
    def validate(path):
        assert path == m.REPORT
        return {'path': str(path), 'sha256': m.common.digest(path), 'report': deepcopy(report), 'status': report['status'],
            'files_sha256': {str(path): m.common.digest(path), str(env.evidence_file): m.common.digest(env.evidence_file)},
            'directory_files': {str(env.directory): [str(env.evidence_file)]}}
    validator = Mock(side_effect=validate)
    monkeypatch.setattr(m.importlib, 'import_module', lambda name: SimpleNamespace(validate_confirmation_report=validator))
    return report, validator


@pytest.mark.parametrize('passed', [True, False])
def test_preview_authenticates_complete_report_and_honestly_renders_both_outcomes(env, monkeypatch, passed):
    report, validator = report_fixture(env, monkeypatch, passed)
    text, evidence = m.build_summary()
    assert ('All five comparisons passed' in text) is passed
    assert ('did not meet all fixed-reference tolerances' in text) is not passed
    assert text.count('Fresh V3, round 2') == 2 and text.count('Retained V2, round 1') == 3
    for domain in m.common.DOMAINS:
        for split in m.common.SPLITS:
            assert str(m.common.DATASET / domain / split) in text
        assert m.pair(report['domains'][domain]['means']['baseline']) in text
        assert m.pair(report['domains'][domain]['means']['candidate']) in text
        assert m.pair(report['domains'][domain]['differences']) in text
    assert '39,424' in text and '8,192' in text and '40,960' in text
    assert 'not a claim of statistical equivalence' in text and 'not five fresh comparisons' in text
    assert 'No treatment training has started' in text and 'fixed, completed historical V2' in text
    assert evidence['source_sha256'] == m.common.digest(m.SOURCE)
    assert not m.OUTPUT.exists() and not m.INTENT.exists()
    validator.assert_called_once_with(path=m.REPORT)


@pytest.mark.parametrize('mutation', ['overall', 'gate', 'delta', 'origin', 'reference', 'equivalence', 'counts', 'grade', 'dataset'])
def test_report_display_rejects_inconsistent_numerical_or_methodology_claim(env, monkeypatch, mutation):
    report, _ = report_fixture(env, monkeypatch)
    changed = deepcopy(report)
    if mutation == 'overall': changed['confirmation_match_verified'] = False
    elif mutation == 'gate': changed['domains']['graph_coloring']['within_tolerance']['pass1'] = False
    elif mutation == 'delta': changed['domains']['graph_coloring']['differences']['pass1'] = .5
    elif mutation == 'origin': changed['domains']['pantry']['fresh_candidate_confirmation'] = True
    elif mutation == 'reference': changed['information_boundary']['historical_level1_confirmation_used_as_fixed_reference'] = False
    elif mutation == 'equivalence': changed['information_boundary']['statistical_equivalence_claimed'] = True
    elif mutation == 'counts': changed['attempts'] = 4096
    elif mutation == 'grade': changed['all_attempts_regraded_with_original_grader'] = False
    else: changed['dataset']['split_sizes'] = {'train': 128, 'dev': 128, 'eval': 128}
    with pytest.raises(ValueError): m.validate_display_contract(changed)


def test_absent_report_default_reads_no_outcomes_or_publishes(env, monkeypatch, capsys):
    never = Mock(side_effect=AssertionError('premature report authentication'))
    monkeypatch.setattr(m, 'build_summary', never)
    assert m.main([]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'waiting_for_completed_confirmation_report'
    never.assert_not_called(); assert not m.OUTPUT.exists()


def test_completed_default_is_readonly_preview(env, monkeypatch, capsys):
    report_fixture(env, monkeypatch)
    assert m.main([]) == 0
    assert 'All five comparisons passed' in capsys.readouterr().out
    assert not m.OUTPUT.exists() and not m.HERE.exists()


@pytest.mark.parametrize('passed', [True, False])
def test_explicit_publication_pins_source_report_evidence_and_writes_once(env, monkeypatch, passed):
    report, validator = report_fixture(env, monkeypatch, passed)
    record = m.publish(m.common.digest(m.REPORT))
    assert record['status'] == report['status'] and record['summary_sha256'] == m.common.digest(m.OUTPUT)
    assert record['files_sha256'][str(m.REPORT)] == m.common.digest(m.REPORT)
    assert record['files_sha256'][str(m.SOURCE)] == m.common.digest(m.SOURCE)
    assert record['files_sha256'][str(m.TEST)] == m.common.digest(m.TEST)
    assert record['intent_sha256'] == m.common.digest(m.INTENT)
    assert m.common.read(m.PUBLICATION) == record
    with pytest.raises(ValueError, match='forbids retry'): m.publish(m.common.digest(m.REPORT))
    validator.assert_called_once()


def test_publication_requires_explicit_correct_report_pin(env, monkeypatch):
    _, validator = report_fixture(env, monkeypatch)
    with pytest.raises(ValueError, match='explicit completed report'): m.publish(None)
    with pytest.raises(ValueError, match='report pin differs'): m.publish('0' * 64)
    validator.assert_not_called(); assert not m.INTENT.exists() and not m.OUTPUT.exists()


def test_changed_report_during_authentication_is_not_published(env, monkeypatch):
    _, validator = report_fixture(env, monkeypatch)
    original = validator.side_effect
    def changed(path):
        result = original(path); path.write_text('{}'); return result
    validator.side_effect = changed
    with pytest.raises(ValueError, match='changed during authentication'): m.publish(m.common.digest(m.REPORT))
    assert not m.OUTPUT.exists() and not m.INTENT.exists()


def test_interrupted_publication_keeps_intent_and_forbids_retry(env, monkeypatch):
    report_fixture(env, monkeypatch)
    pin = m.common.digest(m.REPORT)
    monkeypatch.setattr(m, 'atomic_new_text', Mock(side_effect=KeyboardInterrupt('interrupted')))
    with pytest.raises(KeyboardInterrupt): m.publish(pin)
    assert m.INTENT.is_file() and not m.PUBLICATION.exists()
    with pytest.raises(ValueError, match='forbids retry'): m.publish(pin)


def test_late_extra_evidence_file_prevents_success_receipt(env, monkeypatch):
    report_fixture(env, monkeypatch)
    original = m.atomic_new_text
    def add_extra(path, text):
        original(path, text); (env.directory / 'late_extra').write_text('changed inventory')
    monkeypatch.setattr(m, 'atomic_new_text', add_extra)
    with pytest.raises(ValueError, match='directory inventory changed'): m.publish(m.common.digest(m.REPORT))
    assert m.INTENT.exists() and m.OUTPUT.exists() and not m.PUBLICATION.exists()
