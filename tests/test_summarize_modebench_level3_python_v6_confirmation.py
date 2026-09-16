"""Reporting guards use synthetic evidence and never publish the real summary."""
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'ops/exp_scaling/summarize_modebench_level3_python_v6_confirmation.py'
SPEC = importlib.util.spec_from_file_location('confirmation_summary_under_test', SOURCE)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def report_fixture(matched=True):
    result = {
        'schema': 'modebench-level3-independent-observed-match-audit-v2', 'phase': 'confirmation',
        'status': 'observed_approximate_match' if matched else 'outside_match_tolerance',
        'all_five_domains_complete': True, 'errors': {}, 'missing_domains': [],
        'confirmation_match_verified': matched, 'all_five_observed_approximate_match': matched,
        'criteria': {'absolute_pass1_difference_at_most': .04, 'absolute_pass8_difference_at_most': .08,
            'distinct8_is_diagnostic_only': True, 'statistical_equivalence_claimed': False,
            'confirmation_rows_per_domain_per_model': 128, 'confirmation_draws_per_prompt': 4,
            'samples_per_draw': 8, 'all_five_domains_required': True,
            'expected_confirmation_seeds': [6329000, 6329001, 6329002, 6329003]},
        'evidence_validation': {key: True for key in (
            'metrics_recomputed_from_attempts', 'source_rows_verified_when_dataset_supplied',
            'independent_seed_schedules_recomputed', 'legacy_overlapping_seed_evidence_rejected',
            'level1_controls_reused_with_corrected_sampling', 'fresh_heldout_claim_applies_to_level3_only',
            'frozen_recipe_settings_required', 'baseline_source_pinned_to_level1_confirmation')},
        'prospective_confirmation_seal': {'path': '/synthetic/seal.json'}, 'domains': {},
    }
    for domain in summary.DOMAINS:
        fail = not matched and domain == 'python_factors'
        p1 = .3 if fail else .21
        result['domains'][domain] = {'baseline_rows': 128, 'candidate_rows': 128, 'draws_per_prompt': 4,
            'means': {'baseline': {'pass1': .2, 'pass8': .5}, 'candidate': {'pass1': p1, 'pass8': .55}},
            'candidate_minus_baseline': {'pass1': p1 - .2, 'pass8': .55 - .5},
            'within_tolerance': {'pass1': not fail, 'pass8': True}, 'observed_approximate_match': not fail}
    return result


@pytest.mark.parametrize('matched', [False, True])
def test_complete_success_or_honest_outside_tolerance_report_is_supported(matched):
    assert summary.validate_report(report_fixture(matched)) is matched


@pytest.mark.parametrize('mutation', [
    'legacy_schema', 'development', 'incomplete', 'missing_domain', 'errors', 'missing_list',
    'wrong_threshold', 'different_seeds', 'short_rows', 'short_draws', 'nonfinite', 'bool_metric',
    'out_of_range', 'false_delta', 'false_gate', 'integer_gate', 'domain_match', 'overall_match',
    'wrong_status', 'missing_control_limitation', 'missing_seal',
])
def test_incomplete_unbound_or_inconsistent_reports_are_refused(mutation):
    report = report_fixture()
    row = report['domains']['python_factors']
    if mutation == 'legacy_schema': report['schema'] = 'legacy'
    elif mutation == 'development': report['phase'] = 'development'
    elif mutation == 'incomplete': report['all_five_domains_complete'] = False
    elif mutation == 'missing_domain': del report['domains']['mathir']
    elif mutation == 'errors': report['errors'] = {'mathir': 'invalid receipt'}
    elif mutation == 'missing_list': report['missing_domains'] = ['mathir']
    elif mutation == 'wrong_threshold': report['criteria']['absolute_pass1_difference_at_most'] = .05
    elif mutation == 'different_seeds': report['criteria']['expected_confirmation_seeds'][0] += 1
    elif mutation == 'short_rows': row['candidate_rows'] = 64
    elif mutation == 'short_draws': row['draws_per_prompt'] = 3
    elif mutation == 'nonfinite': row['means']['candidate']['pass1'] = float('nan')
    elif mutation == 'bool_metric': row['means']['baseline']['pass1'] = True
    elif mutation == 'out_of_range': row['means']['candidate']['pass1'] = 1.1
    elif mutation == 'false_delta': row['candidate_minus_baseline']['pass1'] = 0
    elif mutation == 'false_gate': row['within_tolerance']['pass1'] = False
    elif mutation == 'integer_gate': row['within_tolerance']['pass1'] = 1
    elif mutation == 'domain_match': row['observed_approximate_match'] = False
    elif mutation == 'overall_match': report['confirmation_match_verified'] = False
    elif mutation == 'wrong_status': report['status'] = 'outside_match_tolerance'
    elif mutation == 'missing_control_limitation': report['evidence_validation']['fresh_heldout_claim_applies_to_level3_only'] = False
    else: report['prospective_confirmation_seal'] = None
    with pytest.raises(ValueError): summary.validate_report(report)


def render_inputs(matched=True):
    report = report_fixture(matched)
    identity = {'domains': {}, 'recipe_sha256': {domain: 'r' * 64 for domain in summary.DOMAINS}}
    recipes = {}
    for domain in summary.DOMAINS:
        identity['domains'][domain] = {split: {'path': str(summary.DATASET / domain / split), 'seed': 1000 + index}
                                      for index, split in enumerate(summary.SPLITS)}
        recipes[domain] = {'weights': [.25, .25, .25, .25]}
    return report, identity, {'finalizer_source_sha256': 'f' * 64}, recipes, {
        'bundle': 'b' * 64, 'identity': 'i' * 64, 'report': 'q' * 64}


@pytest.mark.parametrize('matched', [False, True])
def test_render_covers_every_domain_split_gate_and_control_limitation(matched):
    text = summary.render_summary(*render_inputs(matched))
    for domain, label in summary.LABELS.items():
        assert label in text
        for split in summary.SPLITS:
            assert str(summary.DATASET / domain / split) in text
        assert str(summary.DATASET / 'recipes' / (domain + '.json')) in text
    assert 'Train (384)' in text and 'Development (128)' in text and 'Evaluation (128)' in text
    assert 'Fresh held-out problems apply to Level3 only' in text
    assert 'Level1 evaluation controls were reused' in text
    assert 'frozen recipe bundle' in text and 'generation seed offset is 1,000,000' in text
    assert ('all five domains match' in text) is matched
    assert ('| FAIL |' in text) is (not matched)
    assert '+5.0000' in text


def test_missing_canonical_report_cannot_publish_or_import_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, 'REPORT', tmp_path / 'absent.json')
    monkeypatch.setattr(summary, 'OUTPUT', tmp_path / 'summary.md')
    importer = Mock(side_effect=AssertionError('scientific imports must not occur'))
    monkeypatch.setattr(summary.importlib.util, 'spec_from_file_location', importer)
    with pytest.raises(ValueError, match='canonical confirmation report is absent'):
        summary.publish()
    assert not summary.OUTPUT.exists()
    importer.assert_not_called()


def test_existing_summary_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / 'summary.md'; output.write_text('preserved')
    monkeypatch.setattr(summary, 'OUTPUT', output)
    authenticate = Mock(side_effect=AssertionError('existing file must fail first'))
    monkeypatch.setattr(summary, 'authenticated_inputs', authenticate)
    with pytest.raises(ValueError, match='already exists'): summary.publish()
    assert output.read_text() == 'preserved'
    authenticate.assert_not_called()


def test_report_mutation_between_authentication_and_publication_is_refused(tmp_path, monkeypatch):
    report_path = tmp_path / 'report.json'; report_path.write_text('{}')
    values = list(render_inputs())
    values[-1]['report'] = summary.digest(report_path)
    monkeypatch.setattr(summary, 'REPORT', report_path)
    monkeypatch.setattr(summary, 'OUTPUT', tmp_path / 'summary.md')
    monkeypatch.setattr(summary, 'authenticated_inputs', lambda: values)
    def render(*args):
        report_path.write_text('{"changed":true}')
        return 'must not publish'
    monkeypatch.setattr(summary, 'render_summary', render)
    with pytest.raises(ValueError, match='report changed'): summary.publish()
    assert not summary.OUTPUT.exists()
