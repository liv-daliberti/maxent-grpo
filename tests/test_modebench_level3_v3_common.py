"""Boundary guards for isolated v3 fixed-reference authentication."""
import json
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_level3_v3_common as common


def test_atomic_publication_never_replaces_existing_evidence(tmp_path):
    path = tmp_path / 'evidence.json'
    common.atomic_new(path, {'kept': True})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        common.atomic_new(path, {'kept': False})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_failed_serialization_publishes_nothing(tmp_path):
    with pytest.raises(ValueError):
        common.atomic_new(tmp_path / 'evidence.json', {'metric': float('nan')})
    assert list(tmp_path.iterdir()) == []


def test_conflicting_inherited_pin_rejected():
    with pytest.raises(ValueError):
        common.merge_pins({'a': 'first'}, {'a': 'second'})
    assert common.merge_pins({'a': 'same'}, {'a': 'same', 'b': 'other'}) == {'a': 'same', 'b': 'other'}


def test_modified_sealed_bytes_rejected(tmp_path):
    path = tmp_path / 'fixed.json'; path.write_text('original')
    files = {str(path): common.digest(path)}
    common.verify_pins(files)
    path.write_text('modified')
    with pytest.raises(ValueError):
        common.verify_pins(files)


def test_new_file_in_frozen_inventory_rejected(tmp_path):
    path = tmp_path / 'fixed.json'; path.write_text('original')
    files = {str(path): common.digest(path)}; trees = {str(tmp_path): [str(path)]}
    common.verify_pins(files, trees)
    (tmp_path / 'unexpected.json').write_text('new')
    with pytest.raises(ValueError):
        common.verify_pins(files, trees)


def test_contract_cannot_mutate_shared_policy():
    record = common.contract()
    record['tolerances']['pass1'] = 1.0
    record['generation_seed_bases']['graph_coloring']['eval'] = 0
    record['revised_domains'].clear()
    assert common.contract()['tolerances'] == {'pass1': .04, 'pass8': .08}
    assert common.contract()['generation_seed_bases']['graph_coloring']['eval'] == 9337100
    assert common.contract()['revised_domains'] == ['graph_coloring', 'python_factors']


def test_union_calibration_covers_eval_only_cells(monkeypatch):
    hist = {'train': Counter({(4,): 384}), 'dev': Counter({(4,): 80, (6,): 48}),
            'eval': Counter({(4,): 66, (6,): 60, (5,): 2})}
    monkeypatch.setattr(common, 'reference_histograms', lambda domain: hist)
    actual = common.calibration_histogram('graph_coloring')
    assert actual == Counter({(4,): 80, (6,): 60, (5,): 2})
    assert sum(actual.values()) == 142


@pytest.mark.parametrize('domain', common.RETAINED)
def test_retained_domains_cannot_be_recalibrated(domain):
    with pytest.raises(ValueError):
        common.calibration_histogram(domain)


@pytest.mark.parametrize('labels', [common.CONF_LABELS, common.REFERENCE_LABELS, (6328000,6328001,6328002,6328003)])
def test_non_development_draws_cannot_enter_candidate_fit(labels):
    with pytest.raises(ValueError):
        common.validate_new_candidate_receipt('must-not-be-opened.json', 'python_factors', labels)


def registration_fixture(tmp_path, monkeypatch):
    old = tmp_path / 'old.txt'; old.write_text('immutable history')
    inherited = {'files_sha256': {str(old): common.digest(old)}, 'directory_files': {}}
    monkeypatch.setattr(common, 'authenticate_inherited', lambda: inherited)
    path = tmp_path / 'registration.json'
    monkeypatch.setattr(common, 'REGISTRATION', path)
    record = {'schema': common.REGISTRATION_SCHEMA, 'contract': common.contract(),
              'inherited_confirmation': common.inherited_metadata(), 'benchmark_targets': common.benchmark_metadata(),
              'candidate_revisions': {domain: {} for domain in common.REVISED},
              'files_sha256': {**inherited['files_sha256'], str(Path(common.__file__).resolve()): common.digest(common.__file__)},
              'directory_files': {}}
    for domain, revision in zip(common.REVISED, ('graph_v8', 'python_v7')):
        descriptor = {'name': revision, 'domain': domain, 'pool_root': str(common.POOL_ROOTS[domain]),
                      'rows_per_tier': common.contract()['calibration_rows_per_tier'][domain],
                      'development_receipts': {str(tier): str(common.RESULTS / f'calibration_3b_{revision}_d{tier}.json')
                                               for tier in range(4)}, 'source_snapshot': inherited}
        for kind, relative in (('generator', f'ops/exp_scaling/modebench_level3_{revision}.py'),
                               ('materializer', f'ops/exp_scaling/materialize_modebench_level3_{revision}.py'),
                               ('tests', f'tests/test_modebench_level3_{revision}.py')):
            source = str(ROOT / relative)
            descriptor[kind + '_path'] = source
            descriptor[kind + '_sha256'] = common.digest(source)
            record['files_sha256'][source] = common.digest(source)
        for key, seed_kind in (('development_seed_base', 'development'), ('capacity_seed_base', 'capacity'),
                               ('final_train_seed_base', 'train'), ('final_eval_seed_base', 'eval')):
            descriptor[key] = common.SEEDS[domain][seed_kind]
        record['candidate_revisions'][domain] = descriptor
    return path, record, old


def write_registration(path, record):
    path.write_text(json.dumps(record, sort_keys=True))
    return common.digest(path)


def test_registration_requires_external_canonical_pin(tmp_path, monkeypatch):
    path, record, _ = registration_fixture(tmp_path, monkeypatch)
    expected = write_registration(path, record)
    assert common.validate_registration(path, expected) == record
    with pytest.raises(ValueError):
        common.validate_registration(path, '0' * 64)


def test_registration_cannot_pin_itself(tmp_path, monkeypatch):
    path, record, _ = registration_fixture(tmp_path, monkeypatch)
    record['files_sha256'][str(path)] = '0' * 64
    expected = write_registration(path, record)
    with pytest.raises(ValueError):
        common.validate_registration(path, expected)


def test_registration_cannot_drop_old_failure_evidence(tmp_path, monkeypatch):
    path, record, old = registration_fixture(tmp_path, monkeypatch)
    record['files_sha256'].pop(str(old))
    expected = write_registration(path, record)
    with pytest.raises(ValueError):
        common.validate_registration(path, expected)


def test_registration_cannot_change_fixed_measured_target(tmp_path, monkeypatch):
    path, record, _ = registration_fixture(tmp_path, monkeypatch)
    record['benchmark_targets']['python_factors']['receipt_sha256'] = '0' * 64
    expected = write_registration(path, record)
    with pytest.raises(ValueError):
        common.validate_registration(path, expected)


@pytest.mark.parametrize('field,value', [
    ('development_draw_labels', [6328000,6328001,6328002,6328003]),
    ('confirmation_draw_labels', [6329000,6329001,6329002,6329003]),
    ('selected_development_sets_scored', 2), ('selected_evaluation_sets_scored', 1), ('selection_seed', 42),
    ('selection_algorithm', 'old_selection_search'), ('weight_objective', 'selected_dev_best'),
    ('all_five_fresh_same_round', True), ('statistical_equivalence_claimed', True),
    ('historical_level3_confirmation_used_for_mixture_fitting', True),
])
def test_registration_cannot_relax_information_boundary(tmp_path, monkeypatch, field, value):
    path, record, _ = registration_fixture(tmp_path, monkeypatch)
    record['contract'][field] = value
    expected = write_registration(path, record)
    with pytest.raises(ValueError):
        common.validate_registration(path, expected)


@pytest.mark.parametrize('field,value', [
    ('name', 'python_v6'), ('pool_root', '/old/pool'), ('rows_per_tier', 128),
    ('generator_sha256', '0' * 64), ('materializer_sha256', '0' * 64), ('tests_sha256', '0' * 64),
    ('development_seed_base', 0), ('final_eval_seed_base', 0),
    ('development_receipts', {}), ('source_snapshot', {}),
])
def test_registration_authenticates_new_revision_descriptor(tmp_path, monkeypatch, field, value):
    path, record, _ = registration_fixture(tmp_path, monkeypatch)
    record['candidate_revisions']['python_factors'][field] = value
    expected = write_registration(path, record)
    with pytest.raises(ValueError):
        common.validate_registration(path, expected)
