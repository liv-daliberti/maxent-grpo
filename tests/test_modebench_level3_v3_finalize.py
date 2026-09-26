"""Guard fresh split generation and byte-identical retained-domain routing."""
from collections import Counter
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_level3_v3_finalize as finalizer
common = finalizer.common


@pytest.mark.parametrize('domain', common.RETAINED)
def test_retained_domains_cannot_enter_fresh_generation(domain):
    with pytest.raises(ValueError):
        finalizer.candidate_modules(domain)
    with pytest.raises(ValueError):
        finalizer.allocation_targets(domain, 'train', {})


@pytest.mark.parametrize('split', ['dev', 'test', 'confirmation'])
def test_only_train_and_eval_receive_new_generation(split):
    with pytest.raises(ValueError):
        finalizer.allocation_targets('graph_coloring', split, {})


def test_copy_retained_domain_keeps_every_byte(tmp_path, monkeypatch):
    old = tmp_path / 'old'; staging = tmp_path / 'staging'; staging.mkdir()
    for split in common.SPLITS:
        folder = old / 'pantry' / split; folder.mkdir(parents=True)
        (folder / 'fixed.arrow').write_bytes(bytes(range(256)))
        (folder / 'state.json').write_text('{"source":"original"}\n')
    monkeypatch.setattr(common, 'OLD_DATASET', old)
    before = finalizer.byte_tree(old / 'pantry')
    assert finalizer.copy_retained_domain('pantry', staging) == before
    assert finalizer.byte_tree(staging / 'pantry') == before
    assert finalizer.byte_tree(old / 'pantry') == before
    with pytest.raises(FileExistsError):
        finalizer.copy_retained_domain('pantry', staging)


def test_revised_domain_cannot_be_passed_off_as_retained(tmp_path):
    with pytest.raises(ValueError):
        finalizer.copy_retained_domain('graph_coloring', tmp_path)


def test_retained_copy_detects_source_change(tmp_path, monkeypatch):
    old = tmp_path / 'old'; source = old / 'countdown'; source.mkdir(parents=True)
    (source / 'rows.arrow').write_bytes(b'original')
    staging = tmp_path / 'staging'; staging.mkdir()
    monkeypatch.setattr(common, 'OLD_DATASET', old)
    original = finalizer.shutil.copytree
    def altered(src, dst):
        result = original(src, dst)
        (src / 'rows.arrow').write_bytes(b'changed concurrently')
        return result
    monkeypatch.setattr(finalizer.shutil, 'copytree', altered)
    with pytest.raises(ValueError):
        finalizer.copy_retained_domain('countdown', staging)


def test_boundary_discloses_only_two_fresh_candidate_domains():
    boundary = finalizer.information_boundary()
    assert boundary['retained_domains'] == ['countdown', 'mathir', 'pantry']
    assert boundary['fresh_candidate_domains'] == ['graph_coloring', 'python_factors']
    assert boundary['all_five_fresh_same_round'] is False
    assert boundary['statistical_equivalence_claimed'] is False
    boundary['fresh_candidate_domains'].clear()
    assert len(finalizer.information_boundary()['fresh_candidate_domains']) == 2


def test_fixed_history_uses_registered_snapshot_not_future_dataset(tmp_path, monkeypatch):
    domain = 'graph_coloring'
    old = tmp_path / 'modebench_level3_old' / domain / 'eval'; old.mkdir(parents=True)
    marker = old / 'dataset_dict.json'; marker.write_text('{}')
    future = tmp_path / 'modebench_level3_matched_v3' / domain / 'eval'; future.mkdir(parents=True)
    (future / 'dataset_dict.json').write_text('{}')
    root = tmp_path / 'new_pools'; pool_dir = root / 'pools' / domain; pool_dir.mkdir(parents=True)
    for tier in range(4): (pool_dir / f'difficulty_{tier}.jsonl').write_text(json.dumps({'id': tier + 10}) + '\n')
    monkeypatch.setattr(common, 'POOL_ROOTS', {domain: root})
    monkeypatch.setattr(finalizer.materializer, 'existing_ids', lambda domain: {1})
    monkeypatch.setattr(finalizer, 'source_rows', lambda path: [{'id': 2}] if path == old else [{'id': 999}])
    monkeypatch.setattr(finalizer.materializer, 'identity_set', lambda domain, rows: {row['id'] for row in rows})
    history_hash = finalizer.materializer.row_hash([1, 2])
    descriptor = {'source_snapshot': {'files_sha256': {str(marker): common.digest(marker)},
        'directory_files': {}, 'candidate_pool_paths': [], 'historical_identity_sha256': history_hash}}
    history, blocked, record = finalizer.fixed_history(domain, descriptor)
    assert history == {1, 2} and blocked == {1, 2, 10, 11, 12, 13} and 999 not in blocked
    assert len(record['candidate_pools']) == 4
    descriptor['source_snapshot']['historical_identity_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        finalizer.fixed_history(domain, descriptor)


def test_allocation_uses_actual_split_histogram_and_frozen_weights(monkeypatch):
    targets = {'dev': Counter({(4,): 128}), 'eval': Counter({(4,): 64, (5,): 64}), 'train': Counter({(4,): 384})}
    monkeypatch.setattr(common, 'reference_histograms', lambda domain: targets)
    recipe = {'weight_units': [10, 10, 0, 0]}
    assigned = finalizer.allocation_targets('graph_coloring', 'eval', recipe)
    assert sum(assigned, Counter()) == targets['eval']
    assert [sum(cells.values()) for cells in assigned] == [64, 64, 0, 0]
    assert all(cells[(5,)] == 32 for cells in assigned[:2])


def test_existing_output_refuses_before_any_model_or_proof_work(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'DATASET', tmp_path)
    with pytest.raises(ValueError):
        finalizer.finalize(registration_sha256='a' * 64, execution_seal_sha256='b' * 64)


def test_arbitrary_execution_seal_is_not_accepted(tmp_path):
    fake = tmp_path / 'seal.json'; fake.write_text('{}')
    with pytest.raises(ValueError):
        finalizer.authenticate_execution_seal(fake, common.digest(fake), 'a' * 64)


def test_readonly_authentication_does_not_accept_an_arbitrary_dataset(tmp_path, monkeypatch):
    path = tmp_path / 'identity.json'; path.write_text('{}')
    bundle = tmp_path / 'frozen_recipes.json'; bundle.write_text('{}')
    monkeypatch.setattr(finalizer, 'IDENTITY', path); monkeypatch.setattr(finalizer, 'BUNDLE', bundle)
    monkeypatch.setattr(common, 'validate_registration', lambda *args: {})
    with pytest.raises((ValueError, KeyError)):
        finalizer.authenticate_dataset(registration_sha256='a' * 64)


def test_fresh_generation_uses_registered_tier_seed_and_excludes_prior_rows(monkeypatch):
    import modebench_level3_graph_v8 as generator
    from types import SimpleNamespace
    domain = 'graph_coloring'; calls = []
    actual = generator.build_pool
    def record(domain, target, excluded, seed, tag, difficulty, multiplier=1):
        calls.append((seed, tag, difficulty, set(excluded)))
        return actual(domain, target, excluded, seed, tag, difficulty, multiplier)
    monkeypatch.setattr(finalizer, 'candidate_modules', lambda domain: (
        SimpleNamespace(build_pool=record), SimpleNamespace(verify_witnesses=lambda rows: len(rows) * 4)))
    target = Counter({(4,): 4})
    monkeypatch.setattr(common, 'SPLITS', {'eval': 4})
    monkeypatch.setattr(common, 'reference_histograms', lambda domain: {'eval': target})
    reference = [{'modebench_task': domain, 'answer': json.dumps({'verifier': domain}), 'answer_mode_count': 4} for _ in range(4)]
    monkeypatch.setattr(finalizer.materializer, 'reference_rows', lambda domain, split: reference)
    recipe = {'weight_units': [10, 10, 0, 0]}
    first, checks, provenance = finalizer.build_fresh_split(domain, 'eval', recipe, set(), witnesses=False)
    assert [call[0] for call in calls] == [9337100, 9338100]
    assert all(call[1] == 'level3_v3_eval' for call in calls)
    assert provenance['tier_seeds'] == {'0': 9337100, '1': 9338100, '2': 9339100, '3': 9340100}
    assert all(checks.values()) and len(first) == 4
    excluded = finalizer.materializer.identity_set(domain, first)
    second, _, _ = finalizer.build_fresh_split(domain, 'eval', recipe, excluded, witnesses=False)
    assert not finalizer.materializer.identity_set(domain, second) & excluded
    repeated, _, _ = finalizer.build_fresh_split(domain, 'eval', recipe, set(), witnesses=False)
    assert repeated == first
