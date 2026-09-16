"""Current-source audits must detect later overlaps without hiding external copies."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from datasets import Dataset, DatasetDict, disable_progress_bars
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import modebench_scale_source_disjointness as audit

disable_progress_bars()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def saved(path, rows, subset='multi_answer'):
    DatasetDict({subset: Dataset.from_list(rows)}).save_to_disk(str(path))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'LEVEL1', {'countdown': {}})
    monkeypatch.setattr(audit, 'NATIVE_SOURCE_ROOTS', {})
    search = tmp_path / 'data'; search.mkdir()
    root = search / 'modebench_revisions' / 'r2'
    base = root / 'level4/dataset/countdown'
    own_pools = root / 'level4/pools/countdown'
    rows = {}
    index = 0
    for split, (count, subset) in audit.SPLITS.items():
        rows[split] = []
        for _ in range(count):
            index += 1
            rows[split].append({'problem': 'mechanical fixture ' + str(index),
                'answer': json.dumps({'numbers': [1, 2, 3, index + 10], 'target': index}),
                'answer_mode_count': 2})
        jsonl(base / (split + '.jsonl'), rows[split]); saved(base / split, rows[split], subset)
    protocol = root / 'protocol.json'; recipe = root / 'level4/recipes/countdown.json'
    write(protocol, {'histograms': {'countdown': {s: [{'cell': [2], 'rows': n}]
                                                  for s, (n, _) in audit.SPLITS.items()}}, 'files_sha256': {}})
    write(recipe, {'mechanical_fixture_only': True})
    identity = {'schema': 'modebench_scale_domain_revision_dataset_v1', 'level': 'level4', 'domain': 'countdown',
                'status': 'frozen_pending_heldout_confirmation', 'protocol_sha256': audit.file_sha(protocol),
                'recipe_sha256': audit.file_sha(recipe), 'test_split': 'eval',
                'splits': {s: {'rows': len(r), 'rows_sha256': audit.sha(r)} for s, r in rows.items()}}
    write(base / 'identity.json', identity)
    jsonl(own_pools / 'difficulty_0.jsonl', rows['dev'])
    return SimpleNamespace(search=search, root=root, base=base, own_pools=own_pools, rows=rows, identity=identity,
                           verify=lambda: audit.verify_dataset(root, 'level4', 'countdown', search_root=search))


def test_own_selected_dev_pool_allowed_and_report_deterministic(campaign):
    first = campaign.verify()
    assert first == campaign.verify()
    assert first['split_rows'] == {'train': 384, 'dev': 128, 'eval': 128}
    assert first['sources_audited'] == 1
    own = first['sources'][0]
    assert own['canonical_own_pool'] and own['own_dev_pool_identity_overlaps_allowed'] == 128
    assert own['compared_splits'] == ['train', 'eval']
    assert str(campaign.base / 'eval.jsonl') in first['files_sha256']
    assert str(campaign.base / 'eval/multi_answer/data-00000-of-00001.arrow') in first['files_sha256']


@pytest.mark.parametrize('split', ['train', 'eval'])
def test_own_pool_never_exempts_train_or_test(campaign, split):
    jsonl(campaign.own_pools / 'difficulty_1.jsonl', campaign.rows[split][:1])
    with pytest.raises(ValueError, match='cross-source overlap'):
        campaign.verify()


def test_future_other_root_copy_of_own_dev_is_not_hidden(campaign):
    campaign.verify()
    external = campaign.search / 'modebench_future/level5/pools/countdown/difficulty_0.jsonl'
    jsonl(external, campaign.rows['dev'][:1])
    with pytest.raises(ValueError, match='cross-source overlap') as caught:
        campaign.verify()
    assert str(external) in str(caught.value)


def test_future_arrow_cross_level_dataset_detected(campaign):
    campaign.verify()
    external = campaign.root / 'level5/dataset/countdown/eval'
    saved(external, campaign.rows['eval'][:1])
    with pytest.raises(ValueError, match='cross-source overlap'):
        campaign.verify()


def test_different_question_same_semantic_identity_detected(campaign):
    row = deepcopy(campaign.rows['train'][0]); row['problem'] = 'different exact prompt'
    jsonl(campaign.search / 'modebench_other/pools/countdown/difficulty_0.jsonl', [row])
    with pytest.raises(ValueError, match='semantic_identities'):
        campaign.verify()


def test_same_question_different_semantic_identity_detected(campaign):
    row = deepcopy(campaign.rows['eval'][0]); row['answer'] = json.dumps({'numbers': [9, 8, 7, 6], 'target': 9876})
    jsonl(campaign.search / 'modebench_other/pools/countdown/difficulty_0.jsonl', [row])
    with pytest.raises(ValueError, match='exact_prompts'):
        campaign.verify()


def test_canonical_own_dataset_alias_allowed_but_copy_rejected(campaign):
    alias = campaign.search / 'modebench_alias/level5/dataset/countdown'
    alias.parent.mkdir(parents=True); alias.symlink_to(campaign.base, target_is_directory=True)
    campaign.verify()
    alias.unlink(); shutil.copytree(campaign.base, alias)
    with pytest.raises(ValueError, match='cross-source overlap'):
        campaign.verify()


def test_level1_paths_outside_search_tree_checked(campaign, monkeypatch, tmp_path):
    external = tmp_path / 'native/eval'; saved(external, campaign.rows['train'][:1])
    monkeypatch.setattr(audit, 'LEVEL1', {'countdown': {'eval': external}})
    with pytest.raises(ValueError, match='cross-source overlap'):
        campaign.verify()


def test_native_historical_eval_outside_reserve_paths_checked(campaign, monkeypatch, tmp_path):
    native = tmp_path / 'old_native'; saved(native / 'eval', campaign.rows['eval'][:1])
    monkeypatch.setattr(audit, 'NATIVE_SOURCE_ROOTS', {'countdown': native})
    with pytest.raises(ValueError, match='cross-source overlap'):
        campaign.verify()


@pytest.mark.parametrize('change', ['jsonl', 'arrow', 'identity', 'recipe', 'histogram'])
def test_changed_frozen_input_rejected(campaign, change):
    if change == 'jsonl':
        jsonl(campaign.base / 'eval.jsonl', campaign.rows['eval'][:-1])
    elif change == 'arrow':
        shutil.rmtree(campaign.base / 'eval'); saved(campaign.base / 'eval', campaign.rows['eval'][:-1])
    elif change == 'identity':
        identity = deepcopy(campaign.identity); identity['level'] = 'level5'; write(campaign.base / 'identity.json', identity)
    elif change == 'recipe':
        write(campaign.root / 'level4/recipes/countdown.json', {'changed': True})
    else:
        protocol = audit.read(campaign.root / 'protocol.json')
        protocol['histograms']['countdown']['eval'] = [{'cell': [3], 'rows': 128}]
        write(campaign.root / 'protocol.json', protocol)
        identity = deepcopy(campaign.identity); identity['protocol_sha256'] = audit.file_sha(campaign.root / 'protocol.json')
        write(campaign.base / 'identity.json', identity)
    with pytest.raises(ValueError):
        campaign.verify()


def test_current_cross_split_duplicate_rejected_even_if_record_rehashed(campaign):
    rows = deepcopy(campaign.rows['eval']); rows[0] = deepcopy(campaign.rows['train'][0])
    jsonl(campaign.base / 'eval.jsonl', rows)
    shutil.rmtree(campaign.base / 'eval'); saved(campaign.base / 'eval', rows)
    identity = deepcopy(campaign.identity); identity['splits']['eval']['rows_sha256'] = audit.sha(rows)
    write(campaign.base / 'identity.json', identity)
    with pytest.raises(ValueError, match='frozen splits overlap'):
        campaign.verify()


def test_new_disjoint_source_is_pinned_and_counted(campaign):
    before = campaign.verify()
    row = deepcopy(campaign.rows['eval'][0])
    row.update(problem='new source', answer=json.dumps({'numbers': [11, 12, 13, 14], 'target': 9000}))
    path = campaign.search / 'modebench_future/pools/countdown/difficulty_0.jsonl'; jsonl(path, [row])
    after = campaign.verify()
    assert after['sources_audited'] == before['sources_audited'] + 1
    assert str(path) in after['files_sha256']
    assert after['sources_with_new_files_since_baseline'] == before['sources_with_new_files_since_baseline'] + 1


@pytest.mark.parametrize('overlapping', [False, True])
def test_freeze_exclusion_snapshot_does_not_hide_later_sources(campaign, overlapping):
    snapshot = campaign.root / 'exclusions/freeze.json'
    write(snapshot, {'protocol_sha256': campaign.identity['protocol_sha256'],
                     'files_sha256': {str(campaign.own_pools / 'difficulty_0.jsonl'):
                                      audit.file_sha(campaign.own_pools / 'difficulty_0.jsonl')}})
    identity = deepcopy(campaign.identity)
    identity.update(exclusions=str(snapshot), exclusions_sha256=audit.file_sha(snapshot))
    write(campaign.base / 'identity.json', identity)
    before = campaign.verify()
    assert before['discovery_baseline'] == 'freeze_exclusion_snapshot'
    assert before['sources_with_new_files_since_baseline'] == 0
    row = deepcopy(campaign.rows['dev'][0])
    if not overlapping:
        row.update(problem='new disjoint source', answer=json.dumps({'numbers': [8, 9, 10, 11], 'target': 8888}))
    jsonl(campaign.search / 'modebench_later/pools/countdown/difficulty_0.jsonl', [row])
    if overlapping:
        with pytest.raises(ValueError, match='cross-source overlap'):
            campaign.verify()
    else:
        after = campaign.verify()
        assert after['sources_with_new_files_since_baseline'] == 1
        assert str(snapshot) in after['files_sha256']


def test_concurrent_source_inventory_growth_requires_fresh_audit(campaign, monkeypatch):
    original = audit.discover_sources
    calls = 0
    def changing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            jsonl(campaign.search / 'modebench_concurrent/pools/countdown/difficulty_0.jsonl', campaign.rows['eval'][:1])
        return original(*args, **kwargs)
    monkeypatch.setattr(audit, 'discover_sources', changing)
    with pytest.raises(ValueError, match='source inventory changed'):
        campaign.verify()
