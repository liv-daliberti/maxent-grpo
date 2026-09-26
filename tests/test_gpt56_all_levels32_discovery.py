"""Guard broader problem sampling, unchanged native requests, and safe resume."""
from copy import deepcopy
from functools import lru_cache
import json
from types import SimpleNamespace

import pytest

from ops import prepare_gpt56_all_levels32_discovery as stage


@lru_cache(maxsize=1)
def inputs():
    rows = [json.loads(line) for line in (stage.SOURCE / 'rows.jsonl').read_text().splitlines()]
    templates = {stage.identity(r): r for r in
                 (json.loads(line) for line in (stage.SOURCE / 'requests.jsonl').read_text().splitlines())
                 if r['sample_index'] == 0}
    return rows, templates


def test_all_cells_retain_exact_prior_problems_and_add_next_sixteen():
    rows, _ = inputs()
    prior = {stage.identity(r) for r in stage.load_helper().rows(stage.PRIOR_BASE / 'rows.jsonl')}
    retained, added = set(), set()
    for level in stage.LEVELS:
        for domain in stage.DOMAINS:
            old, new, ranked = stage.selected_cell(rows, level, domain)
            old_keys = {stage.identity(r) for r in old}
            new_keys = {stage.identity(r) for r in new}
            assert len(old_keys) == len(new_keys) == 16
            assert old_keys == {key for key in prior if key[:2] == (level, domain)}
            assert old == ranked[:16] and new == ranked[16:32]
            assert old_keys.isdisjoint(new_keys)
            retained.update(old_keys)
            added.update(new_keys)
    assert retained == prior and len(retained) == len(added) == 240
    assert retained.isdisjoint(added) and len(retained | added) == 480
    old_slots = {(*key, draw) for key in retained for draw in range(512)}
    new_slots = {(*key, draw) for key in added for draw in range(512)}
    assert old_slots.isdisjoint(new_slots)
    assert len(old_slots) == len(new_slots) == 122880
    assert len(old_slots | new_slots) == 245760
    assert max(slot[-1] for slot in old_slots | new_slots) == 511


def test_selection_ignores_outcomes_problem_content_and_input_order():
    rows, _ = inputs()
    before = stage.selected_cell(rows, 2, 'pantry_plan')
    changed = deepcopy(rows)
    for row in changed:
        row['problem'] = 'different problem text'
        row['answer'] = 'different answer'
        row['metadata'] = {'verified_modes': -100}
    after = stage.selected_cell(list(reversed(changed)), 2, 'pantry_plan')
    assert [[stage.identity(r) for r in group] for group in before] == [[stage.identity(r) for r in group] for group in after]
    missing = [row for row in rows if stage.identity(row) != (2, 'pantry_plan', 0)]
    duplicate = missing + [next(row for row in rows if stage.identity(row) == (2, 'pantry_plan', 1))]
    for invalid in (missing, duplicate):
        with pytest.raises(ValueError, match='all128'):
            stage.selected_cell(invalid, 2, 'pantry_plan')


@pytest.mark.parametrize('level', stage.LEVELS)
@pytest.mark.parametrize('domain', stage.DOMAINS)
def test_each_new_native_interface_has512_fresh_draws(level, domain):
    rows, templates = inputs()
    _, added, _ = stage.selected_cell(rows, level, domain)
    row = added[0]
    reference = templates[stage.identity(row)]
    expanded = stage.expand([row], templates)
    assert len(expanded) == 512
    assert {r['sample_index'] for r in expanded} == set(range(512))
    assert len({r['sample_id'] for r in expanded}) == 512
    for item in expanded:
        assert item['request'] == reference['request']
        assert item['request_sha256'] == reference['request_sha256']
        assert item['row_sha256'] == reference['row_sha256']
        assert item['reference_sample_id'] == reference['sample_id']
        assert item['choice_index'] == 0 and item['group_id'] == item['sample_id']
        assert item['fresh_response_cohort'] is True
        assert item['experiment_condition'] == stage.CONDITION
        assert item['prompt_arm'] == 'original'
        assert 'output_text' not in item and 'provider_sample_identity' not in item
    assert {k: v for k, v in reference['request'].items() if k not in ('input', 'model')} == stage.PARAMETERS


def test_expansion_rejects_changed_prompt_or_generation_controls():
    rows, templates = inputs()
    _, added, _ = stage.selected_cell(rows, 1, 'graph_coloring')
    row = added[0]
    key = stage.identity(row)
    changed = {key: deepcopy(templates[key])}
    changed[key]['request']['reasoning'] = {'effort': 'low'}
    changed[key]['request_sha256'] = stage.load_helper().object_digest(changed[key]['request'])
    with pytest.raises(ValueError, match='generation controls'):
        stage.expand([row], changed)
    altered_row = deepcopy(row)
    altered_row['problem'] += ' changed'
    with pytest.raises(ValueError, match='Source row identity changed'):
        stage.expand([altered_row], templates)


def resume_fixture(tmp_path, monkeypatch, records):
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'sample_receipts').mkdir()
    (out / 'manifest.json').write_text('{"test": true}\n')
    h = stage.load_helper()
    for record in records:
        (out / 'sample_receipts' / (record['sample_id'] + '.json')).write_text(json.dumps(record))
    def no_credentials(_):
        pytest.fail('Resume attempted credential access or a model request')
    helper = SimpleNamespace(require=h.require, digest=h.digest, write=h.write, now=h.now,
                             expected_returned_controls=lambda: {'model': 'gpt-5.6-sol'}, credential=no_credentials)
    monkeypatch.setattr(stage, 'load_helper', lambda: helper)
    monkeypatch.setattr(stage, 'authenticate_base', lambda *_: {'collection_runs': {'cell': {'run_dir': str(out), 'new_samples': 8192}}})
    monkeypatch.setattr(stage, 'audit_records', lambda _: records)
    return out


def test_preflight_recovers_authenticated_response_without_another_request(tmp_path, monkeypatch):
    records = [{'sample_id': 'L1_graph_coloring_042_0', 'provider_sample_identity': ['native-id', 0]}]
    out = resume_fixture(tmp_path, monkeypatch, records)
    first = stage.collect_one(tmp_path, 'cell', 'preflight')
    assert first['new_calls'] == 0 and first['authenticated_samples'] == 1
    assert json.loads((out / 'preflight.json').read_text())['retained_in_production'] is True
    assert stage.collect_one(tmp_path, 'cell', 'preflight') == first
    with (out / 'sample_receipts' / (records[0]['sample_id'] + '.json')).open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='Changed preflight evidence'):
        stage.collect_one(tmp_path, 'cell', 'full')


def test_production_requires_authenticated_preflight_before_credentials(tmp_path, monkeypatch):
    resume_fixture(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError, match='authenticated preflight before production'):
        stage.collect_one(tmp_path, 'cell', 'full')


def test_existing_complete_pool_does_not_access_credentials(tmp_path, monkeypatch):
    records = [{'sample_id': f'L1_graph_coloring_042_{i}', 'provider_sample_identity': ['native-id', i]} for i in range(2)]
    resume_fixture(tmp_path, monkeypatch, records)
    monkeypatch.setattr(stage, 'authenticate_base', lambda *_: {'collection_runs': {'cell': {'run_dir': str(tmp_path / 'run'), 'new_samples': 2}}})
    result = stage.collect_one(tmp_path, 'cell', 'full')
    assert result['complete'] is True and result['new_calls'] == 0
