"""Offline invariant checks for the paired, system-only ModeBench ablation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import prepare_modebench_prompt_ablation as prep


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp('ablation_source')
    rows, requests = [], []
    profiles = {}
    for level in (1, 2, 3):
        for domain in ('graph_coloring', 'countdown', *prep.DOMAINS):
            profiles[f'{level}/{domain}'] = {'level': level, 'domain': domain}
            for index in range(128):
                row = {'level': level, 'domain': domain, 'row_index': index,
                       'cell_id': f'level{level}/{domain}',
                       'problem': f'Public problem {index}; keep this exact text, including seeds or oats.',
                       'answer': json.dumps({'verifier': domain, 'private_spec': index}),
                       'metadata': {'answer_mode_count': index + 1}}
                rows.append(row)
                messages = [{'role': 'system', 'content': prep.ORIGINAL_SYSTEMS.get(domain, 'Box the answer.')},
                            {'role': 'user', 'content': row['problem']}]
                payload = {'model': 'gpt-5.6-sol', 'input': messages, 'max_output_tokens': 8192,
                           'reasoning': {'effort': 'medium'}, 'store': False}
                for draw in range(8):
                    requests.append({'level': level, 'domain': domain, 'row_index': index,
                                     'sample_id': prep.pair_id(row) + '_' + str(draw),
                                     'sample_index': draw, 'row_sha256': prep.sha(row),
                                     'request': payload, 'request_sha256': prep.sha(payload)})
    prep.write_jsonl(directory / 'rows.jsonl', rows)
    prep.write_jsonl(directory / 'requests.jsonl', requests)
    prep.write_json(directory / 'datasets.json', {'cells': list(profiles.values())})
    code_names = ['ops/frontier_modebench_contract.py', 'src/oat_drgrpo/math_grader.py',
                  'src/oat_drgrpo/templates.py']
    for name in code_names:
        target = directory / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# Frozen source fixture\n')
    prep.write_json(directory / 'manifest.json', {
        'model': 'gpt-5.6-sol', 'endpoint': 'https://example.invalid/responses',
        'max_output_tokens': 8192, 'reasoning_effort': 'medium', 'temperature': None,
        'top_p': None, 'seed': None, 'training': False, 'tools': [], 'conversation_state': False,
        'profiles': profiles,
        'artifact_sha256': {name: prep.file_sha(directory / name)
                            for name in ('rows.jsonl', 'requests.jsonl', 'datasets.json')},
        'code_sha256': {name: prep.file_sha(directory / 'code' / name) for name in code_names},
    })
    return directory


@pytest.fixture(scope='module')
def prepared(source, tmp_path_factory):
    directory = tmp_path_factory.mktemp('ablation_output')
    manifest = prep.prepare(directory, source)
    return directory, manifest


@pytest.mark.parametrize('domain,required,forbidden', [
    ('python_factors', ['Construct one allowed lambda expression.', 'Output exactly the boxed lambda.'],
     ['Test small divisors', 'You may instead dispatch']),
    ('mathir', ['Match the operations to the shuffled menu IDs.'],
     ['Use algebraic isolation:', 'right-side x term', 'combined coefficient', 'those operations']),
    ('pantry_plan', ['Choose stepped amounts, check every bound, and output 2 to 4 ingredient_id=grams pairs.'],
     ['Prefer allowed', 'especially seeds or oats']),
])
def test_only_strategy_guidance_changes(domain, required, forbidden):
    original = [{'role': 'system', 'content': prep.ORIGINAL_SYSTEMS[domain]},
                {'role': 'user', 'content': 'All user wording is retained: ' + prep.HINTS[domain]}]
    before = deepcopy(original)
    result = prep.neutral_messages(2, domain, original)
    assert original == before
    assert result[1] == original[1]
    assert result[0]['content'].startswith(prep.COMMON)
    assert all(text in result[0]['content'] for text in required)
    assert all(text not in result[0]['content'] for text in forbidden)
    assert [message['role'] for message in result] == ['system', 'user']


def test_transform_rejects_unknown_systems_and_unsupported_cells():
    messages = [{'role': 'system', 'content': prep.ORIGINAL_SYSTEMS['mathir']},
                {'role': 'user', 'content': 'Problem'}]
    with pytest.raises(ValueError, match='six predeclared'):
        prep.neutral_messages(1, 'mathir', messages)
    messages[0]['content'] += ' Extra instruction.'
    with pytest.raises(ValueError, match='Unrecognized frozen'):
        prep.neutral_messages(2, 'mathir', messages)
    messages[0]['content'] = prep.ORIGINAL_SYSTEMS['mathir']
    messages.append({'role': 'assistant', 'content': 'Answer'})
    with pytest.raises(ValueError, match='exactly the frozen'):
        prep.neutral_messages(2, 'mathir', messages)


def test_selection_is_outcome_and_input_order_independent(source):
    rows = prep.read_jsonl(source / 'rows.jsonl')
    expected, _ = prep.select_rows(rows)
    mutated = deepcopy(list(reversed(rows)))
    for row in mutated:
        row['answer'] = 'Changed private spec'
        row['metadata'] = {'answer_mode_count': 100000, 'success_rate': 0.99}
        row['verified'] = True
    actual, ledger = prep.select_rows(mutated)
    assert [prep.identity(row) for row in actual] == [prep.identity(row) for row in expected]
    assert len(actual) == 192
    for level, domain in prep.CELLS:
        candidates = [record for record in ledger if (record['level'], record['domain']) == (level, domain)]
        assert len(candidates) == 128
        assert [r['rank_in_cell'] for r in candidates if r['selected']] == list(range(1, 33))
        assert [r['selection_sha256'] for r in candidates] == sorted(r['selection_sha256'] for r in candidates)


def test_selection_rejects_missing_and_duplicate_candidate_identities(source):
    rows = prep.read_jsonl(source / 'rows.jsonl')
    rows = [row for row in rows if prep.identity(row) != (2, 'python_factors', 0)]
    with pytest.raises(ValueError, match='128 rows'):
        prep.select_rows(rows)
    with pytest.raises(ValueError, match='Duplicate source row'):
        prep.select_rows(rows + [rows[-1]])


def test_frozen_pairs_preserve_problem_specs_and_every_nonprompt_request_field(source, prepared):
    output, manifest = prepared
    source_rows = {prep.identity(row): row for row in prep.read_jsonl(source / 'rows.jsonl')}
    source_requests = {r['sample_id']: r for r in prep.read_jsonl(source / 'requests.jsonl')}
    chosen = prep.read_jsonl(output / 'rows.jsonl')
    assert len(chosen) == 192
    for row in chosen:
        assert row == source_rows[prep.identity(row)]
    arm_requests = {}
    for arm in prep.ARMS:
        assert (output / arm / 'rows.jsonl').read_bytes() == (output / 'rows.jsonl').read_bytes()
        records = prep.read_jsonl(output / arm / 'requests.jsonl')
        assert len(records) == 1536
        arm_requests[arm] = {r['sample_id']: r for r in records}
        for record in records:
            source_item = source_requests[record['sample_id']]
            for key in ('domain', 'level', 'row_index', 'sample_index', 'sample_id', 'row_sha256'):
                assert record[key] == source_item[key]
            assert record['request_sha256'] == prep.sha(record['request'])
            assert record['reference_request_sha256'] == source_item['request_sha256']
            assert record['request']['input'][1] == source_item['request']['input'][1]
            assert {k: v for k, v in record['request'].items() if k != 'input'} == {
                k: v for k, v in source_item['request'].items() if k != 'input'}
            if arm == 'original':
                assert record['request'] == source_item['request']
            else:
                assert record['request']['input'][0] != source_item['request']['input'][0]
    assert arm_requests['original'].keys() == arm_requests['neutral'].keys()
    assert manifest['fresh_both_arms'] is True and manifest['outcomes_read'] is False
    assert not list(output.rglob('samples.jsonl'))
    prep.verify_snapshot(output, manifest)


def test_prompt_ledger_matches_requests_and_schedule_is_complete_counterbalanced(prepared):
    output, _ = prepared
    prompts = prep.read_jsonl(output / 'prompts.jsonl')
    assert len(prompts) == 384
    lookup = {(p['arm'], p['pair_id']): p for p in prompts}
    assert len(lookup) == 384
    for prompt in prompts:
        assert prompt['messages_sha256'] == prep.sha(prompt['messages'])
        assert prompt['problem_sha256'] == prep.text_sha(prompt['messages'][1]['content'])
    for arm in prep.ARMS:
        for request in prep.read_jsonl(output / arm / 'requests.jsonl'):
            assert request['request']['input'] == lookup[arm, request['pair_id']]['messages']
    schedule = prep.read_jsonl(output / 'execution_order.jsonl')
    assert len(schedule) == len({(r['arm'], r['sample_id']) for r in schedule}) == 3072
    for pair in {p['pair_id'] for p in prompts}:
        entries = [r for r in schedule if r['pair_id'] == pair]
        assert len(entries) == 16
        assert sum(r['arm'] == 'original' for r in entries[::2]) == 4
        for offset in range(0, 16, 2):
            assert {r['arm'] for r in entries[offset:offset + 2]} == set(prep.ARMS)
            assert len({r['sample_id'] for r in entries[offset:offset + 2]}) == 1


def test_preparation_is_idempotent_without_reading_outcomes(source, prepared, monkeypatch):
    output, manifest = prepared
    old_read = Path.read_text
    def restricted_read(path, *args, **kwargs):
        assert path.name not in {'samples.jsonl', 'summary.json', 'normalized_samples.jsonl'}
        return old_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', restricted_read)
    assert prep.prepare(output, source) == manifest


def test_snapshot_rejects_tampered_prompt_artifact(tmp_path):
    (tmp_path / 'prompts.jsonl').write_text('original\n')
    manifest = {'artifact_sha256': {'prompts.jsonl': prep.file_sha(tmp_path / 'prompts.jsonl')}, 'code_sha256': {}}
    (tmp_path / 'prompts.jsonl').write_text('tampered\n')
    with pytest.raises(ValueError, match='Frozen artifact changed: prompts.jsonl'):
        prep.verify_snapshot(tmp_path, manifest)
