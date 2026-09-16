"""Exercise finite-pool estimands and authenticated response bookkeeping."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ops import analyze_gpt56_all_domain_discovery as analysis


@pytest.fixture(autouse=True)
def native_auditor_boundary(monkeypatch):
    """Mock only the already-tested frozen native adapter, keeping file auth real.

    The analyzer must call that boundary before accepting a saved sample. Its
    full Responses protocol is exercised by the repository's native-runner tests.
    """
    def validate_completed(directory, item, group, sample, cache):
        receipt = analysis.read_json(directory / sample['raw_receipt'])
        body = receipt['body']
        text = ''.join(part['text'] for output in body['output']
                       for part in output.get('content', []) if part['type'] == 'output_text')
        analysis.require(text == sample['text'], 'Native text differs from saved sample')
        analysis.require(sample['provider_sample_identity'] == [body['id'], 0],
                         'Invalid native choice identity')
    monkeypatch.setattr(analysis, 'load_native_auditor',
                        lambda directory: SimpleNamespace(validate_completed=validate_completed))


@pytest.mark.parametrize('pool', [[], [None], ['a', 'a', 'b', None, None], ['a', 'b', 'c']])
def test_rarefaction_matches_enumerating_every_subset(pool):
    if not pool:
        with pytest.raises(ValueError, match='Invalid finite-pool'):
            analysis.rarefaction([], 0, 0)
        return
    counts = list(Counter(x for x in pool if x is not None).values())
    for k in range(len(pool) + 1):
        subsets = [tuple(pool[i] for i in indices) for indices in combinations(range(len(pool)), k)]
        distinct = [len({x for x in subset if x is not None}) for subset in subsets]
        passed = [int(any(x is not None for x in subset)) for subset in subsets]
        result = analysis.rarefaction(counts, len(pool), k)
        assert result['distinct'] == pytest.approx(np.mean(distinct))
        assert result['pass'] == pytest.approx(np.mean(passed))
        assert result['breadth'] == pytest.approx(np.mean(distinct) - np.mean(passed))


@pytest.mark.parametrize('counts,total,k', [([2, 2], 3, 2), ([0], 4, 1), ([-1], 4, 1),
                                           ([1.0], 4, 1), ([1], 64, 128)])
def test_rarefaction_rejects_invalid_or_extrapolated_pools(counts, total, k):
    with pytest.raises(ValueError):
        analysis.rarefaction(counts, total, k)


def prompts(total):
    return [{'level': 2, 'domain': 'graph_coloring', 'row_index': i,
             'row_sha256': f'row{i}',
             'support': {'support_count': 4, 'support_kind': 'exact'},
             'keys': ([None] * total if i < 8 else ['a'] * (total - 1) + ['b'])}
            for i in range(16)]


@pytest.mark.parametrize('total', [64, 128])
def test_cell_retains_failures_and_bootstraps_paired_tail(total, monkeypatch):
    monkeypatch.setattr(analysis, 'REPLICATES', 300)
    result = analysis.summarize_cell(prompts(total), 2, 'graph_coloring')
    endpoint = result['points'][-1]
    assert endpoint['k'] == total
    assert endpoint['pass']['estimate'] == .5
    assert endpoint['distinct']['estimate'] == 1
    assert endpoint['breadth']['estimate'] == .5
    assert result['tail']['distinct']['estimate'] == pytest.approx(.25)
    assert result['tail']['pass']['estimate'] == 0
    # For half the prompts the final singleton contributes .5 of a mode; for
    # the all-failure half it contributes zero. The same prompt bootstrap must
    # be applied to that paired difference rather than independent endpoints.
    seed = result['bootstrap']['seed']
    indices = np.random.default_rng(seed).integers(0, 16, size=(300, 16))
    expected = np.asarray([0.] * 8 + [.5] * 8)[indices].mean(axis=1)
    assert result['tail']['distinct']['ci95'] == pytest.approx(np.quantile(expected, [.025, .975]))
    assert [p['correct_draws'] for p in result['prompts'][:8]] == [0] * 8
    assert [p['prefix'][str(total)]['distinct'] for p in result['prompts'][:8]] == [0] * 8


def test_cell_rejects_missing_prompts_or_partial_extension():
    with pytest.raises(ValueError, match='16 fixed prompts'):
        analysis.summarize_cell(prompts(64)[:-1], 2, 'graph_coloring')
    rows = prompts(128)
    rows[-1]['keys'].pop()
    with pytest.raises(ValueError, match='Incomplete or unequal'):
        analysis.summarize_cell(rows, 2, 'graph_coloring')


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def write_jsonl(path, values):
    path.write_text(''.join(json.dumps(v, sort_keys=True) + '\n' for v in values))


def refresh_audit(directory):
    write_json(directory / 'all_domain_grading_audit.json', {
        'status': 'complete', 'samples': analysis.binding(directory / 'samples.jsonl'),
        'grades': analysis.binding(directory / 'discovery_hosted_grades.jsonl'),
        'responses': 2, 'api_calls': 0,
        'grader': analysis.binding(directory / 'code/ops/frontier_modebench_contract.py'),
        'normalizer': analysis.binding(directory / 'code/ops/frontier_modebench_normalization.py')})


def make_run(directory, *, provider_prefix='provider', duplicate_provider=False, problem='Color graph'):
    directory.mkdir(parents=True)
    (directory / 'raw').mkdir()
    (directory / 'code/ops').mkdir(parents=True)
    for name in ('frontier_modebench_contract.py', 'frontier_modebench_normalization.py'):
        (directory / 'code/ops' / name).write_text('# fixture frozen source\n')
    row = {'level': 2, 'domain': 'graph_coloring', 'row_index': 0,
           'problem': problem,
           'answer': json.dumps({'verifier': 'graph_coloring', 'n': 2, 'edges': [[1, 2]]})}
    request = {'model': 'gpt-5.6-sol', 'reasoning': {'effort': 'medium'},
               'max_output_tokens': 8192, 'store': False,
               'input': [{'role': 'user', 'content': problem}]}
    requests, samples, grades, groups = [], [], [], []
    for i, text in enumerate(['\\boxed{12}', 'invalid']):
        sample_id = f'L2_graph_coloring_000_{i}'
        response_id = f'{provider_prefix}_{0 if duplicate_provider else i}'
        native = {'id': response_id, 'model': 'gpt-5.6-sol', 'status': 'completed',
                  'reasoning': {'effort': 'medium'}, 'temperature': 1., 'top_p': .98,
                  'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                              'content': [{'type': 'output_text', 'text': text}]}]}
        receipt = {'request_sha256': analysis.object_sha(request), 'sample_ids': [sample_id],
                   'body': native, 'response': native, 'http_body_text': json.dumps(native),
                   'http_status': 200, 'headers': {'x-ms-served-model': 'gpt-5.6-sol-2026-07-09'}}
        path = directory / f'raw/{i}.json'
        write_json(path, receipt)
        item = {key: row[key] for key in ('level', 'domain', 'row_index')}
        item.update(sample_index=i, sample_id=sample_id, group_id=sample_id, request=request,
                    request_sha256=analysis.object_sha(request), row_sha256=analysis.object_sha(row))
        requests.append(item)
        groups.append({'group_id': sample_id})
        strict = {'verified': i == 0, 'canonical_key': 'graph_coloring:12' if i == 0 else None}
        sample = {k: item[k] for k in ('level', 'domain', 'row_index', 'sample_index',
                                      'sample_id', 'request_sha256', 'row_sha256')}
        sample.update(text=text, **strict, model='gpt-5.6-sol',
                      provider_sample_identity=[response_id, 0], response_id=response_id,
                      choice_index=0, raw_receipt=f'raw/{i}.json', raw_receipt_sha256=analysis.object_sha(receipt),
                      response_status='completed', stop_reason='completed', incomplete_details=None,
                      reasoning={'effort': 'medium'}, temperature=1., top_p=.98)
        samples.append(sample)
        grades.append({k: sample[k] for k in ('level', 'domain', 'row_index', 'sample_index')})
        grades[-1].update(strict=strict, normalization=dict(strict), raw_sample_sha256=analysis.object_sha(sample))
    write_jsonl(directory / 'http_requests.jsonl', groups)
    write_jsonl(directory / 'rows.jsonl', [row])
    write_jsonl(directory / 'requests.jsonl', requests)
    write_jsonl(directory / 'samples.jsonl', samples)
    write_jsonl(directory / 'discovery_hosted_grades.jsonl', grades)
    write_json(directory / 'manifest.json', {
        'model': 'gpt-5.6-sol', 'request_count': 2,
        'artifact_sha256': {name: analysis.sha(directory / name) for name in ('rows.jsonl', 'requests.jsonl', 'http_requests.jsonl')},
        'code_sha256': {f'ops/{name}': analysis.sha(directory / 'code/ops' / name)
                       for name in ('frontier_modebench_contract.py', 'frontier_modebench_normalization.py')}})
    refresh_audit(directory)
    return directory


def test_authenticated_run_retains_invalid_response_slot(tmp_path):
    pools = {}
    result = analysis.load_run(make_run(tmp_path / 'run'), pools, set())
    assert result['responses'] == 2
    assert pools[2, 'graph_coloring', 0]['strict'] == {0: 'graph_coloring:12', 1: None}


def test_incomplete_run_fails_closed(tmp_path):
    directory = make_run(tmp_path / 'run')
    samples = analysis.read_jsonl(directory / 'samples.jsonl')
    write_jsonl(directory / 'samples.jsonl', samples[:1])
    refresh_audit(directory)
    with pytest.raises(ValueError):
        analysis.load_run(directory, {}, set())


def test_changed_immutable_problem_fails_closed(tmp_path):
    directory = make_run(tmp_path / 'run')
    rows = analysis.read_jsonl(directory / 'rows.jsonl')
    rows[0]['problem'] = 'different problem'
    write_jsonl(directory / 'rows.jsonl', rows)
    with pytest.raises(ValueError, match='Changed immutable input'):
        analysis.load_run(directory, {}, set())


def test_duplicate_native_response_fails_closed(tmp_path):
    directory = make_run(tmp_path / 'run', duplicate_provider=True)
    with pytest.raises(ValueError, match='Repeated provider'):
        analysis.load_run(directory, {}, set())


def test_overlapping_draw_slots_cannot_extend_pool(tmp_path):
    pools, providers = {}, set()
    analysis.load_run(make_run(tmp_path / 'first'), pools, providers)
    with pytest.raises(ValueError, match='Overlapping sample indices'):
        analysis.load_run(make_run(tmp_path / 'second', provider_prefix='new'), pools, providers)


def test_problem_revision_cannot_be_mixed_across_runs(tmp_path):
    pools, providers = {}, set()
    analysis.load_run(make_run(tmp_path / 'first'), pools, providers)
    with pytest.raises(ValueError, match='Cannot combine different prompts'):
        analysis.load_run(make_run(tmp_path / 'second', provider_prefix='new', problem='New instructions'), pools, providers)


def test_self_consistent_sample_and_grade_tamper_still_disagrees_with_native_receipt(tmp_path):
    directory = make_run(tmp_path / 'run')
    samples = analysis.read_jsonl(directory / 'samples.jsonl')
    grades = analysis.read_jsonl(directory / 'discovery_hosted_grades.jsonl')
    samples[0]['text'] = '\\boxed{21}'
    samples[0]['canonical_key'] = 'graph_coloring:21'
    for grading in ('strict', 'normalization'):
        grades[0][grading]['canonical_key'] = 'graph_coloring:21'
    grades[0]['raw_sample_sha256'] = analysis.object_sha(samples[0])
    write_jsonl(directory / 'samples.jsonl', samples)
    write_jsonl(directory / 'discovery_hosted_grades.jsonl', grades)
    refresh_audit(directory)
    with pytest.raises(ValueError):
        analysis.load_run(directory, {}, set())


def test_grade_cache_tamper_is_rejected_even_when_sample_is_unchanged(tmp_path):
    directory = make_run(tmp_path / 'run')
    grades = analysis.read_jsonl(directory / 'discovery_hosted_grades.jsonl')
    grades[0]['normalization']['canonical_key'] = 'graph_coloring:21'
    write_jsonl(directory / 'discovery_hosted_grades.jsonl', grades)
    with pytest.raises(ValueError):
        analysis.load_run(directory, {}, set())
