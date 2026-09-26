"""Adversarial accounting, real native receipt, and execution-lock checks."""
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from ops import audit_gpt56_all_levels32_collection as audit
from ops import prepare_gpt56_all_levels512_discovery as old_collector


@pytest.fixture(scope='module')
def inventory():
    rows = [{'level': level, 'domain': domain, 'row_index': index}
            for level in audit.LEVELS for domain in audit.DOMAINS for index in range(32)]
    records = [dict(row, sample_index=draw,
                    provider_sample_identity=[f"resp_{row['level']}/{row['domain']}/{row['row_index']}/{draw}", 0])
               for row in rows for draw in range(512)]
    old_rows = [row for row in rows if row['row_index'] < 16]
    new_rows = [row for row in rows if row['row_index'] >= 16]
    old = [row for row in records if row['row_index'] < 16]
    new = [row for row in records if row['row_index'] >= 16]
    return rows, records, old_rows, new_rows, old, new


def test_complete_32_problem_expansion(inventory):
    rows, records, old_rows, new_rows, old, new = inventory
    result = audit.check_expansion(old, new, old_rows, new_rows, rows)
    assert result['total_authenticated_responses'] == result['unique_provider_samples'] == 245760
    assert result['unique_prompts'] == 480 and result['prompts_per_cell'] == 32
    assert len(result['cells']) == 15 and set(result['cells'].values()) == {16384}
    assert result['retained_and_new_problem_inventories_disjoint'] is True


@pytest.mark.parametrize('corruption', ['provider_reuse', 'duplicate_slot', 'outside_draw',
                                      'unexpected_problem', 'wrong_provider_choice', 'boolean_draw'])
def test_inventory_rejects_identity_corruption(inventory, corruption):
    rows, records, *_ = inventory
    changed = dict(records[-1])
    if corruption == 'provider_reuse':
        changed['provider_sample_identity'] = records[0]['provider_sample_identity']
    elif corruption == 'duplicate_slot':
        changed['sample_index'] = 510
    elif corruption == 'outside_draw':
        changed['sample_index'] = 512
    elif corruption == 'unexpected_problem':
        changed['row_index'] = 99
    elif corruption == 'wrong_provider_choice':
        changed['provider_sample_identity'] = ['a-different-response', 1]
    else:
        changed['sample_index'] = True
    with pytest.raises(ValueError):
        audit.check_inventory([*records[:-1], changed], rows)


def test_missing_draw_and_duplicate_selection_are_rejected(inventory):
    rows, records, *_ = inventory
    with pytest.raises(ValueError, match='inventory'):
        audit.check_inventory(records[:-1], rows)
    with pytest.raises(ValueError, match='Duplicate problem row'):
        audit.check_inventory(records, [*rows, rows[0]])


def test_cell_names_and_counts_are_fixed(inventory):
    rows, records, *_ = inventory
    changed = [dict(row, domain='unknown') if row['domain'] == audit.DOMAINS[0] else row for row in rows]
    with pytest.raises(ValueError, match='fixed domain'):
        audit.check_inventory(records, changed)


def test_retained_and_new_problems_cannot_be_substituted(inventory):
    rows, records, old_rows, new_rows, old, new = inventory
    overlap = [dict(row, row_index=row['row_index'] - 16) for row in new_rows]
    with pytest.raises(ValueError, match='overlap'):
        audit.check_expansion(old, new, old_rows, overlap, rows)
    with pytest.raises(ValueError, match='reassigned'):
        audit.check_expansion(new, old, old_rows, new_rows, rows)
    altered = [dict(old_rows[0], problem='changed bytes'), *old_rows[1:]]
    with pytest.raises(ValueError, match='revision'):
        audit.check_expansion(old, new, altered, new_rows, rows)


@pytest.fixture(scope='module')
def authentic_fixture():
    h = old_collector.load_helper()
    out = old_collector.BASE / 'hosted/gpt56sol/L1_graph_coloring'
    record = json.loads(next((out / 'sample_receipts').glob('*.json')).read_text())
    request = next(row for row in h.rows(out / 'requests.jsonl') if row['sample_id'] == record['sample_id'])
    group = next(row for row in h.rows(out / 'http_requests.jsonl') if row['group_id'] == record['group_id'])
    raw = json.loads((out / record['raw_receipt']).read_text())
    return h, out, record, request, group, raw


@pytest.fixture

def native_run(tmp_path, authentic_fixture):
    h, source, record, request, group, raw = authentic_fixture
    for directory in ('sample_receipts', 'raw_responses', 'code/ops'):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'code/ops/evaluate_native_prompt_ablation.py',
                 tmp_path / 'code/ops/evaluate_native_prompt_ablation.py')
    h.write_rows(tmp_path / 'requests.jsonl', [request])
    h.write_rows(tmp_path / 'http_requests.jsonl', [group])
    h.write_rows(tmp_path / 'samples.jsonl', [record])
    h.write(tmp_path / 'sample_receipts' / (record['sample_id'] + '.json'), record)
    h.write(tmp_path / record['raw_receipt'], raw)
    h.write(tmp_path / 'manifest.json', {
        'artifact_sha256': {name: h.digest(tmp_path / name)
                           for name in ('requests.jsonl', 'http_requests.jsonl')},
        'code_sha256': {'ops/evaluate_native_prompt_ablation.py':
                       h.digest(tmp_path / 'code/ops/evaluate_native_prompt_ablation.py')}})
    return h, tmp_path, deepcopy(record), deepcopy(raw)


def rewrite(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def test_real_frozen_native_receipt_authenticates(native_run):
    h, out, record, _ = native_run
    assert audit.authenticate_saved_run(h, out, 1) == [record]


@pytest.mark.parametrize('corruption', ['snapshot', 'controls', 'raw_digest', 'native_identity', 'sample_text'])
def test_real_native_receipt_tampering_is_rejected(native_run, corruption):
    h, out, record, raw = native_run
    adapter = h.load_adapter(out)
    if corruption == 'snapshot':
        for key in list(raw['headers']):
            if key.lower() == 'x-ms-served-model':
                raw['headers'][key] = 'gpt-5.6-sol-another-snapshot'
        record['raw_receipt_sha256'] = adapter.sha(raw)
    elif corruption == 'controls':
        raw['response']['max_output_tokens'] = 4096
        record['raw_receipt_sha256'] = adapter.sha(raw)
    elif corruption == 'raw_digest':
        raw['latency_seconds'] += 1
    elif corruption == 'native_identity':
        record['provider_sample_identity'] = ['substituted-provider-id', 0]
    else:
        record['text'] += ' substituted text'
    rewrite(out / record['raw_receipt'], raw)
    rewrite(out / 'sample_receipts' / (record['sample_id'] + '.json'), record)
    with pytest.raises(ValueError):
        audit.authenticate_receipts(h, out)


def test_saved_samples_must_equal_native_receipts(native_run):
    h, out, record, _ = native_run
    rewrite(out / 'samples.jsonl', dict(record, text='altered only in the saved aggregate'))
    with pytest.raises(ValueError, match='Saved samples differ'):
        audit.authenticate_saved_run(h, out, 1)


def test_frozen_adapter_change_is_rejected(native_run):
    h, out, record, _ = native_run
    path = out / 'code/ops/evaluate_native_prompt_ablation.py'
    path.write_text(path.read_text() + '\n# changed verifier source\n')
    with pytest.raises(ValueError, match='Changed frozen code'):
        audit.authenticate_saved_run(h, out, 1)


def test_duplicate_native_request_is_rejected(native_run):
    h, out, record, _ = native_run
    path = out / 'requests.jsonl'
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match='Duplicate request'):
        audit.authenticate_receipts(h, out)


@pytest.mark.parametrize('lock_name', ['.production_scheduler.lock', '.all_levels_orchestrator.lock', '.runner.lock'])
def test_active_execution_lock_prevents_completion(tmp_path, monkeypatch, lock_name):
    entries = {}
    (tmp_path / '.production_scheduler.lock').touch()
    for level in audit.LEVELS:
        for domain in audit.DOMAINS:
            name = f'L{level}_{domain}'
            out = tmp_path / name
            out.mkdir()
            for lock in ('.all_levels_orchestrator.lock', '.runner.lock'):
                (out / lock).touch()
            entries[name] = {'run_dir': str(out)}
    collector = SimpleNamespace(load_helper=lambda: SimpleNamespace(), SNAPSHOT=audit.SNAPSHOT,
                                authenticate_base=lambda base: {'collection_runs': entries})
    monkeypatch.setattr(audit, 'load_collector', lambda: collector)
    path = (tmp_path if lock_name == '.production_scheduler.lock'
            else Path(next(iter(entries.values()))['run_dir'])) / lock_name
    with path.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            audit.audit(tmp_path)
    assert not (tmp_path / 'collection_completion_audit.json').exists()


def test_retained_lock_is_not_bypassed(tmp_path):
    path = tmp_path / '.runner.lock'
    with path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ExitStack() as stack, pytest.raises(BlockingIOError):
            audit.lock_file(stack, path)


def test_relative_and_absolute_bindings_refer_to_same_frozen_file():
    item = {'path': 'ops/audit_gpt56_all_levels32_collection.py', 'sha256': 'same-hash'}
    absolute = {'path': str(audit.ROOT / item['path']), 'sha256': 'same-hash'}
    assert audit.same_binding(item, absolute)
    assert not audit.same_binding(item, absolute | {'sha256': 'changed'})


@pytest.mark.parametrize('change', [{'complete': False}, {'completed_samples': 8191},
                                   {'failed_groups_this_session': 1}])
def test_incomplete_native_status_is_never_attested(tmp_path, change):
    state = {'complete': True, 'completed_samples': 8192, 'failed_groups_this_session': 0} | change
    rewrite(tmp_path / 'status.json', state)
    collector = SimpleNamespace(load_helper=lambda: SimpleNamespace())
    with pytest.raises(ValueError, match='Incomplete native run'):
        audit.audit_new_run(collector, 'L1_graph_coloring', {'run_dir': str(tmp_path), 'new_samples': 8192})


def test_native_usage_total_cannot_be_substituted(tmp_path, monkeypatch):
    rewrite(tmp_path / 'status.json', {'complete': True, 'completed_samples': 8192,
            'failed_groups_this_session': 0, 'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}})
    sample = {'usage': {'input_tokens': 1, 'output_tokens': 2, 'total_tokens': 3}}
    monkeypatch.setattr(audit, 'authenticate_saved_run', lambda h, out, expected: [sample] * expected)
    original_glob = Path.glob
    monkeypatch.setattr(Path, 'glob', lambda path, pattern: iter(range(8192)) if path.name == 'raw_responses'
                        else original_glob(path, pattern))
    collector = SimpleNamespace(load_helper=lambda: SimpleNamespace())
    with pytest.raises(ValueError, match='Status usage differs'):
        audit.audit_new_run(collector, 'L1_graph_coloring', {'run_dir': str(tmp_path), 'new_samples': 8192})
