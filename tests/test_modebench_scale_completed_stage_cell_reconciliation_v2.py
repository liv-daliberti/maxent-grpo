"""Scratch fixtures only: no scheduler commands, model loads or real grading."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

# Reuse the reviewed v1 synthetic scientific fixture; its grader is a call spy.
from test_modebench_scale_completed_stage_cell_reconciliation import factory, mutate, write

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/reconcile_modebench_scale_completed_stage_cell_v2_20260912.py'


def load_adapter():
    spec = importlib.util.spec_from_file_location('scratch_completed_cell_v2', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapted(factory, monkeypatch):
    def create(index=2):
        c = factory()
        a = load_adapter()
        b = a._base
        # Inject only scratch paths/dependencies; inherited implementation stays.
        for key in ('ROOT', 'COMPOSITE', 'TRANSPORT', 'REVISION', 'scientific_modules', 'tokenizer_for'):
            monkeypatch.setattr(b, key, getattr(c.a, key))
        directory = c.directory.with_name(str(index))
        c.directory.rename(directory)
        raw, code, start, end = a.OBSERVED_CELLS[index]
        human = '31254520_'+str(index)
        plan = a.read(c.path)
        plan['cells'] = [copy.deepcopy(c.cell) for _ in range(index+1)]
        write(c.path, plan)
        plan_sha = a.file_sha(c.path)
        monkeypatch.setattr(b, 'DEV_PLAN_SHA', plan_sha)
        write(c.path.parent/'plan.sha256.json', {'sha256': plan_sha})
        intent = c.path.parent/'submission_intent.json'
        mutate(intent, lambda value: value.update(plan_sha256=plan_sha))
        txroot = c.path.parent/'frozen_transport'
        txintent = txroot/'submission_intent.json'
        mutate(txintent, lambda value: value.update(canonical_intent_sha256=a.file_sha(intent)))
        effective = txroot/'submission_result.json'
        mutate(effective, lambda value: value.update(intent_sha256=a.file_sha(txintent)))
        start_time = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_time = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        runtime_path = c.path.parent/'runtime'/f'{index}.json'
        runtime_path.with_name('0.json').rename(runtime_path)
        mutate(runtime_path, lambda value: value.update(plan_sha256=plan_sha,
            at=(start_time+timedelta(seconds=15)).isoformat()))
        observed_path = txroot/'runtime'/f'{index}.json'
        observed_path.with_name('0.json').rename(observed_path)
        def update_observed(value):
            value.update(array_index=index, stage_plan_sha256=plan_sha,
                effective_submission_result_sha256=a.file_sha(effective),
                at_utc=(start_time+timedelta(seconds=10)).isoformat())
            value['environment'].update(SLURM_JOB_ID=raw, SLURM_ARRAY_TASK_ID=str(index))
        mutate(observed_path, update_observed)
        for receipt in c.receipts:
            mutate(receipt, lambda value: value.update(generated_at=(end_time-timedelta(seconds=5)).isoformat()))
        logs = []
        for suffix in ('.out', '.err'):
            old = c.path.parent/'logs'/('31254520_0'+suffix)
            new = old.with_name(human+suffix)
            old.rename(new)
            logs.append(new)
        terminal_path = directory/'terminal_accounting.json'
        def update_terminal(value):
            value.update(array_index=index, job_id=human, observer_host='spin.cs.princeton.edu',
                command=['sacct', '-j', human, '-n', '-P', '--format='+a.FIELDS],
                captured_at_utc=(end_time+timedelta(minutes=2)).isoformat(),
                logs_sha256={str(path): a.file_sha(path) for path in logs})
            rows = [line.split('|') for line in value['stdout'].splitlines()]
            for row, suffix in zip(rows, ('', '.batch', '.extern')):
                row[:6] = [raw+suffix, human+suffix, 'COMPLETED' if suffix=='.extern' else 'FAILED',
                            '0:0' if suffix=='.extern' else code, start, end]
                if suffix:
                    row[10] = start
            value['stdout'] = ''.join('|'.join(row)+'\n' for row in rows)
        mutate(terminal_path, update_terminal)
        return SimpleNamespace(a=a, b=b, c=c, index=index, path=c.path, directory=directory,
            terminal_path=terminal_path, runtime_path=runtime_path, observed_path=observed_path,
            digest=a.file_sha(terminal_path), receipts=c.receipts, grading=c.grading)
    return create


@pytest.mark.parametrize('index,code', [(2, '143:0'), (3, '1:0'), (4, '1:0')])
def test_exact_observed_cells_keep_actual_code_and_replay_once(adapted, index, code):
    f = adapted(index)
    original_terminal = f.terminal_path.read_bytes()
    value = f.a.audit('revision_development', index, f.digest)
    assert value['state'] == 'FAILED' and value['exit_code'] == code
    assert value['scheduler_success'] is False and value['exit_cause'] == 'unknown'
    assert value['completed_receipts'] == 4 and value['attempts_validated'] == 1280
    assert value['new_grader_invocations'] == 1280 and value['completed_batches'] == 32
    assert value['files_sha256'][str(f.a.BASE_SOURCE)] == f.a.BASE_SHA
    assert value['files_sha256'][str(f.a.SOURCE)] == f.a.file_sha(f.a.SOURCE)
    registration = f.a.read(f.directory/'registration.json')
    assert registration['script'] == str(f.a.SOURCE)
    assert f.a.read(f.terminal_path)['observer_host'] == 'spin.cs.princeton.edu'
    assert f.terminal_path.read_bytes() == original_terminal
    assert f.grading == [(str(path), True) for path in f.receipts]
    assert f.a.verify_reconciliation('revision_development', index) == value
    assert len(f.grading) == 4
    with pytest.raises(ValueError, match='already claimed'):
        f.a.audit('revision_development', index, f.digest)
    assert len(f.grading) == 4


@pytest.mark.parametrize('stage,index', [('revision_development', 0), ('revision_development', 1),
    ('revision_development', 5), ('revision_development', 6), ('revision_development', True),
    ('revision_development', -1), ('confirmation', 2)])
def test_no_existing_future_or_confirmation_cell_can_be_affected(stage, index):
    a = load_adapter()
    with pytest.raises(ValueError, match='only covers explicitly observed'):
        a.register(stage, index, '0'*64)


def test_original_scientific_and_exact_once_functions_are_reused():
    a = load_adapter()
    assert a.audit is a._base.audit and a.verify_reconciliation is a._base.verify_reconciliation
    assert a.audit.__globals__['inspect_cell'] is a.inspect_cell
    assert a.register.__globals__['validate_terminal'] is a.validate_terminal
    assert a.register.__globals__['SOURCE'] == a.SOURCE
    for name in ('completed_task', 'runtime_linkage', 'native_prompts', 'tokenizer_for',
                 'audit', 'registered', 'register', 'build_certificate', 'action_lock', 'prior_tier'):
        assert Path(inspect.unwrap(getattr(a._base, name)).__code__.co_filename) == a.BASE_SOURCE
    assert hashlib.sha256(a.BASE_SOURCE.read_bytes()).hexdigest() == a.BASE_SHA


@pytest.mark.parametrize('field,value', [('observer_host', 'wash.cs.princeton.edu'),
    ('observer_host', 'other.cs.princeton.edu'), ('returncode', 1), ('environment', {'TZ': 'EST'}),
    ('stderr', 'query failed'), ('job_id', '31257662'), ('array_job_id', 123),
    ('captured_at_utc', '2026-09-12T08:29:59+00:00')])
def test_capture_identity_or_chronology_change_cannot_register(adapted, field, value):
    f = adapted()
    mutate(f.terminal_path, lambda record: record.update({field: value}))
    with pytest.raises(ValueError):
        f.a.audit('revision_development', 2, f.a.file_sha(f.terminal_path))
    assert f.grading == [] and not (f.directory/'claim.json').exists()


@pytest.mark.parametrize('row,column,value', [(0, 0, '31257663'), (0, 2, 'NODE_FAIL'),
    (0, 2, 'COMPLETED'), (0, 3, '1:0'), (0, 3, '0:15'), (1, 3, '1:0'),
    (2, 3, '143:0'), (0, 4, '2026-09-12T05:52:43'), (0, 5, '2026-09-12T08:30:05'),
    (0, 6, 'node106'), (1, 7, '8'), (0, 8, '59G'), (1, 9, 'cpu=6,node=1,mem=60G,gres/gpu=1,gres/gpu:a5000=1')])
def test_exact_rows_resources_exit_status_are_mandatory(adapted, row, column, value):
    f = adapted()
    def change(record):
        rows = [line.split('|') for line in record['stdout'].splitlines()]
        rows[row][column] = value
        record['stdout'] = ''.join('|'.join(values)+'\n' for values in rows)
    mutate(f.terminal_path, change)
    with pytest.raises(ValueError):
        f.a.register('revision_development', 2, f.a.file_sha(f.terminal_path))
    assert f.grading == [] and not (f.directory/'registration.json').exists()


def test_wrong_terminal_sha_fails_before_claim(adapted):
    f = adapted()
    with pytest.raises(ValueError, match='explicit captured terminal'):
        f.a.audit('revision_development', 2, '0'*64)
    assert f.grading == [] and not (f.directory/'claim.json').exists()


@pytest.mark.parametrize('kind', ['receipt', 'batch', 'extra_batch', 'run', 'native_prompt', 'runtime_raw', 'rng'])
def test_incomplete_or_changed_science_cannot_be_reconciled(adapted, kind):
    f = adapted()
    directory = Path(str(f.receipts[3])+'.batches')
    if kind == 'receipt':
        f.receipts[3].unlink()
    elif kind == 'batch':
        next(directory.glob('seed-*')).unlink()
    elif kind == 'extra_batch':
        write(directory/'extra.json', {})
    elif kind == 'run':
        mutate(directory/'run.json', lambda value: value.update(identity_sha256='bad'))
    elif kind == 'native_prompt':
        f.b.tokenizer_for = lambda context: SimpleNamespace(apply_chat_template=lambda *args, **kw: 'changed',
                                                          encode=lambda *args, **kw: [1])
    elif kind == 'runtime_raw':
        mutate(f.observed_path, lambda value: value['environment'].update(SLURM_JOB_ID='31257952'))
    elif kind == 'rng':
        mutate(f.receipts[0], lambda value: value['identity'].update(seeds=[1,2,3,4]))
    with pytest.raises((ValueError, FileNotFoundError)):
        f.a.audit('revision_development', 2, f.digest)
    assert f.grading == []
    if kind == 'native_prompt':
        assert (f.directory/'claim.json').exists() and (f.directory/'failure.json').exists()
    else:
        assert not (f.directory/'claim.json').exists()


def test_pin_drift_after_registration_stops_before_claim(adapted):
    f = adapted()
    f.a.register('revision_development', 2, f.digest)
    mutate(f.receipts[0], lambda value: value['metrics'].update(pass1=.9))
    with pytest.raises(ValueError, match='pinned evidence changed'):
        f.a.audit('revision_development', 2, f.digest)
    assert not (f.directory/'claim.json').exists() and f.grading == []


def test_sealed_base_drift_stops_inspection(adapted, monkeypatch, tmp_path):
    f = adapted()
    substitute = tmp_path/'changed_base.py'
    substitute.write_text('changed')
    monkeypatch.setattr(f.a, 'BASE_SOURCE', substitute)
    with pytest.raises(ValueError, match='sealed original'):
        f.a.audit('revision_development', 2, f.digest)
    assert not (f.directory/'claim.json').exists() and f.grading == []


def test_grader_failure_is_durable_and_cannot_repeat(adapted):
    f = adapted()
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise ValueError('scratch grader mismatch')
    f.c.modules.revision.receipt_scores = fail
    with pytest.raises(ValueError, match='scratch grader mismatch'):
        f.a.audit('revision_development', 2, f.digest)
    assert len(calls) == 1 and (f.directory/'failure.json').exists()
    with pytest.raises(ValueError, match='already claimed'):
        f.a.audit('revision_development', 2, f.digest)
    assert len(calls) == 1


def test_action_lock_rejects_concurrent_audit(adapted):
    f = adapted()
    with f.b.action_lock(f.directory):
        with pytest.raises(BlockingIOError):
            f.a.audit('revision_development', 2, f.digest)
    assert not (f.directory/'claim.json').exists() and f.grading == []
