from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import zipfile

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'ops/exp_scaling/e124_storage_admission.py'
spec = importlib.util.spec_from_file_location('e124_storage', SOURCE)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


@pytest.fixture
def context(tmp_path, monkeypatch):
    (tmp_path / 'var/data').mkdir(parents=True)
    (tmp_path / 'var/artifacts').mkdir()
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    monkeypatch.setattr(s.os, 'statvfs', lambda _: SimpleNamespace(
        f_bavail=1000 * s.GIB, f_frsize=1, f_favail=100000))
    plan = {'root': str(tmp_path), 'cells': [{'run_dir': str(tmp_path / 'var/data/e124/l1/maxrl'), 'model_size': '7b'}]}
    return tmp_path, plan


def job(root, job_id='10', model='3b', state='RUNNING', **kwargs):
    return dict(job_id=job_id, name='training', state=state, gres='gres/gpu:1',
                reason='None', run_dir=str(root / 'var/data' / ('external-' + job_id)),
                model_choice=model, **kwargs)


def test_exact_disk_boundary_and_no_mutation(context, monkeypatch):
    root, plan = context
    before = sorted(root.rglob('*'))
    required = (220 + 64) * s.GIB
    monkeypatch.setattr(s.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=required, f_frsize=1, f_favail=100000))
    result = s.storage_report(plan, external_snapshot={'jobs': []})
    assert result['allowed'] and result['required_bytes'] == required
    monkeypatch.setattr(s.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=required - 1, f_frsize=1, f_favail=100000))
    assert s.storage_report(plan, external_snapshot={'jobs': []})['blocked_reason'] == 'waiting_disk'
    assert before == sorted(root.rglob('*'))


@pytest.mark.parametrize('model,copies_bytes', [('05b', 14), ('3b', 82), ('7b', 200)])
def test_pending_external_reserves_two_checkpoint_writes(context, model, copies_bytes):
    root, plan = context
    result = s.storage_report(plan, external_snapshot={'jobs': [job(root, model=model, state='PENDING')]})
    assert not result['errors']
    assert result['external_reserve_bytes'] == copies_bytes * s.GIB
    assert result['reservations'][0]['copies_reserved'] == 2


def test_held_jobs_and_cpu_guards_do_not_consume_checkpoint_reservations(context):
    root, plan = context
    held = job(root, state='PENDING'); held['reason'] = 'JobHeldUser'
    cpu = {'job_id': '11', 'state': 'RUNNING', 'gres': 'N/A', 'name': 'guard'}
    result = s.storage_report(plan, external_snapshot={'jobs': [held, cpu]})
    assert result['allowed'] and result['external_reserve_bytes'] == 0


@pytest.mark.parametrize('state', ['RUNNING', 'PENDING', 'COMPLETING'])
def test_own_writer_detected_from_scheduler_enforces_singleton(context, state):
    root, plan = context
    own = dict(plan['cells'][0], job_id='77')
    queued = dict(job(root, job_id='77', state=state, model='7b'), run_dir=own['run_dir'])
    result = s.storage_report(plan, own_live_runs=[own], external_snapshot={'jobs': [queued]})
    assert not result['allowed'] and result['own_live_count'] == 1
    assert result['blocked_reason'] == 'own_concurrency_cap'


def test_ambiguous_own_release_blocks_without_scheduler_presence(context):
    _, plan = context
    result = s.storage_report(plan, own_live_runs=plan['cells'], external_snapshot={'jobs': []})
    assert result['blocked_reason'] == 'own_concurrency_cap'


def test_benchmark_shares_science_concurrency_cap(context):
    root, plan = context
    benchmark = {'run_dir': str(root / 'var/data/e124/benchmark'), 'model_size': '7b', 'job_id': '78'}
    plan['benchmark_cells'] = [benchmark]
    result = s.storage_report(plan, own_live_runs=[benchmark], external_snapshot={'jobs': []})
    assert result['blocked_reason'] == 'own_concurrency_cap'


@pytest.mark.parametrize('mutation', ['model', 'unmapped', 'state', 'gres', 'escape', 'root', 'symlink'])
def test_unresolved_or_escaping_identity_fails_closed(context, mutation):
    root, plan = context
    writer = job(root)
    if mutation == 'model': writer['model_choice'] = 'mystery-13b'
    elif mutation == 'unmapped': writer.pop('run_dir')
    elif mutation == 'state': writer['state'] = 'MYSTERY'
    elif mutation == 'gres': writer.pop('gres')
    elif mutation == 'escape': writer['run_dir'] = '/tmp/outside'
    elif mutation == 'root': plan['root'] = '/tmp'
    elif mutation == 'symlink':
        (root / 'var/data/link').symlink_to('/tmp', target_is_directory=True)
        writer['run_dir'] = str(root / 'var/data/link/outside')
    result = s.storage_report(plan, external_snapshot={'jobs': [writer]})
    assert not result['allowed'] and result['errors']


def write_checkpoint(path, corrupt=False):
    path.mkdir(parents=True)
    for name in ['mp_rank_00_model_states.pt', 'zero_pp_rank_0_mp_rank_00_optim_states.pt']:
        if corrupt:
            (path / name).write_bytes(b'incomplete')
        else:
            with zipfile.ZipFile(path / name, 'w') as z:
                z.writestr('archive/data.pkl', b'x' * 300)


def test_complete_current_checkpoint_reduces_only_next_overlap(context, monkeypatch):
    root, plan = context
    writer = job(root)
    cp = Path(writer['run_dir']) / 'debug_job10/checkpoints/step_00096'
    write_checkpoint(cp)
    actual = sum(p.stat().st_size for p in cp.iterdir())
    monkeypatch.setitem(s.CHECKPOINT_BYTES, '3b', actual)
    result = s.storage_report(plan, external_snapshot={'jobs': [writer]})
    assert result['external_reserve_bytes'] == actual
    assert result['reservations'][0]['copies_reserved'] == 1
    assert result['reservations'][0]['existing_current_checkpoint']['path'] == str(cp)


@pytest.mark.parametrize('kind', ['corrupt', 'tiny', 'previous_job'])
def test_partial_or_historical_checkpoint_does_not_reduce_reserve(context, kind):
    root, plan = context
    writer = job(root)
    cp = Path(writer['run_dir']) / ('debug_job9' if kind == 'previous_job' else 'debug_job10') / 'checkpoints/step_00096'
    write_checkpoint(cp, corrupt=kind == 'corrupt')
    result = s.storage_report(plan, external_snapshot={'jobs': [writer]})
    assert result['external_reserve_bytes'] == 82 * s.GIB


def test_oversized_checkpoint_fails_closed(context, monkeypatch):
    root, plan = context
    writer = job(root)
    write_checkpoint(Path(writer['run_dir']) / 'debug_job10/checkpoints/step_00096')
    monkeypatch.setitem(s.CHECKPOINT_BYTES, '3b', 1)
    result = s.storage_report(plan, external_snapshot={'jobs': [writer]})
    assert not result['allowed'] and 'exceeds' in result['errors'][0]


def test_canonical_continuation_uses_same_run_model_without_reading_outcomes(context, monkeypatch):
    root, _ = context
    path = str(root / 'var/data/e119/run')
    (root / 'var/artifacts/source.json').write_text(json.dumps({'model': 'Qwen2.5-0.5B-Instruct', 'runs': [{'job_id': 1, 'run_dir': path}]}))
    (root / 'var/artifacts/continuation.json').write_text(json.dumps({'continuations': [{'original_job_id': 1, 'continuation_job_id': 2, 'run_dir': path}]}))
    monkeypatch.setattr(s, 'LEDGERS', ('source.json', 'continuation.json'))
    registry = s.canonical_registry()
    assert registry['1']['model_choice'] == registry['2']['model_choice'] == '05b'


def test_scheduler_failure_is_a_readonly_block(context, monkeypatch):
    _, plan = context
    def fail(): raise s.subprocess.TimeoutExpired('squeue', 20)
    monkeypatch.setattr(s, 'scheduler_snapshot', fail)
    result = s.storage_report(plan)
    assert not result['allowed'] and result['errors']


def test_duplicate_plan_paths_and_missing_own_identity_fail_closed(context):
    _, plan = context
    plan['cells'].append(dict(plan['cells'][0]))
    assert s.storage_report(plan, external_snapshot={'jobs': []})['errors']
    plan['cells'].pop()
    assert s.storage_report(plan, own_live_runs=['123'], external_snapshot={'jobs': []})['errors']


def test_explicit_plan_root_and_model_aliases(context):
    _, plan = context
    assert s.storage_report(plan, external_snapshot={'jobs': []})['allowed']
    for alias in ('Qwen/Qwen2.5-7B-Instruct', 'qwen7b', '7b'):
        assert s.model_choice({'model': alias}) == '7b'
    with pytest.raises(ValueError): s.model_choice({'model_choice': '3b', 'model_size': '7b'})


def test_exact_artifact_benchmark_root_supported_but_adjacent_paths_rejected(context):
    root, plan = context
    benchmark = {'run_dir': str(root / 'var/artifacts/e124_qwen7b_three_level/systems'), 'model_size': '7b'}
    plan['benchmark_cells'] = [benchmark]
    assert s.storage_report(plan, external_snapshot={'jobs': []})['allowed']
    assert s.storage_report(plan, own_live_runs=[benchmark], external_snapshot={'jobs': []})['blocked_reason'] == 'own_concurrency_cap'
    benchmark['run_dir'] = str(root / 'var/artifacts/e124_qwen7b_three_level/other')
    assert s.storage_report(plan, external_snapshot={'jobs': []})['errors']


def test_duplicate_external_writer_is_not_deduplicated_into_one_reserve(context):
    root, plan = context
    first, second = job(root), job(root, job_id='11')
    second['run_dir'] = first['run_dir']
    result = s.storage_report(plan, external_snapshot={'jobs': [first, second]})
    assert not result['allowed'] and 'multiple released writers' in result['errors'][0]


def test_frozen_helper_honors_explicit_repository_environment(tmp_path, monkeypatch):
    copied = tmp_path / 'control/frozen/e124_storage_admission.py'
    copied.parent.mkdir(parents=True)
    copied.write_bytes(SOURCE.read_bytes())
    monkeypatch.setenv('OAT_ZERO_REPO_ROOT', str(tmp_path))
    frozen_spec = importlib.util.spec_from_file_location('frozen_e124_storage', copied)
    frozen = importlib.util.module_from_spec(frozen_spec)
    frozen_spec.loader.exec_module(frozen)
    assert frozen.ROOT == tmp_path


def test_model_conflict_is_not_repaired_by_another_ledger(context, monkeypatch):
    root, _ = context
    path = str(root / 'var/data/conflict')
    (root / 'var/artifacts/source.json').write_text(json.dumps({'runs': [{'job_id': 1, 'run_dir': path, 'model_choice': '3b'}]}))
    (root / 'var/artifacts/other.json').write_text(json.dumps({'runs': [{'job_id': 2, 'run_dir': path, 'model_choice': '7b'}]}))
    monkeypatch.setattr(s, 'LEDGERS', ('source.json', 'other.json'))
    with pytest.raises(ValueError, match='conflicting model identity'):
        s.canonical_registry()


def test_data_root_symlink_cannot_expand_authorized_filesystem_scope(context, tmp_path):
    root, plan = context
    outside = root.parent / (root.name + '-outside')
    outside.mkdir()
    (root / 'var/data').rmdir()
    (root / 'var/data').symlink_to(outside, target_is_directory=True)
    result = s.storage_report(plan, external_snapshot={'jobs': []})
    assert not result['allowed'] and 'escapes approved' in result['errors'][0]
