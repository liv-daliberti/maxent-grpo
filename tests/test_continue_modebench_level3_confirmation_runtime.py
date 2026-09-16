"""Ten-worker runtime authentication fixtures; no scheduler or inference calls."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('confirmation_runtime_driver_test',
    ROOT / 'ops/exp_scaling/continue_modebench_level3_v2.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
COMPLETION_SPEC = importlib.util.spec_from_file_location('confirmation_runtime_completion_test',
    ROOT / 'ops/audit_modebench_level3_recovery_execution.py')
completion = importlib.util.module_from_spec(COMPLETION_SPEC)
COMPLETION_SPEC.loader.exec_module(completion)


def write(path, value):
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def execution(tmp_path, monkeypatch):
    folder = tmp_path / 'confirmation'
    logs = tmp_path / 'var/logs/modebench_level3'
    folder.mkdir()
    logs.mkdir(parents=True)
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'CONFIRMATION', folder)
    monkeypatch.setattr(driver, 'module', lambda *args: completion)
    plan = {'jobs': [], 'models': {'05b': '/fixed/05b', '3b': '/fixed/3b'}}
    ids, accounts, fixtures = [], {}, []
    for index in range(10):
        label = '05b' if index % 2 == 0 else '3b'
        name = label + '_' + driver.DOMAINS[index // 2]
        task = folder / f'{name}_tasks.json'
        write(task, [{'registered_source': name}])
        job = {'name': name, 'model_label': label, 'tasks': str(task)}
        job_id = str(40000 + index)
        account = {'JobIDRaw': job_id, 'State': 'COMPLETED', 'ExitCode': '0:0', 'Partition': 'all',
            'AllocCPUS': '6', 'ReqMem': '48G', 'Timelimit': '01:00:00', 'NodeList': 'node007',
            'Start': '2026-09-09T12:00:00', 'End': '2026-09-09T12:30:00',
            'ReqTRES': 'cpu=6,mem=48G,gres/gpu:rtx_6000=1', 'AllocTRES': 'cpu=6,mem=48G,gres/gpu:rtx_6000=1'}
        runtime = {'partition_preempt_mode': 'OFF', 'attention_backend': 'XFORMERS', 'engine': 'V0',
            'scheduler': {'JobId': job_id, 'JobState': 'RUNNING', 'UserId': f'test({os.getuid()})',
                'JobName': 'mb-l3-v2-confirm-' + name, 'Command': str(folder / 'worker.slurm'),
                'Partition': 'all', 'QOS': 'none', 'NumCPUs': '6', 'MinMemoryNode': '48G',
                'TimeLimit': '01:00:00', 'TresPerNode': 'gpu:rtx_6000:1', 'NodeList': 'node007',
                'Restarts': '0', 'StartTime': '2026-09-09T12:00:00'}}
        claim = {'identity': {'job_id': job_id, 'seal_sha256': 'fixed-seal', 'task_sha256': driver.digest(task)},
                 'runtime': runtime, 'gpu_names': ['Quadro RTX 6000']}
        claim_path = folder / f'worker_{index:02d}_execution_claim.json'
        write(claim_path, claim)
        event = {'event': 'worker_authenticated', 'job_id': job_id, 'cell': name, 'seal_sha256': 'fixed-seal',
                 'runtime': deepcopy(runtime), 'gpu_names': ['Quadro RTX 6000']}
        outpath, errpath = logs / (job_id + '.out'), logs / (job_id + '.err')
        engine = f"Initializing a V0 LLM engine (v0.8.4) model='{plan['models'][label]}', dtype=torch.float16, tensor_parallel_size=1, quantization=None\nUsing XFormers backend."
        outpath.write_text(json.dumps(event) + '\n' + engine)
        errpath.write_text('')
        fixtures.append(SimpleNamespace(job=job, account=account, claim=claim, event=event, engine=engine,
            task=task, claim_path=claim_path, outpath=outpath, errpath=errpath))
        plan['jobs'].append(job)
        ids.append(job_id)
        accounts[job_id] = account
    return SimpleNamespace(plan=plan, ids=ids, accounts=accounts, cells=fixtures)


def authenticate(execution):
    return driver.authenticate_confirmation_execution(execution.plan, execution.ids, execution.accounts, 'fixed-seal')


def test_all_ten_actual_worker_claims_logs_and_normalized_qos_are_authenticated(execution):
    evidence, files = authenticate(execution)
    assert evidence['all_ten_workers_authenticated'] is True
    assert [row['job_id'] for row in evidence['jobs']] == execution.ids
    assert len(files) == 30
    for cell in execution.cells:
        assert all(files[str(path)] == driver.digest(path) for path in (cell.claim_path, cell.outpath, cell.errpath))


@pytest.mark.parametrize('change', ['missing_cell', 'duplicate_job', 'missing_account', 'missing_claim', 'missing_log'])
def test_incomplete_or_ambiguous_ten_worker_inventory_is_rejected(execution, change):
    if change == 'missing_cell': execution.plan['jobs'].pop()
    elif change == 'duplicate_job': execution.ids[-1] = execution.ids[0]
    elif change == 'missing_account': execution.accounts.pop(execution.ids[-1])
    elif change == 'missing_claim': execution.cells[-1].claim_path.unlink()
    else: execution.cells[-1].outpath.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        authenticate(execution)


@pytest.mark.parametrize('change', ['ownership', 'task', 'timeout', 'exit', 'memory', 'gpu', 'partition',
                                   'backend', 'node', 'model', 'engine', 'dtype', 'missing_auth_event'])
def test_late_cell_runtime_or_model_mismatch_is_rejected(execution, change):
    cell = execution.cells[8]
    if change == 'ownership': cell.claim['identity']['job_id'] = execution.ids[0]
    elif change == 'task': cell.task.write_text('changed source manifest')
    elif change == 'timeout': cell.account['State'] = 'TIMEOUT'
    elif change == 'exit': cell.account['ExitCode'] = '1:0'
    elif change == 'memory': cell.account['ReqMem'] = '24G'
    elif change == 'gpu': cell.claim['gpu_names'] = ['A6000']
    elif change == 'partition': cell.claim['runtime']['scheduler']['Partition'] = 'lowprio'
    elif change == 'backend': cell.claim['runtime']['attention_backend'] = 'FLASH_ATTN'
    elif change == 'node': cell.account['NodeList'] = 'node008'
    elif change == 'model': cell.engine = cell.engine.replace('/fixed/05b', '/other/model')
    elif change == 'engine': cell.engine = cell.engine.replace('V0', 'V1')
    elif change == 'dtype': cell.engine = cell.engine.replace('float16', 'bfloat16')
    write(cell.claim_path, cell.claim)
    cell.outpath.write_text(('' if change == 'missing_auth_event' else json.dumps(cell.event) + '\n') + cell.engine)
    with pytest.raises(ValueError):
        authenticate(execution)


def test_same_id_restart_accepts_preserved_initial_claim_and_only_latest_log(execution):
    cell = execution.cells[4]
    original = cell.claim_path.read_bytes()
    cell.event['runtime']['scheduler'].update(Restarts='1', StartTime='2026-09-09T12:10:00')
    cell.outpath.write_text(json.dumps(cell.event) + '\n' + cell.engine)
    authenticate(execution)
    assert cell.claim_path.read_bytes() == original


def test_runtime_epoch_cannot_change_without_a_recorded_same_id_restart(execution):
    cell = execution.cells[4]
    cell.event['runtime']['scheduler']['StartTime'] = '2026-09-09T12:10:00'
    cell.outpath.write_text(json.dumps(cell.event) + '\n' + cell.engine)
    with pytest.raises(ValueError, match='without same-ID restart'):
        authenticate(execution)


def test_inflight_runtime_evidence_mutation_is_rejected(execution, monkeypatch):
    check = completion.verify_engine_log
    cell = execution.cells[0]
    def mutate_after_read(text, model):
        check(text, model)
        cell.outpath.write_text(cell.outpath.read_text() + '\nchanged during validation')
    monkeypatch.setattr(completion, 'verify_engine_log', mutate_after_read)
    with pytest.raises(ValueError, match='bytes changed'):
        authenticate(execution)
