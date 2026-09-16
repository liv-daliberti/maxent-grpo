import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import continue_e122_slurm2511 as continuation


@pytest.fixture
def journals(tmp_path, monkeypatch):
    original = continuation.original
    cells = [dict(domain='countdown', dataset_domain='countdown', arm='drgrpo',
                  target_steps=3072, seed=i, run_stamp=f'test_{i}',
                  run_dir=str(tmp_path / f'unused_run_{i}'),
                  environment={'OAT_ZERO_LEARNING_RATE': '2e-7'},
                  command=['sbatch', '--hold', f'cell_{i}']) for i in range(100)]
    plan = {'cells': cells, 'model': 'Qwen2.5-0.5B-Instruct',
            'model_revision': 'fixed-model-revision', 'admission_proof': {'all_pass': True},
            'snapshot_root': str(tmp_path / 'immutable_snapshot')}
    ledger = {'schema': original.LEDGER_SCHEMA, 'runs': [], 'released': False,
              'planned_runs': cells}
    here = tmp_path / 'journals'
    here.mkdir()
    ledger_path = tmp_path / 'ledger.json'
    ledger_path.write_text(json.dumps(ledger))
    initial_sha = original.digest(ledger_path)
    (here / 'submission_claim.json').write_text(json.dumps({
        'schema': 'e122_once_only_submission_claim_v1',
        'plan_sha256': continuation.PLAN_SHA256,
        'initial_ledger_sha256': initial_sha, 'jobs': 100}))
    (here / 'submission_000_intent.json').write_text(json.dumps({
        'cell': cells[0], 'command_sha256': original.sha(cells[0]['command'])}))
    (here / 'submission_000_result.json').write_text(json.dumps({
        'returncode': 0, 'stdout': '31158645\n', 'stderr': ''}))
    monkeypatch.setattr(original, 'HERE', here)
    monkeypatch.setattr(original, 'CLAIM', here / 'submission_claim.json')
    monkeypatch.setattr(original, 'LEDGER', ledger_path)
    monkeypatch.setattr(continuation, 'INITIAL_LEDGER_SHA256', initial_sha)
    return plan, ledger, here


def test_known_first_submission_reused(journals):
    plan, ledger, _ = journals
    assert continuation.reconcile_journals(plan, ledger) == 31158645


@pytest.mark.parametrize('name', [
    'submission_001_intent.json', 'submission_001_result.json', 'held_audit_000.json',
    'slurm2511_continuation_intent.json', 'held_submission_complete.json',
    'prospective_ledger_before_submission.json',
])
def test_any_additional_attempt_or_claim_forbids_continuation(journals, name):
    plan, ledger, here = journals
    (here / name).write_text('{}')
    with pytest.raises(ValueError):
        continuation.reconcile_journals(plan, ledger)


@pytest.mark.parametrize('replacement', [
    {'returncode': 1, 'stdout': '31158645\n'},
    {'returncode': 0, 'stdout': '31158646\n'},
    {'returncode': 0, 'stdout': '31158645\n31158646\n'},
    {'status': 'ambiguous_exception', 'error': 'timeout'},
])
def test_first_result_must_be_exact_known_success(journals, replacement):
    plan, ledger, here = journals
    (here / 'submission_000_result.json').write_text(json.dumps(replacement))
    with pytest.raises(ValueError):
        continuation.reconcile_journals(plan, ledger)


def test_first_command_cannot_be_reinterpreted(journals):
    plan, ledger, here = journals
    path = here / 'submission_000_intent.json'
    value = json.loads(path.read_text())
    value['cell']['environment']['OAT_ZERO_LEARNING_RATE'] = '1e-7'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='first submitted command'):
        continuation.reconcile_journals(plan, ledger)


def test_initial_ledger_drift_rejected(journals):
    plan, ledger, _ = journals
    continuation.original.LEDGER.write_text(json.dumps(ledger) + '\n\n')
    with pytest.raises(ValueError, match='prospective ledger changed'):
        continuation.reconcile_journals(plan, ledger)


def test_record_keeps_original_raw_display():
    class Adapter:
        def audit_held(self, job_id, cell):
            assert job_id == 31158645
            return 'JobId=31158645 NumNodes=1-1'
    cell = dict(domain='countdown', dataset_domain='countdown', arm='drgrpo', seed=43,
                run_stamp='test', run_dir='/tmp/unused', target_steps=3072)
    record = continuation.record_for(cell, 31158645, Adapter())
    assert record['held_scheduler_record'].endswith('NumNodes=1-1')
    assert record['job_id'] == 31158645


@pytest.fixture
def execution(journals, monkeypatch, tmp_path):
    """Use real exclusive journal writes and submit-one parsing; never call Slurm."""
    plan, ledger, here = journals
    original = continuation.original
    attempts, audits, verifications = [], [], []
    amendment = tmp_path / 'amendment.json'
    amendment.write_text('{}')
    amendment_sha = original.digest(amendment)
    pins = {str(p): original.digest(p) for p in (continuation.SOURCE, continuation.TEST)}

    def authenticate(path, expected_sha):
        assert Path(path) == amendment and expected_sha == amendment_sha
        return {'files_sha256': pins}

    def verify_plan(*, expected_sha256, model_choice):
        assert expected_sha256 == continuation.PLAN_SHA256 and model_choice == '05b'
        verifications.append((expected_sha256, model_choice))
        return plan

    def queue(first_job_id):
        assert first_job_id == continuation.FIRST_JOB_ID
        return '31158645|e122_level3_countdown_drgrpo_s43|PENDING|JobHeldUser\n'

    class Adapter:
        def audit_held(self, job_id, cell):
            audits.append((job_id, cell['run_stamp']))
            return f'JobId={job_id} NumNodes=1-1 JobName={cell["run_stamp"]}'

    def fake_sbatch(command, **kwargs):
        assert command[0:2] == ['sbatch', '--hold']
        index = int(command[-1].split('_')[-1])
        attempts.append(index)
        if state['fail_at'] == index:
            raise TimeoutError('scheduler reply lost')
        return type('Reply', (), {'returncode': 0,
                                 'stdout': str(continuation.FIRST_JOB_ID + index) + '\n',
                                 'stderr': ''})()

    real_submit_one = original.submit_one_held

    def submit_one(cell, index):
        assert cell == plan['cells'][index]
        return real_submit_one(cell, index, directory=here, runner=fake_sbatch)

    state = {'fail_at': None, 'attempts': attempts, 'audits': audits,
             'verifications': verifications, 'plan': plan, 'ledger': ledger, 'here': here,
             'amendment': amendment, 'amendment_sha': amendment_sha}
    monkeypatch.setattr(continuation.compatibility, 'authenticate_amendment', authenticate)
    monkeypatch.setattr(continuation.compatibility, 'LauncherAdapter', lambda source: Adapter())
    monkeypatch.setattr(original, 'verify_plan', verify_plan)
    monkeypatch.setattr(original, 'submit_one_held', submit_one)
    monkeypatch.setattr(continuation, 'reconcile_queue', queue)
    return state


def test_readonly_execution_does_not_claim_or_submit(execution):
    state = execution
    original = continuation.original
    before = {p.name: p.read_bytes() for p in state['here'].glob('*.json')}
    initial_ledger = original.LEDGER.read_bytes()
    result = continuation.execute(state['amendment'], state['amendment_sha'])
    assert result == {'status': 'single_successful_submission_reconciled',
                      'existing_job_id': continuation.FIRST_JOB_ID,
                      'remaining_unsubmitted': 99, 'scheduler_mutation': False}
    assert state['attempts'] == []
    assert {p.name: p.read_bytes() for p in state['here'].glob('*.json')} == before
    assert original.LEDGER.read_bytes() == initial_ledger


def test_success_continues_exactly_99_without_resubmitting_first(execution):
    state = execution
    original, here = continuation.original, state['here']
    first = {p.name: p.read_bytes() for p in here.glob('submission_000_*.json')}
    initial_ledger = json.loads(original.LEDGER.read_text())
    result = continuation.execute(state['amendment'], state['amendment_sha'], continue_held=True)
    assert result['held'] == 100 and result['released'] == 0
    assert state['attempts'] == list(range(1, 100))
    assert {p.name: p.read_bytes() for p in here.glob('submission_000_*.json')} == first
    assert len(list(here.glob('submission_*_intent.json'))) == 100
    assert len(list(here.glob('submission_*_result.json'))) == 100
    assert len(list(here.glob('held_audit_*.json'))) == 100
    published = json.loads(original.LEDGER.read_text())
    assert published['released'] is False and published['status'] == 'held_audited'
    assert published['model_choice'] == '05b'
    ids = [row['job_id'] for row in published['runs']]
    assert ids == list(range(continuation.FIRST_JOB_ID, continuation.FIRST_JOB_ID + 100))
    assert len(set(ids)) == 100
    assert published['planned_runs'] == initial_ledger['planned_runs']
    assert json.loads((here / 'prospective_ledger_before_submission.json').read_text()) == initial_ledger
    complete = json.loads((here / 'held_submission_complete.json').read_text())
    assert complete['job_ids'] == ids and complete['first_job_reused_without_resubmission'] is True
    assert complete['ledger_sha256'] == original.digest(original.LEDGER)
    assert len(state['audits']) == 200
    assert len(state['verifications']) == 2
    with pytest.raises(ValueError):
        continuation.execute(state['amendment'], state['amendment_sha'], continue_held=True)
    assert state['attempts'] == list(range(1, 100))


def test_ambiguous_new_submission_preserves_evidence_and_forbids_retry(execution):
    state = execution
    original, here = continuation.original, state['here']
    initial_ledger = original.LEDGER.read_bytes()
    state['fail_at'] = 2
    with pytest.raises(TimeoutError, match='scheduler reply lost'):
        continuation.execute(state['amendment'], state['amendment_sha'], continue_held=True)
    assert state['attempts'] == [1, 2]
    assert original.LEDGER.read_bytes() == initial_ledger
    assert (here / 'held_audit_000.json').exists()
    assert (here / 'held_audit_001.json').exists()
    assert not (here / 'held_audit_002.json').exists()
    ambiguity = json.loads((here / 'submission_002_result.json').read_text())
    assert ambiguity['status'] == 'ambiguous_exception'
    assert ambiguity['error_type'] == 'TimeoutError'
    assert not (here / 'held_submission_complete.json').exists()
    with pytest.raises(ValueError):
        continuation.execute(state['amendment'], state['amendment_sha'], continue_held=True)
    assert state['attempts'] == [1, 2]
