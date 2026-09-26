#!/usr/bin/env python3
"""Read-only proof of the eleven registered development execution recoveries.

No generation, fitting, scheduler mutation, or confirmation outcomes are used.
--publish creates the canonical immutable attestation only after every replacement
has completed successfully and all 43,008 original-law attempts regrade exactly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import statistics
from collections import Counter
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
from evaluate_modebench_level3_independent import sha, file_sha, atomic_new, load_rows

CAMPAIGN = ROOT / 'var/artifacts/modebench_level3_v2/regular_queue_recovery'
SEAL = CAMPAIGN / 'implementation_seal.json'
SEAL_SHA256 = '65b3990674e8560d084f328a4e189709bfc99cd012e8e6620efd81879c8d2cab'
LEDGER = CAMPAIGN / 'development_jobs.json'
LEDGER_SHA256 = 'bfcfc53886ecc0b44e52a8c5c8657a3e09d399d7b85f29bc5d6ed71c0516035d'
ATTESTATION = CAMPAIGN / 'completion_attestation.json'
SCHEMA = 'modebench_level3_regular_queue_recovery_completion_v1'
SCIENTIFIC_SHA256 = '3b9f126eb04ddd42ae9aba78b15fe69273f50c74c52a2c6f6b5f3807c518e931'
ACCOUNT_FIELDS = ('JobIDRaw', 'State', 'ExitCode', 'Partition', 'AllocCPUS', 'ReqMem',
                  'Timelimit', 'NodeList', 'Start', 'End', 'ReqTRES', 'AllocTRES')
ACTIVE_STATES = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'REQUEUED'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return file_sha(Path(path))


def recovery_helper():
    # Authenticate this module before executing any of its source.
    require(digest(SEAL) == SEAL_SHA256, 'canonical recovery execution seal changed')
    seal = read(SEAL)
    path = CAMPAIGN / 'recovery_worker.py'
    require(seal['files_sha256'].get(str(path)) == digest(path), 'recovery helper is not sealed')
    spec = importlib.util.spec_from_file_location('authenticated_recovery_worker', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def add_pin(pins, path, expected=None):
    path = str(Path(path).resolve())
    actual = digest(path)
    require(expected is None or actual == expected, 'execution chain file changed: ' + path)
    require(path not in pins or pins[path] == actual, 'conflicting execution chain pin: ' + path)
    pins[path] = actual
    return actual


def authenticate_submission_chain(helper):
    require(digest(LEDGER) == LEDGER_SHA256, 'canonical recovery submission ledger changed')
    seal, ledger = read(SEAL), read(LEDGER)
    plan = read(CAMPAIGN / 'protocol.json')
    helper.verify_recovery_seal(plan, seal)
    require(len(seal['files_sha256']) == 1065 and seal['preserved_batch_count'] == 163
            and seal['new_rng_blocks_added'] == 0, 'recovery seal inventory differs')
    require(ledger['schema'] == 'modebench_level3_regular_queue_recovery_submission_ledger_v1'
            and ledger['status'] == 'all_eleven_submitted'
            and ledger['implementation_seal_path'] == str(SEAL)
            and ledger['implementation_seal_sha256'] == SEAL_SHA256
            and ledger['scientific_seal_sha256'] == SCIENTIFIC_SHA256
            and ledger['scientific_draws_added'] is False
            and ledger['preserved_original_batches'] == 163, 'recovery ledger contract differs')
    pins = dict(seal['files_sha256'])
    add_pin(pins, SEAL, SEAL_SHA256)
    add_pin(pins, LEDGER, LEDGER_SHA256)
    protocol_sha = add_pin(pins, CAMPAIGN / 'protocol.json', ledger['protocol_sha256'])
    require(ledger['protocol_path'] == str(CAMPAIGN / 'protocol.json'), 'ledger protocol path differs')
    records = ledger['jobs']
    require(len(records) == len(plan['jobs']) == 11, 'recovery must cover eleven cells')
    old_ids = [job['old_job_id'] for job in plan['jobs']]
    new_ids = [record['new_job_id'] for record in records]
    require(len(set(new_ids)) == 11 and not set(new_ids) & set(old_ids)
            and all(type(value) is str and re.fullmatch(r'[1-9][0-9]*', value) for value in new_ids),
            'replacement scheduler identities are not distinct genuine new IDs')
    claim_path = CAMPAIGN / 'development_execution_claim.json'
    add_pin(pins, claim_path, ledger['execution_claim_sha256'])
    claim = read(claim_path)
    require(claim['seal_sha256'] == SEAL_SHA256 and claim['protocol_sha256'] == protocol_sha
            and claim['scientific_seal_sha256'] == SCIENTIFIC_SHA256
            and claim['jobs'] == 11 and claim['old_job_ids'] == old_ids, 'recovery campaign claim differs')
    completion_path = CAMPAIGN / 'submission_completion.json'
    add_pin(pins, completion_path, ledger['submission_completion_sha256'])
    completion = read(completion_path)
    require(completion['jobs'] == 11 and completion['old_job_ids'] == old_ids
            and completion['new_job_ids'] == new_ids and completion['seal_sha256'] == SEAL_SHA256,
            'recovery submission completion differs')
    cancellation_path = CAMPAIGN / 'cancellation_verification.json'
    require(ledger['cancellation_verification_path'] == str(cancellation_path), 'cancellation path differs')
    add_pin(pins, cancellation_path, ledger['cancellation_verification_sha256'])
    command = ['scancel', '--state=PENDING', *old_ids]
    result_path = CAMPAIGN / 'cancellation_result.json'
    add_pin(pins, result_path)
    result = read(result_path)
    require(result['command'] == command and result['old_job_ids'] == old_ids
            and result['returncode'] == 0 and result['seal_sha256'] == SEAL_SHA256,
            'registered cancellation result differs')
    preserved = 0
    for index, (job, record) in enumerate(zip(plan['jobs'], records)):
        require(record['index'] == index and type(record['index']) is int, 'ledger cell index differs')
        for key in ('old_job_id', 'old_campaign', 'old_index', 'name', 'domain', 'model_label', 'tasks', 'output', 'old_seal_sha256'):
            require(record[key] == job[key], 'ledger cell identity differs: ' + key)
        require(record['job_id'] == record['new_job_id'], 'ledger job ID alias differs')
        snapshot = seal['job_snapshots'][job['old_job_id']]
        require(record['original_saved_batches'] == snapshot['batch_count'], 'ledger preserved batch count differs')
        preserved += snapshot['batch_count']
        verify_preserved_runtime(job, snapshot, plan['models'][job['model_label']])
        helper.canonical_job(job)
        helper.cancellation_evidence(job, snapshot)
        cancel_intent_path = CAMPAIGN / f'cancellation_{index:02d}_intent.json'
        add_pin(pins, cancel_intent_path)
        cancel_intent = read(cancel_intent_path)
        require(cancel_intent['command'] == command and cancel_intent['old_job_id'] == job['old_job_id']
                and cancel_intent['snapshot_sha256'] == sha(snapshot)
                and cancel_intent['protocol_sha256'] == protocol_sha and cancel_intent['seal_sha256'] == SEAL_SHA256,
                'per-cell cancellation intent differs')
        submission_path = CAMPAIGN / f'submission_{index:02d}_result.json'
        require(record['submission_record'] == str(submission_path), 'ledger submission path differs')
        add_pin(pins, submission_path, record['submission_record_sha256'])
        helper.worker_submission_identity(index, job, record['new_job_id'], SEAL_SHA256)
        for kind in ('intent', 'result'):
            path = CAMPAIGN / f'submission_{index:02d}_{kind}.json'
            add_pin(pins, path)
            entry = read(path)
            require(entry['protocol_sha256'] == protocol_sha and entry['seal_sha256'] == SEAL_SHA256
                    and entry['task_sha256'] == job['tasks_sha256'], 'submission protocol/task/seal differs')
        require(record['worker_claim_path'] == str(CAMPAIGN / f'worker_{index:02d}_execution_claim.json'),
                'ledger worker claim path differs')
    require(preserved == 163, 'original committed batch coverage differs')
    return helper, plan, seal, ledger, pins


def scheduler_records(job_ids):
    command = ['sacct', '--duplicates', '-X', '-j', ','.join(job_ids), '-n', '-P', '-o', ','.join(ACCOUNT_FIELDS)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    records = {}
    for line in result.stdout.splitlines():
        parts = line.split('|')
        if parts and parts[-1] == '' and len(parts) == len(ACCOUNT_FIELDS) + 1:
            parts.pop()
        require(len(parts) == len(ACCOUNT_FIELDS), 'malformed accounting record')
        record = dict(zip(ACCOUNT_FIELDS, parts))
        job_id = record['JobIDRaw']
        require(job_id in job_ids, 'unexpected replacement accounting record')
        records.setdefault(job_id, []).append(record)
    require(set(records) == set(job_ids), 'authoritative replacement accounting is incomplete')
    result = {}
    for job_id, history in records.items():
        require(not any(row['State'].split()[0] in ACTIVE_STATES for row in history[:-1]),
                'ambiguous active historical replacement accounting record')
        result[job_id] = {**history[-1], 'accounting_history': history}
    return result


def completion_status(records):
    failed, waiting = [], []
    for job_id, record in records.items():
        state = record['State'].split()[0]
        if state == 'COMPLETED' and record['ExitCode'] == '0:0':
            continue
        if state in ACTIVE_STATES:
            waiting.append(job_id)
        else:
            failed.append(job_id)
    return ('failed' if failed else 'pending' if waiting else 'complete'), failed, waiting


def verify_engine_log(log_text, model_path):
    engine_lines = [line for line in log_text.splitlines() if 'Initializing a V0 LLM engine (v0.8.4)' in line]
    require(engine_lines and all(f"model='{model_path}'" in line and 'dtype=torch.float16' in line
            and 'tensor_parallel_size=1' in line and 'quantization=None' in line for line in engine_lines)
            and 'Using XFormers backend.' in log_text, 'actual model/FP16/V0/XFormers engine log differs')


def verify_preserved_runtime(job, snapshot, model_path):
    if not snapshot['batch_count']:
        return
    out_path = CAMPAIGN / 'source_evidence' / job['old_job_id'] / (job['old_job_id'] + '.out')
    err_path = out_path.with_suffix('.err')
    for path in (out_path, err_path):
        require(job['source_evidence'].get(str(path)) == digest(path), 'original runtime archive differs')
    text = out_path.read_text() + '\n' + err_path.read_text()
    verify_engine_log(text, model_path)
    events = []
    for line in text.splitlines():
        if line.startswith('{'):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('event') == 'worker_authenticated':
                events.append(event)
    require(events and all(event['job_id'] == job['old_job_id'] and event['cell'] == job['name']
            and event['seal_sha256'] == job['old_seal_sha256']
            and event['gpu_names'] == ['Quadro RTX 6000'] for event in events),
            'original worker GPU/identity runtime differs')


def verify_runtime_evidence(job, record, claim, account, log_text):
    job_id = record['new_job_id']
    require(account['JobIDRaw'] == job_id and account['State'] == 'COMPLETED' and account['ExitCode'] == '0:0',
            'replacement did not complete successfully')
    require(account['Partition'] == 'all' and account['AllocCPUS'] == '6'
            and account['ReqMem'] in ('48G', '48Gn') and account['Timelimit'] == '01:00:00'
            and account['NodeList'] not in ('', 'None assigned', 'Unknown')
            and account['Start'] not in ('', 'Unknown') and account['End'] not in ('', 'Unknown')
            and 'gres/gpu:rtx_6000=1' in account['ReqTRES']
            and 'gres/gpu:rtx_6000=1' in account['AllocTRES'], 'completed runtime resources differ')
    def check_runtime(runtime, *, final=False):
        scheduler = runtime['scheduler']
        require(runtime['gpu_names'] == ['Quadro RTX 6000'] and runtime['attention_backend'] == 'XFORMERS'
                and runtime['engine'] == 'V0' and runtime['partition_preempt_mode'] == 'OFF',
                'worker GPU/backend/engine runtime differs')
        require(scheduler['JobId'] == job_id and scheduler['JobState'] == 'RUNNING'
                and scheduler['Partition'] == 'all' and scheduler['NumCPUs'] == '6'
                and scheduler['MinMemoryNode'] == '48G' and scheduler['TimeLimit'] == '01:00:00'
                and 'gpu:rtx_6000:1' in scheduler.get('TresPerNode', ''), 'worker scheduler claim differs')
        if final:
            require(scheduler['NodeList'] == account['NodeList'], 'final worker allocation differs from accounting')
    check_runtime(claim['runtime'])
    events = []
    for line in log_text.splitlines():
        if line.startswith('{'):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('event') == 'recovery_worker_authenticated':
                events.append(event)
    require(events and all(event['old_job_id'] == job['old_job_id'] and event['new_job_id'] == job_id
            and event['cell'] == job['name'] and event['recovery_seal_sha256'] == SEAL_SHA256
            and event['scientific_seal_sha256'] == SCIENTIFIC_SHA256 for event in events),
            'actual worker authentication log differs')
    require(events[0]['runtime'] == claim['runtime'] or events[0].get('resumed_same_new_id') is True,
            'first worker runtime differs from immutable ownership claim without same-ID resume')
    for event in events:
        check_runtime(event['runtime'])
    check_runtime(events[-1]['runtime'], final=True)
    verify_engine_log(log_text, job['_model_path'])


def grade_completed_receipt(receipt, rows):
    """Replay the original grader; publication requires this full CPU-only check."""
    from oat_drgrpo.math_grader import validated_modebench_outcome_key
    require(len(receipt['prompt_results']) == len(rows), 'regrade requires full source rows')
    attempts = 0
    for row, recorded in zip(rows, receipt['prompt_results']):
        require(recorded['row_sha256'] == sha(row), 'regrade source row differs')
        require(len(recorded['draws']) == 4, 'regrade requires all four draws')
        for draw in recorded['draws']:
            require(len(draw['attempts']) == 8, 'regrade requires all eight attempts')
            for attempt in draw['attempts']:
                require(isinstance(attempt['text'], str), 'recorded sample text must be a string')
                key = validated_modebench_outcome_key(attempt['text'], row['answer'])
                require(sha(key) == sha(attempt['canonical_key']) and attempt['verified'] is (key is not None),
                        'original grader disagrees with recorded attempt')
                attempts += 1
    return attempts


def verify_completed_samples(helper, job, snapshot, *, regrade=True):
    checked = helper.validate_snapshot(job, snapshot, allow_progress=True)
    require(checked['batch_count'] == checked['expected_batches'] and checked['existing_receipt_sha256'],
            'completed replacement lacks full batch/receipt coverage')
    task = read(job['tasks'])[0]
    rows, _ = load_rows(task)
    receipt = read(job['output'])
    require(receipt['sampling'] == {**receipt['identity']['interface'], 'seeds': task['seeds']}
            and receipt['information_boundary']['evaluation_prompts_loaded'] is False
            and receipt['information_boundary']['treatment_training_started'] is False,
            'receipt sampling/information boundary differs')
    require(receipt['answer_mode_histogram'] == dict(Counter(str(row.get('answer_mode_count')) for row in rows)),
            'receipt source support histogram differs')
    folder = Path(job['output'] + '.batches')
    seen, attempts = set(), 0
    for path in sorted(folder.glob('seed-*.json')):
        batch = read(path)
        for index, draw in enumerate(batch['draws'], batch['start']):
            label_index = task['seeds'].index(batch['seed'])
            coordinate = (index, label_index)
            require(coordinate not in seen, 'duplicated committed draw')
            seen.add(coordinate)
            require(sha(draw) == sha(receipt['prompt_results'][index]['draws'][label_index]),
                    'receipt draw differs from committed cached draw')
            for attempt in draw['attempts']:
                require(isinstance(attempt['text'], str), 'recorded sample text must be a string')
                require(type(attempt['verified']) is bool and attempt['verified'] == (attempt['canonical_key'] is not None),
                        'recorded verification flag and canonical key differ')
                require(type(attempt['token_count']) is int and 0 <= attempt['token_count'] <= receipt['identity']['interface']['max_tokens'],
                        'recorded token count differs from frozen output bound')
                attempts += 1
            count = sum(attempt['verified'] for attempt in draw['attempts'])
            distinct = len({sha(attempt['canonical_key']) for attempt in draw['attempts'] if attempt['verified']})
            for metric, expected in {'verified_count': count, 'pass1': count / 8, 'pass8': float(count > 0), 'distinct8': distinct}.items():
                actual = draw[metric]
                require(type(actual) in (int, float) and actual == expected, 'regraded draw metric differs')
    rebuilt = []
    for index, (row, recorded) in enumerate(zip(rows, receipt['prompt_results'])):
        spec = json.loads(row['answer']) if isinstance(row['answer'], str) else row['answer']
        reconstructed = {'row_index': index, 'row_sha256': sha(row), 'problem_sha256': sha(row['problem']),
            'spec_sha256': sha(spec), 'row_metadata': {key: value for key, value in row.items() if key not in ('problem', 'answer')},
            'draws': recorded['draws']}
        reconstructed.update({metric: statistics.mean(draw[metric] for draw in recorded['draws'])
                              for metric in ('pass1', 'pass8', 'distinct8')})
        require(sha(reconstructed) == sha(recorded), 'receipt row/source metadata or derived metric differs')
        rebuilt.append(reconstructed)
    from evaluate_modebench_level3 import summarize
    require(sha(receipt['metrics']) == sha(summarize(rebuilt)), 'receipt summary differs from regraded attempts')
    require(seen == {(index, label) for index in range(len(rows)) for label in range(4)}
            and attempts == len(rows) * 32, 'complete prompt/draw/attempt coverage differs')
    if regrade:
        require(grade_completed_receipt(receipt, rows) == attempts, 'full original-grader replay coverage differs')
    return checked, attempts


def audit_recovery_execution(*, publish=False, regrade=True):
    require(not publish or regrade, 'publishing requires full original-grader replay')
    helper, plan, seal, ledger, pins = authenticate_submission_chain(recovery_helper())
    accounts = scheduler_records([record['new_job_id'] for record in ledger['jobs']])
    status, failed, waiting = completion_status(accounts)
    if status != 'complete':
        return {'status': status, 'failed_job_ids': failed, 'waiting_job_ids': waiting,
                'scheduler_records': accounts, 'completion_attestation_published': False}
    records, attempts, rows_count, request_blocks = [], 0, 0, set()
    for index, (job, entry) in enumerate(zip(plan['jobs'], ledger['jobs'])):
        snapshot = seal['job_snapshots'][job['old_job_id']]
        claim_path = CAMPAIGN / f'worker_{index:02d}_execution_claim.json'
        add_pin(pins, claim_path)
        claim = read(claim_path)
        require(claim['identity'] == helper.recovery_claim_identity(job, entry['new_job_id'], SEAL_SHA256, snapshot),
                'completed worker ownership claim differs')
        log_path = ROOT / 'var/logs/modebench_level3' / (entry['new_job_id'] + '.out')
        err_path = log_path.with_suffix('.err')
        add_pin(pins, log_path)
        add_pin(pins, err_path)
        verify_runtime_evidence({**job, '_model_path': plan['models'][job['model_label']]}, entry, claim,
                                accounts[entry['new_job_id']], log_path.read_text() + '\n' + err_path.read_text())
        checked, count = verify_completed_samples(helper, job, snapshot, regrade=regrade)
        schedule = read(job['output'])['identity']['seed_schedule']
        blocks = [base for row in schedule['request_seeds'] for base in row]
        require(len(blocks) == len(set(blocks)) and not set(blocks) & request_blocks,
                'recovered logical RNG blocks overlap another registered cell')
        request_blocks.update(blocks)
        for path, expected in checked['cache_snapshot'].items():
            add_pin(pins, path, expected)
        add_pin(pins, job['output'], checked['existing_receipt_sha256'])
        attempts += count
        rows_count += checked['stable_identity']['rows']
        records.append({'name': job['name'], 'old_job_id': job['old_job_id'], 'new_job_id': entry['new_job_id'],
            'tasks': job['tasks'], 'output': job['output'], 'worker_claim_path': str(claim_path),
            'initial_snapshot_sha256': sha(snapshot), 'final_snapshot': checked,
            'scheduler': accounts[entry['new_job_id']], 'runtime': claim['runtime'],
            'attempts_regraded': count if regrade else 0})
    require(rows_count == 1344 and attempts == 43008 and len(request_blocks) == 5376, 'eleven-cell attempt inventory differs')
    for path in (Path(__file__).resolve(), ROOT / 'tests/test_modebench_level3_recovery_execution.py'):
        add_pin(pins, path)
    payload = {'schema': SCHEMA, 'status': 'complete', 'recovery_seal_path': str(SEAL),
        'recovery_seal_sha256': SEAL_SHA256, 'scientific_seal_sha256': SCIENTIFIC_SHA256,
        'submission_ledger_path': str(LEDGER), 'submission_ledger_sha256': LEDGER_SHA256,
        'jobs': records, 'rows': rows_count, 'attempts': attempts, 'preserved_original_batches': 163,
        'logical_request_blocks': 5376, 'new_logical_request_blocks_added': 0,
        'scientific_development_sources': 33, 'scientific_development_request_blocks': 16640,
        'all_attempts_regraded_with_original_grader': regrade, 'confirmation_outcomes_loaded': False,
        'development_fit_choices_made': False, 'files_sha256': pins}
    # Detect mutation during the audit, including complete log/cache/receipt bytes.
    helper.verify_recovery_seal(plan, seal)
    for path, expected in pins.items():
        require(digest(path) == expected, 'execution evidence changed during completion audit: ' + path)
    if publish:
        require(not ATTESTATION.exists(), 'completion attestation already exists; use its validator')
        atomic_new(ATTESTATION, {**payload, 'created_at': datetime.now(timezone.utc).isoformat()})
    return payload


def validate_completion_attestation(path=ATTESTATION):
    path = Path(path).resolve()
    require(path == ATTESTATION.resolve(), 'canonical recovery completion attestation required')
    require(path.is_file(), 'all eleven recoveries must complete before confirmation sealing')
    before = digest(path)
    saved = read(path)
    require(saved.get('schema') == SCHEMA and saved.get('status') == 'complete', 'invalid recovery completion attestation')
    for source, expected in saved['files_sha256'].items():
        require(digest(source) == expected, 'attested execution chain changed: ' + source)
    require(saved['all_attempts_regraded_with_original_grader'] is True
            and sum(record['attempts_regraded'] for record in saved['jobs']) == 43008
            and all(record['attempts_regraded'] == record['final_snapshot']['stable_identity']['rows'] * 32
                    for record in saved['jobs']), 'published original-grader coverage differs')
    # Publication code cannot omit replay; its own digest and every graded byte
    # are pinned above. Reauthenticate unchanged samples/metrics and execution
    # semantics, without repeating the expensive symbolic grader on each worker.
    current = audit_recovery_execution(regrade=False)
    expected = {key: value for key, value in saved.items() if key != 'created_at'}
    expected['all_attempts_regraded_with_original_grader'] = False
    expected['jobs'] = [{**record, 'attempts_regraded': 0} for record in saved['jobs']]
    require(sha(current) == sha(expected),
            'completion attestation differs from reauthenticated current execution')
    require(digest(path) == before, 'completion attestation changed during validation')
    pins = dict(saved['files_sha256'])
    add_pin(pins, path, before)
    return {'path': str(path), 'sha256': before, 'files_sha256': pins,
            'recovery_seal_path': str(SEAL), 'recovery_seal_sha256': SEAL_SHA256,
            'scientific_seal_sha256': SCIENTIFIC_SHA256, 'jobs': len(saved['jobs']), 'attempts': saved['attempts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    require(not (args.publish and args.validate), 'choose publish or validate')
    if args.validate:
        result = validate_completion_attestation()
        print(json.dumps({key: value for key, value in result.items() if key != 'files_sha256'}, sort_keys=True))
        return 0
    result = audit_recovery_execution(publish=args.publish)
    if result['status'] == 'complete':
        print(json.dumps({key: value for key, value in result.items() if key not in ('files_sha256', 'jobs')}, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True))
    return {'complete': 0, 'pending': 2, 'failed': 3}[result['status']]


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({'status': 'blocked_execution_integrity', 'detail': str(error)}), file=sys.stderr)
        raise SystemExit(1)
