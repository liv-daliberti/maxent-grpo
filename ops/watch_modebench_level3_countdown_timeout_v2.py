#!/usr/bin/env python3
"""One same-ID TIMEOUT recovery for two explicitly authorized v2 Countdown jobs.

Only --watch can mutate scheduler state. No replacement submissions, resource
changes, score decisions, or other job IDs are supported. Persistent claims and
per-batch pins survive watcher restarts. Any ambiguous state or integrity failure
stops that job without taking a scheduler action.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'var/artifacts/modebench_level3_v2'
RECOVERY = CAMPAIGN / 'countdown_timeout_recovery'
SEAL_PATH = CAMPAIGN / 'implementation_seal.json'
SEAL_SHA = 'e03ffa74a476638401ddadb49b950f3d74377457114bf13b630ed8834be34736'
PROTOCOL_SHA = '783cacd9cff61d4ae25d6d02c10fb188f686a2cb539334e89dc78d0c3f2dc7a7'
AUTHORIZED = {'31146855': 3, '31146856': 4}
POLL_SECONDS = 30
TEST_PATH = ROOT / 'artifacts/test_modebench_level3_countdown_timeout_watch.py'
TOKENIZERS = {}
SHAPE_KEYS = ('UserId', 'GroupId', 'Account', 'QOS', 'Partition', 'TimeLimit',
              'ExcNodeList', 'NumCPUs', 'CPUs/Task', 'MinMemoryNode', 'TresPerNode',
              'Command', 'SubmitLine', 'WorkDir', 'StdOut', 'StdErr', 'NumNodes')
LIVE_STATES = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
               'STOPPED', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'RESIZING', 'SIGNALING'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def pin_once(path, payload):
    if Path(path).exists():
        require(read(path) == payload, f'persistent recovery pin changed: {path}')
    else:
        atomic_new(path, payload)


def run(command):
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20, check=False)


def record_process(result, command):
    return {'command': command, 'returncode': result.returncode, 'stdout': result.stdout,
            'stderr': result.stderr, 'captured_at': now()}


def parse_job(text):
    matches = list(re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/-]*)=', text.strip()))
    return {match.group(1): text.strip()[match.end():matches[index + 1].start() if index + 1 < len(matches) else None].strip()
            for index, match in enumerate(matches)}


def inspect_job(jobid, runner=run):
    require(jobid in AUTHORIZED, 'job ID outside explicit recovery authorization')
    command = ['scontrol', 'show', 'job', jobid, '-o']
    result = runner(command)
    record = record_process(result, command)
    require(result.returncode == 0, f'controller record unavailable for {jobid}; no recovery permitted')
    fields = parse_job(result.stdout)
    require(fields.get('JobId') == jobid and fields.get('JobState'), 'ambiguous controller job identity/state')
    return fields, record


def validate_scheduler(fields, jobid, original_command):
    require(jobid in AUTHORIZED and fields.get('JobId') == jobid, 'unauthorized scheduler identity')
    require(fields.get('UserId', '').endswith(f'({os.getuid()})'), 'scheduler job is not owned by current user')
    required = {'Requeue': '1', 'Partition': 'lowprio', 'TimeLimit': '01:00:00',
                'ExcNodeList': 'node103', 'NumCPUs': '6', 'CPUs/Task': '6', 'MinMemoryNode': '48G',
                'TresPerNode': 'gres/gpu:rtx_6000:1', 'NumNodes': '1', 'WorkDir': str(ROOT),
                'Command': str(CAMPAIGN / 'worker.slurm'),
                'StdOut': str(ROOT / f'var/logs/modebench_level3/{jobid}.out'),
                'StdErr': str(ROOT / f'var/logs/modebench_level3/{jobid}.err')}
    require(all(fields.get(key) == value for key, value in required.items()), 'scheduler resources/paths/requeue eligibility changed')
    require(fields.get('BatchFlag', '').isdigit() and int(fields['BatchFlag']) >= 1,
            'scheduler record is not a retained batch job')
    require(fields.get('Restarts', '').isdigit(), 'scheduler restart count unavailable')
    require(shlex.split(fields.get('SubmitLine', '')) == original_command, 'original scheduler submission command changed')
    return {key: fields[key] for key in SHAPE_KEYS}


def verify_sealed_inputs():
    require(digest(SEAL_PATH) == SEAL_SHA, 'calibration seal hash changed')
    seal = read(SEAL_PATH)
    require(len(seal['files_sha256']) == 358, 'expected exactly 358 frozen calibration file pins')
    for path, expected in seal['files_sha256'].items():
        require(digest(path) == expected, f'frozen calibration file changed: {path}')
    source = CAMPAIGN / 'launch.py'
    spec = importlib.util.spec_from_file_location('sealed_countdown_recovery_launcher', source)
    launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
    launcher.verify_seal(seal)
    require(digest(CAMPAIGN / 'protocol.json') == PROTOCOL_SHA, 'calibration protocol hash changed')
    return seal, launcher


def validate_cached_batch(batch, *, run_sha, label, start, end, schedule, evaluator):
    require(batch.get('identity_sha256') == run_sha and type(batch.get('seed')) is int
            and batch['seed'] == label and type(batch.get('start')) is int and batch['start'] == start
            and type(batch.get('end')) is int and batch['end'] == end
            and isinstance(batch.get('draws'), list) and len(batch['draws']) == end - start
            and batch.get('draws_sha256') == evaluator.sha(batch['draws']), 'invalid cached batch identity/hash/coverage')
    draw_index = schedule['draw_labels'].index(label)
    for index, draw in enumerate(batch['draws'], start):
        evaluator.validate_draw_seed_metadata(draw, schedule, index, draw_index)
        attempts = draw.get('attempts')
        require(isinstance(attempts, list) and len(attempts) == 8, 'cached draw lacks exactly eight attempts')
        for attempt in attempts:
            require(type(attempt.get('verified')) is bool and attempt['verified'] == (attempt.get('canonical_key') is not None)
                    and isinstance(attempt.get('text'), str) and type(attempt.get('token_count')) is int
                    and 0 <= attempt['token_count'] <= 192, 'invalid cached attempt metadata')
        correct = sum(attempt['verified'] for attempt in attempts)
        distinct = len({evaluator.sha(attempt['canonical_key']) for attempt in attempts if attempt['verified']})
        require(type(draw.get('verified_count')) is int and draw['verified_count'] == correct, 'invalid cached verified count')
        for metric, expected in [('pass1', correct / 8), ('pass8', float(correct > 0)), ('distinct8', float(distinct))]:
            require(type(draw.get(metric)) in (int, float) and math.isfinite(draw[metric])
                    and abs(draw[metric] - expected) <= 1e-12, 'cached metric integrity mismatch')


def audit_batches(jobid, job, seal, job_dir):
    import evaluate_modebench_level3_independent as evaluator
    task = read(job['tasks'])[0]
    rows, source = evaluator.load_rows(task)
    require(len(rows) == 128 and source['total_rows'] == 128, 'full registered Countdown source required')
    schedule = evaluator.schedule_record('countdown', rows, task['seeds'])
    batch_dir = Path(job['output'] + '.batches')
    run_path = batch_dir / 'run.json'
    manifest = read(run_path)
    identity, run_sha = manifest['identity'], manifest['identity_sha256']
    require(run_sha == evaluator.sha(identity), 'run manifest identity hash mismatch')
    model_path = seal['models']['3b']['path']
    if model_path not in TOKENIZERS:
        from transformers import AutoTokenizer
        TOKENIZERS[model_path] = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    interface = evaluator.frozen_interface('countdown')
    prompts = [TOKENIZERS[model_path].apply_chat_template(
        evaluator.prompt_messages('countdown', row['problem'], interface['prompt_profile']),
        tokenize=False, add_generation_prompt=True) for row in rows]
    expected = {'schema': evaluator.SCHEMA, 'domain': 'countdown', 'level': 'level3', 'split': 'dev',
                'model': seal['models']['3b'], 'interface': evaluator.frozen_interface('countdown'),
                'source': source, 'seeds': task['seeds'], 'batch_size': 8,
                'code_sha256': evaluator.code_identity(), 'seed_schedule': schedule,
                'seed_schedule_sha256': evaluator.sha(schedule), 'sampling_engine': evaluator.ENGINE_CONTRACT,
                'interface_sha256': evaluator.sha(interface), 'rendered_prompts_sha256': evaluator.sha(prompts)}
    require(identity == expected, 'run manifest differs from frozen scientific inputs')
    pin_once(job_dir / 'run_manifest_pin.json', {'path': str(run_path), 'sha256': digest(run_path), 'identity_sha256': run_sha})
    valid_names = {f'seed-{label}__rows-{start:06d}-{start + 8:06d}.json': (label, start, start + 8)
                   for label in task['seeds'] for start in range(0, 128, 8)}
    paths = sorted(batch_dir.glob('seed-*.json'))
    require(all(path.name in valid_names for path in paths), 'unexpected cached batch filename')
    pins_dir = job_dir / 'batch_pins'
    old_names = {p.name[:-len('.pin.json')] for p in pins_dir.glob('*.pin.json')} if pins_dir.exists() else set()
    require(old_names <= {path.name for path in paths}, 'previously saved batch disappeared')
    hashes = {}
    for path in paths:
        before = digest(path)
        pinned = pins_dir / (path.name + '.pin.json')
        pin = {'path': str(path), 'sha256': before}
        if pinned.exists():
            require(read(pinned) == pin, 'previously saved batch changed')
        label, start, end = valid_names[path.name]
        validate_cached_batch(read(path), run_sha=run_sha, label=label, start=start, end=end,
                              schedule=schedule, evaluator=evaluator)
        require(digest(path) == before, 'cached batch changed during validation')
        pin_once(pinned, pin)
        hashes[str(path)] = before
    return {'run_manifest_sha256': digest(run_path), 'run_identity_sha256': run_sha,
            'completed_batches': len(paths), 'total_batches': 64, 'batch_files_sha256': hashes,
            'checks_use_integrity_only': True, 'correctness_scores_recorded_or_used_for_recovery': False}


def preserve_file(source, destination):
    require(Path(source).is_file(), f'required pre-requeue log missing: {source}')
    before = digest(source)
    with Path(source).open('rb') as src, Path(destination).open('xb') as dst:
        while block := src.read(1 << 20):
            dst.write(block)
        dst.flush(); os.fsync(dst.fileno())
    require(digest(source) == before == digest(destination), f'log changed during TIMEOUT snapshot: {source}')
    return {'source': str(source), 'copy': str(destination), 'sha256': before}


class Watcher:
    def __init__(self, directory=RECOVERY, runner=run):
        self.directory, self.runner = Path(directory), runner
        self.active = set(AUTHORIZED)
        self.status = {}

    def emit(self, event, **details):
        payload = {'event': event, 'time': now(), **details}
        print(json.dumps(payload, sort_keys=True), flush=True)
        with (self.directory / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(payload, sort_keys=True) + '\n'); handle.flush()

    def finish(self, jobid, reason):
        self.active.discard(jobid)
        self.emit('job_watch_finished', job_id=jobid, reason=reason)

    def authenticate_job(self, jobid, seal, launcher):
        require(jobid in AUTHORIZED, 'unauthorized recovery job')
        index = AUTHORIZED[jobid]
        job = read(CAMPAIGN / 'protocol.json')['jobs'][index]
        require(job['name'] == '3b_countdown_d' + str(index - 1), 'authorized cell index changed')
        expected_command = launcher.command_for(index, job, SEAL_SHA)
        submission = read(CAMPAIGN / f'submission_{index:02d}_result.json')
        require(submission['returncode'] == 0 and submission['stdout'].strip().split(';', 1)[0] == jobid
                and submission['command'] == expected_command, 'original successful submission identity changed')
        identity = {'job_id': jobid, 'seal_sha256': SEAL_SHA, 'task_sha256': digest(job['tasks'])}
        require(read(CAMPAIGN / f'worker_{index:02d}_execution_claim.json')['identity'] == identity,
                'worker execution ownership changed')
        claim = read(CAMPAIGN / 'development_execution_claim.json')
        require(claim['seal_sha256'] == SEAL_SHA and claim['protocol_sha256'] == PROTOCOL_SHA, 'campaign claim changed')
        job_dir = self.directory / jobid; job_dir.mkdir(parents=True, exist_ok=True)
        files = [CAMPAIGN / f'submission_{index:02d}_{kind}.json' for kind in ('intent', 'result')]
        files += [CAMPAIGN / f'worker_{index:02d}_execution_claim.json', CAMPAIGN / 'development_execution_claim.json']
        pin_once(job_dir / 'ownership_pins.json', {str(path): digest(path) for path in files})
        return job, expected_command, job_dir

    def recover(self, jobid, job, original_command, job_dir, sealed, fields, scheduler_record, batches):
        require(fields.get('JobState') == 'TIMEOUT', 'only actual TIMEOUT can trigger recovery')
        require(not Path(job['output']).exists(), 'final receipt exists; recovery forbidden')
        intent_path = job_dir / 'requeue_intent.json'
        require(not intent_path.exists(), 'requeue was already attempted; automatic retry forbidden')
        snapshot = job_dir / ('before_requeue_' + stamp()); snapshot.mkdir()
        atomic_new(snapshot / 'scheduler_timeout.json', scheduler_record)
        command = ['sacct', '--duplicates', '-X', '-j', jobid,
                   '--format=JobIDRaw,State,ExitCode,Elapsed,Start,End,NodeList', '-P']
        accounting = self.runner(command)
        atomic_new(snapshot / 'accounting.json', record_process(accounting, command))
        require(accounting.returncode == 0, 'accounting snapshot failed; recovery refused')
        require(any(line.split('|')[0] == jobid and line.split('|')[1].split()[0] == 'TIMEOUT'
                    for line in accounting.stdout.splitlines() if '|' in line), 'accounting does not confirm TIMEOUT')
        logs = [preserve_file(fields[key], snapshot / suffix) for key, suffix in [('StdOut', 'stdout.log'), ('StdErr', 'stderr.log')]]
        atomic_new(snapshot / 'batches.json', batches)
        atomic_new(snapshot / 'preservation.json', {'logs': logs, 'ownership_pins': read(job_dir / 'ownership_pins.json'),
                                                  'seal_sha256': SEAL_SHA, 'recorded_at': now()})
        verify_sealed_inputs()
        current_batches = audit_batches(jobid, job, sealed, job_dir)
        require(current_batches == batches, 'batch inventory changed after TIMEOUT snapshot')
        fresh_fields, fresh_record = inspect_job(jobid, self.runner)
        require(fresh_fields['JobState'] == 'TIMEOUT', 'job state changed before requeue')
        shape = validate_scheduler(fresh_fields, jobid, original_command)
        require(shape == read(job_dir / 'scheduler_shape.json'), 'scheduler resources changed before requeue')
        require(not Path(job['output']).exists(), 'receipt appeared before requeue')
        atomic_new(snapshot / 'scheduler_final_check.json', fresh_record)
        command = ['scontrol', 'requeue', jobid]
        atomic_new(intent_path, {'created_at': now(), 'job_id': jobid, 'command': command,
                                'reason': 'confirmed_actual_TIMEOUT_only', 'snapshot': str(snapshot),
                                'seal_sha256': SEAL_SHA, 'prior_restarts': int(fresh_fields['Restarts']),
                                'authorized_attempt_limit': 1, 'resources_changed': False})
        # An exception or ambiguous scheduler response leaves the immutable
        # intent in place. Neither this process nor a restart may retry it.
        result = self.runner(command)
        atomic_new(job_dir / 'requeue_result.json', record_process(result, command))
        require(result.returncode == 0, 'owner requeue rejected; no automatic retry')
        after, after_record = inspect_job(jobid, self.runner)
        atomic_new(job_dir / 'scheduler_after_requeue.json', after_record)
        require(after['JobState'] in LIVE_STATES and int(after.get('Restarts', '-1')) > int(fresh_fields['Restarts']),
                'requeue acceptance or restart count is ambiguous; no further action')
        require(validate_scheduler(after, jobid, original_command) == shape, 'resource settings changed during requeue')
        self.emit('same_id_requeue_accepted', job_id=jobid, completed_batches=batches['completed_batches'],
                  restarts=int(after['Restarts']), snapshot=str(snapshot))

    def poll_job(self, jobid, sealed, launcher):
        job, original_command, job_dir = self.authenticate_job(jobid, sealed, launcher)
        fields, scheduler_record = inspect_job(jobid, self.runner)
        shape = validate_scheduler(fields, jobid, original_command)
        pin_once(job_dir / 'scheduler_shape.json', shape)
        state = fields['JobState']
        if Path(job['output']).exists():
            atomic_new(job_dir / ('observed_receipt_' + stamp() + '.json'),
                       {'state': state, 'receipt_path': job['output'], 'receipt_sha256': digest(job['output']),
                        'scores_read': False, 'requeue_action': False})
            self.finish(jobid, 'final_receipt_exists_no_action')
            return
        if state != 'TIMEOUT' and state not in LIVE_STATES:
            atomic_new(job_dir / ('terminal_' + stamp() + '.json'), scheduler_record)
            self.finish(jobid, 'terminal_' + state + '_no_action')
            return
        if state == 'TIMEOUT' and (job_dir / 'requeue_intent.json').exists():
            self.finish(jobid, 'TIMEOUT_after_existing_requeue_intent_no_second_attempt')
            return
        batches = audit_batches(jobid, job, sealed, job_dir)
        value = (state, batches['completed_batches'], fields['Restarts'])
        if self.status.get(jobid) != value:
            self.status[jobid] = value
            self.emit('job_observation', job_id=jobid, state=state, completed_batches=batches['completed_batches'],
                      restarts=int(fields['Restarts']))
        if state == 'TIMEOUT':
            self.recover(jobid, job, original_command, job_dir, sealed, fields, scheduler_record, batches)

    def watch(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / '.watcher.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            own_pins = {str(Path(__file__).resolve()): digest(__file__), str(TEST_PATH): digest(TEST_PATH)}
            pin_once(self.directory / 'watcher_implementation_pins.json', own_pins)
            atomic_new(self.directory / ('watcher_start_' + stamp() + '.json'),
                       {'pid': os.getpid(), 'authorized_jobs': AUTHORIZED, 'poll_seconds': POLL_SECONDS,
                        'seal_sha256': SEAL_SHA, 'created_at': now(), 'maximum_owner_requeues_per_job': 1})
            self.emit('watcher_started', pid=os.getpid(), authorized_jobs=sorted(AUTHORIZED), poll_seconds=POLL_SECONDS)
            while self.active:
                cycle_started = time.monotonic()
                for path, expected in own_pins.items():
                    require(digest(path) == expected, 'watcher implementation changed while armed')
                sealed, launcher = verify_sealed_inputs()
                for jobid in sorted(self.active):
                    try:
                        self.poll_job(jobid, sealed, launcher)
                    except (ValueError, OSError, subprocess.SubprocessError, KeyError, TypeError) as error:
                        job_dir = self.directory / jobid; job_dir.mkdir(parents=True, exist_ok=True)
                        atomic_new(job_dir / ('blocked_' + stamp() + '.json'), {'error': str(error), 'created_at': now(), 'job_id': jobid})
                        self.finish(jobid, 'fail_closed: ' + str(error))
                if self.active:
                    time.sleep(max(0, POLL_SECONDS - (time.monotonic() - cycle_started)))
            self.emit('watcher_complete', pid=os.getpid())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='arm the explicitly authorized same-ID TIMEOUT recovery')
    args = parser.parse_args()
    require(args.watch, 'explicit --watch required; no default scheduler mutations')
    watcher = Watcher()
    try:
        watcher.watch()
    except Exception as error:
        if RECOVERY.is_dir():
            atomic_new(RECOVERY / ('watcher_fatal_' + stamp() + '.json'), {'error': str(error), 'created_at': now()})
        raise


if __name__ == '__main__':
    main()
