#!/usr/bin/env python3
"""Continue only the registered Level3-v2 pipeline after an explicit local start.

No arguments is read-only. --prepare pins this implementation and its inputs;
--advance performs available phases once; --watch repeats pending observations.
Both action modes require the prepared seal hash. The four passing recipes are retained; the sealed Python v6 fitter and independent
development auditor own their outputs. This driver waits for their proof.
An interrupted action is never replayed automatically, even if its output exists.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'var/artifacts/modebench_level3_v2'
HERE = CAMPAIGN / 'continuation_python_v6'
SEAL = HERE / 'seal.json'
RESULT = HERE / 'result.json'
RECOVERY = CAMPAIGN / 'regular_queue_recovery'
RECOVERY_SEAL = RECOVERY / 'implementation_seal.json'
RECOVERY_SHA = '65b3990674e8560d084f328a4e189709bfc99cd012e8e6620efd81879c8d2cab'
RECOVERY_LEDGER = RECOVERY / 'development_jobs.json'
LEDGER_SHA = 'bfcfc53886ecc0b44e52a8c5c8657a3e09d399d7b85f29bc5d6ed71c0516035d'
READINESS = RECOVERY / 'completion_audit_readiness.json'
READINESS_SHA = 'f98706ad856797aefd2b28eba19c19114c72babd4698665755e6157070812ed4'
CONFIRMATION = CAMPAIGN / 'confirmation_python_v6'
DATASET = ROOT / 'var/data/modebench_level3_matched_v2'
MAPPING = HERE / 'accepted_recipes.json'
REPORT = CONFIRMATION / 'confirmation_report.json'
PYTHON = ROOT / 'var/seed_paper_eval/paper310/bin/python'
FINALIZER = ROOT / 'ops/exp_scaling/finalize_modebench_level3_python_v6_independent.py'
FITTER = ROOT / 'ops/exp_scaling/fit_modebench_level3_independent.py'
MATHIR_WATCHER = ROOT / 'ops/exp_scaling/watch_modebench_level3_mathir_v2.py'
MATHIR_WATCHER_SHA = '17a628d2d6cc491e212e405dfcea192ef52dccc6bcd30f6eec8676ee5a358086'
AUDITOR = ROOT / 'ops/audit_modebench_level3_python_v6_independent_match.py'
COMPLETION = ROOT / 'ops/audit_modebench_level3_recovery_execution.py'
PYTHON_WATCHER = ROOT / 'ops/exp_scaling/watch_modebench_level3_python_v6.py'
V6_AUDITOR = ROOT / 'ops/audit_modebench_level3_python_v6_development.py'
V6_AUDIT = CAMPAIGN / 'python_v6/independent_completed_development_audit.json'
V6_SEAL = CAMPAIGN / 'python_v6/implementation_seal.json'
V6_SHA = '7f3a49ae3314ef314accac14be7668c920454b8d9206ca07026e81c772a0f266'
V6_LAUNCHER = CAMPAIGN / 'python_v6/launch.py'
V6_CLAIM = CAMPAIGN / 'python_v6/development_execution_claim.json'
V6_FIT_FAILURE = CAMPAIGN / 'python_v6/development_fit_failure.json'
INTEGRATION_READINESS = CONFIRMATION / 'integration_readiness.json'
INTEGRATION_SHA = '5f6e3aa3eef1b6782ffb8a6a7fad62f36b2b4693c1907fee61d819d24abc5440'
OWNER_STARTS = {str(CAMPAIGN / 'python_v6' / name): sha for name, sha in (
    ('fit_watcher_start_intent.json', '4326ba853c63905c073623c2337052c3cacca6a0039fbbf918d2ca20c0413c7f'),
    ('fit_watcher_start_result.json', '17619ce2335fe205014335cf547e15adee191e2f2706e1e313b51786972e3a4b'),
    ('audit_start_intent.json', '5ce0704ea23a86dba47fbb659ddb6e243266db413a3b43e502227029dba3d0d1'),
    ('audit_start.json', '6596696cbc82f497cffaa62d187100b85a6b3cd6dbd7a21c7bf6359b9492c86e'),
)}
LAUNCHER = CONFIRMATION / 'launch.py'
TESTS = ROOT / 'tests/test_continue_modebench_level3_python_v6.py'
DOMAINS = ('countdown', 'graph_coloring', 'mathir', 'python_factors', 'pantry')
RECIPES = {domain: CAMPAIGN / relative for domain, relative in zip(DOMAINS, (
    'recipes/countdown.json', 'graph_v7/recipe.json', 'recipes/mathir.json',
    'python_v6/recipe.json', 'recipes/pantry.json'))}
PHASES = ('python_v6_development_attestation', 'accepted_recipes', 'finalize', 'recovery_attestation',
          'confirmation_prepare', 'confirmation_submit', 'confirmation_audit')
NEW_IDS = ('31149147', '31149148', '31149149', '31149150', '31149151', '31149152',
           '31149153', '31149154', '31149155', '31149156', '31149158')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def pins(paths):
    return {str(Path(path).resolve()): digest(path) for path in paths}


def tree_pins(path):
    return pins(sorted(p for p in Path(path).rglob('*') if p.is_file()))


def merge_pins(*mappings):
    result = {}
    for mapping in mappings:
        for path, expected in mapping.items():
            require(path not in result or result[path] == expected, f'conflicting source pin: {path}')
            result[path] = expected
    return result


def parse_job_id(stdout):
    require(isinstance(stdout, str) and re.fullmatch(r'[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?', stdout.strip()),
            'ambiguous scheduler response; root review required')
    return stdout.strip().split(';', 1)[0]


def verify_pins(files, directories=None):
    for path, expected in files.items():
        require(digest(path) == expected, f'input/output bytes changed: {path}')
    for directory, expected in (directories or {}).items():
        actual = sorted(str(p.resolve()) for p in Path(directory).rglob('*') if p.is_file())
        require(actual == sorted(expected), f'input inventory changed: {directory}')


def module(path, name):
    for directory in ('ops', 'ops/exp_scaling', 'src'):
        if str(ROOT / directory) not in sys.path:
            sys.path.insert(0, str(ROOT / directory))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def verify_models(models):
    evaluator = module(ROOT / 'ops/evaluate_modebench_level3.py', 'continuation_evaluator')
    require(set(models) == {'05b', '3b'}, 'exactly both frozen models required')
    for label, expected in models.items():
        current = evaluator.model_identity(Path(expected['path']), label)
        current['vllm_version'] = importlib.metadata.version('vllm')
        require(current == expected, f'sealed model or engine changed: {label}')


def authenticate_confirmation_execution(plan, job_ids, accounts, seal_sha):
    """Bind all ten completed workers to their submitted IDs and actual runtime."""
    require(len(plan['jobs']) == len(job_ids) == len(set(job_ids)) == 10
            and set(accounts) == set(job_ids), 'exact ten-worker execution inventory required')
    completion = module(COMPLETION, 'continuation_completion')
    files, records = {}, []
    for index, (job, job_id) in enumerate(zip(plan['jobs'], job_ids)):
        account = accounts[job_id]
        require(account['JobIDRaw'] == job_id and account['State'] == 'COMPLETED' and account['ExitCode'] == '0:0'
                and account['Partition'] == 'all' and account['AllocCPUS'] == '6'
                and account['ReqMem'] in ('48G', '48Gn') and account['Timelimit'] == '01:00:00'
                and account['NodeList'] not in ('', 'Unknown', 'None assigned')
                and account['Start'] not in ('', 'Unknown') and account['End'] not in ('', 'Unknown')
                and 'gres/gpu:rtx_6000=1' in account['ReqTRES']
                and 'gres/gpu:rtx_6000=1' in account['AllocTRES'], 'confirmation completed resources differ')
        claim_path = CONFIRMATION / f'worker_{index:02d}_execution_claim.json'
        out = ROOT / 'var/logs/modebench_level3' / (job_id + '.out')
        err = out.with_suffix('.err')
        current = pins([claim_path, out, err])
        claim = read(claim_path)
        require(claim['identity'] == {'job_id': job_id, 'seal_sha256': seal_sha, 'task_sha256': digest(job['tasks'])},
                'confirmation worker ownership differs')
        def check_runtime(runtime, gpu_names, final=False):
            scheduler = runtime['scheduler']
            require(gpu_names in (['Quadro RTX 6000'], ['Quadro RTX6000'])
                    and runtime['partition_preempt_mode'] == 'OFF'
                    and runtime['attention_backend'] == 'XFORMERS' and runtime['engine'] == 'V0',
                    'confirmation actual GPU/backend differs')
            require(scheduler['JobId'] == job_id and scheduler['JobState'] == 'RUNNING'
                    and scheduler['UserId'].endswith(f'({os.getuid()})')
                    and scheduler['JobName'] == 'mb-l3-v2-confirm-pyv6-' + job['name']
                    and scheduler['Command'] == str(CONFIRMATION / 'worker.slurm')
                    and scheduler['Partition'] == 'all' and scheduler['QOS'] in ('normal', 'none')
                    and scheduler['NumCPUs'] == '6' and scheduler['MinMemoryNode'] == '48G'
                    and scheduler['TimeLimit'] == '01:00:00'
                    and 'gpu:rtx_6000:1' in scheduler.get('TresPerNode', ''), 'confirmation actual scheduler identity differs')
            if final:
                require(scheduler['NodeList'] == account['NodeList'], 'confirmation final allocation differs')
        check_runtime(claim['runtime'], claim['gpu_names'])
        text = out.read_text() + '\n' + err.read_text()
        events = []
        for line in text.splitlines():
            if line.startswith('{'):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get('event') == 'worker_authenticated':
                    events.append(event)
        require(events, 'confirmation authentication log missing')
        for event in events:
            require(event['job_id'] == job_id and event['cell'] == job['name']
                    and event['seal_sha256'] == seal_sha, 'confirmation runtime event belongs to another worker')
            check_runtime(event['runtime'], event['gpu_names'])
        if events[0]['runtime'] != claim['runtime']:
            require(int(events[0]['runtime']['scheduler']['Restarts']) > int(claim['runtime']['scheduler']['Restarts']),
                    'confirmation runtime changed without same-ID restart')
        check_runtime(events[-1]['runtime'], events[-1]['gpu_names'], final=True)
        completion.verify_engine_log(text, plan['models'][job['model_label']])
        verify_pins(current)
        files = merge_pins(files, current)
        records.append({'job_id': job_id, 'cell': job['name'], 'claim': str(claim_path),
                        'claim_sha256': digest(claim_path), 'accounting': account,
                        'authenticated_runtime_events': len(events)})
    return {'jobs': records, 'seal_sha256': seal_sha, 'all_ten_workers_authenticated': True}, files


def authenticate_python_v6_submissions():
    """Bind four actual v6 IDs to the immutable development submission chain."""
    require(V6_SHA != 'NOT_READY' and digest(V6_SEAL) == V6_SHA,
            'registered Python v6 development seal changed')
    sealed = read(V6_SEAL)
    require(sealed['files_sha256'].get(str(V6_LAUNCHER)) == digest(V6_LAUNCHER),
            'Python v6 launcher is not in the fixed development seal')
    launcher = module(V6_LAUNCHER, 'continuation_python_v6_development_launcher')
    launcher.authenticate_saved_seal(V6_SHA)
    plan_path = V6_LAUNCHER.with_name('protocol.json')
    plan = read(plan_path)
    launcher.check_static(plan)
    claim = read(V6_CLAIM)
    require(claim.get('jobs') == 4 and claim.get('candidate_revision') == 'python_v6'
            and claim.get('seal_sha256') == V6_SHA
            and claim.get('protocol_sha256') == digest(plan_path),
            'Python v6 development execution claim differs')
    require(len(plan['jobs']) == 4, 'exactly four Python v6 development jobs required')
    files = pins([V6_CLAIM, plan_path])
    job_ids = []
    prior_ids = set(NEW_IDS) | {job['old_job_id'] for job in read(RECOVERY_LEDGER)['jobs']}
    for index, job in enumerate(plan['jobs']):
        intent_path = V6_CLAIM.with_name(f'submission_{index:02d}_intent.json')
        result_path = V6_CLAIM.with_name(f'submission_{index:02d}_result.json')
        intent, result = read(intent_path), read(result_path)
        command = launcher.command_for(index, job, V6_SHA)
        require(intent['command'] == result['command'] == command
                and intent['cell'] == result['cell'] == job['name']
                and intent['task_sha256'] == digest(job['tasks'])
                and intent['seal_sha256'] == V6_SHA and result['returncode'] == 0,
                'ambiguous Python v6 development submission; root review required')
        job_id = parse_job_id(result['stdout'])
        require(job_id not in job_ids and job_id not in prior_ids,
                'invalid or repeated Python v6 scheduler ID')
        job_ids.append(job_id)
        files = merge_pins(files, pins([intent_path, result_path]))
    ledger = V6_CLAIM.with_name('development_jobs.json')
    if ledger.exists():
        files = merge_pins(files, pins([ledger]))
    return {'job_ids': job_ids, 'seal_sha256': V6_SHA, 'outputs': [job['output'] for job in plan['jobs']]}, files


def fixed_configuration():
    return {'recipes': {domain: str(path) for domain, path in RECIPES.items()},
            'external_fit_owners': {'python_factors': str(PYTHON_WATCHER)},
            'dataset': str(DATASET), 'mapping': str(MAPPING), 'report': str(REPORT),
            'recovery_job_ids': list(NEW_IDS), 'phases': list(PHASES),
            'python_v6_seal':str(V6_SEAL),'python_v6_seal_sha256':V6_SHA,
            'python_v6_completed_audit':str(V6_AUDIT),
            'owner_start_records_sha256': OWNER_STARTS, 'integration_readiness_sha256': INTEGRATION_SHA}


def prepare():
    require(not SEAL.exists(), 'continuation seal already exists')
    require(digest(RECOVERY_SEAL) == RECOVERY_SHA and digest(RECOVERY_LEDGER) == LEDGER_SHA,
            'registered recovery seal or ledger changed')
    require(digest(READINESS) == READINESS_SHA, 'reviewed completion readiness changed')
    require(digest(MATHIR_WATCHER) == MATHIR_WATCHER_SHA, 'reviewed MathIR-only fitting owner changed')
    require(digest(V6_SEAL) == V6_SHA, 'registered Python v6 development seal changed')
    require(digest(INTEGRATION_READINESS) == INTEGRATION_SHA, 'reviewed future integration readiness changed')
    verify_pins(OWNER_STARTS)
    inherited = read(V6_SEAL)
    _, development_files = authenticate_python_v6_submissions()
    files = merge_pins(development_files, OWNER_STARTS, read(INTEGRATION_READINESS)['files_sha256'],
                       inherited['files_sha256'], read(READINESS)['files_sha256'], pins([RECOVERY_SEAL, RECOVERY_LEDGER, READINESS, Path(__file__), TESTS,
                       FINALIZER, FITTER, MATHIR_WATCHER, PYTHON_WATCHER, AUDITOR, COMPLETION, LAUNCHER, V6_SEAL, V6_AUDITOR, INTEGRATION_READINESS,
                       ROOT / 'tests/test_modebench_level3_python_v6_development_audit.py',
                       ROOT / 'tests/test_modebench_level3_python_v6_watcher.py',
                       ROOT / 'tests/test_modebench_python_v6_independent_finalizer.py',
                       ROOT / 'tests/test_continue_modebench_level3_confirmation_runtime.py',
                       ROOT / 'tests/test_watch_modebench_level3_mathir_v2.py',
                       CONFIRMATION / 'pairs.json', CONFIRMATION / 'plan.json', CAMPAIGN / 'confirmation_control_amendment.json',
                       RECIPES['countdown'], RECIPES['graph_coloring'], RECIPES['mathir'], RECIPES['pantry']]))
    plan = read(CONFIRMATION / 'plan.json')
    files = merge_pins(files, plan['immutable_inputs_sha256'], pins(job['tasks'] for job in plan['jobs']))
    verify_pins(files, inherited['directory_files'])
    verify_models(inherited['models'])
    # All late imports come from the inherited scientific closure or the reviewed
    # finalizer/auditors/confirmation implementation pinned above.
    finalizer = module(FINALIZER, 'continuation_finalizer')
    for domain in ('countdown', 'graph_coloring', 'mathir', 'pantry'):
        finalizer.load_recipe(RECIPES[domain], domain)
    require(not RESULT.exists() and not any(HERE.glob('*_intent.json')),
            'continuation action journal exists before preparation')
    payload = {'schema': 'modebench_level3_fixed_continuation_seal_v1', 'created_at': now(),
               'configuration': fixed_configuration(), 'recovery_seal_sha256': RECOVERY_SHA,
               'files_sha256': files, 'directory_files': inherited['directory_files'],
               'models': inherited['models'],'python_v6_seal_sha256':V6_SHA}
    atomic_new(SEAL, payload)
    return {'status': 'prepared_not_started', 'seal': str(SEAL), 'seal_sha256': digest(SEAL)}


def verify_seal(expected):
    require(expected and digest(SEAL) == expected, 'explicit continuation seal hash required')
    seal = read(SEAL)
    require(seal.get('schema') == 'modebench_level3_fixed_continuation_seal_v1'
            and seal['configuration'] == fixed_configuration()
            and seal['recovery_seal_sha256'] == RECOVERY_SHA and seal['python_v6_seal_sha256'] == V6_SHA, 'continuation configuration changed')
    require(seal['files_sha256'].get(str(Path(__file__).resolve())) == digest(__file__),
            'continuation implementation is not sealed')
    verify_pins(seal['files_sha256'], seal['directory_files'])
    require(seal['models'] == read(RECOVERY_SEAL)['models'], 'model binding differs from recovery seal')
    verify_models(seal['models'])
    return seal


@contextmanager
def coordinator_lock():
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / '.coordinator.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Driver:
    def __init__(self, seal_sha):
        self.seal_sha = seal_sha

    def verify(self):
        verify_seal(self.seal_sha)

    def journal(self):
        completed = {}
        for name in PHASES:
            intent, result = HERE / f'{name}_intent.json', HERE / f'{name}_result.json'
            require(intent.exists() == result.exists(), f'ambiguous interrupted phase {name}; root review required')
            if not intent.exists():
                continue
            record = read(result)
            require(read(intent)['seal_sha256'] == self.seal_sha
                    and record['intent_sha256'] == digest(intent)
                    and record['seal_sha256'] == self.seal_sha,
                    f'phase ownership changed: {name}')
            require(record['status'] == 'complete', f'failed phase {name}; root review required')
            verify_pins(record['files_sha256'])
            completed[name] = record
        return completed

    def phase(self, name, inputs, action):
        require(name in PHASES, 'unregistered continuation phase')
        completed = self.journal()
        if name in completed:
            require(read(HERE / f'{name}_intent.json')['inputs'] == inputs,
                    f'phase inputs changed: {name}')
            return completed[name]['evidence']
        self.verify()
        intent = HERE / f'{name}_intent.json'
        atomic_new(intent, {'created_at': now(), 'phase': name, 'seal_sha256': self.seal_sha, 'inputs': inputs})
        try:
            evidence, files = action()
            logs = [HERE / (name + suffix) for suffix in ('_command.json', '.stdout', '.stderr')]
            files = merge_pins(files, pins(path for path in logs if path.exists()))
            self.verify()
            verify_pins(files)
            record = {'status': 'complete', 'evidence': evidence, 'files_sha256': files}
        except BaseException as error:
            atomic_new(HERE / f'{name}_result.json', {'status': 'failed', 'created_at': now(),
                       'seal_sha256': self.seal_sha, 'intent_sha256': digest(intent),
                       'error': f'{type(error).__name__}: {error}'})
            raise
        atomic_new(HERE / f'{name}_result.json', {**record, 'created_at': now(),
                   'seal_sha256': self.seal_sha, 'intent_sha256': digest(intent)})
        return evidence

    def command(self, name, command):
        # Durable logs remain available even if the parent is interrupted. Such
        # interruption leaves an unmatched intent and is deliberately not retried.
        atomic_new(HERE / f'{name}_command.json', {'command': list(map(str, command)), 'cwd': str(ROOT),
                   'phase': name, 'seal_sha256': self.seal_sha, 'created_at': now()})
        with (HERE / f'{name}.stdout').open('x') as stdout, (HERE / f'{name}.stderr').open('x') as stderr:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stdout, stderr=stderr)
        require(result.returncode == 0, f'{name} returned {result.returncode}; inspect preserved logs')

    def finish(self, status, **details):
        self.verify()
        payload = {'status': status, 'created_at': now(), 'seal_sha256': self.seal_sha, **details}
        atomic_new(RESULT, payload)
        return payload

    def authenticate_recipes(self):
        finalizer = module(FINALIZER, 'continuation_finalizer')
        before = pins(RECIPES.values())
        for domain, path in RECIPES.items():
            recipe = finalizer.load_recipe(path, domain)
            expected_revision = {'graph_coloring': 'graph_v7', 'python_factors': 'python_v6'}.get(domain)
            require(recipe.get('candidate_revision', {}).get('name') == expected_revision,
                    f'{domain}: unexpected registered candidate revision')
            result_root = ROOT / 'var/results/modebench_level3_v2'
            candidate = {'graph_coloring': 'graph_v7', 'python_factors': 'python_v6'}.get(domain, domain)
            require(recipe['provenance']['baseline_receipt_path'] == str(result_root / f'calibration_05b_{domain}.json')
                    and [recipe['provenance']['pools'][str(tier)]['receipt_path'] for tier in range(4)]
                    == [str(result_root / f'calibration_3b_{candidate}_d{tier}.json') for tier in range(4)],
                    f'{domain}: recipe uses different registered development receipts')
        verify_pins(before)
        return before

    def authenticate_dataset(self, recipe_pins):
        launcher = module(LAUNCHER, 'continuation_confirmation')
        identity, _, _ = launcher.validate_dataset_before_confirmation(read(CONFIRMATION / 'plan.json'))
        require(identity['recipe_sha256'] == {domain: recipe_pins[str(path)] for domain, path in RECIPES.items()},
                'published dataset uses a different accepted recipe')
        require(set(identity['domains']) == set(DOMAINS)
                and all(set(identity['domains'][domain]) == {'train', 'dev', 'eval'} for domain in DOMAINS),
                'all fifteen registered splits required')
        return {'splits_authenticated': 15, 'identity_sha256': digest(DATASET / 'identity.json')}

    def confirmation_submissions(self):
        launcher = module(LAUNCHER, 'continuation_confirmation')
        seal_sha = digest(CONFIRMATION / 'seal.json')
        launcher.authenticate_saved_seal(seal_sha)
        plan = read(CONFIRMATION / 'plan.json')
        claim = CONFIRMATION / 'confirmation_execution_claim.json'
        value = read(claim)
        require(value['jobs'] == 10 and value['seal_sha256'] == seal_sha
                and value['plan_sha256'] == digest(CONFIRMATION / 'plan.json'), 'confirmation execution claim differs')
        files, job_ids = pins([claim]), []
        development, _ = authenticate_python_v6_submissions()
        require(len(plan['jobs']) == 10, 'exactly ten confirmation jobs required')
        for index, job in enumerate(plan['jobs']):
            intent = CONFIRMATION / f'submission_{index:02d}_intent.json'
            result = CONFIRMATION / f'submission_{index:02d}_result.json'
            before, after = read(intent), read(result)
            command = launcher.command_for(index, job, seal_sha)
            require(before['command'] == after['command'] == command
                    and before['cell'] == after['cell'] == job['name']
                    and before['task_sha256'] == digest(job['tasks'])
                    and before['seal_sha256'] == seal_sha and after['returncode'] == 0,
                    'ambiguous confirmation submission; root review required')
            job_id = parse_job_id(after['stdout'])
            prior_ids = set(NEW_IDS) | set(development['job_ids']) | {job['old_job_id'] for job in read(RECOVERY_LEDGER)['jobs']}
            require(job_id not in job_ids and job_id not in prior_ids,
                    'invalid or repeated confirmation job ID')
            job_ids.append(job_id)
            files.update(pins([intent, result]))
        return {'job_ids': job_ids, 'seal_sha256': seal_sha}, files

    def tick(self):
        self.verify()
        completed = self.journal()
        if RESULT.exists():
            result = read(RESULT)
            require(result['seal_sha256'] == self.seal_sha, 'terminal result belongs to another seal')
            return result
        completion = module(COMPLETION, 'continuation_completion')
        accounts = completion.scheduler_records(list(NEW_IDS))
        state, failed, _ = completion.completion_status(accounts)
        if failed:
            return self.finish('needs_execution_review', failed_job_ids=failed, scheduler_records=accounts)
        if state == 'complete':
            missing = [job['output'] for job in read(RECOVERY_LEDGER)['jobs'] if not Path(job['output']).is_file()]
            if missing:
                return {'status': 'waiting_for_completed_recovery_receipt_visibility', 'missing_receipts': missing}
        development, _ = authenticate_python_v6_submissions()
        dev_accounts = completion.scheduler_records(development['job_ids'])
        dev_state, dev_failed, _ = completion.completion_status(dev_accounts)
        if dev_failed:
            return self.finish('needs_execution_review', failed_python_v6_job_ids=dev_failed, scheduler_records=dev_accounts)
        if dev_state == 'complete':
            missing = [path for path in development['outputs'] if not Path(path).is_file()]
            if missing:
                return {'status': 'waiting_for_python_v6_receipt_visibility', 'missing_receipts': missing}
        if V6_FIT_FAILURE.exists():
            return self.finish('needs_execution_review', python_v6_fit_failure=str(V6_FIT_FAILURE),
                               failure_sha256=digest(V6_FIT_FAILURE))
        for domain, path in RECIPES.items():
            if path.exists() and read(path).get('development_fit_pass') is not True:
                return self.finish('needs_calibration_revision', domain=domain, recipe=str(path), recipe_sha256=digest(path))
        missing = [domain for domain, path in RECIPES.items() if not path.is_file()]
        if missing:
            return {'status': 'waiting_for_registered_recipes', 'missing_domains': missing}
        if not V6_AUDIT.exists():
            return {'status': 'waiting_for_python_v6_independent_completed_development_audit'}
        def attest_v6():
            validator = module(V6_AUDITOR, 'continuation_python_v6_completed_auditor')
            evidence = validator.validate_completed_development_audit(path=V6_AUDIT)
            require(evidence['status'] == 'passed_development', 'Python v6 independent development audit did not pass')
            return evidence, merge_pins(evidence['files_sha256'], pins([V6_AUDIT]))
        self.phase('python_v6_development_attestation', {'scientific_seal_sha256': V6_SHA}, attest_v6)
        # Phase result pins authenticate unchanged recipes and code after first refit.
        recipe_pins = pins(RECIPES.values()) if 'accepted_recipes' in completed else self.authenticate_recipes()
        mapping = {domain: str(path) for domain, path in RECIPES.items()}
        def accepted():
            atomic_new(MAPPING, mapping)
            return {'mapping': str(MAPPING)}, {**recipe_pins, **pins([MAPPING])}
        self.phase('accepted_recipes', {'recipe_sha256': recipe_pins}, accepted)
        def finalize():
            require(not DATASET.exists(), 'fresh final dataset path required')
            self.command('finalize', [PYTHON, FINALIZER, '--recipes-json', MAPPING, '--output-root', DATASET,
                         '--protocol-amendment', CAMPAIGN / 'confirmation_control_amendment.json'])
            return self.authenticate_dataset(recipe_pins), tree_pins(DATASET)
        self.phase('finalize', {'mapping_sha256': digest(MAPPING)}, finalize)
        if state != 'complete':
            return {'status': 'waiting_for_completed_recovery_execution'}
        def attest():
            path = RECOVERY / 'completion_attestation.json'
            if not path.exists():
                outcome = completion.audit_recovery_execution(publish=True, regrade=True)
                require(outcome['status'] == 'complete', 'recovery completion changed before publication')
            evidence = completion.validate_completion_attestation()
            return evidence, evidence['files_sha256']
        self.phase('recovery_attestation', {'recovery_seal_sha256': RECOVERY_SHA}, attest)
        def prepare_confirmation():
            require(not (CONFIRMATION / 'seal.json').exists(), 'confirmation seal already exists outside this driver')
            self.command('confirmation_prepare', [PYTHON, LAUNCHER, '--prepare'])
            launcher = module(LAUNCHER, 'continuation_confirmation')
            current = digest(CONFIRMATION / 'seal.json')
            launcher.authenticate_saved_seal(current)
            return {'seal_sha256': current}, pins([CONFIRMATION / 'seal.json'])
        sealed = self.phase('confirmation_prepare', {'dataset_identity_sha256': digest(DATASET / 'identity.json')}, prepare_confirmation)
        def submit():
            require(not (CONFIRMATION / 'confirmation_execution_claim.json').exists()
                    and not list(CONFIRMATION.glob('submission_*_intent.json'))
                    and not list(CONFIRMATION.glob('submission_*_result.json')),
                    'confirmation submission evidence already exists; root review required')
            self.command('confirmation_submit', [PYTHON, LAUNCHER, '--submit', '--seal-sha256', sealed['seal_sha256']])
            return self.confirmation_submissions()
        submitted = self.phase('confirmation_submit', sealed, submit)
        current, _ = self.confirmation_submissions()
        require(current == submitted, 'confirmation submission IDs changed')
        accounts = completion.scheduler_records(submitted['job_ids'])
        state, failed, waiting = completion.completion_status(accounts)
        if failed:
            return self.finish('needs_execution_review', failed_job_ids=failed, scheduler_records=accounts)
        plan = read(CONFIRMATION / 'plan.json')
        if waiting:
            return {'status': 'waiting_for_confirmation', 'waiting_job_ids': waiting}
        missing = [job['output'] for job in plan['jobs'] if not Path(job['output']).is_file()]
        if missing:
            return {'status': 'waiting_for_confirmation_receipt_visibility', 'missing_receipts': missing}
        def audit():
            require(not REPORT.exists(), 'fresh final audit report required')
            execution, execution_files = authenticate_confirmation_execution(plan, submitted['job_ids'], accounts, sealed['seal_sha256'])
            execution_path = HERE / 'confirmation_completed_execution.json'
            atomic_new(execution_path, execution)
            execution_files = merge_pins(execution_files, pins([execution_path]))
            self.command('confirmation_audit', [PYTHON, AUDITOR, '--pairs-json', CONFIRMATION / 'pairs.json',
                         '--dataset-root', DATASET, '--output', REPORT])
            report = read(REPORT)
            require(report['status'] in ('observed_approximate_match', 'outside_match_tolerance')
                    and report['all_five_domains_complete'] is True and not report['errors'],
                    'final audit rejected evidence; root review required')
            return {'status': report['status'], 'report': str(REPORT), 'execution': str(execution_path)}, merge_pins(execution_files, pins([REPORT, *(job['output'] for job in plan['jobs'])]))
        outcome = self.phase('confirmation_audit', {'job_ids': submitted['job_ids']}, audit)
        return self.finish(outcome['status'], report=str(REPORT), report_sha256=digest(REPORT),
                           confirmation_match_verified=read(REPORT)['confirmation_match_verified'])


def readonly_status():
    return {'status': 'read_only', 'prepared': SEAL.exists(), 'terminal_result': str(RESULT) if RESULT.exists() else None,
            'registered_recipes_present': {domain: path.is_file() for domain, path in RECIPES.items()},
            'configuration': fixed_configuration(), 'scheduler_actions_taken': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--advance', action='store_true')
    actions.add_argument('--watch', action='store_true')
    parser.add_argument('--seal-sha256')
    args = parser.parse_args(argv)
    if not (args.prepare or args.advance or args.watch):
        print(json.dumps(readonly_status(), sort_keys=True), flush=True)
        return 0
    with coordinator_lock():
        if args.prepare:
            print(json.dumps(prepare(), sort_keys=True), flush=True)
            return 0
        driver = Driver(args.seal_sha256)
        while True:
            try:
                result = driver.tick()
            except Exception as error:
                # Never clear intents, rewrite results, or recover ambiguity.
                result = {'status': 'needs_root_review', 'error': f'{type(error).__name__}: {error}'}
                print(json.dumps(result, sort_keys=True), flush=True)
                return 1
            print(json.dumps(result, sort_keys=True), flush=True)
            if not result['status'].startswith('waiting_'):
                return 0 if result['status'] == 'observed_approximate_match' else 2
            if not args.watch:
                return 0
            time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main())
