#!/usr/bin/env python3
"""Continue only the registered Level3-v2 pipeline after an explicit local start.

No arguments is read-only. --prepare pins this implementation and its inputs;
--advance performs available phases once; --watch repeats pending observations.
Both action modes require the prepared seal hash. Pantry fitting additionally
requires --own-pantry-fit; the existing MathIR and Python fit owners are retained.
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
HERE = CAMPAIGN / 'continuation'
SEAL = HERE / 'seal.json'
RESULT = HERE / 'result.json'
RECOVERY = CAMPAIGN / 'regular_queue_recovery'
RECOVERY_SEAL = RECOVERY / 'implementation_seal.json'
RECOVERY_SHA = '65b3990674e8560d084f328a4e189709bfc99cd012e8e6620efd81879c8d2cab'
RECOVERY_LEDGER = RECOVERY / 'development_jobs.json'
LEDGER_SHA = 'bfcfc53886ecc0b44e52a8c5c8657a3e09d399d7b85f29bc5d6ed71c0516035d'
READINESS = RECOVERY / 'completion_audit_readiness.json'
READINESS_SHA = 'f98706ad856797aefd2b28eba19c19114c72babd4698665755e6157070812ed4'
CONFIRMATION = CAMPAIGN / 'confirmation'
DATASET = ROOT / 'var/data/modebench_level3_matched_v2'
MAPPING = CAMPAIGN / 'accepted_recipes.json'
REPORT = CONFIRMATION / 'confirmation_report.json'
PYTHON = ROOT / 'var/seed_paper_eval/paper310/bin/python'
FINALIZER = ROOT / 'ops/exp_scaling/finalize_modebench_level3_independent.py'
FITTER = ROOT / 'ops/exp_scaling/fit_modebench_level3_independent.py'
MATHIR_WATCHER = ROOT / 'ops/exp_scaling/watch_modebench_level3_mathir_v2.py'
MATHIR_WATCHER_SHA = '17a628d2d6cc491e212e405dfcea192ef52dccc6bcd30f6eec8676ee5a358086'
AUDITOR = ROOT / 'ops/audit_modebench_level3_independent_match.py'
COMPLETION = ROOT / 'ops/audit_modebench_level3_recovery_execution.py'
LAUNCHER = CONFIRMATION / 'launch.py'
TESTS = ROOT / 'tests/test_continue_modebench_level3_v2.py'
DOMAINS = ('countdown', 'graph_coloring', 'mathir', 'python_factors', 'pantry')
RECIPES = {domain: CAMPAIGN / relative for domain, relative in zip(DOMAINS, (
    'recipes/countdown.json', 'graph_v7/recipe.json', 'recipes/mathir.json',
    'python_v5/recipe.json', 'recipes/pantry.json'))}
PANTRY_RECEIPTS = [ROOT / 'var/results/modebench_level3_v2' / name for name in (
    'calibration_05b_pantry.json', *(f'calibration_3b_pantry_d{tier}.json' for tier in range(4)))]
PHASES = ('pantry_fit', 'accepted_recipes', 'finalize', 'recovery_attestation',
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
                    and scheduler['JobName'] == 'mb-l3-v2-confirm-' + job['name']
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


def fixed_configuration():
    return {'recipes': {domain: str(path) for domain, path in RECIPES.items()},
            'pantry_receipts': [str(path) for path in PANTRY_RECEIPTS],
            'external_fit_owners': {'mathir': str(MATHIR_WATCHER), 'python_factors': 'root_autofit_17081'},
            'dataset': str(DATASET), 'mapping': str(MAPPING), 'report': str(REPORT),
            'recovery_job_ids': list(NEW_IDS), 'phases': list(PHASES)}


def prepare():
    require(not SEAL.exists(), 'continuation seal already exists')
    require(digest(RECOVERY_SEAL) == RECOVERY_SHA and digest(RECOVERY_LEDGER) == LEDGER_SHA,
            'registered recovery seal or ledger changed')
    require(digest(READINESS) == READINESS_SHA, 'reviewed completion readiness changed')
    require(digest(MATHIR_WATCHER) == MATHIR_WATCHER_SHA, 'reviewed MathIR-only fitting owner changed')
    inherited = read(RECOVERY_SEAL)
    files = merge_pins(inherited['files_sha256'], read(READINESS)['files_sha256'], pins([RECOVERY_SEAL, RECOVERY_LEDGER, READINESS, Path(__file__), TESTS,
                       FINALIZER, FITTER, MATHIR_WATCHER, AUDITOR, COMPLETION, LAUNCHER,
                       ROOT / 'tests/test_modebench_independent_finalizer.py',
                       ROOT / 'tests/test_continue_modebench_level3_confirmation_runtime.py',
                       ROOT / 'tests/test_watch_modebench_level3_mathir_v2.py',
                       CONFIRMATION / 'pairs.json', CAMPAIGN / 'confirmation_control_amendment.json',
                       RECIPES['countdown'], RECIPES['graph_coloring']]))
    plan = read(CONFIRMATION / 'plan.json')
    files = merge_pins(files, plan['immutable_inputs_sha256'], pins(job['tasks'] for job in plan['jobs']))
    verify_pins(files, inherited['directory_files'])
    verify_models(inherited['models'])
    # All late imports come from the inherited scientific closure or the reviewed
    # finalizer/auditors/confirmation implementation pinned above.
    finalizer = module(FINALIZER, 'continuation_finalizer')
    for domain in ('countdown', 'graph_coloring'):
        finalizer.load_recipe(RECIPES[domain], domain)
    require(not RESULT.exists() and not any(HERE.glob('*_intent.json')),
            'continuation action journal exists before preparation')
    payload = {'schema': 'modebench_level3_fixed_continuation_seal_v1', 'created_at': now(),
               'configuration': fixed_configuration(), 'recovery_seal_sha256': RECOVERY_SHA,
               'files_sha256': files, 'directory_files': inherited['directory_files'],
               'models': inherited['models']}
    atomic_new(SEAL, payload)
    return {'status': 'prepared_not_started', 'seal': str(SEAL), 'seal_sha256': digest(SEAL)}


def verify_seal(expected):
    require(expected and digest(SEAL) == expected, 'explicit continuation seal hash required')
    seal = read(SEAL)
    require(seal.get('schema') == 'modebench_level3_fixed_continuation_seal_v1'
            and seal['configuration'] == fixed_configuration()
            and seal['recovery_seal_sha256'] == RECOVERY_SHA, 'continuation configuration changed')
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
    def __init__(self, seal_sha, *, own_pantry_fit=False):
        self.seal_sha = seal_sha
        self.own_pantry_fit = own_pantry_fit

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
            expected_revision = {'graph_coloring': 'graph_v7', 'python_factors': 'python_v5'}.get(domain)
            require(recipe.get('candidate_revision', {}).get('name') == expected_revision,
                    f'{domain}: unexpected registered candidate revision')
            result_root = ROOT / 'var/results/modebench_level3_v2'
            candidate = {'graph_coloring': 'graph_v7', 'python_factors': 'python_v5'}.get(domain, domain)
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

    def pantry_fit(self):
        require(self.own_pantry_fit, 'explicit Pantry fitting ownership required')
        auditor = module(AUDITOR, 'continuation_auditor')
        receipt_pins = pins(PANTRY_RECEIPTS)
        for index, path in enumerate(PANTRY_RECEIPTS):
            auditor.validate_receipt(read(path), domain='pantry', role='baseline' if index == 0 else 'candidate', development=True)
        verify_pins(receipt_pins)
        def action():
            require(not RECIPES['pantry'].exists(), 'Pantry recipe appeared under another owner')
            self.command('pantry_fit', [PYTHON, FITTER, '--baseline', PANTRY_RECEIPTS[0], '--scores',
                         *PANTRY_RECEIPTS[1:], '--domain', 'pantry', '--output', RECIPES['pantry']])
            verify_pins(receipt_pins)
            return {'recipe': str(RECIPES['pantry'])}, {**receipt_pins, **pins([RECIPES['pantry']])}
        return self.phase('pantry_fit', {'receipts_sha256': receipt_pins}, action)

    def confirmation_submissions(self):
        launcher = module(LAUNCHER, 'continuation_confirmation')
        seal_sha = digest(CONFIRMATION / 'seal.json')
        launcher.authenticate_saved_seal(seal_sha)
        plan = read(CONFIRMATION / 'plan.json')
        claim = CAMPAIGN / 'confirmation_execution_claim.json'
        value = read(claim)
        require(value['jobs'] == 10 and value['seal_sha256'] == seal_sha
                and value['plan_sha256'] == digest(CONFIRMATION / 'plan.json'), 'confirmation execution claim differs')
        files, job_ids = pins([claim]), []
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
            prior_ids = set(NEW_IDS) | {job['old_job_id'] for job in read(RECOVERY_LEDGER)['jobs']}
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
                return self.finish('needs_execution_review', missing_receipts=missing, scheduler_records=accounts)
        for domain, path in RECIPES.items():
            if path.exists() and read(path).get('development_fit_pass') is not True:
                return self.finish('needs_calibration_revision', domain=domain, recipe=str(path), recipe_sha256=digest(path))
        if not RECIPES['pantry'].exists() and all(path.is_file() for path in PANTRY_RECEIPTS):
            if not self.own_pantry_fit:
                return {'status': 'waiting_for_explicit_pantry_fit_ownership'}
            self.pantry_fit()
            if read(RECIPES['pantry']).get('development_fit_pass') is not True:
                return self.finish('needs_calibration_revision', domain='pantry', recipe=str(RECIPES['pantry']),
                                   recipe_sha256=digest(RECIPES['pantry']))
        missing = [domain for domain, path in RECIPES.items() if not path.is_file()]
        if missing:
            return {'status': 'waiting_for_registered_recipes', 'missing_domains': missing}
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
            require(not (CAMPAIGN / 'confirmation_execution_claim.json').exists()
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
            return self.finish('needs_execution_review', missing_receipts=missing, scheduler_records=accounts)
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
    parser.add_argument('--own-pantry-fit', action='store_true')
    args = parser.parse_args(argv)
    if not (args.prepare or args.advance or args.watch):
        require(not args.own_pantry_fit, 'fit ownership requires explicit --advance or --watch')
        print(json.dumps(readonly_status(), sort_keys=True), flush=True)
        return 0
    require(not args.prepare or not args.own_pantry_fit, '--prepare does not start or own fitting')
    with coordinator_lock():
        if args.prepare:
            print(json.dumps(prepare(), sort_keys=True), flush=True)
            return 0
        driver = Driver(args.seal_sha256, own_pantry_fit=args.own_pantry_fit)
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
