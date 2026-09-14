#!/usr/bin/env python3
"""Continue the adaptive fixed-reference v3 campaign under one explicit owner.

Default invocation is read-only. Preparation seals the stable future pipeline
and eight actual DEV submissions; it does not fit, finalize, or submit work.
Only --advance/--watch with that explicit additive seal may execute phases.
An interrupted or failed action is preserved and never retried automatically.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
import modebench_level3_v3_common as common
from fit_modebench_level3 import local_dependency_sources

HERE = common.CAMPAIGN / 'continuation'
SEAL = HERE / 'seal.json'
RESULT = HERE / 'result.json'
SOURCE = Path(__file__).resolve()
TEST = ROOT / 'tests/test_continue_modebench_level3_v3.py'
PYTHON = ROOT / 'var/seed_paper_eval/paper310/bin/python'
DEV_LAUNCHER = ROOT / 'ops/exp_scaling/modebench_level3_v3_development.py'
FITTER = ROOT / 'ops/exp_scaling/fit_modebench_level3_fixed_reference.py'
AUDITOR = ROOT / 'ops/audit_modebench_level3_v3.py'
FINALIZER = ROOT / 'ops/exp_scaling/modebench_level3_v3_finalize.py'
CONF_LAUNCHER = ROOT / 'ops/exp_scaling/modebench_level3_v3_confirmation.py'
COMPLETION = ROOT / 'ops/audit_modebench_level3_recovery_execution.py'
DEV_SEAL = common.CAMPAIGN / 'development/seal.json'
DEV_AUDIT = common.CAMPAIGN / 'development/completed_development_audit.json'
CONFIRMATION = common.CAMPAIGN / 'confirmation'
REPORT = CONFIRMATION / 'confirmation_report.json'
RECIPES = {domain: common.CAMPAIGN / 'recipes' / (domain + '.json') for domain in common.REVISED}
SCHEMA = 'modebench_level3_v3_continuation_seal_v1'
PHASES = ('development_completed', 'fit_graph_coloring', 'fit_python_factors',
          'development_audit', 'finalize', 'confirmation_prepare',
          'confirmation_submit', 'confirmation_audit')
IMPLEMENTATION = (SOURCE, TEST, DEV_LAUNCHER, FITTER, AUDITOR, FINALIZER, CONF_LAUNCHER,
    ROOT / 'tests/test_modebench_level3_v3_development.py',
    ROOT / 'tests/test_fit_modebench_level3_fixed_reference.py',
    ROOT / 'tests/test_modebench_level3_v3_audit.py',
    ROOT / 'tests/test_modebench_level3_v3_finalize.py',
    ROOT / 'tests/test_modebench_level3_v3_confirmation.py')
require = common.require
read = common.read
digest = common.digest
atomic_new = common.atomic_new
merge_pins = common.merge_pins
verify_pins = common.verify_pins


def now():
    return datetime.now(timezone.utc).isoformat()


def pins(paths):
    return {str(Path(path).resolve()): digest(path) for path in paths}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def configuration():
    return {'contract': common.contract(), 'phases': list(PHASES),
            'development_jobs': 8, 'development_original_grader_attempts': 39424,
            'confirmation_jobs': 2, 'confirmation_original_grader_attempts': 8192,
            'retained_domains': list(common.RETAINED), 'revised_domains': list(common.REVISED),
            'recipe_paths': {domain: str(path) for domain, path in RECIPES.items()},
            'dataset_root': str(common.DATASET), 'terminal_result': str(RESULT),
            'research_authority': 'root', 'failed_recipe_policy': 'immutable_no_alternate_fit',
            'execution_ambiguity_policy': 'preserve_and_stop_without_retry',
            'historical_level1_role': 'frozen_benchmark_reference',
            'all_five_fresh_same_round': False, 'treatment_training_started': False}


def require_subset(expected_files, expected_trees, files, trees, message):
    require(all(files.get(path) == value for path, value in expected_files.items())
            and all(trees.get(path) == value for path, value in expected_trees.items()), message)


def no_future_work():
    require(not SEAL.exists() and not RESULT.exists() and not (HERE / 'prepare_intent.json').exists(),
            'existing continuation preparation or terminal record forbids retry')
    require(not any((HERE / (name + suffix)).exists() for name in PHASES
                    for suffix in ('_intent.json', '_result.json', '_command.json', '.stdout', '.stderr')),
            'continuation phase evidence exists before preparation')
    require(not any(path.exists() or path.with_suffix('.fit_intent.json').exists() for path in RECIPES.values())
            and not DEV_AUDIT.exists() and not common.DATASET.exists() and not REPORT.exists(),
            'future fit, audit or dataset work exists before additive sealing')
    require(not any((CONFIRMATION / name).exists() for name in
                    ('seal.json', 'plan.json', 'prepare_intent.json', 'confirmation_execution_claim.json'))
            and not list(CONFIRMATION.glob('submission_*_*.json'))
            and not (common.RESULTS / 'confirmation').exists(),
            'future confirmation ownership or sampling already exists')


def prepare(registration_sha256, scientific_seal_sha256):
    """Pin actual DEV submissions and stable future implementations, no actions."""
    no_future_work()
    for path in IMPLEMENTATION:
        require(path.is_file(), 'future implementation is not ready: ' + str(path))
    registration = common.validate_registration(common.REGISTRATION, registration_sha256)
    launcher = module(DEV_LAUNCHER, 'v3_continuation_prepare_development')
    scientific = launcher.authenticate_saved_seal(scientific_seal_sha256)
    require(scientific['registration_sha256'] == registration_sha256, 'DEV scientific registration differs')
    submissions = launcher.authenticate_submissions(scientific_seal_sha256)
    require(len(submissions['jobs']) == len(submissions['job_ids']) == len(set(submissions['job_ids'])) == 8,
            'all eight actual DEV submission chains required')
    files = merge_pins(scientific['files_sha256'], submissions['files_sha256'],
                      pins([common.REGISTRATION, DEV_SEAL, *IMPLEMENTATION]))
    for relative, expected in local_dependency_sources(list(IMPLEMENTATION)).items():
        files = merge_pins(files, {str((ROOT / relative).resolve()): expected})
    trees = dict(scientific['directory_files'])
    require_subset(registration['files_sha256'], registration['directory_files'], files, trees,
                   'additive seal omits scientific registration')
    verify_pins(files, trees)
    no_future_work()
    atomic_new(HERE / 'prepare_intent.json', {'schema': 'modebench_level3_v3_continuation_prepare_intent_v1',
        'created_at': now(), 'registration_sha256': registration_sha256,
        'scientific_seal_sha256': scientific_seal_sha256, 'source_sha256': digest(SOURCE)})
    files = merge_pins(files, pins([HERE / 'prepare_intent.json']))
    seal = {'schema': SCHEMA, 'created_at': now(), 'configuration': configuration(),
            'registration_path': str(common.REGISTRATION), 'registration_sha256': registration_sha256,
            'scientific_seal_path': str(DEV_SEAL), 'scientific_seal_sha256': scientific_seal_sha256,
            'development_submissions': submissions, 'models': scientific['models'],
            'files_sha256': files, 'directory_files': trees,
            'new_candidate_outcomes_loaded': False, 'new_actions_started': False}
    verify_pins(files, trees)
    atomic_new(SEAL, seal)
    return {'status': 'prepared_not_armed', 'seal': str(SEAL), 'seal_sha256': digest(SEAL),
            'registration_sha256': registration_sha256, 'scientific_seal_sha256': scientific_seal_sha256,
            'files': len(files), 'inventories': len(trees), 'development_job_ids': submissions['job_ids']}


def verify_seal(expected_sha256):
    require(isinstance(expected_sha256, str) and re.fullmatch(r'[a-f0-9]{64}', expected_sha256)
            and digest(SEAL) == expected_sha256, 'explicit saved additive seal hash required')
    seal = read(SEAL)
    require(seal['schema'] == SCHEMA and seal['configuration'] == configuration()
            and seal['registration_path'] == str(common.REGISTRATION)
            and seal['scientific_seal_path'] == str(DEV_SEAL)
            and seal['new_candidate_outcomes_loaded'] is False and seal['new_actions_started'] is False,
            'additive execution contract differs')
    registration = common.validate_registration(common.REGISTRATION, seal['registration_sha256'])
    launcher = module(DEV_LAUNCHER, 'v3_continuation_verify_development')
    scientific = launcher.authenticate_saved_seal(seal['scientific_seal_sha256'])
    require(scientific['registration_sha256'] == seal['registration_sha256']
            and scientific['models'] == seal['models'], 'scientific registration or models differ')
    submissions = launcher.authenticate_submissions(seal['scientific_seal_sha256'])
    require(submissions == seal['development_submissions'], 'actual DEV submissions changed')
    require_subset(scientific['files_sha256'], scientific['directory_files'],
                   seal['files_sha256'], seal['directory_files'], 'additive seal omits scientific inputs')
    require_subset(submissions['files_sha256'], {}, seal['files_sha256'], seal['directory_files'],
                   'additive seal omits actual submission ownership')
    required = pins([common.REGISTRATION, DEV_SEAL, HERE / 'prepare_intent.json', *IMPLEMENTATION])
    for relative, expected in local_dependency_sources(list(IMPLEMENTATION)).items():
        required = merge_pins(required, {str((ROOT / relative).resolve()): expected})
    require_subset(required, {}, seal['files_sha256'], seal['directory_files'], 'future implementation is not sealed')
    require(SEAL.as_posix() not in seal['files_sha256'], 'additive seal cannot pin itself')
    verify_pins(seal['files_sha256'], seal['directory_files'])
    require(digest(SEAL) == expected_sha256, 'additive seal changed during authentication')
    return seal


@contextmanager
def coordinator_lock():
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / '.coordinator.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Driver:
    def __init__(self, seal_sha256):
        self.seal_sha = seal_sha256

    def verify(self):
        return verify_seal(self.seal_sha)

    def command_binding(self):
        require(digest(SEAL) == self.seal_sha, 'command binding additive seal changed')
        return read(SEAL)

    def expected_command(self, name, inputs, seal):
        registration_sha = seal['registration_sha256']
        if name.startswith('fit_'):
            domain = name[len('fit_'):]
            require(domain in common.REVISED and inputs == {'registration_sha256': registration_sha, 'domain': domain},
                    'fit command inputs differ from fixed registered domain')
            command = [PYTHON, '-B', FITTER, '--registration', common.REGISTRATION,
                       '--registration-sha256', registration_sha, '--domain', domain, '--output', RECIPES[domain]]
        elif name in ('development_audit', 'confirmation_audit'):
            stage = name.removesuffix('_audit')
            if stage == 'development':
                require(inputs['scientific_seal_sha256'] == seal['scientific_seal_sha256'], 'DEV audit seal differs')
                scientific_sha = seal['scientific_seal_sha256']
            else:
                scientific_sha = read(HERE / 'confirmation_prepare_result.json')['evidence']['seal_sha256']
                require(inputs['scientific_seal_sha256'] == scientific_sha, 'CONF audit seal differs')
            command = [PYTHON, '-B', AUDITOR, '--stage', stage, '--seal-sha256', scientific_sha,
                       '--registration-sha256', registration_sha, '--source-sha256', digest(AUDITOR), '--publish']
        elif name == 'finalize':
            command = [PYTHON, '-B', FINALIZER, '--registration-sha256', registration_sha,
                       '--execution-seal', SEAL, '--execution-seal-sha256', self.seal_sha, '--publish']
        elif name == 'confirmation_prepare':
            command = [PYTHON, '-B', CONF_LAUNCHER, '--prepare', '--registration-sha256', registration_sha,
                       '--execution-seal-sha256', self.seal_sha]
        elif name == 'confirmation_submit':
            scientific_sha = read(HERE / 'confirmation_prepare_result.json')['evidence']['seal_sha256']
            require(inputs['seal_sha256'] == scientific_sha, 'CONF submission seal differs')
            command = [PYTHON, '-B', CONF_LAUNCHER, '--submit', '--seal-sha256', scientific_sha]
        else:
            raise ValueError('phase has no registered child command')
        return list(map(str, command))

    def validate_command(self, name, inputs, files, seal=None):
        if name == 'development_completed':
            return
        command_path = HERE / (name + '_command.json')
        paths = [command_path, HERE / (name + '.stdout'), HERE / (name + '.stderr')]
        require(all(files.get(str(path)) == digest(path) for path in paths), 'completed phase omits command or durable logs')
        record = read(command_path)
        expected = self.expected_command(name, inputs, self.command_binding() if seal is None else seal)
        require(record['phase'] == name and record['seal_sha256'] == self.seal_sha
                and record['cwd'] == str(ROOT) and record['command'] == expected,
                'completed child command differs from sealed phase contract')

    def journal(self):
        completed = {}
        gap = False
        for name in PHASES:
            intent, result = HERE / f'{name}_intent.json', HERE / f'{name}_result.json'
            require(intent.exists() == result.exists(), 'ambiguous interrupted phase ' + name + '; root review required')
            if not intent.exists():
                gap = True
                continue
            require(not gap, 'phase order differs from registered continuation')
            before, after = read(intent), read(result)
            require(before['phase'] == name and before['seal_sha256'] == self.seal_sha
                    and after['phase'] == name and after['seal_sha256'] == self.seal_sha
                    and after['intent_sha256'] == digest(intent), 'phase ownership changed: ' + name)
            require(after['status'] == 'complete', 'failed phase ' + name + '; root review required')
            verify_pins(after['files_sha256'], after['directory_files'])
            self.validate_command(name, before['inputs'], after['files_sha256'])
            completed[name] = after
        return completed

    def phase(self, name, inputs, action):
        require(name in PHASES, 'unregistered continuation phase')
        completed = self.journal()
        if name in completed:
            require(read(HERE / f'{name}_intent.json')['inputs'] == inputs, 'phase inputs changed: ' + name)
            return completed[name]['evidence']
        require(len(completed) < len(PHASES) and PHASES[len(completed)] == name, 'phase attempted out of order')
        self.verify()
        intent = HERE / f'{name}_intent.json'
        atomic_new(intent, {'created_at': now(), 'phase': name, 'seal_sha256': self.seal_sha, 'inputs': inputs})
        try:
            evidence, files, trees = action()
            logs = [HERE / (name + suffix) for suffix in ('_command.json', '.stdout', '.stderr')]
            files = merge_pins(files, pins(path for path in logs if path.exists()))
            binding = self.verify()
            verify_pins(files, trees)
            self.validate_command(name, inputs, files, binding)
            result = {'status': 'complete', 'evidence': evidence, 'files_sha256': files, 'directory_files': trees}
        except BaseException as error:
            atomic_new(HERE / f'{name}_result.json', {'status': 'failed', 'created_at': now(), 'phase': name,
                'seal_sha256': self.seal_sha, 'intent_sha256': digest(intent),
                'error': f'{type(error).__name__}: {error}'})
            raise
        atomic_new(HERE / f'{name}_result.json', {**result, 'created_at': now(), 'phase': name,
                   'seal_sha256': self.seal_sha, 'intent_sha256': digest(intent)})
        return evidence

    def command(self, name, command):
        atomic_new(HERE / f'{name}_command.json', {'command': list(map(str, command)), 'cwd': str(ROOT),
                   'phase': name, 'seal_sha256': self.seal_sha, 'created_at': now()})
        with (HERE / f'{name}.stdout').open('x') as stdout, (HERE / f'{name}.stderr').open('x') as stderr:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stdout, stderr=stderr)
        require(result.returncode == 0, f'{name} returned {result.returncode}; preserve logs and review')

    def finish(self, status, **details):
        require(status in ('needs_execution_review', 'needs_calibration_revision',
                           'matched_fixed_reference', 'outside_fixed_reference_tolerance'), 'unknown terminal status')
        self.verify()
        result = {'status': status, 'created_at': now(), 'seal_sha256': self.seal_sha,
                  'historical_level1_role': 'frozen_benchmark_reference', 'all_five_fresh_same_round': False,
                  'treatment_training_started': False, **details}
        atomic_new(RESULT, result)
        return result

    def observe(self, stage, submissions):
        completion = module(COMPLETION, 'v3_continuation_completion')
        accounts = completion.scheduler_records(submissions['job_ids'])
        state, failed, waiting = completion.completion_status(accounts)
        if failed:
            return self.finish('needs_execution_review', stage=stage, failed_job_ids=failed, scheduler_records=accounts), accounts
        if state != 'complete':
            return {'status': 'waiting_for_' + stage, 'waiting_job_ids': waiting}, accounts
        missing = [record['job']['output'] for record in submissions['jobs']
                   if not Path(record['job']['output']).is_file()]
        if missing:
            return {'status': 'waiting_for_' + stage + '_receipt_visibility', 'missing_receipts': missing}, accounts
        return None, accounts

    def fit(self, domain, registration_sha256):
        path = RECIPES[domain]
        require(not path.exists() and not path.with_suffix('.fit_intent.json').exists(),
                'existing recipe or fit ownership forbids another fit')
        name = 'fit_' + domain
        self.command(name, [PYTHON, '-B', FITTER, '--registration', common.REGISTRATION,
                           '--registration-sha256', registration_sha256, '--domain', domain, '--output', path])
        auditor = module(AUDITOR, 'v3_continuation_recipe_validation')
        recipe = read(path)
        auditor.validate_recipe_gates(recipe)
        return {'domain': domain, 'recipe': str(path), 'recipe_sha256': digest(path),
                'development_fit_pass': recipe['development_fit_pass']}, pins([path, path.with_suffix('.fit_intent.json')]), {}

    def audit(self, stage, scientific_seal_sha256, registration_sha256):
        output = DEV_AUDIT if stage == 'development' else REPORT
        require(not output.exists(), 'existing canonical audit must not be replayed or replaced')
        self.command(stage + '_audit', [PYTHON, '-B', AUDITOR, '--stage', stage,
            '--seal-sha256', scientific_seal_sha256, '--registration-sha256', registration_sha256,
            '--source-sha256', digest(AUDITOR), '--publish'])
        auditor = module(AUDITOR, 'v3_continuation_audit_validation')
        saved = read(output)
        if stage == 'development' and saved['status'] == 'needs_calibration_revision':
            require(saved['schema'] == auditor.DEV_SCHEMA and saved['jobs'] == 8 and saved['rows'] == 1232
                    and saved['attempts'] == 39424 and saved['all_revised_development_gates_pass'] is False
                    and saved['all_attempts_regraded_with_original_grader'] is True
                    and saved['every_attempt_including_failures_and_full_canonical_keys_compared'] is True
                    and saved['scientific_seal_sha256'] == scientific_seal_sha256
                    and saved['registration_sha256'] == registration_sha256, 'failed fit audit is incomplete')
            evidence = {'status': saved['status'], 'path': str(output), 'sha256': digest(output)}
            return evidence, merge_pins(saved['files_sha256'], pins([output])), saved['directory_files']
        evidence = (auditor.validate_completed_development_audit(path=output) if stage == 'development'
                    else auditor.validate_confirmation_report(path=output))
        require(evidence['status'] in ('passed_development', 'matched_fixed_reference', 'outside_fixed_reference_tolerance'),
                'canonical audit has unexpected status')
        summary = {key: evidence[key] for key in ('status', 'path', 'sha256')}
        if stage == 'confirmation':
            summary['confirmation_match_verified'] = evidence['confirmation_match_verified']
        return summary, evidence['files_sha256'], evidence['directory_files']

    def tick(self):
        seal = self.verify()
        completed = self.journal()
        if RESULT.exists():
            result = read(RESULT)
            require(result['seal_sha256'] == self.seal_sha, 'terminal result belongs to another owner')
            return result
        registration_sha = seal['registration_sha256']
        development = seal['development_submissions']
        if 'development_completed' not in completed:
            pending, accounts = self.observe('development', development)
            if pending is not None:
                return pending
            def complete_development():
                record = {'job_ids': development['job_ids'], 'accounting': accounts,
                          'receipts': [item['job']['output'] for item in development['jobs']]}
                path = HERE / 'development_completed_execution.json'
                atomic_new(path, record)
                return record, pins([path, *record['receipts']]), {}
            self.phase('development_completed', {'job_ids': development['job_ids']}, complete_development)
        for domain in common.REVISED:
            self.phase('fit_' + domain, {'registration_sha256': registration_sha, 'domain': domain},
                       lambda domain=domain: self.fit(domain, registration_sha))
        evidence = self.phase('development_audit', {'scientific_seal_sha256': seal['scientific_seal_sha256']},
                             lambda: self.audit('development', seal['scientific_seal_sha256'], registration_sha))
        if evidence['status'] == 'needs_calibration_revision':
            return self.finish('needs_calibration_revision', development_audit=evidence,
                               recipes={domain: {'path': str(path), 'sha256': digest(path)} for domain, path in RECIPES.items()})
        def finalize():
            require(not common.DATASET.exists(), 'fresh final dataset required')
            self.command('finalize', [PYTHON, '-B', FINALIZER, '--registration-sha256', registration_sha,
                '--execution-seal', SEAL, '--execution-seal-sha256', self.seal_sha, '--publish'])
            helper = module(FINALIZER, 'v3_continuation_finalize_validation')
            proof = helper.authenticate_dataset(registration_sha256=registration_sha)
            return proof['identity_metadata'], proof['files_sha256'], proof['directory_files']
        dataset = self.phase('finalize', {'development_audit_sha256': evidence['sha256']}, finalize)
        def prepare_confirmation():
            require(not (CONFIRMATION / 'seal.json').exists(), 'fresh confirmation preparation required')
            self.command('confirmation_prepare', [PYTHON, '-B', CONF_LAUNCHER, '--prepare',
                '--registration-sha256', registration_sha, '--execution-seal-sha256', self.seal_sha])
            launcher = module(CONF_LAUNCHER, 'v3_continuation_confirmation_prepare')
            current_sha = digest(launcher.SEAL)
            current = launcher.authenticate_saved_seal(current_sha)
            return {'seal_sha256': current_sha}, merge_pins(current['files_sha256'], pins([launcher.SEAL])), current['directory_files']
        confirmation = self.phase('confirmation_prepare', {'dataset': dataset}, prepare_confirmation)
        def submit_confirmation():
            launcher = module(CONF_LAUNCHER, 'v3_continuation_confirmation_submit')
            require(not launcher.CLAIM.exists() and not list(launcher.HERE.glob('submission_*_*.json')),
                    'existing confirmation submission ownership forbids retry')
            self.command('confirmation_submit', [PYTHON, '-B', CONF_LAUNCHER, '--submit', '--seal-sha256', confirmation['seal_sha256']])
            submissions = launcher.authenticate_submissions(confirmation['seal_sha256'])
            require(len(submissions['jobs']) == len(submissions['job_ids']) == len(set(submissions['job_ids'])) == 2
                    and not set(submissions['job_ids']) & set(development['job_ids']), 'exactly two fresh CONF submissions required')
            return submissions, submissions['files_sha256'], {}
        submissions = self.phase('confirmation_submit', confirmation, submit_confirmation)
        pending, accounts = self.observe('confirmation', submissions)
        if pending is not None:
            return pending
        evidence = self.phase('confirmation_audit', {'job_ids': submissions['job_ids'], 'scientific_seal_sha256': confirmation['seal_sha256']},
            lambda: self.audit('confirmation', confirmation['seal_sha256'], registration_sha))
        return self.finish(evidence['status'], report=evidence['path'], report_sha256=evidence['sha256'],
                           confirmation_match_verified=evidence['confirmation_match_verified'])


def readonly_status():
    return {'status': 'read_only', 'prepared': SEAL.is_file(), 'terminal_record_present': RESULT.is_file(),
            'configuration': configuration(), 'outcomes_read': False, 'actions_started': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--advance', action='store_true')
    actions.add_argument('--watch', action='store_true')
    parser.add_argument('--registration-sha256')
    parser.add_argument('--scientific-seal-sha256')
    parser.add_argument('--seal-sha256')
    args = parser.parse_args(argv)
    if not (args.prepare or args.advance or args.watch):
        print(json.dumps(readonly_status(), sort_keys=True), flush=True)
        return 0
    if args.prepare:
        require(args.registration_sha256 and args.scientific_seal_sha256 and not args.seal_sha256,
                'prepare requires explicit scientific registration and DEV seal hashes only')
    else:
        require(args.seal_sha256 and not args.registration_sha256 and not args.scientific_seal_sha256,
                'action requires the explicit prepared additive seal hash only')
    with coordinator_lock():
        if args.prepare:
            print(json.dumps(prepare(args.registration_sha256, args.scientific_seal_sha256), sort_keys=True), flush=True)
            return 0
        driver = Driver(args.seal_sha256)
        while True:
            try:
                result = driver.tick()
            except Exception as error:
                print(json.dumps({'status': 'needs_root_review', 'error': f'{type(error).__name__}: {error}',
                                  'seal_sha256': args.seal_sha256}, sort_keys=True), flush=True)
                return 1
            print(json.dumps(result, sort_keys=True), flush=True)
            if not result['status'].startswith('waiting_'):
                return 0 if result['status'] == 'matched_fixed_reference' else 2
            if not args.watch:
                return 0
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
