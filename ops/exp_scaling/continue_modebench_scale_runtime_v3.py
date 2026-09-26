#!/usr/bin/env python3
"""Continue the prospectively amended batch-eight A5000 runtime; default is read-only status.

Copied from preserved runtime v2; both its SHA256 and the sealed v1 SHA256
are recorded below. The
scientific fitting, frozen data checks, release gates, and retry policy remain
unchanged. Only execution placement and its prospective provenance are revised.
Use --advance to fit, freeze, submit confirmation, and publish after every held-out gate passes. --watch repeats the sweep. Failed gates require a new development revision; an unresolved
submission intent requires reconciliation. Neither condition triggers a retry.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import tempfile

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
import fit_modebench_scale as fit
import launch_modebench_scale_runtime_v3 as launch
import materialize_modebench_scale as materialize
from fit_modebench_level3 import atomic_new, file_sha, sha
from materialize_modebench_harder_v2 import load_rows, SPLITS
from materialize_modebench_scale import DOMAINS, LEVELS, authenticate, read, require

DEFAULT_DATA = ROOT / 'var/data/modebench_scale_v1'
DEFAULT_ARTIFACTS = ROOT / 'var/artifacts/modebench_scale_runtime_v3'
ORIGINAL_CONTROLLER = ROOT / 'ops/exp_scaling/continue_modebench_scale.py'
ORIGINAL_CONTROLLER_SHA256 = 'bde7611223aed892e8c13a90a9a50eea628d0c137b822af86f629a9386246af5'
PREVIOUS_CONTROLLER = ROOT / 'ops/exp_scaling/continue_modebench_scale_runtime_v2.py'
PREVIOUS_CONTROLLER_SHA256 = '3fb49221c97f3d55ea46fcd56586e07d98f429a380ec015dc5f5bac65f4106ab'
REGISTERED_PROTOCOL_SHA256 = 'b764df4f5e23cf02f7bf4d3339226134739c5f1ccedf7a884c5947c65dd2f6d4'
FORMER_ARRAY_JOB_ID = 31242953
AMENDMENT_SCHEMA = 'modebench_scale_runtime_amendment_v3'


def runtime_profile():
    return dict(launch.RUNTIME_PROFILE)


def amendment_sources(data):
    return (Path(__file__).resolve(), ORIGINAL_CONTROLLER, PREVIOUS_CONTROLLER, Path(launch.__file__).resolve(),
            Path(fit.__file__).resolve(), Path(materialize.__file__).resolve(), data / 'protocol.json')


def runtime_amendment(data, artifacts):
    path = artifacts / 'runtime_amendment.json'
    require(path.is_file(), 'prospective runtime amendment required before advance')
    value = read(path)
    require(value.get('schema') == AMENDMENT_SCHEMA
            and value.get('data_root') == str(data) and value.get('artifacts_root') == str(artifacts)
            and value.get('protocol_sha256') == file_sha(data / 'protocol.json') == REGISTERED_PROTOCOL_SHA256
            and value.get('original_controller_sha256') == file_sha(ORIGINAL_CONTROLLER) == ORIGINAL_CONTROLLER_SHA256
            and file_sha(PREVIOUS_CONTROLLER) == PREVIOUS_CONTROLLER_SHA256
            and value.get('scientific_protocol_unchanged') is True
            and value.get('model_identities_unchanged') is True
            and value.get('candidate_outcomes_observed_before_replacement') is False
            and value.get('former_array_job_id') == FORMER_ARRAY_JOB_ID
            and isinstance(value.get('replacement_reason'), str) and value['replacement_reason'].strip()
            and value.get('runtime_profile') == runtime_profile(),
            'runtime amendment identity or prospective conditions changed')
    pins = value.get('files_sha256')
    require(isinstance(pins, dict) and all(str(source) in pins for source in amendment_sources(data)),
            'runtime amendment omits required source pins')
    check_pins(pins)
    smoke = value.get('runtime_smoke_receipts')
    require(isinstance(smoke, dict) and set(smoke) == set(LEVELS.values()),
            'passing runtime smoke receipts required for both model scales')
    for label, reference in smoke.items():
        require(isinstance(reference, dict) and isinstance(reference.get('path'), str)
                and Path(reference['path']).is_absolute(), 'runtime smoke receipt path required')
        receipt_path = Path(reference['path'])
        require(reference.get('sha256') == file_sha(receipt_path), 'runtime smoke receipt changed')
        receipt = read(receipt_path)
        require(receipt.get('status') == 'pass' and receipt.get('model_label') == label
                and receipt.get('runtime_profile') == runtime_profile(), 'runtime smoke receipt did not pass')
    return path, value

TERMINAL = {'admitted', 'needs_new_development_revision', 'heldout_confirmation_failed', 'execution_failed', 'submission_needs_reconciliation'}

def receipt_paths(data, level, phase):
    return [data / level / 'results' / ('development' if phase == 'dev' else 'confirmation') /
            domain / f'difficulty_{tier}.json' if phase == 'dev' else
            data / level / 'results/confirmation' / (domain + '.json')
            for domain in DOMAINS for tier in (range(4) if phase == 'dev' else (None,))]

def check_pins(pins):
    require(isinstance(pins, dict) and pins, 'nonempty recorded input pins required')
    for path, digest in pins.items():
        require(file_sha(path) == digest, 'recorded input changed: ' + path)

def ready_receipts(data, level, phase):
    paths = receipt_paths(data, level, phase)
    if not all(path.is_file() for path in paths):
        return False
    identities = set()
    for path in paths:
        receipt = read(path)
        domain = path.parent.name if phase == 'dev' else path.stem
        require(receipt.get('schema') == 'modebench-scale-calibration-independent-v1'
                and receipt.get('status') == 'complete' and receipt.get('level') == level
                and receipt.get('domain') == domain and receipt.get('split') == phase
                and receipt.get('model_label') == LEVELS[level], 'wrong or incomplete receipt: ' + str(path))
        digest = receipt.get('identity_sha256')
        require(isinstance(digest, str) and digest not in identities, 'duplicate receipt identity')
        identities.add(digest)
    return True

def job_failure(plan_path, level, runner=None):
    result_path = plan_path.parent / 'submission_result.json'
    if not result_path.exists():
        return None
    job, plan = read(result_path)['array_job_id'], read(plan_path)
    cells = {f'{job}_{i}': cell for i, cell in enumerate(plan['cells']) if cell['id'].startswith(level + '_')}
    # JobIDRaw is a numeric allocation ID here, even for array elements. Ask
    # Slurm to expand untouched pending/cancelled elements and use JobID so
    # every state can be attributed to its registered level/domain.
    command = ['sacct', '-n', '-X', '-P', '--array', '-j', str(job), '--format=JobID,State,End']
    try:
        result = (runner or subprocess.run)(command, capture_output=True, text=True, check=False, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as error:
        return {'status': 'scheduler_query_unavailable', 'array_job_id': job,
                'error': type(error).__name__, 'detail': str(error)[:2000]}
    if result.returncode != 0:
        return {'status': 'scheduler_query_unavailable', 'array_job_id': job,
                'error': 'sacct_nonzero_exit', 'returncode': result.returncode,
                'detail': (result.stderr or '')[:2000]}
    failed = []
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if len(fields) < 2 or fields[0].strip() not in cells:
            continue
        task, state = fields[0].strip(), fields[1].strip().split(' ')[0].rstrip('+')
        domain = cells[task]['id'][len(level) + 1:]
        missing = [p for p in receipt_paths(Path(plan['data_root']), level, plan['phase'])
                   if (p.parent.name if plan['phase'] == 'dev' else p.stem) == domain and not p.is_file()]
        if not missing:
            continue
        stale = False
        if state == 'COMPLETED' and len(fields) > 2 and fields[2].strip() not in ('', 'Unknown'):
            ended = datetime.fromisoformat(fields[2].strip())
            stale = (datetime.now(ended.tzinfo) - ended).total_seconds() >= 120
        if state in {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'} or stale:
            failed.append({'job': task, 'state': state, **({'error': 'missing_receipts_after_visibility_grace'} if stale else {})})
    return {'status': 'execution_failed', 'jobs': failed} if failed else None

def recipe(data, level, domain, protocol, *, advance):
    path = data / level / 'recipes' / (domain + '.json')
    if not path.exists():
        return fit.fit_domain(data, level, domain) if advance else None
    value = read(path)
    require(value.get('schema') == fit.SCHEMA and value.get('level') == level
            and value.get('domain') == domain and value.get('protocol_sha256') == file_sha(data / 'protocol.json')
            and value.get('target') == protocol['targets'][domain]
            and value.get('fitter_sha256') == file_sha(fit.__file__), 'saved development recipe changed')
    check_pins(value['input_sha256'])
    return value

def dataset(data, level, domain):
    base = data / level / 'dataset' / domain
    value = read(base / 'identity.json')
    require(value.get('schema') == 'modebench_scale_frozen_domain_v1'
            and value.get('status') == 'frozen_pending_heldout_confirmation'
            and value.get('level') == level and value.get('domain') == domain
            and value.get('protocol_sha256') == file_sha(data / 'protocol.json')
            and value.get('recipe_sha256') == file_sha(data / level / 'recipes' / (domain + '.json')),
            'frozen dataset identity changed')
    for split, (count, subset) in SPLITS.items():
        rows = [json.loads(line) for line in (base / (split + '.jsonl')).read_text().splitlines() if line.strip()]
        require(len(rows) == value['splits'][split]['rows'] == count
                and sha(rows) == value['splits'][split]['rows_sha256']
                and load_rows(base / split, subset) == rows, 'frozen dataset rows changed: ' + str(base / split))
    return value

def confirmation(data, level, domain, protocol):
    path = data / level / 'confirmation' / (domain + '.json')
    if not path.exists():
        return fit.confirm_domain(data, level, domain)
    value = read(path)
    receipt_path = data / level / 'results/confirmation' / (domain + '.json')
    receipt = read(receipt_path)
    require(value.get('schema') == 'modebench_scale_confirmation_v1'
            and value.get('level') == level and value.get('domain') == domain
            and value.get('receipt_sha256') == file_sha(receipt_path)
            and value.get('recipe_sha256') == file_sha(data / level / 'recipes' / (domain + '.json'))
            and value.get('dataset_identity_sha256') == file_sha(data / level / 'dataset' / domain / 'identity.json')
            and value.get('metrics') == receipt['metrics']
            and value.get('target') == protocol['targets'][domain]['metrics']
            and value.get('original_grader_replayed_attempts') == SPLITS['eval'][0] * 32,
            'saved held-out audit inputs changed')
    delta = {metric: receipt['metrics'][metric] - value['target'][metric] for metric in fit.TOLERANCES}
    gates = {metric: abs(delta[metric]) <= fit.TOLERANCES[metric] for metric in fit.TOLERANCES}
    require(value.get('differences') == delta and value.get('gates') == gates
            and value.get('difficulty_matched') is all(gates.values()), 'saved held-out audit gates changed')
    return value

def publish_level(data, level, protocol, *, publish=True):
    pins, domains = {str(data / 'protocol.json'): file_sha(data / 'protocol.json')}, {}
    for domain in DOMAINS:
        base = data / level / 'dataset' / domain
        require(recipe(data, level, domain, protocol, advance=False)['development_fit_pass'], 'passing recipe required')
        value = dataset(data, level, domain)
        require((data / level / 'confirmation' / (domain + '.json')).is_file(), 'missing release audit')
        require(confirmation(data, level, domain, protocol)['difficulty_matched'] is True,
                'all five held-out domains must pass before release')
        audit = data / level / 'confirmation' / (domain + '.json')
        for path in (base / 'identity.json', audit, data / level / 'recipes' / (domain + '.json'),
                     data / level / 'results/confirmation' / (domain + '.json')):
            pins[str(path)] = file_sha(path)
        domains[domain] = {split: {'path': str(base / split), **value['splits'][split]}
                           for split in ('train', 'dev', 'eval')}
    value = {'schema': 'modebench_scale_level_admission_v1', 'level': level, 'model_label': LEVELS[level],
             'difficulty_matched': True, 'target': protocol['targets'], 'domains': domains,
             'files_sha256': pins, 'test_split': 'eval', 'treatment_training_started': False}
    if publish:
        atomic_new(data / level / 'admission.json', value)
    return value

def publish_readme(data, level):
    path = data / level / 'README.md'
    text = (f'# ModeBench {level} ({LEVELS[level]})\n\n'
            'All five domains passed the registered held-out difficulty gates. '
            'Each domain has 384 training rows, 128 development rows, and 128 test rows. '
            'The test split is named `eval`; no model training was run.\n\n'
            '| Domain | Train | Test |\n|---|---|---|\n' + ''.join(
                f'| {domain} | [train](dataset/{domain}/train) | [test](dataset/{domain}/eval) |\n'
                for domain in DOMAINS))
    if path.exists():
        require(path.read_text() == text, 'release README changed')
    else:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(handle.name, path)

def prior_submissions(artifacts):
    paths = [artifacts / 'development/submission_result.json', *(
        artifacts / ('confirmation_' + level) / 'submission_result.json' for level in LEVELS)]
    ids = []
    for path in paths:
        if path == paths[0] or path.exists() or (path.parent / 'submission_intent.json').exists():
            if not path.exists():
                return None
            value = read(path)
            require(value.get('status') == 'submitted' and type(value.get('array_job_id')) is int
                    and value['array_job_id'] > 0, 'invalid prior array submission')
            ids.append(value['array_job_id'])
    return sorted(set(ids))

def advance_level(data, artifacts, level, protocol, *, advance):
    admission = data / level / 'admission.json'
    if admission.exists():
        value = read(admission)
        check_pins(value['files_sha256'])
        require(value == publish_level(data, level, protocol, publish=False), 'existing admission changed')
        if advance:
            publish_readme(data, level)
        return {'status': 'admitted', 'admission': str(admission)}
    if not ready_receipts(data, level, 'dev'):
        return job_failure(artifacts / 'development/plan.json', level) or {'status': 'waiting_development', 'complete': sum(p.is_file() for p in receipt_paths(data, level, 'dev')), 'expected': 20}
    recipes = {domain: recipe(data, level, domain, protocol, advance=advance) for domain in DOMAINS}
    if any(value is None for value in recipes.values()):
        return {'status': 'ready_to_fit'}
    failed = [domain for domain, value in recipes.items() if not value['development_fit_pass']]
    if failed:
        return {'status': 'needs_new_development_revision', 'failed_domains': failed}
    for domain in DOMAINS:
        base = data / level / 'dataset' / domain
        if not base.exists():
            if not advance:
                return {'status': 'ready_to_freeze'}
            materialize.freeze_dataset(data, level, domain)
        dataset(data, level, domain)
    plan_path = artifacts / ('confirmation_' + level) / 'plan.json'
    dependencies = prior_submissions(artifacts) if not (plan_path.parent / 'submission_result.json').exists() else []
    if dependencies is None:
        return {'status': 'submission_needs_reconciliation', 'reason': 'unresolved prior array submission'}
    if not plan_path.exists():
        if not advance:
            return {'status': 'ready_confirmation_submission'}
        launch.prepare(None, plan_path.parent, {label: model['path'] for label, model in protocol['models'].items()},
                       phase='eval', concurrency=1, data_root=data, levels=[level], dependency_ids=dependencies)
    plan = launch.verify(plan_path)
    require(plan.get('phase') == 'eval' and Path(plan['data_root']).resolve() == data
            and {cell['id'] for cell in plan['cells']} == {level + '_' + domain for domain in DOMAINS}
            and len(plan['cells']) == len(DOMAINS) and plan.get('concurrency') == 1,
            'wrong confirmation plan')
    intent, submitted = plan_path.parent / 'submission_intent.json', plan_path.parent / 'submission_result.json'
    if not submitted.exists():
        if intent.exists():
            return {'status': 'submission_needs_reconciliation', 'intent': str(intent)}
        if not advance:
            return {'status': 'ready_confirmation_submission'}
        require(set(dependencies) <= set(plan.get('dependency_ids', [])),
                'confirmation plan dependencies omit prior arrays; reconcile the unsubmitted plan')
        launch.submit(plan_path)
    require(intent.exists() and read(intent).get('plan_sha256') == file_sha(plan_path)
            and read(submitted).get('status') == 'submitted'
            and type(read(submitted).get('array_job_id')) is int, 'submission provenance changed')
    if not ready_receipts(data, level, 'eval'):
        return job_failure(plan_path, level) or {'status': 'waiting_confirmation', 'complete': sum(p.is_file() for p in receipt_paths(data, level, 'eval')), 'expected': 5}
    if not advance and not all((data / level / 'confirmation' / (domain + '.json')).exists() for domain in DOMAINS):
        return {'status': 'ready_to_audit'}
    audits = {domain: confirmation(data, level, domain, protocol) for domain in DOMAINS}
    failed = [domain for domain, value in audits.items() if not value['difficulty_matched']]
    if failed:
        return {'status': 'heldout_confirmation_failed', 'failed_domains': failed}
    if not advance:
        return {'status': 'ready_to_release'}
    publish_level(data, level, protocol)
    publish_readme(data, level)
    return {'status': 'admitted', 'admission': str(admission)}

@contextmanager
def controller_lock(artifacts):
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / 'controller.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def sweep(data=DEFAULT_DATA, artifacts=DEFAULT_ARTIFACTS, *, advance=False):
    data, artifacts = Path(data).resolve(), Path(artifacts).resolve()
    protocol = authenticate(data / 'protocol.json')
    if advance:
        with controller_lock(artifacts):
            amendment_path, _ = runtime_amendment(data, artifacts)
            seal = {'schema': 'modebench_scale_controller_runtime_v3', 'data_root': str(data),
                    'original_controller_sha256': ORIGINAL_CONTROLLER_SHA256,
                    'previous_controller_sha256': PREVIOUS_CONTROLLER_SHA256,
                    'runtime_profile': runtime_profile(),
                    'files_sha256': {str(path): file_sha(path) for path in (Path(__file__),
                        Path(fit.__file__), Path(materialize.__file__), Path(launch.__file__),
                        data / 'protocol.json', artifacts / 'development/plan.json', ORIGINAL_CONTROLLER, PREVIOUS_CONTROLLER, amendment_path)}}
            seal_path = artifacts / 'controller_identity.json'
            if seal_path.exists():
                require(read(seal_path) == seal, 'controller or registered inputs changed after arm')
            else:
                atomic_new(seal_path, seal)
            return _sweep(data, artifacts, protocol, advance=True)
    return _sweep(data, artifacts, protocol, advance=False)

def _sweep(data, artifacts, protocol, *, advance):
    development = launch.verify(artifacts / 'development/plan.json')
    require(development.get('phase') == 'dev' and Path(development['data_root']).resolve() == data
            and len(development['cells']) == 10 and development.get('concurrency') == 1
            and {cell['id'] for cell in development['cells']} == {level + '_' + domain for level in LEVELS for domain in DOMAINS},
            'wrong development plan')
    result = {level: advance_level(data, artifacts, level, protocol, advance=advance) for level in LEVELS}
    if advance and all(value['status'] == 'admitted' for value in result.values()):
        value = {'schema': 'modebench_scale_campaign_admission_v1', 'difficulty_matched': True,
                 'levels': result, 'files_sha256': {str(data / level / 'admission.json'):
                                                   file_sha(data / level / 'admission.json') for level in LEVELS}}
        path = data / 'admission.json'
        if path.exists():
            require(read(path) == value, 'campaign admission changed')
        else:
            atomic_new(path, value)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--artifacts-root', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--advance', action='store_true', help='perform authorized transitions and submit confirmation once')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='one sweep (default)')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args(argv)
    require(1 <= args.interval <= 60, 'watch interval must be between 1 and 60 seconds')
    while True:
        result = sweep(args.data_root, args.artifacts_root, advance=args.advance)
        print(json.dumps(result, sort_keys=True), flush=True)
        if not args.watch or all(value['status'] in TERMINAL for value in result.values()):
            return result
        time.sleep(args.interval)

if __name__ == '__main__':
    main()
