#!/usr/bin/env python3
"""Additive Slurm25.11 readback compatibility and exact first-job continuation.

The frozen E122 launcher, controller, plan and old submission evidence are never
edited. Only the equivalent single-node spelling 1-1 is normalized for the old
strict auditor; ledger records always retain the original scheduler bytes.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
SOURCE = Path(__file__).resolve()
TEST = ROOT / 'tests/test_e122_slurm_2511_compat.py'
LAUNCHER = ROOT / 'ops/exp_scaling/launch_e122_level3_factorial.py'
CONTROLLER = ROOT / 'ops/exp_scaling/control_e122_level3_release.py'
PLAN = ROOT / 'var/artifacts/e122_level3_factorial_plan.json'
HERE = ROOT / 'var/artifacts/e122_level3_factorial'
COMPAT_ROOT = HERE / 'slurm_2511_compat'
RESUME_CLAIM = COMPAT_ROOT / 'resume_claim.json'
FIRST_JOB = 31158645
PLAN_SHA256 = '67c506a40b9a7fb7b984335d2ac47c859e01c9a62e5eddba5891aca9401cec2f'
INITIAL_LEDGER_SHA256 = 'e02c8395be9ad2cd8f0246cd452b38687e8a480456e716ce18e1236f432997b6'
FROZEN_PINS = {
    str(LAUNCHER): 'bdf72437c4971361c54982726c4a49e0ef04daf2d3d9b8e9ac01edfb5b8f85cd',
    str(CONTROLLER): 'e123029bf55dde97d87eced134ee1b5db2c3dc4c4dd10c57f47200113a6f43c1',
    str(PLAN): PLAN_SHA256,
}
FIRST_PINS = {
    'submission_claim.json': 'a4bdff6e85bb515591192e71e436d6dc0aa0454a50e686da0c1bdf2644d43fb5',
    'submission_000_intent.json': 'ac1c8ef8df11b766a8bc923230ab624382c6247b95398d04953942b46a93f33b',
    'submission_000_result.json': '3d862fc664630d2a120efffdb45ad802304c28c06d57f3e0f40f09a2a9a509fd',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_bindings(compat_source_sha256, compat_test_sha256):
    for value in (compat_source_sha256, compat_test_sha256):
        require(re.fullmatch(r'[a-f0-9]{64}', value or ''), 'explicit compatibility source/test SHA256 required')
    pins = {**FROZEN_PINS, str(SOURCE): compat_source_sha256, str(TEST): compat_test_sha256}
    for path, expected in pins.items():
        require(digest(path) == expected, 'compatibility or frozen input changed: ' + path)
    return pins


def normalize_single_node(record):
    matches = list(re.finditer(r'(?<!\S)NumNodes=([^\s]+)', record))
    require(len(matches) == 1 and matches[0].group(1) in ('1', '1-1'),
            'only exact single-node NumNodes=1 or NumNodes=1-1 is compatible')
    match = matches[0]
    return record[:match.start(1)] + '1' + record[match.end(1):]


def make_auditor(original, verify):
    def audit(record, job_id, cell):
        verify()
        original(normalize_single_node(record), job_id, cell)
        return record
    return audit


def install_compat(launcher, compat_source_sha256, compat_test_sha256):
    verify_bindings(compat_source_sha256, compat_test_sha256)
    require(Path(launcher.__file__).resolve() == LAUNCHER, 'exact frozen E122 launcher module required')
    binding = (compat_source_sha256, compat_test_sha256)
    installed = getattr(launcher, '_slurm_2511_compat_binding', None)
    require(installed is None or installed == binding, 'another compatibility binding is already installed')
    if installed is None:
        launcher.audit_held_record = make_auditor(launcher.audit_held_record,
            lambda: verify_bindings(compat_source_sha256, compat_test_sha256))
        launcher._slurm_2511_compat_binding = binding
    return launcher


def queue_e122_jobs():
    result = subprocess.run(['squeue', '--noheader', '--user', str(__import__('os').getuid()),
        '--format=%i|%j|%T'], capture_output=True, text=True, check=False, timeout=60)
    require(result.returncode == 0, 'cannot enumerate existing E122 jobs: ' + result.stderr)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split('|')
        require(len(fields) == 3, 'malformed scheduler queue row')
        if fields[1].lower().startswith('e122'):
            rows.append({'job_id': fields[0], 'job_name': fields[1], 'state': fields[2]})
    return rows


def validate_initial_evidence(launcher, plan, jobs):
    """Read-only exact-state guard; no historical intent can be reset or adopted loosely."""
    require(not RESUME_CLAIM.exists(), 'compatibility resume claim already exists; never retry')
    for name, expected in FIRST_PINS.items():
        require(digest(HERE / name) == expected, 'known first submission evidence changed: ' + name)
    expected = {'submission_000_intent.json', 'submission_000_result.json'}
    observed = {path.name for pattern in ('submission_*_intent.json', 'submission_*_result.json')
                for path in HERE.glob(pattern)}
    require(observed == expected and not list(HERE.glob('held_audit_*.json')),
            'unexpected additional submission intent/result or held audit')
    require(not (HERE / 'held_submission_complete.json').exists()
            and not (HERE / 'prospective_ledger_before_submission.json').exists(), 'submission already finalized')
    claim = read(HERE / 'submission_claim.json')
    result = read(HERE / 'submission_000_result.json')
    intent = read(HERE / 'submission_000_intent.json')
    require(claim['schema'] == 'e122_once_only_submission_claim_v1' and claim['jobs'] == 100
            and claim['plan_sha256'] == PLAN_SHA256
            and claim['initial_ledger_sha256'] == INITIAL_LEDGER_SHA256, 'original submission claim differs')
    require(result['returncode'] == 0 and result['stdout'] == str(FIRST_JOB) + '\n' and result['stderr'] == '',
            'only the exact successful first submission result can be reconciled')
    require(intent['cell'] == plan['cells'][0]
            and intent['command_sha256'] == launcher.sha(plan['cells'][0]['command']), 'first submitted cell differs')
    require(jobs == [{'job_id': str(FIRST_JOB), 'job_name': plan['cells'][0]['run_stamp'], 'state': 'PENDING'}],
            'exactly the known first E122 job must exist and remain pending')
    require(digest(launcher.LEDGER) == INITIAL_LEDGER_SHA256, 'prospective ledger changed')
    ledger = read(launcher.LEDGER)
    launcher.verify_initial_ledger(ledger, plan['cells'])
    require(not any(Path(cell['run_dir']).exists() for cell in plan['cells']), 'all E122 run directories must remain fresh')
    return ledger


def compatibility_record(compat_source_sha256, compat_test_sha256):
    return {'schema': 'e122_slurm_2511_compat_v1', 'source': str(SOURCE),
        'source_sha256': compat_source_sha256, 'tests': str(TEST), 'tests_sha256': compat_test_sha256,
        'frozen_pins': dict(FROZEN_PINS), 'only_normalization': 'NumNodes=1-1 to NumNodes=1 for strict audit',
        'raw_scheduler_record_preserved': True}


def resume_known_first(launcher, compat_source_sha256, compat_test_sha256):
    """Adopt only job31158645, then submit indices1..99 once under new ownership."""
    install_compat(launcher, compat_source_sha256, compat_test_sha256)
    plan = launcher.verify_plan(expected_sha256=PLAN_SHA256, model_choice='05b')
    validate_initial_evidence(launcher, plan, queue_e122_jobs())
    with (HERE / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = validate_initial_evidence(launcher, plan, queue_e122_jobs())
        first_raw = launcher.audit_held(FIRST_JOB, plan['cells'][0])
        binding = compatibility_record(compat_source_sha256, compat_test_sha256)
        launcher.atomic_new(RESUME_CLAIM, {'schema': 'e122_slurm_2511_resume_claim_v1',
            'created_at': launcher.now(), 'first_job_id': FIRST_JOB, 'adopted_index': 0,
            'submit_only_indices': list(range(1, 100)), 'original_claim_sha256': FIRST_PINS['submission_claim.json'],
            'initial_ledger_sha256': INITIAL_LEDGER_SHA256, 'compatibility': binding})
        records = []
        for index, cell in enumerate(plan['cells']):
            verify_bindings(compat_source_sha256, compat_test_sha256)
            for name, expected in FIRST_PINS.items():
                require(digest(HERE / name) == expected, 'original submission evidence changed: ' + name)
            job_id = FIRST_JOB if index == 0 else launcher.submit_one_held(cell, index)
            raw = first_raw if index == 0 else launcher.audit_held(job_id, cell)
            record = {key: cell[key] for key in
                ('domain', 'dataset_domain', 'arm', 'seed', 'run_stamp', 'run_dir', 'target_steps')}
            record.update(job_id=job_id, held_scheduler_record=raw)
            launcher.atomic_new(HERE / f'held_audit_{index:03d}.json', record)
            records.append(record)
            print(json.dumps({'event': 'held_audited', 'count': len(records), 'job_id': job_id,
                              'adopted_without_submission': index == 0}), flush=True)
        require(len({row['job_id'] for row in records}) == 100, 'duplicate E122 job ID')
        launcher.verify_plan(expected_sha256=PLAN_SHA256, model_choice='05b')
        for record, cell in zip(records, plan['cells']):
            launcher.audit_held(record['job_id'], cell)
        require(digest(launcher.LEDGER) == INITIAL_LEDGER_SHA256, 'prospective ledger changed during continuation')
        launcher.atomic_new(HERE / 'prospective_ledger_before_submission.json', ledger)
        ledger.update(runs=records, status='held_audited', released=False,
            model_choice='05b', model_choice_pending=False, model=plan['model'], model_revision=plan['model_revision'],
            model_family='qwen05b', admission_proof=plan['admission_proof'], plan_path=str(PLAN),
            plan_sha256=PLAN_SHA256, snapshot_root=plan['snapshot_root'], held_audited_at=launcher.now(),
            slurm_readback_compatibility=binding)
        launcher.e119.e78.atomic_json(launcher.LEDGER, ledger)
        launcher.atomic_new(HERE / 'held_submission_complete.json', {'schema': 'e122_held_submission_complete_v1',
            'created_at': launcher.now(), 'plan_sha256': PLAN_SHA256, 'ledger_sha256': digest(launcher.LEDGER),
            'job_ids': [record['job_id'] for record in records], 'released': False, 'compatibility': binding})
    return {'held': 100, 'released': 0, 'adopted_first_job': FIRST_JOB,
            'submitted_new_jobs': 99, 'ledger': str(launcher.LEDGER), 'ledger_sha256': digest(launcher.LEDGER)}


def verify_controller_ledger_binding(path, expected_sha256, compat_source_sha256, compat_test_sha256):
    require(digest(path) == expected_sha256, 'explicit held ledger SHA256 differs')
    require(read(path).get('slurm_readback_compatibility')
            == compatibility_record(compat_source_sha256, compat_test_sha256),
            'held ledger compatibility implementation binding differs')


def controller_main(arguments, compat_source_sha256, compat_test_sha256):
    launcher = importlib.import_module('launch_e122_level3_factorial')
    install_compat(launcher, compat_source_sha256, compat_test_sha256)
    controller = importlib.import_module('control_e122_level3_release')
    parsed = controller.parse_args(arguments)
    require(Path(parsed.plan).resolve() == PLAN and parsed.plan_sha256 == PLAN_SHA256
            and Path(parsed.held_ledger).resolve() == launcher.LEDGER and parsed.model_choice == '05b',
            'compatibility controller requires the original exact E122 plan, ledger and model')
    verify_controller_ledger_binding(Path(parsed.held_ledger), parsed.held_ledger_sha256,
                                     compat_source_sha256, compat_test_sha256)
    return controller.main(arguments)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compat-source-sha256', required=True)
    parser.add_argument('--compat-test-sha256', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--resume-known-first', action='store_true')
    action.add_argument('--controller', action='store_true')
    args, forwarded = parser.parse_known_args(argv)
    verify_bindings(args.compat_source_sha256, args.compat_test_sha256)
    if args.resume_known_first:
        require(not forwarded, 'unexpected resume arguments')
        launcher = importlib.import_module('launch_e122_level3_factorial')
        result = resume_known_first(launcher, args.compat_source_sha256, args.compat_test_sha256)
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    return controller_main(forwarded, args.compat_source_sha256, args.compat_test_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
