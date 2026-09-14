#!/usr/bin/env python3
"""Queue an isolated E123 benchmark, then launch the registered factorial on success.

User-authorized automatic handoff; only the measured physical runtime settings
may differ from the reviewed 100-cell preview. A failed or ambiguous scheduler
operation stops without retry. Existing experiment jobs are never modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
HERE = ROOT / 'var/artifacts/e123_a100_launch_20260909'
BENCH = ROOT / 'ops/exp_scaling/benchmark_e123_a100_runtime.py'
CONTROLLER = ROOT / 'ops/exp_scaling/control_e123_level3_release.py'
PYTHON = ROOT / 'var/seed_paper_eval/paper310/bin/python'
sys.path.insert(0, str(SOURCE.parent))
import launch_e123_level3_qwen3b_factorial as launch


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def record_state(status, **kw):
    result = {'schema': 'e123_automatic_launch_state_v1', 'status': status,
              'observed_at': now(), 'pid': os.getpid(), **kw}
    write(HERE / 'state.json', result)
    print(json.dumps(result), flush=True)
    return result


def science_projection(plan):
    """Physical settings may change; the scientific cells and runtime may not."""
    return {
        'snapshot': plan['snapshot'], 'model_identity': plan['model_identity'],
        'admission_proof': plan['admission_proof'], 'files_sha256': plan['files_sha256'],
        'cells': [{k: v for k, v in cell.items() if k not in ('environment', 'resources', 'command')}
                  | {'environment': {k: v for k, v in cell['environment'].items()
                                     if k not in launch.SYSTEMS_ENV_KEYS}}
                  for cell in plan['cells']],
    }


def check(plan):
    for path, sha in plan['files_sha256'].items():
        launch.require(launch.digest(path) == sha, 'orchestration input changed: ' + path)
    preview = launch.read(plan['preview_path'])
    launch.require(launch.digest(plan['preview_path']) == plan['preview_sha256'], 'reviewed preview changed')
    launch.verify_snapshot(preview['snapshot'])
    return preview


def prepare():
    launch.require(not (HERE / 'orchestration.json').exists(), 'orchestration already prepared')
    HERE.mkdir(parents=True, exist_ok=True)
    preview = launch.build_plan('3b')
    launch.require(preview['admission_proof']['status'] == 'matched_fixed_reference', 'Level-3 admission required')
    launch.publish_snapshot(preview['snapshot'])
    preview_path = HERE / 'reviewed_preview.json'
    launch.atomic_new(preview_path, preview)
    if not launch.LEDGER.exists():
        launch.atomic_new(launch.LEDGER, launch.prospective_ledger(preview))
    else:
        launch.verify_initial_ledger(launch.read(launch.LEDGER), preview['cells'])
    subprocess.run([str(PYTHON), str(BENCH), 'prepare', '--manifest', str(preview_path),
                    '--output', str(HERE / 'benchmark')], check=True)
    pins = dict(preview['files_sha256'])
    for path in (SOURCE, BENCH, ROOT / 'tests/test_e123_benchmark_then_launch.py', HERE / 'benchmark/plan.json'):
        pins[str(path)] = launch.digest(path)
    wrapper = HERE / 'benchmark.slurm'
    wrapper.write_text('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(ROOT)) + '\n'
        + 'source ' + shlex.quote(str(Path(preview['snapshot_root']) / 'ops/repo_env.sh')) + '\n'
        + 'export LD_LIBRARY_PATH=' + shlex.quote(str(PYTHON.parent.parent / 'lib')) + '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
        + 'export USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_TF=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
        + 'exec ' + shlex.join([str(PYTHON), str(BENCH), 'suite', '--plan', str(HERE / 'benchmark/plan.json'),
                               '--warmup', '8', '--steps', '32']) + '\n')
    wrapper.chmod(0o755)
    pins[str(wrapper)] = launch.digest(wrapper)
    plan = {'schema': 'e123_automatic_launch_plan_v1', 'created_at': now(),
        'authorization': 'User requested measured A100 runtime setup followed by full E123 Qwen-3B Level-3 factorial launch.',
        'preview_path': str(preview_path), 'preview_sha256': launch.digest(preview_path),
        'files_sha256': pins, 'selected_profile_path': str(HERE / 'benchmark/selected_profile.json'),
        'benchmark_resources': {'node': 'node302', 'gpus': 1, 'cpus': 8, 'memory_gib': 116, 'hours': 24},
        'benchmark_command': ['sbatch', '--parsable', '--hold', '--job-name=e123-a100-benchmark',
            '--partition=mltheory', '--account=mltheory', '--nodelist=node302', '--gres=gpu:a100:1',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=116G', '--time=1-00:00:00',
            '--nice=0', '--no-requeue', '--chdir=' + str(ROOT), '--export=ALL',
            '--output=' + str(HERE / 'benchmark-%j.log'), '--error=' + str(HERE / 'benchmark-%j.log'), str(wrapper)],
        'launch_policy': 'passing selected profile only; exact frozen science; all 100 held audited; measured concurrency cap and live storage controller',
        'old_jobs_modified': False}
    launch.atomic_new(HERE / 'orchestration.json', plan)
    check(plan)
    return record_state('prepared', planned_cells=100, benchmark_submitted=False)


def audit_benchmark(record, job_id, plan):
    expected = {'JobId': str(job_id), 'JobName': 'e123-a100-benchmark', 'JobState': 'PENDING',
        'Reason': 'JobHeldUser', 'Account': 'mltheory', 'Partition': 'mltheory', 'ReqNodeList': 'node302',
        'NumCPUs': '8', 'NumTasks': '1', 'MinMemoryNode': '116G', 'TimeLimit': '1-00:00:00',
        'Command': plan['benchmark_command'][-1], 'WorkDir': str(ROOT), 'Restarts': '0', 'Requeue': '0',
        'Nice': '0', 'QOS': 'none', 'RunTime': '00:00:00'}
    for key, value in expected.items():
        actual = launch.field(record, key)
        launch.require(actual == value or (key == 'MinMemoryNode' and actual == '118784'),
                       'benchmark held field differs: ' + key)
    launch.require(launch.field(record, 'NumNodes') in ('1', '1-1'), 'benchmark must use one node')
    launch.require(launch.field(record, 'UserId').endswith(f'({os.getuid()})'), 'benchmark owner differs')
    launch.require(launch.field(record, 'TresPerNode') == 'gres/gpu:a100:1', 'benchmark GPU allocation differs')


def submit():
    plan = launch.read(HERE / 'orchestration.json'); check(plan)
    # This exclusive intent remains after every failure; never repeat sbatch.
    launch.atomic_new(HERE / 'benchmark_submission_intent.json', {'at': now(), 'command': plan['benchmark_command']})
    result = subprocess.run(plan['benchmark_command'], env=launch.clean_submit_environment(),
                            capture_output=True, text=True, timeout=120)
    launch.atomic_new(HERE / 'benchmark_submission_result.json', {'returncode': result.returncode,
                      'stdout': result.stdout, 'stderr': result.stderr})
    launch.require(result.returncode == 0 and re.fullmatch(r'[1-9]\d*(?:;[^\s]+)?\s*', result.stdout),
                   'benchmark submission failed or ambiguous; retained intent requires reconciliation')
    job_id = int(result.stdout.strip().split(';')[0])
    record = subprocess.run(['scontrol', 'show', 'job', '-dd', '-o', str(job_id)],
                            capture_output=True, text=True, check=True, timeout=60).stdout
    audit_benchmark(record, job_id, plan)
    launch.atomic_new(HERE / 'benchmark_held_audit.json', {'job_id': job_id, 'record': record})
    check(plan)
    launch.atomic_new(HERE / 'benchmark_release_intent.json', {'job_id': job_id, 'at': now()})
    result = subprocess.run(['scontrol', 'release', str(job_id)], capture_output=True, text=True, timeout=60)
    launch.atomic_new(HERE / 'benchmark_release_result.json', {'job_id': job_id, 'returncode': result.returncode,
                      'stdout': result.stdout, 'stderr': result.stderr})
    launch.require(result.returncode == 0, 'benchmark release uncertain; do not retry')
    return record_state('benchmark_queued', benchmark_job_id=job_id, planned_cells=100, science_jobs_submitted=0)


def benchmark_status(job_id):
    result = subprocess.run(['sacct', '--noheader', '--parsable2', '--jobs', str(job_id),
        '--format=JobIDRaw,State,ExitCode'], capture_output=True, text=True, check=True, timeout=60)
    rows = [line.split('|') for line in result.stdout.splitlines() if line.split('|')[0] == str(job_id)]
    if not rows:
        queue = subprocess.run(['squeue', '--noheader', '--jobs', str(job_id), '--format=%i|%T'],
            capture_output=True, text=True, check=True, timeout=60)
        queued = [line.strip().split('|') for line in queue.stdout.splitlines()
                  if line.strip().split('|')[0] == str(job_id)]
        if len(queued) == 1 and queued[0][1] in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
            return queued[0][1], 'accounting_pending'
        launch.require(not queued and not queue.stdout.strip(), 'benchmark accounting missing or ambiguous')
        return 'WAITING_ACCOUNTING', 'unknown'
    launch.require(len(rows) == 1, 'benchmark accounting missing or ambiguous')
    return rows[0][1].split()[0].rstrip('+'), rows[0][2]


def launch_measured(plan, preview):
    profile = Path(plan['selected_profile_path'])
    draft = launch.build_plan('3b', require_admission=True, profile_path=profile,
                             profile_sha256=launch.digest(profile))
    launch.require(science_projection(draft) == science_projection(preview),
                   'measured plan changed the reviewed scientific design or runtime')
    path = HERE / 'measured_draft.json'
    launch.atomic_new(path, draft)
    prepared = launch.prepare(path, launch.digest(path), '3b')
    record_state('submitting_science_held', plan_sha256=prepared['plan_sha256'], profile=str(profile))
    submitted = launch.submit_held(prepared['plan_sha256'], '3b')
    command = [sys.executable, str(CONTROLLER), '--plan', prepared['plan'], '--plan-sha256', prepared['plan_sha256'],
        '--held-ledger', submitted['ledger'], '--held-ledger-sha256', submitted['ledger_sha256'],
        '--model-choice', '3b', '--watch', '--interval-seconds', '60']
    launch.atomic_new(HERE / 'controller_handoff.json', {'at': now(), 'command': command, 'held_jobs': 100})
    record_state('campaign_submitted_controller_handoff', science_jobs_submitted=100, controller_command=command)
    with (HERE / 'release_controller.log').open('x') as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    return record_state('controller_exited', returncode=result.returncode, science_jobs_submitted=100)


def watch():
    plan = launch.read(HERE / 'orchestration.json'); preview = check(plan)
    job_id = launch.read(HERE / 'benchmark_held_audit.json')['job_id']
    release = launch.read(HERE / 'benchmark_release_result.json')
    launch.require(release['returncode'] == 0, 'benchmark release not confirmed')
    with (HERE / '.watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        missing_count = 0
        while True:
            state, code = benchmark_status(job_id)
            missing_count = missing_count + 1 if state == 'WAITING_ACCOUNTING' else 0
            launch.require(missing_count <= 10, 'benchmark allocation absent from queue and accounting for ten polls')
            if state == 'COMPLETED':
                launch.require(code == '0:0', 'benchmark did not exit successfully')
                check(plan)
                return launch_measured(plan, preview)
            launch.require(state in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'WAITING_ACCOUNTING'),
                           'benchmark stopped: ' + state + ' / ' + code)
            record_state('benchmark_' + state.lower(), benchmark_job_id=job_id,
                         planned_cells=100, science_jobs_submitted=0)
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'submit', 'watch', 'status'))
    args = parser.parse_args()
    try:
        if args.phase == 'status':
            print(json.dumps(launch.read(HERE / 'state.json'), indent=2)); return 0
        globals()[args.phase](); return 0
    except Exception as exc:
        if HERE.exists(): record_state('stopped_needs_review', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
