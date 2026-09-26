#!/usr/bin/env python3
"""Recover E123's failed bootstrap without changing its qualification or science."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e123_benchmark_then_launch as auto

ROOT = auto.ROOT
SOURCE = Path(__file__).resolve()
OLD = ROOT / 'var/artifacts/e123_a100_launch_20260909'
HERE = ROOT / 'var/artifacts/e123_a100_bootstrap_recovery_20260910'
PROTOCOL = ROOT / 'paper/preregistration/e123_a100_bootstrap_recovery_20260910.md'
TEST = ROOT / 'tests/test_e123_benchmark_bootstrap_recovery.py'
FAILED_JOB = 31160003


def wrapper(preview, command):
    """Bind mutable infrastructure to the real repo before sourcing a snapshot."""
    exports = {
        'OAT_ZERO_REPO_ROOT': str(ROOT), 'ROOT_DIR': str(ROOT),
        'MAXENT_GRPO_ROOT': str(ROOT), 'MAXENT_GRPO_VAR_ROOT': str(ROOT / 'var'),
        'CUDA_HOME': str(ROOT / 'var/cuda124_toolkit'),
        'TORCH_EXTENSIONS_DIR': str(ROOT / 'var/cache/torch_extensions'),
        'PYTHONPYCACHEPREFIX': str(ROOT / 'var/pycache'),
        'TMPDIR': str(ROOT / 'var/tmp'),
    }
    lines = ['#!/bin/bash', 'set -euo pipefail', 'cd ' + shlex.quote(str(ROOT))]
    lines += ['export ' + key + '=' + shlex.quote(value) for key, value in exports.items()]
    lines += [
        'source ' + shlex.quote(str(Path(preview['snapshot_root']) / 'ops/repo_env.sh')),
        'test -x "$CUDA_HOME/bin/nvcc"',
        'export LD_LIBRARY_PATH=' + shlex.quote(str(auto.PYTHON.parent.parent / 'lib')) + '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
        'export USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_TF=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        'exec ' + shlex.join(command),
    ]
    return '\n'.join(lines) + '\n'


def inventory(snapshot):
    root = Path(snapshot['root'])
    expected = set(snapshot['inventory']) | {'SNAPSHOT_IDENTITY.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    auto.launch.require(expected <= actual, 'frozen snapshot files are missing')
    for relative, item in snapshot['inventory'].items():
        auto.launch.require(auto.launch.digest(root / relative) == item['sha256'], 'frozen file bytes changed: ' + relative)
    extras = actual - expected
    auto.launch.require(all(x.startswith('var/') for x in extras), 'unexpected files outside bootstrap-created var tree')
    auto.launch.require(not any(x.startswith('var/') for x in expected), 'registered snapshot includes var; do not move it')
    return sorted(extras)


def quarantine_cache(snapshot):
    extras = inventory(snapshot)
    root = Path(snapshot['root'])
    target = HERE / 'quarantined_snapshot_var'
    intent = HERE / 'cache_quarantine_intent.json'
    receipt = HERE / 'cache_quarantine_result.json'
    if not extras:
        auto.launch.verify_snapshot(snapshot)
        return
    auto.launch.require(not target.exists(), 'quarantine destination exists while extras remain')
    observed = {relative: {'sha256': auto.launch.digest(root / relative), 'bytes': (root / relative).stat().st_size}
                for relative in extras}
    payload = {'source': str(root / 'var'), 'destination': str(target), 'files': observed,
               'expected_frozen_files_changed': False, 'at': auto.now()}
    if intent.exists():
        saved = auto.launch.read(intent)
        auto.launch.require(saved['source'] == payload['source'] and saved['destination'] == payload['destination']
                            and saved['files'] == observed, 'quarantine intent changed')
    else:
        auto.launch.atomic_new(intent, payload)
    os.rename(root / 'var', target)
    auto.launch.verify_snapshot(snapshot)
    auto.write(receipt, {'status': 'complete', 'at': auto.now(), 'files_preserved': len(extras),
                         'destination': str(target), 'frozen_snapshot_inventory_restored': True})


def prepare():
    auto.launch.require(not (HERE / 'orchestration.json').exists(), 'recovery already prepared; inspect before retrying')
    old = auto.launch.read(OLD / 'orchestration.json')
    auto.launch.require(auto.benchmark_status(FAILED_JOB) == ('FAILED', '1:0'), 'old benchmark failure state changed')
    queue = subprocess.run(['squeue', '--noheader', '--user', str(os.getuid()), '--format=%i'],
                           capture_output=True, text=True, check=True)
    auto.launch.require(str(FAILED_JOB) not in queue.stdout.split(), 'old benchmark remains queued')
    auto.launch.require(auto.launch.read(OLD / 'benchmark_held_audit.json')['job_id'] == FAILED_JOB,
                        'old benchmark identity changed')
    auto.launch.require(not auto.launch.PLAN.exists() and not auto.launch.CLAIM.exists(), 'science launch already began')
    preview = auto.launch.read(old['preview_path'])
    auto.launch.require(auto.launch.digest(old['preview_path']) == old['preview_sha256'], 'reviewed preview changed')
    auto.launch.verify_initial_ledger(auto.launch.read(auto.launch.LEDGER), preview['cells'])
    auto.launch.common.verify_pins(old['files_sha256'])
    HERE.mkdir(parents=True, exist_ok=True)
    with (OLD / '.watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        extras = inventory(preview['snapshot'])
    plan_path = OLD / 'benchmark/plan.json'
    benchmark = auto.launch.read(plan_path)
    auto.launch.require(set((OLD / 'benchmark').iterdir()) == {plan_path},
                        'unexpected benchmark evidence exists; reconcile before reusing output')
    script = HERE / 'benchmark.slurm'
    script.write_text(wrapper(preview, [str(auto.PYTHON), str(auto.BENCH), 'suite', '--plan', str(plan_path), '--warmup', '8', '--steps', '32']))
    script.chmod(0o755)
    preflight = HERE / 'preflight.sh'
    preflight.write_text(wrapper(preview, [str(auto.PYTHON), str(auto.BENCH), 'contract', '--plan', str(plan_path)]))
    preflight.chmod(0o755)
    result = copy.deepcopy(old)
    result.update(created_at=auto.now(), recovery={'reason': 'snapshot-relative CUDA_HOME and Python cache root',
        'failed_job_id': FAILED_JOB, 'predecessor_orchestration': str(OLD / 'orchestration.json'),
        'predecessor_sha256': auto.launch.digest(OLD / 'orchestration.json'),
        'science_or_benchmark_algorithm_changed': False, 'scientific_jobs_submitted': 0})
    result['benchmark_command'] = [x.replace(str(OLD / 'benchmark-%j.log'), str(HERE / 'benchmark-%j.log'))
                                   for x in old['benchmark_command'][:-1]] + [str(script)]
    for p in (SOURCE, PROTOCOL, TEST, script, preflight, OLD / 'orchestration.json'):
        result['files_sha256'][str(p)] = auto.launch.digest(p)
    auto.launch.atomic_new(HERE / 'orchestration.json', result)
    auto.HERE = HERE
    auto.record_state('prepared_bootstrap_recovery', failed_job_id=FAILED_JOB,
                      planned_cells=100, science_jobs_submitted=0,
                      unregistered_snapshot_cache_files=len(extras),
                      preflight_command=['bash', str(preflight)])
    print(json.dumps({'prepared': True, 'orchestration': str(HERE / 'orchestration.json'),
                      'preflight': ['bash', str(preflight)], 'submit': [str(auto.PYTHON), str(SOURCE), 'submit'],
                      'watch': [str(auto.PYTHON), str(SOURCE), 'watch']}))


def repair_bootstrap():
    plan = auto.launch.read(HERE / 'orchestration.json')
    auto.launch.common.verify_pins(plan['files_sha256'])
    preview = auto.launch.read(plan['preview_path'])
    auto.launch.require(auto.launch.digest(plan['preview_path']) == plan['preview_sha256'], 'preview changed')
    with (OLD / '.watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        quarantine_cache(preview['snapshot'])
    auto.check(plan)
    output = HERE / 'preflight.log'
    auto.launch.require(not output.exists(), 'preflight evidence exists; inspect before retrying')
    with output.open('x') as log:
        result = subprocess.run(['bash', str(HERE / 'preflight.sh')], stdout=log, stderr=subprocess.STDOUT)
    auto.write(HERE / 'preflight_result.json', {'at': auto.now(), 'returncode': result.returncode, 'log': str(output)})
    auto.launch.require(result.returncode == 0, 'corrected-bootstrap CPU preflight failed')
    auto.check(plan)
    auto.record_state('bootstrap_repaired_preflight_passed', planned_cells=100, science_jobs_submitted=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'repair-bootstrap', 'submit', 'watch', 'status'))
    args = parser.parse_args()
    auto.HERE = HERE
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'repair-bootstrap':
        repair_bootstrap()
    elif args.phase == 'status':
        print(json.dumps(auto.launch.read(HERE / 'state.json'), indent=2))
    else:
        if args.phase == 'submit':
            contract = auto.launch.read(OLD / 'benchmark/production_loss_contract.json')
            auto.launch.require(contract.get('status') == 'passed' and len(contract.get('comparisons', [])) == 12,
                                'corrected-bootstrap production-loss preflight must pass before submission')
        try:
            getattr(auto, args.phase)()
        except Exception as exc:
            auto.record_state('stopped_needs_review', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
