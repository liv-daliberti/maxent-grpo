#!/usr/bin/env python3
"""Versioned E123 diagnostic repair; prepare/preflight perform no submission."""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e123_benchmark_then_launch as auto
import recover_e123_benchmark_bootstrap_20260910 as bootstrap

ROOT = auto.ROOT
SOURCE = Path(__file__).resolve()
OLD = ROOT / 'var/artifacts/e123_a100_bootstrap_recovery_20260910'
ORIGINAL = ROOT / 'var/artifacts/e123_a100_launch_20260909'
HERE = ROOT / 'var/artifacts/e123_runtime_harness_repair_20260911'
BENCH = SOURCE.with_name('benchmark_e123_a100_runtime_repaired_20260911.py')
ORIGINAL_BENCH = auto.BENCH
PROTOCOL = ROOT / 'paper/preregistration/e123_runtime_harness_repair_20260911.md'
TEST = ROOT / 'tests/test_e123_runtime_harness_repair_20260911.py'
FAILED_JOB = 31170074


def repaired_benchmark():
    spec = importlib.util.spec_from_file_location('e123_repaired_benchmark', BENCH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def wrapper(preview, command):
    text = bootstrap.wrapper(preview, command)
    marker = 'test -x "$CUDA_HOME/bin/nvcc"\n'
    auto.launch.require(text.count(marker) == 1, 'bootstrap wrapper layout changed')
    prefix = shlex.quote(str(auto.PYTHON.parent))
    additions = ('export PATH=' + prefix + ':"$PATH"\n'
                 + 'test "$(command -v ninja)" = ' + shlex.quote(str(auto.PYTHON.parent / 'ninja')) + '\n'
                 + 'ninja --version\n')
    return text.replace(marker, marker + additions)


def verify_harness_delta():
    old = ast.parse(ORIGINAL_BENCH.read_text())
    new = ast.parse(BENCH.read_text())
    old_functions = {n.name: n for n in old.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new_functions = {n.name: n for n in new.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    auto.launch.require(set(new_functions) - set(old_functions) == {'optimizer_sketch_indices'}, 'unexpected added benchmark functions')
    auto.launch.require(set(old_functions) <= set(new_functions), 'benchmark function removed')
    for name in old_functions:
        if name not in ('optimizer_sketch', 'selected_profile', 'suite'):
            auto.launch.require(ast.dump(old_functions[name]) == ast.dump(new_functions[name]), 'benchmark gate or algorithm changed: ' + name)
    old_body = [n for n in old.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    new_body = [n for n in new.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    auto.launch.require([ast.dump(n) for n in old_body] == [ast.dump(n) for n in new_body], 'benchmark constants or entrypoint changed')
    return {'unchanged_original_functions': sorted(set(old_functions) - {'optimizer_sketch', 'selected_profile', 'suite'}),
            'changed_functions': ['optimizer_sketch', 'selected_profile', 'suite'], 'added_function': 'optimizer_sketch_indices', 'eligibility_amendment': 'gpu_adam only after both combined profiles fail qualification'}


def archive_inputs():
    paths = [ORIGINAL_BENCH, OLD / 'benchmark.slurm', OLD / 'orchestration.json',
             OLD / 'state.json', OLD / 'benchmark-31170074.log',
             ORIGINAL / 'reviewed_preview.json', ORIGINAL / 'benchmark/plan.json',
             ORIGINAL / 'benchmark/suite_status.json', ORIGINAL / 'benchmark/production_loss_contract.json']
    for name in repaired_benchmark().VARIANTS:
        paths += [ORIGINAL / 'benchmark' / name / item for item in ('candidate_result.json', 'worker.log')]
    paths += sorted(OLD.glob('benchmark_*intent.json')) + sorted(OLD.glob('benchmark_*result.json'))
    paths += [OLD / 'benchmark_held_audit.json']
    inventory = {}
    for path in paths:
        target = HERE / 'failed_inputs' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        with target.open('xb') as stream:
            stream.write(content)
        auto.launch.require(auto.launch.digest(path) == auto.launch.digest(target), 'archive differs: ' + str(path))
        inventory[str(path)] = {'archived_path': str(target), 'sha256': auto.launch.digest(target), 'bytes': len(content)}
    return inventory


def prepare():
    auto.launch.require(not HERE.exists(), 'repair directory exists; inspect rather than repeating preparation')
    old = auto.launch.read(OLD / 'orchestration.json')
    auto.launch.require(auto.benchmark_status(FAILED_JOB) == ('FAILED', '1:0'), 'predecessor is not the recorded failed benchmark')
    auto.launch.require(auto.launch.read(OLD / 'benchmark_held_audit.json')['job_id'] == FAILED_JOB, 'predecessor job identity differs')
    auto.launch.require(not auto.launch.PLAN.exists() and not auto.launch.CLAIM.exists(), 'E123 science submission already began')
    preview = auto.check(old)
    auto.launch.verify_initial_ledger(auto.launch.read(auto.launch.LEDGER), preview['cells'])
    delta = verify_harness_delta()
    with (OLD / '.watch.lock').open('a+') as old_lock:
        fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        HERE.mkdir(parents=True)
        archived = archive_inputs()
        auto.launch.atomic_new(HERE / 'failed_input_archive.json', {'at': auto.now(), 'job_id': FAILED_JOB, 'files': archived})
    bench = repaired_benchmark()
    plan = bench.prepare(old['preview_path'], HERE / 'benchmark')
    previous = auto.launch.read(ORIGINAL / 'benchmark/plan.json')
    projection = lambda p: {k: v for k, v in p.items() if k not in ('output_root', 'runtime', 'plan_sha256')}
    auto.launch.require(projection(plan) == projection(previous), 'benchmark design, defaults or gates changed')
    before_runtime, after_runtime = copy.deepcopy(previous['runtime']), copy.deepcopy(plan['runtime'])
    before_pins, after_pins = before_runtime.pop('files_sha256'), after_runtime.pop('files_sha256')
    auto.launch.require(before_runtime == after_runtime, 'immutable training runtime differs')
    auto.launch.require({k: v for k, v in before_pins.items() if k != str(ORIGINAL_BENCH)} ==
                        {k: v for k, v in after_pins.items() if k != str(BENCH)}, 'training runtime pins differ')
    script, preflight = HERE / 'benchmark.slurm', HERE / 'preflight.sh'
    command = [str(auto.PYTHON), str(BENCH)]
    script.write_text(wrapper(preview, command + ['suite', '--plan', str(HERE / 'benchmark/plan.json'), '--warmup', '8', '--steps', '32']))
    preflight.write_text(wrapper(preview, command + ['contract', '--plan', str(HERE / 'benchmark/plan.json')]))
    script.chmod(0o755); preflight.chmod(0o755)
    result = copy.deepcopy(old)
    result.update(created_at=auto.now(), selected_profile_path=str(HERE / 'benchmark/selected_profile.json'),
        recovery={'reason': 'Ninja PATH, exact integer indices, prospectively approved existing GPU Adam fallback', 'failed_job_id': FAILED_JOB,
                  'predecessor_orchestration': str(OLD / 'orchestration.json'), 'predecessor_sha256': auto.launch.digest(OLD / 'orchestration.json'),
                  'scientific_inputs_and_memory_numerical_evaluation_gates_unchanged': True, 'harness_delta': delta, 'science_jobs_submitted': 0})
    result['launch_policy'] = 'PREPARED ONLY: no scheduler submission, watcher or scientific release authorized by this packet'
    result['release_blockers'] = [
        'The shared storage classifier does not yet admit this benchmark as a known writer.',
        'A fresh full reserve including released pending external writers, E122 terminal reserves and benchmark peak must fit under the shared admission lock.',
        'No immediate eligible node302 A100/memory slot was established.',
        'The existing E123 science controller requires separately reviewed shared storage coordination before it can be armed.',
        'Any future measured launch must audit the approved operational source-pin transition against all 100 frozen scientific cells.']
    result['benchmark_command'] = [x.replace(str(OLD / 'benchmark-%j.log'), str(HERE / 'benchmark-%j.log')) for x in old['benchmark_command'][:-1]] + [str(script)]
    for path in (SOURCE, BENCH, PROTOCOL, TEST, SOURCE.with_name('launch_e123_level3_qwen3b_factorial_runtime_repaired_20260911.py'), SOURCE.with_name('control_e123_level3_release_runtime_repaired_20260911.py'), script, preflight, HERE / 'benchmark/plan.json',
                 HERE / 'failed_input_archive.json', OLD / 'orchestration.json', auto.PYTHON.parent / 'ninja'):
        result['files_sha256'][str(path)] = auto.launch.digest(path)
    auto.launch.atomic_new(HERE / 'orchestration.json', result)
    auto.HERE = HERE
    auto.check(result)
    auto.record_state('prepared_runtime_repair', benchmark_submitted=False, science_jobs_submitted=0,
                      seven_profiles_unchanged=True, prior_combined_gpu_profiles_oom=True)
    print(json.dumps({'plan': str(HERE / 'orchestration.json'), 'sha256': auto.launch.digest(HERE / 'orchestration.json'),
        'preflight': [str(auto.PYTHON), str(SOURCE), 'preflight'], 'benchmark_command_for_review': result['benchmark_command'],
        'submission_blocked': True}, indent=2))


def preflight():
    plan = auto.launch.read(HERE / 'orchestration.json'); auto.check(plan)
    with (HERE / 'preflight.log').open('x') as stream:
        result = subprocess.run(['bash', str(HERE / 'preflight.sh')], stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    auto.launch.atomic_new(HERE / 'preflight_result.json', {'at': auto.now(), 'returncode': result.returncode})
    auto.launch.require(result.returncode == 0, 'repaired runtime production-loss contract failed')
    auto.check(plan)
    auto.record_state('cpu_preflight_passed', benchmark_submitted=False, science_jobs_submitted=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'preflight', 'status'))
    args = parser.parse_args()
    auto.HERE = HERE
    if args.phase in ('prepare', 'preflight'):
        globals()[args.phase]()
    elif args.phase == 'status':
        print(json.dumps(auto.launch.read(HERE / 'state.json'), indent=2))



if __name__ == '__main__':
    main()
