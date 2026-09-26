#!/usr/bin/env python3
"""Verify supplemental execution evidence after both scale levels are admitted.

The historical array plan remains its original submission record. This additive
artifact records the independently observed scheduling amendment; it neither
changes scientific admission nor contacts the scheduler. Default is read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEDULE = ROOT / 'var/artifacts/modebench_scale_schedule_v2'
DEFAULT_RELEASE = ROOT / 'var/data/modebench_scale_release_v1'
REGISTERED_V1_SHA256 = '6f9743cf1a445920abcb3abaa25a1cca1938531ad527fe6741d3002a8be0df57'
REGISTERED_AMENDMENT_SHA256 = '613cd33537b4777ffb7547730143c279b94c58a19e1adf058b8fe66b0690a970'
SCHEMA = 'modebench_scale_supplemental_execution_provenance_v1'
ARRAY_JOB_ID = 31243495
LEVELS = {'level4': '7b', 'level5': '14b'}
DOMAINS = ('countdown', 'graph_coloring', 'python_factors', 'mathir', 'pantry')
SPLITS = {'train': 384, 'dev': 128, 'eval': 128}
COMMAND = ['scontrol', 'update', 'JobId=31243495', 'ArrayTaskThrottle=2']
RESOURCE_FIELDS = {'Account', 'ArrayJobId', 'CPUs/Task', 'Command', 'NumCPUs', 'NumTasks',
                   'Partition', 'ReqNodeList', 'Requeue', 'Restarts', 'TimeLimit', 'TresPerNode',
                   'TresPerTask', 'WorkDir'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_file(), 'missing evidence: ' + str(path))
    return json.loads(path.read_text())


def file_sha(path):
    path = Path(path)
    require(path.is_file(), 'missing pinned file: ' + str(path))
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pins_for(paths):
    return {str(Path(path).resolve()): file_sha(path) for path in paths}


def check_pins(pins):
    require(isinstance(pins, dict) and pins, 'nonempty evidence pins required')
    for path, digest in pins.items():
        require(isinstance(path, str) and Path(path).is_absolute() and str(Path(path).resolve()) == path,
                'canonical evidence pin required: ' + str(path))
        require(file_sha(path) == digest, 'pinned evidence changed: ' + path)
    return dict(pins)


def require_pinned(pins, paths):
    expected = pins_for(paths)
    require(all(pins.get(path) == digest for path, digest in expected.items()), 'required evidence pin missing or changed')
    return expected


def scheduler_fields(raw):
    matches = list(re.finditer(r'(?<!\S)([A-Za-z][A-Za-z0-9_/:]*)=', raw))
    keys = [match[1] for match in matches]
    require(len(keys) == len(set(keys)), 'duplicate scheduler evidence field')
    return {match[1]: raw[match.end():matches[index + 1].start()].strip()
            if index + 1 < len(matches) else raw[match.end():].strip()
            for index, match in enumerate(matches)}


def unique_record(pins, name):
    paths = [Path(path) for path in pins if Path(path).name == name]
    require(len(paths) == 1, 'exactly one pinned execution record required: ' + name)
    return paths[0]


def validate_original_execution(amendment, release):
    pins = {}
    plan_path = pinned_schema_record(amendment['files_sha256'], 'modebench-scale-array-runtime-v3')
    activation_path = unique_record(amendment['files_sha256'], 'controller_activation.json')
    runtime_path = unique_record(amendment['files_sha256'], 'runtime_amendment.json')
    identity_path = unique_record(amendment['files_sha256'], 'controller_identity.json')
    arm_path = unique_record(amendment['files_sha256'], 'controller_arm_result_v2.json')
    plan, activation, runtime, identity, arm = [read(path) for path in
                                              (plan_path, activation_path, runtime_path, identity_path, arm_path)]
    for value, field in ((plan, 'immutable_inputs_sha256'), (activation, 'files_sha256'),
                         (runtime, 'files_sha256'), (identity, 'files_sha256')):
        pins.update(check_pins(value.get(field)))
    original_intent_path, original_result_path = [plan_path.parent / name for name in
                                                 ('submission_intent.json', 'submission_result.json')]
    require_pinned(amendment['files_sha256'], [original_intent_path, original_result_path])
    original_intent, original_result = read(original_intent_path), read(original_result_path)
    require(plan.get('schema') == 'modebench-scale-array-runtime-v3' and plan.get('phase') == 'dev'
            and plan.get('concurrency') == 1 and '--array=0-9%1' in plan.get('submit_command', [])
            and original_intent.get('plan_sha256') == file_sha(plan_path)
            and original_result.get('status') == 'submitted' and original_result.get('array_job_id') == ARRAY_JOB_ID,
            'historical initial submission provenance changed')
    require(activation.get('schema') == 'modebench_scale_composite_activation_v1'
            and activation.get('parent_array_job_id') == ARRAY_JOB_ID
            and activation.get('release_root') == str(release)
            and identity.get('release_root') == str(release)
            and activation.get('parent_root') == identity.get('parent_root') == plan.get('data_root')
            and activation.get('runtime_profile') == runtime.get('runtime_profile') == plan.get('runtime_profile')
            and arm.get('status') == 'started' and arm.get('activation_sha256') == file_sha(activation_path)
            and all(arm.get(key) == amendment['controller'].get(key) for key in ('pid', 'start_ticks', 'command'))
            and amendment['controller'].get('unchanged') is True,
            'sealed controller activation or runtime provenance changed')
    return plan, pins

def pinned_schema_record(pins, schema):
    paths = [Path(path) for path in pins if Path(path).name == 'plan.json'
             and read(path).get('schema') == schema]
    require(len(paths) == 1, 'exactly one pinned execution schema required: ' + schema)
    return paths[0]


def qualify_memory(receipt):
    """Recompute observed memory gates using only the recorded job's scopes."""
    snapshots = receipt['memory_snapshots']; job = str(receipt['job_id'])
    require(set(snapshots) == {'before_load', 'after_load', 'after_generation'} and job.isdigit(),
            'complete own-job memory snapshots required')
    job_paths, peaks, limits = set(), [], []
    for snapshot in snapshots.values():
        scopes = snapshot['scopes_leaf_to_job']; job_path = Path(snapshot['job_scope'])
        require(snapshot.get('job_id') == job and snapshot.get('shared_ancestors_excluded') is True
                and scopes and scopes[-1].get('path') == str(job_path)
                and scopes[-1].get('is_job_scope') is True
                and job_path.name in ('job_' + job, 'job-' + job, 'job_' + job + '.scope', 'job-' + job + '.scope'),
                'memory evidence includes an unrelated or shared job scope')
        job_paths.add(str(job_path))
        for index, scope in enumerate(scopes):
            path = Path(scope['path'])
            require(path == job_path or path.is_relative_to(job_path), 'shared cgroup ancestors cannot qualify job memory')
            require(index == len(scopes) - 1 or scope.get('is_job_scope') is False, 'ambiguous job memory scope')
            if index:
                require(Path(scopes[index - 1]['path']).parent == path, 'discontinuous memory scope chain')
            limit = scope['counters'].get('memory.max')
            if type(limit) is int:
                limits.append(limit)
        counters = scopes[-1]['counters']; peak = counters.get('memory.peak')
        require(type(peak) is int and peak >= 0 and type(counters.get('memory.current')) is int
                and counters['memory.current'] <= peak and 'memory.max' in counters
                and isinstance(counters.get('memory.events'), dict), 'incomplete job memory counters')
        require(all(counters['memory.events'].get(key, 0) == 0 for key in ('oom', 'oom_kill', 'oom_group_kill')),
                'job memory OOM events prevent qualification')
        peaks.append(peak)
    require(len(job_paths) == 1 and max(peaks) < 59 * 1024**3, 'observed job memory does not fit 59 GiB')
    visible_limit = min(limits) if limits else None
    summary = {'requested_host_limit_bytes': 59 * 1024**3, 'observed_job_peak_bytes': max(peaks),
               'visible_process_ancestor_limit_bytes': visible_limit,
               'hard_ceiling_at_or_below_59_gib_visible': visible_limit is not None and visible_limit <= 59 * 1024**3,
               'cache_state': 'uncontrolled_existing_node_cache', 'cold_load_claim': False}
    recorded = receipt['memory_qualification']
    require(all(recorded.get(key) == value for key, value in summary.items()) and recorded.get('passed') is True
            and recorded.get('gates') == {'job_peak_below_59_gib': True, 'job_oom_events_zero': True},
            'recorded memory qualification or its limitations changed')
    return summary


def validate_qualification(reference, baseline, amendment_pins):
    root = Path(reference['root']).resolve()
    paths = {key: Path(reference[key]).resolve() for key in ('receipt', 'plan', 'terminal_accounting')}
    require(paths == {'receipt': root / '14b.json', 'plan': root / 'plan.json',
                      'terminal_accounting': root / 'terminal_sacct.txt'}, 'wrong qualification evidence routing')
    manifest_path = root / 'manifest.json'; source = root / 'smoke.py'
    require_pinned(amendment_pins, [*paths.values(), manifest_path, source])
    plan, receipt, manifest = read(paths['plan']), read(paths['receipt']), read(manifest_path)
    pins = check_pins(plan['files_sha256']); pins.update(check_pins(manifest['files_sha256']))
    require_pinned(plan['files_sha256'], [manifest_path, source, root / 'worker.slurm'])
    expected_settings = {key: baseline['runtime_profile'][key] for key in
                         ('dtype', 'tensor_parallel_size', 'gpu_memory_utilization', 'swap_space', 'enable_prefix_caching')}
    expected_settings['max_model_len'] = 2048
    profile = {**baseline['runtime_profile'], 'memory': '59G', 'partition': 'all', 'time': '00:15:00'}
    require(plan.get('schema') == 'modebench_scale_14b_host_memory_smoke_plan_v1'
            and manifest.get('schema') == 'modebench_scale_14b_host_memory_smoke_manifest_v1'
            and receipt.get('schema') == 'modebench_scale_14b_host_memory_smoke_v1'
            and receipt.get('status') == 'pass' and receipt.get('model_label') == '14b'
            and receipt.get('actual_llm_settings') == manifest.get('numerical_settings') == expected_settings
            and receipt.get('runtime_profile') == manifest.get('runtime_profile') == plan.get('runtime_profile') == profile
            and receipt.get('model') == manifest['models']['14b'] == plan.get('model_identity')
            and receipt.get('manifest_sha256') == file_sha(manifest_path)
            and receipt.get('smoke_source_sha256') == manifest.get('smoke_source_sha256') == file_sha(source)
            and receipt.get('benchmark_candidates_loaded') is False and receipt.get('calibration_outcomes_produced') is False,
            'qualified model, numerical settings, or synthetic workload provenance changed')
    qualified, original = receipt['model'], baseline['models']['14b']
    require(qualified.get('label') == original.get('label') == '14b'
            and qualified.get('path') == original.get('path')
            and qualified.get('weight_file_manifest') == original.get('weights')
            and all(original['configuration_sha256'].get(name) == digest
                    for name, digest in qualified['configuration_sha256'].items()),
            'qualification checkpoint differs from the original plan')
    for name, digest in original['configuration_sha256'].items():
        require(file_sha(Path(original['path']) / name) == digest, 'qualified model configuration changed')
    for weight in original['weights']:
        path = Path(original['path']) / weight['name']
        require(str(path.resolve()) == weight['resolved_path'] and path.stat().st_size == weight['bytes']
                and path.stat().st_mtime_ns == weight['mtime_ns'], 'qualified model shard identity changed')
    sampling = {'prompts': 8, 'n': 8, 'temperature': 1.0, 'top_p': 1.0, 'max_tokens': 192,
                'min_tokens': 192, 'ignore_eos': True, 'seeds': list(range(1000, 9000, 1000))}
    require(all(manifest['synthetic_sampling'].get(key) == value for key, value in sampling.items()),
            'synthetic sampling settings changed')
    counts = receipt.get('output_token_counts', []); prompts = receipt.get('prompt_tokens', [])
    require(len(counts) == 8 and all(len(row) == 8 and all(value == 192 for value in row) for row in counts)
            and len(prompts) == 8 and all(type(value) is int and 1500 <= value <= 2048 - 192 for value in prompts),
            'qualification requires eight long prompts and eight full 192-token samples each')
    allocation = receipt['allocation_environment']
    require(all(allocation.get(key) == value for key, value in
                {'SLURM_MEM_PER_NODE': '60416', 'SLURM_CPUS_PER_TASK': '6', 'SLURM_JOB_PARTITION': 'all'}.items())
            and receipt.get('host', '').split('.')[0] == 'node105'
            and len(receipt.get('devices', [])) == 2 and all('A5000' in value['name'] for value in receipt['devices']),
            'wrong synthetic qualification allocation')
    submission_paths = [root / name for name in ('submission_intent.json', 'submission_result.json')]
    require_pinned(amendment_pins, submission_paths)
    intent, submitted = [read(path) for path in submission_paths]
    pins.update(check_pins(intent['files_sha256']))
    require(intent.get('plan_sha256') == submitted.get('plan_sha256') == file_sha(paths['plan'])
            and submitted.get('intent_sha256') == file_sha(submission_paths[0])
            and intent.get('command') == plan.get('command'), 'qualification submission intent changed')
    require(submitted.get('status') == 'submitted' and str(submitted.get('job_id')) == str(receipt.get('job_id'))
            and submitted.get('command') == plan.get('command'), 'qualification job submission changed')
    accounting = [line.split('|') for line in paths['terminal_accounting'].read_text().splitlines() if line.strip()]
    matching = [row for row in accounting if row[0] == str(receipt['job_id'])]
    require(len(matching) == 1 and matching[0][1] == 'COMPLETED' and len(matching[0]) >= 3 and matching[0][2] == '0:0',
            'qualification job must be terminal COMPLETED with exit 0:0')
    summary = qualify_memory(receipt)
    audit_path = Path(reference['audit']).resolve()
    require(audit_path == root / 'qualification_audit.json', 'wrong qualification audit routing')
    require_pinned(amendment_pins, [audit_path])
    audit = read(audit_path); pins.update(check_pins(audit['files_sha256']))
    require(audit.get('schema') == 'modebench_scale_14b_59g_qualification_audit_v1'
            and audit.get('status') == 'passed' and str(audit.get('job_id')) == str(receipt['job_id'])
            and reference.get('job_id') == audit['job_id'] and audit.get('model_label') == '14b'
            and audit.get('outputs') == 64 and audit.get('synthetic_prompt_count') == 8
            and audit.get('tokens_per_output') == 192 and audit.get('terminal_completed_exit0') is True
            and audit.get('unchanged_numerical_settings') is True
            and audit.get('memory_qualification') == reference.get('memory_qualification') == receipt['memory_qualification']
            and audit.get('actual_allocation') == receipt['allocation_environment'], 'qualification audit changed')
    pins.update(pins_for([audit_path]))
    summary.update({'job_id': str(receipt['job_id']), 'receipt': str(paths['receipt']),
                    'workload': '8 prompts x 8 samples x 192 output tokens',
                    'claim': 'Observed own-job memory below the requested reservation; cache state was uncontrolled.'})
    pins.update(pins_for([*paths.values(), manifest_path, source, *submission_paths]))
    return summary, pins


def expected_commands():
    return [['scontrol', 'update', f'JobId={ARRAY_JOB_ID}_{index}', 'MinMemoryNode=60416']
            for index in range(5, 10)] + [COMMAND]


def validate_task_snapshot(path, index, expected, memory, throttle, *, pending=False):
    fields = scheduler_fields(Path(path).read_text())
    require(fields.get('ArrayTaskId') == str(index) and fields.get('MinMemoryNode') == memory
            and fields.get('ArrayTaskThrottle') == str(throttle)
            and all(fields.get(key) == value for key, value in expected.items())
            and '--array=0-9%1' in fields.get('SubmitLine', '')
            and (not pending or fields.get('JobState') == 'PENDING'),
            'raw scheduler task identity, memory, throttle, or qualified resources changed')
    return fields


def validate_capacity_snapshot(path):
    node = scheduler_fields(Path(path).read_text())
    configured = dict(item.split('=', 1) for item in node['CfgTRES'].split(','))
    allocated = dict(item.split('=', 1) for item in node['AllocTRES'].split(','))
    require(node.get('NodeName') == 'node105'
            and int(configured['gres/gpu:a5000']) - int(allocated.get('gres/gpu:a5000', '0')) >= 2
            and int(node['CPUTot']) - int(node['CPUAlloc']) >= 6
            and int(node['RealMemory']) - int(node['AllocMem']) >= 60416,
            'recorded spare resources do not support the qualified reservation')


def validate_mutation(schedule, amendment):
    commands = expected_commands(); expected = amendment['unchanged_per_cell']
    require(set(expected) == RESOURCE_FIELDS and expected['ArrayJobId'] == str(ARRAY_JOB_ID),
            'incomplete unchanged per-cell contract')
    before = [schedule / f'before_task_{index}.txt' for index in range(4, 10)]
    require_pinned(amendment['files_sha256'], before)
    for index, path in zip(range(4, 10), before):
        validate_task_snapshot(path, index, expected, '60G', 1, pending=index > 4)
    intent_path, result_path = schedule / 'mutation_intent.json', schedule / 'mutation_result.json'
    intent, result = read(intent_path), read(result_path)
    require(intent.get('schema') == 'modebench_scale_schedule_mutation_intent_v2'
            and result.get('schema') == 'modebench_scale_schedule_mutation_result_v2'
            and all(value.get('array_job_id') == ARRAY_JOB_ID and value.get('commands') == commands
                    and value.get('before_throttle') == 1 and value.get('after_throttle') == 2 for value in (intent, result))
            and result.get('status') == 'applied_observed' and result.get('per_cell_memory_verified') is True
            and result.get('unchanged_per_cell_verified') is True,
            'schedule mutation is partial or requires reconciliation')
    pins = check_pins(intent['files_sha256']); pins.update(check_pins(result['files_sha256']))
    amendment_paths = [schedule / 'amendment.json', schedule / 'amendment.sha256.json']
    initial_paths = [schedule / 'apply_schedule.py', schedule / 'pre_apply_qualification_sacct.txt',
                     schedule / 'pre_apply_node105.txt', *[schedule / f'pre_apply_task_{index}.txt' for index in range(4, 10)]]
    require_pinned(intent['files_sha256'], [*amendment_paths, *initial_paths])
    require_pinned(result['files_sha256'], [*amendment_paths, intent_path, *initial_paths])
    validate_capacity_snapshot(schedule / 'pre_apply_node105.txt')
    job = str(read(amendment['qualification']['receipt'])['job_id'])
    accounting = [line.split('|') for line in (schedule / 'pre_apply_qualification_sacct.txt').read_text().splitlines()]
    require(any(row[:3] == [job, 'COMPLETED', '0:0'] for row in accounting), 'qualification was not completed before mutation')
    for index in range(4, 10):
        validate_task_snapshot(schedule / f'pre_apply_task_{index}.txt', index, expected, '60G', 1, pending=index > 4)
    dispatches = result.get('dispatches', [])
    require(len(dispatches) == 6, 'all six successful dispatches are required')
    previous_time = datetime.fromisoformat(intent['created_at'])
    require(previous_time.tzinfo is not None and datetime.fromisoformat(amendment['created_at']) <= previous_time,
            'mutation intent predates the amendment')
    for index, command in enumerate(commands):
        paths = [schedule / f'action_{index}_{suffix}.json' for suffix in ('intent', 'dispatch', 'result')]
        require_pinned(result['files_sha256'], paths)
        action_intent, dispatch_record, action_result = [read(path) for path in paths]
        require(action_intent.get('schema') == 'modebench_scale_schedule_action_intent_v2'
                and action_result.get('schema') == 'modebench_scale_schedule_action_result_v2'
                and action_result.get('status') == 'applied_observed', 'action ledger schema or status changed')
        for value in (action_intent, action_result):
            pins.update(check_pins(value['files_sha256']))
        for value in (action_intent, dispatch_record, action_result):
            require(value.get('index') == index and value.get('command') == command, 'mutation action order or command changed')
            moment = datetime.fromisoformat(value['created_at'])
            require(moment.tzinfo is not None and moment >= previous_time, 'mutation action chronology changed')
            previous_time = moment
        before_paths = [schedule / f'action_{index}_before_task_{task}.txt' for task in range(4, 10)]
        linked = [intent_path, *before_paths]
        if index: linked.append(schedule / f'action_{index - 1}_result.json')
        if index == 5:
            linked.append(schedule / 'before_throttle_node105.txt')
            validate_capacity_snapshot(linked[-1])
        require_pinned(action_intent['files_sha256'], linked)
        for task, path in zip(range(4, 10), before_paths):
            memory = '59G' if 4 < task <= index + 4 else '60G'
            validate_task_snapshot(path, task, expected, memory, 1, pending=task > 4)
        after_paths = [schedule / f'action_{index}_after_task_{task}.txt' for task in range(4, 10)]
        require_pinned(action_result['files_sha256'], [paths[0], paths[1], *after_paths])
        keys = ('index', 'command', 'returncode', 'stdout', 'stderr', 'timed_out')
        dispatch = {key: action_result[key] for key in keys}
        require(dispatch == {key: dispatch_record[key] for key in keys} == dispatches[index]
                and type(dispatch['returncode']) is int and dispatch['returncode'] == 0
                and dispatch['timed_out'] is False, 'ambiguous or failed scheduler action requires reconciliation')
        require(set(action_result.get('tasks', {})) == {str(task) for task in range(4, 10)}, 'missing per-action task observations')
        for task, path in zip(range(4, 10), after_paths):
            memory = '59G' if 4 < task <= index + 5 else '60G'
            fields = validate_task_snapshot(path, task, expected, memory, 2 if index == 5 else 1,
                                            pending=index < 5 and task > 4)
            require(action_result['tasks'][str(task)] == fields, 'action task summary differs from raw state')
    require(datetime.fromisoformat(result['created_at']) >= previous_time, 'mutation result predates its actions')
    after_paths = [schedule / f'after_task_{index}.txt' for index in range(4, 10)]
    require_pinned(result['files_sha256'], [*after_paths, schedule / 'after_sacct.txt', schedule / 'after_node105.txt'])
    require(set(result.get('tasks', {})) == {str(task) for task in range(4, 10)}, 'missing final task observations')
    for index, path in zip(range(4, 10), after_paths):
        after = validate_task_snapshot(path, index, expected, '60G' if index == 4 else '59G', 2)
        require(after.get('SubmitLine') == scheduler_fields(before[index - 4].read_text()).get('SubmitLine')
                and result['tasks'].get(str(index)) == after, 'final task summary or historical submission line changed')
    pins.update(pins_for([intent_path, result_path]))
    return pins


def validate_schedule(schedule=DEFAULT_SCHEDULE, release=DEFAULT_RELEASE, *,
                      expected_amendment_sha256=REGISTERED_AMENDMENT_SHA256):
    schedule, release = Path(schedule).resolve(), Path(release).resolve()
    amendment_path, sidecar = schedule / 'amendment.json', schedule / 'amendment.sha256.json'
    digest = file_sha(amendment_path)
    require(digest == read(sidecar).get('sha256')
            and (expected_amendment_sha256 is None or digest == expected_amendment_sha256), 'registered amendment or sidecar changed')
    amendment = read(amendment_path)
    overrides = {str(index): {'before': '60G', 'after': '59G', 'before_mb': 61440, 'after_mb': 60416}
                 for index in range(5, 10)}
    require(amendment.get('schema') == 'modebench_scale_parent_array_schedule_amendment_v2'
            and amendment.get('status') == 'prospectively_registered_before_scheduler_change'
            and amendment.get('array_job_id') == ARRAY_JOB_ID
            and amendment.get('array_throttle_before') == 1 and amendment.get('array_throttle_after') == 2
            and amendment.get('per_cell_memory_overrides') == overrides
            and amendment.get('mutation_commands') == expected_commands()
            and amendment.get('initial_submit_array_spec') == '0-9%1'
            and amendment.get('scientific_changes') == [] and amendment.get('heldout_outcomes_observed') is False,
            'wrong v2 scheduling amendment scope')
    destination = amendment['required_final_execution_provenance']
    require(destination.get('schema') == SCHEMA and destination.get('path') == str(release / 'execution_provenance.json')
            and destination.get('required_before_goal_completion') is True, 'wrong final provenance destination')
    pins = check_pins(amendment['files_sha256']); pins.update(pins_for([amendment_path, sidecar]))
    # Preserve the rejected proposal and failed smoke as part of the true history.
    v1 = schedule.parent / 'modebench_scale_schedule_v1'
    preserved = [v1 / 'amendment.json', v1 / 'amendment.sha256.json', v1 / 'resource_guard_failure.json',
                 schedule / 'capacity_smoke_owner_r2/failure_audit.json']
    require_pinned(amendment['files_sha256'], preserved)
    require(file_sha(preserved[0]) == read(preserved[1]).get('sha256') == REGISTERED_V1_SHA256,
            'preserved v1 amendment changed')
    guard, failed_smoke = read(preserved[2]), read(preserved[3])
    require(guard.get('status') == 'not_applied' and guard.get('scientific_jobs_changed') is False
            and failed_smoke.get('status') == 'failed_before_model_loading'
            and failed_smoke.get('scientific_jobs_changed') is False
            and not (v1 / 'mutation_intent.json').exists() and not (v1 / 'mutation_result.json').exists(),
            'preserved unapplied amendment or failed qualification history changed')
    pins.update(check_pins(read(preserved[0])['files_sha256']))
    pins.update(check_pins(guard['files_sha256'])); pins.update(check_pins(failed_smoke['files_sha256']))
    baseline, original_pins = validate_original_execution(amendment, release); pins.update(original_pins)
    qualification, smoke_pins = validate_qualification(amendment['qualification'], baseline, amendment['files_sha256'])
    pins.update(smoke_pins); pins.update(validate_mutation(schedule, amendment))
    return amendment, pins, qualification


def validate_release(release=DEFAULT_RELEASE):
    """Authenticate all admitted sources and independently recount final rows."""
    release = Path(release).resolve()
    campaign_path = release / 'admission.json'; campaign = read(campaign_path)
    require(campaign.get('schema') == 'modebench_scale_composite_campaign_admission_v1'
            and campaign.get('difficulty_matched') is True and set(campaign.get('levels', {})) == set(LEVELS),
            'both admitted levels are required')
    pins = check_pins(campaign.get('files_sha256')); summaries = {}
    for level, model in LEVELS.items():
        path = release / level / 'admission.json'; value = read(path)
        require_pinned(campaign['files_sha256'], [path])
        require(campaign['levels'][level].get('status') == 'admitted'
                and campaign['levels'][level].get('admission') == str(path)
                and value.get('schema') == 'modebench_scale_composite_level_admission_v1'
                and value.get('level') == level and value.get('model_label') == model
                and value.get('difficulty_matched') is True and value.get('test_split') == 'eval'
                and value.get('treatment_training_started') is False
                and set(value.get('domains', {})) == set(DOMAINS), 'incomplete or misrouted level admission')
        domain_pins = check_pins(value.get('files_sha256')); pins.update(domain_pins)
        manifest_path = release / level / 'source_manifest.json'; require_pinned(domain_pins, [manifest_path])
        manifest = read(manifest_path)
        require(manifest.get('level') == level and manifest.get('model_label') == model
                and set(manifest.get('sources', {})) == set(DOMAINS), 'wrong admitted source manifest')
        for domain, source in value['domains'].items():
            root = Path(source['source_root']).resolve(); base = root / level / 'dataset' / domain
            require(source.get('source_kind') in ('campaign_v1', 'domain_revision_v1')
                    and all(source.get(key) == item for key, item in manifest['sources'][domain].items()),
                    'admitted domain source routing changed')
            identity_path = base / 'identity.json'; identity = read(identity_path)
            protocol, recipe = root / 'protocol.json', root / level / 'recipes' / (domain + '.json')
            audit_path = root / level / 'confirmation' / (domain + '.json')
            receipt = root / level / 'results/confirmation' / (domain + '.json')
            require_pinned(domain_pins, [protocol, recipe, identity_path, audit_path, receipt])
            require(identity.get('level') == level and identity.get('domain') == domain
                    and identity.get('protocol_sha256') == file_sha(protocol) == source.get('protocol_sha256')
                    and identity.get('recipe_sha256') == file_sha(recipe), 'admitted dataset identity changed')
            audit = read(audit_path)
            require(audit.get('level') == level and audit.get('domain') == domain
                    and audit.get('difficulty_matched') is True and audit.get('original_grader_replayed_attempts') == 4096
                    and audit.get('receipt_sha256') == file_sha(receipt)
                    and audit.get('recipe_sha256') == file_sha(recipe)
                    and audit.get('dataset_identity_sha256') == file_sha(identity_path),
                    'complete heldout replay evidence required for every domain')
            require(set(source.get('splits', {})) == set(SPLITS), 'complete train/dev/test splits required')
            for split, count in SPLITS.items():
                record = source['splits'][split]; jsonl = base / (split + '.jsonl')
                require(record.get('path') == str(base / split) and record.get('rows') == count
                        and identity['splits'][split].get('rows') == count, 'wrong admitted split path or count')
                rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
                require(len(rows) == count and sha(rows) == record.get('rows_sha256')
                        == identity['splits'][split].get('rows_sha256'), 'admitted split rows changed')
                files = [item for item in (base / split).rglob('*') if item.is_file()]
                require(files, 'missing admitted Arrow split')
                require_pinned(domain_pins, [jsonl, *files])
            link = release / level / 'dataset' / domain
            require(link.is_symlink() and link.resolve() == base, 'admitted dataset pointer changed')
        summaries[level] = {'model_label': model, 'admission': str(path), 'domains': list(DOMAINS),
                            'rows_per_domain': dict(SPLITS)}
    pins.update(pins_for([campaign_path]))
    return summaries, pins


def verify(schedule=DEFAULT_SCHEDULE, release=DEFAULT_RELEASE, *, publish=False,
           expected_amendment_sha256=REGISTERED_AMENDMENT_SHA256):
    amendment, pins, qualification = validate_schedule(schedule, release, expected_amendment_sha256=expected_amendment_sha256)
    levels, release_pins = validate_release(release); pins.update(release_pins)
    pins.update(pins_for([Path(__file__).resolve()]))
    value = {'schema': SCHEMA, 'status': 'verified', 'difficulty_matched': True,
             'scope': 'Supplemental execution provenance; existing scientific admissions remain authoritative.',
             'parent_array_job_id': ARRAY_JOB_ID, 'initial_submit_array_spec': amendment['initial_submit_array_spec'],
             'effective_parent_array_task_throttle': 2, 'maximum_parent_array_gpus': 4,
             'per_cell_numerical_settings_unchanged': True, 'scientific_changes': [],
             'per_cell_memory_overrides': amendment['per_cell_memory_overrides'], 'qualification': qualification,
             'later_array_scheduling': 'This amendment applies only to the parent array; later arrays retain their own sealed plans.',
             'levels': levels, 'files_sha256': pins}
    destination = Path(release).resolve() / 'execution_provenance.json'
    if destination.exists():
        require(read(destination) == value, 'immutable supplemental execution provenance changed')
    elif publish:
        fd, temporary = tempfile.mkstemp(prefix='.execution_provenance.', dir=destination.parent)
        try:
            with os.fdopen(fd, 'w') as handle:
                json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
            os.link(temporary, destination)
        finally:
            os.unlink(temporary)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule-root', type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument('--release-root', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--publish', action='store_true', help='Create the immutable supplemental artifact after all checks pass.')
    args = parser.parse_args(argv)
    result = verify(args.schedule_root, args.release_root, publish=args.publish)
    destination = args.release_root.resolve() / 'execution_provenance.json'
    print(json.dumps({'status': 'verified' if destination.exists() else 'ready_to_publish',
                      'path': str(destination), 'schema': SCHEMA, 'pinned_files': len(result['files_sha256'])}, sort_keys=True))
    return result


if __name__ == '__main__':
    main()
