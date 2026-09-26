#!/usr/bin/env python3
"""Add one authenticated offline-inference array to unchanged training reserves.

Read-only: callers serialize release intents under the common storage/controller
locks. Unknown arrays or GPU writers fail closed. No checkpoint credit is given
to inference, and each queued array task retains its parent/task identity.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import e122_shared_storage_admission as original
import bounded_checkpoint_zip_metadata_20260911 as bounded_zip

ROOT = original.ROOT
CERTIFICATE = ROOT / 'var/artifacts/e122_array_writer_classification_20260911/classification.json'
CERTIFICATE_SHA256 = '6cfd677b1eb4153867a1c89dddc56460b2e5e7fc95a5c3ec61721f64e8858656'
ORIGINAL_SHA256 = '183d2171dae461d19849c6584e1c1e36f52a473a1d66eec9add3acffb5267b3d'
BASE_SHA256 = '8c0e13885a63c1ce07c2f0e3d61ffef2964c74999c32909d42a8e37dbe870934'
ARRAY_ID = 31243495
PER_TASK_BYTES = 32 * 1024**3
SYSTEMS_ID = 31161634
SYSTEMS_BYTES = 220 * 1024**3
SYSTEMS_CERTIFICATE = CERTIFICATE.with_name('e124_systems_classification.json')
SYSTEMS_CERTIFICATE_SHA256 = 'a20f5bcfb9c468e87c4fa7af3156c011f7773f78c11cf7f2e6eec211f5d99e47'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_json(path, limit=32*1024**2):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(ROOT) and not path.is_symlink()
            and path.resolve(strict=True) == path and path.is_file() and path.stat().st_size <= limit,
            'unsafe or oversized storage evidence')
    return json.loads(path.read_text())


def certificate():
    require(original.digest(CERTIFICATE) == CERTIFICATE_SHA256, 'array writer certificate changed')
    require(original.digest(Path(original.__file__)) == ORIGINAL_SHA256
            and original.digest(Path(original.base.__file__)) == BASE_SHA256, 'original storage helper changed')
    value = bounded_json(CERTIFICATE)
    require(value['schema'] == 'bounded_modebench_array_writer_classification_v1'
            and value['status'] == 'qualified_for_explicit_budget_adapter'
            and value['array_job_id'] == ARRAY_ID and len(value['cells']) == 10
            and value['per_task_estimated_future_reserve_bytes'] == PER_TASK_BYTES,
            'array writer qualification differs')
    require(len(value['source_pins']) <= 1200, 'unbounded array source pins')
    for source, expected in value['source_pins'].items():
        path = Path(source)
        require(path.is_absolute() and path.is_relative_to(ROOT) and not path.is_symlink()
                and path.resolve(strict=True) == path and path.is_file()
                and path.stat().st_size <= 32*1024**2, 'unsafe array source pin')
        require(original.digest(path) == expected, 'array inference source changed')
    plan = bounded_json(value['plan_path'])
    require(original.digest(Path(value['plan_path'])) == value['plan_sha256']
            and plan['concurrency'] == 1 and plan['treatment_training_started'] is False
            and len(plan['cells']) == 10, 'array plan changed')
    return value, plan


def systems_certificate():
    require(original.digest(SYSTEMS_CERTIFICATE) == SYSTEMS_CERTIFICATE_SHA256,
            'E124 systems certificate changed')
    value = bounded_json(SYSTEMS_CERTIFICATE)
    require(value['schema'] == 'e122_exact_e124_systems_writer_v1'
            and value['job_id'] == SYSTEMS_ID and value['model_choice'] == '7b'
            and value['future_reserve_bytes'] == SYSTEMS_BYTES
            and value['checkpoint_credit_bytes'] == 0, 'E124 systems profile differs')
    require(len(value['source_pins']) <= 1200, 'unbounded systems source pins')
    for source, expected in value['source_pins'].items():
        path = Path(source)
        require(path.is_absolute() and path.is_relative_to(ROOT) and not path.is_symlink()
                and path.resolve(strict=True) == path and path.is_file()
                and path.stat().st_size <= 32*1024**2, 'unsafe systems source pin')
        require(original.digest(path) == expected, 'E124 systems source changed')
    original.base.bounded_run_path(value['run_dir'], benchmark=True)
    return value


def optional_number(value, label):
    require(isinstance(value, dict) and type(value.get('set')) is bool
            and value.get('infinite') is False and type(value.get('number')) is int
            and value['number'] >= 0, 'invalid Slurm ' + label)
    return value['number'] if value['set'] else None


def array_indices(text):
    require(isinstance(text, str) and len(text) <= 64, 'unbounded array range')
    match = re.fullmatch(r'([0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*)((?:%[0-9]+)?)', text)
    require(match is not None and match.group(2) in ('', '%1'), 'unrecognized array range/throttle')
    result = []
    for part in match.group(1).split(','):
        ends = [int(x) for x in part.split('-')]
        start, end = ends[0], ends[-1]
        require(0 <= start <= end < 10, 'array index outside pinned plan')
        result.extend(range(start, end + 1))
    require(len(result) == len(set(result)), 'duplicate array task in range')
    return result


def classify_snapshot(snapshot, qualified, plan, systems=None):
    require(isinstance(snapshot, dict) and not snapshot.get('errors')
            and isinstance(snapshot.get('jobs'), list) and len(snapshot['jobs']) <= 2000,
            'invalid full scheduler snapshot')
    registry = original.base.canonical_registry()
    jobs, inference, benchmarks, seen_records, seen_tasks = [], [], [], set(), set()
    cells = {r['array_task_id']: r for r in qualified['cells']}
    require(set(cells) == set(range(10)), 'inference cell index differs')
    expected_command = str(Path(qualified['plan_path']).parent / 'worker.slurm')
    expected_submit = plan['submit_command']
    for raw in snapshot['jobs']:
        jid = raw.get('job_id')
        require(type(jid) is int and jid > 0 and jid not in seen_records, 'duplicate or invalid scheduler numeric ID')
        seen_records.add(jid)
        state = raw.get('job_state')
        require(isinstance(state, list) and len(state) == 1
                and state[0] in original.base.ACTIVE_STATES, 'unrecognized active scheduler state')
        reason, gres = raw.get('state_reason'), raw.get('tres_per_node')
        require(isinstance(reason, str) and isinstance(gres, str), 'missing scheduler resource/reason identity')
        parent = optional_number(raw.get('array_job_id'), 'array parent')
        task = optional_number(raw.get('array_task_id'), 'array task')
        if parent in (None, 0):
            require(task is None and raw.get('array_task_string') in (None, ''), 'ambiguous non-array scheduler row')
            if jid == SYSTEMS_ID:
                require(systems is not None and raw.get('command') == systems['command']
                        and raw.get('name') == systems['name'] and gres == systems['gres']
                        and raw.get('comment') == systems['comment']
                        and raw.get('account') == 'mltheory' and raw.get('partition') == 'lowprio'
                        and shlex.split(raw.get('submit_line', '')) == systems['submit_command'],
                        'unmapped or changed E124 systems GPU writer')
                benchmarks.append(dict(job_id=str(jid), numeric_job_id=str(jid),
                    scheduler_record_id=str(jid), array_job_id=None, array_task_id=None,
                    state=state[0], reason=reason, gres=gres, run_dir=systems['run_dir'],
                    model_choice='7b', ledger=systems['plan_path'],
                    plan_sha256=systems['plan_sha256'], writer_class='e124_systems_benchmark',
                    future_reserve_bytes=SYSTEMS_BYTES, checkpoint_credit_bytes=0))
                continue
            jobs.append(dict(job_id=str(jid), name=raw.get('name'), state=state[0], gres=gres, reason=reason))
            continue
        require(parent == ARRAY_ID and raw.get('command') == expected_command
                and raw.get('name') == 'modebench-scale-v3-dev'
                and shlex.split(raw.get('submit_line', '')) == expected_submit
                and gres == 'gres/gpu:a5000:2', 'unmapped or changed array GPU writer')
        if task is not None:
            require(0 <= task < 10 and raw.get('array_task_string') in (None, ''), 'ambiguous concrete array task')
            indices = [task]
        else:
            require(state == ['PENDING'] and jid == parent, 'unresolved running array identity')
            indices = array_indices(raw.get('array_task_string'))
        for index in indices:
            display = f'{parent}_{index}'
            require(display not in seen_tasks, 'duplicate pending/running array task')
            seen_tasks.add(display)
            cell = cells[index]
            require(cell['recommended_future_reserve_bytes'] == PER_TASK_BYTES
                    and cell['model_label'] in ('7b', '14b'), 'inference reserve or model differs')
            inference.append(dict(job_id=display, scheduler_record_id=str(jid),
                numeric_job_id=str(jid) if task is not None else None,
                array_job_id=str(parent), array_task_id=index, state=state[0], reason=reason,
                gres=gres, run_dir=str(Path(cell['outputs'][0]['output']).parent),
                model_choice='inference_' + cell['model_label'], model_label=cell['model_label'],
                model_source=cell['model_source'], cell_id=cell['cell_id'],
                ledger=qualified['plan_path'], plan_sha256=qualified['plan_sha256'],
                writer_class='offline_modebench_scale_inference',
                future_reserve_bytes=PER_TASK_BYTES, checkpoint_credit_bytes=0))
    return {'jobs': jobs, 'registry': registry, 'benchmark_writers': benchmarks}, inference


def storage_report(include_held_job_ids=()):
    report = dict(schema='e122_shared_storage_admission_v1', status='rejected', allowed=False,
        errors=[], unknown_writers=[], evaluation_writers=[], writer_profiles=[], writer_job_ids=[],
        free_bytes=None, required_bytes=None, external_peak_reserve_bytes=None,
        e122_terminal_reserve_bytes=original.E122_TERMINAL_BYTES,
        e122_peak_reserve_bytes=original.E122_PEAK_BYTES,
        shared_headroom_bytes=original.SHARED_HEADROOM_BYTES,
        blocked_reason='unresolved_storage_safety')
    try:
        qualified, plan = certificate()
        result = subprocess.run(['squeue', '--json', '--user', str(os.getuid()),
            '--states=' + ','.join(sorted(original.base.ACTIVE_STATES))],
            capture_output=True, text=True, timeout=20, check=True)
        snapshot = json.loads(result.stdout)
        training, inference = classify_snapshot(snapshot, qualified, plan, systems_certificate())
        benchmarks = training.pop('benchmark_writers')
        # Only the serialized budget call uses bounded ZIP directory reads;
        # restore the global class before any scientific authentication resumes.
        previous_zip = original.base.zipfile.ZipFile
        original.base.zipfile.ZipFile = bounded_zip.BoundedZipFile
        try:
            report = original.storage_report(include_held_job_ids=include_held_job_ids, external_snapshot=training)
        finally:
            original.base.zipfile.ZipFile = previous_zip
        report['static_evidence'] = [dict(path=str(p), sha256=original.digest(p)) for p in (
            CERTIFICATE, SYSTEMS_CERTIFICATE, Path(bounded_zip.__file__),
            ROOT/'tests/test_e122_storage_with_modebench_array_20260911.py')]

        report['benchmark_writers'] = benchmarks
        report['benchmark_future_reserve_bytes'] = sum(r['future_reserve_bytes'] for r in benchmarks)
        report['evaluation_writers'] = inference
        report['evaluation_future_reserve_bytes'] = sum(r['future_reserve_bytes'] for r in inference)
        report['storage_adapter'] = dict(path=str(Path(__file__).resolve()),
            sha256=original.digest(Path(__file__)), certificate_path=str(CERTIFICATE),
            certificate_sha256=CERTIFICATE_SHA256)
        # An unresolved original writer can never be excused by this adapter.
        if report.get('errors') or report.get('required_bytes') is None:
            return report
        extra = report['evaluation_future_reserve_bytes'] + report['benchmark_future_reserve_bytes']
        report['required_bytes'] += extra
        report['external_peak_reserve_bytes'] += extra
        report['reservations'].extend(dict(row, bytes=row['future_reserve_bytes'],
            reason='full per-task offline inference output/cache allowance; no checkpoint credit') for row in inference)
        report['reservations'].extend(dict(row, bytes=row['future_reserve_bytes'],
            reason='full original E124 systems peak allowance; no checkpoint credit') for row in benchmarks)
        report['writer_profiles'].extend(inference + benchmarks)
        report['writer_profiles'].sort(key=lambda r: r['job_id'])
        report['writer_job_ids'] = [r['job_id'] for r in report['writer_profiles']]
        require(len(report['writer_job_ids']) == len(set(report['writer_job_ids'])), 'duplicate full writer identity')
        report['writer_identity_sha256'] = hashlib.sha256(json.dumps(report['writer_profiles'],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        report['margin_bytes'] = report['free_bytes'] - report['required_bytes']
        report['allowed'] = report['free_bytes'] >= report['required_bytes'] and report['free_inodes'] >= 10000
        report['status'] = 'approved' if report['allowed'] else 'rejected'
        report['blocked_reason'] = None if report['allowed'] else ('insufficient_inodes' if report['free_inodes'] < 10000 else 'waiting_disk')
        report['policy'] += '; additionally reserve32GiB per authenticated queued inference task and full220GiB for exact E124 systems allocation'
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report.update(status='rejected', allowed=False, blocked_reason='unresolved_storage_safety')
        report.setdefault('errors', []).append(type(error).__name__ + ': ' + str(error))
    finally:
        report['observed_at'] = datetime.now(timezone.utc).isoformat()
    return report
