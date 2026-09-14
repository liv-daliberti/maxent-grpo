#!/usr/bin/env python3
"""Read-only shared-filesystem admission for one additional E124 7B writer.

This report does not reserve space. The caller must serialize E124 release
intents, refresh this report immediately before release, and preserve ambiguous
intents as live runs. Other campaigns' released pending jobs are reserved even
if Slurm forecasts a late start. No metrics, outcomes, or completion markers
are read. Occupied checkpoints are already reflected in statvfs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(os.environ.get('OAT_ZERO_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
GIB = 1024 ** 3
NEW_RUN_PEAK_BYTES = 220 * GIB
SHARED_HEADROOM_BYTES = 64 * GIB
MAX_OWN_LIVE = 1
# Rounded up from serialized model+Adam state. The 7B value is an estimate:
# 15,231,233,024 BF16 parameter bytes * 7 = 99.297 GiB, not a measurement.
CHECKPOINT_BYTES = {'05b': 7 * GIB, '3b': 41 * GIB, '7b': 100 * GIB}
LEDGERS = (
    'e118_all_scales_maxrl_verified_replay_jobs.json',
    'e119_level2_qwen05b_factorial_jobs.json',
    'e119_level2_continuation_jobs.json',
    'e120r1_frequency_weighted_replay_jobs.json',
    'e120r1_scheduler_continuation_jobs.json',
    'e122_level3_factorial_jobs.json',
    'e123_level3_factorial_jobs.json',
)
ACTIVE_STATES = {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'PENDING'}
HELD_REASONS = {'JobHeldUser', 'JobHeldAdmin', 'job requeued in held state'}


def bounded_run_path(value, *, benchmark=False):
    """Reject lexical/symlink escape and a different mounted filesystem."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if '..' in path.parts:
        raise ValueError('run path contains parent traversal')
    path = path.resolve()
    base = ROOT / 'var/data'
    benchmark_root = ROOT / 'var/artifacts/e124_qwen7b_three_level/systems'
    benchmark_allowed = benchmark and (path == benchmark_root or path.is_relative_to(benchmark_root))
    if not benchmark_allowed and (path == base or not path.is_relative_to(base)):
        raise ValueError('run path escapes approved science/benchmark roots')
    existing = path
    while not existing.exists():
        existing = existing.parent
    if existing.stat().st_dev != ROOT.stat().st_dev:
        raise ValueError('run path is on a different filesystem')
    return path


def model_choice(row, default=None):
    names = [row.get(k) for k in ('model_choice', 'model_size', 'model_key', 'scale', 'model')]
    names.append(default)
    found = set()
    for value in names:
        if not isinstance(value, str):
            continue
        text = value.lower().replace('_', '-').replace('0p5', '0.5')
        if text in ('05b', '0.5b', 'qwen05b') or re.search(r'qwen(?:2\.5)?-?0\.5b', text):
            found.add('05b')
        elif text in ('3b', 'qwen3b') or re.search(r'qwen(?:2\.5)?-?3b', text):
            found.add('3b')
        elif text in ('7b', 'qwen7b') or re.search(r'qwen(?:2\.5)?-?7b', text):
            found.add('7b')
        elif value not in ('custom', ''):
            raise ValueError('unknown model size: ' + value)
    if not found:
        raise ValueError('missing model size')
    if len(found) != 1:
        raise ValueError('conflicting model size')
    return found.pop()


def canonical_registry():
    """Read exact identity fields only from bounded, named campaign ledgers."""
    registry = {}
    records = []
    by_path = {}
    for name in LEDGERS:
        path = ROOT / 'var/artifacts' / name
        if not path.exists():
            continue
        if path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('unsafe or oversized canonical ledger: ' + name)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError('malformed canonical ledger: ' + name)
        for key in ('runs', 'continuations'):
            rows = payload.get(key, [])
            if not isinstance(rows, list) or len(rows) > 1000:
                raise ValueError('malformed or oversized canonical rows: ' + name)
            for row in rows:
                if not isinstance(row, dict) or not row.get('run_dir'):
                    raise ValueError('canonical row lacks run identity: ' + name)
                identity = {'run_dir': row['run_dir'], 'ledger': str(path)}
                try:
                    identity['model_choice'] = model_choice(row, payload.get('model_choice') or payload.get('model'))
                except ValueError as exc:
                    # Unsupported historical models block only if actually queued.
                    identity['model_error'] = str(exc)
                if 'model_choice' in identity:
                    previous_choice = by_path.get(row['run_dir'])
                    if previous_choice is not None and previous_choice != identity['model_choice']:
                        raise ValueError('conflicting model identity for canonical run')
                    by_path[row['run_dir']] = identity['model_choice']
                for field in ('job_id', 'continuation_job_id', 'original_job_id'):
                    job_id = row.get(field)
                    if job_id is None:
                        continue
                    job_id = str(job_id)
                    if not job_id.isdecimal():
                        raise ValueError('invalid canonical job ID')
                    records.append((job_id, identity.copy()))
    for job_id, identity in records:
        if identity.get('model_error') == 'missing model size' and identity['run_dir'] in by_path:
            identity.pop('model_error')
            identity['model_choice'] = by_path[identity['run_dir']]
        previous = registry.get(job_id)
        if previous is not None and any(previous.get(k) != identity.get(k) for k in ('run_dir', 'model_choice', 'model_error')):
            raise ValueError('conflicting canonical identity: ' + job_id)
        registry[job_id] = identity
    return registry


def scheduler_snapshot():
    completed = subprocess.run(
        ['squeue', '--noheader', '--user', str(os.getuid()),
         '--states=' + ','.join(sorted(ACTIVE_STATES)),
         '--format=%i|%j|%T|%b|%r'],
        capture_output=True, text=True, timeout=20, check=True,
    )
    jobs = []
    for line in completed.stdout.splitlines():
        fields = line.split('|')
        if len(fields) != 5 or not fields[0].isdecimal():
            raise ValueError('malformed scheduler writer row')
        jobs.append(dict(zip(('job_id', 'name', 'state', 'gres', 'reason'), fields)))
    return {'jobs': jobs, 'registry': canonical_registry()}


def existing_current_checkpoint(run_dir, job_id, checkpoint_bytes):
    """Inspect small ZIP directories and sizes, never deserialize tensors.

    A complete current-job checkpoint allows a one-next-checkpoint reserve;
    otherwise reserve two writes, including the first rolling overlap. Older
    job directories do not qualify: their files may remain during recovery.
    """
    cp_root = run_dir / ('debug_job' + job_id) / 'checkpoints'
    if not cp_root.exists():
        return None
    if cp_root.is_symlink() or not cp_root.resolve().is_relative_to(run_dir):
        raise ValueError('checkpoint root escapes run')
    steps = sorted(cp_root.iterdir(), reverse=True)
    if len(steps) > 64:
        raise ValueError('unbounded checkpoint inventory')
    for step in steps:
        if not re.fullmatch(r'step_\d+', step.name):
            continue
        if step.is_symlink() or not step.is_dir():
            raise ValueError('unsafe checkpoint step')
        files = list(step.rglob('*.pt'))
        if len(files) > 32:
            raise ValueError('unbounded checkpoint shards')
        if not any(p.name.endswith('model_states.pt') for p in files) or not any(p.name.endswith('optim_states.pt') for p in files):
            continue
        total = 0
        complete = True
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(run_dir):
                raise ValueError('checkpoint shard escapes run')
            total += path.stat().st_size
            try:
                with zipfile.ZipFile(path) as archive:
                    members = archive.infolist()
                    if not any(m.filename.endswith('/data.pkl') for m in members):
                        complete = False
            except (OSError, zipfile.BadZipFile):
                complete = False
        # Serialized BF16+FP32 Adam is within the rounded-up allocation.
        # Reject partial, tiny archives and sizes above the known bound.
        if total > checkpoint_bytes:
            raise ValueError('observed checkpoint exceeds known model reservation')
        if complete and total >= checkpoint_bytes * 0.9:
            return {'path': str(step), 'bytes': total, 'evidence': 'ZIP directories and size only; no outcomes inspected'}
    return None


def storage_report(plan, own_live_runs=(), external_snapshot=None):
    """Return JSON-safe allowed/free/required/reservations; never mutate state.

    plan has ``runs`` or ``cells`` with run_dir and model_choice/model metadata.
    own_live_runs contains exact job IDs or such rows, including release intents
    whose scheduler acknowledgement is unknown. Injected snapshots may provide
    resolved run_dir/model_choice on jobs or a registry keyed by job ID.
    """
    report = {'schema': 'e124_storage_admission_v1',
              'observed_at': datetime.now(timezone.utc).isoformat(), 'allowed': False,
              'free_bytes': None, 'required_bytes': None, 'reservations': [], 'errors': [],
              'new_7b_peak_bytes': NEW_RUN_PEAK_BYTES, 'shared_headroom_bytes': SHARED_HEADROOM_BYTES,
              'max_own_live_runs': MAX_OWN_LIVE, 'own_live_count': 0,
              'policy': 'occupied bytes already used; reserve next checkpoint for proven current checkpoint, otherwise first plus overlap; include released pending writers',
              'measurement': '7B 100GiB checkpoint / 220GiB total peak is conservative estimate; 3B 41GiB and 0.5B 7GiB round measured checkpoint sizes upward'}
    try:
        if Path(plan.get('root', ROOT)).resolve() != ROOT:
            raise ValueError('plan root differs from configured repository root')
        rows = plan.get('runs') or plan.get('cells')
        if not isinstance(rows, list) or not rows or len(rows) > 1000:
            raise ValueError('plan needs bounded nonempty runs or cells')
        benchmark_rows = plan.get('benchmark_cells', [])
        if not isinstance(benchmark_rows, list) or len(benchmark_rows) > 20:
            raise ValueError('invalid benchmark cells')
        rows = rows + benchmark_rows
        own_paths = set()
        plan_ids = {}
        for row in rows:
            path = str(bounded_run_path(row['run_dir'], benchmark=row in benchmark_rows))
            if path in own_paths:
                raise ValueError('duplicate plan run directory')
            own_paths.add(path)
            if model_choice(row, plan.get('model_choice') or plan.get('model')) != '7b':
                raise ValueError('E124 plan contains a non-7B model')
            if row.get('job_id') is not None:
                job_id = str(row['job_id'])
                if not job_id.isdecimal() or job_id in plan_ids:
                    raise ValueError('duplicate or invalid plan job ID')
                plan_ids[job_id] = path
        own = set()
        for row in own_live_runs:
            if isinstance(row, dict):
                path = str(bounded_run_path(row['run_dir'], benchmark=True))
                if path not in own_paths:
                    raise ValueError('own live run is outside plan')
                own.add(path)
                if row.get('job_id') is not None:
                    job_id = str(row['job_id'])
                    if not job_id.isdecimal() or (job_id in plan_ids and plan_ids[job_id] != path):
                        raise ValueError('invalid or conflicting own live job ID')
                    plan_ids[job_id] = path
            else:
                if str(row) not in plan_ids:
                    raise ValueError('own live job ID is outside plan')
                own.add(plan_ids[str(row)])
        snapshot = external_snapshot if external_snapshot is not None else scheduler_snapshot()
        jobs = snapshot['jobs']
        if not isinstance(jobs, list) or len(jobs) > 2000:
            raise ValueError('unbounded or malformed scheduler snapshot')
        registry = snapshot.get('registry', {})
        seen_ids = set()
        writer_paths = {}
        for job in jobs:
            job_id = str(job['job_id'])
            if not job_id.isdecimal() or job_id in seen_ids:
                raise ValueError('duplicate or malformed scheduler job ID')
            seen_ids.add(job_id)
            if job['state'] not in ACTIVE_STATES:
                raise ValueError('unknown scheduler writer state: ' + job['state'])
            gres = job.get('gres')
            if not isinstance(gres, str):
                raise ValueError('missing scheduler GPU resource evidence')
            if gres in ('N/A', '(null)', '', 'none'):
                continue
            if 'gpu' not in gres.lower():
                raise ValueError('unknown scheduler resource classification')
            if job['state'] == 'PENDING' and job.get('reason') in HELD_REASONS:
                continue
            identity = ({'run_dir': plan_ids[job_id], 'model_choice': '7b'} if job_id in plan_ids
                        else registry.get(job_id) or job)
            if identity.get('model_error'):
                raise ValueError('unresolved external model ' + job_id + ': ' + identity['model_error'])
            if not identity.get('run_dir'):
                raise ValueError('unmapped external GPU writer: ' + job_id)
            path = bounded_run_path(identity['run_dir'], benchmark=True)
            if str(path) in writer_paths:
                raise ValueError('multiple released writers for one run path')
            writer_paths[str(path)] = job_id
            choice = model_choice(identity)
            if str(path) in own_paths:
                if choice != '7b':
                    raise ValueError('own scheduler model differs from plan')
                own.add(str(path))
                continue
            if path.is_relative_to(ROOT / 'var/artifacts'):
                raise ValueError('untracked external benchmark writer')
            size = CHECKPOINT_BYTES[choice]
            evidence = existing_current_checkpoint(path, job_id, size) if job['state'] != 'PENDING' else None
            copies = 1 if evidence else 2
            report['reservations'].append({'job_id': job_id, 'state': job['state'],
                'run_dir': str(path), 'model_choice': choice, 'checkpoint_bytes': size,
                'copies_reserved': copies, 'bytes': size * copies,
                'existing_current_checkpoint': evidence,
                'reason': 'next overlap' if evidence else 'initial save plus overlap; includes released pending jobs'})
        report['own_live_count'] = len(own)
        own_reserve = len(own) * NEW_RUN_PEAK_BYTES
        external_reserve = sum(r['bytes'] for r in report['reservations'])
        stat = os.statvfs(ROOT)
        report.update(free_bytes=stat.f_bavail * stat.f_frsize,
                      free_inodes=stat.f_favail, device=ROOT.stat().st_dev,
                      own_live_reserve_bytes=own_reserve, external_reserve_bytes=external_reserve,
                      required_bytes=NEW_RUN_PEAK_BYTES + SHARED_HEADROOM_BYTES + own_reserve + external_reserve)
        report['allowed'] = len(own) < MAX_OWN_LIVE and report['free_bytes'] >= report['required_bytes'] and stat.f_favail >= 10000
        report['blocked_reason'] = ('own_concurrency_cap' if len(own) >= MAX_OWN_LIVE else
                                    'insufficient_inodes' if stat.f_favail < 10000 else
                                    'waiting_disk' if report['free_bytes'] < report['required_bytes'] else None)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
        report['blocked_reason'] = 'unresolved_storage_safety'
    return report
