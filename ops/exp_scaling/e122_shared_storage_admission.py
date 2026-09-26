#!/usr/bin/env python3
"""Read-only shared-storage admission for the finite E122 six-writer burst.

Occupied files are included in statvfs usage already. Reserve one future rounded
checkpoint when this exact job ID has a structurally complete checkpoint, and
two otherwise (first save plus rolling overlap). A pending requeued job may use
its own existing checkpoint; a predecessor's checkpoint never earns this credit.
E122 always receives its full 125 GiB terminal + 6*16 GiB peak allowance, in
addition to 64 GiB shared headroom. Unknown released GPU writers fail closed.
This function neither holds nor releases anything and reserves no filesystem
space; callers must serialize release intents and recheck immediately.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import e124_storage_admission as base

ROOT = base.ROOT
GIB = base.GIB
E122_LEDGER = ROOT / 'var/artifacts/e122_level3_factorial_jobs.json'
E124_LEDGER = ROOT / 'var/artifacts/e124_qwen7b_three_level_jobs.json'
E122_TERMINAL_BYTES = 125 * GIB
E122_PEAK_BYTES = 6 * 16 * GIB
SHARED_HEADROOM_BYTES = 64 * GIB


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def storage_report(include_held_job_ids=(), external_snapshot=None):
    """JSON-safe report; include held IDs to qualify their proposed release.

    Normal calls count every non-held active GPU writer. Explicit held IDs must
    be present, canonical GPU jobs; they are charged exactly as released jobs.
    The caller must journal ambiguous releases and pass their IDs until Slurm
    positively reports them non-held or terminal. Never exclude a released job.
    Injected snapshots are supported solely for bounded verification fixtures.
    """
    result = dict(schema='e122_shared_storage_admission_v1',
        observed_at=datetime.now(timezone.utc).isoformat(), status='rejected',
        allowed=False, errors=[], unknown_writers=[], reservations=[],
        writer_profiles=[], held_gpu_job_ids=[], forced_held_job_ids=[],
        free_bytes=None, required_bytes=None, external_peak_reserve_bytes=None,
        e122_terminal_reserve_bytes=E122_TERMINAL_BYTES,
        e122_peak_reserve_bytes=E122_PEAK_BYTES,
        shared_headroom_bytes=SHARED_HEADROOM_BYTES,
        policy='current-job complete checkpoint earns one-next-checkpoint reserve, including pending; otherwise first plus overlap; full E122 cap6 peak and all100 terminals retained',
        evidence=[])
    try:
        forced = {str(x) for x in include_held_job_ids}
        if len(forced) > 100 or any(not x.isdecimal() for x in forced):
            raise ValueError('invalid forced held IDs')
        snap = external_snapshot if external_snapshot is not None else base.scheduler_snapshot()
        registry = dict(snap['registry'])
        # Named E124 canonical ledger is also admitted as an external model7B.
        if E124_LEDGER.exists():
            if E124_LEDGER.is_symlink() or E124_LEDGER.stat().st_size > 32 * 1024**2:
                raise ValueError('unsafe E124 ledger')
            e124 = json.loads(E124_LEDGER.read_text())
            rows = e124['runs']
            if not isinstance(rows, list) or len(rows) > 1000:
                raise ValueError('invalid E124 rows')
            for row in rows:
                job_id = str(row['job_id'])
                identity = dict(run_dir=row['run_dir'], model_choice=base.model_choice(row, e124.get('model_size')), ledger=str(E124_LEDGER))
                if not job_id.isdecimal() or identity['model_choice'] != '7b':
                    raise ValueError('invalid E124 identity')
                if job_id in registry and any(registry[job_id].get(k) != identity[k] for k in ('run_dir', 'model_choice')):
                    raise ValueError('conflicting E124 identity')
                registry[job_id] = identity
        e122 = json.loads(E122_LEDGER.read_text())
        if len(e122['runs']) != 100:
            raise ValueError('E122 cardinality changed')
        own = {str(row['job_id']): str(base.bounded_run_path(row['run_dir'])) for row in e122['runs']}
        if len(own) != 100 or len(set(own.values())) != 100:
            raise ValueError('duplicate E122 identity')
        jobs = snap['jobs']
        if not isinstance(jobs, list) or len(jobs) > 2000:
            raise ValueError('invalid scheduler snapshot')
        seen, writer_paths, present_forced = set(), set(), set()
        own_count = 0
        for job in jobs:
            jid = str(job['job_id'])
            if not jid.isdecimal() or jid in seen or job['state'] not in base.ACTIVE_STATES:
                raise ValueError('invalid or duplicate scheduler row')
            seen.add(jid)
            gres = job['gres']
            if not isinstance(gres, str):
                raise ValueError('missing GPU classification')
            if gres in ('N/A', '(null)', '', 'none'):
                if jid in forced:
                    raise ValueError('forced held ID is not a GPU writer')
                continue
            if 'gpu' not in gres.lower():
                raise ValueError('unknown scheduler resource classification')
            held = job['state'] == 'PENDING' and job.get('reason') in base.HELD_REASONS
            if held:
                result['held_gpu_job_ids'].append(jid)
                if jid not in forced:
                    continue
                result['forced_held_job_ids'].append(jid)
            if jid in forced:
                present_forced.add(jid)
            identity = registry.get(jid)
            if identity is None or identity.get('model_error'):
                result['unknown_writers'].append(jid)
                raise ValueError('unmapped or unresolved GPU writer: ' + jid)
            path = base.bounded_run_path(identity['run_dir'], benchmark=True)
            if str(path) in writer_paths:
                raise ValueError('multiple released writers for one run path')
            writer_paths.add(str(path))
            choice = base.model_choice(identity)
            profile = dict(job_id=jid, run_dir=str(path), model_choice=choice, gres=gres, ledger=identity['ledger'])
            result['writer_profiles'].append(profile)
            if jid in own:
                if str(path) != own[jid] or choice != '05b':
                    raise ValueError('E122 writer identity changed')
                own_count += 1
                continue
            if path.is_relative_to(ROOT / 'var/artifacts'):
                raise ValueError('untracked external benchmark writer')
            size = base.CHECKPOINT_BYTES[choice]
            checkpoint = base.existing_current_checkpoint(path, jid, size)
            copies = 1 if checkpoint else 2
            result['reservations'].append(dict(profile, state=job['state'],
                checkpoint_bytes=size, copies_reserved=copies, bytes=size*copies,
                existing_current_checkpoint=checkpoint,
                reason='next overlap' if checkpoint else 'first save plus overlap'))
        if forced - present_forced:
            raise ValueError('forced held IDs absent from active GPU queue: ' + ','.join(sorted(forced-present_forced)))
        if own_count > 6:
            raise ValueError('E122 released writers exceed finite cap6')
        stat = os.statvfs(ROOT)
        external = sum(row['bytes'] for row in result['reservations'])
        required = external + E122_TERMINAL_BYTES + E122_PEAK_BYTES + SHARED_HEADROOM_BYTES
        result.update(free_bytes=stat.f_bavail*stat.f_frsize, free_inodes=stat.f_favail,
                      external_peak_reserve_bytes=external, required_bytes=required,
                      e122_live_count=own_count)
        result['writer_profiles'].sort(key=lambda r: int(r['job_id']))
        result['writer_job_ids'] = [r['job_id'] for r in result['writer_profiles']]
        result['writer_identity_sha256'] = hashlib.sha256(json.dumps(result['writer_profiles'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        paths = [Path(__file__), Path(base.__file__), E124_LEDGER] + [ROOT/'var/artifacts'/n for n in base.LEDGERS]
        result['evidence'] = [dict(path=str(p), sha256=digest(p)) for p in paths if p.exists()]
        result['allowed'] = result['free_bytes'] >= required and stat.f_favail >= 10000
        result['status'] = 'approved' if result['allowed'] else 'rejected'
        result['blocked_reason'] = None if result['allowed'] else ('insufficient_inodes' if stat.f_favail < 10000 else 'waiting_disk')
        result['margin_bytes'] = result['free_bytes'] - required
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
        result['blocked_reason'] = 'unresolved_storage_safety'
    result['observed_at'] = datetime.now(timezone.utc).isoformat()
    return result


if __name__ == '__main__':
    print(json.dumps(storage_report(), indent=2))
