#!/usr/bin/env python3
"""Read-only E124 admission adapter for the authenticated ModeBench array.

Keep the immutable E124 one-run/220-GiB admission implementation and use the
reviewed E122 JSON scheduler classifier. Add every inference task's full
32-GiB reserve, plus the full E122 terminal/peak reserve. This intentionally
retains E122's ordinary checkpoint charges too, a conservative overestimate.
No controller is patched or started by this module. Deployment must serialize
all release decisions under the shared admission lock and existing E124 lock.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e122_storage_with_modebench_array_20260911 as arrays
import modebench_array_throttle2_storage_20260911 as throttle

base = arrays.original.base
ROOT = base.ROOT
ADAPTER_SHA256 = 'ffe7063c1a9e729ad4b0dc744e2dfff791c4bae0bc8ec1f5b1de37eca98cbbd0'
ZIP_SHA256 = 'fa42ec7308592e44963b8084240d8235dc80a776876b020a709f764f1aa0a899'
THROTTLE_SHA256 = 'c75832e7f1cdfd17119015f98a3889894d544791788b3e1f75ca766df6b83ecc'


def pins():
    for path, expected in ((Path(arrays.__file__), ADAPTER_SHA256),
                           (Path(arrays.bounded_zip.__file__), ZIP_SHA256),
                           (Path(throttle.__file__), THROTTLE_SHA256)):
        arrays.require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                       'reviewed array/ZIP adapter changed')
    # These verify the original admission sources, exact array and benchmark
    # identities, and their complete frozen source inventories.
    qualified, array_plan = arrays.certificate()
    return qualified, array_plan, arrays.systems_certificate()


def storage_report(plan, own_live_runs=()):
    """Return a read-only diagnostic using the original E124 function contract."""
    report = dict(schema='e124_storage_admission_v1', allowed=False, errors=[],
                  blocked_reason='unresolved_storage_safety', free_bytes=None,
                  required_bytes=None, reservations=[], own_live_count=0)
    try:
        qualified, array_plan, systems = pins()
        queried = subprocess.run(['squeue', '--json', '--user', str(os.getuid()),
            '--states=' + ','.join(sorted(base.ACTIVE_STATES))],
            capture_output=True, text=True, timeout=20, check=True)
        with throttle.compatible_array_parser():
            snapshot, inference = arrays.classify_snapshot(
                json.loads(queried.stdout), qualified, array_plan, systems)
        benchmarks = snapshot.pop('benchmark_writers')
        # Restore the exact systems row to the original E124 budget so that it
        # consumes the unchanged own-concurrency slot and complete 220-GiB peak.
        for row in benchmarks:
            identity = {'run_dir': row['run_dir'], 'model_choice': '7b'}
            previous = snapshot['registry'].get(row['job_id'])
            arrays.require(previous is None or all(previous.get(k) == v for k, v in identity.items()),
                           'conflicting E124 systems registry identity')
            snapshot['registry'][row['job_id']] = identity
            snapshot['jobs'].append({k: row[k] for k in ('job_id', 'state', 'gres', 'reason')})
        previous_zip = base.zipfile.ZipFile
        base.zipfile.ZipFile = arrays.bounded_zip.BoundedZipFile
        try:
            report = base.storage_report(plan, own_live_runs=own_live_runs,
                                         external_snapshot=snapshot)
        finally:
            base.zipfile.ZipFile = previous_zip
        report['compatibility_adapter'] = str(Path(__file__).resolve())
        report['evaluation_writers'] = inference
        report['evaluation_future_reserve_bytes'] = sum(r['future_reserve_bytes'] for r in inference)
        report['e122_terminal_reserve_bytes'] = arrays.original.E122_TERMINAL_BYTES
        report['e122_peak_reserve_bytes'] = arrays.original.E122_PEAK_BYTES
        if report.get('errors') or report.get('required_bytes') is None:
            return report
        # Never relax an original gate or subtract an existing reservation.
        extra = (report['evaluation_future_reserve_bytes']
                 + report['e122_terminal_reserve_bytes'] + report['e122_peak_reserve_bytes'])
        report['original_required_bytes'] = report['required_bytes']
        report['additional_shared_reserve_bytes'] = extra
        report['required_bytes'] += extra
        report['reservations'].extend(dict(row, bytes=row['future_reserve_bytes'],
            reason='full authenticated inference allowance; no checkpoint credit') for row in inference)
        report['margin_bytes'] = report['free_bytes'] - report['required_bytes']
        if report['allowed'] and report['margin_bytes'] < 0:
            report.update(allowed=False, blocked_reason='waiting_disk')
        report['policy'] += '; full E122125GiB terminal +96GiB peak and32GiB per authenticated inference task; original E122 checkpoint charges retained conservatively'
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report.update(allowed=False, blocked_reason='unresolved_storage_safety')
        report.setdefault('errors', []).append(type(error).__name__ + ': ' + str(error))
    finally:
        report['observed_at'] = datetime.now(timezone.utc).isoformat()
    return report


def controller_storage_report(plan, tx, *, releasing=None):
    """Exact original live-intent selection; no transaction or plan mutation."""
    own = []
    for row in [*plan['benchmark_cells'], *plan['cells']]:
        item = tx['rows'].get(row['cell_id'], {})
        if row['cell_id'] != releasing and item.get('status') in (
                'released', 'release_intent', 'requeue_intent', 'held_retry'):
            own.append(row | {'job_id': item['job_id']})
    return storage_report(plan, own_live_runs=own)


if __name__ == '__main__':
    # Diagnostic only. A later deployment must use fresh values while holding
    # the shared and E124 admission locks; this command cannot authorize release.
    art = ROOT / 'var/artifacts/e124_qwen7b_three_level'
    print(json.dumps(controller_storage_report(
        json.loads((art / 'plan.json').read_text()),
        json.loads((art / 'transaction.json').read_text())), indent=2))
