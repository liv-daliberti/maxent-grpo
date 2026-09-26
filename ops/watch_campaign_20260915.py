#!/usr/bin/env python3
"""One status view over the three things running for the 2026-09-15 campaign.

  resample   the PCMD re-measurement under disjoint sampling streams
  upload     the paused model archive, resumed to 715
  e95r       the plain-GRPO replication of the 55 cells whose weights were lost

Read-only: reads receipts, the archive's own status files, and the scheduler.
Prints one JSON object, so it is equally usable from a shell, a watch loop, or a
notifier. Nothing here submits, releases, or mutates anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RESAMPLE = ROOT / 'var/artifacts/pmd_independent_resample_20260915'
ARCHIVE = ROOT / 'var/artifacts/hf_model_archive_20260911/all'
UPLOAD_LOG = ROOT / 'var/artifacts/hf_archive_resume_20260915/upload.log'
E95R_LEDGER = ROOT / 'var/artifacts/e95r_plain_grpo_replication_jobs.json'
UPLOAD_FIX_UTC = '2026-09-15T17:51'


def squeue_states(names: list[str]) -> dict[str, int]:
    if not names:
        return {}
    result = subprocess.run(['squeue', '-u', 'od2961', '-h', '-o', '%j|%T'],
                            capture_output=True, text=True)
    counts: dict[str, int] = {}
    wanted = set(names)
    for line in result.stdout.splitlines():
        job, _, state = line.partition('|')
        if job in wanted:
            counts[state] = counts.get(state, 0) + 1
    return counts


def resample_status() -> dict:
    counts = {mode: len(list((RESAMPLE / 'receipts' / mode).glob('*.json.gz')))
              for mode in ('reproduce', 'independent', 'greedy')}
    result = subprocess.run(['squeue', '-u', 'od2961', '-h', '-o', '%j'],
                            capture_output=True, text=True)
    inflight = sum(1 for line in result.stdout.split() if line.startswith('pmd-'))
    paired = len(json.loads((RESAMPLE / 'cohort_manifest.json').read_text())['cells'])
    grpo_manifest = RESAMPLE / 'grpo_cohort_manifest.json'
    grpo = len(json.loads(grpo_manifest.read_text())['cells']) if grpo_manifest.is_file() else 0
    target = paired + grpo
    return {'receipts': counts, 'target_per_arm': target, 'inflight': inflight,
            'complete': counts['reproduce'] >= target and counts['independent'] >= target}


def upload_status() -> dict:
    verified = len(list(ARCHIVE.glob('models/*/verification.json')))
    retired = len(list(ARCHIVE.glob('models/*/retired.json')))
    failures = []
    if UPLOAD_LOG.is_file():
        for line in UPLOAD_LOG.read_text(errors='ignore').splitlines():
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('event') == 'stopped_model' and row.get('at_utc', '') > UPLOAD_FIX_UTC:
                failures.append(row['model'])
    running = subprocess.run(['pgrep', '-f', 'resume_model_archive.py run'],
                             capture_output=True, text=True).returncode == 0
    return {'verified': verified, 'retired_locally': retired, 'target': 715,
            'remaining': 715 - verified, 'process_running': running,
            'failures_since_cache_fix': failures}


def e95r_status() -> dict:
    if not E95R_LEDGER.is_file():
        return {'submitted': 0}
    ledger = json.loads(E95R_LEDGER.read_text())
    stamps = [r['stamp'] for r in ledger['runs']]
    states = squeue_states(stamps)
    complete = sum(1 for r in ledger['runs']
                   if (Path(r['save_path']) / 'TRAINING_COMPLETE.json').is_file())
    return {'submitted': len(stamps), 'queue_states': states, 'training_complete': complete,
            'is_recovery': ledger.get('is_recovery'), 'note': ledger.get('caveat', '')[:80]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    report = {'at_utc': datetime.now(timezone.utc).isoformat(),
              'resample': resample_status(), 'upload': upload_status(), 'e95r': e95r_status()}
    if args.compact:
        r, u, e = report['resample'], report['upload'], report['e95r']
        print(f"resample {r['receipts']['reproduce']}/{r['target_per_arm']} rep, "
              f"{r['receipts']['independent']}/{r['target_per_arm']} ind, inflight {r['inflight']} | "
              f"upload {u['verified']}/715 fail {len(u['failures_since_cache_fix'])} "
              f"{'running' if u['process_running'] else 'STOPPED'} | "
              f"e95r {e.get('training_complete', 0)}/{e.get('submitted', 0)} done "
              f"{e.get('queue_states', {})}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
