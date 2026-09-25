#!/usr/bin/env python3
"""Keep the node302-pinned E129 cells from starving E124 Level 1 on the only a100.

E129 and E129x reproduce E78 cells and pin each one to the GPU model its source cell
trained on -- placement() maps source_node node302 -> gpu:a100:1 and node105 ->
gpu:a5000:1. The a100 requests are therefore a matching constraint, not an oversight,
and 78 of them carry ReqNodeList=node302.

node302 is the only a100 in mltheory and is where E124 Level 1 runs. Both campaigns sit
at priority 8975, but an E129 cell asks 48G or 80G against an E124 cell's 160-256G, so
E129 wins every backfill gap. The cell at 2975/3072 can be locked out indefinitely by
its own sibling campaign.

  --hold     set a user hold on the node302-pinned E129 cells (recommended)
  --release  lift that hold
  --repin    move them to a5000/node105 instead -- BREAKS the source-node match, so it
             also requires --break-source-node-match

Holding costs nothing scientifically and is reversible: the cells were already queued
behind E124 on a single node, so the hold only removes their ability to take a gap.

Dry run by default. Pass --apply to act.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/n/fs/similarity/maxent-grpo')
LEDGERS = [ROOT / 'var/artifacts/e129_drgrpo_reference_kl_05b_jobs.json',
           ROOT / 'var/artifacts/e129x_reference_kl_high_beta_05b_jobs.json']
JOURNAL = ROOT / 'var/artifacts/e129_a100_gate_journal.json'


def run(argv, check=True):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if check and p.returncode != 0:
        raise RuntimeError(f'{argv[0]} failed ({p.returncode}): {p.stderr.strip()}')
    return p.stdout


def fields(job_id):
    out = run(['scontrol', 'show', 'job', str(job_id)], check=False)
    return dict(t.split('=', 1) for t in out.split() if '=' in t) if out.strip() else {}


def targets():
    """Pending E129 cells pinned to node302, with their source node from the ledger."""
    found = []
    for ledger in LEDGERS:
        if not ledger.exists():
            continue
        data = json.loads(ledger.read_text())
        for r in data.get('runs', []):
            job = r.get('job_id')
            if not job:
                continue
            f = fields(job)
            if not f or f.get('JobState') != 'PENDING':
                continue
            if f.get('ReqNodeList') != 'node302':
                continue
            found.append({'job_id': job, 'name': f.get('JobName'), 'mem': f.get('MinMemoryNode'),
                          'priority': f.get('Priority'), 'reason': f.get('Reason'),
                          'source_node': r.get('source_node'), 'ledger': ledger.name})
    return found


def journal(action, rows):
    entry = {'at': datetime.now(timezone.utc).isoformat(), 'action': action,
             'jobs': [r['job_id'] for r in rows], 'count': len(rows),
             'why': ('E124 Level 1 runs on node302, the only a100 in mltheory. E129 cells are '
                     'pinned there by source-node matching and are small enough to win every '
                     'backfill gap at equal priority.')}
    log = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {'schema': 'e129_a100_gate_v1', 'entries': []}
    log['entries'].append(entry)
    JOURNAL.write_text(json.dumps(log, indent=1) + '\n')
    print(f'\njournalled to {JOURNAL.name}')


def do_hold(rows, apply, release=False):
    verb = 'release' if release else 'hold'
    print(f'\n=== {verb} {len(rows)} node302-pinned E129 cells ===')
    acted = []
    for r in rows:
        held = r['reason'] == 'JobHeldUser'
        if release and not held:
            print(f"  {r['job_id']} {r['name']:<28} already not held")
            continue
        if not release and held:
            print(f"  {r['job_id']} {r['name']:<28} already held")
            continue
        if not apply:
            print(f"  {r['job_id']} {r['name']:<28} {r['mem']:>5}  would {verb}")
            continue
        run(['scontrol', verb, str(r['job_id'])])
        after = fields(r['job_id'])
        ok = (after.get('Priority') == '0') if not release else (after.get('Priority') != '0')
        print(f"  {r['job_id']} {r['name']:<28} {r['mem']:>5}  {verb}ed"
              f"{'' if ok else '  <-- DID NOT TAKE, CHECK'}")
        acted.append(r)
    if apply and acted:
        journal(verb, acted)
    return acted


def do_repin(rows, apply, confirmed):
    if not confirmed:
        print('\nREFUSING --repin without --break-source-node-match.')
        print('  E129 pins each cell to the GPU model of the E78 cell it reproduces.')
        print('  Re-pinning to a5000 breaks that match for every cell listed above,')
        print('  and the mismatch must then be disclosed wherever E129 is reported.')
        return []
    print(f'\n=== re-pin {len(rows)} cells to a5000/node105 (source-node match BROKEN) ===')
    acted = []
    for r in rows:
        if not apply:
            print(f"  {r['job_id']} {r['name']:<28} would move node302/a100 -> node105/a5000")
            continue
        # Try in place first; nodelist and gres are both submit-time pins.
        p = subprocess.run(['scontrol', 'update', f'jobid={r["job_id"]}',
                            'ReqNodeList=node105', 'Gres=gpu:a5000:1'],
                           capture_output=True, text=True, timeout=120)
        after = fields(r['job_id'])
        if p.returncode == 0 and after.get('ReqNodeList') == 'node105':
            print(f"  {r['job_id']} {r['name']:<28} moved in place")
            acted.append(r)
        else:
            print(f"  {r['job_id']} {r['name']:<28} IN-PLACE FAILED: {p.stderr.strip()[:70]}")
            print('         needs cancel and resubmit from the ledger SubmitLine with '
                  '--nodelist/--gres changed')
    if apply and acted:
        journal('repin_a5000', acted)
    return acted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--hold', action='store_true')
    ap.add_argument('--release', action='store_true')
    ap.add_argument('--repin', action='store_true')
    ap.add_argument('--break-source-node-match', action='store_true')
    a = ap.parse_args()
    if sum([a.hold, a.release, a.repin]) != 1:
        ap.error('pick exactly one of --hold / --release / --repin')
    if not a.apply:
        print('DRY RUN -- pass --apply to act')

    rows = targets()
    if not rows:
        print('no pending E129 cells are pinned to node302')
        return 0
    by_mem = {}
    for r in rows:
        by_mem[r['mem']] = by_mem.get(r['mem'], 0) + 1
    print(f'\n{len(rows)} pending E129 cells pinned to node302: '
          + ', '.join(f'{n} x {m}' for m, n in sorted(by_mem.items())))

    f = dict(t.split('=', 1) for t in run(['scontrol', 'show', 'node', 'node302']).split() if '=' in t)
    free = (int(f['RealMemory']) - int(f['AllocMem'])) / 1024
    print(f'node302 free right now: {free:.0f} GiB '
          f'({sum(1 for r in rows if int(r["mem"].rstrip("G")) <= free)} of them fit immediately)')

    if a.repin:
        do_repin(rows, a.apply, a.break_source_node_match)
    else:
        do_hold(rows, a.apply, release=a.release)
    return 0


if __name__ == '__main__':
    sys.exit(main())
