#!/usr/bin/env python3
"""Report which trained policies still exist, and which are only one rm away.

On 2026-08-24 the weight files of 55 completed plain-GRPO runs were removed in a
single sub-second pass, from run directories that both retention scripts are
scoped to skip. They had never been uploaded, so those policies are gone: their
metrics survive as recorded numbers that nobody -- including us -- can ever
re-evaluate.

The failure was not the deletion itself but that nothing was watching the
invariant. Every completed run's terminal weights should be in at least one of
two places, a local export or the Hub archive, and a run in neither is a loss
that has already happened. This reports that invariant:

  archived        receipt present; weights restorable from the Hub
  both            receipt present and weights still local
  local_only      weights local, never uploaded -- one deletion from gone
  never_exported  the run declared no terminal export; nothing was ever saved
  LOST            an export was declared and its weights are gone everywhere

``local_only`` is the backlog worth uploading. ``LOST`` is damage already done
and can only be counted, not fixed. A non-zero exit means the LOST set grew
beyond the baseline the repository already acknowledges.

Filesystem discipline: scans only the known run roots at bounded depth, and
stats one export directory per run rather than walking trees.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOTS = (ROOT / 'var/data', ROOT / 'var/checkpoints')
OUT = ROOT / 'var/artifacts/checkpoint_custody_latest.json'
WEIGHT_GLOBS = ('*.safetensors', 'pytorch_model*.bin')
# Exploratory runs are worth distinguishing from analytical ones: losing a
# smoke test is not the same event as losing a cohort cell.
DEVELOPMENT_MARKERS = ('smoke', 'pilot', 'probe', 'preflight', 'diagnostic',
                       'unit_test', 'tiny', 'sentinel')
# The E95 plain-GRPO weights removed on 2026-08-24, already recorded by the
# archive as ``unavailable_E95_weights``. Growth beyond this is a new loss.
KNOWN_LOST_BASELINE = 55


def weight_files(export: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in WEIGHT_GLOBS:
        found.extend(export.glob(pattern))
    return sorted(found)


def classify(run_dir: Path) -> dict | None:
    complete = run_dir / 'TRAINING_COMPLETE.json'
    if not complete.is_file():
        return None
    try:
        receipt = json.loads(complete.read_text())
    except (json.JSONDecodeError, OSError):
        return {'run_dir': str(run_dir), 'state': 'unreadable_receipt'}
    # Some completion receipts record no terminal export at all.
    declared = receipt.get('terminal_export') or ''
    export = Path(declared) if declared else None
    archive = run_dir / 'MODEL_ARCHIVE.json'
    archived = archive.is_file()
    local = weight_files(export) if export and export.is_dir() else []
    if archived and local:
        state = 'both'
    elif archived:
        state = 'archived'
    elif local:
        state = 'local_only'
    elif not declared:
        # The run never declared a terminal export, so no policy was ever
        # saved to lose. Counting these as losses would inflate the number
        # with runs that were configured not to export in the first place.
        state = 'never_exported'
    else:
        state = 'LOST'
    name = run_dir.name
    row = {'run_dir': str(run_dir), 'state': state,
           'development': any(m in name for m in DEVELOPMENT_MARKERS),
           'terminal_step': receipt.get('terminal_step'),
           'export_declared': bool(declared),
           'export_exists': bool(export and export.is_dir()),
           'local_weight_bytes': sum(p.stat().st_size for p in local)}
    if archived:
        try:
            payload = json.loads(archive.read_text())
            row['repo_id'] = payload.get('repo_id')
            row['repo_prefix'] = payload.get('repo_prefix')
        except (json.JSONDecodeError, OSError):
            row['state'] = 'unreadable_archive_receipt'
    if state == 'LOST' and export and export.is_dir():
        row['surviving_export_files'] = sorted(p.name for p in export.iterdir())
        row['removed_at_utc'] = datetime.fromtimestamp(
            export.stat().st_mtime, timezone.utc).isoformat()
    return row


def audit(roots=RUN_ROOTS) -> dict:
    rows = []
    for root in roots:
        if not root.is_dir():
            continue
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            row = classify(run_dir)
            if row is not None:
                rows.append({**row, 'root': root.name})
    states = Counter(r['state'] for r in rows)
    at_risk = [r for r in rows if r['state'] == 'local_only']
    lost = [r for r in rows if r['state'] == 'LOST']
    lost_analytical = [r for r in lost if not r['development']]
    by_day = Counter(r.get('removed_at_utc', '')[:10] for r in lost)
    return {
        'schema': 'checkpoint-custody-audit-v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'roots': [str(r) for r in roots],
        'completed_runs': len(rows),
        'states': dict(sorted(states.items())),
        'at_risk_count': len(at_risk),
        'at_risk_bytes': sum(r['local_weight_bytes'] for r in at_risk),
        'lost_count': len(lost),
        'lost_analytical_count': len(lost_analytical),
        'lost_development_count': len(lost) - len(lost_analytical),
        'known_lost_baseline': KNOWN_LOST_BASELINE,
        'new_losses_beyond_baseline': max(0, len(lost) - KNOWN_LOST_BASELINE),
        'lost_by_removal_date': dict(sorted(by_day.items())),
        'runs': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    summary = {k: report[k] for k in
               ('completed_runs', 'states', 'at_risk_count', 'at_risk_bytes',
                'lost_count', 'lost_analytical_count', 'lost_development_count',
                'new_losses_beyond_baseline', 'lost_by_removal_date')}
    summary['at_risk_terabytes'] = round(report['at_risk_bytes'] / 1e12, 2)
    summary['output'] = str(args.output)
    if not args.quiet:
        print(json.dumps(summary, indent=2))
    raise SystemExit(1 if report['new_losses_beyond_baseline'] else 0)


if __name__ == '__main__':
    main()
