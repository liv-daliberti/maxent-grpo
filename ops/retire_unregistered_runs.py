#!/usr/bin/env python3
"""Delete weight files of completed runs that no cohort, plan or paper claims.

On 2026-08-24 the weights of 55 completed plain-GRPO runs were removed in a
sub-second pass that left no record. All that survived was a directory mtime,
so the loss could only be characterised months later by inference. This does the
same kind of work deliberately: it writes a manifest of exactly what it will
remove, refuses anything that is claimed by a cohort ledger, an archive plan,
the paper tree, or a live scheduler job, and removes only weight files.

What is kept: the completion receipt, training metrics, evaluation outputs,
config and tokenizer. Those are small and they are what lets a later reader know
the run happened and what it measured. What is destroyed is the policy itself,
and unlike the archive's retirement there is no remote copy -- these runs were
never uploaded, which is precisely why they were judged not worth publishing.

Dry run by default. ``--apply`` deletes, and only after the manifest is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops' / 'exp_scaling'))
import cohorts as registry  # noqa: E402

RUN_ROOTS = (ROOT / 'var/data', ROOT / 'var/checkpoints')
PLANS = (ROOT / 'var/artifacts/hf_model_archive_20260911/all/plan_combined_856_20260915.json',
         ROOT / 'var/artifacts/hf_model_archive_20260911/all/plan_completed_delta715.json')
PAPER_REFS = ROOT / 'var/artifacts/tier_b_retirement_20260915/paper_referenced_run_dirs.txt'
OUT_DIR = ROOT / 'var/artifacts/tier_b_retirement_20260915'
WEIGHTS = ('*.safetensors', 'pytorch_model*.bin')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def claimed_by_cohort() -> set[str]:
    claimed = set()
    for cohort in registry.REGISTRY:
        path = cohort.path()
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for run in (payload.get('runs') or []):
            run_dir = run.get('run_dir') or run.get('save_path')
            if run_dir:
                claimed.add(str(run_dir))
    return claimed


def claimed_by_plan() -> set[str]:
    claimed = set()
    for plan_path in PLANS:
        if not plan_path.is_file():
            continue
        for model in json.loads(plan_path.read_text())['models']:
            claimed.add(str(model['terminal_export']))
            claimed.add(str(model['run_dir']))
    return claimed


def referenced_by_paper() -> set[str]:
    """Run directories cited anywhere under paper/, precomputed by --scan-paper."""
    if not PAPER_REFS.is_file():
        raise SystemExit(f'missing paper reference scan: {PAPER_REFS}; run --scan-paper first')
    return {line.strip() for line in PAPER_REFS.read_text().splitlines() if line.strip()}


def scan_paper() -> int:
    """One bounded pass over paper/ recording every run directory it mentions."""
    result = subprocess.run(
        ['grep', '-rhoE', r'/n/fs/similarity/maxent-grpo/var/(data|checkpoints)/[A-Za-z0-9_.-]+',
         str(ROOT / 'paper')], capture_output=True, text=True)
    refs = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    PAPER_REFS.parent.mkdir(parents=True, exist_ok=True)
    PAPER_REFS.write_text('\n'.join(refs) + '\n')
    return len(refs)


def live_scheduler_text() -> str:
    result = subprocess.run(['squeue', '-u', 'od2961', '-h', '-o', '%i|%j|%Z'],
                            capture_output=True, text=True)
    return result.stdout


def candidates() -> tuple[list[dict], list[dict]]:
    cohort_claims, plan_claims = claimed_by_cohort(), claimed_by_plan()
    paper_claims = referenced_by_paper()
    queue = live_scheduler_text()
    chosen, held = [], []
    for root in RUN_ROOTS:
        if not root.is_dir():
            continue
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            complete = run_dir / 'TRAINING_COMPLETE.json'
            if not complete.is_file():
                continue
            try:
                receipt = json.loads(complete.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            export = Path(receipt.get('terminal_export') or '')
            if not export.is_dir():
                continue
            files = sorted(p for pattern in WEIGHTS for p in export.glob(pattern))
            if not files:
                continue
            row = {'run_dir': str(run_dir), 'terminal_export': str(export),
                   'terminal_step': receipt.get('terminal_step'),
                   'weight_files': [{'name': p.name, 'bytes': p.stat().st_size,
                                     'mtime_utc': datetime.fromtimestamp(
                                         p.stat().st_mtime, timezone.utc).isoformat()}
                                    for p in files],
                   'bytes': sum(p.stat().st_size for p in files)}
            reasons = []
            if str(run_dir) in cohort_claims:
                reasons.append('claimed by a registered cohort ledger')
            if str(run_dir) in plan_claims or str(export) in plan_claims:
                reasons.append('claimed by an archive plan')
            if str(run_dir) in paper_claims:
                reasons.append('referenced under paper/')
            if (run_dir / 'MODEL_ARCHIVE.json').is_file():
                reasons.append('already archived to the Hub')
            if run_dir.name in queue:
                reasons.append('a live scheduler job references this run')
            if reasons:
                held.append({**row, 'held_because': reasons})
            else:
                chosen.append(row)
    return chosen, held


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-paper', action='store_true',
                        help='refresh the record of run directories cited under paper/')
    parser.add_argument('--apply', action='store_true', help='actually delete')
    args = parser.parse_args()
    if args.scan_paper:
        print(json.dumps({'paper_referenced_run_dirs': scan_paper(),
                          'written': str(PAPER_REFS)}, indent=2))
        return
    chosen, held = candidates()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': 'unregistered-run-weight-deletion-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'applied': False,
        'retained_per_run': ['TRAINING_COMPLETE.json', 'train_metrics.jsonl',
                             'eval_results/', 'eval_mode_coverage_draws.jsonl',
                             'config.json', 'tokenizer files'],
        'irreversible': ('these runs were never uploaded, so deletion is permanent and no '
                         'restore path exists'),
        'digests': ('file sizes and mtimes are recorded; content is not hashed, because reading '
                    'every weight would put hundreds of gigabytes of avoidable load on shared '
                    'storage and a digest of a deleted file cannot restore it'),
        'selected_count': len(chosen), 'selected_bytes': sum(r['bytes'] for r in chosen),
        'held_count': len(held), 'held_bytes': sum(r['bytes'] for r in held),
        'selected': chosen, 'held': held,
    }
    path = OUT_DIR / ('deletion_manifest.json' if args.apply else 'deletion_manifest_dryrun.json')
    removed = 0
    if args.apply:
        require(chosen, 'nothing selected')
        manifest['applied'] = True
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        for row in chosen:
            export = Path(row['terminal_export'])
            for item in row['weight_files']:
                target = export / item['name']
                if target.is_file():
                    target.unlink()
                    removed += item['bytes']
        manifest['removed_bytes'] = removed
        manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'applied': args.apply, 'selected_runs': len(chosen),
                      'selected_terabytes': round(manifest['selected_bytes'] / 1e12, 3),
                      'held_runs': len(held),
                      'removed_terabytes': round(removed / 1e12, 3) if args.apply else 0,
                      'manifest': str(path)}, indent=2))


if __name__ == '__main__':
    main()
