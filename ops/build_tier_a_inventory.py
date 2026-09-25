#!/usr/bin/env python3
"""Inventory the registered treatment-arm runs that are still only on local disk.

The 2026-09-11 archive covered 715 models. The custody audit found 853 completed
runs whose weights exist in exactly one place, and 671 of those sit outside that
plan. This selects the part of that backlog that belongs in a public archive:
runs claimed by a cohort registered in ``ops/exp_scaling/cohorts.py`` whose kind
is an actual treatment arm.

Cohorts of kind ``repair`` are deliberately excluded. Those are mechanism gates
and preflights -- ``E111 verified-support discovery gate``, ``E117-R1
same-plumbing mechanism preflight`` -- carrying ``family=None, arm=None``. They
are registered so campaign tracking can see them, not because they report a
result, and no manuscript claim rests on their policies. Excluding them from the
archive does not delete them; they stay on disk and keep appearing in the
custody audit as ``local_only``.

Emits the record shape ``archive_completed_models.prepare`` consumes. Every
field is derived from the run's own completion receipt and export, so a record
that cannot be built honestly is reported rather than guessed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops' / 'exp_scaling'))
import cohorts as registry  # noqa: E402

DATA_ROOT = ROOT / 'var/data'
# The 2026-09-11 plan is being archived separately; a run in both plans would be
# uploaded twice and its local weights retired by whichever pass reached it first.
PLAN_715 = ROOT / 'var/artifacts/hf_model_archive_20260911/all/plan_completed_delta715.json'
# The Tier-A pass ran with keep_local, so its 141 models carry no MODEL_ARCHIVE.json
# and reappear here unless the plan that published them is excluded too. Reading
# only the 715 plan re-emits them, which would re-upload 226 GB already on the Hub
# and collide on their repo prefixes when the combined plan is prepared.
PLAN_COMBINED = ROOT / 'var/artifacts/hf_model_archive_20260911/all/plan_combined_856_20260915.json'
PUBLISHED_PLANS = (PLAN_715, PLAN_COMBINED)
OUT = ROOT / 'var/artifacts/hf_tier_a_20260915/inventory.json'
ADMITTED_KINDS = ('semantic', 'replay_dose', 'paired')
# Some studies are registered as several cohorts, one per model scale, so their
# tags carry a scale suffix the archive does not use: the published plan records
# 'e115', never 'e115_05b'. archive_expanded_registry keys its public study text
# the same way, and prepare_expanded_archive_plan rejects a source_experiment it
# cannot find there. Map the split tags back to the study they belong to; any tag
# that is already a study key passes through unchanged.
STUDY_KEYS = {'e95_3b': 'e95', 'e95_1b': 'e95', 'e95_05b': 'e95', 'e95r_05b': 'e95r',
              'e115_05b': 'e115', 'e115_3b': 'e115', 'e116_05b': 'e116', 'e116_3b': 'e116'}
# Run directories spell the half-billion scale three ways; all are the same model.
# E95-R was launched from the current source tree, which names the directory after
# the pretrained snapshot rather than the campaign's own short scale token, so its
# runs read xdr_qwen_qwen2.5-0.5b-instruct_. Verified against the export itself:
# Qwen2ForCausalLM, hidden 896, 24 layers, 14 heads, vocab 151936.
MODEL_KEYS = {'qwen25_0p5b_instruct': 'qwen05b', 'qwen25_05b_instruct': 'qwen05b',
              'qwen_qwen2.5-0.5b-instruct': 'qwen05b',
              'qwen25_3b_instruct': 'qwen3b', 'falcon3_1b_instruct': 'falcon1b'}
WEIGHTS = ('*.safetensors', 'pytorch_model*.bin')


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_key_of(run_dir: Path) -> str | None:
    name = run_dir.name
    for prefix, key in MODEL_KEYS.items():
        if name.startswith(f'xdr_{prefix}_'):
            return key
    return None


def cohort_index() -> dict[str, registry.Cohort]:
    index: dict[str, registry.Cohort] = {}
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
                index[str(run_dir)] = (cohort, run)
    return index


def already_planned() -> set[str]:
    planned: set[str] = set()
    for path in PUBLISHED_PLANS:
        if path.is_file():
            planned |= {m['terminal_export'] for m in json.loads(path.read_text())['models']}
    return planned


def build() -> dict:
    index = cohort_index()
    planned = already_planned()
    records, skipped = [], []
    for run_dir_text, (cohort, run) in sorted(index.items()):
        if cohort.kind not in ADMITTED_KINDS:
            continue
        run_dir = Path(run_dir_text)
        receipt_path = run_dir / 'TRAINING_COMPLETE.json'
        if (run_dir / 'MODEL_ARCHIVE.json').is_file() or not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        export = Path(receipt.get('terminal_export') or '')
        if not export.is_dir():
            continue
        weights = sorted(p for pattern in WEIGHTS for p in export.glob(pattern))
        if not weights:
            continue
        if str(export) in planned:
            continue  # covered by the 2026-09-11 plan
        reason = None
        model_key = model_key_of(run_dir)
        if model_key is None:
            reason = f'unrecognised model prefix in {run_dir.name}'
        elif DATA_ROOT not in run_dir.parents:
            reason = 'run directory is outside var/data'
        elif int(receipt.get('terminal_step', 0)) < 3072:
            reason = f'terminal step {receipt.get("terminal_step")} below 3072'
        if reason:
            skipped.append({'run_dir': run_dir_text, 'cohort': cohort.tag, 'reason': reason})
            continue
        job_ids = [str(run['job_id'])] if run.get('job_id') else []
        attempt = Path(receipt['terminal_attempt']).name
        hit = re.match(r'debug_job(\d+)$', attempt)
        if hit and hit.group(1) not in job_ids:
            job_ids.append(hit.group(1))
        if not job_ids:
            skipped.append({'run_dir': run_dir_text, 'cohort': cohort.tag,
                            'reason': 'no scheduler identity to check for live consumers'})
            continue
        study = STUDY_KEYS.get(cohort.tag, cohort.tag)
        records.append({
            'audited_candidate': True, 'endpoint_status': 'admitted',
            'source_experiment': study, 'cohort_label': cohort.label,
            # Cross-list key rendered onto the public model card, so a reader can
            # see which campaign a model belongs to without the run directory.
            'campaign': study,
            'cohort_kind': cohort.kind, 'model_key': model_key,
            'scale': model_key, 'domain': run.get('domain'),
            'source_arm': run.get('arm'), 'arm': run.get('arm'),
            'seed': int(run['seed']) if run.get('seed') is not None else None,
            'run_dir': run_dir_text, 'run_stamp': run.get('run_stamp') or run_dir.name,
            'receipt_path': str(receipt_path), 'receipt_present': True,
            'receipt_sha256': digest(receipt_path),
            'terminal_export': str(export), 'terminal_step': int(receipt['terminal_step']),
            'terminal_bytes': sum(p.stat().st_size for p in export.iterdir() if p.is_file()),
            'terminal_files': sorted(p.name for p in export.iterdir() if p.is_file()),
            'large_weight_files': [p.name for p in weights],
            'all_related_job_ids': job_ids,
        })
    # Pin every ledger that contributed a record. The engine re-checks these
    # before each upload and before each deletion, so a cohort ledger edited
    # mid-run aborts the pass instead of silently changing what is in scope --
    # which is exactly how the 2026-09-11 archive stopped.
    pins = {}
    for cohort in registry.REGISTRY:
        if cohort.kind not in ADMITTED_KINDS:
            continue
        path = cohort.path()
        study = STUDY_KEYS.get(cohort.tag, cohort.tag)
        if path.is_file() and any(r['source_experiment'] == study for r in records):
            pins[str(path)] = digest(path)
    return {
        'schema': 'hf-tier-a-inventory-v1',
        'ledger_pins': pins,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection': ('local-only completed runs claimed by a registered cohort whose kind is '
                      'a treatment arm; repair-kind cohorts are excluded, not deleted'),
        'admitted_kinds': list(ADMITTED_KINDS),
        'excludes_published_plans': [str(p.relative_to(ROOT)) for p in PUBLISHED_PLANS if p.is_file()],
        'records': records, 'record_count': len(records),
        'total_bytes': sum(r['terminal_bytes'] for r in records),
        'by_cohort': dict(sorted(Counter(r['source_experiment'] for r in records).items())),
        'skipped': skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'records': payload['record_count'],
                      'terabytes': round(payload['total_bytes'] / 1e12, 3),
                      'by_cohort': payload['by_cohort'],
                      'skipped': len(payload['skipped']),
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
