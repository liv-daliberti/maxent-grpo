#!/usr/bin/env python3
"""Pin the plain-GRPO terminal cells that can still be re-measured.

The paired cohort excludes plain GRPO, but the manuscript's concentration claim
covers it: "all twelve Dr.GRPO and GRPO before/after comparisons move toward
concentration". Those GRPO values carry the same eleven-stream sampling as the
paired arms, so they deserve the same correction.

Only part of that is possible. Of the 75 GRPO cells, the 55 from E95 have export
directories holding config and tokenizer files but no weight file, and no
archive receipt -- their policies were never uploaded and no longer exist
locally, which the archive's own bookkeeping already records as
``unavailable_E95_weights: 55``. Those cells cannot be re-measured by anyone and
are reported here as permanently unresampleable rather than silently dropped.
The 20 E114 Qwen-3B cells (seeds 71-74) were archived and can be restored.

Cohort records for GRPO carry no ``run_dir`` field, so the run directory is
recovered from the checkpoint origin paths the frozen record cites.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import build_terminal_cohort_manifest as base  # noqa: E402
from draw_tail import terminal_draws  # noqa: E402

SCHEMA = 'pmd-independent-resample-grpo-cohort-v1'
OUT = ROOT / 'var/artifacts/pmd_independent_resample_20260915/grpo_cohort_manifest.json'
METHOD = 'grpo'


def run_dir_of(cell: dict) -> Path:
    """Recover the run directory from the origins the frozen record cites."""
    paths = {Path(origin['path'])
             for group in cell['checkpoints'].values()
             for entry in group['origins']
             for origin in entry['origins']}
    roots = {p.parent.parent for p in paths}
    base.require(len(roots) == 1, f'ambiguous run directory for {cell}: {sorted(roots)}')
    root = roots.pop()
    return root if root.is_absolute() else ROOT / root


def build() -> dict:
    configs = base.domain_eval_config()
    with gzip.open(base.COHORT, 'rt') as handle:
        records = [json.loads(line) for line in handle]
    cohort = [r for r in records if r.get('record_kind') == 'cell'
              and r.get('method') == METHOD and r.get('level') == base.LEVEL]
    base.require(cohort, 'no plain-GRPO Level-1 cells in the cohort archive')

    cells, unresampleable = [], []
    for cell in sorted(cohort, key=lambda c: (c['scale'], c['domain'], c['seed'])):
        run_dir = run_dir_of(cell)
        receipt = run_dir / 'MODEL_ARCHIVE.json'
        complete = run_dir / 'TRAINING_COMPLETE.json'
        if not receipt.is_file():
            export = (Path(json.loads(complete.read_text())['terminal_export'])
                      if complete.is_file() else None)
            unresampleable.append({
                'scale': cell['scale'], 'domain': cell['domain'], 'seed': int(cell['seed']),
                'run_dir': str(run_dir),
                'reason': ('terminal weights absent locally and never archived; the policy '
                           'no longer exists and this cell cannot be re-measured'),
                'export_dir': str(export) if export else None,
                'export_files': sorted(p.name for p in export.iterdir()) if export and export.is_dir() else [],
            })
            continue
        draws_path = next(run_dir.glob('debug_job*/eval_mode_coverage_draws.jsonl'))
        if complete.is_file():
            attempt = Path(json.loads(complete.read_text())['terminal_attempt'])
            candidate = attempt / 'eval_mode_coverage_draws.jsonl'
            if candidate.is_file():
                draws_path = candidate
        draws = terminal_draws(draws_path, base.TERMINAL_STEP)
        config = dict(configs[cell['domain']])
        base.require(len(draws) == config['eval_mode_coverage_draws'],
                     f'{run_dir}: expected {config["eval_mode_coverage_draws"]} terminal draws')
        prompts = sorted(({'prompt_index': int(p['prompt_index']), 'problem': p['prompt'],
                           'reference': p['reference']} for p in draws[0]['prompts']),
                         key=lambda p: p['prompt_index'])
        cells.append({
            'scale': cell['scale'], 'level': base.LEVEL, 'domain': cell['domain'],
            'method': METHOD, 'seed': int(cell['seed']), 'run_dir': str(run_dir),
            'registered_job_id': None, 'terminal_step': base.TERMINAL_STEP,
            'draws_path': str(draws_path), 'gpu': None,
            'prompt_count': len(prompts),
            'prompt_set_sha256': base.sha_text(
                json.dumps(prompts, sort_keys=True, separators=(',', ':'))),
            'original_draw_seeds': [int(d['seed']) for d in draws],
            'eval_config': {**config, 'prompt_template': base.scale_template(
                cell['scale'], config['prompt_template'])},
            'weights': base.weights_record(run_dir),
            'registered_terminal_reference': (cell.get('frozen_reference_metrics') or {}).get(
                str(base.TERMINAL_STEP)),
        })
    return {
        'schema': SCHEMA, 'level': base.LEVEL, 'terminal_step': base.TERMINAL_STEP,
        'method': METHOD,
        'purpose': ('re-measure the plain-GRPO terminal cells that still have weights, so the '
                    'manuscript concentration claim is not left resting entirely on '
                    'eleven-stream sampling'),
        'sources': {'cohort': {'path': str(base.COHORT.relative_to(ROOT)),
                               'sha256': base.file_sha(base.COHORT)}},
        'builder': {'path': str(Path(__file__).resolve().relative_to(ROOT)),
                    'sha256': base.file_sha(Path(__file__).resolve())},
        'cell_count': len(cells),
        'unresampleable_count': len(unresampleable),
        'unresampleable': unresampleable,
        'cells': cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    import collections
    print(json.dumps({
        'resampleable_cells': payload['cell_count'],
        'unresampleable_cells': payload['unresampleable_count'],
        'unresampleable_by_scale': dict(collections.Counter(
            u['scale'] for u in payload['unresampleable'])),
        'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
