#!/usr/bin/env python3
"""Freeze the E122 Level-3 Qwen-0.5B factorial terminal cohort and its PCMD.

Level 1 and Level 2 reach the manuscript through
``freeze_paper_modebench_levels.py``, which binds an already dated endpoint
audit. E122 has no such audit, so admission is decided here by the same
source-level rule the audit itself applies: a cell is admitted when its run
directory holds one complete, unconflicted four-draw sampled evaluation at the
target step. Nothing is selected by outcome, and a cell that cannot be read
that way is recorded as a gap with its exact reason rather than dropped.

PCMD travels with the endpoints because the frozen verified-sample archive that
serves Levels 1 and 2 is bound to its own census. The estimator is imported
from the same modules that built the archive's numbers, so the Level-3 values
are the identical statistic on identical inputs, read from the run directories
instead of from the archive's copy of them.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

from build_paper_modebench_level_comparison import (  # noqa: E402
    DOMAINS, METHODS, METRICS, SEEDS, TARGET_STEP,
)
from mode_diversity import DEFAULT_MIN_DEFINED_PROMPTS  # noqa: E402
from modebench_checkpoint_pmd import checkpoint_pmd, definition  # noqa: E402
from snapshot_evaluation_coverage import read_cell  # noqa: E402

LEVEL = 'level3'
BASELINE_STEP = 0
SCHEMA = 'modebench-level3-comparison-frozen-snapshot-v1'
OUTPUT = ROOT / 'paper/results/modebench_level3_comparison_snapshot.json'
LEDGER = ROOT / 'var/artifacts/e122_level3_factorial_jobs.json'
# The Python cells run under the committed CLI-runtime successors, so the
# ledger's original run directories are provenance and not where they wrote.
CLI_RECOVERY = ROOT / 'var/artifacts/python_level3_cli_recovery_20260912/committed.json'
CAMPAIGN = 'e122'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def registered_cells(ledger_path: Path = LEDGER, recovery_path: Path = CLI_RECOVERY) -> list[dict]:
    """The 100 registered cells, each pointed at the directory it wrote."""
    ledger = json.loads(ledger_path.read_text())
    successors = {}
    for replacement in json.loads(recovery_path.read_text())['replacements']:
        if replacement['campaign'] != CAMPAIGN:
            continue
        cell = replacement['cell']
        key = (cell['arm'], int(cell['seed']))
        if key in successors:
            raise RuntimeError(f'duplicate committed Python successor: {key}')
        successors[key] = cell['environment']['SAVE_PATH']
    cells = {}
    for run in ledger['runs']:
        key = (run['domain'], run['arm'], int(run['seed']))
        if key in cells:
            raise RuntimeError(f'duplicate registered Level-3 cell: {key}')
        run_dir = run['run_dir']
        if run['domain'] == 'python_factors':
            run_dir = successors[run['arm'], int(run['seed'])]
        cells[key] = {'domain': key[0], 'method': key[1], 'seed': key[2],
                      'run_dir': run_dir, 'registered_job_id': run['job_id'],
                      'registered_run_dir': run['run_dir']}
    required = {(domain, method, seed) for domain in DOMAINS
                for method in METHODS for seed in SEEDS}
    if set(cells) != required:
        raise RuntimeError('the Level-3 ledger does not enumerate the 100 registered cells')
    return [cells[key] for key in sorted(cells)]


def draw_rows(cell: dict, checkpoint: dict, step: int) -> list[dict]:
    """Copy a checkpoint's four draws verbatim; never recompute their metrics."""
    if checkpoint['draw_count'] != 4:
        raise RuntimeError('admitted Level-3 checkpoint lacks four draws')
    rows = []
    for draw in checkpoint['draws']:
        meta = draw['metadata']
        if (meta['prompt_count'], meta['sample_count'], meta['temperature']) != (128, 8, 1):
            raise RuntimeError('Level-3 draw violates the fixed 128-prompt K8 temperature1 contract')
        rows.append({
            'level': LEVEL, 'domain': cell['domain'], 'method': cell['method'],
            'seed': cell['seed'], 'step': step, 'draw_index': draw['draw_index'],
            'evaluation_kind': meta['evaluation_kind'], 'sample_count': meta['sample_count'],
            'metrics': draw['metrics'], 'evaluation_metadata': meta, 'origins': draw['origins'],
        })
    return rows


def compose(cells: list[dict], coverage: dict[tuple, dict], min_defined: int,
            sources: dict) -> dict:
    admission, terminal, pmd_cells, gaps = [], [], [], []
    baseline_evaluations, baseline_cells, baseline_pmd = [], [], []
    for cell in cells:
        key = (cell['domain'], cell['method'], cell['seed'])
        read = coverage[key]
        admitted = str(TARGET_STEP) in read['complete_checkpoints']
        record = {'level': LEVEL, 'domain': key[0], 'method': key[1], 'seed': key[2],
                  'admitted': admitted, 'endpoint': None}
        if admitted:
            rows = draw_rows(cell, read['complete_checkpoints'][str(TARGET_STEP)], TARGET_STEP)
            record['endpoint'] = {
                metric: sum(row['metrics'][field] for row in rows) / 4
                for metric, field in METRICS.items()
            }
            terminal.extend(rows)
            pmd_cells.append({'level': LEVEL, 'domain': key[0], 'method': key[1], 'seed': key[2],
                              **checkpoint_pmd(read['complete_checkpoints'][str(TARGET_STEP)],
                                               min_defined)})
        else:
            incomplete = read['incomplete_checkpoints'].get(str(TARGET_STEP))
            gaps.append({
                'level': LEVEL, 'domain': key[0], 'method': key[1], 'seed': key[2],
                'run_dir': cell['run_dir'],
                'reason': ('conflicted_or_invalid_terminal_draws' if incomplete
                           else 'no_terminal_evaluation_observed'),
                'observed_terminal_draws': incomplete,
                'latest_complete_step': max(read['complete_steps']) if read['complete_steps'] else None,
            })
        admission.append(record)
        # The untrained checkpoint is the same policy in all four arms, so it is
        # recorded per cell and averaged later rather than assigned to a method.
        baseline = read['complete_checkpoints'].get(str(BASELINE_STEP))
        baseline_cells.append({'level': LEVEL, 'domain': key[0], 'method': key[1],
                               'seed': key[2], 'admitted': baseline is not None,
                               'endpoint': None})
        if baseline is not None:
            rows = draw_rows(cell, baseline, BASELINE_STEP)
            baseline_cells[-1]['endpoint'] = {
                metric: sum(row['metrics'][field] for row in rows) / 4
                for metric, field in METRICS.items()
            }
            baseline_evaluations.extend(rows)
            baseline_pmd.append({'level': LEVEL, 'domain': key[0], 'method': key[1],
                                 'seed': key[2],
                                 **checkpoint_pmd(baseline, min_defined)})
    return {
        'schema': SCHEMA,
        'campaign': CAMPAIGN,
        'level': LEVEL,
        'model': 'Qwen2.5-0.5B-Instruct',
        'target_step': TARGET_STEP,
        'baseline_step': BASELINE_STEP,
        'collected_at_utc': now(),
        'admission_rule': (
            'A cell is admitted when its registered run directory holds one complete, '
            'unconflicted four-draw fixed-seed sampled K=8 evaluation at the target step. '
            'Admission is decided from coverage alone, before any metric is read, and a '
            'cell that fails it is recorded as a gap with its reason. The untrained step '
            'is admitted the same way and separately, so a level keeps a sound endpoint '
            'when its frozen checkpoint could not be measured.'
        ),
        'pmd_definition': definition(min_defined, digest),
        'sources': sources,
        'cells': [dict(cell) for cell in cells],
        'coverage': [{'domain': key[0], 'method': key[1], 'seed': key[2],
                      **{field: coverage[key][field] for field in (
                          'run_dir', 'complete_steps', 'invalid_or_conflicted_steps',
                          'source_files')}}
                     for key in sorted(coverage)],
        'terminal_admission': admission,
        'terminal_evaluations': terminal,
        'pmd_cells': pmd_cells,
        'baseline_admission': baseline_cells,
        'baseline_evaluations': baseline_evaluations,
        'baseline_pmd_cells': baseline_pmd,
        'unadmitted_cells': gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--ledger', type=Path, default=LEDGER)
    parser.add_argument('--recovery', type=Path, default=CLI_RECOVERY)
    parser.add_argument('--min-defined', type=int, default=DEFAULT_MIN_DEFINED_PROMPTS)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    cells = registered_cells(args.ledger, args.recovery)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        read = list(pool.map(lambda cell: read_cell(cell['run_dir']), cells))
    coverage = {(cell['domain'], cell['method'], cell['seed']): value
                for cell, value in zip(cells, read)}
    sources = {str(path.relative_to(ROOT)): digest(path.read_bytes())
               for path in (args.ledger, args.recovery)}
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__).read_bytes())
    sources[str((ROOT / 'ops/exp_scaling/snapshot_evaluation_coverage.py').relative_to(ROOT))] = digest(
        (ROOT / 'ops/exp_scaling/snapshot_evaluation_coverage.py').read_bytes())
    snapshot = compose(cells, coverage, args.min_defined, sources)
    write_json(args.output, snapshot)
    admitted = sum(row['admitted'] for row in snapshot['terminal_admission'])
    baseline = sum(row['admitted'] for row in snapshot['baseline_admission'])
    print(json.dumps({'event': 'frozen', 'output': str(args.output),
                      'registered_cells': len(cells), 'admitted_terminal_cells': admitted,
                      'admitted_baseline_cells': baseline,
                      'gaps': [f"{g['domain']}/{g['method']}/s{g['seed']}: {g['reason']}"
                               for g in snapshot['unadmitted_cells']]}, indent=2))


if __name__ == '__main__':
    main()
