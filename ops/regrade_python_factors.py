#!/usr/bin/env python3
"""Re-grade the stored PythonFactors responses under the corrected size caps.

The verifier rejected any candidate whose AST exceeded 64 nodes with "candidate
AST is too large". The prompt never states a size limit - it tells the model it
may use conditional expressions - so answers written inside the stated grammar
were discarded for exceeding an undisclosed bound. On one cell that removed 72%
of well-formed lambdas, many of them correct on every case.

Every receipt retains each attempt's raw text, so this re-grades saved
responses. Nothing is resampled: the models are not run again, and the draw
order, seeds and text are untouched. Only `verified` and `canonical_key` can
change, and only from false to true, since the caps were loosened.

Originals are copied aside before anything is written.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in ('ops', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
import evaluate_modebench_base_grid as grid
from oat_drgrpo.math_grader import validated_modebench_outcome_key as grader

sha = grid.sha
DOMAIN = 'python_factors'


def regrade_receipt(path: Path, rows_cache: dict) -> dict | None:
    receipt = json.loads(path.read_text())
    if receipt.get('domain') != DOMAIN or receipt.get('status') != 'complete':
        return None
    if 'regrade' in receipt:
        return None          # already corrected; makes the sweep resumable
    rows_jsonl = receipt['identity']['dataset_binding']['rows_jsonl']
    if rows_jsonl not in rows_cache:
        rows_cache[rows_jsonl] = grid.load_rows(
            {'rows_jsonl': rows_jsonl, 'row_offset': 0, 'row_limit': 0})[0]
    rows = rows_cache[rows_jsonl]

    before = receipt['metrics']['pass8']
    changed = 0
    for result in receipt['prompt_results']:
        answer = rows[result['row_index']]['answer']
        for draw in result['draws']:
            for attempt in draw['attempts']:
                key = grader(attempt['text'], answer)
                verified = key is not None
                if verified != attempt['verified']:
                    changed += 1
                attempt['verified'] = verified
                attempt['canonical_key'] = key
            verified_count = sum(a['verified'] for a in draw['attempts'])
            distinct = len({sha(a['canonical_key']) for a in draw['attempts'] if a['verified']})
            draw['verified_count'] = verified_count
            draw['pass1'] = verified_count / len(draw['attempts'])
            draw['pass8'] = float(verified_count > 0)
            draw['distinct8'] = distinct
        for metric in ('pass1', 'pass8', 'distinct8'):
            result[metric] = statistics.fmean(d[metric] for d in result['draws'])

    results = receipt['prompt_results']
    for metric in ('pass1', 'pass8', 'distinct8'):
        values = [r[metric] for r in results]
        receipt['metrics'][metric] = statistics.fmean(values)
        receipt['metrics'][metric + '_standard_error_across_prompts'] = (
            statistics.stdev(values) / len(values) ** .5 if len(values) > 1 else None)
    # `identity` describes the sampling run - model, seeds, prompts and the code
    # that produced the draws - and re-grading changed none of that. Leaving it
    # untouched keeps identity_sha256 valid and keeps the receipt bound to its
    # retained batch files. The grader that reinterpreted the text is recorded
    # separately, where it belongs.
    receipt['regrade'] = {
        'reason': 'python_factor size caps raised; saved responses re-graded, not resampled',
        'attempts_changed': changed,
        'pass8_before': before,
        'pass8_after': receipt['metrics']['pass8'],
        'grader_code_sha256': grid.frozen.code_identity(),
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, nargs='+', required=True,
                        help='receipt directories to sweep')
    parser.add_argument('--backup-suffix', default='_pre_python_ast_fix')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    rows_cache: dict = {}
    report = []
    for directory in args.receipts:
        directory = Path(directory).resolve()
        paths = sorted(Path(directory).glob(f'*_{DOMAIN}.json'))
        if not paths:
            continue
        backup = Path(directory).parent / (Path(directory).name + args.backup_suffix)
        for path in paths:
            updated = regrade_receipt(path, rows_cache)
            if updated is None:
                continue
            row = {'receipt': str(path.resolve().relative_to(ROOT)), **updated['regrade']}
            row['delta'] = row['pass8_after'] - row['pass8_before']
            report.append(row)
            if args.dry_run:
                continue
            backup.mkdir(parents=True, exist_ok=True)
            if not (backup / path.name).exists():
                shutil.copy2(path, backup / path.name)
            path.write_text(json.dumps(updated, indent=1, sort_keys=True,
                                       allow_nan=False) + '\n')
    report.sort(key=lambda r: -r['delta'])
    for row in report:
        print(f"{Path(row['receipt']).stem:36s} pass8 {row['pass8_before']:.3f} ->"
              f" {row['pass8_after']:.3f} ({row['delta']:+.3f})  attempts changed {row['attempts_changed']}")
    moved = [r for r in report if r['delta'] > 0]
    print(f"\ncells: {len(report)}  changed: {len(moved)}  "
          f"mean delta on changed: {statistics.fmean(r['delta'] for r in moved) if moved else 0:+.3f}")
    print('DRY RUN - nothing written' if args.dry_run else 'receipts rewritten; originals copied aside')


if __name__ == '__main__':
    main()
