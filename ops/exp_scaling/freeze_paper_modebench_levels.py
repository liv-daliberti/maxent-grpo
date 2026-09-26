#!/usr/bin/env python3
"""Freeze all 200 Level1/Level2 cells and select matched four-draw checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from build_paper_core_terminal_endpoints import approved_run_exclusion
from build_paper_modebench_level_comparison import (
    DOMAINS, LEVELS, METHODS, SEEDS, METRICS, build_interim_comparison,
    build_terminal_comparison,
)
from snapshot_evaluation_coverage import read_cell

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'paper/results/modebench_level_comparison_snapshot.json'
SOURCES = (
    ('level1', ROOT / 'var/artifacts/e78_verified_replay_only_05b_jobs.json'),
    ('level1', ROOT / 'var/artifacts/e118_all_scales_maxrl_verified_replay_jobs.json'),
    ('level2', ROOT / 'var/artifacts/e119_level2_qwen05b_factorial_jobs.json'),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def terminal_admissions(audit_path: Path, core_path: Path) -> tuple[list[dict], dict]:
    """Anchor the terminal cohort to the shared dated audit, not live progress."""
    audit = json.loads(audit_path.read_text())
    core = json.loads(core_path.read_text())
    if audit.get('schema') != 'paper-latest-endpoint-audit-v1':
        raise RuntimeError('wrong shared endpoint audit schema')
    endpoints = {}
    for campaign, level in (('e118', 'level1'), ('e119', 'level2')):
        for row in audit['campaigns'][campaign]['rows']:
            if campaign == 'e118' and row['scale'] != 'qwen05b':
                continue
            key = (level, row['domain'], row['arm'], row['seed'])
            if key in endpoints:
                raise RuntimeError(f'duplicate endpoint audit cell: {key}')
            endpoints[key] = row['endpoint'] if row['endpoint_status'] == 'admitted' else None
    for domain, block in core['models']['Qwen2.5-0.5B']['domains'].items():
        for arm, method in (('control', 'drgrpo'), ('replay', 'replay_drgrpo')):
            values = block['methods'][arm]['per_seed']
            for seed in SEEDS:
                endpoints['level1', domain, method, seed] = values.get(str(seed))
    required = {(level, domain, method, seed) for level in LEVELS
                for domain in DOMAINS for method in METHODS for seed in SEEDS}
    if set(endpoints) != required:
        raise RuntimeError('shared endpoint audits do not enumerate the 200 comparison cells')
    admission = [{'level': key[0], 'domain': key[1], 'method': key[2], 'seed': key[3],
                  'admitted': endpoint is not None,
                  'endpoint': {metric: endpoint[metric] for metric in METRICS} if endpoint else None}
                 for key, endpoint in sorted(endpoints.items())]
    sources = {str(path.resolve()): digest(path.read_bytes()) for path in (audit_path, core_path)}
    return admission, sources


def compose(raw: dict[str, dict], paths: dict[str, Path], reference: dict,
            audit_path: Path, core_path: Path) -> tuple[dict, dict]:
    """Choose availability first, then copy exact admitted draws and provenance."""
    index, availability = {}, []
    for level, record in raw.items():
        for cell in record['cells']:
            key = (level, cell['domain'], cell['method'], int(cell['seed']))
            if key in index:
                raise RuntimeError(f'duplicate registered comparison cell: {key}')
            index[key] = cell
            availability.append({
                'level': level, 'domain': cell['domain'], 'method': cell['method'],
                'seed': int(cell['seed']),
                **{field: cell[field] for field in (
                    'complete_steps', 'invalid_or_conflicted_steps', 'source_files', 'run_dir',
                )},
            })
    required = {(level, domain, method, seed) for level in LEVELS
                for domain in DOMAINS for method in METHODS for seed in SEEDS}
    if set(index) != required:
        raise RuntimeError('comparison must enumerate exactly the 200 registered cells')
    chosen = {}
    for domain in DOMAINS:
        for seed in SEEDS:
            common = set.intersection(*(
                set(index[level, domain, method, seed]['complete_steps'])
                for level in LEVELS for method in METHODS
            ))
            if common:
                chosen[domain, seed] = max(common)

    def normalized(level, domain, method, seed, step):
        checkpoint = index[level, domain, method, seed]['complete_checkpoints'][str(step)]
        if checkpoint['draw_count'] != 4:
            raise RuntimeError('selected checkpoint lacks four draws')
        rows = []
        for draw in checkpoint['draws']:
            meta = draw['metadata']
            if (meta['prompt_count'], meta['sample_count'], meta['temperature']) != (128, 8, 1):
                raise RuntimeError('selected draw violates fixed 128-prompt K8 temperature1 contract')
            rows.append({
                'level': level, 'domain': domain, 'method': method, 'seed': seed,
                'step': step, 'draw_index': draw['draw_index'],
                'evaluation_kind': meta['evaluation_kind'], 'sample_count': meta['sample_count'],
                'metrics': draw['metrics'], 'evaluation_metadata': meta, 'origins': draw['origins'],
            })
        return rows

    evaluations = [row for (domain, seed), step in chosen.items()
                   for level in LEVELS for method in METHODS
                   for row in normalized(level, domain, method, seed, step)]
    admission, terminal_sources = terminal_admissions(audit_path, core_path)
    terminal = []
    for cell in admission:
        if not cell['admitted']:
            continue
        key = tuple(cell[field] for field in ('level', 'domain', 'method', 'seed'))
        if 3072 not in index[key]['complete_steps']:
            raise RuntimeError(f'admitted terminal endpoint absent from frozen draws: {key}')
        rows = normalized(*key, 3072)
        for metric, field in METRICS.items():
            mean = sum(row['metrics'][field] for row in rows) / 4
            if mean != cell['endpoint'][metric]:
                raise RuntimeError(f'frozen draws disagree with shared endpoint audit: {key}, {metric}')
        terminal.extend(rows)
    terminal_comparison = build_terminal_comparison(terminal, admission)
    comparison = build_interim_comparison(evaluations, availability)
    snapshot = {
        'schema': 'modebench-level-comparison-frozen-snapshot-v1',
        'collected_at_utc': now(),
        'selection_policy': (
            'Latest observed shared nonnegative checkpoint for every domain/seed across all8 series; '
            'actual valid step0 allowed based on coverage before inspecting effects. All5domains must '
            'have observations. No missing-value imputation, no substitution of admission estimates, '
            'no retry conflict selection.'
        ),
        'input_snapshots': {level: {'path': str(path.relative_to(ROOT)),
                                  'sha256': digest(path.read_bytes())}
                            for level, path in paths.items()},
        'collection_intervals': {level: {'start': record['started_at_utc'],
                                         'end': record['finished_at_utc']}
                                 for level, record in raw.items()},
        'reference_figure': reference, 'availability': availability,
        'evaluations': evaluations,
        'terminal_evaluations': terminal, 'terminal_admission': admission,
        'terminal_sources': terminal_sources,
        'terminal_selection_policy': terminal_comparison['selection_rule'],
        'level2_terminal_evaluations': [row for row in terminal if row['level'] == 'level2'],
    }
    return snapshot, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--endpoint-audit', type=Path, required=True)
    parser.add_argument('--core-endpoints', type=Path, default=ROOT / 'paper/results/core_terminal_endpoints.json')
    parser.add_argument('--reuse-coverage', action='store_true', help='Rebuild from the exact coverage snapshots already in the audit directory')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    audit = args.audit_directory.resolve()
    audit.relative_to(ROOT)
    audit.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    previous = output.read_bytes()
    reference = json.loads(previous)['reference_figure']
    before = audit / 'before_modebench_level_comparison_snapshot.json'
    if not before.exists():
        before.write_bytes(previous)
    paths = {level: audit / f'{level}_coverage_snapshot.json' for level in LEVELS}
    if args.reuse_coverage:
        raw = {level: json.loads(path.read_text()) for level, path in paths.items()}
    else:
        sources, tasks = {}, []
        for level, path in SOURCES:
            frozen = path.read_bytes()
            sources[str(path.relative_to(ROOT))] = digest(frozen)
            (audit / path.name).write_bytes(frozen)
            for run in json.loads(frozen)['runs']:
                if 'e118' in path.name and run['scale'] != 'qwen05b':
                    continue
                if approved_run_exclusion(Path(run['run_dir'])):
                    raise RuntimeError('unexpected excluded Qwen-0.5B comparison source')
                method = {'control': 'drgrpo', 'replay': 'replay_drgrpo'}.get(run['arm'], run['arm'])
                tasks.append((level, path, run, method))
        started = now()
        raw = {level: {
            'schema': 'modebench-evaluation-coverage-v1', 'started_at_utc': started,
            'sources': sources, 'reader_sha256': digest((ROOT / 'ops/exp_scaling/snapshot_evaluation_coverage.py').read_bytes()),
            'collector_sha256': digest(Path(__file__).read_bytes()), 'cells': [],
        } for level in LEVELS}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(read_cell, run['run_dir']): (level, path, run, method)
                       for level, path, run, method in tasks}
            for count, future in enumerate(as_completed(futures), start=1):
                level, path, run, method = futures[future]
                cell = future.result()
                cell.update(level=level, domain=run['domain'], method=method, seed=int(run['seed']),
                            registered_job_id=run['job_id'], ledger=str(path.relative_to(ROOT)))
                raw[level]['cells'].append(cell)
                if count % 10 == 0:
                    print(f'Frozen {count}/{len(tasks)} comparison cells', flush=True)
        for level in LEVELS:
            raw[level]['finished_at_utc'] = now()
            raw[level]['cells'].sort(key=lambda row: (row['domain'], row['method'], row['seed']))
            write_json(paths[level], raw[level])
    snapshot, comparison = compose(raw, paths, reference, args.endpoint_audit, args.core_endpoints)
    write_json(output, snapshot)
    summary = {key: comparison[key] for key in (
        'coverage_by_domain', 'eligible_domain_seed_cells', 'initial_only_domains', 'means', 'domain_means',
    )}
    summary['level2_terminal_cells'] = len(snapshot['level2_terminal_evaluations']) // 4
    summary['terminal_comparison'] = build_terminal_comparison(
        snapshot['terminal_evaluations'], snapshot['terminal_admission'])
    write_json(audit / 'selection_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
