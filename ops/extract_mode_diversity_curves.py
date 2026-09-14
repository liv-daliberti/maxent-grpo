#!/usr/bin/env python3
"""Extract per-step verified key streams for PMD training curves, once.

The saved training evaluations keep every response body, so the full set of
source files is roughly 22 GiB on shared storage. The curves only need each
sampled response's canonical key and whether it verified, which is a tiny
fraction of that. This walks the frozen source files a single time and writes a
compact gzip container so no later figure has to touch the raw run directories
again.

Reading is deliberately modest: the source list is fixed and enumerated from
the frozen archive rather than discovered by traversal, each file is streamed
once line by line, and worker count stays small because the cost here is shared
filesystem bandwidth rather than local CPU.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))

ARCHIVE = ROOT / 'paper/audits/conditional_concentration_20260912/verified_samples_completed_cohort.jsonl.gz'
OUT = ROOT / 'var/artifacts/mode_diversity_curves/verified_keys_by_step.jsonl.gz'
SAMPLED = 'fixed_seed_sampled_k_neutral'


def cell_index(archive: Path) -> dict[str, dict]:
    """Map each source file to the scientific cell that registered it."""
    index: dict[str, dict] = {}
    with gzip.open(archive, 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('record_kind') != 'cell':
                continue
            meta = {k: record[k] for k in ('level', 'scale', 'domain', 'method', 'seed')}
            meta['run_dir'] = record.get('run_dir')
            for check in record.get('source_checks') or []:
                path = check.get('path')
                if path:
                    index.setdefault(path, meta)
    return index


def extract_file(path: str) -> list[dict]:
    """Return one record per (step, draw) holding only verified canonical keys."""
    out = []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('evaluation_kind') != SAMPLED or row.get('step') is None:
                continue
            prompts = {}
            for prompt in row.get('prompts') or []:
                keys, rewards = prompt.get('answer_keys'), prompt.get('rewards')
                if not isinstance(keys, list) or not isinstance(rewards, list) or len(keys) != len(rewards):
                    continue
                # A key counts only when its own sample earned a verification reward.
                prompts[str(prompt['prompt_index'])] = [
                    key if (reward > 0 and key is not None) else None
                    for key, reward in zip(keys, rewards)]
            out.append({'step': row['step'], 'draw_index': row.get('draw_index'),
                        'seed': row.get('seed'), 'prompts': prompts})
    return out


def _worker(path: str) -> tuple[str, list[dict] | None, str | None]:
    try:
        return path, extract_file(path), None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return path, None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=6,
                        help='kept small on purpose; the bottleneck is shared storage')
    args = parser.parse_args()

    index = cell_index(args.archive)
    paths = sorted(index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with gzip.open(args.output, 'wt') as out:
        out.write(json.dumps({'record_kind': 'manifest',
                              'schema': 'paper-mode-diversity-curve-keys-v1',
                              'source_files': len(paths)}) + '\n')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done, (path, rows, error) in enumerate(pool.map(_worker, paths, chunksize=1), 1):
                if error is not None:
                    stats['failed'] += 1
                    print(json.dumps({'event': 'source_failed', 'path': path, 'reason': error}), flush=True)
                    continue
                meta = index[path]
                out.write(json.dumps({'record_kind': 'source', 'path': path,
                                      **meta, 'draws': rows}) + '\n')
                stats['files'] += 1
                stats['draws'] += len(rows)
                if done % 25 == 0:
                    print(json.dumps({'event': 'progress', 'done': done, 'of': len(paths), **stats}), flush=True)
    print(json.dumps({'event': 'complete', 'output': str(args.output), **stats}), flush=True)


if __name__ == '__main__':
    main()
