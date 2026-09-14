#!/usr/bin/env python3
"""Freeze sampled evaluation coverage and metrics without choosing by outcomes.

read_cell(run_dir, excluded_job_ids=()) returns all complete K=8, four-draw
checkpoints, their metric payloads and exact source lines. Conflicting duplicate
(step, draw) payloads invalidate the whole checkpoint. Reads a bounded snapshot
of each file, so concurrent appends cannot change an admitted record afterward.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_METRICS = ('any_correct_at_k', 'distinct_correct_modes_at_k')


def read_cell(run_dir: str | Path, excluded_job_ids=(), target_step: int = 3072) -> dict[str, Any]:
    root = Path(run_dir)
    excluded = {str(x) for x in excluded_job_ids}
    draws: dict[int, dict[int, dict]] = {}
    bad_steps: set[int] = set()
    files, issues = [], []
    ignored = 0
    for path in sorted(root.glob('debug_job*/eval_mode_coverage_draws.jsonl')):
        if path.parent.name.removeprefix('debug_job') in excluded:
            issues.append({'kind': 'excluded_source', 'path': str(path)})
            continue
        before = path.stat()
        digest = hashlib.sha256()
        read_bytes = 0
        line_count = 0
        with path.open('rb') as handle:
            while read_bytes < before.st_size:
                raw = handle.readline(before.st_size - read_bytes)
                if not raw:
                    break
                read_bytes += len(raw)
                line_count += 1
                digest.update(raw)
                origin = {'path': str(path), 'line': line_count}
                if not raw.endswith(b'\n'):
                    issues.append({'kind': 'incomplete_final_line', **origin})
                    continue
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    issues.append({'kind': 'invalid_json', 'error': str(exc), **origin})
                    continue
                if row.get('evaluation_kind') != 'fixed_seed_sampled_k_neutral':
                    ignored += 1
                    continue
                step, draw = row.get('step'), row.get('draw_index')
                if type(step) is not int or step < 0 or step > target_step:
                    issues.append({'kind': 'invalid_step', 'step': step, **origin})
                    continue
                metrics = row.get('metrics', {})
                valid = (row.get('sample_count') == 8 and type(draw) is int and draw in range(4)
                         and row.get('temperature') == 1.0 and isinstance(metrics, dict)
                         and all(isinstance(metrics.get(k), (int, float))
                                 and not isinstance(metrics.get(k), bool)
                                 and math.isfinite(metrics[k]) for k in REQUIRED_METRICS))
                if not valid:
                    issues.append({'kind': 'invalid_sampled_draw', 'step': step, 'draw': draw, **origin})
                    bad_steps.add(step)
                    continue
                meta = {k: row.get(k) for k in ('benchmark', 'evaluation_kind', 'sample_count', 'seed', 'temperature')}
                meta['prompt_count'] = len(row.get('prompts', []))
                payload = {'metrics': metrics, 'metadata': meta}
                signature = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
                existing = draws.setdefault(step, {}).get(draw)
                if existing is not None:
                    if existing['_signature'] != signature:
                        bad_steps.add(step)
                        issues.append({'kind': 'conflicting_duplicate', 'step': step, 'draw': draw,
                                       'first_origin': existing['origins'][0], 'other_origin': origin,
                                       'metric_payload_equal': existing['metrics'] == metrics,
                                       'other_metrics': metrics, 'other_metadata': meta})
                    else:
                        existing['origins'].append(origin)
                else:
                    draws[step][draw] = {**payload, 'origins': [origin], '_signature': signature}
        after = path.stat()
        files.append({'path': str(path), 'sha256_read_prefix': digest.hexdigest(), 'read_bytes': read_bytes,
                      'line_count': line_count, 'size_before': before.st_size, 'size_after': after.st_size,
                      'mtime_ns_before': before.st_mtime_ns, 'mtime_ns_after': after.st_mtime_ns})
    complete, incomplete = {}, {}
    for step, by_draw in sorted(draws.items()):
        if set(by_draw) != {0, 1, 2, 3} or step in bad_steps:
            incomplete[str(step)] = {'draw_indices': sorted(by_draw), 'conflicted_or_invalid': step in bad_steps}
            continue
        ordered = []
        for index in range(4):
            record = {k: v for k, v in by_draw[index].items() if k != '_signature'}
            ordered.append({'draw_index': index, **record})
        complete[str(step)] = {'step': step, 'draw_count': 4, 'draws': ordered,
                              'mean_metrics': {key: sum(x['metrics'][key] for x in ordered) / 4
                                               for key in REQUIRED_METRICS}}
    return {'run_dir': str(root), 'complete_steps': sorted(map(int, complete)),
            'complete_checkpoints': complete, 'incomplete_checkpoints': incomplete,
            'invalid_or_conflicted_steps': sorted(bad_steps), 'source_files': files,
            'issues': issues, 'ignored_non_sampled_rows': ignored}
