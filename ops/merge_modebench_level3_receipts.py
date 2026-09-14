#!/usr/bin/env python3
"""Merge immutable ModeBench receipts for disjoint sampling seeds on identical rows.

Inputs must use the exact same source, checkpoint, frozen interface, decoding
budget, and evaluator/verifier code. Each stored draw and aggregate is checked
against its attempts; verification itself is not rerun. Seed draws are pooled
with equal draw weight, never by averaging unequal-sized receipt aggregates.
Both development and explicitly authorized confirmation receipts are supported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_modebench_level3 import DOMAINS, SCHEMA, atomic_new, file_sha, frozen_interface, sha, summarize
from audit_modebench_level3_match import METRICS, close, require, valid_sha

MERGE_SCHEMA = 'modebench-level3-disjoint-seed-merge-v1'


def validate_input(receipt: dict[str, Any]) -> None:
    require(receipt.get('schema') == SCHEMA and receipt.get('status') == 'complete',
            'a complete frozen-calibration receipt is required')
    domain, split = receipt.get('domain'), receipt.get('split')
    require(domain in DOMAINS and split in ('dev', 'eval'), 'invalid domain or split')
    require(receipt.get('model_label') in ('05b', '3b'), 'invalid model label')
    require(receipt.get('level') in ('level1', 'level2', 'level3'), 'invalid level')
    identity = receipt.get('identity', {})
    require(receipt.get('identity_sha256') == sha(identity), 'receipt identity hash mismatch')
    for field in ('schema', 'domain', 'split', 'level'):
        require(identity.get(field) == receipt.get(field), f'inconsistent identity {field}')
    require(identity.get('model', {}).get('label') == receipt['model_label'], 'inconsistent model label')
    require(bool(identity.get('code_sha256')), 'missing evaluator/verifier code identity')
    interface = identity.get('interface', {})
    require(interface == frozen_interface(domain, interface.get('name')), 'input interface is not frozen')
    require(identity.get('interface_sha256') == sha(interface), 'interface hash mismatch')
    seeds = identity.get('seeds', [])
    require(isinstance(seeds, list) and seeds and all(type(seed) is int and seed >= 0 for seed in seeds)
            and len(seeds) == len(set(seeds)), 'invalid or repeated seed')
    require(receipt.get('sampling') == {**interface, 'seeds': seeds}, 'inconsistent sampling settings')
    boundary = receipt.get('information_boundary', {})
    require(boundary.get('evaluation_prompts_loaded') is (split == 'eval'), 'inconsistent evaluation boundary')
    require(boundary.get('treatment_training_started') is False, 'receipt does not describe an untrained model')
    if split == 'eval':
        require(boundary.get('confirmation_explicitly_authorized') is True, 'confirmation was not explicitly authorized')
    source, rows = identity.get('source', {}), receipt.get('prompt_results', [])
    require(isinstance(rows, list) and len(rows) > 0, 'missing prompt results')
    require(source.get('selected_rows') == len(rows), 'source row count mismatch')
    require(valid_sha(source.get('rows_sha256')) and valid_sha(source.get('all_rows_sha256')),
            'missing source hashes')
    offset = source.get('row_offset', 0)
    require(type(offset) is int and offset >= 0, 'invalid row offset')
    require([row.get('row_index') for row in rows] == list(range(offset, offset + len(rows))),
            'missing, reordered, or repeated prompt indices')
    require(len({row.get('row_sha256') for row in rows}) == len(rows), 'duplicate source rows')
    if split == 'eval':
        require(len(rows) == 128 and source.get('total_rows') == 128 and offset == 0
                and source.get('row_limit', 0) == 0, 'confirmation requires the full 128-row split')
    for index, row in enumerate(rows):
        context = f'row {index}'
        require(all(valid_sha(row.get(field)) for field in ('row_sha256', 'problem_sha256', 'spec_sha256')),
                f'{context}: invalid row/spec/prompt hash')
        require(isinstance(row.get('row_metadata'), dict), f'{context}: missing row metadata')
        draws = row.get('draws', [])
        require([draw.get('seed') for draw in draws] == seeds, f'{context}: missing or repeated draw')
        values = []
        for draw in draws:
            attempts = draw.get('attempts', [])
            require(len(attempts) == 8, f'{context}: expected eight samples in each draw')
            for attempt in attempts:
                require(type(attempt.get('verified')) is bool and
                        attempt['verified'] == (attempt.get('canonical_key') is not None),
                        f'{context}: verification/canonical-key disagreement')
                require(isinstance(attempt.get('text'), str), f'{context}: missing completion text')
                require(type(attempt.get('token_count')) is int and
                        0 <= attempt['token_count'] <= interface['max_tokens'], f'{context}: token budget violation')
            correct = sum(attempt['verified'] for attempt in attempts)
            require(type(draw.get('verified_count')) is int and draw['verified_count'] == correct,
                    f'{context}: verified sample count mismatch')
            expected = {'pass1': correct / 8, 'pass8': float(correct > 0),
                        'distinct8': len({sha(attempt['canonical_key']) for attempt in attempts if attempt['verified']})}
            for metric, value in expected.items():
                close(draw.get(metric), value, f'{context}/seed {draw["seed"]}/{metric}')
            values.append(expected)
        for metric in METRICS:
            close(row.get(metric), statistics.mean(value[metric] for value in values), f'{context}/{metric}')
    summary = summarize(rows)
    require(receipt.get('metrics', {}).get('rows') == len(rows), 'summary row count mismatch')
    for metric, value in summary.items():
        if metric in METRICS or metric in receipt.get('metrics', {}):
            if value is None:
                require(receipt['metrics'].get(metric) is None, f'{metric}: standard error mismatch')
            else:
                close(receipt['metrics'].get(metric), value, f'summary/{metric}')
    histogram = dict(Counter(str(row['row_metadata'].get('answer_mode_count')) for row in rows))
    require(receipt.get('answer_mode_histogram') == histogram, 'support histogram differs from source metadata')


def comparable_identity(identity: dict[str, Any]) -> dict[str, Any]:
    # Merging earlier merges is safe when their underlying run identities agree;
    # their separate provenance trees remain fully recorded in the new receipt.
    return {key: value for key, value in identity.items() if key not in ('seeds', 'aggregation')}


def row_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in ('draws', *METRICS)}


def merge_receipts(paths: list[Path], output: Path | None = None) -> dict[str, Any]:
    require(len(paths) >= 2, 'at least two input receipts are required')
    if output is not None and output.exists():
        raise FileExistsError(f'fresh merged receipt required: {output}')
    loaded, provenance, all_seeds = [], [], set()
    for path in paths:
        path = Path(path).resolve()
        raw = path.read_bytes()
        import hashlib
        digest = hashlib.sha256(raw).hexdigest()
        receipt = json.loads(raw)
        validate_input(receipt)
        seeds = receipt['identity']['seeds']
        require(not all_seeds.intersection(seeds), f'overlapping sampling seeds: {sorted(all_seeds.intersection(seeds))}')
        all_seeds.update(seeds)
        if loaded:
            first = loaded[0]
            require(comparable_identity(receipt['identity']) == comparable_identity(first['identity']),
                    'input source/model/interface/settings/code identities differ')
            require(receipt['information_boundary'] == first['information_boundary'], 'information boundaries differ')
            require(len(receipt['prompt_results']) == len(first['prompt_results']), 'source row counts differ')
            require([row_identity(row) for row in receipt['prompt_results']] ==
                    [row_identity(row) for row in first['prompt_results']], 'source rows/specs/metadata differ')
            require(receipt.get('metric_definitions') == first.get('metric_definitions'), 'metric definitions differ')
        provenance.append({'path': str(path), 'sha256': digest, 'identity_sha256': receipt['identity_sha256'],
                           'seeds': list(seeds), 'aggregation': receipt.get('aggregation')})
        loaded.append(receipt)
    ordered_seeds = sorted(all_seeds)
    merged = deepcopy(loaded[0])
    for index, row in enumerate(merged['prompt_results']):
        draws = {draw['seed']: deepcopy(draw) for receipt in loaded for draw in receipt['prompt_results'][index]['draws']}
        row['draws'] = [draws[seed] for seed in ordered_seeds]
        for metric in METRICS:
            row[metric] = statistics.mean(draw[metric] for draw in row['draws'])
    aggregation = {'schema': MERGE_SCHEMA, 'input_receipts': provenance,
                   'merger_source_sha256': file_sha(Path(__file__)),
                   'seed_order': ordered_seeds, 'draws_per_prompt': len(ordered_seeds),
                   'aggregation_unit': 'all seed draws for each original prompt, with equal weight per draw',
                   'metrics_recomputed_from_attempts': True, 'verification_outcomes_regraded': False}
    merged['identity'] = {**comparable_identity(merged['identity']), 'seeds': ordered_seeds, 'aggregation': aggregation}
    merged['identity_sha256'] = sha(merged['identity'])
    merged['sampling'] = {**merged['identity']['interface'], 'seeds': ordered_seeds}
    merged['metrics'] = summarize(merged['prompt_results'])
    merged['aggregation'] = aggregation
    merged['generated_at'] = datetime.now(timezone.utc).isoformat()
    validate_input(merged)
    if output is not None:
        atomic_new(output, merged)
    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = merge_receipts(args.receipts, args.output)
    print(json.dumps({'output': str(args.output), 'domain': result['domain'], 'level': result['level'],
                      'model_label': result['model_label'], 'seeds': result['sampling']['seeds'],
                      'metrics': result['metrics']}, sort_keys=True))


if __name__ == '__main__':
    main()
