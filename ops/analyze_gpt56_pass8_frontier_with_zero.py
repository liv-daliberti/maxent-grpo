#!/usr/bin/env python3
"""Add empirical pass@8 to the authenticated, saved GPT temperature experiment.

Offline only. Each observation is an intact eight-draw prompt group, including
all-zero groups. Original accuracy, mode, collision, count, and source fields
are preserved; no model calls or regrading are performed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_frontier_temperature_ablation import DOMAINS, LEVELS, counts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/frontier_temperature_20260911/GPT56_TEMPERATURE_CURVE_WITH_ZERO.json'
OUTPUT = SOURCE.with_name('GPT56_PASS8_FRONTIER_WITH_ZERO')
TEMPERATURES = ('0.5', '1.0', '1.5', '2.0')
EXTENDED_TEMPERATURES = ('0.0', *TEMPERATURES)
GRADINGS = ('strict', 'normalized_secondary')
REPLICATES = 20000
SEED = 20260916
GROUPS = {'five_domain_macro': DOMAINS, 'four_finite_support_domain_macro': DOMAINS[1:]}
CELL_KEYS = tuple(f'level{level}/{domain}' for level in LEVELS for domain in DOMAINS)
CONTRASTS = ((1, 2), (1, 3), (2, 3))
REQUIRED_VALIDATION = ('all_3840_native_receipts_authenticated', 'same_120_prompts_all_eight_slots',
                       'only_temperature_varies_within_curve', 'all_returned_controls_match_requests',
                       'source_summaries_reconstructed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate(value, root=ROOT, cache=None):
    """Check every existing recursive path/sha256 binding, without API access."""
    if cache is None:
        cache = {}
    verified = 0
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            path = Path(value['path'])
            path = (path if path.is_absolute() else root / path).resolve()
            if path not in cache:
                cache[path] = file_sha(path)
            require(cache[path] == value['sha256'], f'Stale GPT temperature source binding: {path}')
            verified += 1
        for child in value.values():
            verified += authenticate(child, root, cache)
    elif isinstance(value, list):
        for child in value:
            verified += authenticate(child, root, cache)
    return verified


def row_identity(record):
    return record['level'], record['domain'], record['row_index']


def validate_records(records, label):
    require(isinstance(records, list) and len(records) == 120,
            f'{label}: require all 120 complete prompt groups')
    identities, digests = {}, set()
    for record in records:
        require(isinstance(record, dict), f'{label}: invalid prompt record')
        require(type(record.get('level')) is int and record['level'] in LEVELS and
                record.get('domain') in DOMAINS and type(record.get('row_index')) is int and
                record['row_index'] >= 0, f'{label}: invalid prompt row identity')
        identity = row_identity(record)
        require(identity not in identities, f'{label}: duplicate prompt row identity')
        digest = record.get('row_sha256')
        require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest),
                f'{label}: invalid prompt row digest')
        require(digest not in digests, f'{label}: duplicate prompt row digest')
        require(type(record.get('responses')) is int and record['responses'] == 8,
                f'{label}: every prompt must retain all eight responses')
        correct, distinct = record.get('correct_responses'), record.get('distinct8')
        require(type(correct) is int and 0 <= correct <= 8 and type(distinct) is int and
                0 <= distinct <= correct and (distinct > 0) == (correct > 0),
                f'{label}: invalid correct-response or distinct-mode count')
        require(record.get('correct_pairs') == correct * (correct - 1) // 2 and
                type(record.get('colliding_correct_pairs')) is int and
                0 <= record['colliding_correct_pairs'] <= record['correct_pairs'] and
                type(record.get('collision_eligible')) is bool and
                record['collision_eligible'] == (correct >= 2), f'{label}: inconsistent prompt pair counts')
        for name in ('native_refusals', 'native_content_filtered', 'truncated_responses', 'empty_answers'):
            if name in record or name != 'empty_answers':
                require(type(record.get(name)) is int and 0 <= record[name] <= 8,
                        f'{label}: invalid retained outcome count {name}')
        identities[identity] = digest
        digests.add(digest)
    by_cell = {key: sorted((r for r in records if f"level{r['level']}/{r['domain']}" == key),
                          key=row_identity) for key in CELL_KEYS}
    require(all(len(group) == 8 for group in by_cell.values()),
            f'{label}: require eight prompts in every domain/level cell')
    return identities, by_cell


def validate_counts(declared, records, label):
    expected = counts(records)
    require(isinstance(declared, dict) and all(type(declared.get(k)) is int and declared[k] == v
                                             for k, v in expected.items()),
            f'{label}: source counts differ from complete prompt groups')


def validate_block(block, label, paired_identities):
    identities, by_cell = validate_records(block['prompt_records'], label)
    require(identities == paired_identities, f'{label}: unpaired prompt identities or row digests')
    require(set(block['cells']) == set(CELL_KEYS), f'{label}: missing or extra source cells')
    validate_counts(block['counts'], block['prompt_records'], label)
    for key, records in by_cell.items():
        cell = block['cells'][key]
        validate_counts(cell['counts'], records, label + '/' + key)
        for metric, expected in [('accuracy', sum(r['correct_responses'] for r in records) / 64),
                                 ('distinct8', sum(r['distinct8'] for r in records) / 8)]:
            value = cell[metric]['estimate']
            require(type(value) in (int, float) and math.isfinite(value) and
                    abs(value - expected) < 1e-12, f'{label}/{key}: source {metric} differs from prompt counts')
        if 'empty_answers' in cell:
            require(cell['empty_answers'] == sum(r['empty_answers'] for r in records),
                    f'{label}/{key}: empty-answer source counts differ')
    return by_cell


def describe(point, boots):
    require(math.isfinite(point) and np.isfinite(boots).all(), 'Nonfinite empirical pass@8 metric')
    return {'estimate': float(point), 'ci95': np.quantile(boots, [.025, .975]).tolist(),
            'defined_replicates': len(boots)}


def pass8_bootstrap(by_cell, indices):
    """A prompt succeeds iff at least one of its eight saved draws is correct."""
    points, boots = {}, {}
    for level in LEVELS:
        for domain in DOMAINS:
            key = f'level{level}/{domain}'
            values = np.asarray([record['correct_responses'] > 0 for record in by_cell[key]], dtype=float)
            points[key] = float(values.mean())
            boots[key] = values[indices[level, domain]].mean(axis=1)
    return points, boots


def add_metrics(block, points, boots, *, contrast=False):
    require(set(block['cells']) == set(CELL_KEYS), 'Missing or extra contrast cells')
    require(set(block['groups']) == set(GROUPS), 'Missing or extra macro groups')
    limit = (-1, 1) if contrast else (0, 1)
    for key in CELL_KEYS:
        require(limit[0] <= points[key] <= limit[1], 'Empirical pass@8 outside valid bounds')
        block['cells'][key]['pass8'] = describe(points[key], boots[key])
    for name, domains in GROUPS.items():
        group = block['groups'][name]
        require(set(group['levels']) == {str(l) for l in LEVELS} and
                set(group['level_contrasts']) == {f'L{b}-L{a}' for a, b in CONTRASTS},
                'Missing or extra macro levels or level contrasts')
        selected = [key for key in CELL_KEYS if key.split('/')[1] in domains]
        group['overall']['pass8'] = describe(float(np.mean([points[k] for k in selected])),
                                            np.mean([boots[k] for k in selected], axis=0))
        level_points, level_boots = {}, {}
        for level in LEVELS:
            keys = [k for k in selected if k.startswith(f'level{level}/')]
            level_points[level] = float(np.mean([points[k] for k in keys]))
            level_boots[level] = np.mean([boots[k] for k in keys], axis=0)
            group['levels'][str(level)]['pass8'] = describe(level_points[level], level_boots[level])
        for a, b in CONTRASTS:
            group['level_contrasts'][f'L{b}-L{a}']['pass8'] = describe(
                level_points[b] - level_points[a], level_boots[b] - level_boots[a])


def difference(points, boots, target, baseline):
    return ({key: points[target][key] - points[baseline][key] for key in CELL_KEYS},
            {key: boots[target][key] - boots[baseline][key] for key in CELL_KEYS})


def empty_metric_block():
    return {'cells': {key: {} for key in CELL_KEYS}, 'groups': {
        name: {'overall': {}, 'levels': {str(level): {} for level in LEVELS},
               'level_contrasts': {f'L{b}-L{a}': {} for a, b in CONTRASTS}} for name in GROUPS}}


def report_temperatures(source):
    version = source.get('schema', '').rsplit('-v', 1)[-1]
    grid = TEMPERATURES if version == '1' else EXTENDED_TEMPERATURES
    require(source.get('temperatures') == [float(t) for t in grid],
            'Require the complete registered temperature grid for this report version')
    return grid


def analyze(source):
    """Augment a validated source report in memory, leaving every old metric intact."""
    require(source.get('schema') in ('gpt56-none-temperature-curve-v1', 'gpt56-none-temperature-curve-v2') and
            source.get('status') == 'complete' and source.get('model') == 'gpt-5.6-sol' and
            source.get('reasoning_effort') == 'none', 'Require a complete authenticated GPT reasoning=none report')
    grid = report_temperatures(source)
    extended = grid == EXTENDED_TEMPERATURES
    require(set(source['analyses']) == set(GRADINGS), 'Require strict and normalized analyses')
    required = REQUIRED_VALIDATION[1:] + (('all_native_receipts_authenticated',
        'zero_extension_gate_authenticated') if extended else REQUIRED_VALIDATION[:1])
    require(all(source.get('validation', {}).get(k) is True for k in required),
            'The complete-cohort validation is missing')
    if extended:
        require(source.get('total_registered_responses') == 4800 and
                source.get('responses_per_condition') == 960 and source.get('prompts_per_condition') == 120 and
                source.get('draws_per_prompt') == 8 and
                source['validation'].get('authenticated_native_receipt_count') == 4800,
                'The zero-extended response inventory is incomplete')
    require(source['bootstrap']['seed'] == SEED and source['bootstrap']['replicates'] == REPLICATES and
            source['bootstrap']['individual_draw_pairing'] is False,
            'Require the original paired prompt bootstrap seed and 20000 replicates')
    medium = source['matched_medium_reference']
    require(medium['model'] == 'gpt-5.6-sol' and medium['reasoning_effort'] == 'medium' and
            medium['requested_temperature'] is None and medium['returned_temperature'] == 1. and
            medium['connect_to_temperature_curve'] is False and set(medium['analyses']) == set(GRADINGS),
            'The matched medium reference must remain a separate condition')
    paired_identities, _ = validate_records(source['analyses']['strict']['temperatures']['1.0']['prompt_records'],
                                           'strict/1.0')
    result = copy.deepcopy(source)
    # This exact domain and level order reproduces the original resampling schedule.
    rng = np.random.default_rng(SEED)
    indices = {(level, domain): rng.integers(0, 8, (REPLICATES, 8))
               for level in LEVELS for domain in DOMAINS}
    for grading in GRADINGS:
        data = result['analyses'][grading]
        require(set(data['temperatures']) == set(grid) and
                set(data['paired_contrasts_vs_t1p0']) == set(grid),
                f'{grading}: incomplete temperature or paired contrast grid')
        points, boots = {}, {}
        for temperature in grid:
            block = data['temperatures'][temperature]
            by_cell = validate_block(block, grading + '/' + temperature, paired_identities)
            points[temperature], boots[temperature] = pass8_bootstrap(by_cell, indices)
            add_metrics(block, points[temperature], boots[temperature])
        for temperature in grid:
            delta = difference(points, boots, temperature, '1.0')
            add_metrics(data['paired_contrasts_vs_t1p0'][temperature], *delta, contrast=True)
        require(data['paired_endpoint_contrast']['comparison'] == f'T{grid[-1]}-T{grid[0]}',
                'Changed endpoint comparison')
        add_metrics(data['paired_endpoint_contrast'], *difference(points, boots, grid[-1], grid[0]), contrast=True)
        data['pass8_pairwise_contrasts'] = {}
        for i, baseline in enumerate(grid):
            for target in grid[i + 1:]:
                block = empty_metric_block()
                block['comparison'] = f'T{target}-T{baseline}'
                add_metrics(block, *difference(points, boots, target, baseline), contrast=True)
                data['pass8_pairwise_contrasts'][f'{target}-{baseline}'] = block
        block = result['matched_medium_reference']['analyses'][grading]
        by_cell = validate_block(block, grading + '/medium', paired_identities)
        add_metrics(block, *pass8_bootstrap(by_cell, indices))
    result['schema'] = 'gpt56-pass8-temperature-frontier-v2' if extended else 'gpt56-pass8-temperature-frontier-v1'
    result['validation']['pass8_computed_from_complete_prompt_groups'] = True
    result['metric_definitions'] = {
        'pass8': 'Empirical fraction of complete eight-draw prompt groups with at least one verified correct response: mean(correct_responses > 0).',
        'accuracy': 'Per-response correctness (pass@1), retained unchanged from the original report.',
        'distinct8': 'Mean distinct correct canonical modes per complete eight-draw prompt group, including zero-mode groups; retained unchanged.',
        'aggregation': 'Equal mean across included domain/level cells; every cell contains eight prompts.',
        'pass8_inference': 'Whole-prompt groups are resampled with the same paired, domain/level-stratified 20000-replicate schedule as the original analysis.',
        'pass8_warning': 'Pass@8 cannot be estimated as 1-(1-pass@1)^8 from aggregate accuracy; heterogeneous prompt success rates require the empirical any-correct indicator.',
        'temperature_curve': 'Segments connect measured temperatures under reasoning=none; these points do not establish a population Pareto frontier.',
        'medium_reference': 'Matched original medium-reasoning prompt groups are retained as a separate historical condition in the analysis; they are not part of the temperature sweep.'}
    return result


def build_report(path=SOURCE):
    path = Path(path).resolve()
    content = path.read_bytes()
    source = json.loads(content)
    binding_count = authenticate(source)
    require(binding_count > 0, 'The original report has no authenticated source bindings')
    result = analyze(source)
    result['analysis_sources']['original_accuracy_analysis'] = {
        'path': str(path), 'sha256': hashlib.sha256(content).hexdigest()}
    result['analysis_sources'][Path(__file__).name] = {'path': str(Path(__file__).resolve()),
                                                     'sha256': file_sha(__file__)}
    result['validation']['original_accuracy_analysis_bindings_authenticated'] = binding_count
    return result


def table_rows(result):
    grid = report_temperatures(result)
    for grading in GRADINGS:
        data = result['analyses'][grading]
        conditions = [('temperature', t, 'none', data['temperatures'][t]) for t in grid]
        conditions.append(('matched_medium_reference', '1.0', 'medium',
                           result['matched_medium_reference']['analyses'][grading]))
        conditions.append((f'paired_endpoint_T{grid[-1]}_minus_T{grid[0]}', '', 'none', data['paired_endpoint_contrast']))
        for condition, temperature, reasoning, block in conditions:
            for group_name, group in block['groups'].items():
                for level, metrics in [('all', group['overall']), *group['levels'].items()]:
                    row = {'grading': grading, 'condition': condition, 'reasoning_effort': reasoning,
                           'temperature': temperature, 'aggregation': group_name, 'level': level}
                    for name in ('pass8', 'distinct8', 'accuracy'):
                        value = metrics[name]
                        row.update({name: value['estimate'], name + '_ci95_low': value['ci95'][0],
                                    name + '_ci95_high': value['ci95'][1],
                                    name + '_defined_replicates': value['defined_replicates']})
                    yield row


def csv_text(result):
    rows = list(table_rows(result))
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def format_interval(row, metric, scale=1):
    return (f"{row[metric] * scale:.3f} [{row[metric + '_ci95_low'] * scale:.3f}, "
            f"{row[metric + '_ci95_high'] * scale:.3f}]")


def markdown(result):
    grid = report_temperatures(result)
    lines = ['# GPT-5.6 Sol: modes versus empirical pass@8 across temperatures', '',
             f"Offline reanalysis of all {len(grid) * 960:,} saved temperature-sweep responses: temperatures {', '.join(grid)}; "
             'the same 120 prompts at each temperature; eight draws per prompt; five domains '
             'and Levels 1–3. All sweep points use reasoning=none. No additional API calls or regrading.', '',
             'Pass@8 is the fraction of complete prompt groups containing at least one verified correct '
             'response. It is not per-response accuracy and is not calculated as 1-(1-pass@1)^8. '
             'Distinct8 counts distinct correct canonical modes per eight draws, retaining every zero-mode '
             'group. The original accuracy, modes, collision statistics, and source bindings are unchanged.', '',
             'Intervals are pointwise 95% percentiles from 20,000 whole-prompt bootstrap replicates '
             '(seed 20260916), paired across temperatures and stratified by domain and level. '
             'The same resampling schedule is used for the matched medium reference. Eight prompts per '
             'cell make this exploratory; temperature connections do not establish a population Pareto frontier.', '',
             'The medium reference uses the matched 120 prompts and first eight original draws (960 of '
             'the original 15,360 responses). Its requests omitted temperature, and native responses '
             'returned T=1.0. Different reasoning and collection time prevent a causal reasoning comparison; '
             'this reference remains separate from the temperature curve.', '']
    rows = list(table_rows(result))
    for grading in GRADINGS:
        for aggregation in GROUPS:
            lines += [f'## {grading}: {aggregation}', '',
                      '| Condition | Level | Pass@8 %, 95% CI | Distinct8, 95% CI | Accuracy %, 95% CI |',
                      '|---|---:|---:|---:|---:|']
            for row in rows:
                if row['grading'] != grading or row['aggregation'] != aggregation:
                    continue
                condition = ('T=' + row['temperature'] if row['condition'] == 'temperature' else
                             'Medium reference' if row['condition'] == 'matched_medium_reference' else
                             f'Paired T{grid[-1]} − T{grid[0]}')
                lines.append(f"| {condition} | {row['level']} | {format_interval(row, 'pass8', 100)} | "
                             f"{format_interval(row, 'distinct8')} | {format_interval(row, 'accuracy', 100)} |")
            lines.append('')
    lines += [f'Endpoint probability differences are percentage points. All {len(grid)} contrasts against '
              'T=1.0, all cell estimates, all level contrasts, both grading rules, source hashes, '
              'and the unchanged prompt records are included in the JSON. The CSV contains both macro '
              'aggregations, all temperatures, the medium reference, and paired endpoint contrasts.', '']
    lines += ['## Paired differences between temperatures', '',
              'Pointwise intervals do not adjust for selecting the temperature with the highest observed '
              'pass@8. The largest observed value is not proof of an optimal temperature.', '',
              '| Grading | Comparison | Overall five-domain pass@8 difference, pp [95% CI] |',
              '|---|---|---:|']
    for grading in GRADINGS:
        for name, block in result['analyses'][grading]['pass8_pairwise_contrasts'].items():
            metric = block['groups']['five_domain_macro']['overall']['pass8']
            lines.append(f"| {grading} | {name} | {100 * metric['estimate']:.3f} "
                         f"[{100 * metric['ci95'][0]:.3f}, {100 * metric['ci95'][1]:.3f}] |")
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, help='Output stem for JSON, Markdown, and CSV; a v2 source defaults to a separate WITH_ZERO report')
    args = parser.parse_args()
    result = build_report(args.source)
    grid = report_temperatures(result)
    output = args.output or (args.source.with_name('GPT56_PASS8_FRONTIER_WITH_ZERO') if grid == EXTENDED_TEMPERATURES else OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    output.with_suffix('.md').write_text(markdown(result))
    output.with_suffix('.csv').write_text(csv_text(result))
    print(json.dumps({'status': 'complete', 'api_calls': 0, 'responses': len(grid) * 960,
                      'matched_medium_responses': 960, 'output': str(output.with_suffix('.json')),
                      'normalized_pass8': {t: result['analyses']['normalized_secondary']['temperatures'][t]
                          ['groups']['five_domain_macro']['overall']['pass8'] for t in grid}}))


if __name__ == '__main__':
    main()
