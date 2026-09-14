#!/usr/bin/env python3
"""Authenticate the additive 480-prompt GPT temperature sweep, entirely offline.

The old 120-prompt measurements remain immutable. New measurements contribute
24 additional prompts per domain/level, with eight intact draws at each of five
temperatures. Cohort sensitivity uses the same frozen grading and bootstrap.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / '.git').exists())
EXPERIMENT = ROOT / 'artifacts/frontier_temperature_20260911'
FROZEN_OPS = EXPERIMENT / 'gpt56_zero_analysis_code/ops'
sys.path.insert(0, str(FROZEN_OPS))
import analyze_gpt56_temperature_curve as legacy
import analyze_gpt56_pass8_frontier_with_zero as old_pass8

paired = legacy.paired
native, integrity, provider = legacy.native, legacy.integrity, legacy.provider
sha, file_sha, require = legacy.sha, legacy.file_sha, legacy.require
read_jsonl, identity, row_identity, unique = legacy.read_jsonl, legacy.identity, legacy.row_identity, legacy.unique
bound_file, within = legacy.bound_file, legacy.within
DOMAINS, LEVELS = legacy.DOMAINS, legacy.LEVELS
TEMPERATURES = (0.0, 0.5, 1.0, 1.5, 2.0)
GRADINGS = ('strict', 'normalized_secondary')
REPLICATES = 20000
SEED = 20260916
PLAN = EXPERIMENT / 'prompt_expansion_32_per_cell/plan.json'
OUTPUT = EXPERIMENT / 'GPT56_TEMPERATURE_CURVE_EXPANDED480'
CURVE_SCHEMA = 'gpt56-none-temperature-curve-expanded480-v1'
FRONTIER_SCHEMA = 'gpt56-pass8-temperature-frontier-expanded480-v1'
CELL_KEYS = tuple(f'level{level}/{domain}' for level in LEVELS for domain in DOMAINS)
COHORT_SIZES = {'combined_480': 32, 'original_120': 8, 'additional_360': 24}


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def binding(path):
    path = resolve(path)
    return bound_file(path, file_sha(path))


def authenticate_plan(path=PLAN):
    """Rebuild the outcome-blind selection and all approved request digests."""
    path = resolve(path)
    plan = json.loads(path.read_text())
    require(plan.get('schema') == 'gpt56-temperature-prompt-expansion-plan-v1' and
            plan.get('temperatures') == list(TEMPERATURES) and
            plan.get('model') == legacy.MODEL and plan.get('reasoning_effort') == 'none',
            'Wrong approved expanded temperature plan')
    require(plan['counts']['combined_prompts'] == 480 and
            plan['counts']['existing_authenticated_samples'] == 4800 and
            plan['counts']['new_sample_slots'] == 14400 and
            plan['counts']['draws_per_prompt'] == 8 and
            plan['selection']['seed'] == paired.SELECTION_SEED and
            plan['selection']['uses_outcomes'] is False, 'Expanded plan counts or selection changed')
    old_pass8.authenticate(plan, ROOT)
    original_rows = read_jsonl(resolve(plan['sources']['original_rows']['path']))
    selected = paired.selected_original_rows(original_rows, 32)
    retained = paired.selected_original_rows(original_rows, 8)
    selection = unique(read_jsonl(resolve(plan['artifacts']['selection.jsonl']['path'])),
                       row_identity, 'expanded selection')
    require(set(selection) == set(selected), 'Expanded selection misses or adds prompt identities')
    for key, row in selected.items():
        item = selection[key]
        cohort = 'original_120' if key in retained else 'additional_360'
        # Preserve the exact cohort names sealed in the approved selection.
        require(item['cohort'] == ('existing_120' if key in retained else 'additional_360') and
                item['row_sha256'] == sha(row) and item['sample_indices'] == list(range(8)),
                'Expanded selection changed cohort, row bytes, or draw slots')
        ranked = sorted((r for r in original_rows if row_identity(r)[:2] == key[:2]),
                        key=lambda r: __import__('hashlib').sha256(
                            (paired.SELECTION_SEED + ':' + sha(r)).encode()).hexdigest())
        rank = next(i + 1 for i, r in enumerate(ranked) if row_identity(r) == key)
        selection_hash = __import__('hashlib').sha256((paired.SELECTION_SEED + ':' + sha(row)).encode()).hexdigest()
        require(item['selection_rank_in_cell'] == rank and item['selection_hash'] == selection_hash,
                'Expanded selection rank differs from frozen hash ordering')
    original_requests = unique(read_jsonl(resolve(plan['sources']['original_requests']['path'])),
                               identity, 'original request')
    hashes = unique(read_jsonl(resolve(plan['artifacts']['new_request_hashes.jsonl']['path'])),
                    lambda r: (float(r['temperature']), identity(r)), 'planned new request')
    new_rows = {k: r for k, r in selected.items() if k not in retained}
    require(set(hashes) == {(t, (*key, i)) for t in TEMPERATURES for key in new_rows for i in range(8)},
            'Approved request inventory misses or adds new draw slots')
    for (temperature, key), record in hashes.items():
        original = original_requests[key]
        expected = copy.deepcopy(original['request'])
        expected['reasoning']['effort'] = 'none'
        expected['temperature'] = temperature
        require(record['planned_request_sha256'] == sha(expected) and
                record['reference_request_sha256'] == original['request_sha256'] and
                record['sample_id'] == original['sample_id'] and
                record['row_sha256'] == sha(new_rows[key[:3]]),
                'Approved request changed prompt bytes or other sampling configuration')
    receipts = unique(read_jsonl(resolve(plan['artifacts']['retained_measurement_bindings.jsonl']['path'])),
                      lambda r: (float(r['temperature']), r['sample_id']), 'retained binding')
    require(len(receipts) == 4800, 'Require all 4,800 original receipt bindings')
    old_pass8.authenticate(list(receipts.values()), ROOT)
    return {'path': path, 'plan': plan, 'rows': selected, 'old_rows': retained,
            'new_rows': new_rows, 'original_requests': original_requests,
            'new_hashes': hashes, 'retained_receipts': receipts}


def validate_records(records, rows, prompts_per_cell, label):
    require(len(records) == prompts_per_cell * 15, f'{label}: incomplete prompt cohort')
    by_identity = unique(records, row_identity, label + ' prompt')
    require(set(by_identity) == set(rows), f'{label}: missing or extra prompt identity')
    groups = {key: [] for key in CELL_KEYS}
    for key, record in by_identity.items():
        require(type(record.get('level')) is int and record['level'] in LEVELS and
                record.get('domain') in DOMAINS and type(record.get('row_index')) is int and
                record['row_index'] >= 0, f'{label}: invalid prompt identity')
        require(record['row_sha256'] == sha(rows[key]) and record['responses'] == 8,
                f'{label}: changed row bytes or incomplete eight-draw group')
        correct, distinct = record['correct_responses'], record['distinct8']
        require(type(correct) is int and 0 <= correct <= 8 and type(distinct) is int and
                0 <= distinct <= correct and (distinct > 0) == (correct > 0),
                f'{label}: invalid correct or distinct count')
        require(record['correct_pairs'] == correct * (correct - 1) // 2 and
                type(record['colliding_correct_pairs']) is int and
                0 <= record['colliding_correct_pairs'] <= record['correct_pairs'] and
                type(record['collision_eligible']) is bool and record['collision_eligible'] == (correct >= 2),
                f'{label}: inconsistent pair counts')
        support = rows[key]['metadata'].get('answer_mode_count')
        if key[1] == 'countdown' or rows[key]['metadata'].get('support_is_open'):
            support = None
        require(record.get('certified_support_count') == support and
                record.get('uniform_expected_colliding_correct_pairs') ==
                (record['correct_pairs'] / support if support else None),
                f'{label}: support baseline differs from frozen row metadata')
        for name in ('native_refusals', 'native_content_filtered', 'truncated_responses', 'empty_answers'):
            require(type(record.get(name)) is int and 0 <= record[name] <= 8,
                    f'{label}: invalid retained outcome {name}')
        groups[f'level{key[0]}/{key[1]}'].append(record)
    require(all(len(group) == prompts_per_cell for group in groups.values()),
            f'{label}: unequal domain/level cell sizes')
    return {key: sorted(group, key=row_identity) for key, group in groups.items()}


def add_pass8(block, points, boots):
    for key in CELL_KEYS:
        block['cells'][key]['pass8'] = old_pass8.describe(points[key], boots[key])
    for name, domains in old_pass8.GROUPS.items():
        group = block['groups'][name]
        keys = [key for key in CELL_KEYS if key.split('/')[1] in domains]
        group['overall']['pass8'] = old_pass8.describe(
            float(np.mean([points[k] for k in keys])), np.mean([boots[k] for k in keys], axis=0))
        level_points, level_boots = {}, {}
        for level in LEVELS:
            selected = [k for k in keys if k.startswith(f'level{level}/')]
            level_points[level] = float(np.mean([points[k] for k in selected]))
            level_boots[level] = np.mean([boots[k] for k in selected], axis=0)
            group['levels'][str(level)]['pass8'] = old_pass8.describe(level_points[level], level_boots[level])
        for a, b in old_pass8.CONTRASTS:
            group['level_contrasts'][f'L{b}-L{a}']['pass8'] = old_pass8.describe(
                level_points[b] - level_points[a], level_boots[b] - level_boots[a])


def analyze_records(temperature_records, rows, prompts_per_cell, *, replicates=REPLICATES, seed=SEED):
    """One whole-prompt resampling schedule shared across every temperature."""
    require(set(temperature_records) == set(TEMPERATURES), 'Missing or extra temperature')
    cells = {t: validate_records(rs, rows, prompts_per_cell, str(t)) for t, rs in temperature_records.items()}
    rng = np.random.default_rng(seed)
    indices = {key: rng.integers(0, prompts_per_cell, (replicates, prompts_per_cell)) for key in CELL_KEYS}
    points, boots, successes, success_boots, blocks = {}, {}, {}, {}, {}
    for temperature in TEMPERATURES:
        points[temperature], boots[temperature], successes[temperature], success_boots[temperature] = {}, {}, {}, {}
        for key in CELL_KEYS:
            records = cells[temperature][key]
            values = paired.matrix(records)
            points[temperature][key] = paired.rates(values.sum(axis=0))
            boot = np.empty((replicates, len(paired.METRICS)))
            for start in range(0, replicates, 256):
                boot[start:start+256] = paired.rates(values[indices[key][start:start+256]].sum(axis=1))
            boots[temperature][key] = boot
            passed = np.asarray([r['correct_responses'] > 0 for r in records], dtype=float)
            successes[temperature][key] = float(passed.mean())
            success_boots[temperature][key] = passed[indices[key]].mean(axis=1)
        block = {'cells': {key: paired.describe(points[temperature][key], boots[temperature][key]) |
                          {'counts': paired.counts(cells[temperature][key]),
                           'empty_answers': sum(r['empty_answers'] for r in cells[temperature][key])}
                          for key in CELL_KEYS},
                 'groups': legacy.groups(points[temperature], boots[temperature]),
                 'counts': paired.counts(temperature_records[temperature]),
                 'prompt_records': sorted(temperature_records[temperature], key=row_identity)}
        add_pass8(block, successes[temperature], success_boots[temperature])
        blocks[str(temperature)] = block
    def contrast(target, baseline):
        p = {key: points[target][key] - points[baseline][key] for key in CELL_KEYS}
        b = {key: boots[target][key] - boots[baseline][key] for key in CELL_KEYS}
        out = {'comparison': f'T{target:.1f}-T{baseline:.1f}',
               'cells': {key: paired.describe(p[key], b[key]) for key in CELL_KEYS},
               'groups': legacy.groups(p, b)}
        add_pass8(out, {key: successes[target][key] - successes[baseline][key] for key in CELL_KEYS},
                  {key: success_boots[target][key] - success_boots[baseline][key] for key in CELL_KEYS})
        return out
    return {'temperatures': blocks,
            'paired_contrasts_vs_t1p0': {str(t): contrast(t, 1.0) for t in TEMPERATURES},
            'paired_endpoint_contrast': contrast(2.0, 0.0),
            'paired_high_temperature_contrast': contrast(2.0, 1.5)}


def analyze_cohorts(cohorts, plan, *, replicates=REPLICATES):
    records = {grading: {temperature: sorted(
        legacy.prompt_records(cohorts[temperature]['old'], grading) +
        legacy.prompt_records(cohorts[temperature]['new'], grading), key=row_identity)
        for temperature in TEMPERATURES} for grading in GRADINGS}
    result = {'analyses': {}, 'cohort_sensitivity': {}}
    for cohort, size in COHORT_SIZES.items():
        rows = plan['rows'] if cohort == 'combined_480' else plan['old_rows'] if cohort == 'original_120' else plan['new_rows']
        analyses = {grading: analyze_records({t: [r for r in rs if row_identity(r) in rows]
                     for t, rs in records[grading].items()}, rows, size, replicates=replicates)
                    for grading in GRADINGS}
        if cohort == 'combined_480':
            result['analyses'] = analyses
        else:
            result['cohort_sensitivity'][cohort] = {
                'prompts_per_condition': len(rows), 'prompts_per_domain_level': size,
                'responses_per_condition': 8 * len(rows), 'analyses': analyses}
    result['cohort_sensitivity']['interpretation'] = (
        'Original 120 and additional 360 are disjoint fixed prompt subsets collected in different batches. '
        'Differences can reflect prompt composition or service drift and are not a causal collection-time effect. '
        'The combined estimate weights every domain/level equally and every retained prompt equally within its cell.')
    return result


def markdown(result):
    lines = ['# GPT-5.6 Sol temperature sweep: 480 matched prompts', '',
        '19,200 authenticated responses: 480 fixed prompts, eight draws each, temperatures 0, 0.5, 1, 1.5, and 2. '
        'Reasoning is none throughout. The original 4,800 responses are retained, with 14,400 additional responses. '
        'Each of 15 domain/level cells contains 32 prompts. Equal-cell means include every zero-success, refused, '
        'invalid, and truncated prompt group. Intervals use 20,000 paired, stratified whole-prompt bootstrap replicates.', '',
        '| Cohort | Grading | T | pass@8 [95% CI] | distinct@8 [95% CI] |',
        '|---|---|---:|---:|---:|']
    cohorts = {'combined_480': result['analyses'], **{k: result['cohort_sensitivity'][k]['analyses']
               for k in ('original_120', 'additional_360')}}
    for cohort, analyses in cohorts.items():
        for grading, analysis in analyses.items():
            for temperature in TEMPERATURES:
                overall = analysis['temperatures'][str(temperature)]['groups']['five_domain_macro']['overall']
                lines.append(f"| {cohort} | {grading} | {temperature:g} | " +
                    paired.fmt(overall['pass8'], interval=True) + '% | ' +
                    paired.fmt(overall['distinct8'], 1, True) + ' |')
    lines += ['', '## Paired T=2 minus T=1.5 contrasts', '',
              '| Cohort | Grading | pass@8 difference, pp [95% CI] | distinct@8 difference [95% CI] |',
              '|---|---|---:|---:|']
    for cohort, analyses in cohorts.items():
        for grading, analysis in analyses.items():
            stats = analysis['paired_high_temperature_contrast']['groups']['five_domain_macro']['overall']
            lines.append(f"| {cohort} | {grading} | " + paired.fmt(stats['pass8'], interval=True) +
                         ' | ' + paired.fmt(stats['distinct8'], 1, True) + ' |')
    lines += ['', result['cohort_sensitivity']['interpretation'], '',
              'Intervals are pointwise, without multiplicity adjustment. Provider echoes describe exposed controls; '
              'they do not independently reveal internal sampling. Historical medium-reasoning measurements cover '
              'only the original 120 prompts and are excluded from this expanded curve.', '']
    return '\n'.join(lines)


def authenticate_new_condition(run, plan, collection, io_workers=16):
    """Authenticate every arm-view byte against mixed collection and approved plan."""
    run = resolve(run)
    inv = native.load_inventory(run, expected_samples=2880)
    m = inv['manifest']
    temperature = m['temperature']
    require(m.get('expansion_view_schema') == 'gpt56-temperature-expansion-arm-view-v1' and
            m.get('model') == legacy.MODEL and m.get('reasoning_effort') == 'none' and
            temperature in TEMPERATURES and m.get('sample_count') == 8 and
            m.get('prompt_count') == 360 and not inv['grouped'], 'Wrong expanded arm-view design')
    require(inv['rows'] == plan['new_rows'], 'Expanded arm has unapproved rows or old prompts')
    requests = unique(inv['requests'], identity, 'expanded arm request')
    expected_keys = {(*key, i) for key in plan['new_rows'] for i in range(8)}
    require(set(requests) == expected_keys, 'Expanded arm misses or adds draw slots')
    parent_path = run / 'parent_collection_binding.json'
    parent = json.loads(parent_path.read_text())
    require(parent.get('schema') == 'gpt56-temperature-expansion-arm-binding-v1' and
            resolve(parent['parent_collection']) == collection['directory'] and
            parent['parent_manifest_sha256'] == file_sha(collection['directory'] / 'manifest.json') and
            parent['parent_requests_sha256'] == file_sha(collection['directory'] / 'requests.jsonl') and
            parent['temperature'] == temperature, 'Wrong expanded arm parent binding')
    parent_samples = unique(parent['samples'], lambda r: r['sample_id'], 'parent arm sample')
    require(set(parent_samples) == {r['sample_id'] for r in requests.values()},
            'Parent arm binding has missing or extra samples')
    for item in parent_samples.values():
        for kind in ('sample_receipt', 'raw_receipt'):
            bound_file(within(run, item[kind]), item[kind + '_sha256'])
            bound_file(within(collection['directory'], item[kind]), item[kind + '_sha256'])
    bound_file(parent_path, m['artifact_sha256'].get('parent_collection_binding.json'))
    for key, request in requests.items():
        original = plan['original_requests'][key]
        planned = plan['new_hashes'][temperature, key]
        expected = copy.deepcopy(original['request'])
        expected['reasoning']['effort'] = 'none'
        expected['temperature'] = temperature
        require(request['request'] == expected and request['request_sha256'] == planned['planned_request_sha256'] and
                request['reference_request_sha256'] == original['request_sha256'] and
                request['original_sample_id'] == original['sample_id'] and
                request['row_sha256'] == sha(plan['new_rows'][key[:3]]) and
                request['temperature_condition'] == temperature,
                'Expanded request differs from approved prompt bytes or controls')
        require(request == collection['requests'][request['sample_id']],
                'Arm-view request differs from mixed collection registration')
    summary = json.loads((run / 'summary.json').read_text())
    require(summary.get('status') == 'complete' and summary.get('received_responses') ==
            summary.get('expected_responses') == 2880 and summary.get('complete_prompts') == 360 and
            summary.get('models_returned') == {legacy.MODEL: 2880}, 'Incomplete expanded arm summary')
    audit = json.loads((run / 'completion_audit.json').read_text())
    require(audit.get('status') == 'pass' and audit.get('saved_samples') == audit.get('expected_responses') ==
            audit.get('unique_response_ids') == audit.get('unique_response_choice_ids') == 2880,
            'Expanded native completion audit is incomplete')
    evidence = json.loads((run / 'evidence_file_sha256.json').read_text())
    require(sha(evidence) == audit['evidence_inventory_sha256'], 'Expanded native evidence inventory changed')
    for name in ('manifest.json', 'rows.jsonl', 'datasets.json', 'requests.jsonl', 'samples.jsonl'):
        bound_file(run / name, evidence.get(name))
    for name, digest in summary['input_sha256'].items():
        if name == 'errors.jsonl' and digest is None:
            require(not (run / name).exists(), 'Unbound expanded error log appeared')
        else:
            bound_file(within(run, name), digest)
    history_path = run / 'transport_history_binding.json'
    history = json.loads(history_path.read_text())
    require(history.get('schema') == 'gpt56-temperature-expansion-transport-view-v1' and
            resolve(history['parent_collection']) == collection['directory'] and
            history['parent_manifest_sha256'] == file_sha(collection['directory'] / 'manifest.json') and
            history['view_manifest_sha256'] == file_sha(run / 'manifest.json') and
            history['temperature'] == temperature and history['logical_samples'] == 2880 and
            history['attempts_sha256'] == sha(history['attempts']), 'Expanded transport history binding differs')
    exporter = Path(__file__).resolve().parent / 'complete_gpt56_prompt_expansion_views.py'
    bound_file(exporter, history['exporter_sha256'])
    expected_dependencies = {'ops/run_gpt56_prompt_expansion.py', 'ops/prepare_gpt56_prompt_expansion.py',
                             'ops/evaluate_frontier_modebench.py'}
    require(set(history.get('collector_dependency_sha256', {})) == expected_dependencies,
            'Expanded transport helper lacks sealed collector dependency authentication')
    for relative, digest in history['collector_dependency_sha256'].items():
        require(digest == collection['manifest']['code_sha256'][relative], 'Transport dependency differs from collection seal')
        bound_file(within(collection['directory'], 'code/' + relative), digest)
    for name, log in history['logs'].items():
        bound_file(within(run, name), log['view_sha256'])
        bound_file(within(collection['directory'], name), log['parent_sha256'])
        ids = {item['sample_id'] for item in requests.values()}
        selected = [item for item in read_jsonl(collection['directory'] / name) if item.get('sample_id') in ids]
        require(read_jsonl(run / name) == selected and len(selected) == log['records'],
                'Expanded transport view omitted or added parent events')
    require('events.jsonl' in history['logs'], 'Expanded transport history lacks request-start events')
    attempts = unique(history['attempts'], lambda r: (r['sample_id'], r['attempt']), 'expanded raw attempt')
    require(len(attempts) == history['raw_attempts'] and
            {r['path'] for r in attempts.values()} == set(
                str(p.relative_to(run)) for p in (run / 'raw_responses').glob('*.json')),
            'Expanded transport view has missing or extra raw attempts')
    for item in attempts.values():
        bound_file(within(run, item['path']), item['sha256'])
        bound_file(within(collection['directory'], item['path']), item['sha256'])
    raw = unique(read_jsonl(run / 'samples.jsonl'), identity, 'expanded sample')
    require(set(raw) == set(requests), 'Expanded samples miss or add draw slots')
    require({p.stem for p in (run / 'sample_receipts').glob('*.json')} ==
            {s['sample_id'] for s in raw.values()}, 'Extra or missing expanded atomic sample')
    for key, sample in raw.items():
        request = requests[key]
        for name in ('sample_id', 'row_sha256', 'request_sha256', 'temperature_condition'):
            require(sample.get(name) == request[name], 'Expanded sample/request identity differs')
        integrity.grade_valid(sample)
        relative = sample['raw_receipt']
        atomic = 'sample_receipts/' + sample['sample_id'] + '.json'
        bound_file(within(run, relative), evidence.get(relative))
        bound_file(within(run, atomic), evidence.get(atomic))
        require(json.loads((run / atomic).read_text()) == sample, 'Expanded atomic sample differs from export')
        require(sample == collection['samples'][sample['sample_id']], 'Arm-view sample differs from collection receipt')
        require(file_sha(run / relative) == file_sha(collection['directory'] / relative) and
                file_sha(run / atomic) == file_sha(collection['directory'] / atomic),
                'Arm-view native or atomic bytes differ from collection receipt')
    bodies = native.validate_native_records(inv, list(raw.values()), io_workers)
    returned = legacy.returned_controls(raw, bodies, temperature)
    primary, grading_audit = integrity.validate_primary(run, summary, raw, evidence, 2880)
    normalized = paired.normalized_grades(run, summary, raw, primary)
    outcomes = {key: provider.classify_native(bodies[sample['raw_receipt']]['response'], 'responses')
                for key, sample in raw.items()}
    provider_report = json.loads((run / 'provider_outcomes.json').read_text())
    require(provider_report['status'] == 'complete' and provider_report['responses'] ==
            provider_report['expected_responses'] == 2880 and provider_report['native_protocol'] == 'responses',
            'Expanded provider outcome report is incomplete')
    old_pass8.authenticate(provider_report, ROOT)
    sidecar = unique(read_jsonl(run / 'provider_outcome_samples.jsonl'), identity, 'expanded provider outcome')
    require(set(sidecar) == set(raw), 'Expanded provider sidecar inventory differs')
    for key, classified in outcomes.items():
        require(all(sidecar[key].get(k) == v for k, v in classified.items()) and
                all(sidecar[key].get(k) == raw[key][k] for k in ('sample_id', 'row_sha256', 'request_sha256',
                    'raw_receipt', 'raw_receipt_sha256', 'response_id')) and
                sidecar[key]['raw_receipt_file_sha256'] == evidence[raw[key]['raw_receipt']],
                'Expanded provider outcome differs from native receipt')
    require(provider_report['totals'] == provider.aggregate(list(outcomes.values())),
            'Expanded provider outcome totals differ from native receipts')
    marker = json.loads((run / 'analysis_complete.json').read_text())
    require(marker.get('status') == 'complete' and marker.get('received_responses') == 2880 and
            marker.get('complete_prompts') == 360 and marker.get('api_calls') == 0,
            'Expanded postprocessing marker is incomplete')
    for name, digest in {**marker['evidence_sha256'], **marker['output_sha256']}.items():
        bound_file(within(run, name), digest)
    return {'directory': run, 'manifest': m, 'rows': inv['rows'], 'requests': requests,
            'raw': raw, 'bodies': bodies, 'summary': summary, 'primary': primary,
            'normalized': normalized, 'grading_audit': grading_audit, 'outcomes': outcomes,
            'returned_controls': returned, 'sources': {name: binding(run / name) for name in
            ('manifest.json', 'parent_collection_binding.json', 'transport_history_binding.json', 'rows.jsonl', 'requests.jsonl',
             'samples.jsonl', 'summary.json', 'completion_audit.json', 'evidence_file_sha256.json',
             'primary_python_regrade_audit.json', 'audited_primary_samples.jsonl', 'normalized_samples.jsonl',
             'native_control_audit.json', 'provider_outcomes.json', 'provider_outcome_samples.jsonl',
             'analysis_complete.json', 'postprocessing_source_manifest.json')}}


def audit_native_configuration(cohorts):
    """Independently retain exact native settings, served snapshots, and times."""
    seen_ids, seen_samples = set(), set()
    summaries, expected_settings, expected_grades, expected_normalization = {}, None, None, None
    for temperature in TEMPERATURES:
        summaries[str(temperature)] = {}
        for name in ('old', 'new'):
            cohort = cohorts[temperature][name]
            counts, timestamps = Counter(), []
            raw_bodies = cohort.get('bodies')
            for key, sample in cohort['raw'].items():
                uid = sample['response_id']
                require(uid not in seen_ids, 'A native response ID was reused across cohorts or temperatures')
                seen_ids.add(uid)
                sample_key = (temperature, *key)
                require(sample_key not in seen_samples, 'An eight-draw slot is duplicated across collection cohorts')
                seen_samples.add(sample_key)
                raw = (raw_bodies[sample['raw_receipt']] if raw_bodies is not None else
                       json.loads((cohort['directory'] / sample['raw_receipt']).read_text()))
                body = raw['response']
                request = cohort['requests'][key]['request']
                require(body.get('temperature') == temperature and
                        body.get('reasoning', {}).get('effort') == 'none' and
                        body.get('model') == legacy.MODEL and
                        body.get('max_output_tokens') == request.get('max_output_tokens') == 8192 and
                        body.get('store') == request.get('store') is False,
                        'Native controls differ from approved expanded request')
                snapshot = raw.get('headers', {}).get('x-ms-served-model')
                require(isinstance(snapshot, str) and snapshot, 'Served-model snapshot is unavailable')
                settings = {k: body.get(k) for k in ('model', 'reasoning', 'top_p', 'max_output_tokens', 'store')}
                settings['served_model'] = snapshot
                if expected_settings is None:
                    expected_settings = settings
                require(settings == expected_settings, 'Native non-temperature controls or served model drifted')
                counts[json.dumps(settings, sort_keys=True)] += 1
                stamp = raw.get('received_at_utc')
                require(isinstance(stamp, str), 'Missing native collection timestamp')
                datetime.fromisoformat(stamp.replace('Z', '+00:00'))
                timestamps.append(stamp)
            normalized = {field: cohort['summary']['normalized_secondary'][field] for field in
                ('normalization_source_sha256', 'initial_rule_audit_sha256', 'frozen_grader_contract_sha256')}
            grades = {k: v['sha256'] for k, v in cohort['grading_audit']['frozen_grader_modules'].items()}
            if expected_normalization is None:
                expected_normalization, expected_grades = normalized, grades
            require(normalized == expected_normalization and grades == expected_grades,
                    'Frozen graders or formatting normalization changed between cohorts')
            summaries[str(temperature)][name] = {'responses': len(timestamps),
                'first_received_at_utc': min(timestamps), 'last_received_at_utc': max(timestamps),
                'native_settings_counts': dict(counts)}
    require(len(seen_ids) == 19200, 'The expanded curve does not contain 19,200 unique native responses')
    return {'shared_native_configuration': expected_settings, 'collection_batches': summaries,
            'normalization_contract': expected_normalization, 'frozen_grader_modules_sha256': expected_grades,
            'interpretation': 'Matching provider echoes authenticate the exposed configuration and named snapshot; '
                              'they do not independently reveal internal sampling or rule out service drift.'}


def validate_source_summary(cohort):
    for grading in GRADINGS:
        records = legacy.prompt_records(cohort, grading)
        source = cohort['summary'] if grading == 'strict' else cohort['summary']['normalized_secondary']
        for key in CELL_KEYS:
            level, domain = key.split('/')
            group = [r for r in records if r['level'] == int(level[5:]) and r['domain'] == domain]
            values = paired.rates(paired.matrix(group).sum(axis=0))
            for i, metric in enumerate(paired.SOURCE_METRICS):
                expected = source['cells'][key]['metrics'][metric]['estimate']
                require((expected is None and not np.isfinite(values[i])) or
                        (expected is not None and abs(values[i] - expected) < 1e-12),
                        'Reconstructed expanded metric differs from frozen per-arm summary')


def authenticate_collection(directory, plan):
    directory = resolve(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    require(manifest.get('schema') == 'gpt56-temperature-expansion-collection-v1',
            'Wrong interleaved expanded collection schema')
    require(manifest.get('approved_plan_sha256') == file_sha(plan['path']) and
            manifest.get('request_count') == 14400 and manifest.get('prompt_count') == 360 and
            manifest.get('temperatures') == list(TEMPERATURES) and manifest.get('model') == legacy.MODEL and
            manifest.get('reasoning_effort') == 'none', 'Expanded collection registration changed')
    for name, digest in manifest['artifact_sha256'].items():
        bound_file(within(directory, name), digest)
    for name, digest in manifest['code_sha256'].items():
        bound_file(within(directory, 'code/' + name), digest)
    requests_list = read_jsonl(directory / 'requests.jsonl')
    requests = unique(requests_list, lambda r: r['sample_id'], 'mixed expanded request')
    require(len(requests) == 14400 and
            unique(read_jsonl(directory / 'rows.jsonl'), row_identity, 'mixed expanded row') == plan['new_rows'],
            'Mixed expanded collection is not the approved additional-360 cohort')
    inventory = unique(requests_list, lambda r: (r['temperature_condition'], identity(r)), 'mixed expanded slot')
    require(set(inventory) == set(plan['new_hashes']), 'Mixed collection misses or adds approved slots')
    for index in range(0, len(requests_list), 5):
        group = requests_list[index:index+5]
        require([r['temperature_condition'] for r in group] == list(TEMPERATURES) and
                len({identity(r) for r in group}) == 1,
                'Expanded request order is not interleaved within each prompt/draw slot')
    samples = unique(read_jsonl(directory / 'samples.jsonl'), lambda r: r['sample_id'], 'mixed expanded sample')
    require(set(samples) == set(requests), 'Mixed expanded collection is incomplete or contains extra samples')
    require({p.stem for p in (directory / 'sample_receipts').glob('*.json')} == set(requests),
            'Mixed expanded atomic inventory differs from approved requests')
    gate_path = directory / 'gate.json'
    gate = json.loads(gate_path.read_text())
    require(gate.get('schema') == 'gpt56-temperature-expansion-preflight-gate-v1' and
            gate.get('status') == 'authorized_supported_controls' and
            gate.get('manifest_sha256') == file_sha(directory / 'manifest.json') and
            gate.get('approved_plan_sha256') == file_sha(plan['path']) and
            gate.get('registered_requests') == 14400 and
            gate.get('preflights_count_toward_registered_requests') is True and
            gate.get('returned_controls_required') == manifest['returned_controls_required'],
            'Expanded preflight gate is missing or changed')
    require(len(gate['preflight_samples']) == 5, 'Require exactly five retained preflight samples')
    for item, preflight in zip(requests_list[:5], gate['preflight_samples']):
        sample = samples[item['sample_id']]
        require(preflight['sample_id'] == item['sample_id'] and
                preflight['temperature'] == item['temperature_condition'] and
                preflight['request_sha256'] == item['request_sha256'] and
                preflight['raw_receipt'] == sample['raw_receipt'], 'Preflight is not the registered first draw')
        bound_file(directory / 'sample_receipts' / (item['sample_id'] + '.json'), preflight['sample_receipt_sha256'])
        bound_file(within(directory, sample['raw_receipt']), preflight['raw_receipt_sha256'])
    return {'directory': directory, 'manifest': manifest, 'requests': requests, 'samples': samples,
            'manifest_binding': binding(directory / 'manifest.json'), 'gate_binding': binding(gate_path)}


def build_report(plan_path=PLAN, collection_directory=None, io_workers=16):
    plan = authenticate_plan(plan_path)
    collection_directory = collection_directory or plan['path'].parent / 'collection_v1'
    collection = authenticate_collection(collection_directory, plan)
    gate = legacy.authenticate_gate(EXPERIMENT)
    zero_gate = legacy.authenticate_extension_gate(EXPERIMENT)
    cohorts = {}
    for temperature in TEMPERATURES:
        arm = next(arm for arm in plan['plan']['arms'] if arm['temperature'] == temperature)
        old = legacy.authenticate_condition(resolve(arm['existing_directory']), io_workers)
        old_evidence = json.loads((old['directory'] / 'evidence_file_sha256.json').read_text())
        for subdirectory in ('raw_responses', 'sample_receipts'):
            require({str(p.relative_to(old['directory'])) for p in (old['directory'] / subdirectory).glob('*.json')} ==
                    {name for name in old_evidence if name.startswith(subdirectory + '/')},
                    'Original cohort contains missing or extra native evidence files')
        for sample in old['raw'].values():
            retained = plan['retained_receipts'].get((temperature, sample['sample_id']))
            require(retained is not None and retained['request_sha256'] == sample['request_sha256'] and
                    resolve(retained['sample_receipt']['path']) == old['directory'] / 'sample_receipts' / (sample['sample_id'] + '.json') and
                    resolve(retained['raw_receipt']['path']) == old['directory'] / sample['raw_receipt'],
                    'Original measurement does not match retained approved receipt binding')
        suffix = f't{temperature:.1f}'.replace('.', 'p')
        new = authenticate_new_condition(collection['directory'] / 'arms' / suffix, plan, collection, io_workers)
        cohorts[temperature] = {'old': old, 'new': new}
        validate_source_summary(old)
        validate_source_summary(new)
    native_configuration = audit_native_configuration(cohorts)
    result = analyze_cohorts(cohorts, plan)
    result.update(schema=CURVE_SCHEMA, status='complete', model=legacy.MODEL, reasoning_effort='none',
        temperatures=list(TEMPERATURES), total_registered_responses=19200,
        responses_per_condition=3840, prompts_per_condition=480, draws_per_prompt=8,
        created_at_utc=max(json.loads((cs['new']['directory'] / 'analysis_complete.json').read_text())['completed_at_utc']
                          for cs in cohorts.values()),
        timestamp_origin='Latest immutable per-arm offline-analysis completion timestamp; deterministic reconstruction.',
        conditions={str(t): {'cohorts': {name: {'directory': str(c['directory']), 'source_sha256': c['sources'],
            'returned_controls': c['returned_controls'], 'outcome_categories': legacy.outcome_categories(c)}
            for name, c in cs.items()}} for t, cs in cohorts.items()},
        approved_expansion_plan=binding(plan['path']),
        collection_gate=collection['gate_binding'], collection_manifest=collection['manifest_binding'],
        original_collection_gate=gate, original_zero_extension_gate=zero_gate,
        native_configuration_audit=native_configuration,
        bootstrap={'replicates': REPLICATES, 'seed': SEED,
            'unit': 'Whole eight-draw prompts, paired between temperatures and stratified by domain and level.',
            'individual_draw_pairing': False, 'interval': 'Pointwise 95% percentile; no multiplicity adjustment.',
            'undefined': 'Undefined cells propagate into macros; finite complete-macro replicates determine intervals.',
            'cohort_sensitivity': 'Separately resample eight original or 24 additional prompts per cell; combined resamples all 32.'},
        validation={'all_native_receipts_authenticated': True, 'authenticated_native_receipt_count': 19200,
            'same_480_prompts_all_eight_slots': True, 'only_temperature_varies_within_curve': True,
            'all_returned_controls_match_requests': True, 'source_summaries_reconstructed': True,
            'pass8_computed_from_complete_prompt_groups': True, 'expansion_plan_authenticated': True,
            'original_4800_measurements_retained_unchanged': True,
            'same_served_snapshot_and_native_non_temperature_controls': True,
            'cohort_sensitivity_reported': True, 'api_calls_during_analysis': 0},
        analysis_sources={Path(path).name: binding(path) for path in
            (__file__, legacy.__file__, paired.__file__, old_pass8.__file__, native.__file__, integrity.__file__, provider.__file__)})
    original = json.loads(resolve(plan['plan']['sources']['existing_pass8_analysis']['path']).read_text())
    result['historical_medium_reference_subset120'] = {
        'prompts_per_condition': 120, 'role': 'Historical original subset; excluded from expanded 480-prompt curve.',
        'source': plan['plan']['sources']['existing_pass8_analysis'],
        'reference': original['matched_medium_reference']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--collection', type=Path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--io-workers', type=int, default=16)
    args = parser.parse_args()
    result = build_report(args.plan, args.collection, args.io_workers)
    native.atomic(args.output.with_suffix('.json'), result)
    args.output.with_suffix('.md').write_text(markdown(result))
    print(json.dumps({'status': 'complete', 'responses': 19200, 'output': str(args.output.with_suffix('.json'))}))

if __name__ == '__main__':
    main()
