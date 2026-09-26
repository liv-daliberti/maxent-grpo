#!/usr/bin/env python3
"""Authenticate and compare paired, frozen hosted-model temperature conditions.

Offline only: validates native receipts and existing frozen grades, never calls
a model or regrades outputs. Bootstrap units are paired eight-draw prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_hosted_modebench_completion as native
import audit_hosted_provider_outcomes as provider
import analyze_frontier_python_prompt_condition as integrity

DOMAINS = ('countdown', 'graph_coloring', 'mathir', 'pantry_plan', 'python_factors')
LEVELS = (1, 2, 3)
CONDITION = 'temperature_ablation_v1'
SELECTION_SEED = 'frontier-temperature-ablation-20260911-v1'
METRICS = ('accuracy', 'distinct8', 'collision', 'uniform_collision', 'excess_uniform')
SOURCE_METRICS = ('pass1', 'distinct8', 'correct_pair_collision',
                  'uniform_correct_pair_collision', 'correct_pair_collision_excess_uniform')
MODEL_SLUGS = {'grok43': 'grok-4.3', 'kimi_k3': 'FW-Kimi-K3'}
sha, file_sha = integrity.sha, integrity.file_sha
require, bound_file, within = integrity.require, integrity.bound_file, integrity.within
read_jsonl, identity, row_identity, unique = (
    integrity.read_jsonl, integrity.identity, integrity.row_identity, integrity.unique)


def selected_original_rows(rows, count=8):
    selected = []
    for level in LEVELS:
        for domain in DOMAINS:
            group = [r for r in rows if (r['level'], r['domain']) == (level, domain)]
            require(len(group) == 128, 'Reference must retain all 128 original rows per cell')
            selected.extend(sorted(group, key=lambda r: __import__('hashlib').sha256(
                (SELECTION_SEED + ':' + sha(r)).encode()).hexdigest())[:count])
    return unique(selected, row_identity, 'selected original row')


def authenticate_collection_gate(root):
    root = Path(root).resolve()
    path = root / 'collection_gate.json'
    gate = json.loads(path.read_text())
    require(gate.get('status') == 'authorized_supported_controls' and
            gate.get('admitted_slugs') == list(MODEL_SLUGS) and
            gate.get('temperatures') == [1.0, 1.5] and
            gate.get('planned_terminal_responses') == 3840 and
            gate.get('responses_per_condition') == 960 and
            gate.get('prompt_count_per_condition') == 120 and
            gate.get('preflight_responses_included_in_total') == 4,
            'Collection gate differs from the fixed four-condition design')
    support = bound_file(gate['support_review'], gate['support_review_sha256'])
    for name, digest in gate['source_sha256'].items():
        bound_file(within(root / 'code' / 'ops', name), digest)
    expected = {slug + suffix for slug in MODEL_SLUGS for suffix in ('_t1p0', '_t1p5')}
    require(set(gate['preflight_evidence']) == expected, 'Missing admitted preflight binding')
    for label, evidence in gate['preflight_evidence'].items():
        run = root / label
        bound_file(run / 'manifest.json', evidence['manifest_sha256'])
        bound_file(run / 'preflight_result.json', evidence['result_sha256'])
        preflight = json.loads((run / 'preflight_result.json').read_text())
        require(preflight.get('exit_code') == 0 and preflight.get('terminal_samples') == 1 and
                preflight.get('condition') == label, 'First registered preflight did not complete')
        first = read_jsonl(run / 'requests.jsonl')[0]
        sample_path = run / 'sample_receipts' / (first['sample_id'] + '.json')
        bound_file(sample_path, evidence['sample_receipt_sha256'])
        sample = json.loads(sample_path.read_text())
        require(identity(sample) == identity(first) and sample['request_sha256'] == first['request_sha256'],
                'Preflight is not the first registered sample')
        bound_file(within(run, sample['raw_receipt']), evidence['raw_receipt_sha256'])
    return {'path': str(path), 'sha256': file_sha(path), 'support_review': support,
            'details': gate, 'first_registered_samples_bound': True}


def normalized_grades(run, summary, raw, primary):
    secondary = summary['normalized_secondary']
    for path, digest in ((run / 'normalized_samples.jsonl', secondary['cache_sha256']),
                         (secondary['normalization_source_path'], secondary['normalization_source_sha256']),
                         (secondary['frozen_grader_contract_path'], secondary['frozen_grader_contract_sha256']),
                         (run / 'secondary_initial15_audit.json', secondary['initial_rule_audit_sha256'])):
        bound_file(path, digest)
    initial = json.loads((run / 'secondary_initial15_audit.json').read_text())
    require(initial['normalizer']['sha256'] == secondary['normalization_source_sha256'],
            'Normalizer differs from its original frozen development audit')
    cache = {}
    for item in read_jsonl(run / 'normalized_samples.jsonl'):
        if item.get('normalization_source_sha256') != secondary['normalization_source_sha256']:
            continue
        key = item['strict_receipt_sha256']
        require(key not in cache or cache[key] == item, 'Conflicting normalization cache entries')
        cache[key] = item
    output = {}
    for key, sample in primary.items():
        receipt = {**raw[key], **{name: sample[name] for name in integrity.GRADES}}
        cache_key = sha(receipt)
        require(cache_key in cache, 'Missing current audited normalization cache entry')
        item = cache[cache_key]
        require(identity(item) == key, 'Normalization cache identity differs')
        grade = item['normalization']
        integrity.grade_valid(grade)
        require(grade.get('original_text') == sample['text'], 'Normalization original text differs')
        require(not sample['verified'] or
                (grade['verified'] and grade['canonical_key'] == sample['canonical_key']),
                'Normalization changed a strict success')
        output[key] = grade
    return output


def validate_provider_sidecar(run, summary, raw, raw_bodies, evidence):
    path = run / 'provider_outcomes.json'
    report = json.loads(path.read_text())
    expected, model = len(raw), summary['run_configuration']['model']
    require(report.get('status') == 'complete' and report.get('model') == model and
            report.get('responses') == expected and report.get('expected_responses') == expected and
            Path(report['source_run']).resolve() == run, 'Incomplete or mismatched provider report')
    for name in ('manifest.json', 'samples.jsonl', 'completion_audit.json', 'evidence_file_sha256.json'):
        source = report['sources'][name]
        require(Path(source['path']).resolve() == run / name, 'Provider report uses another cohort')
        bound_file(run / name, source['sha256'])
    source = Path(report['audit_source_path'])
    bound_file(source, report['audit_source_sha256'])
    bound_file(source.with_name('audit_hosted_modebench_completion.py'), report['audit_helper_source_sha256'])
    sidecar = report['sample_outcomes']
    require(Path(sidecar['path']).resolve() == run / 'provider_outcome_samples.jsonl',
            'Provider sidecar lies outside selected condition')
    bound_file(sidecar['path'], sidecar['sha256'])
    outcomes = unique(read_jsonl(sidecar['path']), identity, 'provider outcome')
    require(set(outcomes) == set(raw) and sidecar['records'] == expected, 'Provider inventory differs')
    for key, sample in raw.items():
        outcome, body = outcomes[key], raw_bodies[sample['raw_receipt']]
        require(all(outcome.get(name) == sample[name] for name in
                    ('sample_id', 'row_sha256', 'request_sha256', 'raw_receipt', 'raw_receipt_sha256', 'response_id')),
                'Provider sidecar identity differs from native sample')
        require(outcome['raw_receipt_file_sha256'] == evidence[sample['raw_receipt']],
                'Provider sidecar native file binding differs')
        classified = provider.classify_native(body['response'], 'chat_completions', 0)
        require(all(outcome.get(k) == value for k, value in classified.items()),
                'Provider outcomes differ from native response metadata')
    require(report['totals'] == provider.aggregate(list(outcomes.values())), 'Provider totals differ')
    for level in LEVELS:
        for domain in DOMAINS:
            records = [o for k, o in outcomes.items() if k[:2] == (level, domain)]
            require(report['cells'][f'level{level}/{domain}'] == provider.aggregate(records),
                    'Provider cell totals differ')
    return outcomes


def authenticate_condition(run, prompts_per_cell=8, io_workers=16):
    run = Path(run).resolve()
    expected = len(LEVELS) * len(DOMAINS) * prompts_per_cell * 8
    inventory = native.load_inventory(run, expected_samples=expected)
    manifest = inventory['manifest']
    model, temperature = manifest['model'], manifest.get('temperature')
    require(model in MODEL_SLUGS.values() and temperature in (1.0, 1.5) and
            manifest.get('experiment_condition') == CONDITION and
            manifest.get('original_prompt_cohort') is False and manifest.get('sample_count') == 8 and
            manifest.get('protocol') == 'chat_completions', 'Wrong registered ablation condition')
    require(len(inventory['groups']) == expected and all(g['sample_count'] == 1 for g in inventory['groups'].values()),
            'Require independent single-choice requests')
    condition = json.loads((run / 'temperature_condition.json').read_text())
    require(condition.get('condition') == CONDITION and condition.get('requested_temperature') == temperature and
            condition.get('selection_seed') == SELECTION_SEED and condition.get('selection_uses_outcomes') is False and
            condition.get('changed_request_fields') == ['temperature'], 'Frozen condition metadata differs')
    summary = json.loads((run / 'summary.json').read_text())
    require(summary.get('status') == 'complete' and summary.get('received_responses') == expected and
            summary.get('expected_responses') == expected and summary.get('complete_prompts') == expected // 8 and
            summary.get('models_returned') == {model: expected} and
            summary['run_configuration']['model'] == model, 'Summary is incomplete or wrong model')
    audit = json.loads((run / 'completion_audit.json').read_text())
    require(audit.get('status') == 'pass' and audit.get('model') == model and
            audit.get('expected_responses') == expected and audit.get('saved_samples') == expected and
            audit.get('unique_response_ids') == expected and audit.get('unique_response_choice_ids') == expected,
            'Completion audit is incomplete or has reused responses')
    evidence = json.loads((run / 'evidence_file_sha256.json').read_text())
    require(sha(evidence) == audit['evidence_inventory_sha256'], 'Completion inventory digest differs')
    for name in ('manifest.json', 'rows.jsonl', 'datasets.json', 'requests.jsonl', 'http_requests.jsonl', 'samples.jsonl'):
        bound_file(run / name, evidence.get(name))
    for name, digest in summary['input_sha256'].items():
        path = within(run, name)
        if name == 'errors.jsonl' and digest is None:
            require(not path.exists(), 'Unbound error log appeared after summary')
        else:
            bound_file(path, digest)
    rows = inventory['rows']
    require(Counter(k[:2] for k in rows) == Counter({(l, d): prompts_per_cell for l in LEVELS for d in DOMAINS}),
            'Not the complete stratified prompt inventory')
    require(condition.get('row_sha256') == [sha(r) for r in read_jsonl(run / 'rows.jsonl')],
            'Condition row digest list differs')
    raw = unique(read_jsonl(run / 'samples.jsonl'), identity, 'sample identity')
    requests = unique(inventory['requests'], identity, 'request identity')
    expected_keys = {(*key, i) for key in rows for i in range(8)}
    require(set(raw) == set(requests) == expected_keys and len(raw) == expected and
            len({s['response_id'] for s in raw.values()}) == expected, 'Missing or reused eight-draw slot')
    for key, sample in raw.items():
        item = requests[key]
        require(item.get('temperature_condition') == temperature and item['request'].get('temperature') == temperature,
                'Requested temperature differs from condition')
        require(all(sample.get(name) == item[name] for name in
                    ('sample_id', 'level', 'domain', 'row_index', 'sample_index', 'row_sha256', 'request_sha256')),
                'Sample identity differs from frozen request')
        integrity.grade_valid(sample)
        bound_file(within(run, sample['raw_receipt']), evidence.get(sample['raw_receipt']))
        atomic = 'sample_receipts/' + sample['sample_id'] + '.json'
        bound_file(within(run, atomic), evidence.get(atomic))
        require(json.loads((run / atomic).read_text()) == sample, 'Atomic sample differs from export')
    raw_bodies = native.validate_native_records(inventory, list(raw.values()), io_workers)
    primary, grading_audit = integrity.validate_primary(run, summary, raw, evidence, expected)
    normalized = normalized_grades(run, summary, raw, primary)
    outcomes = validate_provider_sidecar(run, summary, raw, raw_bodies, evidence)
    sources = {name: {'path': str(run / name), 'sha256': file_sha(run / name)} for name in
               ('summary.json', 'manifest.json', 'temperature_condition.json', 'rows.jsonl', 'requests.jsonl',
                'samples.jsonl', 'completion_audit.json', 'evidence_file_sha256.json',
                'primary_python_regrade_audit.json', 'normalized_samples.jsonl', 'provider_outcomes.json',
                'provider_outcome_samples.jsonl')}
    sources['audited_primary'] = bound_file(summary['primary_samples_path'], summary['primary_samples_sha256'])
    return {'directory': run, 'manifest': manifest, 'summary': summary, 'rows': rows, 'requests': requests,
            'raw': raw, 'primary': primary, 'normalized': normalized, 'outcomes': outcomes,
            'grading_audit': grading_audit, 'sources': sources}


def validate_pairing(low, high, prompts_per_cell=8):
    require(low['directory'] != high['directory'] and low['rows'] == high['rows'],
            'Temperatures must use separate cohorts with identical mathematical tasks')
    require(low['manifest']['temperature'] == 1.0 and high['manifest']['temperature'] == 1.5 and
            low['manifest']['model'] == high['manifest']['model'], 'Mismatched model or temperature pair')
    for field in ('reference_run', 'reference_manifest_sha256', 'reference_artifact_sha256'):
        require(low['manifest'][field] == high['manifest'][field], 'Different original reference cohorts')
    reference = Path(low['manifest']['reference_run']).resolve()
    bound_file(reference / 'manifest.json', low['manifest']['reference_manifest_sha256'])
    for name in ('rows.jsonl', 'requests.jsonl'):
        bound_file(reference / name, low['manifest']['reference_artifact_sha256'][name])
    reference_rows = selected_original_rows(read_jsonl(reference / 'rows.jsonl'), prompts_per_cell)
    require(low['rows'] == reference_rows, 'Rows differ from the fixed outcome-blind selection')
    originals = unique(read_jsonl(reference / 'requests.jsonl'), identity, 'original request')
    for key in low['requests']:
        original = originals[key]
        require(sha(original['request']) == original['request_sha256'], 'Original payload digest differs')
        for cohort in (low, high):
            item = cohort['requests'][key]
            require(item.get('reference_request_sha256') == original['request_sha256'] and
                    item['row_sha256'] == original['row_sha256'], 'Lost original task/request binding')
            require({k: v for k, v in item['request'].items() if k != 'temperature'} ==
                    {k: v for k, v in original['request'].items() if k != 'temperature'},
                    'A non-temperature payload field changed')
        require({k: v for k, v in low['requests'][key]['request'].items() if k != 'temperature'} ==
                {k: v for k, v in high['requests'][key]['request'].items() if k != 'temperature'},
                'Paired payloads differ beyond temperature')
    require(not {s['response_id'] for s in low['raw'].values()}.intersection(
                s['response_id'] for s in high['raw'].values()), 'A response was reused across temperatures')
    for field in ('normalization_source_sha256', 'initial_rule_audit_sha256', 'frozen_grader_contract_sha256'):
        require(low['summary']['normalized_secondary'][field] == high['summary']['normalized_secondary'][field],
                'Frozen normalizer or grading contract differs across temperatures')
    require({k: v['sha256'] for k, v in low['grading_audit']['frozen_grader_modules'].items()} ==
            {k: v['sha256'] for k, v in high['grading_audit']['frozen_grader_modules'].items()},
            'Executable graders differ across temperatures')


def prompt_records(cohort, grading):
    grades = cohort['primary'] if grading == 'strict' else cohort['normalized']
    records = []
    for key, row in sorted(cohort['rows'].items()):
        keys = [(*key, i) for i in range(8)]
        modes = Counter(sha(grades[k]['canonical_key']) for k in keys if grades[k]['verified'])
        correct = sum(modes.values())
        pairs = correct * (correct - 1) // 2
        support = row['metadata'].get('answer_mode_count')
        if row['domain'] == 'countdown' or row['metadata'].get('support_is_open'):
            support = None
        require(support is None or (isinstance(support, int) and support > 0), 'Invalid finite support count')
        records.append({'level': key[0], 'domain': key[1], 'row_index': key[2], 'row_sha256': sha(row),
                        'responses': 8, 'correct_responses': correct, 'distinct8': len(modes),
                        'correct_pairs': pairs, 'colliding_correct_pairs': sum(n*(n-1)//2 for n in modes.values()),
                        'certified_support_count': support,
                        'uniform_expected_colliding_correct_pairs': pairs / support if support else None,
                        'collision_eligible': correct >= 2,
                        'native_refusals': sum(cohort['outcomes'][k]['refusal'] for k in keys),
                        'native_content_filtered': sum(cohort['outcomes'][k]['content_filtered'] for k in keys),
                        'truncated_responses': sum(cohort['raw'][k]['stop_reason'] == 'length' for k in keys)})
    return records


def matrix(records):
    return np.asarray([[1, r['correct_responses'], r['distinct8'], r['correct_pairs'],
                        r['colliding_correct_pairs'], r['uniform_expected_colliding_correct_pairs'] or 0,
                        r['correct_pairs'] if r['certified_support_count'] else 0,
                        r['colliding_correct_pairs'] if r['certified_support_count'] else 0]
                       for r in records], dtype=float)


def rates(totals):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.stack((totals[..., 1] / (8 * totals[..., 0]), totals[..., 2] / totals[..., 0],
                         totals[..., 4] / totals[..., 3], totals[..., 5] / totals[..., 6],
                         (totals[..., 7] - totals[..., 5]) / totals[..., 6]), axis=-1)


def describe(point, boots):
    output = {}
    for j, name in enumerate(METRICS):
        valid = boots[:, j][np.isfinite(boots[:, j])]
        output[name] = {'estimate': float(point[j]) if np.isfinite(point[j]) else None,
                        'ci95': np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                        'defined_replicates': len(valid)}
    return output


def counts(records):
    result = {name: sum(r[name] for r in records) for name in
              ('responses', 'correct_responses', 'distinct8', 'correct_pairs', 'colliding_correct_pairs',
               'native_refusals', 'native_content_filtered', 'truncated_responses')}
    result.update(prompts=len(records), collision_eligible_prompts=sum(r['collision_eligible'] for r in records),
                  prompts_with_correct_answer=sum(r['correct_responses'] > 0 for r in records))
    return result


def joint_eligibility(low, high):
    require([r['row_sha256'] for r in low] == [r['row_sha256'] for r in high], 'Prompt pairing order differs')
    return {'both_eligible': sum(a['collision_eligible'] and b['collision_eligible'] for a, b in zip(low, high)),
            'only_t1p0_eligible': sum(a['collision_eligible'] and not b['collision_eligible'] for a, b in zip(low, high)),
            'only_t1p5_eligible': sum(not a['collision_eligible'] and b['collision_eligible'] for a, b in zip(low, high)),
            'neither_eligible': sum(not a['collision_eligible'] and not b['collision_eligible'] for a, b in zip(low, high))}


def paired_bootstrap(low, high, indices):
    require(low.shape == high.shape and len(low) > 0, 'Unequal or empty paired prompt matrices')
    points = {'t1p0': rates(low.sum(axis=0)), 't1p5': rates(high.sum(axis=0))}
    boots = {label: np.empty((len(indices), len(METRICS))) for label in points}
    for label, values in (('t1p0', low), ('t1p5', high)):
        for start in range(0, len(indices), 256):
            boots[label][start:start+256] = rates(values[indices[start:start+256]].sum(axis=1))
    points['t1p5_minus_t1p0'] = points['t1p5'] - points['t1p0']
    boots['t1p5_minus_t1p0'] = boots['t1p5'] - boots['t1p0']
    return points, boots


def macro(points, boots, keys):
    # np.mean deliberately propagates undefined cells; never silently drop them.
    return {label: describe(np.mean([points[k][label] for k in keys], axis=0),
                            np.mean([boots[k][label] for k in keys], axis=0))
            for label in ('t1p0', 't1p5', 't1p5_minus_t1p0')}


def analyze_pair(low, high, indices):
    validate_pairing(low, high)
    output = {'model': low['manifest']['model'], 'conditions': {
        label: {'directory': str(c['directory']), 'source_sha256': c['sources'],
                'requested_temperature': c['manifest']['temperature'],
                'returned_sampling': c['summary'].get('returned_sampling'),
                'api_error_attempts': c['summary'].get('api_error_attempts')}
        for label, c in (('t1p0', low), ('t1p5', high))}, 'analyses': {}}
    for grading in ('strict', 'normalized_secondary'):
        records = {label: prompt_records(c, grading) for label, c in (('t1p0', low), ('t1p5', high))}
        cells, points, boots = {}, {}, {}
        for level in LEVELS:
            for domain in DOMAINS:
                key = f'level{level}/{domain}'
                subset = {label: [r for r in rs if (r['level'], r['domain']) == (level, domain)]
                          for label, rs in records.items()}
                joint = joint_eligibility(subset['t1p0'], subset['t1p5'])
                point, boot = paired_bootstrap(matrix(subset['t1p0']), matrix(subset['t1p5']), indices[level, domain])
                for label, cohort in (('t1p0', low), ('t1p5', high)):
                    source = cohort['summary'] if grading == 'strict' else cohort['summary']['normalized_secondary']
                    for j, name in enumerate(SOURCE_METRICS):
                        expected = source['cells'][key]['metrics'][name]['estimate']
                        require((expected is None and not np.isfinite(point[label][j])) or
                                (expected is not None and abs(expected - point[label][j]) < 1e-12),
                                'Reconstructed statistic differs from source summary: ' + key + '/' + name)
                cells[key] = {label: describe(point[label], boot[label]) for label in point}
                cells[key]['counts'] = {label: counts(rs) for label, rs in subset.items()}
                cells[key]['joint_eligibility'] = joint
                points[key], boots[key] = point, boot
        groups = {'five_domain_macro': {'levels': {}, 'overall': macro(points, boots, list(points))},
                  'four_finite_support_domain_macro': {'levels': {}, 'overall': macro(
                      points, boots, [k for k in points if not k.endswith('/countdown')])}}
        for level in LEVELS:
            keys = [k for k in points if k.startswith(f'level{level}/')]
            groups['five_domain_macro']['levels'][str(level)] = macro(points, boots, keys)
            groups['four_finite_support_domain_macro']['levels'][str(level)] = macro(
                points, boots, [k for k in keys if not k.endswith('/countdown')])
        output['analyses'][grading] = {'cells': cells, 'groups': groups,
            'totals': {label: counts(rs) for label, rs in records.items()},
            'joint_eligibility': joint_eligibility(records['t1p0'], records['t1p5']),
            'prompt_records': records}
    output['validation'] = {'all_1920_native_receipts_authenticated': True,
        'same_120_rows_and_all_eight_draw_slots': True, 'temperature_only_payload_change': True,
        'normalizer_and_audited_primary_source_bound': True, 'reconstructed_cells_match_summaries': True}
    return output


def fmt(metric, scale=100, interval=False):
    if metric['estimate'] is None:
        return '—'
    text = f"{scale * metric['estimate']:.2f}"
    if interval and metric['ci95'] is not None:
        text += f" [{scale * metric['ci95'][0]:.2f}, {scale * metric['ci95'][1]:.2f}]"
    return text


def markdown(result):
    lines = ['# Hosted-model temperature ablation', '',
        f"{len(result['models'])} complete paired model{'s' if len(result['models']) != 1 else ''}; each temperature condition contains 960 saved responses "
        '(eight fixed held-out prompts in each of 15 domain-level cells, eight independent draws per prompt). '
        'Prompts, requested medium reasoning and the 8,192-token limit remain fixed; only requested temperature changes from 1.0 to 1.5.', '',
        'This is a small exploratory sampling ablation. HTTP acceptance validates the API contract, while effective '
        'internal sampling settings are not independently observed. All returned invalid answers, refusals and '
        'truncations remain in the accuracy and distinct8 denominators. Collision uses only pairs of correct answers, '
        'with eligibility reported separately. All original full cohorts are unchanged.', '',
        'Intervals are pointwise 95% paired whole-prompt bootstrap intervals. Prompt identities are paired; '
        'individual generated draws are independent. Macros weight every included domain-level cell equally '
        'and propagate undefined collision cells. Countdown has no finite uniform-support reference.', '']
    for model, record in result['models'].items():
        overall = record['analyses']['strict']['groups']['five_domain_macro']['overall']
        a, b, delta = (overall[k] for k in ('t1p0', 't1p5', 't1p5_minus_t1p0'))
        totals = record['analyses']['strict']['totals']
        lines += [f"**{model}:** across all 15 cells, mean distinct correct modes per eight draws change "
            f"from {fmt(a['distinct8'], 1)} to {fmt(b['distinct8'], 1)} "
            f"(difference {fmt(delta['distinct8'], 1, True)}). Correct-pair collision changes "
            f"from {fmt(a['collision'])}% to {fmt(b['collision'])}% "
            f"(difference {fmt(delta['collision'], interval=True)} pp); accuracy changes "
            f"from {fmt(a['accuracy'])}% to {fmt(b['accuracy'])}% "
            f"(difference {fmt(delta['accuracy'], interval=True)} pp). "
            f"Retained truncations: {totals['t1p0']['truncated_responses']} at T=1.0 and "
            f"{totals['t1p5']['truncated_responses']} at T=1.5, out of 960 answers each.", '']
    for grading in ('strict', 'normalized_secondary'):
        lines += ['## ' + grading.replace('_', ' '), '',
            '| Model | Scope | Accuracy T1 / T1.5 % | Δ accuracy pp [95% CI] | Modes T1 / T1.5 | Δ modes [95% CI] | Collision T1 / T1.5 % | Δ collision pp [95% CI] |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for model, record in result['models'].items():
            group = record['analyses'][grading]['groups']['five_domain_macro']
            for scope, values in [*group['levels'].items(), ('overall', group['overall'])]:
                a, b, delta = (values[k] for k in ('t1p0', 't1p5', 't1p5_minus_t1p0'))
                lines.append(f"| {model} | {scope} | {fmt(a['accuracy'])} / {fmt(b['accuracy'])} | {fmt(delta['accuracy'], interval=True)} | "
                    f"{fmt(a['distinct8'], 1)} / {fmt(b['distinct8'], 1)} | {fmt(delta['distinct8'], 1, True)} | "
                    f"{fmt(a['collision'])} / {fmt(b['collision'])} | {fmt(delta['collision'], interval=True)} |")
        lines += ['', '| Model | Cell | Accuracy T1 / T1.5 % | Modes T1 / T1.5 | Collision T1 / T1.5 % | Δ collision pp [95% CI] | Uniform T1 / T1.5 % | Eligible prompts T1 / T1.5 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for model, record in result['models'].items():
            for key, cell in record['analyses'][grading]['cells'].items():
                a, b, delta = (cell[k] for k in ('t1p0', 't1p5', 't1p5_minus_t1p0'))
                ca, cb = (cell['counts'][k] for k in ('t1p0', 't1p5'))
                lines.append(f"| {model} | {key} | {fmt(a['accuracy'])} / {fmt(b['accuracy'])} | "
                    f"{fmt(a['distinct8'], 1)} / {fmt(b['distinct8'], 1)} | {fmt(a['collision'])} / {fmt(b['collision'])} | "
                    f"{fmt(delta['collision'], interval=True)} | {fmt(a['uniform_collision'])} / {fmt(b['uniform_collision'])} | "
                    f"{ca['collision_eligible_prompts']} / {cb['collision_eligible_prompts']} |")
        lines.append('')
    lines += ['## Retained provider outcomes', '', '| Model | Cell | Refusals T1 / T1.5 | Filter signals T1 / T1.5 | Truncations T1 / T1.5 |',
              '|---|---|---:|---:|---:|']
    for model, record in result['models'].items():
        for key, cell in record['analyses']['strict']['cells'].items():
            a, b = (cell['counts'][k] for k in ('t1p0', 't1p5'))
            lines.append(f"| {model} | {key} | {a['native_refusals']} / {b['native_refusals']} | "
                         f"{a['native_content_filtered']} / {b['native_content_filtered']} | {a['truncated_responses']} / {b['truncated_responses']} |")
    lines += ['', 'The JSON retains every paired prompt group, per-cell accuracy/mode/collision intervals, '
        'finite-support references, collision eligibility, returned settings and source digests. '
        'Eight prompts per cell yield coarse uncertainty estimates; no multiplicity correction or independent '
        'resampling of API draws is applied. A zero-width interval can occur when all observed prompt statistics '
        'coincide; it does not prove exact equality in the population. A temperature effect on valid-mode concentration need not be '
        'monotone, and these static results do not identify training-induced collapse.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/frontier_temperature_20260911'))
    parser.add_argument('--slug', choices=tuple(MODEL_SLUGS), action='append')
    parser.add_argument('--replicates', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--io-workers', type=int, default=16)
    args = parser.parse_args()
    require(args.replicates > 0, 'Replicates must be positive')
    rng = np.random.default_rng(args.seed)
    indices = {(l, d): rng.integers(0, 8, (args.replicates, 8)) for l in LEVELS for d in DOMAINS}
    result = {'schema': 'hosted-temperature-ablation-comparison-v1', 'status': 'complete',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'models': {},
        'bootstrap': {'replicates': args.replicates, 'seed': args.seed,
            'unit': 'Paired whole eight-draw prompt groups, independently stratified by domain and level.',
            'shared_indices_across_models': True, 'individual_draw_pairing': False,
            'interval': 'Pointwise 95% percentile; no multiplicity adjustment.'},
        'analysis_sources': {p.name: {'path': str(p.resolve()), 'sha256': file_sha(p)} for p in
             (Path(__file__), Path(native.__file__), Path(provider.__file__), Path(integrity.__file__))},
        'definitions': {'accuracy': 'Correct answers / all retained sampled answers.',
            'distinct8': 'Mean number of distinct correct canonical keys per eight draws; zero-correct prompts contribute zero.',
            'collision': 'Colliding correct pairs / correct pairs, pooled within each domain-level cell.',
            'uniform_collision': 'Correct-pair-weighted uniform collision over certified finite canonical modes; undefined for Countdown.',
            'macro': 'Equal mean of every included cell; any undefined member makes that macro metric undefined.',
            'normalized_secondary': 'Previously frozen formatting-only normalizer applied to source-bound audited strict receipts.'},
        'limitations': ['Small exploratory follow-up: eight prompts per cell.',
            'API acceptance is not independent observation of the internal sampling actuator.',
            'Eligibility can differ across temperatures; collision conditions on correct answers.',
            'Original full model cohorts and their provider defaults are not replaced.',
            'Level populations differ and do not establish monotonically greater model difficulty.',
            'Static inference concentration does not identify a training or parameter-scale cause.']}
    result['collection_gate'] = authenticate_collection_gate(args.root)
    row_digests = set()
    for slug in args.slug or list(MODEL_SLUGS):
        low = authenticate_condition(args.root / (slug + '_t1p0'), io_workers=args.io_workers)
        high = authenticate_condition(args.root / (slug + '_t1p5'), io_workers=args.io_workers)
        require(low['manifest']['model'] == MODEL_SLUGS[slug], 'Condition directory has the wrong model')
        row_digests.add(sha([r for _, r in sorted(low['rows'].items())]))
        result['models'][MODEL_SLUGS[slug]] = analyze_pair(low, high, indices)
    require(len(row_digests) == 1, 'Models use different selected task populations')
    result['selected_rows_sha256'] = next(iter(row_digests))
    native.atomic(args.root / 'TEMPERATURE_ABLATION.json', result)
    (args.root / 'TEMPERATURE_ABLATION.md').write_text(markdown(result))
    print(json.dumps({'models': list(result['models']), 'status': 'complete', 'output': str(args.root)}))


if __name__ == '__main__':
    main()
