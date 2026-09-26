#!/usr/bin/env python3
"""Offline, paired-prompt comparison of original and plain Python conditions.

No API calls or grading are performed. Completed, audited native receipts and
the already frozen formatting cache are authenticated before any statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = 'claude-opus-5'
DOMAIN = 'python_factors'
PLAIN_CONDITION = 'python_plain_user_no_system_v1'
LEVELS = (1, 2, 3)
GRADES = ('verified', 'canonical_key', 'graded_text')
METRICS = ('native_refusal_rate', 'strict_accuracy', 'normalized_accuracy',
           'strict_distinct8', 'normalized_distinct8',
           'strict_correct_pair_collision', 'normalized_correct_pair_collision')


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as handle:
        result = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_file(path, expected):
    path = Path(path).resolve()
    require(isinstance(expected, str) and file_sha(path) == expected,
            'Changed or unbound evidence: ' + str(path))
    return {'path': str(path), 'sha256': expected}


def within(run, relative):
    path = (run / relative).resolve()
    require(path.is_relative_to(run), 'Evidence path escapes cohort: ' + str(relative))
    return path


def identity(record):
    return (record['level'], record['domain'], record['row_index'], record['sample_index'])


def row_identity(row):
    return (row['level'], row['domain'], row['row_index'])


def plain_prompt(cases):
    return (f'For the inputs n in {cases}, write one arithmetic expression in Python notation lambda n: EXPR. '
            'For every listed input, the result d must be an integer satisfying 1 < d < n and n % d == 0. '
            'Use only integer literals, n, +, -, *, //, %, comparisons, Boolean operators, and conditional expressions. '
            'Any proper divisor meeting these conditions is acceptable. '
            r'Return only the one-line lambda inside \boxed{}.')


def unique(records, key, label):
    result = {key(row): row for row in records}
    require(len(result) == len(records), 'Duplicate ' + label)
    return result


def grade_valid(grade):
    require(isinstance(grade.get('verified'), bool) and
            grade['verified'] == (grade.get('canonical_key') is not None),
            'Inconsistent verified/canonical-key fields')


def normalization_grades(run, summary, raw, audited):
    """Use the exact current raw-receipt + audited-grade cache key, never fallback."""
    secondary = summary['normalized_secondary']
    bound_file(run / 'normalized_samples.jsonl', secondary['cache_sha256'])
    bound_file(secondary['normalization_source_path'], secondary['normalization_source_sha256'])
    bound_file(secondary['frozen_grader_contract_path'], secondary['frozen_grader_contract_sha256'])
    bound_file(run / 'secondary_initial15_audit.json', secondary['initial_rule_audit_sha256'])
    initial = json.loads((run / 'secondary_initial15_audit.json').read_text())
    require(initial['normalizer']['sha256'] == secondary['normalization_source_sha256'],
            'Normalizer differs from original frozen rule audit')
    cache = {}
    for record in read_jsonl(run / 'normalized_samples.jsonl'):
        if record.get('normalization_source_sha256') != secondary['normalization_source_sha256']:
            continue
        key = record['strict_receipt_sha256']
        require(key not in cache or cache[key] == record, 'Conflicting duplicate normalization cache key')
        cache[key] = record
    output = {}
    for key, sample in audited.items():
        if sample['domain'] != DOMAIN:
            continue
        receipt = {**raw[key], **{name: sample[name] for name in GRADES}}
        cache_key = sha(receipt)
        require(cache_key in cache, 'No authenticated current normalization entry: ' + sample['sample_id'])
        entry = cache[cache_key]
        require(identity(entry) == key, 'Normalization entry has wrong sample identity')
        grade = entry['normalization']
        grade_valid(grade)
        require(grade.get('original_text') == sample['text'], 'Normalization changed original answer text')
        require(not sample['verified'] or
                (grade['verified'] and grade['canonical_key'] == sample['canonical_key']),
                'Normalization changed an audited strict success')
        output[key] = grade
    return output


def validate_primary(run, summary, raw, evidence, expected):
    audit_path = run / 'primary_python_regrade_audit.json'
    declared = summary['primary_grading_audit']
    require(Path(declared['path']).resolve() == audit_path, 'Summary names a different primary audit')
    bound_file(audit_path, declared['sha256'])
    audit = json.loads(audit_path.read_text())
    require(audit.get('status') == 'complete_for_snapshot' and audit.get('full_sampling_complete') is True
            and audit.get('snapshot_samples') == expected and audit.get('expected_samples') == expected
            and audit.get('unresolved_groups') == 0,
            'Serial Python audit is incomplete, stale, or unresolved')
    bound_file(run / 'manifest.json', audit['primary_manifest_sha256'])
    require(audit['python_samples'] == sum(s['domain'] == DOMAIN for s in raw.values()),
            'Serial audit did not cover all Python samples')
    receipts = audit['source_receipt_manifest']
    require(sha(receipts) == audit['source_receipt_manifest_sha256']
            and set(receipts) == {s['sample_id'] for s in raw.values()},
            'Serial audit source-receipt inventory mismatch')
    for record in receipts.values():
        require(evidence.get(record['path']) == record['sha256'],
                'Serial audit receipt is not bound to completion audit')
    for source in audit['frozen_grader_modules'].values():
        bound_file(source['path'], source['sha256'])
    bound_file(within(run, audit['audit_source']), audit['audit_source_sha256'])
    bound_file(within(run, audit['audit_helper_source']), audit['audit_helper_sha256'])
    derived = audit['derived_primary']
    bound_file(within(run, derived['path']), derived['sha256'])
    primary_path = Path(summary['primary_samples_path']).resolve()
    require(primary_path.is_relative_to(run) and primary_path != run / 'samples.jsonl',
            'Require the separate audited-primary file in the selected cohort')
    bound_file(primary_path, summary['primary_samples_sha256'])
    require(summary['primary_samples_sha256'] == derived['sha256'] and derived['records'] == expected,
            'Summary does not use the complete audited primary')
    records = read_jsonl(primary_path)
    require(sha(records) == summary['response_records_sha256'], 'Summary primary snapshot changed')
    primary = unique(records, identity, 'audited sample identity')
    require(set(primary) == set(raw), 'Audited-primary sample inventory differs')
    for key, sample in primary.items():
        original = raw[key]
        grade_valid(sample)
        require(all(sample.get(name) == value for name, value in original.items() if name not in GRADES),
                'Audited primary changed a non-grading field')
        receipt = receipts[sample['sample_id']]
        require(sample['raw_strict_receipt'] == receipt['path'] and
                sample['raw_strict_receipt_sha256'] == receipt['sha256'] and
                all(sample['raw_strict_' + name] == original[name] for name in GRADES),
                'Audited primary lost its original strict receipt binding')
    return primary, audit


def native_outcome(body):
    """Independent Anthropic-only reconstruction of explicit provider outcomes."""
    blocks = body.get('content') or []
    text = ''.join(part.get('text', '') for part in blocks if part.get('type') == 'text')
    details = body.get('stop_details')
    details = details if isinstance(details, dict) else {}
    refusal = (body.get('stop_reason') == 'refusal' or details.get('type') == 'refusal'
               or any(p.get('type') == 'refusal' for p in blocks) or bool(body.get('refusal')))
    return {'refusal': refusal, 'native_stop_reason': body.get('stop_reason'),
            'native_stop_details': body.get('stop_details'),
            'category_labels': [details['category']] if isinstance(details.get('category'), str) else [],
            'answer_text_empty': not text.strip(),
            'answer_text_sha256': hashlib.sha256(text.encode()).hexdigest()}, text


def validate_python_receipt(run, item, sample, outcome, evidence):
    relative = sample['raw_receipt']
    path = within(run, relative)
    require(path.parent == run / 'raw_responses', 'Unexpected raw receipt location')
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    require(evidence.get(relative) == digest == outcome['raw_receipt_file_sha256'],
            'Native receipt bytes differ from completion/outcome audit')
    raw = json.loads(raw_bytes)
    require(sha(raw) == sample['raw_receipt_sha256'] == outcome['raw_receipt_sha256'],
            'Native receipt canonical digest mismatch')
    require(raw.get('sample_id') == sample['sample_id'] and raw.get('relative_path') == relative
            and raw.get('request_sha256') == item['request_sha256'] and raw.get('http_status') == 200,
            'Native receipt does not identify the expected successful request')
    body = raw['response']
    require(body.get('model') == MODEL and body.get('id') == sample['response_id'] == outcome['response_id'],
            'Native response deployment/identity mismatch')
    require(body.get('stop_reason') in ('end_turn', 'stop_sequence', 'max_tokens', 'refusal'),
            'Native body is not a completed or token-limited sampled response')
    classified, text = native_outcome(body)
    require(text == sample['text'] and all(outcome.get(k) == v for k, v in classified.items()),
            'Provider sidecar/sample differs from independently extracted native outcome')
    atomic_path = 'sample_receipts/' + sample['sample_id'] + '.json'
    atomic_bytes = within(run, atomic_path).read_bytes()
    require(hashlib.sha256(atomic_bytes).hexdigest() == evidence.get(atomic_path)
            and json.loads(atomic_bytes) == sample, 'Atomic sample receipt differs from completed export')
    return classified


def authenticate_cohort(run, expected_responses, prompts_per_level=128, io_workers=16):
    run = Path(run).resolve()
    summary = json.loads((run / 'summary.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    audit = json.loads((run / 'completion_audit.json').read_text())
    evidence = json.loads((run / 'evidence_file_sha256.json').read_text())
    require(manifest.get('model') == MODEL and manifest.get('sample_count') == 8
            and manifest.get('request_count') == expected_responses,
            'Wrong model or planned cohort size')
    require(summary.get('status') == 'complete' and summary.get('received_responses') == expected_responses
            and summary.get('expected_responses') == expected_responses
            and summary.get('complete_prompts') == expected_responses // 8
            and summary.get('models_returned') == {MODEL: expected_responses}, 'Summary is not a complete cohort')
    require(audit.get('status') == 'pass' and audit.get('expected_responses') == expected_responses
            and audit.get('saved_samples') == expected_responses
            and audit.get('unique_response_ids') == expected_responses and audit.get('model') == MODEL,
            'Completion audit is not a complete, unique cohort')
    require(sha(evidence) == audit['evidence_inventory_sha256'], 'Completion evidence inventory changed')
    for name in ('manifest.json', 'rows.jsonl', 'datasets.json', 'requests.jsonl', 'samples.jsonl'):
        bound_file(run / name, evidence.get(name))
        if name in summary['input_sha256']:
            bound_file(run / name, summary['input_sha256'][name])
    for name, digest in manifest['artifact_sha256'].items():
        bound_file(within(run, name), digest)
    rows = unique(read_jsonl(run / 'rows.jsonl'), row_identity, 'row identity')
    require(len(rows) * 8 == expected_responses, 'Row and response counts differ')
    python_rows = {k: v for k, v in rows.items() if k[1] == DOMAIN}
    expected_python = {(level, DOMAIN, row_index) for level in LEVELS for row_index in range(prompts_per_level)}
    require(set(python_rows) == expected_python, 'Python prompt population is incomplete or unexpected')
    raw = unique(read_jsonl(run / 'samples.jsonl'), identity, 'raw sample identity')
    requests = unique(read_jsonl(run / 'requests.jsonl'), identity, 'request identity')
    expected_keys = {(*key, i) for key in rows for i in range(8)}
    require(len(raw) == expected_responses and set(raw) == set(requests) == expected_keys,
            'Not exactly eight independently requested responses per frozen row')
    require(len({s['response_id'] for s in raw.values()}) == expected_responses
            and len({s['sample_id'] for s in raw.values()}) == expected_responses,
            'Duplicate response/sample identifier')
    for key, sample in raw.items():
        item = requests[key]
        require(all(sample.get(name) == item[name] for name in
                    ('sample_id', 'level', 'domain', 'row_index', 'sample_index', 'row_sha256', 'request_sha256'))
                and item['row_sha256'] == sha(rows[key[:3]]) and item['request_sha256'] == sha(item['request']),
                'Saved request/row/sample identity mismatch')
        require(sample['model'] == MODEL and item['request']['model'] == MODEL, 'Mixed deployment cohort')
        grade_valid(sample)
    primary, grading_audit = validate_primary(run, summary, raw, evidence, expected_responses)
    normalized = normalization_grades(run, summary, raw, primary)
    outcomes_path = run / 'provider_outcomes.json'
    outcomes = json.loads(outcomes_path.read_text())
    require(outcomes.get('status') == 'complete' and outcomes.get('model') == MODEL
            and outcomes.get('responses') == expected_responses
            and outcomes.get('expected_responses') == expected_responses
            and Path(outcomes['source_run']).resolve() == run, 'Provider-outcome audit is incomplete or wrong cohort')
    for name in ('manifest.json', 'samples.jsonl', 'completion_audit.json', 'evidence_file_sha256.json'):
        source = outcomes['sources'][name]
        require(Path(source['path']).resolve() == run / name, 'Provider audit names a different source run')
        bound_file(run / name, source['sha256'])
    bound_file(outcomes['audit_source_path'], outcomes['audit_source_sha256'])
    outcome_sidecar = outcomes['sample_outcomes']
    bound_file(outcome_sidecar['path'], outcome_sidecar['sha256'])
    all_outcomes = unique(read_jsonl(outcome_sidecar['path']), identity, 'provider-outcome sample')
    require(set(all_outcomes) == set(raw) and outcome_sidecar['records'] == expected_responses,
            'Provider-outcome sample inventory differs')
    for key, outcome in all_outcomes.items():
        require(all(outcome.get(name) == raw[key][name] for name in
                    ('sample_id', 'row_sha256', 'request_sha256', 'raw_receipt', 'raw_receipt_sha256', 'response_id')),
                'Outcome/sample receipt identity differs')
    require(outcomes['totals']['responses'] == expected_responses and
            outcomes['totals']['refusals'] == sum(o['refusal'] for o in all_outcomes.values()),
            'Provider-outcome total differs from complete sidecar')
    python_keys = sorted(k for k in raw if k[1] == DOMAIN)
    def read_native(key):
        return key, validate_python_receipt(run, requests[key], raw[key], all_outcomes[key], evidence)
    require(1 <= io_workers <= 32, 'I/O workers must be 1..32')
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        native = dict(pool.map(read_native, python_keys))
    for level in LEVELS:
        cell = outcomes['cells'][f'level{level}/{DOMAIN}']
        subset = [v for k, v in native.items() if k[0] == level]
        require(cell['responses'] == len(subset) and cell['refusals'] == sum(v['refusal'] for v in subset)
                and cell['category_counts'] == dict(Counter(c for v in subset for c in v['category_labels'])),
                'Provider per-cell counts differ from authoritative native receipts')
    sources = {name: {'path': str(run / name), 'sha256': file_sha(run / name)} for name in
               ('summary.json', 'manifest.json', 'rows.jsonl', 'requests.jsonl', 'samples.jsonl',
                'completion_audit.json', 'evidence_file_sha256.json', 'primary_python_regrade_audit.json',
                'normalized_samples.jsonl', 'provider_outcomes.json')}
    sources['audited_primary'] = bound_file(summary['primary_samples_path'], summary['primary_samples_sha256'])
    sources['provider_outcome_samples'] = bound_file(outcome_sidecar['path'], outcome_sidecar['sha256'])
    return {'directory': run, 'summary': summary, 'manifest': manifest, 'rows': python_rows,
            'requests': {k: requests[k] for k in python_keys}, 'raw': {k: raw[k] for k in python_keys},
            'primary': {k: primary[k] for k in python_keys}, 'normalized': normalized, 'native': native,
            'grading_audit': grading_audit, 'sources': sources}


def prompt_records(cohort):
    result = []
    for key, row in sorted(cohort['rows'].items()):
        keys = [(*key, index) for index in range(8)]
        refused = sum(cohort['native'][k]['refusal'] for k in keys)
        record = {'level': key[0], 'row_index': key[2], 'row_sha256': sha(row), 'responses': 8,
                  'native_refusals': refused, 'all_refused': refused == 8, 'any_refused': refused > 0,
                  'native_category_counts': dict(Counter(c for k in keys for c in cohort['native'][k]['category_labels']))}
        for label, source in (('strict', 'primary'), ('normalized', 'normalized')):
            modes = Counter(sha(cohort[source][k]['canonical_key']) for k in keys if cohort[source][k]['verified'])
            count = sum(modes.values())
            record[label] = {'correct_responses': count, 'distinct8': len(modes),
                             'correct_pairs': count * (count - 1) // 2,
                             'colliding_correct_pairs': sum(n * (n - 1) // 2 for n in modes.values()),
                             'eligible_accuracy': True, 'eligible_distinct8': True,
                             'eligible_correct_pair_collision': count >= 2,
                             'has_correct_answer': count >= 1}
        result.append(record)
    return result


def matrix(prompts):
    return np.asarray([[1, p['responses'], p['native_refusals'],
                        p['strict']['correct_responses'], p['normalized']['correct_responses'],
                        p['strict']['distinct8'], p['normalized']['distinct8'],
                        p['strict']['correct_pairs'], p['strict']['colliding_correct_pairs'],
                        p['normalized']['correct_pairs'], p['normalized']['colliding_correct_pairs']]
                       for p in prompts], dtype=float)


def rates(totals):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.stack((totals[..., 2] / totals[..., 1], totals[..., 3] / totals[..., 1],
                         totals[..., 4] / totals[..., 1], totals[..., 5] / totals[..., 0],
                         totals[..., 6] / totals[..., 0], totals[..., 8] / totals[..., 7],
                         totals[..., 10] / totals[..., 9]), axis=-1)


def estimates(point):
    return {name: float(value) if np.isfinite(value) else None for name, value in zip(METRICS, point)}


def paired_contrast(original, plain, replicates, rng):
    require(original.shape == plain.shape and len(original) > 0 and replicates > 0,
            'Paired bootstrap requires equally sized nonempty prompt groups and positive replicates')
    point = rates(plain.sum(axis=0)) - rates(original.sum(axis=0))
    # Independent draws within each prompt are not paired by sample_index.
    # The complete observed eight-draw groups are paired only by task identity.
    boots = np.empty((replicates, 3))
    for start in range(0, replicates, 256):
        end = min(start + 256, replicates)
        indices = rng.integers(0, len(original), (end - start, len(original)))
        boots[start:end] = (rates(plain[indices].sum(axis=1)) - rates(original[indices].sum(axis=1)))[:, :3]
    return {name: {'estimate': float(point[index]),
                   'ci95': np.quantile(boots[:, index], [.025, .975]).tolist(),
                   'defined_replicates': replicates}
            for index, name in enumerate(METRICS[:3])}


def describe(prompts):
    counts = {'prompts': len(prompts), 'responses': 8 * len(prompts),
              'native_refusals': sum(p['native_refusals'] for p in prompts),
              'all_refused_prompts': sum(p['all_refused'] for p in prompts),
              'any_refused_prompts': sum(p['any_refused'] for p in prompts)}
    for label in ('strict', 'normalized'):
        counts[label] = {name: sum(p[label][name] for p in prompts) for name in
                         ('correct_responses', 'distinct8', 'correct_pairs', 'colliding_correct_pairs')}
        counts[label]['prompts_with_correct_answer'] = sum(p[label]['has_correct_answer'] for p in prompts)
        counts[label]['collision_eligible_prompts'] = sum(p[label]['eligible_correct_pair_collision'] for p in prompts)
        counts[label]['collision_ineligible_prompts'] = len(prompts) - counts[label]['collision_eligible_prompts']
    return {'metrics': estimates(rates(matrix(prompts).sum(axis=0))), 'counts': counts}


def validate_summary_cells(cohort, prompts):
    for level in LEVELS:
        independent = describe([p for p in prompts if p['level'] == level])['metrics']
        for label, source in (('strict', cohort['summary']), ('normalized', cohort['summary']['normalized_secondary'])):
            cell = source['cells'][f'level{level}/{DOMAIN}']['metrics']
            for target, summary_key in (('accuracy', 'pass1'), ('distinct8', 'distinct8'),
                                        ('correct_pair_collision', 'correct_pair_collision')):
                got, expected = independent[label + '_' + target], cell[summary_key]['estimate']
                require((got is None and expected is None) or
                        (got is not None and expected is not None and abs(got - expected) < 1e-12),
                        'Independently reconstructed statistic differs from source summary')


def validate_pairing(original, plain):
    require(original['directory'] != plain['directory'], 'Conditions must be separate cohorts')
    require(original['rows'] == plain['rows'], 'Prompt conditions changed mathematical tasks or row identities')
    require(plain['manifest'].get('experiment_condition') == PLAIN_CONDITION,
            'Missing or different frozen plain-prompt condition identity')
    require(Path(plain['manifest']['reference_run']).resolve() == original['directory'] and
            plain['manifest']['reference_manifest_sha256'] == original['sources']['manifest.json']['sha256'],
            'Plain condition does not reference the selected original cohort')
    for source_key in ('normalization_source_sha256', 'initial_rule_audit_sha256', 'frozen_grader_contract_sha256'):
        require(original['summary']['normalized_secondary'][source_key] == plain['summary']['normalized_secondary'][source_key],
                'Conditions have different frozen normalization/grader contracts')
    require({k: v['sha256'] for k, v in original['grading_audit']['frozen_grader_modules'].items()} ==
            {k: v['sha256'] for k, v in plain['grading_audit']['frozen_grader_modules'].items()},
            'Conditions have different executable grader modules')
    original_ids = {s['response_id'] for s in original['raw'].values()}
    require(not original_ids.intersection(s['response_id'] for s in plain['raw'].values()),
            'Plain condition reuses original model responses')
    for key, item in plain['requests'].items():
        before = original['requests'][key]['request']
        after = item['request']
        require(item.get('condition') == PLAIN_CONDITION and
                item.get('reference_request_sha256') == original['requests'][key]['request_sha256'],
                'Plain request lost its condition/original-request binding')
        require('system' not in after, 'Plain condition includes a system field')
        require({k: v for k, v in before.items() if k not in ('system', 'messages')} ==
                {k: v for k, v in after.items() if k not in ('system', 'messages')},
                'Prompt condition changed non-prompt request controls')
        require(before['messages'] != after['messages'] and len(after['messages']) == 1
                and after['messages'][0]['role'] == 'user', 'Condition is not the intended single-user prompt change')
        cases = json.loads(plain['rows'][key[:3]]['answer'])['cases']
        require(after['messages'][0]['content'] == plain_prompt(cases),
                'Plain request differs from the single fixed educational template')


def compare(original, plain, replicates=20000, seed=20260914):
    validate_pairing(original, plain)
    records = {'original': prompt_records(original), 'plain': prompt_records(plain)}
    validate_summary_cells(original, records['original'])
    validate_summary_cells(plain, records['plain'])
    rng = np.random.default_rng(seed)
    levels = {}
    for level in LEVELS:
        selected = {label: [p for p in rows if p['level'] == level] for label, rows in records.items()}
        require([p['row_sha256'] for p in selected['original']] == [p['row_sha256'] for p in selected['plain']],
                'Paired prompt order differs')
        levels[str(level)] = {label: describe(rows) for label, rows in selected.items()}
        levels[str(level)]['plain_minus_original'] = paired_contrast(
            matrix(selected['original']), matrix(selected['plain']), replicates, rng)
        levels[str(level)]['joint_eligibility'] = {label: {
            'both_collision_eligible': sum(a[label]['eligible_correct_pair_collision'] and b[label]['eligible_correct_pair_collision']
                                           for a, b in zip(selected['original'], selected['plain'])),
            'only_original_collision_eligible': sum(a[label]['eligible_correct_pair_collision'] and not b[label]['eligible_correct_pair_collision']
                                                    for a, b in zip(selected['original'], selected['plain'])),
            'only_plain_collision_eligible': sum(not a[label]['eligible_correct_pair_collision'] and b[label]['eligible_correct_pair_collision']
                                                 for a, b in zip(selected['original'], selected['plain']))}
            for label in ('strict', 'normalized')}
    return {'schema': 'hosted-python-prompt-condition-comparison-v1', 'status': 'complete',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'model': MODEL,
            'analysis_source': {'path': str(Path(__file__).resolve()), 'sha256': file_sha(__file__)},
            'conditions': {label: {'directory': str(cohort['directory']), 'source_sha256': cohort['sources'],
                                   'selected_domain': DOMAIN, 'selected_prompts': len(cohort['rows']),
                                   'selected_responses': len(cohort['raw']),
                                   'source_cohort_responses': cohort['summary']['received_responses'],
                                   'selected_rows_sha256': sha([r for _, r in sorted(cohort['rows'].items())]),
                                   'selected_native_receipts_sha256': sha({s['sample_id']: s['raw_receipt_sha256']
                                                                          for s in cohort['raw'].values()})}
                           for label, cohort in (('original', original), ('plain', plain))},
            'bootstrap': {'replicates': replicates, 'seed': seed, 'confidence_level': .95,
                          'unit': 'Whole eight-draw prompt groups, paired across conditions by exact original task row.',
                          'draw_pairing': False, 'independent_indices_across_levels': True,
                          'interval': 'Pointwise percentile; no multiplicity adjustment; API draws are not separately resampled.'},
            'levels': levels, 'totals': {label: describe(rows) for label, rows in records.items()},
            'prompt_records': records,
            'definitions': {
                'native_refusal_rate': 'Explicit provider-declared refusals divided by all 1,024 sampled responses per level.',
                'accuracy': 'Correct responses divided by all sampled responses, including refusals and token-limited answers.',
                'distinct8': 'Mean number of distinct correct canonical modes across all prompts; zero-correct prompts contribute zero.',
                'correct_pair_collision': 'Sum of colliding correct pairs / sum of correct pairs within prompts. Undefined if denominator is zero.',
                'collision_eligibility': 'A prompt needs at least two correct answers; eligibility is reported separately under each grading and condition.',
                'normalized': 'Frozen formatting-only secondary analysis; every strict success and canonical mode are retained.'},
            'validation': {'all_python_native_receipts_reauthenticated': True,
                           'provider_refusal_counts_reconstructed_from_native_bodies': True,
                           'audited_primary_and_frozen_normalizer_hashes_authenticated': True,
                           'independent_metrics_match_source_summaries': True,
                           'same_task_rows_and_non_prompt_request_controls': True,
                           'distinct_response_ids_across_conditions': True},
            'limitations': [
                'This is a separately collected prompt condition, not a replacement for the original cohort or its refusals.',
                'The plain condition was selected after observing original refusals and synthetic development probes; this is an exploratory follow-up.',
                'The shared task prompts justify paired prompt resampling. Sample indices do not pair independent model outputs.',
                'Provider state and collection time can differ. This comparison does not identify a classifier mechanism or a general causal prompting effect.',
                'Collision describes correct answers only, weights prompts by their numbers of correct pairs, and can compare different eligible prompt populations.',
                'Level populations differ; higher benchmark level does not itself establish greater model difficulty.',
                'Static concentration is not evidence of training-induced collapse or zero probability of unobserved modes.']}


def markdown(result):
    lines = ['# Opus 5 Python prompt condition', '',
             'Separate original and plain-prompt conditions: 384 identical Python tasks and 3,072 independent responses each. '
             'Original samples are the complete Python subset of the original 15,360-response cohort. '
             'All eight responses remain in each prompt, including provider refusals.', '',
             f"Native refusals: **{result['totals']['original']['counts']['native_refusals']:,} / 3,072 original**, "
             f"**{result['totals']['plain']['counts']['native_refusals']:,} / 3,072 plain**. "
             'The new condition omits the system field and uses one fixed positive educational template with each original input list. '
             'Model, requested adaptive/medium reasoning, token limit, mathematical constraints, verifier, and canonical answer interface are unchanged.', '',
             '| Level | Condition | Refusals / 1,024 | Strict accuracy % | Normalized accuracy % | Strict distinct / 8 | Normalized distinct / 8 | Strict collision % | Normalized collision % |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def percent(x):
        return '—' if x is None else f'{100*x:.2f}'
    for level, record in result['levels'].items():
        for label in ('original', 'plain'):
            m, c = record[label]['metrics'], record[label]['counts']
            lines.append(f"| {level} | {label} | {c['native_refusals']} | {percent(m['strict_accuracy'])} | "
                         f"{percent(m['normalized_accuracy'])} | {m['strict_distinct8']:.3f} | {m['normalized_distinct8']:.3f} | "
                         f"{percent(m['strict_correct_pair_collision'])} | {percent(m['normalized_correct_pair_collision'])} |")
    lines += ['', 'Plain minus original, percentage points; paired whole-prompt 95% percentile intervals:', '',
              '| Level | Native refusal difference | Strict accuracy difference | Normalized accuracy difference |',
              '|---|---:|---:|---:|']
    for level, record in result['levels'].items():
        cells = []
        for name in METRICS[:3]:
            x = record['plain_minus_original'][name]
            cells.append(f"{100*x['estimate']:+.2f} [{100*x['ci95'][0]:+.2f}, {100*x['ci95'][1]:+.2f}]")
        lines.append('| ' + level + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '| Level | Grading | Original collision-eligible prompts / 128 | Plain eligible / 128 | Eligible in both / 128 |',
              '|---|---|---:|---:|---:|']
    for level, record in result['levels'].items():
        for label in ('strict', 'normalized'):
            lines.append(f"| {level} | {label} | {record['original']['counts'][label]['collision_eligible_prompts']} | "
                         f"{record['plain']['counts'][label]['collision_eligible_prompts']} | "
                         f"{record['joint_eligibility'][label]['both_collision_eligible']} |")
    lines += ['', 'Collision pools pairs of correct answers within prompts and is undefined when no such pairs exist. '
              'All prompts contribute to accuracy and distinct-mode means. The JSON retains per-prompt eligibility and evidence hashes.', '',
              *[f'- {line}' for line in result['limitations']], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--plain', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--io-workers', type=int, default=16)
    args = parser.parse_args()
    require(args.replicates > 0, 'Replicates must be positive')
    original = authenticate_cohort(args.original, 15360, io_workers=args.io_workers)
    plain = authenticate_cohort(args.plain, 3072, io_workers=args.io_workers)
    result = compare(original, plain, args.replicates, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'paired_prompt_comparison.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (args.output / 'PYTHON_PROMPT_COMPARISON.md').write_text(markdown(result))
    print(json.dumps({'status': result['status'], 'paired_prompts': 384, 'responses_per_condition': 3072,
                      'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
