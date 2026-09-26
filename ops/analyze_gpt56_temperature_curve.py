#!/usr/bin/env python3
"""Authenticate the separate GPT-5.6 none-reasoning temperature curve, offline."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_frontier_temperature_ablation as paired

native, provider, integrity = paired.native, paired.provider, paired.integrity
sha, file_sha, require = paired.sha, paired.file_sha, paired.require
bound_file, within = paired.bound_file, paired.within
read_jsonl, identity, row_identity, unique = paired.read_jsonl, paired.identity, paired.row_identity, paired.unique
DOMAINS, LEVELS, METRICS = paired.DOMAINS, paired.LEVELS, paired.METRICS
MODEL = 'gpt-5.6-sol'
CONDITION = 'gpt56_none_temperature_curve_v1'
TEMPERATURES = (0.5, 1.0, 1.5, 2.0)
EXTENDED_TEMPERATURES = (0.0, *TEMPERATURES)
SLUGS = {0.0: 'gpt56_none_t0p0', 0.5: 'gpt56_none_t0p5', 1.0: 'gpt56_none_t1p0',
         1.5: 'gpt56_none_t1p5', 2.0: 'gpt56_none_t2p0'}


def authenticate_gate(root):
    root = Path(root).resolve()
    path = root / 'gpt56_curve_collection_gate.json'
    gate = json.loads(path.read_text())
    require(gate['schema'] == 'frontier-gpt56-temperature-gate-v1' and
            gate['status'] == 'authorized_supported_controls' and gate['model'] == MODEL and
            gate['reasoning_effort'] == 'none' and gate['temperatures'] == list(TEMPERATURES) and
            gate['total_registered_responses'] == 3840 and gate['responses_per_condition'] == 960 and
            gate['prompts_per_condition'] == 120 and gate['draws_per_prompt'] == 8 and
            gate['domains'] == 5 and gate['levels'] == list(LEVELS), 'Wrong GPT curve collection gate')
    expected_sources = {'collection_code/run_gpt56_temperature_curve.py',
        'gpt56_control_probe/probe_results.json', 'gpt56_curve_independent_design_audit.json',
        'gpt56_curve_preparation_audit.json'}
    require(set(gate['evidence']) == expected_sources, 'Missing or changed gate source inventory')
    for relative, digest in gate['evidence'].items():
        bound_file(within(root, relative), digest)
    authenticate_gate_conditions(root, gate, TEMPERATURES)
    return {'path': str(path), 'sha256': file_sha(path), 'details': gate,
            'all_four_registered_preflights_bound': True}


def authenticate_gate_conditions(root, gate, temperatures):
    conditions = unique(gate['conditions'], lambda c: c['temperature'], 'gate temperature')
    require(set(conditions) == set(temperatures), 'Missing gate temperature')
    for temperature, evidence in conditions.items():
        require(evidence['condition'] == SLUGS[temperature] and evidence['reasoning_effort'] == 'none' and
                evidence['registered_responses'] == 960, 'Gate condition differs')
        run = root / evidence['condition']
        for name in ('manifest', 'requests', 'rows'):
            suffix = '.json' if name == 'manifest' else '.jsonl'
            bound_file(run / (name + suffix), evidence[name + '_sha256'])
        bound_file(run / 'preflight_result.json', evidence['preflight_result_sha256'])
        preflight = json.loads((run / 'preflight_result.json').read_text())
        require(preflight['exit_code'] == 0 and preflight['terminal_samples'] == 1,
                'Registered preflight did not complete')
        raw_path = within(root, evidence['preflight_receipt'])
        bound_file(raw_path, evidence['preflight_receipt_sha256'])
        request = read_jsonl(run / 'requests.jsonl')[0]
        sample = json.loads((run / 'sample_receipts' / (request['sample_id'] + '.json')).read_text())
        require(identity(sample) == identity(request) and sample['request_sha256'] == request['request_sha256'] and
                within(run, sample['raw_receipt']) == raw_path, 'Preflight is not the retained first registered draw')
        raw = json.loads(raw_path.read_text())
        require(raw['http_status'] == 200 and raw['response']['status'] in ('completed', 'incomplete') and
                raw['response']['id'] == sample['response_id'], 'Preflight is not the retained terminal response')
        returned_controls({request['sample_id']: sample}, {sample['raw_receipt']: raw}, temperature)


def authenticate_extension_gate(root):
    """Authenticate the zero arm without changing the original four-arm gate."""
    root = Path(root).resolve()
    path = root / 'gpt56_zero_extension_collection_gate.json'
    gate = json.loads(path.read_text())
    require(gate['schema'] == 'frontier-gpt56-temperature-extension-gate-v1' and
            gate['status'] == 'authorized_supported_controls' and gate['model'] == MODEL and
            gate['reasoning_effort'] == 'none' and gate['temperatures'] == [0.0] and
            gate['total_registered_responses'] == gate['responses_per_condition'] == 960 and
            gate['prompts_per_condition'] == 120 and gate['draws_per_prompt'] == 8 and
            gate['domains'] == 5 and gate['levels'] == list(LEVELS), 'Wrong GPT zero extension gate')
    expected_sources = {'gpt56_curve_collection_gate.json',
        'gpt56_requested_grid_probe/probe_results.json', 'gpt56_zero_control_probe/probe_results.json',
        'collection_code/run_gpt56_zero_extension.py'}
    require(set(gate['evidence']) == expected_sources, 'Missing or changed extension gate source inventory')
    for relative, digest in gate['evidence'].items():
        bound_file(within(root, relative), digest)
    authenticate_gate_conditions(root, gate, (0.0,))
    return {'path': str(path), 'sha256': file_sha(path), 'details': gate,
            'zero_registered_preflight_bound': True}


def validate_design(cohort):
    m, rows, requests = (cohort[k] for k in ('manifest', 'rows', 'requests'))
    temperature = m['temperature']
    require(m['model'] == MODEL and m['reasoning_effort'] == 'none' and temperature in EXTENDED_TEMPERATURES and
            m['experiment_condition'] == CONDITION and m['schema'] == 'frontier-modebench-responses-v1' and
            m['sample_count'] == 8 and m['original_prompt_cohort'] is False and m['original_prompt_bytes'] is True,
            'Wrong GPT no-reasoning curve registration')
    reference = Path(m['reference_run'])
    bound_file(reference / 'manifest.json', m['reference_manifest_sha256'])
    for name in ('rows.jsonl', 'requests.jsonl'):
        bound_file(reference / name, m['reference_artifact_sha256'][name])
    require(rows == paired.selected_original_rows(read_jsonl(reference / 'rows.jsonl')),
            'Curve rows differ from fixed outcome-blind selection')
    originals = unique(read_jsonl(reference / 'requests.jsonl'), identity, 'original request')
    require(set(requests) == {(*key, i) for key in rows for i in range(8)}, 'Missing or extra curve draw slot')
    for key, request in requests.items():
        original = originals[key]
        require(original['request']['reasoning']['effort'] == 'medium', 'Original reference reasoning changed')
        expected = copy.deepcopy(original['request'])
        expected['reasoning']['effort'] = 'none'
        expected['temperature'] = temperature
        require(request['request'] == expected and request['request_sha256'] == sha(expected) and
                request['reference_request_sha256'] == original['request_sha256'] and
                request['row_sha256'] == original['row_sha256'] and
                request['temperature_condition'] == temperature,
                'Curve changed a field besides registered reasoning and temperature')
    return reference


def provider_outcomes(run, raw, raw_bodies, evidence):
    report_path = run / 'provider_outcomes.json'
    report = json.loads(report_path.read_text())
    require(report['status'] == 'complete' and report['model'] == MODEL and
            report['responses'] == report['expected_responses'] == 960 and
            report['native_protocol'] == 'responses' and Path(report['source_run']).resolve() == run,
            'Incomplete or wrong provider outcome report')
    for name in ('manifest.json', 'samples.jsonl', 'completion_audit.json', 'evidence_file_sha256.json'):
        source = report['sources'][name]
        require(Path(source['path']).resolve() == run / name, 'Provider sources use another condition')
        bound_file(run / name, source['sha256'])
    auditor = Path(report['audit_source_path'])
    bound_file(auditor, report['audit_source_sha256'])
    bound_file(auditor.with_name('audit_hosted_modebench_completion.py'), report['audit_helper_source_sha256'])
    source = report['sample_outcomes']
    require(Path(source['path']).resolve() == run / 'provider_outcome_samples.jsonl', 'Wrong provider sidecar')
    bound_file(source['path'], source['sha256'])
    outcomes = unique(read_jsonl(source['path']), identity, 'provider outcome')
    require(set(outcomes) == set(raw) and source['records'] == 960, 'Wrong provider outcome inventory')
    for key, sample in raw.items():
        outcome = outcomes[key]
        require(all(outcome.get(k) == sample[k] for k in
                    ('sample_id', 'row_sha256', 'request_sha256', 'raw_receipt', 'raw_receipt_sha256', 'response_id')),
                'Provider sidecar lost native sample identity')
        require(outcome['raw_receipt_file_sha256'] == evidence[sample['raw_receipt']], 'Provider receipt binding differs')
        classified = provider.classify_native(raw_bodies[sample['raw_receipt']]['response'], 'responses')
        require(all(outcome.get(k) == v for k, v in classified.items()), 'Provider outcomes differ from native response')
    require(report['totals'] == provider.aggregate(list(outcomes.values())), 'Provider totals differ')
    for level in LEVELS:
        for domain in DOMAINS:
            require(report['cells'][f'level{level}/{domain}'] == provider.aggregate(
                [v for k, v in outcomes.items() if k[:2] == (level, domain)]), 'Provider cell totals differ')
    return outcomes


def returned_controls(raw, bodies, temperature):
    returned = Counter()
    for sample in raw.values():
        body = bodies[sample['raw_receipt']]['response']
        require(body.get('temperature') == temperature and
                (body.get('reasoning') or {}).get('effort') == 'none',
                'Returned temperature or reasoning differs')
        returned[json.dumps({k: body.get(k) for k in ('temperature', 'reasoning', 'top_p')}, sort_keys=True)] += 1
    return dict(returned)


def validate_control_audit(run, raw, bodies, temperature):
    path = run / 'native_control_audit.json'
    audit = json.loads(path.read_text())
    require(audit['status'] == 'pass' and audit['responses'] == 960 and audit['api_calls'] == 0 and
            audit['requested_temperature'] == temperature and audit['requested_reasoning_effort'] == 'none',
            'Incomplete or wrong native-control audit')
    bound_file(run / 'manifest.json', audit['manifest_sha256'])
    bound_file(run / 'samples.jsonl', audit['samples_sha256'])
    bound_file(run / 'analysis_code/ops/audit_gpt56_temperature_controls.py', audit['auditor_sha256'])
    native_bodies = [bodies[s['raw_receipt']]['response'] for s in raw.values()]
    for field, values in (
        ('returned_temperature_counts', [str(b.get('temperature')) for b in native_bodies]),
        ('returned_reasoning_effort_counts', [(b.get('reasoning') or {}).get('effort') for b in native_bodies]),
        ('returned_top_p_counts', [str(b.get('top_p')) for b in native_bodies]),
        ('returned_model_counts', [str(b.get('model')) for b in native_bodies])):
        require(audit[field] == dict(Counter(values)), 'Native-control audit counts differ from receipts')
    return bound_file(path, file_sha(path))


def authenticate_condition(run, io_workers=16):
    run = Path(run).resolve()
    inv = native.load_inventory(run, expected_samples=960)
    m = inv['manifest']
    requests = unique(inv['requests'], identity, 'request')
    cohort = {'directory': run, 'manifest': m, 'rows': inv['rows'], 'requests': requests}
    validate_design(cohort)
    require(not inv['grouped'] and len(inv['groups']) == 960, 'Require stateless independent Responses requests')
    condition = json.loads((run / 'temperature_condition.json').read_text())
    require(condition['condition'] == CONDITION and condition['reasoning_effort'] == 'none' and
            condition['requested_temperature'] == m['temperature'] and
            condition['selection_seed'] == paired.SELECTION_SEED and condition['selection_uses_outcomes'] is False and
            condition['changed_fields_vs_original'] == ['reasoning.effort', 'temperature'] and
            condition['changed_fields_within_curve'] == ['temperature'] and
            condition['row_sha256'] == [sha(r) for r in read_jsonl(run / 'rows.jsonl')],
            'Frozen curve condition metadata differs')
    summary = json.loads((run / 'summary.json').read_text())
    require(summary['status'] == 'complete' and summary['received_responses'] == summary['expected_responses'] == 960 and
            summary['complete_prompts'] == 120 and summary['models_returned'] == {MODEL: 960} and
            summary['run_configuration']['model'] == MODEL, 'Incomplete or wrong curve summary')
    audit = json.loads((run / 'completion_audit.json').read_text())
    require(audit['status'] == 'pass' and audit['model'] == MODEL and audit['saved_samples'] == 960 and
            audit['expected_responses'] == audit['unique_response_ids'] == audit['unique_response_choice_ids'] == 960,
            'Incomplete native receipt audit')
    evidence = json.loads((run / 'evidence_file_sha256.json').read_text())
    require(sha(evidence) == audit['evidence_inventory_sha256'], 'Native evidence inventory changed')
    for name in ('manifest.json', 'rows.jsonl', 'datasets.json', 'requests.jsonl', 'samples.jsonl'):
        bound_file(run / name, evidence.get(name))
    for name, digest in summary['input_sha256'].items():
        path = within(run, name)
        if name == 'errors.jsonl' and digest is None:
            require(not path.exists(), 'Unbound error log appeared after summary')
        else:
            bound_file(path, digest)
    raw = unique(read_jsonl(run / 'samples.jsonl'), identity, 'sample')
    require(set(raw) == set(requests) and len({s['response_id'] for s in raw.values()}) == 960,
            'Missing or reused curve sample')
    for key, sample in raw.items():
        require(all(sample.get(name) == requests[key][name] for name in
                    ('sample_id', 'row_sha256', 'request_sha256', 'temperature_condition')), 'Sample/request identity differs')
        integrity.grade_valid(sample)
        bound_file(within(run, sample['raw_receipt']), evidence.get(sample['raw_receipt']))
        atomic = 'sample_receipts/' + sample['sample_id'] + '.json'
        bound_file(within(run, atomic), evidence.get(atomic))
        require(json.loads((run / atomic).read_text()) == sample, 'Atomic sample differs from export')
    bodies = native.validate_native_records(inv, list(raw.values()), io_workers)
    returned = returned_controls(raw, bodies, m['temperature'])
    control_audit = validate_control_audit(run, raw, bodies, m['temperature'])
    primary, grading_audit = integrity.validate_primary(run, summary, raw, evidence, 960)
    normalized = paired.normalized_grades(run, summary, raw, primary)
    outcomes = provider_outcomes(run, raw, bodies, evidence)
    sources = {name: bound_file(run / name, file_sha(run / name)) for name in
        ('manifest.json', 'temperature_condition.json', 'capability_probe_results.json', 'rows.jsonl', 'requests.jsonl',
         'samples.jsonl', 'summary.json', 'completion_audit.json', 'evidence_file_sha256.json',
         'primary_python_regrade_audit.json', 'audited_primary_samples.jsonl', 'normalized_samples.jsonl',
         'provider_outcomes.json', 'provider_outcome_samples.jsonl')}
    sources['native_control_audit.json'] = control_audit
    return cohort | {'summary': summary, 'raw': raw, 'primary': primary, 'normalized': normalized,
        'grading_audit': grading_audit, 'outcomes': outcomes, 'sources': sources, 'returned_controls': dict(returned)}


def validate_legacy_medium_primary(run, summary, raw, evidence, expected):
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
    require(audit['python_samples'] == sum(s['domain'] == 'python_factors' for s in raw.values()),
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
    require('audit_helper_source' not in audit and
            audit['schema'] == 'frontier-modebench-primary-python-serial-audit-v1' and
            audit['protocol']['all_python_positives_and_negatives_regraded'] is True and
            audit['protocol']['api_calls'] == 0 and
            audit['protocol']['strict_text_and_raw_receipts_changed'] is False and
            audit['protocol']['unchanged_frozen_executable_grader'] is True,
            'Wrong standalone legacy Python audit protocol')
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
        integrity.grade_valid(sample)
        require(all(sample.get(name) == value for name, value in original.items() if name not in integrity.GRADES),
                'Audited primary changed a non-grading field')
        receipt = receipts[sample['sample_id']]
        require(sample['raw_strict_receipt'] == receipt['path'] and
                sample['raw_strict_receipt_sha256'] == receipt['sha256'] and
                all(sample['raw_strict_' + name] == original[name] for name in integrity.GRADES),
                'Audited primary lost its original strict receipt binding')
    return primary, audit


def normalized_medium_subset(run, summary, raw, primary):
    secondary = summary['normalized_secondary']
    for path, digest in ((run / 'normalized_samples.jsonl', secondary['cache_sha256']),
                         (secondary['normalization_source_path'], secondary['normalization_source_sha256']),
                         (secondary['frozen_grader_contract_path'], secondary['frozen_grader_contract_sha256']),
                         (run / 'secondary_initial15_audit.json', secondary['initial_rule_audit_sha256'])):
        bound_file(path, digest)
    initial = json.loads((run / 'secondary_initial15_audit.json').read_text())
    require(initial['normalizer']['sha256'] == secondary['normalization_source_sha256'],
            'Normalizer differs from its original frozen development audit')
    needed = {sha({**raw[k], **{name: sample[name] for name in integrity.GRADES}})
              for k, sample in primary.items()}
    cache = {}
    for item in read_jsonl(run / 'normalized_samples.jsonl'):
        if item.get('normalization_source_sha256') != secondary['normalization_source_sha256']:
            continue
        key = item['strict_receipt_sha256']
        if key not in needed:
            continue
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


def authenticate_medium_reference(run, selected_rows, io_workers=16):
    run = Path(run).resolve()
    inv = native.load_inventory(run, expected_samples=15360)
    m = inv['manifest']
    require(m['model'] == MODEL and m['reasoning_effort'] == 'medium', 'Wrong original medium reference')
    summary = json.loads((run / 'summary.json').read_text())
    require(summary['status'] == 'complete' and summary['received_responses'] == summary['expected_responses'] == 15360 and
            summary['models_returned'] == {MODEL: 15360}, 'Original reference summary is incomplete')
    audit = json.loads((run / 'completion_audit.json').read_text())
    require(audit['status'] == 'pass' and audit['saved_samples'] == audit['expected_responses'] ==
            audit['unique_response_ids'] == 15360, 'Original reference native audit is incomplete')
    evidence = json.loads((run / 'evidence_file_sha256.json').read_text())
    require(sha(evidence) == audit['evidence_inventory_sha256'], 'Original evidence inventory changed')
    for name in ('manifest.json', 'rows.jsonl', 'datasets.json', 'requests.jsonl', 'samples.jsonl'):
        bound_file(run / name, evidence.get(name))
    for name, digest in summary['input_sha256'].items():
        path = within(run, name)
        if name == 'errors.jsonl' and digest is None:
            require(not path.exists(), 'Unbound original error log appeared')
        else:
            bound_file(path, digest)
    require(selected_rows == paired.selected_original_rows(read_jsonl(run / 'rows.jsonl')),
            'Original reference uses different selected rows')
    all_raw = unique(read_jsonl(run / 'samples.jsonl'), identity, 'original sample')
    all_requests = unique(inv['requests'], identity, 'original request')
    keys = {(*key, i) for key in selected_rows for i in range(8)}
    require(keys.issubset(all_raw) and keys.issubset(all_requests), 'Original reference misses selected slots')
    raw, requests = ({k: values[k] for k in keys} for values in (all_raw, all_requests))
    for key, sample in raw.items():
        require('temperature' not in requests[key]['request'] and
                requests[key]['request']['reasoning']['effort'] == 'medium', 'Original requested controls differ')
        atomic = 'sample_receipts/' + sample['sample_id'] + '.json'
        bound_file(within(run, atomic), evidence.get(atomic))
        bound_file(within(run, sample['raw_receipt']), evidence.get(sample['raw_receipt']))
        require(json.loads((run / atomic).read_text()) == sample, 'Original atomic receipt differs')
    bodies = native.validate_native_records(inv, list(raw.values()), io_workers)
    returned = Counter()
    outcomes = {}
    for key, sample in raw.items():
        body = bodies[sample['raw_receipt']]['response']
        require(body.get('temperature') == 1.0 and (body.get('reasoning') or {}).get('effort') == 'medium',
                'Original reference did not return medium reasoning and temperature 1.0')
        returned[json.dumps({k: body.get(k) for k in ('temperature', 'reasoning', 'top_p')}, sort_keys=True)] += 1
        outcomes[key] = provider.classify_native(body, 'responses')
    all_primary, grading_audit = validate_legacy_medium_primary(run, summary, all_raw, evidence, 15360)
    primary = {k: all_primary[k] for k in keys}
    normalized = normalized_medium_subset(run, summary, raw, primary)
    reconciliation = summary['normalized_secondary']['python_cache_reconciliation']
    require(reconciliation['status'] == 'complete', 'Legacy normalization reconciliation is incomplete')
    bound_file(reconciliation['path'], reconciliation['sha256'])
    sources = {name: bound_file(run / name, file_sha(run / name)) for name in
        ('manifest.json', 'rows.jsonl', 'requests.jsonl', 'samples.jsonl', 'summary.json',
         'completion_audit.json', 'evidence_file_sha256.json', 'primary_python_regrade_audit.json',
         'audited_primary_samples.jsonl', 'normalized_samples.jsonl')}
    sources['normalization_python_reconciliation.json'] = bound_file(reconciliation['path'], reconciliation['sha256'])
    return {'directory': run, 'manifest': m, 'rows': selected_rows, 'requests': requests, 'raw': raw,
        'primary': primary, 'normalized': normalized, 'grading_audit': grading_audit,
        'summary': summary, 'outcomes': outcomes, 'sources': sources, 'returned_controls': dict(returned)}


def outcome_categories(cohort):
    totals, cells = Counter(), {}
    for key, sample in cohort['raw'].items():
        outcome = cohort['outcomes'][key]
        details = sample.get('incomplete_details') or {}
        if outcome['refusal']: category = 'native_refusal'
        elif outcome['content_filtered']: category = 'native_content_filter'
        elif sample.get('response_status') == 'incomplete' and details.get('reason') == 'max_output_tokens':
            category = 'token_limit'
        elif outcome['answer_text_empty']: category = 'empty_answer'
        elif cohort['primary'][key]['verified']: category = 'strict_success'
        elif cohort['normalized'][key]['verified']: category = 'formatting_recovered'
        else: category = 'nonempty_verifier_rejection_after_normalization'
        totals[category] += 1
        cells.setdefault(f'level{key[0]}/{key[1]}', Counter())[category] += 1
    return {'totals': dict(totals), 'cells': {k: dict(v) for k, v in cells.items()},
        'definition': 'Disjoint diagnostic categories, in listed priority order: native refusal, filter, token limit, empty answer, strict success, frozen formatting recovery, remaining nonempty verifier rejection. This does not identify a semantic failure mechanism.'}


def validate_curve(cohorts):
    temperatures = tuple(sorted(cohorts))
    require(temperatures in (TEMPERATURES, EXTENDED_TEMPERATURES),
            'Curve must contain all four registered temperatures, optionally with the zero extension')
    reference = cohorts[1.0]
    ids = set()
    for temperature in temperatures:
        cohort = cohorts[temperature]
        validate_design(cohort)
        require(cohort['manifest']['temperature'] == temperature and cohort['rows'] == reference['rows'],
                'Curve conditions use different rows or temperatures')
        require(cohort['manifest']['reference_manifest_sha256'] == reference['manifest']['reference_manifest_sha256'],
                'Curve conditions use different original references')
        for key, item in cohort['requests'].items():
            require({k: v for k, v in item['request'].items() if k != 'temperature'} ==
                    {k: v for k, v in reference['requests'][key]['request'].items() if k != 'temperature'},
                    'Curve payloads differ beyond temperature')
        new = {s['response_id'] for s in cohort['raw'].values()}
        require(not ids.intersection(new), 'A response was reused between curve temperatures')
        ids.update(new)
        for field in ('normalization_source_sha256', 'initial_rule_audit_sha256', 'frozen_grader_contract_sha256'):
            require(cohort['summary']['normalized_secondary'][field] == reference['summary']['normalized_secondary'][field],
                    'Curve conditions use different normalization contracts')
        require({k: v['sha256'] for k, v in cohort['grading_audit']['frozen_grader_modules'].items()} ==
                {k: v['sha256'] for k, v in reference['grading_audit']['frozen_grader_modules'].items()},
                'Curve conditions use different executable graders')
    return temperatures


def prompt_records(cohort, grading):
    # A temporary analysis view maps the native Responses limit reason to the
    # existing shared whole-prompt counter. Source receipts remain unchanged.
    raw_view = {}
    for key, sample in cohort['raw'].items():
        details = sample.get('incomplete_details') or {}
        is_limit = sample.get('response_status') == 'incomplete' and details.get('reason') == 'max_output_tokens'
        raw_view[key] = sample | {'stop_reason': 'length' if is_limit else sample.get('response_status')}
    records = paired.prompt_records(cohort | {'raw': raw_view}, grading)
    for record in records:
        key = record['level'], record['domain'], record['row_index']
        record['empty_answers'] = sum(cohort['outcomes'][(*key, i)]['answer_text_empty'] for i in range(8))
    return records


def describe_group(points, boots, keys):
    return paired.describe(np.mean([points[k] for k in keys], axis=0),
                           np.mean([boots[k] for k in keys], axis=0))


def groups(points, boots):
    result = {}
    for name, domains in [('five_domain_macro', DOMAINS), ('four_finite_support_domain_macro', DOMAINS[1:])]:
        keys = [k for k in points if k.split('/')[1] in domains]
        by_level = {l: [k for k in keys if k.startswith(f'level{l}/')] for l in LEVELS}
        level_points = {l: np.mean([points[k] for k in ks], axis=0) for l, ks in by_level.items()}
        level_boots = {l: np.mean([boots[k] for k in ks], axis=0) for l, ks in by_level.items()}
        result[name] = {'overall': describe_group(points, boots, keys), 'levels': {
            str(l): paired.describe(level_points[l], level_boots[l]) for l in LEVELS},
            'level_contrasts': {f'L{b}-L{a}': paired.describe(level_points[b] - level_points[a],
                level_boots[b] - level_boots[a]) for a, b in ((1, 2), (1, 3), (2, 3))}}
    return result


def analyze_curve(cohorts, indices):
    grid = validate_curve(cohorts)
    low, high = grid[0], grid[-1]
    output = {'model': MODEL, 'reasoning_effort': 'none', 'temperatures': list(grid),
        'total_registered_responses': len(grid) * 960, 'responses_per_condition': 960,
        'prompts_per_condition': 120, 'draws_per_prompt': 8,
        'conditions': {str(t): {'directory': str(c['directory']), 'source_sha256': c['sources'],
            'returned_controls': c['returned_controls'], 'outcome_categories': outcome_categories(c)}
            for t, c in cohorts.items()}, 'analyses': {}}
    for grading in ('strict', 'normalized_secondary'):
        records = {t: prompt_records(c, grading) for t, c in cohorts.items()}
        temperatures, contrasts, arm_points, arm_boots = {}, {}, {}, {}
        for temperature in grid:
            points, boots, delta_points, delta_boots, cells, delta_cells = {}, {}, {}, {}, {}, {}
            for level in LEVELS:
                for domain in DOMAINS:
                    key = f'level{level}/{domain}'
                    baseline = [r for r in records[1.0] if (r['level'], r['domain']) == (level, domain)]
                    current = [r for r in records[temperature] if (r['level'], r['domain']) == (level, domain)]
                    eligibility = paired.joint_eligibility(baseline, current)
                    joint = {'baseline_temperature': 1.0, 'target_temperature': temperature,
                        'both_eligible': eligibility['both_eligible'],
                        'only_t1p0_eligible': eligibility['only_t1p0_eligible'],
                        'only_target_eligible': eligibility['only_t1p5_eligible'],
                        'neither_eligible': eligibility['neither_eligible']}
                    point, boot = paired.paired_bootstrap(paired.matrix(baseline), paired.matrix(current), indices[level, domain])
                    points[key], boots[key] = point['t1p5'], boot['t1p5']
                    delta_points[key], delta_boots[key] = point['t1p5_minus_t1p0'], boot['t1p5_minus_t1p0']
                    source = cohorts[temperature]['summary']
                    if grading != 'strict': source = source['normalized_secondary']
                    for j, metric in enumerate(paired.SOURCE_METRICS):
                        expected = source['cells'][key]['metrics'][metric]['estimate']
                        require((expected is None and not np.isfinite(points[key][j])) or
                                (expected is not None and abs(expected - points[key][j]) < 1e-12),
                                'Reconstructed curve cell differs from frozen summary')
                    cells[key] = paired.describe(points[key], boots[key]) | {'counts': paired.counts(current),
                        'empty_answers': sum(r['empty_answers'] for r in current)}
                    delta_cells[key] = paired.describe(delta_points[key], delta_boots[key]) | {'joint_eligibility': joint}
            arm_points[temperature], arm_boots[temperature] = points, boots
            temperatures[str(temperature)] = {'cells': cells, 'groups': groups(points, boots),
                'counts': paired.counts(records[temperature]), 'prompt_records': records[temperature]}
            contrasts[str(temperature)] = {'cells': delta_cells, 'groups': groups(delta_points, delta_boots)}
        endpoint_points = {k: arm_points[high][k] - arm_points[low][k] for k in arm_points[low]}
        endpoint_boots = {k: arm_boots[high][k] - arm_boots[low][k] for k in arm_boots[low]}
        endpoint_cells = {k: paired.describe(endpoint_points[k], endpoint_boots[k]) for k in endpoint_points}
        for key, cell in endpoint_cells.items():
            level, domain = key.split('/')
            a, b = ([r for r in records[t] if r['level'] == int(level[5:]) and r['domain'] == domain] for t in (low, high))
            joint = paired.joint_eligibility(a, b)
            cell['joint_eligibility'] = {'both_eligible': joint['both_eligible'],
                f'only_t{low:.1f}_eligible'.replace('.', 'p'): joint['only_t1p0_eligible'],
                f'only_t{high:.1f}_eligible'.replace('.', 'p'): joint['only_t1p5_eligible'],
                'neither_eligible': joint['neither_eligible']}
        output['analyses'][grading] = {'temperatures': temperatures, 'paired_contrasts_vs_t1p0': contrasts,
            'paired_endpoint_contrast': {'comparison': f'T{high:.1f}-T{low:.1f}', 'cells': endpoint_cells,
                'groups': groups(endpoint_points, endpoint_boots)}}
    output['validation'] = {'all_native_receipts_authenticated': True,
        'authenticated_native_receipt_count': len(grid) * 960,
        'same_120_prompts_all_eight_slots': True, 'only_temperature_varies_within_curve': True,
        'all_returned_controls_match_requests': True, 'source_summaries_reconstructed': True}
    if grid == TEMPERATURES:
        output['validation']['all_3840_native_receipts_authenticated'] = True
    return output


def analyze_medium_reference(cohort, curve_cohorts, indices):
    curve_ids = {r['response_id'] for c in curve_cohorts.values() for r in c['raw'].values()}
    require(not curve_ids.intersection(r['response_id'] for r in cohort['raw'].values()),
            'Original reference response was reused in the curve')
    for field in ('normalization_source_sha256', 'initial_rule_audit_sha256', 'frozen_grader_contract_sha256'):
        require(cohort['summary']['normalized_secondary'][field] ==
                curve_cohorts[1.0]['summary']['normalized_secondary'][field],
                'Medium reference and curve use different frozen normalization contracts')
    require({k: v['sha256'] for k, v in cohort['grading_audit']['frozen_grader_modules'].items()} ==
            {k: v['sha256'] for k, v in curve_cohorts[1.0]['grading_audit']['frozen_grader_modules'].items()},
            'Medium reference and curve use different executable graders')
    result = {'model': MODEL, 'reasoning_effort': 'medium', 'requested_temperature': None,
        'returned_temperature': 1.0, 'connect_to_temperature_curve': False,
        'directory': str(cohort['directory']), 'source_sha256': cohort['sources'],
        'returned_controls': cohort['returned_controls'], 'analyses': {},
        'limitations': ['Original requests omitted temperature; native responses returned 1.0.',
            'Different reasoning configuration and collection time; this is not a causal reasoning comparison.'],
        'outcome_categories': outcome_categories(cohort)}
    for grading in ('strict', 'normalized_secondary'):
        records = prompt_records(cohort, grading)
        points, boots, cells = {}, {}, {}
        for level in LEVELS:
            for domain in DOMAINS:
                key = f'level{level}/{domain}'
                subset = [r for r in records if (r['level'], r['domain']) == (level, domain)]
                values = paired.matrix(subset)
                point, boot = paired.paired_bootstrap(values, values, indices[level, domain])
                points[key], boots[key] = point['t1p0'], boot['t1p0']
                cells[key] = paired.describe(points[key], boots[key]) | {'counts': paired.counts(subset)}
        result['analyses'][grading] = {'cells': cells, 'groups': groups(points, boots),
            'counts': paired.counts(records), 'prompt_records': records}
    return result


def markdown(result):
    grid = result['temperatures']
    endpoint_label = f'T{grid[-1]:.1f}−T{grid[0]:.1f}'
    lines = ['# GPT-5.6 temperature curve with reasoning set to none', '',
        f"This separate exploratory condition contains {len(grid) * 960:,} saved responses: {len(grid)} requested temperatures "
        f"({', '.join(f'{t:.1f}' for t in grid)}), 120 fixed held-out prompts each, and eight independent draws per prompt. "
        'Native responses echo every requested temperature and reasoning=none. Provider echoes are metadata, '
        'not independent observation of internal sampling.', '',
        'The original full GPT cohort used medium reasoning. Changing temperature at that setting was rejected '
        'except at 1.0, so this curve uses none throughout. It must not be described as the original medium-reasoning '
        'deployment curve or connected to a medium-reasoning reference point.', '',
        'All invalid, refused and truncated answers remain in accuracy and distinct8 denominators. Collision '
        'conditions on correct pairs. Equal-cell macros propagate undefined members. Pointwise 95% intervals '
        'resample whole prompt groups within each of the 15 cells; all eight draws stay together. Countdown '
        'has no finite uniform-support reference.', '']
    if 'matched_medium_reference' in result:
        reference = result['matched_medium_reference']['analyses']['normalized_secondary']['groups']['five_domain_macro']['overall']
        lines += [f"The same 120 original prompts under medium reasoning have "
            f"{paired.fmt(reference['accuracy'])}% normalized accuracy and "
            f"{paired.fmt(reference['distinct8'], 1)} distinct correct modes per eight draws. "
            'Those original requests omitted temperature; all matched receipts returned T=1.0. '
            'This separate reference was collected at a different time and is never part of the connected curve.', '']
    for grading, analysis in result['analyses'].items():
        lines += ['## ' + grading.replace('_', ' '), '',
                  '| Scope | Temperature | Accuracy % [95% CI] | Distinct correct modes / 8 [95% CI] | Collision % [95% CI] |',
                  '|---|---:|---:|---:|---:|']
        for scope in ('1', '2', '3', 'overall'):
            for temperature in grid:
                group = analysis['temperatures'][str(temperature)]['groups']['five_domain_macro']
                stats = group['overall'] if scope == 'overall' else group['levels'][scope]
                lines.append(f"| {scope} | {temperature:.1f} | {paired.fmt(stats['accuracy'], interval=True)} | "
                    f"{paired.fmt(stats['distinct8'], 1, True)} | {paired.fmt(stats['collision'], interval=True)} |")
        lines += ['', f'| Scope | {endpoint_label} accuracy pp [95% CI] | {endpoint_label} modes [95% CI] | {endpoint_label} collision pp [95% CI] |',
                  '|---|---:|---:|---:|']
        endpoint = analysis['paired_endpoint_contrast']['groups']['five_domain_macro']
        for scope in ('1', '2', '3', 'overall'):
            stats = endpoint['overall'] if scope == 'overall' else endpoint['levels'][scope]
            lines.append(f"| {scope} | {paired.fmt(stats['accuracy'], interval=True)} | "
                f"{paired.fmt(stats['distinct8'], 1, True)} | {paired.fmt(stats['collision'], interval=True)} |")
        lines += ['', '| Cell | Temperature | Accuracy % | Modes / 8 | Collision % | Uniform % | Eligible prompts | Refusals | Truncations |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for level in LEVELS:
            for domain in DOMAINS:
                key = f'level{level}/{domain}'
                for temperature in grid:
                    cell = analysis['temperatures'][str(temperature)]['cells'][key]
                    count = cell['counts']
                    lines.append(f"| {key} | {temperature:.1f} | {paired.fmt(cell['accuracy'])} | "
                        f"{paired.fmt(cell['distinct8'], 1)} | {paired.fmt(cell['collision'])} | "
                        f"{paired.fmt(cell['uniform_collision'])} | {count['collision_eligible_prompts']} | "
                        f"{count['native_refusals']} | {count['truncated_responses']} |")
        lines.append('')
    lines += ['## Retained outcome categories', '',
        '| Temperature | Strict successes | Formatting recoveries | Remaining nonempty verifier rejections | Native refusals | Filters | Token limits | Empty answers |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for temperature in grid:
        counts = result['conditions'][str(temperature)]['outcome_categories']['totals']
        values = [counts.get(k, 0) for k in ('strict_success', 'formatting_recovered',
            'nonempty_verifier_rejection_after_normalization', 'native_refusal', 'native_content_filter',
            'token_limit', 'empty_answer')]
        lines.append('| ' + str(temperature) + ' | ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', 'These diagnostic categories are disjoint and retain every answer. A remaining verifier rejection '
        'does not by itself distinguish incorrect mathematics from an unsupported output form. Per-domain and '
        'per-level category counts are preserved in the JSON.', '']
    lines += ['Only eight prompts per cell make this an exploratory sensitivity analysis. Intervals are not '
        'multiplicity adjusted; a zero-width empirical interval does not prove a population equality. '
        'When collision is undefined in a bootstrap replicate, its interval uses only finite complete-macro '
        'replicates, with their number retained in defined_replicates; sparse correctness limits that inference. '
        'The JSON preserves all paired contrasts against T=1.0, source digests, and per-prompt counts.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/frontier_temperature_20260911'))
    parser.add_argument('--replicates', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--io-workers', type=int, default=16)
    parser.add_argument('--include-zero', action='store_true', help='Authenticate the separate zero extension gate and add its complete condition')
    parser.add_argument('--output', type=Path, help='Report output stem; defaults to a separate WITH_ZERO report for the extension')
    args = parser.parse_args()
    require(args.replicates > 0, 'Replicates must be positive')
    gate = authenticate_gate(args.root)
    extension_gate = authenticate_extension_gate(args.root) if args.include_zero else None
    grid = EXTENDED_TEMPERATURES if args.include_zero else TEMPERATURES
    cohorts = {t: authenticate_condition(args.root / SLUGS[t], args.io_workers) for t in grid}
    rng = np.random.default_rng(args.seed)
    indices = {(l, d): rng.integers(0, 8, (args.replicates, 8)) for l in LEVELS for d in DOMAINS}
    result = analyze_curve(cohorts, indices)
    reference = authenticate_medium_reference(cohorts[1.0]['manifest']['reference_run'], cohorts[1.0]['rows'], args.io_workers)
    result['matched_medium_reference'] = analyze_medium_reference(reference, cohorts, indices)
    result.update(schema='gpt56-none-temperature-curve-v2' if args.include_zero else 'gpt56-none-temperature-curve-v1',
        status='complete', collection_gate=gate,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        bootstrap={'replicates': args.replicates, 'seed': args.seed,
                   'unit': 'Whole eight-draw prompts, paired between temperatures and stratified by domain and level.',
                   'individual_draw_pairing': False, 'interval': 'Pointwise 95% percentile; no multiplicity adjustment.',
                   'undefined': 'Undefined cells propagate into macros; percentile intervals use only finite complete-macro replicates, whose counts are reported.'},
        analysis_sources={p.name: bound_file(p, file_sha(p)) for p in map(Path,
            (__file__, paired.__file__, native.__file__, provider.__file__, integrity.__file__))})
    if extension_gate is not None:
        result['extension_collection_gate'] = extension_gate
        result['validation']['zero_extension_gate_authenticated'] = True
    output = args.output or args.root / ('GPT56_TEMPERATURE_CURVE_WITH_ZERO' if args.include_zero else 'GPT56_TEMPERATURE_CURVE')
    output.parent.mkdir(parents=True, exist_ok=True)
    native.atomic(output.with_suffix('.json'), result)
    output.with_suffix('.md').write_text(markdown(result))
    print(json.dumps({'status': 'complete', 'model': MODEL, 'reasoning_effort': 'none',
                      'responses': len(grid) * 960, 'output': str(output.with_suffix('.json'))}))


if __name__ == '__main__':
    main()
