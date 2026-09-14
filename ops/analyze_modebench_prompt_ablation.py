#!/usr/bin/env python3
"""Offline paired analysis of the frozen ModeBench prompt-hint ablation.

All registered eight-draw slots, including incorrect and truncated responses,
are required. The resampling unit is an entire paired prompt, stratified by
level and domain. This script never calls a model or overwrites original response/grade receipts.
Optional offline modes write separately audited grading sidecars.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

CONDITION = 'prompt_hint_ablation_v1'
SEED = 20260911
DRAWS = 8
PROMPTS_PER_CELL = 32
DOMAINS = ('python_factors', 'mathir', 'pantry_plan')
LEVELS = (2, 3)
ARMS = ('original', 'neutral')
CONTRAST = 'neutral_minus_original'
GRADINGS = ('strict', 'normalized_secondary')
METRICS = ('pass1', 'pass8', 'distinct8', 'b8', 'correct_pair_collision',
           'uniform_correct_pair_collision', 'correct_pair_collision_excess_uniform',
           'uniform_expected_distinct_given_correct')
DOMAIN_LABELS = {'python_factors': 'Python factors', 'mathir': 'MathIR', 'pantry_plan': 'Pantry'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def identity(row):
    require(isinstance(row['level'], int) and not isinstance(row['level'], bool), 'Invalid level')
    require(isinstance(row['row_index'], int) and not isinstance(row['row_index'], bool), 'Invalid row index')
    return row['level'], row['domain'], row['row_index']


def sample_identity(row):
    index = row['sample_index']
    require(isinstance(index, int) and not isinstance(index, bool) and index in range(DRAWS),
            'Invalid eight-draw sample index')
    return (*identity(row), index)


def unique(records, key, label):
    result = {}
    for row in records:
        k = key(row)
        require(k not in result, 'Duplicate ' + label + ': ' + str(k))
        result[k] = row
    return result


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_sha(path)}


def bound_file(path, digest):
    require(isinstance(digest, str) and len(digest) == 64 and Path(path).is_file()
            and file_sha(path) == digest, 'File digest mismatch: ' + str(path))
    return binding(path)


def grade_valid(grade):
    require(isinstance(grade.get('verified'), bool), 'Every draw needs a boolean verified grade')
    require(grade['verified'] == (grade.get('canonical_key') is not None),
            'Success and canonical key are inconsistent')
    if grade['verified']:
        sha(grade['canonical_key'])


def prompt_statistics(row, samples):
    """Empirical P8, distinct correct keys D8, and B8=D8-P8 per prompt."""
    require(len(samples) == DRAWS and {s['sample_index'] for s in samples} == set(range(DRAWS)),
            'Prompt statistics require all eight unique slots')
    for sample in samples:
        grade_valid(sample)
    modes = Counter(sha(s['canonical_key']) for s in samples if s['verified'])
    correct, distinct = sum(modes.values()), len(modes)
    pairs = correct * (correct - 1) // 2
    collisions = sum(n * (n - 1) // 2 for n in modes.values())
    metadata = row.get('metadata', {})
    support = None if metadata.get('support_is_open') else metadata.get('answer_mode_count')
    require(support is None or (isinstance(support, int) and not isinstance(support, bool) and support > 0),
            'Invalid certified support')
    require(support is None or distinct <= support, 'Observed keys exceed certified support')
    uniform_distinct = None if support is None else (
        float(correct > 0) if support == 1 else support * -math.expm1(correct * math.log1p(-1 / support)))
    return {'level': row['level'], 'domain': row['domain'], 'row_index': row['row_index'],
            'row_sha256': sha(row), 'responses': DRAWS, 'correct_draws': correct,
            'pass8': int(correct > 0), 'distinct8': distinct, 'b8': distinct - int(correct > 0),
            'correct_pairs': pairs, 'colliding_correct_pairs': collisions,
            'collision_eligible': correct >= 2, 'certified_support_count': support,
            'uniform_expected_colliding_correct_pairs': pairs / support if support else None,
            'uniform_expected_distinct_given_correct': uniform_distinct,
            'failed_draws': DRAWS - correct,
            'truncated_draws': sum(s.get('stop_reason') in ('length', 'max_tokens') or
                                   s.get('response_status') == 'incomplete' for s in samples),
            'response_status_counts': dict(Counter(s.get('response_status', 'unknown') for s in samples)),
            'stop_reason_counts': dict(Counter(str(s.get('stop_reason', 'unknown')) for s in samples))}


def matrix(records):
    return np.asarray([[1, r['correct_draws'], r['pass8'], r['distinct8'],
                        r['correct_pairs'], r['colliding_correct_pairs'],
                        r['uniform_expected_colliding_correct_pairs'] or 0,
                        r['correct_pairs'] if r['certified_support_count'] else 0,
                        r['colliding_correct_pairs'] if r['certified_support_count'] else 0,
                        int(r['certified_support_count'] is not None),
                        r['uniform_expected_distinct_given_correct'] or 0]
                       for r in records], dtype=float)


def rates(sums):
    s = np.asarray(sums, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.stack((s[..., 1] / (DRAWS * s[..., 0]), s[..., 2] / s[..., 0],
                         s[..., 3] / s[..., 0], (s[..., 3] - s[..., 2]) / s[..., 0],
                         s[..., 5] / s[..., 4], s[..., 6] / s[..., 7],
                         (s[..., 8] - s[..., 6]) / s[..., 7], s[..., 10] / s[..., 9]), axis=-1)


def describe(point, boots):
    result = {}
    for j, metric in enumerate(METRICS):
        valid = boots[:, j][np.isfinite(boots[:, j])]
        estimate = float(point[j]) if np.isfinite(point[j]) else None
        ci = np.quantile(valid,[.025,.975]).tolist() if len(valid) and estimate is not None else None
        result[metric] = {'estimate': estimate, 'ci95':ci,
                          'degenerate_ci':bool(ci is not None and ci[0]==ci[1]),
                          'defined_bootstrap_replicates': len(valid)}
    return result


def paired_bootstrap(original, neutral, indices):
    require(original.shape == neutral.shape and original.shape[0] == indices.shape[1],
            'Paired bootstrap prompt dimensions differ')
    points, boots = {}, {}
    for arm, values in zip(ARMS, (original, neutral)):
        points[arm] = rates(values.sum(axis=0))
        boots[arm] = np.empty((len(indices), len(METRICS)))
        for start in range(0, len(indices), 256):
            boots[arm][start:start+256] = rates(values[indices[start:start+256]].sum(axis=1))
    points[CONTRAST] = points['neutral'] - points['original']
    boots[CONTRAST] = boots['neutral'] - boots['original']
    return points, boots


def counts(records):
    keys = ('responses', 'correct_draws', 'pass8', 'distinct8', 'b8', 'correct_pairs',
            'colliding_correct_pairs', 'failed_draws', 'truncated_draws')
    result = {key: sum(r[key] for r in records) for key in keys}
    result.update(prompts=len(records), collision_eligible_prompts=sum(r['collision_eligible'] for r in records),
                  known_support_prompts=sum(r['certified_support_count'] is not None for r in records))
    for key in ('response_status_counts', 'stop_reason_counts'):
        counter = Counter()
        for row in records:
            counter.update(row[key])
        result[key] = dict(counter)
    return result


def joint_eligibility(original, neutral):
    require([r['row_sha256'] for r in original] == [r['row_sha256'] for r in neutral],
            'Paired prompt order or mathematical task changed')
    categories = Counter(('both' if a['collision_eligible'] and b['collision_eligible'] else
                          'original_only' if a['collision_eligible'] else
                          'neutral_only' if b['collision_eligible'] else 'neither')
                         for a, b in zip(original, neutral))
    return {key: categories[key] for key in ('both', 'original_only', 'neutral_only', 'neither')}


def macro(points, boots, keys):
    # A conditional metric with an undefined cell stays undefined: no omission.
    return {label: describe(np.mean([points[k][label] for k in keys], axis=0),
                            np.mean([boots[k][label] for k in keys], axis=0))
            for label in (*ARMS, CONTRAST)}


def analyze_pair(original, neutral, indices):
    require(original['rows'] == neutral['rows'], 'Pair has different mathematical tasks')
    require(original['model_id'] == neutral['model_id'], 'Pair uses different models')
    analyses = {}
    for grading in GRADINGS:
        records = {}
        for arm, run in zip(ARMS, (original, neutral)):
            records[arm] = [prompt_statistics(row, [run[grading][(*key, i)] for i in range(DRAWS)])
                            for key, row in sorted(run['rows'].items())]
        cells, points, boots = {}, {}, {}
        for level in sorted({key[0] for key in original['rows']}):
            for domain in [d for d in DOMAINS if (level, d) in {k[:2] for k in original['rows']}]:
                key = f'level{level}/{domain}'
                subset = {arm: [r for r in rs if (r['level'], r['domain']) == (level, domain)]
                          for arm, rs in records.items()}
                point, boot = paired_bootstrap(matrix(subset['original']), matrix(subset['neutral']), indices[level, domain])
                cells[key] = {label: describe(point[label], boot[label]) for label in point}
                cells[key]['counts'] = {arm: counts(rs) for arm, rs in subset.items()}
                cells[key]['joint_collision_eligibility'] = joint_eligibility(subset['original'], subset['neutral'])
                cells[key]['paired_prompt_differences'] = paired_difference_counts(subset['original'],subset['neutral'])
                points[key], boots[key] = point, boot
        groups = {'overall': macro(points, boots, list(points)), 'levels': {}}
        for level in sorted({key[0] for key in original['rows']}):
            groups['levels'][str(level)] = macro(points, boots, [k for k in points if k.startswith(f'level{level}/')])
        analyses[grading] = {'cells': cells, 'groups': groups,
                             'totals': {arm: counts(rs) for arm, rs in records.items()},
                             'joint_collision_eligibility': joint_eligibility(records['original'], records['neutral']),
                             'prompts': records}
    return {'model_id': original['model_id'], 'family': original['family'],
            'grading_audits':{arm:run.get('grading_audit') for arm,run in zip(ARMS,(original,neutral))},
            'conditions': {arm: {'directory': str(run['directory']), 'sources': run['sources']}
                           for arm, run in zip(ARMS, (original, neutral))}, 'analyses': analyses}


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_design(base):
    base = Path(base).resolve()
    manifest = json.loads((base / 'manifest.json').read_text())
    require(manifest.get('selection_seed') == SEED and manifest.get('sample_count') == DRAWS
            and manifest.get('prompt_count') == 192 and manifest.get('arm_count') == 2
            and manifest.get('fresh_both_arms') is True and manifest.get('outcomes_read') is False,
            'Wrong registered paired prompt design')
    required = {'rows.jsonl', 'prompts.jsonl', 'selection.json', 'transformations.json', 'protocol.json', 'design.json'}
    require(required <= set(manifest['artifact_sha256']), 'Unbound design artifacts')
    for name, digest in manifest['artifact_sha256'].items():
        bound_file(base / name, digest)
    for name, digest in manifest['code_sha256'].items():
        bound_file(base / 'code' / name, digest)
    protocol = json.loads((base / 'protocol.json').read_text())
    require(protocol['condition'] == CONDITION and protocol['prompts_per_cell'] == PROMPTS_PER_CELL
            and protocol['outcome_based_selection'] is False and protocol['fresh_both_arms'] is True,
            'Protocol changed the fixed selection or fresh two-arm design')
    reference = Path(manifest['reference_run']).resolve()
    bound_file(reference / 'manifest.json', manifest['reference_manifest_sha256'])
    for name in ('rows.jsonl', 'requests.jsonl'):
        bound_file(reference / name, manifest['reference_artifact_sha256'][name])
    builder_name = 'ops/prepare_modebench_prompt_ablation.py'
    require(builder_name in manifest['code_sha256'], 'Unfrozen prompt builder')
    builder = load_module('_frozen_prompt_ablation_builder', base / 'code' / builder_name)
    selected, ledger = builder.select_rows(read_jsonl(reference / 'rows.jsonl'))
    rows = unique(read_jsonl(base / 'rows.jsonl'), identity, 'design row')
    require(rows == unique(selected, identity, 'selected row'), 'Rows differ from outcome-blind hash selection')
    selection = json.loads((base / 'selection.json').read_text())
    require(selection.get('seed') == SEED and selection.get('outcomes_read') is False
            and selection['candidates'] == ledger, 'Selection ranking changed')
    require(Counter(k[:2] for k in rows) == Counter({(l,d): PROMPTS_PER_CELL for l in LEVELS for d in DOMAINS}),
            'Expected all six cells with 32 prompts')
    source_requests = unique(read_jsonl(reference / 'requests.jsonl'), sample_identity, 'source request')
    prompts = unique(read_jsonl(base / 'prompts.jsonl'), lambda r: (r['arm'], *identity(r)), 'arm prompt')
    require(set(prompts) == {(arm,*key) for arm in ARMS for key in rows}, 'Missing or unexpected prompt arm')
    for key, row in rows.items():
        original = source_requests[(*key, 0)]
        source_messages = original['request'].get('input', original['request'].get('messages'))
        for arm in ARMS:
            prompt = prompts[(arm, *key)]
            expected = source_messages if arm == 'original' else builder.neutral_messages(key[0], key[1], source_messages)
            require(prompt['row_sha256'] == sha(row) and prompt['messages_sha256'] == sha(prompt['messages'])
                    and prompt['messages'] == expected and prompt['messages'][1]['content'] == row['problem'],
                    'Prompt text differs from the frozen hint-only intervention')
            require(prompt['source_request_sha256'] == original['request_sha256'], 'Prompt source request changed')
    code = manifest['code_sha256']
    return {'base': base, 'manifest': manifest, 'rows': rows, 'prompts': prompts,
            'manifest_sha256': file_sha(base / 'manifest.json'),
            'normalizer_sha256': code['ops/frontier_modebench_normalization.py'],
            'contract_sha256': code['ops/frontier_modebench_contract.py'],
            'sources': {name: binding(base / name) for name in ('manifest.json', *sorted(required))}}


def native_inventory(run, expected):
    """Use the exact new native adapter while preserving the existing auditor."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import audit_hosted_modebench_completion as native
    native.RUNNERS = dict(native.RUNNERS)
    native.RUNNERS['frontier-modebench-native-chat-responses-v1'] = 'ops/evaluate_native_prompt_ablation.py'
    return native, native.load_inventory(run, expected_samples=expected)


def validate_run_inputs(design, entry):
    from datetime import datetime
    run = Path(entry['run_dir']).resolve()
    arm = entry['arm']
    require(arm in ARMS and entry['family'] == 'frontier', 'Unregistered hosted family or arm')
    bound_file(run / 'manifest.json', entry['manifest_sha256'])
    manifest = json.loads((run / 'manifest.json').read_text())
    require(manifest.get('experiment_condition') == CONDITION and manifest.get('prompt_arm') == arm
            and manifest.get('ablation_manifest_sha256') == design['manifest_sha256']
            and manifest.get('fresh_response_cohort') is True and manifest.get('sample_count') == DRAWS,
            'Require a fresh registered arm; historical controls are inadmissible')
    require(manifest['model'] == entry['model'], 'Registry deployment differs from run')
    expected = len(design['rows']) * DRAWS
    native, inventory = native_inventory(run, expected)
    require(inventory['rows'] == design['rows'], 'Run rows differ from frozen identical tasks')
    requests = unique(inventory['requests'], sample_identity, 'run request')
    expected_keys = {(*key, i) for key in design['rows'] for i in range(DRAWS)}
    require(set(requests) == expected_keys, 'Run request slot inventory differs')
    for key, item in requests.items():
        messages = item['request'].get('input', item['request'].get('messages'))
        require(messages == design['prompts'][(arm, *key[:3])]['messages'], 'Requested prompt changed')
        require(item.get('prompt_arm') == arm and item.get('experiment_condition') == CONDITION,
                'Request lacks fresh arm identity')
    require(manifest['code_sha256']['ops/frontier_modebench_normalization.py'] == design['normalizer_sha256']
            and manifest['code_sha256']['ops/frontier_modebench_contract.py'] == design['contract_sha256'],
            'Frozen normalizer or contract differs across conditions')
    raw = unique(read_jsonl(run / 'samples.jsonl'), sample_identity, 'sample')
    require(set(raw) == expected_keys, 'Incomplete or unexpected eight-draw slots; no inferential report')
    provider_ids = [(s['response_id'], s.get('choice_index', 0)) for s in raw.values()]
    require(len(set(provider_ids)) == expected, 'A native response/choice was reused')
    raw_bodies = native.validate_native_records(inventory, list(raw.values()))
    prepared = datetime.fromisoformat(manifest['prepared_at_utc'].replace('Z', '+00:00'))
    for sample in raw.values():
        grade_valid(sample)
        atomic = run / 'sample_receipts' / (sample['sample_id'] + '.json')
        require(json.loads(atomic.read_text()) == sample, 'Atomic receipt differs from exported sample')
        receipt = raw_bodies[sample['raw_receipt']]
        started = datetime.fromisoformat(receipt['started_at_utc'].replace('Z', '+00:00'))
        require(started >= prepared, 'Historical response predates the registered fresh run')
    return native, inventory, requests, raw


def validate_payload_pair(original, neutral):
    require(original['directory'] != neutral['directory'], 'Both arms point to the same cohort')
    require(set(original['requests']) == set(neutral['requests']), 'Paired request identities differ')
    for key, first in original['requests'].items():
        second = neutral['requests'][key]
        without_prompt = lambda r: {k: v for k,v in r['request'].items() if k not in ('input', 'messages')}
        require(without_prompt(first) == without_prompt(second), 'A non-prompt generation control changed')
        require(first['row_sha256'] == second['row_sha256'], 'Mathematical task differs across arms')
    require(not {s['response_id'] for s in original['raw'].values()} &
                {s['response_id'] for s in neutral['raw'].values()}, 'A response was reused across arms')
    # Runtime grading code is the same; collector adapters can differ by provider.
    graders = lambda r: {k:v for k,v in r['manifest']['code_sha256'].items()
                         if k.startswith('src/oat_drgrpo/') or k == 'ops/frontier_modebench_contract.py'}
    require(graders(original) == graders(neutral), 'Executable graders differ across arms')


def frozen_grade_modules(root):
    """Must run in an isolated process before importing any live grader."""
    for name in sys.modules:
        require(name != 'frontier_modebench_contract' and not name.startswith('oat_drgrpo'),
                'Run offline grading in an isolated process with no preloaded grader')
    sys.path.insert(0, str(root / 'src'))
    sys.path.insert(0, str(root / 'ops'))
    import frontier_modebench_contract as contract
    normalizer = load_module('_prompt_ablation_frozen_normalizer', root / 'ops/frontier_modebench_normalization.py')
    return contract, normalizer


def grade_hosted(base, run_dir):
    """Serial frozen Python recheck plus one previously frozen normalizer."""
    design = authenticate_design(base)
    registry = json.loads((Path(base) / 'hosted_analysis_runs.json').read_text())
    entries = [e for e in registry['runs'] if Path(e['run_dir']).resolve() == Path(run_dir).resolve()]
    require(len(entries) == 1, 'Run not uniquely registered')
    entry = entries[0]
    native, inventory, requests, raw = validate_run_inputs(design, entry)
    run = Path(run_dir).resolve()
    output = run / 'prompt_ablation_grades.jsonl'
    audit_path = run / 'prompt_ablation_grading_audit.json'
    if audit_path.exists():
        authenticate_grades(design, run, raw)
        return json.loads(audit_path.read_text())
    require(not output.exists(), 'Unfinished grading output exists; preserve it before regeneration')
    completion = native.audit(run,expected_samples=len(raw))
    contract, normalizer = frozen_grade_modules(run / 'code')
    warmup = warm_frozen_python(contract)
    entries, changed = [], 0
    for key, sample in sorted(raw.items()):
        row = design['rows'][key[:3]]
        strict = {k: sample[k] for k in ('verified', 'canonical_key', 'graded_text')}
        if key[1] == 'python_factors':
            strict = contract.grade_response(key[0], key[1], row, sample['text'])
        grade_valid(strict)
        changed += any(strict.get(k) != sample.get(k) for k in ('verified', 'canonical_key', 'graded_text'))
        normalized = normalizer.normalize_and_grade(row, sample['text'], strict_grade=strict, grader=contract.grade_response)
        grade_valid(normalized)
        require(not strict['verified'] or (normalized['verified'] and normalized['canonical_key'] == strict['canonical_key']),
                'Normalization changed a strict success')
        entries.append({**{k:sample[k] for k in ('level','domain','row_index','sample_index')},
                        'raw_sample_sha256': sha(sample), 'strict': strict, 'normalization': normalized})
    temporary = output.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(e, sort_keys=True, allow_nan=False)+'\n' for e in entries))
    temporary.replace(output)
    source = freeze_analysis_source(run)
    audit = {'schema': 'prompt-ablation-frozen-grades-v1', 'status': 'complete', 'api_calls': 0,
             'records': len(entries), 'manifest_sha256': file_sha(run/'manifest.json'),
             'completion_audit_sha256':file_sha(run/'completion_audit.json'),
             'evidence_inventory_sha256':file_sha(run/'evidence_file_sha256.json'),
             'native_adapter':'ops/evaluate_native_prompt_ablation.py',
             'physical_attempts':completion['raw_attempts'],
             'attempt_status_counts':completion['attempt_status_counts'],
             'raw_samples_sha256': file_sha(run/'samples.jsonl'), 'rows_sha256': file_sha(run/'rows.jsonl'),
             'cache_sha256': file_sha(output), 'normalizer_sha256': design['normalizer_sha256'],
             'contract_sha256': design['contract_sha256'], 'analyzer_source': source,
             'python_rechecked_serially': sum(k[1]=='python_factors' for k in raw),
             'strict_changed_records': changed, 'python_synthetic_warmup': warmup,
             'strict_verified': sum(e['strict']['verified'] for e in entries),
             'normalized_verified': sum(e['normalization']['verified'] for e in entries),
             'grader_code_sha256': {k:v for k,v in inventory['manifest']['code_sha256'].items()
                                    if k.startswith('src/oat_drgrpo/') or k=='ops/frontier_modebench_contract.py'}}
    write_json(audit_path, audit)
    return audit


def authenticate_grades(design, run, raw):
    audit = json.loads((run / 'prompt_ablation_grading_audit.json').read_text())
    require(audit.get('status') == 'complete' and audit.get('records') == len(raw)
            and audit.get('normalizer_sha256') == design['normalizer_sha256']
            and audit.get('contract_sha256') == design['contract_sha256'], 'Incomplete or changed frozen grading procedure')
    for name, key in [('manifest.json','manifest_sha256'), ('rows.jsonl','rows_sha256'),
                      ('samples.jsonl','raw_samples_sha256'), ('prompt_ablation_grades.jsonl','cache_sha256')]:
        bound_file(run/name, audit[key])
    bound_file(audit['analyzer_source']['path'], audit['analyzer_source']['sha256'])
    bound_file(run/'completion_audit.json',audit['completion_audit_sha256'])
    bound_file(run/'evidence_file_sha256.json',audit['evidence_inventory_sha256'])
    completion = json.loads((run/'completion_audit.json').read_text())
    evidence = json.loads((run/'evidence_file_sha256.json').read_text())
    require(completion['status']=='pass' and completion['expected_responses']==len(raw)
            and completion['saved_samples']==len(raw) and completion['evidence_inventory_sha256']==sha(evidence),
            'Native completion/attempt audit incomplete or changed')
    for name,digest in evidence.items():
        bound_file(run/name,digest)
    manifest = json.loads((run/'manifest.json').read_text())
    require(audit['grader_code_sha256']=={k:v for k,v in manifest['code_sha256'].items()
            if k.startswith('src/oat_drgrpo/') or k=='ops/frontier_modebench_contract.py'},
            'Serial grading audit used different executable grader sources')
    records = unique(read_jsonl(run/'prompt_ablation_grades.jsonl'), sample_identity, 'grade cache')
    require(set(records) == set(raw), 'Grading cache does not cover every sampled slot')
    strict, normalized = {}, {}
    for key, record in records.items():
        sample = raw[key]
        require(record['raw_sample_sha256'] == sha(sample), 'Grade cache belongs to another raw response')
        for field in ('strict', 'normalization'):
            grade_valid(record[field])
        a, b = record['strict'], record['normalization']
        require(b.get('original_text') == sample['text'], 'Normalization original text changed')
        require(not a['verified'] or (b['verified'] and b['canonical_key'] == a['canonical_key']),
                'Normalization changed a strict success')
        if key[1] != 'python_factors':
            require(all(a.get(k) == sample.get(k) for k in ('verified','canonical_key','graded_text')),
                    'Serial Python audit changed another domain')
        strict[key], normalized[key] = {**sample, **a}, {**sample, **b}
    require(audit.get('python_rechecked_serially')==sum(k[1]=='python_factors' for k in raw)
            and audit.get('python_synthetic_warmup',[])[-1:] == [True], 'Incomplete warmed Python recheck')
    require(audit['strict_verified'] == sum(s['verified'] for s in strict.values())
            and audit['normalized_verified'] == sum(s['verified'] for s in normalized.values()), 'Grading totals differ')
    return strict, normalized


def authenticate_hosted(design, entry):
    native, inventory, requests, raw = validate_run_inputs(design, entry)
    run = Path(entry['run_dir']).resolve()
    strict, normalized = authenticate_grades(design, run, raw)
    return {'directory': run, 'model_id': entry['model_id'], 'family': 'frontier',
            'manifest': inventory['manifest'], 'rows': inventory['rows'], 'requests': requests,
            'raw': raw, 'strict': strict, 'normalized_secondary': normalized,
            'grading_audit':json.loads((run/'prompt_ablation_grading_audit.json').read_text()),
            'sources': {name: binding(run/name) for name in ('manifest.json','rows.jsonl','requests.jsonl',
                       'samples.jsonl','prompt_ablation_grades.jsonl','prompt_ablation_grading_audit.json',
                       'completion_audit.json','evidence_file_sha256.json')}}


def freeze_analysis_source(directory):
    source = Path(__file__)
    destination = Path(directory) / 'analysis_code' / (source.stem + '_' + file_sha(source)[:16] + '.py')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(destination.read_bytes() == source.read_bytes(), 'Frozen analysis source collision')
    else:
        destination.write_bytes(source.read_bytes())
    auditor_source=source.parent/'audit_hosted_modebench_completion.py'
    if auditor_source.exists():
        auditor_destination=destination.parent/auditor_source.name
        if auditor_destination.exists():
            require(auditor_destination.read_bytes()==auditor_source.read_bytes(), 'Frozen native auditor changed')
        else:auditor_destination.write_bytes(auditor_source.read_bytes())
    return binding(destination)


def local_key(record):
    domain = 'pantry_plan' if record['domain'] == 'pantry' else record['domain']
    return record['level'], domain, record['row_index']


def local_samples(design, plan_path, checkpoint):
    """Authenticate a pinned checkpoint's complete two-arm local receipts."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    validate_local_registration(design,plan_path,plan)
    for path, digest in {**plan['input_sha256'], **plan['code_sha256']}.items():
        bound_file(path, digest)
    require(Path(plan['rows_path']).resolve() == design['base']/'rows.jsonl'
            and Path(plan['prompts_path']).resolve() == design['base']/'prompts.jsonl',
            'Local plan uses another prompt intervention')
    require(plan['input_sha256'].get(str(design['base']/'manifest.json')) == design['manifest_sha256'],
            'Local checkpoint panel not bound to frozen prompt manifest')
    require(plan['settings']['sample_count'] == DRAWS, 'Local draw count differs')
    directory = Path(plan['output_root']) / checkpoint['label']
    result = json.loads((directory/'result.json').read_text())
    expected_identity = {'schema': plan['schema'], 'plan_sha256': file_sha(plan_path),
                         'checkpoint': checkpoint, 'settings': plan['settings']}
    require(result['identity'] == expected_identity and result['identity_sha256'] == sha(expected_identity)
            and result['status'] == 'complete', 'Local result differs from registered checkpoint/controls')
    require(Path(result['responses_path']).resolve() == directory.resolve()/'responses.jsonl',
            'Local response path lies outside checkpoint cohort')
    bound_file(directory/'responses.jsonl', result['responses_sha256'])
    rows = {key: row for key,row in design['rows'].items()
            if checkpoint['domain'] is None or key[1] == ('pantry_plan' if checkpoint['domain']=='pantry' else checkpoint['domain'])}
    expected = {(arm,*key,i) for arm in ARMS for key in rows for i in range(DRAWS)}
    raw = unique(read_jsonl(directory/'responses.jsonl'), lambda r: (r['arm'],*local_key(r),r['draw_index']), 'local draw')
    require(set(raw) == expected and result['draws'] == len(expected) == checkpoint['expected_draws'],
            'Local checkpoint missing registered prompts or eight-draw slots')
    rebuilt = {}
    for key,sample in raw.items():
        arm, level, domain, index, draw = key
        grade_valid(sample)
        require(sample['checkpoint_label'] == checkpoint['label'] and
                all(sample[field] == checkpoint[field] for field in ('training_method','training_seed','trained_on_level')),
                'Local draw comes from another checkpoint')
        require(sample['row_sha256'] == sha(rows[level,domain,index]) and
                sample['messages_sha256'] == design['prompts'][arm,level,domain,index]['messages_sha256'] and
                sample['pair_id'] == design['prompts'][arm,level,domain,index]['pair_id'],
                'Local draw prompt hash changed')
        peer = raw[('neutral' if arm=='original' else 'original',level,domain,index,draw)]
        expected_seed = plan['settings']['seed_base']+plan['settings']['seed_stride']*((level-2)*10000+DOMAINS.index(domain)*1000+index)
        require(sample['sampling_seed'] == peer['sampling_seed'] == expected_seed,
                'Local sampling seed differs from the frozen disjoint child-seed schedule')
        adapted = {**sample, 'domain':domain, 'sample_index':draw,
                   'stop_reason':sample.get('finish_reason','unknown'), 'graded_text':sample['text']}
        rebuilt[key] = adapted
    # Bind the exported draw records to every original saved generation group.
    grouped = unique(result['prompt_results'], lambda r:(r['arm'],*local_key(r)), 'local prompt result')
    require(set(grouped) == {(a,*k) for a in ARMS for k in rows}, 'Local prompt result inventory differs')
    for key,group in grouped.items():
        require(len(group['attempts']) == DRAWS, 'Local generation group lacks eight attempts')
        for draw, attempt in enumerate(group['attempts']):
            require(all(raw[(*key,draw)].get(field) == value for field,value in attempt.items()),
                    'Local draw export differs from saved generation attempt')
    return directory.resolve(), rows, rebuilt, result


def grade_local(base, plan_path, checkpoint_label):
    design = authenticate_design(base)
    plan = json.loads(Path(plan_path).read_text())
    checkpoints = [c for c in plan['checkpoints'] if c['label'] == checkpoint_label]
    require(len(checkpoints) == 1, 'Local checkpoint not uniquely registered')
    directory, rows, raw, result = local_samples(design, plan_path, checkpoints[0])
    output = directory/'prompt_ablation_normalized.jsonl'
    audit_path = directory/'prompt_ablation_normalization_audit.json'
    if audit_path.exists():
        authenticate_local_normalization(design, directory, raw)
        return json.loads(audit_path.read_text())
    require(not output.exists(), 'Unfinished local normalization cache exists')
    contract, normalizer = frozen_grade_modules(design['base']/'code')
    # Recheck every Python draw serially to remove cold-worker startup artifacts.
    warmup = warm_frozen_python(contract) if any(k[2]=='python_factors' for k in raw) else []
    entries, changed = [], 0
    for key,sample in sorted(raw.items()):
        row = rows[key[1:4]]
        strict = {field:sample[field] for field in ('verified','canonical_key','graded_text')}
        if key[2]=='python_factors':
            strict = contract.grade_response(key[1],key[2],row,sample['text'])
        grade_valid(strict)
        changed += any(strict.get(k)!=sample.get(k) for k in ('verified','canonical_key','graded_text'))
        normalization = normalizer.normalize_and_grade(row, sample['text'], strict_grade=strict,
                                                       grader=contract.grade_response)
        grade_valid(normalization)
        require(not strict['verified'] or (normalization['verified'] and normalization['canonical_key']==strict['canonical_key']),
                'Local normalization changed a strict success')
        entries.append({'arm':key[0],'level':key[1],'domain':key[2],'row_index':key[3],'sample_index':key[4],
                        'raw_sample_sha256':sha(sample), 'strict':strict, 'normalization':normalization})
    temporary = output.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(e, sort_keys=True, allow_nan=False)+'\n' for e in entries))
    temporary.replace(output)
    audit = {'schema':'prompt-ablation-local-normalization-v1', 'status':'complete', 'api_calls':0,
             'records':len(raw), 'result_sha256':file_sha(directory/'result.json'),
             'responses_sha256':file_sha(directory/'responses.jsonl'), 'cache_sha256':file_sha(output),
             'normalizer_sha256':design['normalizer_sha256'], 'contract_sha256':design['contract_sha256'],
             'analyzer_source':freeze_analysis_source(directory),
             'strict_verified':sum(e['strict']['verified'] for e in entries),
             'raw_verified':sum(s['verified'] for s in raw.values()),
             'strict_changed_records':changed, 'python_synthetic_warmup':warmup,
             'python_rechecked_serially':sum(k[2]=='python_factors' for k in raw),
             'normalized_verified':sum(e['normalization']['verified'] for e in entries)}
    write_json(audit_path,audit)
    return audit


def authenticate_local_normalization(design, directory, raw):
    audit = json.loads((directory/'prompt_ablation_normalization_audit.json').read_text())
    require(audit.get('status')=='complete' and audit['records']==len(raw)
            and audit['normalizer_sha256']==design['normalizer_sha256']
            and audit['contract_sha256']==design['contract_sha256'], 'Wrong local normalization inventory or frozen source')
    for name,field in [('result.json','result_sha256'),('responses.jsonl','responses_sha256'),
                       ('prompt_ablation_normalized.jsonl','cache_sha256')]:
        bound_file(directory/name,audit[field])
    bound_file(audit['analyzer_source']['path'],audit['analyzer_source']['sha256'])
    entries = unique(read_jsonl(directory/'prompt_ablation_normalized.jsonl'),
                     lambda r:(r['arm'],*sample_identity(r)), 'local normalization')
    require(set(entries)==set(raw), 'Local normalization lacks registered draws')
    strict, normalized = {}, {}
    for key,sample in raw.items():
        entry = entries[key]
        require(entry['raw_sample_sha256']==sha(sample), 'Local normalized cache belongs to another response')
        primary, grade = entry['strict'], entry['normalization']
        grade_valid(primary)
        grade_valid(grade)
        if key[2]!='python_factors':
            require(all(primary.get(k)==sample.get(k) for k in ('verified','canonical_key','graded_text')),
                    'Local Python audit changed another domain')
        require(grade['original_text']==sample['text'], 'Local normalization changed original text')
        require(not primary['verified'] or (grade['verified'] and grade['canonical_key']==primary['canonical_key']),
                'Local normalization changed a strict success')
        strict[key], normalized[key] = {**sample, **primary}, {**sample, **grade}
    python_count = sum(k[2]=='python_factors' for k in raw)
    require(audit.get('python_rechecked_serially')==python_count
            and (not python_count or audit.get('python_synthetic_warmup',[])[-1:]==[True]),
            'Local serial Python recheck is incomplete or not warmed')
    require(audit['strict_verified']==sum(s['verified'] for s in strict.values())
            and audit['normalized_verified']==sum(s['verified'] for s in normalized.values())
            and audit['raw_verified']==sum(s['verified'] for s in raw.values()), 'Local grading audit totals differ')
    return strict, normalized


def authenticate_local(design, plan_path, checkpoint):
    directory, rows, raw, result = local_samples(design, plan_path, checkpoint)
    strict, normalized = authenticate_local_normalization(design,directory,raw)
    runs = []
    for arm in ARMS:
        runs.append({'directory':directory, 'model_id':checkpoint['label'], 'family':'local',
                     'checkpoint':checkpoint, 'rows':rows,
                     'grading_audit':json.loads((directory/'prompt_ablation_normalization_audit.json').read_text()),
                     'strict':{k[1:]:s for k,s in strict.items() if k[0]==arm},
                     'normalized_secondary':{k[1:]:s for k,s in normalized.items() if k[0]==arm},
                     'sources':{name:binding(directory/name) for name in
                        ('result.json','responses.jsonl','prompt_ablation_normalized.jsonl','prompt_ablation_normalization_audit.json')}})
    return runs


def combine_local_seeds(models, indices, replicates=20000):
    """Keep checkpoint/seed replication distinct from paired prompt replication.

    Fixed-seed prompt CIs resample the SAME task indices for every seed. For
    groups with five seeds, hierarchical CIs additionally resample whole seed
    pairs. Two-seed groups show both effects and their range, without a seed CI.
    """
    groups = {}
    for model in models:
        checkpoint = model.get('checkpoint', {})
        if checkpoint.get('training_method') in (None,'initial'):
            continue
        key = checkpoint['training_method'] + '/' + ('pantry_plan' if checkpoint['domain']=='pantry' else checkpoint['domain'])
        groups.setdefault(key,[]).append(model)
    output = {}
    for key, members in sorted(groups.items()):
        members.sort(key=lambda m:m['checkpoint']['training_seed'])
        seeds = [m['checkpoint']['training_seed'] for m in members]
        require(len(seeds)==len(set(seeds)), 'Duplicate seed checkpoint in local objective/domain')
        domain = key.split('/')[1]
        expected_seeds = [43,46] if domain=='pantry_plan' else list(range(43,48))
        require(seeds == expected_seeds, 'Missing or selected local training seeds')
        seed_indices = np.random.default_rng(SEED).integers(0,len(seeds),(replicates,len(seeds)))
        analyses = {}
        for grading in GRADINGS:
            cells = {}
            for level in LEVELS:
                cell = f'level{level}/{domain}'
                seed_points, seed_boots = [], []
                per_seed = {}
                for model in members:
                    prompts = model['analyses'][grading]['prompts']
                    rs = {arm:[r for r in prompts[arm] if r['level']==level] for arm in ARMS}
                    point, boot = paired_bootstrap(matrix(rs['original']),matrix(rs['neutral']),indices[level,domain])
                    seed_points.append(point);seed_boots.append(boot)
                    per_seed[str(model['checkpoint']['training_seed'])] = {arm:describe(point[arm],boot[arm]) for arm in point}
                combined, hierarchical = {}, {}
                for arm in (*ARMS,CONTRAST):
                    point_array = np.asarray([p[arm] for p in seed_points])
                    boot_array = np.asarray([b[arm] for b in seed_boots])
                    point = np.mean(point_array,axis=0)
                    combined[arm] = describe(point,np.mean(boot_array,axis=0))
                    if len(seeds)>=5:
                        # Select the same seeds for both arms and all metrics;
                        # pair at both sampling stages, preserve prompt strata.
                        hierarchical_boot = np.mean(boot_array[seed_indices,np.arange(replicates)[:,None]],axis=1)
                        hierarchical[arm] = describe(point,hierarchical_boot)
                cells[cell] = {'seed_mean_fixed_seed_prompt_ci':combined,
                               'hierarchical_seed_and_prompt_ci':hierarchical or None,
                               'per_seed':per_seed,
                               'contrast_seed_range':{metric:[min(p[CONTRAST][j] for p in seed_points),
                                                               max(p[CONTRAST][j] for p in seed_points)]
                                                        for j,metric in enumerate(METRICS)
                                                        if all(np.isfinite(p[CONTRAST][j]) for p in seed_points)}}
            analyses[grading] = {'cells':cells}
        output[key] = {'training_method':members[0]['checkpoint']['training_method'],
                       'domain':domain, 'training_seeds':seeds, 'seed_count':len(seeds),
                       'checkpoint_ids':[m['model_id'] for m in members], 'analyses':analyses}
    return output


def inventory_report(design, hosted_registry=None, local_plan=None):
    """A missing cell is listed, never turned into an unsuccessful observation."""
    report = {'status':'complete', 'expected_draws':0, 'finalized_draws':0, 'durable_draws':0, 'runs':[]}
    if hosted_registry:
        registry = json.loads(Path(hosted_registry).read_text())
        require(registry.get('experiment_condition')==CONDITION
                and registry['ablation_manifest_sha256']==design['manifest_sha256'], 'Registry belongs to another experiment')
        seen = unique(registry['runs'],lambda e:(e['model_id'],e['arm']),'registered hosted arm')
        require(set(seen)=={(model,arm) for model in ('gpt56sol','gpt54','grok43') for arm in ARMS},
                'Hosted registry omitted a predeclared model or fresh arm')
        for entry in registry['runs']:
            run = Path(entry['run_dir'])
            path = run/'samples.jsonl'
            records = read_jsonl(path) if path.exists() else []
            keys = unique(records,sample_identity,'hosted inventory sample')
            expected = {(*k,i) for k in design['rows'] for i in range(DRAWS)}
            require(set(keys)<=expected, 'Unexpected hosted sample in inventory')
            cells = {f'level{l}/{d}':{'expected_draws':PROMPTS_PER_CELL*DRAWS,
                       'finalized_draws':sum(k[:2]==(l,d) for k in keys),
                       'failed_draws':sum(k[:2]==(l,d) and s.get('verified') is False for k,s in keys.items())}
                       for l in LEVELS for d in DOMAINS}
            report['runs'].append({'model_id':entry['model_id'],'family':'frontier','arm':entry['arm'],
                                    'directory':str(run),'complete':set(keys)==expected,
                                    'expected_draws':len(expected),'finalized_draws':len(keys),'cells':cells,
                                    'graded':(run/'prompt_ablation_grading_audit.json').exists()})
    if local_plan:
        plan = json.loads(Path(local_plan).read_text())
        validate_local_registration(design,Path(local_plan),plan)
        labels = unique(plan['checkpoints'],lambda c:c['label'],'local checkpoint')
        require(len(labels)==25, 'Local registry must retain the initial model and all 24 trained checkpoints')
        for checkpoint in plan['checkpoints']:
            run = Path(plan['output_root'])/checkpoint['label']
            path = run/'responses.jsonl'
            records = read_jsonl(path) if path.exists() else []
            raw = unique(records,lambda r:(r['arm'],*local_key(r),r['draw_index']),'local inventory draw')
            domain = 'pantry_plan' if checkpoint['domain']=='pantry' else checkpoint['domain']
            cells = [(l,d) for l in LEVELS for d in DOMAINS if domain is None or d==domain]
            expected = {(a,*k,i) for a in ARMS for k in design['rows'] if k[:2] in cells for i in range(DRAWS)}
            require(set(raw)<=expected, 'Unexpected local sample in inventory')
            saved_batch_draws=0
            for batch_path in sorted(run.glob('batch_*.json')):
                batch=json.loads(batch_path.read_text())
                require(batch.get('records_sha256')==sha(batch['records']), 'Partial saved local batch digest differs')
                saved_batch_draws+=sum(len(r['attempts']) for r in batch['records'])
            report['runs'].append({'model_id':checkpoint['label'],'family':'local','arm':'both',
                    'directory':str(run),'complete':set(raw)==expected and (run/'result.json').exists(),
                    'expected_draws':len(expected),'finalized_draws':len(raw),'durable_batch_draws':saved_batch_draws,
                    'cells':{f'{a}/level{l}/{d}':{'expected_draws':PROMPTS_PER_CELL*DRAWS,
                             'finalized_draws':sum(k[:3]==(a,l,d) for k in raw),
                             'failed_draws':sum(k[:3]==(a,l,d) and r.get('verified') is False for k,r in raw.items())}
                             for a in ARMS for l,d in cells},
                    'graded':(run/'prompt_ablation_normalization_audit.json').exists()})
    for run in report['runs']:
        report['expected_draws'] += run['expected_draws']
        report['finalized_draws'] += run['finalized_draws']
        report['durable_draws'] += max(run['finalized_draws'],run.get('durable_batch_draws',0))
        if not run['complete']: report['status']='incomplete'
    return report


def build_report(base, hosted_registry=None, local_plan=None, replicates=20000):
    require(replicates==20000, 'The registered final analysis uses exactly 20,000 replicates')
    require(hosted_registry or local_plan, 'Specify at least one complete registered model panel')
    design = authenticate_design(base)
    inventory = inventory_report(design,hosted_registry,local_plan)
    require(inventory['status']=='complete', 'Sampling incomplete: save --inventory before inferential analysis')
    rng = np.random.default_rng(SEED)
    indices = {(l,d):rng.integers(0,PROMPTS_PER_CELL,(replicates,PROMPTS_PER_CELL)) for l in LEVELS for d in DOMAINS}
    models = []
    if hosted_registry:
        registry = json.loads(Path(hosted_registry).read_text())
        runs = {(e['model_id'],e['arm']):authenticate_hosted(design,e) for e in registry['runs']}
        for model_id in ('gpt56sol','gpt54','grok43'):
            original, neutral = (runs[model_id,arm] for arm in ARMS)
            validate_payload_pair(original,neutral)
            models.append(analyze_pair(original,neutral,indices))
    if local_plan:
        plan = json.loads(Path(local_plan).read_text())
        for checkpoint in plan['checkpoints']:
            original,neutral = authenticate_local(design,local_plan,checkpoint)
            result = analyze_pair(original,neutral,indices)
            result['checkpoint'] = checkpoint
            models.append(result)
    source = binding(Path(__file__))
    return {'schema':'modebench-prompt-ablation-analysis-v1','status':'complete',
            'experiment_status':'complete' if hosted_registry and local_plan else 'partial_panels',
            'scope':{'included_panels':[name for name,path in [('frontier',hosted_registry),('local',local_plan)] if path],
                     'registered_panels':['frontier','local'],
                     'omitted_panels':[name for name,path in [('frontier',hosted_registry),('local',local_plan)] if not path],
                     'completion_applies_to':'All registered models/checkpoints and cells within included panels only.'},
            'design':design['sources'],'inventory':inventory,'analyzer_source':source,
            'analysis_dependencies':{'audit_hosted_modebench_completion.py':binding(Path(__file__).resolve().parent/'audit_hosted_modebench_completion.py')},
            'prospective_analysis_plan':binding(Path(base)/'ANALYSIS_PLAN.md'),
            'prospective_amendments':{str(p.relative_to(Path(base))):binding(p)
              for pattern in ('execution_amendment*.json','local/*amendment*.json')
              for p in sorted(Path(base).glob(pattern))},
            'registries':{label:binding(path) for label,path in [('hosted',hosted_registry),('local',local_plan)] if path},
            'protocol':{'condition':CONDITION,'primary_grading':'strict',
              'primary_metrics':['distinct8','pass8'],'additional_breadth_metric':'b8 = distinct8 - pass8',
              'contrast':CONTRAST,'normalization':'One unchanged frozen formatting normalizer; strict successes/keys retained.',
              'draws_per_prompt_arm':DRAWS,'prompts_per_domain_level':PROMPTS_PER_CELL,
              'bootstrap_replicates':replicates,'seed':SEED,
              'resampling':'Whole paired prompts with all eight draws intact; independently stratified by domain and level. The same index arrays are reused across arms, metrics, gradings and model/checkpoint seeds.',
              'aggregation':'Equal domain-level means; undefined conditional cells propagate instead of being dropped.',
              'collision':'Sum of colliding correct pairs / sum of all correct pairs within each cell; known-support references use exactly the same known-support prompt/pair weights.',
              'local_seeds':'Checkpoint effects reported individually. Fixed-seed prompt CI shares prompt indices across seeds; five-seed groups additionally show hierarchical seed-and-prompt CIs. Two-seed Pantry groups show both seed estimates and ranges.',
              'interval_scope':'Pointwise percentile 95% intervals, no multiplicity adjustment; exploratory 32 prompts per cell, not evidence of equivalence or absence of an effect. Degenerate intervals reflect zero observed resampled variation; paired discordance counts accompany primary effects.'},
            'models':models, 'local_seed_groups':combine_local_seeds([m for m in models if m['family']=='local'],indices,replicates),
            'limitations':['This is a prospectively frozen follow-up to already evaluated benchmark prompts.',
                'Only Python factors, MathIR and Pantry at Levels 2 and 3 are sampled; no inference about other cells.',
                'The intervention removes the registered strategy hints while preserving the mathematical task and remaining format requirements.',
                'Local syntax constraints, token budgets and sampling differ from hosted inference; compare prompt arms within the same model/interface.',
                'Level 3 local checkpoint results are transfer from Level 2 training.',
                'Observed breadth after eight draws does not identify total support or prove that unseen modes have zero probability.',
                'Conditional collision populations can differ when prompt wording changes correctness; joint eligibility counts are reported.']}


def warm_frozen_python(contract):
    from oat_drgrpo.python_modebench_process import _SHARED_VERIFIER
    import time
    attempts = []
    for _ in range(3):
        _SHARED_VERIFIER._start()
        time.sleep(1.25)
        result = contract.grade_response(2,'python_factors',
            {'answer':{'verifier':'python_factor_function','python_version':'factor-v1','cases':[6,8]}},
            r'\boxed{lambda n: 2}')
        attempts.append(result['verified'])
        if attempts[-1]: break
    require(attempts[-1], 'Frozen serial Python verifier failed its synthetic warmup')
    return attempts


MODEL_LABELS = {'gpt56sol':'GPT-5.6 Sol','gpt54':'GPT-5.4','grok43':'Grok 4.3','qwen05b_initial':'Initial Qwen 0.5B'}
METHOD_LABELS = {'drgrpo':'DrGRPO','replay_drgrpo':'Re:Dr.GRPO'}


def display_rows(report, family, grading):
    rows = []
    for model in report['models']:
        if model['family']!=family or (family=='local' and model.get('checkpoint',{}).get('training_method')!='initial'):
            continue
        for cell,values in sorted(model['analyses'][grading]['cells'].items(),key=lambda item:(int(item[0][5]),DOMAINS.index(item[0].split('/')[1]))):
            level,domain = cell.split('/')
            rows.append({'label':MODEL_LABELS.get(model['model_id'],model['model_id']),
                         'level':int(level[-1]),'domain':domain,'seed_count':1,
                         'metrics':values,'interval_kind':'paired prompts','seed_ranges':None})
    if family=='local':
        for group in report['local_seed_groups'].values():
            for cell,values in sorted(group['analyses'][grading]['cells'].items()):
                # Across-seed inference is the displayed interval when available.
                selected = values['hierarchical_seed_and_prompt_ci'] or values['seed_mean_fixed_seed_prompt_ci']
                rows.append({'label':METHOD_LABELS[group['training_method']], 'level':int(cell[5]),
                             'domain':group['domain'],'seed_count':group['seed_count'],'metrics':selected,
                             'interval_kind':'seed + paired prompts' if group['seed_count']>=5 else 'fixed-seed paired prompts',
                             'seed_ranges':values['contrast_seed_range'] if group['seed_count']<5 else None})
    return rows


def tex_escape(value):
    return str(value).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')


def scalar(value, digits=3, signed=False):
    if value is None: return '--'
    return f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'


def interval_tex(metric,digits=3):
    if metric['estimate'] is None: return '--'
    value = scalar(metric['estimate'],digits,True)
    ci = metric['ci95']
    return '$'+value+(r'\;['+','.join(scalar(x,digits,True) for x in ci)+']' if ci else '')+'$'


def render_table(report,family,grading):
    label = 'Hosted models' if family=='frontier' else 'Local Qwen2.5-0.5B checkpoints'
    grade_label = 'strict verification' if grading=='strict' else 'frozen formatting normalization'
    lines = [r'\begin{table}[t]',r'\centering',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',
             r'\caption{'+label+' under '+grade_label+r'. Each arrow gives original $\to$ neutral. '
             r'$P_8$ is empirical \texttt{pass@8}, $D_8$ is verified \texttt{distinct@8}, '
             r'and $B_8=D_8-P_8$. Effects are neutral minus original; brackets are pointwise 95\% intervals. '
             + ('Local rows average the displayed number of training seeds. Five-seed intervals include seed and paired-prompt resampling; two-seed Pantry intervals condition on the two fixed checkpoints. '
                if family=='local' else 'Each model--domain--level row contains 32 paired problems and eight draws per arm. ')+r'}',
             r'\label{tab:prompt-hints-'+family+'-'+grading.replace('_','-')+'}',
             r'\begin{tabular}{llrrrrr}',r'\toprule',
             r'Model & Cell & $P_8$: O$\to$N & $D_8$: O$\to$N & $\Delta P_8$ [95\%] & $\Delta D_8$ [95\%] & $\Delta B_8$ \\',r'\midrule']
    for row in display_rows(report,family,grading):
        metrics = row['metrics']
        original,neutral,delta = (metrics[k] for k in (*ARMS,CONTRAST))
        name = tex_escape(row['label'])+(f" ($s={row['seed_count']}$)" if family=='local' else '')
        cell = tex_escape(DOMAIN_LABELS[row['domain']])+f" L{row['level']}"
        p = '$'+scalar(original['pass8']['estimate'])+r'\to'+scalar(neutral['pass8']['estimate'])+'$'
        d = '$'+scalar(original['distinct8']['estimate'],2)+r'\to'+scalar(neutral['distinct8']['estimate'],2)+'$'
        lines.append(' & '.join([name,cell,p,d,interval_tex(delta['pass8']),interval_tex(delta['distinct8'],2),
                                '$'+scalar(delta['b8']['estimate'],2,True)+'$'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    return '\n'.join(lines)


def render_seed_ranges(report):
    if not report['local_seed_groups']: return ''
    lines = [r'\paragraph{Two-seed Pantry sensitivity.} '
             r'Two checkpoints cannot support a stable estimate of training-seed variability. '
             r'The following lists both strict prompt effects; the intervals in the local table condition on these two checkpoints.']
    fragments = []
    for group in report['local_seed_groups'].values():
        if group['seed_count']!=2: continue
        for cell,values in group['analyses']['strict']['cells'].items():
            seed_text = []
            for seed,metrics in values['per_seed'].items():
                d = metrics[CONTRAST]
                seed_text.append(f"seed {seed}: $\\Delta P_8={scalar(d['pass8']['estimate'],3,True)}$, "
                                 f"$\\Delta D_8={scalar(d['distinct8']['estimate'],2,True)}$")
            fragments.append(METHOD_LABELS[group['training_method']]+f" L{cell[5]} ("+'; '.join(seed_text)+')')
    lines.append('; '.join(fragments)+'.\n')
    return '\n'.join(lines)


def render_appendix(report, figure_prefix='figures/modebench_prompt_ablation'):
    require(report['status']=='complete', 'Appendix cannot present an incomplete panel as a result')
    families = [f for f in ('frontier','local') if any(m['family']==f for m in report['models'])]
    text = [r'\subsection{Removing strategy hints from unchanged problems}',r'\label{sec:prompt-hint-ablation}',
      r'We freeze a paired follow-up on Python factors, MathIR, and Pantry at Levels 2 and 3. '
      r'Within each of these six cells, the 32 smallest SHA-256 hashes of '
      r'$(20260911,\mathrm{level},\mathrm{domain},\mathrm{row\ index})$ select problems from the 128-row evaluation split, '
      r'without consulting outcomes. Both arms generate eight fresh draws per identical problem. '
      r'The original arm retains the published prompt. The neutral arm removes the Python suggestions to test small divisors '
      r'with nested conditionals or dispatch on listed values, the MathIR ordered algebraic-isolation suggestion, '
      r'and the Pantry preference for high-energy/protein, low-sodium ingredients such as seeds or oats. '
      r'MathIR changes ``those operations'' to ``the operations'' to preserve the menu-ID instruction. '
      r'The user problem, answer specification, executable verifier, and remaining format requirements are unchanged.',
      r'We report the paired neutral-minus-original effects on empirical $P_8=\texttt{pass@8}$ and '
      r'$D_8=\texttt{distinct@8}$, with $B_8=D_8-P_8$ as additional verified breadth. '
      r'All returned answers, including incorrect, empty, and truncated outputs, remain in their eight-draw groups; '
      r'only verifier-accepted outputs create a verified key. '
      r'Strict verification is primary; one unchanged, previously frozen formatting normalizer is a sensitivity analysis '
      r'and preserves every strict success and canonical key. Python outputs in both arms undergo the same serial, '
      r'warmed-verifier recheck, with original grades and any corrections retained in separate receipts.',
      r'Intervals use 20,000 whole-prompt bootstrap resamples, paired across arms and stratified by domain and level '
      r'(seed 20260911). Cells contribute equally to reported macros. These are pointwise exploratory intervals for '
      r'32 problems per cell; crossing zero does not establish equivalence or absence of a prompt effect. '
      r'A zero-width bootstrap interval records zero observed resampled variation, including all-tied $P_8$ outcomes; '
      r'it is not precise evidence of population equivalence. '
      r'Correct-pair collision and the certified-support uniform reference are secondary and retain their correct-pair denominators; '
      r'the machine-readable report includes joint eligibility counts and undefined cells. Eight sampled draws do not identify total unseen support.','']
    if report.get('experiment_status')=='partial_panels':
        omitted=', '.join(report['scope']['omitted_panels'])
        text.append(r'This appendix reports the complete '+', '.join(families)+r' panel only. The separately registered '+
                    tex_escape(omitted)+r' panel is not included here, and the overall local-plus-frontier experiment is incomplete.')
    if 'local' in families:
        text.append(r'For local models, we evaluate the initial Qwen2.5-0.5B model and fixed Level-2-trained DrGRPO and '
                    r'Re:Dr.GRPO checkpoints. Python and MathIR use matched training seeds 43--47; Pantry uses seeds 43 and 46. '
                    r'Level 3 is transfer evaluation. Seed means retain the same selected problem set across checkpoints. '
                    r'Five-seed intervals resample whole training seeds and paired prompts; Pantry reports both seed effects '
                    r'and intervals conditional on the two checkpoints. The native local syntax constraints and 192-token budget '
                    r'differ from hosted inference, so effects are interpreted within each model and interface.'+'\n')
    for family in families:
        for grading in GRADINGS:
            text.append(render_table(report,family,grading))
        text += [r'\begin{figure}[t]',r'\centering',
                 r'\includegraphics[width=\linewidth]{'+figure_prefix+'_'+family+r'.pdf}',
                 r'\caption{Prompt-hint removal effects for '+('hosted models' if family=='frontier' else 'local checkpoints')+
                 r', preserving every registered domain--level cell. Blue circles use strict verification; orange squares use '
                 r'the same frozen formatting normalizer. Horizontal bars show the intervals described in the tables; '
                 r'zero is the vertical reference. '+(r'Gray endpoint marks show the two individual Pantry seed effects. '
                 if family=='local' else '')+r'}',r'\label{fig:prompt-hints-'+family+'}',r'\end{figure}','']
    text.append(render_seed_ranges(report))
    return '\n\n'.join(text)


def make_figures(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = {}
    for family in ('frontier','local'):
        rows = display_rows(report,family,'strict')
        if not rows: continue
        fig,axes = plt.subplots(1,2,figsize=(8.0,max(4,0.29*len(rows)+1.0)),sharey=True)
        plotted = []
        for grading,color,marker,offset in [('strict','#2364AA','o',-.12),('normalized_secondary','#D56A25','s',.12)]:
            current = display_rows(report,family,grading)
            for i,row in enumerate(current):
                for ax,metric in zip(axes,('distinct8','pass8')):
                    record = row['metrics'][CONTRAST][metric]
                    y = len(rows)-1-i+offset
                    value,ci = record['estimate'],record['ci95']
                    if value is None: continue
                    if ci:
                        ax.plot(ci,[y,y],color=color,lw=1.3)
                    ax.plot(value,y,marker=marker,markersize=4,color=color,
                            label=('Strict' if grading=='strict' else 'Frozen normalized') if i==0 else None)
                    if row['seed_ranges'] and grading=='strict':
                        lo,hi = row['seed_ranges'][metric]
                        ax.plot([lo,hi],[y,y],'|',color='#555555',markersize=8)
                plotted.append({'grading':grading,**row})
        labels = [f"{r['label']} · {DOMAIN_LABELS[r['domain']]} L{r['level']}"+
                  (f" (s={r['seed_count']})" if family=='local' else '') for r in rows]
        axes[0].set_yticks(range(len(rows)),labels=labels[::-1],fontsize=7)
        for ax,label in zip(axes,(r'$\Delta D_8$: neutral − original',r'$\Delta P_8$: neutral − original')):
            ax.axvline(0,color='#777777',lw=.8,ls='--',zorder=0)
            ax.set_xlabel(label,fontsize=10)
            ax.grid(axis='x',alpha=.15)
            ax.tick_params(axis='y',length=0)
            ax.spines[['top','right','left']].set_visible(False)
        axes[0].legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.075),ncol=2)
        fig.tight_layout()
        stem = output/f'modebench_prompt_ablation_{family}'
        for suffix in ('pdf','png'):
            fig.savefig(stem.with_suffix('.'+suffix),dpi=220,bbox_inches='tight')
        plt.close(fig)
        provenance = {'schema':'prompt-ablation-figure-v1','family':family,
                      'report_sha256':file_sha(output/'analysis.json'),'plotted_records':plotted,
                      'outputs':{suffix:binding(stem.with_suffix('.'+suffix)) for suffix in ('pdf','png')}}
        write_json(stem.with_suffix('.json'),provenance)
        figures[family] = binding(stem.with_suffix('.json'))
    return figures


def write_artifacts(report,output):
    import csv
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    report['analyzer_source']=freeze_analysis_source(output)
    report['analysis_dependencies']={'audit_hosted_modebench_completion.py':binding(output/'analysis_code/audit_hosted_modebench_completion.py')}
    report['publication_source']={'directory':str(output.resolve()),'report_path':str((output/'analysis.json').resolve())}
    write_json(output/'analysis.json',report)
    fields = ['model_id','family','grading','cell','arm','metric','estimate','ci95_low','ci95_high',
              'defined_bootstrap_replicates']
    with (output/'all_cells.csv').open('w') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for model in report['models']:
            for grading,analysis in model['analyses'].items():
                for cell,record in analysis['cells'].items():
                    for arm in (*ARMS,CONTRAST):
                        for metric,value in record[arm].items():
                            ci = value['ci95'] or [None,None]
                            writer.writerow({'model_id':model['model_id'],'family':model['family'],
                                'grading':grading,'cell':cell,'arm':arm,'metric':metric,
                                'estimate':value['estimate'],'ci95_low':ci[0],'ci95_high':ci[1],
                                'defined_bootstrap_replicates':value['defined_bootstrap_replicates']})
    (output/'appendix.tex').write_text(render_appendix(report))
    figures = make_figures(report,output)
    (output/'README.md').write_text(
        '# Frozen prompt-hint ablation\n\nComplete registered panels only. `analysis.json` retains every per-prompt '
        'occupancy, failure count, conditional denominator, source binding, and interval. `all_cells.csv` exposes every '
        'checkpoint/model × domain × level × grading × arm/contrast × metric. `appendix.tex` gives all domain-level '
        'original→neutral P8/D8 cells and local seed means; figures show strict and frozen-normalized effects.\n\n'
        'P8 is the observed fraction of prompts with any correct draw. D8 counts distinct verified canonical keys. '
        'B8=D8−P8. No plugin pass@8 estimate, historical control, failure exclusion, or result-based cell selection is used.\n\n'
        'Local fixed-seed prompt intervals and hierarchical seed/prompt intervals are stored separately. Five-seed '
        'groups display the latter; two-seed Pantry groups display fixed-seed prompt intervals alongside both seed effects. '
        'The initial model has no training-seed replication.\n')
    paths = ['analysis.json','all_cells.csv','appendix.tex','README.md']
    write_json(output/'artifact_manifest.json',{'status':'complete','analyzer_source':binding(Path(__file__)),
               'outputs':{name:binding(output/name) for name in paths},'figures':figures})



def validate_local_registration(design,plan_path,plan):
    amendment_path = design['base']/'local/rng_seed_stride_amendment_v2.json'
    amendment = json.loads(amendment_path.read_text())
    require(Path(plan_path).resolve()==Path(amendment['new_plan_path']).resolve()
            and file_sha(plan_path)==amendment['new_plan_sha256']
            and amendment['outcomes_inspected'] is False
            and amendment['old_sampling_receipts']==0,
            'Superseded local startup data are not a production cohort')
    settings = plan['settings']
    require(settings.get('seed_stride')==16 and settings.get('sample_count')==8
            and settings.get('seed_policy')=='disjoint_n8_child_seed_blocks_per_problem_shared_across_arms_and_checkpoints',
            'Local child-seed independence amendment is missing')
    for name,digest in design['manifest']['code_sha256'].items():
        if name.startswith('src/oat_drgrpo/') or name=='ops/frontier_modebench_contract.py':
            matches = [v for path,v in plan['code_sha256'].items() if path.endswith('/'+name)]
            require(matches==[digest], 'Local executable grader differs from the frozen design: '+name)
    expected = Counter({('initial',None,None):1})
    for method in ('drgrpo','replay_drgrpo'):
        for domain in DOMAINS:
            for seed in ([43,46] if domain=='pantry_plan' else list(range(43,48))):
                expected[method,domain,seed]=1
    actual = Counter((c['training_method'],'pantry_plan' if c['domain']=='pantry' else c['domain'],c['training_seed'])
                     for c in plan['checkpoints'])
    require(actual==expected, 'Local checkpoint panel omitted or duplicated registered objectives/seeds')



def paired_difference_counts(original,neutral):
    require([r['row_sha256'] for r in original]==[r['row_sha256'] for r in neutral],
            'Paired diagnostic order changed')
    result={}
    for metric in ('pass8','distinct8','b8'):
        delta=np.asarray([b[metric]-a[metric] for a,b in zip(original,neutral)])
        result[metric]={'paired_prompts':len(delta),'zero_difference_prompts':int(np.sum(delta==0)),
                        'positive_difference_prompts':int(np.sum(delta>0)),
                        'negative_difference_prompts':int(np.sum(delta<0)),
                        'discordant_prompts':int(np.sum(delta!=0)),
                        'all_paired_differences_identical':bool(len(delta) and np.all(delta==delta[0]))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts/modebench_prompt_ablation_20260911')
    parser.add_argument('--scope',choices=('full','local','hosted'),default='full',help='Default requires both complete panels; a partial panel must be explicitly selected.')
    parser.add_argument('--hosted-registry',type=Path)
    parser.add_argument('--local-plan',type=Path)
    parser.add_argument('--grade-hosted',type=Path,help='One complete hosted run; isolated offline grading only.')
    parser.add_argument('--grade-local',help='One complete local checkpoint label; requires --local-plan.')
    parser.add_argument('--inventory',action='store_true',help='Write completeness with missing cells and stop; no effect estimates.')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.local_plan is None:
        args.local_plan=args.base/'local/plan_v2.json'
    if args.hosted_registry is None:
        args.hosted_registry=args.base/'hosted_analysis_runs.json'
    if args.scope=='local':args.hosted_registry=None
    elif args.scope=='hosted':args.local_plan=None
    if args.grade_hosted:
        require(not args.grade_local, 'Choose one isolated grading mode')
        result = grade_hosted(args.base,args.grade_hosted)
    elif args.grade_local:
        require(args.local_plan is not None,'--grade-local requires --local-plan')
        result = grade_local(args.base,args.local_plan,args.grade_local)
    elif args.inventory:
        result = inventory_report(authenticate_design(args.base),args.hosted_registry,args.local_plan)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            write_json(args.output,result)
    else:
        require(args.output is not None,'Final analysis requires --output directory')
        result = build_report(args.base,args.hosted_registry,args.local_plan)
        write_artifacts(result,args.output)
        result = {'status':'complete','models':len(result['models']),'output':str(args.output)}
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','models','prompt_results')},sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
