#!/usr/bin/env python3
"""Freeze a separate Opus 5 Python prompt condition; never call the API.

The existing eight-sample synthetic diagnostic selected this wording. Every
held-out Python prompt is included, irrespective of its original response.
Original rows, mathematical constraints, verifier, and sampling settings stay
fixed; only the user wording and presence of the system message change.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_claude_modebench import atomic, file_sha, now, sha

REFERENCE = ROOT / 'artifacts/frontier_modebench_claude_opus5_20260911'
DIAGNOSTIC = ROOT / 'artifacts/frontier_models_comparison_20260911/benign_prompt_diagnostic_round2'
DEFAULT_OUTPUT = ROOT / 'artifacts/frontier_modebench_claude_opus5_python_plain_20260911'
CONDITION = 'python_plain_user_no_system_v1'
PROMPT_PREFIX = 'For the inputs n in '
PROMPT_SUFFIX = (
    ', write one arithmetic expression in Python notation lambda n: EXPR. '
    'For every listed input, the result d must be an integer satisfying '
    '1 < d < n and n % d == 0. Use only integer literals, n, +, -, *, //, %, '
    'comparisons, Boolean operators, and conditional expressions. '
    'Any proper divisor meeting these conditions is acceptable. '
    'Return only the one-line lambda inside \\boxed{}.'
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def public_cases(row):
    if row['domain'] != 'python_factors':
        raise ValueError('This condition applies only to Python factor tasks')
    matches = re.findall(r'for each n in (\[[0-9, ]+\])\.', row['problem'])
    if len(matches) != 1:
        raise ValueError('Expected one public input list in the original problem')
    cases = json.loads(matches[0])
    answer = row['answer']
    answer = json.loads(answer) if isinstance(answer, str) else answer
    if (not cases or any(type(n) is not int or n <= 2 for n in cases)
            or answer.get('cases') != cases
            or answer.get('verifier') != 'python_factor_function'
            or answer.get('python_version') != 'factor-v1'):
        raise ValueError('Public input list differs from the frozen mathematical task')
    return cases


def plain_prompt(row):
    return PROMPT_PREFIX + json.dumps(public_cases(row)) + PROMPT_SUFFIX


def adapt_request(original, row):
    if original['row_sha256'] != sha(row) or original['request_sha256'] != sha(original['request']):
        raise ValueError('Original row or request digest mismatch')
    payload = copy.deepcopy(original['request'])
    if (payload.get('model') != 'claude-opus-5'
            or payload.get('messages') != [{'role': 'user', 'content': row['problem']}]
            or 'system' not in payload):
        raise ValueError('Unexpected original model or prompt interface')
    del payload['system']
    payload['messages'] = [{'role': 'user', 'content': plain_prompt(row)}]
    item = {key: value for key, value in original.items()
            if key not in ('request', 'request_sha256', 'reference_request_sha256')}
    item.update(request=payload, request_sha256=sha(payload), condition=CONDITION,
                reference_request_sha256=original['request_sha256'])
    return item


def write_jsonl(path, rows):
    with Path(path).open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def prepare(output=DEFAULT_OUTPUT, reference=REFERENCE, diagnostic=DIAGNOSTIC):
    output, reference, diagnostic = map(lambda p: Path(p).resolve(), (output, reference, diagnostic))
    if output == reference or output in reference.parents or reference in output.parents:
        raise ValueError('The changed-prompt condition requires a separate run directory')
    if (output / 'manifest.json').exists():
        saved = json.loads((output / 'manifest.json').read_text())
        if saved.get('experiment_condition') != CONDITION or saved.get('request_count') != 3072:
            raise ValueError('Existing directory is not this frozen condition')
        for name, digest in saved['artifact_sha256'].items():
            if file_sha(output / name) != digest:
                raise ValueError('Frozen condition artifact changed: ' + name)
        for name, digest in saved['code_sha256'].items():
            if file_sha(output / 'code' / name) != digest:
                raise ValueError('Frozen condition source changed: ' + name)
        return saved
    original_manifest = json.loads((reference / 'manifest.json').read_text())
    if original_manifest['model'] != 'claude-opus-5' or original_manifest['request_count'] != 15360:
        raise ValueError('Expected the original complete-design Opus 5 cohort')
    for name, digest in original_manifest['artifact_sha256'].items():
        if file_sha(reference / name) != digest:
            raise ValueError('Original artifact changed: ' + name)
    for name, digest in original_manifest['code_sha256'].items():
        if file_sha(reference / 'code' / name) != digest:
            raise ValueError('Original verifier/runner source changed: ' + name)
    rows = [row for row in read_jsonl(reference / 'rows.jsonl') if row['domain'] == 'python_factors']
    if Counter(row['level'] for row in rows) != {1: 128, 2: 128, 3: 128}:
        raise ValueError('Expected all 384 original Python prompts')
    lookup = {(row['level'], row['domain'], row['row_index']): row for row in rows}
    original_requests = [item for item in read_jsonl(reference / 'requests.jsonl')
                         if item['domain'] == 'python_factors']
    requests = [adapt_request(item, lookup[item['level'], item['domain'], item['row_index']])
                for item in original_requests]
    if len(requests) != 3072 or len({item['sample_id'] for item in requests}) != 3072:
        raise ValueError('Expected exactly 3,072 unique fixed requests')
    for identity in lookup:
        indices = [item['sample_index'] for item in requests
                   if (item['level'], item['domain'], item['row_index']) == identity]
        if sorted(indices) != list(range(8)):
            raise ValueError('Every original prompt must have all eight sample indices')
    diagnostic_requests = read_jsonl(diagnostic / 'probe_requests.jsonl')
    diagnostic_items = [item for item in diagnostic_requests if item['condition'] == 'no_system_field']
    if len(diagnostic_items) != 8:
        raise ValueError('Expected the original eight-sample synthetic diagnostic')
    for item in diagnostic_items:
        payload = item['request']
        if (item['request_sha256'] != sha(payload) or 'system' in payload
                or payload['messages'] != [{'role': 'user', 'content': plain_prompt(item['fixture'])}]):
            raise ValueError('Proposed wording differs from the saved diagnostic')
        if any(public_cases(row) == public_cases(item['fixture']) for row in rows):
            raise ValueError('Synthetic diagnostic overlaps the held-out Python inputs')
    diagnostic_result = json.loads((diagnostic / 'probe_analysis.json').read_text())
    cell = diagnostic_result['cells']['no_system_field/python_factors']
    if cell['responses'] != 8 or cell['native_refusals'] != 0 or cell['strict_valid'] != 8:
        raise ValueError('Saved synthetic condition did not pass its diagnostic')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Refusing to overwrite an unfrozen nonempty run directory')
    write_jsonl(output / 'rows.jsonl', rows)
    write_jsonl(output / 'requests.jsonl', requests)
    shutil.copyfile(reference / 'datasets.json', output / 'datasets.json')
    atomic(output / 'prompt_condition.json', {
        'condition': CONDITION, 'system_field': 'omitted',
        'user_template': PROMPT_PREFIX + '{original_public_input_list}' + PROMPT_SUFFIX,
        'scope': {'domain': 'python_factors', 'levels': [1, 2, 3], 'prompts': 384, 'requests': 3072},
        'changes': ['Plain mathematical user wording', 'System field omitted at every level'],
        'unchanged': ['Public input integers', 'Proper-divisor conditions', 'Allowed expression grammar',
                      'Boxed one-line lambda output', 'Frozen verifier and canonical modes',
                      'All 384 held-out Python prompts and eight sample indices',
                      'Model, adaptive thinking, medium effort, and 8192 output-token limit'],
        'selection': 'Post-hoc prompt adaptation selected using synthetic development diagnostics',
        'not_original_prompt_cohort': True, 'not_format_normalization': True,
        'not_retry_until_accepted': True, 'no_training': True,
        'diagnostic_run': str(diagnostic),
        'diagnostic_sha256': {name: file_sha(diagnostic / name)
                              for name in ('probe_requests.jsonl', 'fixtures.json', 'probe_analysis.json', 'probe_grades.jsonl')},
    })
    code_hashes = {}
    for name in original_manifest['code_sha256']:
        destination = output / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / 'code' / name, destination)
        code_hashes[name] = file_sha(destination)
    builder_name = 'ops/prepare_frontier_python_plain_prompt.py'
    shutil.copyfile(Path(__file__), output / 'code' / builder_name)
    code_hashes[builder_name] = file_sha(output / 'code' / builder_name)
    manifest = copy.deepcopy(original_manifest)
    manifest.update(prepared_at_utc=now(), experiment_condition=CONDITION,
                    condition_scope='Python only, all three levels; separate changed-prompt cohort',
                    original_prompt_cohort=False, prompt_count=384, request_count=3072,
                    reference_run=str(reference), reference_manifest_sha256=file_sha(reference / 'manifest.json'),
                    reference_artifact_sha256=original_manifest['artifact_sha256'],
                    code_sha256=code_hashes,
                    artifact_sha256={name: file_sha(output / name) for name in
                                     ('datasets.json', 'rows.jsonl', 'requests.jsonl', 'prompt_condition.json')})
    manifest['notes'] = [
        'Separate Python-only prompt condition; never splice responses into the original full-model cohort.',
        'All original Python test rows included, regardless of original refusal/correctness.',
        'Rows retain their original problem text as provenance; requests.jsonl records actual adapted messages.',
        'Input integers are parsed from the public problem, checked against the frozen task; no solution hints added.',
        'No system field; user wording is identical to the saved successful synthetic diagnostic template.',
        'Original native runner, mathematical verifier, output interface, and generation settings retained.',
        'HTTP-200 refusals are saved outcomes and never automatically retried.',
        'The removed original system example suggested a divisor strategy; prompt changes may affect mode concentration.',
        'The full copied datasets.json describes source datasets; rows.jsonl limits this condition to Python.',
    ]
    atomic(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / ('.' + args.output.name + '.prepare.lock')).open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = prepare(args.output)
    print(json.dumps({key: result[key] for key in ('model', 'experiment_condition', 'prompt_count', 'request_count', 'artifact_sha256')}, indent=2))
