#!/usr/bin/env python3
"""Freeze new original/neutral cohorts with the original provider controls."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_native_prompt_ablation import atomic, file_sha, now, sha, validate_profile, native_request, verify_snapshot

BASE = ROOT / 'artifacts/modebench_prompt_ablation_20260911'
SOURCES = {'gpt56sol': 'gpt-5.6-sol', 'gpt54': 'gpt-5.4', 'grok43': 'grok-4.3'}
CONDITION = 'prompt_hint_ablation_v1'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def identity(row):
    return row['level'], row['domain'], row['row_index']


def source_profile(manifest, request):
    field = 'input' if 'input' in request else 'messages'
    profile = {'model': manifest['model'], 'endpoint': manifest['endpoint'],
               'protocol': 'responses' if field == 'input' else 'chat_completions',
               'request_parameters': {k: v for k, v in request.items() if k not in ('model', field)}}
    validate_profile(profile)
    return profile


def adapt_items(originals, templates, profile, arm):
    old_lookup = {item['sample_id']: item for item in originals}
    result = []
    for template in templates:
        old = old_lookup[template['sample_id']]
        if old['request_sha256'] != sha(old['request']) or template['request_sha256'] != sha(template['request']):
            raise ValueError('Corrupt source request')
        if old['row_sha256'] != template['row_sha256'] or identity(old) != identity(template):
            raise ValueError('Changed task identity')
        field = 'input' if 'input' in old['request'] else 'messages'
        messages = template['request']['input']
        if messages[1] != old['request'][field][1]:
            raise ValueError('The user problem must remain unchanged')
        if arm == 'original' and messages != old['request'][field]:
            raise ValueError('Original arm differs from original source prompt')
        payload = native_request(messages, profile)
        if {k: v for k, v in payload.items() if k != field} != {k: v for k, v in old['request'].items() if k != field}:
            raise ValueError('Provider sampling controls changed')
        item = {k: v for k, v in old.items() if k not in ('request', 'request_sha256')}
        item.update(request=payload, request_sha256=sha(payload), group_id=old['sample_id'],
                    choice_index=0, reference_request_sha256=old['request_sha256'],
                    prompt_arm=arm, experiment_condition=CONDITION)
        result.append(item)
    # Both arms share a frozen, outcome-independent shuffle, so their collectors
    # traverse matched sample slots concurrently rather than entire cells apart.
    result.sort(key=lambda item: hashlib.sha256(('hint-ablation-order-v1:' + item['sample_id']).encode()).hexdigest())
    return result


def prepare(slug, arm, base=BASE):
    if slug not in SOURCES or arm not in ('original', 'neutral'):
        raise ValueError('Unregistered deployment or prompt arm')
    base = Path(base).resolve()
    root_manifest_path = base / 'manifest.json'
    root_manifest = json.loads(root_manifest_path.read_text())
    # Authenticated preparation artifacts; support common manifest field names.
    root_hashes = root_manifest.get('artifact_sha256', root_manifest.get('artifacts', {}))
    if not root_hashes:
        raise ValueError('Root preparation requires an artifact hash manifest')
    for name, digest in root_hashes.items():
        if isinstance(digest, dict):
            digest = digest.get('sha256')
        if file_sha(base / name) != digest:
            raise ValueError('Root preparation artifact changed: ' + name)
    output = base / 'hosted' / slug / arm
    reference = ROOT / ('artifacts/frontier_modebench_' + slug + '_20260911')
    source_manifest = json.loads((reference / 'manifest.json').read_text())
    verify_snapshot(reference, source_manifest)
    if source_manifest['model'] != SOURCES[slug]:
        raise ValueError('Unexpected source deployment')
    if (output / 'manifest.json').exists():
        existing = json.loads((output / 'manifest.json').read_text())
        if (existing.get('experiment_condition') != CONDITION or existing.get('prompt_arm') != arm
                or existing.get('ablation_manifest_sha256') != file_sha(root_manifest_path)):
            raise ValueError('Existing cohort belongs to a different experiment')
        verify_snapshot(output, existing)
        return existing
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing nonempty unfrozen output')
    rows = read_jsonl(base / 'rows.jsonl')
    templates = read_jsonl(base / arm / 'requests.jsonl')
    originals = read_jsonl(reference / 'requests.jsonl')
    profile = source_profile(source_manifest, originals[0]['request'])
    requests = adapt_items(originals, templates, profile, arm)
    if len(rows) != 192 or len(requests) != 1536 or len({x['sample_id'] for x in requests}) != 1536:
        raise ValueError('Expected the registered 192 prompts and 1536 sample slots')
    row_lookup = {identity(row): row for row in rows}
    for row_id, row in row_lookup.items():
        members = [item for item in requests if identity(item) == row_id]
        if sorted(item['sample_index'] for item in members) != list(range(8)) or any(item['row_sha256'] != sha(row) for item in members):
            raise ValueError('Invalid eight-draw task group')
    groups = [{'group_id': item['group_id'], 'request': item['request'],
               'request_sha256': item['request_sha256'], 'sample_ids': [item['sample_id']],
               'sample_count': 1, 'protocol': profile['protocol']} for item in requests]
    output.mkdir(parents=True)
    for name, records in [('rows.jsonl', rows), ('requests.jsonl', requests), ('http_requests.jsonl', groups)]:
        write_jsonl(output / name, records)
    shutil.copyfile(reference / 'datasets.json', output / 'datasets.json')
    atomic(output / 'model_profile.json', profile)
    code_hashes = {}
    for name in source_manifest['code_sha256']:
        destination = output / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / 'code' / name, destination)
        code_hashes[name] = file_sha(destination)
    for name in ['ops/evaluate_native_prompt_ablation.py', 'ops/prepare_hosted_prompt_ablation.py',
                 'ops/frontier_modebench_normalization.py']:
        destination = output / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        code_hashes[name] = file_sha(destination)
    manifest = {'schema': 'frontier-modebench-native-chat-responses-v1', 'prepared_at_utc': now(),
                'experiment_condition': CONDITION, 'prompt_arm': arm,
                'ablation_manifest_sha256': file_sha(root_manifest_path), 'fresh_response_cohort': True,
                'model': profile['model'], 'endpoint': profile['endpoint'], 'protocol': profile['protocol'],
                'model_profile': profile, 'model_profile_sha256': sha(profile),
                'sample_count': 8, 'prompt_count': len(rows), 'request_count': len(requests),
                'http_request_count': len(groups), 'samples_per_http_request': 1,
                'max_output_tokens': 8192, 'requested_settings': profile['request_parameters'],
                'training': False, 'tools': [], 'conversation_state': False,
                'reference_run': str(reference), 'reference_manifest_sha256': file_sha(reference / 'manifest.json'),
                'code_sha256': code_hashes,
                'artifact_sha256': {name: file_sha(output / name) for name in
                                   ['datasets.json', 'rows.jsonl', 'requests.jsonl', 'http_requests.jsonl', 'model_profile.json']},
                'notes': ['Fresh contemporaneous original and neutral samples; no historical samples count as controls.',
                          'Only the registered strategy guidance differs; every user problem and generation control is preserved.',
                          'Run both arms concurrently in the same shuffled sample order.',
                          'All returned model failures and truncations count; provider errors and retries remain separately recorded.',
                          'Local and hosted sampling interfaces differ; estimate prompt effects within each model/interface.']}
    atomic(output / 'manifest.json', manifest)
    verify_snapshot(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    entries = []
    for slug in SOURCES:
        for arm in ('original', 'neutral'):
            manifest = prepare(slug, arm, args.base)
            run_dir = args.base.resolve() / 'hosted' / slug / arm
            entries.append({'model_id': slug, 'model': SOURCES[slug], 'family': 'frontier', 'arm': arm,
                            'run_dir': str(run_dir), 'manifest_sha256': file_sha(run_dir / 'manifest.json')})
    registry = {'experiment_condition': CONDITION, 'ablation_manifest_sha256': file_sha(args.base / 'manifest.json'), 'runs': entries}
    path = args.base / 'hosted_analysis_runs.json'
    if path.exists() and json.loads(path.read_text()) != registry:
        raise ValueError('Existing hosted registry differs')
    atomic(path, registry)
    print(json.dumps({'prepared_runs': len(entries), 'total_responses': 9216, 'registry': str(path)}))


if __name__ == '__main__':
    main()
