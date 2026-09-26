#!/usr/bin/env python3
"""Freeze matched temperature-only subsets of original hosted ModeBench runs.

No API calls. Prompt selection uses only frozen row identities, never outcomes.
The original prompts, graders and provider settings remain fixed except for
the explicitly requested temperature. Conditions must be analyzed separately
from the full original cohorts, and require a working provider control.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_chat_frontier_modebench import atomic, file_sha, now, sha, validate_profile, verify_snapshot

SELECTION_SEED = 'frontier-temperature-ablation-20260911-v1'
SOURCES = {'grok43': 'grok-4.3', 'kimi_k3': 'FW-Kimi-K3', 'deepseek_v4_pro': 'DeepSeek-V4-Pro'}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identity(item):
    return item['level'], item['domain'], item['row_index']


def select_rows(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[row['level'], row['domain']].append(row)
    if len(cells) != 15 or any(len(group) != 128 for group in cells.values()):
        raise ValueError('Expected the full original five-domain, three-level row inventory')
    selected = set()
    for group in cells.values():
        ordered = sorted(group, key=lambda row: hashlib.sha256((SELECTION_SEED + ':' + sha(row)).encode()).hexdigest())
        selected.update(identity(row) for row in ordered[:8])
    return [row for row in rows if identity(row) in selected]


def adapt_item(item, temperature):
    if item['request_sha256'] != sha(item['request']):
        raise ValueError('Original request digest differs')
    result = copy.deepcopy(item)
    result['reference_request_sha256'] = item['request_sha256']
    result['request']['temperature'] = temperature
    result['request_sha256'] = sha(result['request'])
    result['temperature_condition'] = temperature
    return result


def prepare(slug, temperature, output):
    if slug not in SOURCES or temperature not in (1.0, 1.5):
        raise ValueError('This registered small ablation supports only the approved models and temperatures')
    reference = ROOT / f'artifacts/frontier_modebench_{slug}_20260911'
    source_manifest = json.loads((reference / 'manifest.json').read_text())
    verify_snapshot(reference, source_manifest)
    if source_manifest['model'] != SOURCES[slug] or source_manifest['request_count'] != 15360:
        raise ValueError('Wrong original model cohort')
    output = Path(output).resolve()
    if output == reference.resolve() or reference.resolve() in output.parents:
        raise ValueError('Ablations must be separate from the original evidence')
    if (output / 'manifest.json').exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if (manifest.get('experiment_condition') != 'temperature_ablation_v1'
                or manifest.get('temperature') != temperature or manifest['model'] != SOURCES[slug]):
            raise ValueError('Existing condition differs')
        verify_snapshot(output, manifest)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing an unfrozen nonempty output directory')
    rows = select_rows(read_jsonl(reference / 'rows.jsonl'))
    lookup = {identity(row): row for row in rows}
    originals = [item for item in read_jsonl(reference / 'requests.jsonl') if identity(item) in lookup]
    if len(rows) != 120 or len(originals) != 960 or len({item['sample_id'] for item in originals}) != 960:
        raise ValueError('Expected 120 prompts and 960 unique eight-draw sample slots')
    requests = [adapt_item(item, temperature) for item in originals]
    for key, row in lookup.items():
        group = [item for item in requests if identity(item) == key]
        if sorted(item['sample_index'] for item in group) != list(range(8)):
            raise ValueError('Incomplete eight-draw prompt')
        if any(item['row_sha256'] != sha(row) for item in group):
            raise ValueError('Row hash differs from original')
    groups = [{'group_id': item['group_id'], 'request': item['request'],
               'request_sha256': item['request_sha256'], 'sample_ids': [item['sample_id']],
               'sample_count': 1, 'protocol': 'chat_completions'} for item in requests]
    if len({group['group_id'] for group in groups}) != 960:
        raise ValueError('Expected independent single-choice HTTP requests')
    profile = copy.deepcopy(source_manifest['model_profile'])
    profile['request_parameters']['temperature'] = temperature
    validate_profile(profile)
    if any({key: value for key, value in item['request'].items() if key != 'messages'} !=
           {'model': profile['model'], **profile['request_parameters']} for item in requests):
        raise ValueError('Payload parameters differ from frozen condition profile')
    output.mkdir(parents=True)
    for name, records in [('rows.jsonl', rows), ('requests.jsonl', requests), ('http_requests.jsonl', groups)]:
        with (output / name).open('x') as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    shutil.copyfile(reference / 'datasets.json', output / 'datasets.json')
    atomic(output / 'model_profile.json', profile)
    atomic(output / 'temperature_condition.json', {
        'condition': 'temperature_ablation_v1', 'requested_temperature': temperature,
        'selection_seed': SELECTION_SEED, 'selection_rule': 'Lowest eight SHA256(seed + colon + canonical row SHA256) values in each domain-level cell; retain original request order.',
        'selection_uses_outcomes': False, 'prompts_per_cell': 8, 'sample_count': 8,
        'cells': 15, 'prompt_count': 120, 'request_count': 960,
        'row_sha256': [sha(row) for row in rows],
        'changed_request_fields': ['temperature'],
        'unchanged': ['system/user prompt bytes', 'held-out row and mathematical verifier', 'sample indices',
                      'provider reasoning setting', 'output token limit', 'other sampling parameters'],
        'reference_run': str(reference.resolve()), 'original_prompt_cohort': False,
        'original_prompt_bytes': True, 'no_training': True,
        'provider_control_requirement': 'API acceptance alone cannot establish that temperature takes effect; analyze only with documented support and saved native preflight.',
    })
    code_hashes = {}
    for name in source_manifest['code_sha256']:
        destination = output / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / 'code' / name, destination)
        code_hashes[name] = file_sha(destination)
    builder = output / 'code/ops/prepare_frontier_temperature_ablation.py'
    shutil.copyfile(Path(__file__), builder)
    code_hashes['ops/prepare_frontier_temperature_ablation.py'] = file_sha(builder)
    manifest = copy.deepcopy(source_manifest)
    manifest.update(prepared_at_utc=now(), experiment_condition='temperature_ablation_v1',
                    original_prompt_cohort=False, original_prompt_bytes=True,
                    prompt_count=120, request_count=960, http_request_count=960,
                    model_profile=profile, model_profile_sha256=sha(profile),
                    requested_settings=profile['request_parameters'], temperature=temperature,
                    reference_run=str(reference.resolve()), reference_manifest_sha256=file_sha(reference / 'manifest.json'),
                    reference_artifact_sha256=source_manifest['artifact_sha256'], code_sha256=code_hashes,
                    artifact_sha256={name: file_sha(output / name) for name in
                                     ['datasets.json', 'rows.jsonl', 'requests.jsonl', 'http_requests.jsonl', 'model_profile.json', 'temperature_condition.json']})
    manifest['notes'] = [
        'Separate matched-subset sampling ablation; never splice into the original full cohort.',
        'Selection uses frozen public row identities only, not correctness, collision, refusals or model outputs.',
        'All five domains and three levels contribute eight prompts with all eight draws.',
        'The two explicit temperatures are the only payload-field change within a model.',
        'Original default sampling and provider-exposed controls must be distinguished from requested settings.',
        'Strict and previously frozen format-normalized results remain separate.',
        'Low mode counts can reflect invalid answers; report accuracy and correct-pair collision together.',
        'Eight prompts per cell is a small exploratory ablation; intervals resample entire prompts with eight draws intact.',
    ]
    atomic(output / 'manifest.json', manifest)
    verify_snapshot(output, manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slug', choices=sorted(SOURCES), required=True)
    parser.add_argument('--temperature', type=float, choices=(1.0, 1.5), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.slug, args.temperature, args.output)
    print(json.dumps({key: result[key] for key in ['model', 'temperature', 'prompt_count', 'request_count', 'artifact_sha256']}, indent=2))
