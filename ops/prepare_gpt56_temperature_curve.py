#!/usr/bin/env python3
"""Prepare an unrun, separately labeled GPT no-reasoning temperature curve."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_frontier_modebench import atomic, file_sha, now, sha
from audit_hosted_modebench_completion import load_inventory
from prepare_frontier_temperature_ablation import SELECTION_SEED, identity, read_jsonl, select_rows

REFERENCE = ROOT / 'artifacts/frontier_modebench_gpt56sol_20260911'
BASE = ROOT / 'artifacts/frontier_temperature_20260911'
TEMPERATURES = (0.0, 0.5, 1.0, 1.5, 2.0)
CONDITION = 'gpt56_none_temperature_curve_v1'


def prepare(temperature, output):
    if temperature not in TEMPERATURES:
        raise ValueError('Unsupported candidate temperature')
    output = Path(output).resolve()
    if output == REFERENCE.resolve() or REFERENCE.resolve() in output.parents:
        raise ValueError('Temperature curve must be separate from original evidence')
    if (output / 'manifest.json').exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest.get('experiment_condition') != CONDITION or manifest.get('temperature') != temperature:
            raise ValueError('Existing candidate condition differs')
        load_inventory(output, expected_samples=960)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing a nonempty unregistered candidate directory')
    original = json.loads((REFERENCE / 'manifest.json').read_text())
    for name, expected in original['artifact_sha256'].items():
        if file_sha(REFERENCE / name) != expected:
            raise ValueError('Original frozen artifact changed: ' + name)
    for name, expected in original['code_sha256'].items():
        if file_sha(REFERENCE / 'code' / name) != expected:
            raise ValueError('Original frozen source changed: ' + name)
    if original['model'] != 'gpt-5.6-sol' or original['schema'] != 'frontier-modebench-responses-v1':
        raise ValueError('Unexpected GPT reference schema')
    rows = select_rows(read_jsonl(REFERENCE / 'rows.jsonl'))
    selected = {identity(row): row for row in rows}
    requests = []
    for item in read_jsonl(REFERENCE / 'requests.jsonl'):
        if identity(item) not in selected:
            continue
        if item['request_sha256'] != sha(item['request']) or item['row_sha256'] != sha(selected[identity(item)]):
            raise ValueError('Original request identity mismatch')
        changed = copy.deepcopy(item)
        changed['reference_request_sha256'] = item['request_sha256']
        changed['request']['reasoning']['effort'] = 'none'
        changed['request']['temperature'] = temperature
        changed['request_sha256'] = sha(changed['request'])
        changed['temperature_condition'] = temperature
        requests.append(changed)
    if len(rows) != 120 or len(requests) != 960 or len({item['sample_id'] for item in requests}) != 960:
        raise ValueError('Expected 120 fixed prompts and 960 independent sample slots')
    for key in selected:
        if sorted(item['sample_index'] for item in requests if identity(item) == key) != list(range(8)):
            raise ValueError('Missing or duplicated sample slot')
    probe = BASE / 'gpt56_control_probe/probe_results.json'
    probe_results = json.loads(probe.read_text())
    accepted = [record for record in probe_results['results'] if record['reasoning_effort'] == 'none'
                and record['requested_temperature'] == 1.5 and record['http_status'] == 200
                and record['returned_temperature'] == 1.5]
    if len(accepted) != 1:
        raise ValueError('Missing saved evidence for the candidate no-reasoning control')
    output.mkdir(parents=True)
    for name, records in [('rows.jsonl', rows), ('requests.jsonl', requests)]:
        with (output / name).open('x') as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    shutil.copyfile(REFERENCE / 'datasets.json', output / 'datasets.json')
    shutil.copyfile(probe, output / 'capability_probe_results.json')
    atomic(output / 'temperature_condition.json', {
        'condition': CONDITION, 'requested_temperature': temperature, 'reasoning_effort': 'none',
        'selection_seed': SELECTION_SEED, 'selection_uses_outcomes': False,
        'selection_rule': 'Same lowest-eight canonical-row-hash selection in each domain/level as the separate Grok/Kimi temperature ablation.',
        'prompts_per_cell': 8, 'cells': 15, 'sample_count': 8, 'prompt_count': 120, 'request_count': 960,
        'changed_fields_vs_original': ['reasoning.effort', 'temperature'],
        'changed_fields_within_curve': ['temperature'],
        'original_prompt_bytes': True, 'original_prompt_cohort': False, 'no_training': True,
        'reference_run': str(REFERENCE.resolve()), 'row_sha256': [sha(row) for row in rows],
        'capability_probe_source': str(probe.resolve()), 'capability_probe_source_sha256': file_sha(probe),
        'capability_limit': 'Only none/T1.5 is supported by the saved candidate probe; every eventual arm requires a successful native preflight and returned-control validation.',
        'interpretation': 'Reasoning effort changes from medium to none relative to the original cohort. Only temperature varies within the candidate arms; this is not the original medium-reasoning model configuration.',
        'prepared_only': True, 'api_calls_during_preparation': 0,
    })
    code_hashes = {}
    for name in original['code_sha256']:
        destination = output / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REFERENCE / 'code' / name, destination)
        code_hashes[name] = file_sha(destination)
    for name in ('prepare_gpt56_temperature_curve.py', 'prepare_frontier_temperature_ablation.py',
                 'evaluate_chat_frontier_modebench.py', 'audit_hosted_modebench_completion.py'):
        source = ROOT / 'ops' / name
        destination = output / 'code/ops' / name
        if destination.exists() and file_sha(destination) != file_sha(source):
            raise ValueError('Would replace an original frozen code file')
        shutil.copyfile(source, destination)
        code_hashes['ops/' + name] = file_sha(destination)
    manifest = copy.deepcopy(original)
    manifest.pop('operational_amendments', None)
    manifest.update(prepared_at_utc=now(), experiment_condition=CONDITION,
                    original_prompt_cohort=False, original_prompt_bytes=True, prompt_count=120,
                    request_count=960, reasoning_effort='none', temperature=temperature,
                    reference_run=str(REFERENCE.resolve()), reference_manifest_sha256=file_sha(REFERENCE / 'manifest.json'),
                    reference_artifact_sha256=original['artifact_sha256'], code_sha256=code_hashes,
                    requested_settings={'reasoning': {'effort': 'none'}, 'temperature': temperature,
                                        'max_output_tokens': 8192, 'store': False},
                    artifact_sha256={name: file_sha(output / name) for name in
                                     ['datasets.json', 'rows.jsonl', 'requests.jsonl', 'temperature_condition.json', 'capability_probe_results.json']})
    manifest['notes'] = ['Prepared candidate only; preparation makes no model calls.',
                         'Separate no-reasoning curve, not the original medium-reasoning GPT cohort.',
                         'Only temperature varies across the candidate arms; original prompt bytes and verifier are fixed.',
                         'Every eventual native response must echo the requested temperature and reasoning effort.',
                         'Refusals, invalid answers and truncations remain outcomes; API or transport failures are separately retained.',
                         'Small exploratory subset: eight fixed prompts per domain and level, all eight stateless draws retained.']
    atomic(output / 'manifest.json', manifest)
    load_inventory(output, expected_samples=960)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temperature', type=float, choices=TEMPERATURES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.temperature, args.output)
    print(json.dumps({key: manifest[key] for key in ('model', 'experiment_condition', 'reasoning_effort', 'temperature', 'prompt_count', 'request_count')}, indent=2))
