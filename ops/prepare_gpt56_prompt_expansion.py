#!/usr/bin/env python3
"""Prepare only the approved 360-prompt, 14,400-response Figure 8 extension."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'artifacts/frontier_modebench_gpt56sol_20260911/manifest.json').is_file())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_frontier_modebench import atomic, file_sha, now, sha

PLAN_DIR = ROOT / 'artifacts/frontier_temperature_20260911/prompt_expansion_32_per_cell'
DEFAULT_OUTPUT = PLAN_DIR / 'collection_v1'
APPROVED_PLAN_SHA256 = '80211566250851ae3726e36fa076855c4a35206606cd52ac97edb4d4e1991213'
SCHEMA = 'gpt56-temperature-expansion-collection-v1'
TEMPERATURES = (0.0, 0.5, 1.0, 1.5, 2.0)
PLAN_FILES = ('plan.json', 'selection.jsonl', 'new_request_hashes.jsonl', 'retained_measurement_bindings.jsonl')
SERVED_MODEL = 'gpt-5.6-sol-2026-07-09'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identity(item):
    return item['level'], item['domain'], item['row_index']


def slug(temperature):
    return 't' + str(float(temperature)).replace('.', 'p')


def write_lines(path, records):
    with Path(path).open('x') as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')


def verify_binding(binding):
    require(file_sha(ROOT / binding['path']) == binding['sha256'], 'Changed approved source: ' + binding['path'])


def derive_requests(plan_dir):
    """Reconstruct approved row/rank identities and exact payloads from source."""
    plan_dir = Path(plan_dir)
    require(file_sha(plan_dir / 'plan.json') == APPROVED_PLAN_SHA256, 'Approved plan digest differs')
    plan = json.loads((plan_dir / 'plan.json').read_text())
    require(plan['schema'] == 'gpt56-temperature-prompt-expansion-plan-v1', 'Unexpected plan schema')
    require(tuple(plan['temperatures']) == TEMPERATURES and plan['counts']['new_sample_slots'] == 14400, 'Approved scope differs')
    for name, binding in plan['artifacts'].items():
        require(file_sha(plan_dir / name) == binding['sha256'], 'Changed approved plan artifact: ' + name)
    for binding in plan['sources'].values():
        verify_binding(binding)
    reference = ROOT / Path(plan['sources']['original_manifest']['path']).parent
    original_manifest = json.loads((reference / 'manifest.json').read_text())
    for name, digest in original_manifest['artifact_sha256'].items():
        require(file_sha(reference / name) == digest, 'Changed original artifact: ' + name)
    for name, digest in original_manifest['code_sha256'].items():
        require(file_sha(reference / 'code' / name) == digest, 'Changed original verifier source: ' + name)
    source_rows = read_lines(reference / 'rows.jsonl')
    lookup = {identity(row): row for row in source_rows}
    require(len(source_rows) == len(lookup) == 1920, 'Original row inventory differs')
    cells = defaultdict(list)
    for row in source_rows:
        cells[row['level'], row['domain']].append(row)
    require(len(cells) == 15 and {len(v) for v in cells.values()} == {128}, 'Original cells differ')
    ranks, selected = {}, set()
    seed = plan['selection']['seed']
    for values in cells.values():
        ranked = sorted(values, key=lambda row: hashlib.sha256((seed + ':' + sha(row)).encode()).hexdigest())
        for rank, row in enumerate(ranked, 1):
            ranks[identity(row)] = rank
        selected.update(identity(row) for row in ranked[:32])
    selection = read_lines(plan_dir / 'selection.jsonl')
    require(len(selection) == 480 and {identity(s) for s in selection} == selected, 'Selection is not the approved 32-row prefix')
    additional = set()
    for item in selection:
        key = identity(item)
        row = lookup[key]
        rank = ranks[key]
        require(item['row_sha256'] == sha(row) and item['selection_rank_in_cell'] == rank, 'Selection row/rank binding differs')
        require(item['sample_indices'] == list(range(8)), 'Selection draw slots differ')
        require(item['cohort'] == ('existing_120' if rank <= 8 else 'additional_360'), 'Selection cohort differs')
        if rank > 8:
            additional.add(key)
    rows = [row for row in source_rows if identity(row) in additional]
    require(len(rows) == 360 and set(Counter((r['level'], r['domain']) for r in rows).values()) == {24}, 'Expected 24 new prompts per cell')
    approved = read_lines(plan_dir / 'new_request_hashes.jsonl')
    approved_map = {(x['temperature'], x['sample_id']): x for x in approved}
    require(len(approved) == len(approved_map) == 14400, 'Approved request inventory differs')
    requests = []
    original_requests = read_lines(reference / 'requests.jsonl')
    require(len(original_requests) == 15360 and len({x['sample_id'] for x in original_requests}) == 15360, 'Reference sample slots differ')
    for original in original_requests:
        require(original['row_sha256'] == sha(lookup[identity(original)]) and original['request_sha256'] == sha(original['request']), 'Original request binding differs')
        if identity(original) not in additional:
            continue
        for temperature in TEMPERATURES:
            item = deepcopy(original)
            item['original_sample_id'] = original['sample_id']
            item['sample_id'] = slug(temperature).upper() + '__' + original['sample_id']
            item['reference_request_sha256'] = original['request_sha256']
            item['request']['reasoning']['effort'] = 'none'
            item['request']['temperature'] = temperature
            item['request_sha256'] = sha(item['request'])
            item['temperature_condition'] = temperature
            expected = approved_map.get((temperature, original['sample_id']))
            require(expected is not None, 'Unapproved request slot')
            for key in ('level', 'domain', 'row_index', 'sample_index', 'row_sha256', 'reference_request_sha256'):
                require(item[key] == expected[key], 'Approved request identity differs: ' + key)
            require(item['request_sha256'] == expected['planned_request_sha256'], 'Approved payload hash differs')
            requests.append(item)
    require(len(requests) == len({x['sample_id'] for x in requests}) == 14400, 'Derived request inventory differs')
    for arm in plan['arms']:
        for binding in arm['existing_sources'].values():
            verify_binding(binding)
    retained = read_lines(plan_dir / 'retained_measurement_bindings.jsonl')
    require(len(retained) == 4800, 'Retained sample count differs')
    for item in retained:
        verify_binding(item['sample_receipt'])
        verify_binding(item['raw_receipt'])
    return plan, reference, original_manifest, rows, requests


def prepare(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    require(output == DEFAULT_OUTPUT.resolve(), 'Collection must use its isolated approved namespace')
    if (output / 'manifest.json').exists():
        from run_gpt56_prompt_expansion import validate_inventory
        return validate_inventory(output)[0]
    require(not output.exists() or not any(output.iterdir()), 'Refusing unregistered nonempty output directory')
    plan, reference, original, rows, requests = derive_requests(PLAN_DIR)
    output.mkdir(parents=True, exist_ok=True)
    for name in PLAN_FILES:
        shutil.copyfile(PLAN_DIR / name, output / ('approved_' + name))
    shutil.copyfile(reference / 'datasets.json', output / 'datasets.json')
    write_lines(output / 'rows.jsonl', rows)
    write_lines(output / 'requests.jsonl', requests)
    authorization = {'schema': 'gpt56-temperature-expansion-authorization-v1', 'recorded_at_utc': now(),
                     'user_instruction': 'Okay... do it?', 'approved_plan_sha256': APPROVED_PLAN_SHA256,
                     'scope': 'Expand Figure 8 from 120 to 480 prompts, retaining all 4,800 prior responses and collecting 14,400 additional responses at five supported temperatures.',
                     'api_calls_during_preparation': 0}
    atomic(output / 'authorization.json', authorization)
    hashes = {}
    for name in original['code_sha256']:
        destination = output / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / 'code' / name, destination)
        hashes[name] = file_sha(destination)
    for name in ('prepare_gpt56_prompt_expansion.py', 'run_gpt56_prompt_expansion.py', 'postprocess_gpt56_temperature_expanded480.py'):
        destination = output / 'code/ops' / name
        shutil.copyfile(ROOT / 'ops' / name, destination)
        hashes['ops/' + name] = file_sha(destination)
    artifacts = ['datasets.json', 'rows.jsonl', 'requests.jsonl', 'authorization.json'] + ['approved_' + name for name in PLAN_FILES]
    manifest = deepcopy(original)
    manifest.pop('operational_amendments', None)
    manifest.update(schema=SCHEMA, prepared_at_utc=now(), experiment_condition='gpt56_none_temperature_expansion_480_v1',
                    reasoning_effort='none', temperature=list(TEMPERATURES), temperatures=list(TEMPERATURES),
                    sample_count=8, prompt_count=360, combined_prompt_count=480, request_count=14400,
                    prompts_per_cell=24, combined_prompts_per_cell=32, retained_request_count=4800,
                    reference_run=str(reference), reference_manifest_sha256=file_sha(reference / 'manifest.json'),
                    approved_plan_sha256=APPROVED_PLAN_SHA256, endpoint=plan['protocol']['endpoint'],
                    collection_order='For each original new-prompt/sample slot, consecutive temperatures 0, .5, 1, 1.5, 2.',
                    returned_controls_required={'model': 'gpt-5.6-sol', 'reasoning.effort': 'none',
                                               'temperature': 'exact request temperature', 'top_p': 0.98,
                                               'x-ms-served-model': SERVED_MODEL},
                    code_sha256=hashes, artifact_sha256={name: file_sha(output / name) for name in artifacts},
                    preflight_sample_ids=[x['sample_id'] for x in requests[:5]],
                    max_attempts_per_slot=8,
                    notes=['Prepared offline; no API calls.', 'The five retained preflights are part of the 14,400 new sample slots.',
                           'Invalid answers, refusals and truncations remain outcomes.',
                           'Transport failures are retained separately; an unfinished dispatch without a durable receipt blocks automatic retry.',
                           'Original 120-prompt results are immutable and enter combined analysis through approved retained bindings.'])
    atomic(output / 'manifest.json', manifest)
    from run_gpt56_prompt_expansion import validate_inventory
    validate_inventory(output)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({k: result[k] for k in ('schema', 'prompt_count', 'request_count', 'combined_prompt_count', 'temperatures')}, indent=2))
