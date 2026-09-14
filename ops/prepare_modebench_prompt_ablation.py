#!/usr/bin/env python3
"""Freeze a paired strategy-hint ablation without reading model outcomes.

This command only reads source inputs/code and writes preparation artifacts. It
never generates responses, submits jobs, rewrites dataset rows, or selects cases
using outcomes. Original and neutral arms both require fresh generation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts/frontier_modebench_gpt56sol_20260911'
OUTPUT = ROOT / 'artifacts/modebench_prompt_ablation_20260911'
SCHEMA = 'modebench-prompt-hint-ablation-v1'
SELECTION_SEED = 20260911
PROMPTS_PER_CELL = 32
SAMPLE_COUNT = 8
DOMAINS = ('python_factors', 'mathir', 'pantry_plan')
CELLS = tuple((level, domain) for level in (2, 3) for domain in DOMAINS)
ARMS = ('original', 'neutral')
COMMON = ('Solve the executable constraint problem carefully. You may reason briefly, '
          'but end with exactly one final answer inside \\boxed{}.')
PYTHON_HINT = (' Test small divisors with nested conditional expressions, for example '
               '2 if n % 2 == 0 else 3 if n % 3 == 0 else 5, but adapt the tests to '
               'every listed case. You may instead dispatch on each listed value.')
MATHIR_HINT = (' Use algebraic isolation: move the right-side x term left, remove the '
               'left constant, then divide by the combined coefficient.')
PANTRY_HINT = (' Prefer allowed high-energy/protein, very-low-sodium ingredients, '
               'especially seeds or oats.')
ORIGINAL_SYSTEMS = {
    'python_factors': COMMON + ' Construct one allowed lambda expression.' + PYTHON_HINT
                      + ' Output exactly the boxed lambda.',
    'mathir': COMMON + MATHIR_HINT + ' Match those operations to the shuffled menu IDs.',
    'pantry_plan': COMMON + PANTRY_HINT + ' Choose stepped amounts, check every bound, '
                   'and output 2 to 4 ingredient_id=grams pairs.',
}
HINTS = {'python_factors': PYTHON_HINT, 'mathir': MATHIR_HINT, 'pantry_plan': PANTRY_HINT}


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value, sort_keys=True, allow_nan=False) + '\n'
                            for value in values))


def identity(item):
    return item['level'], item['domain'], item['row_index']


def pair_id(item):
    return f"L{item['level']}_{item['domain']}_{item['row_index']:03d}"


def selection_hash(item, seed=SELECTION_SEED):
    # Deliberately excludes row contents, specs, support counts, and outcomes.
    return text_sha(f"{seed}\0{item['level']}\0{item['domain']}\0{item['row_index']}")


def neutral_system(domain, original):
    if domain not in ORIGINAL_SYSTEMS or original != ORIGINAL_SYSTEMS[domain]:
        raise ValueError('Unrecognized frozen original system wording: ' + domain)
    result = original.replace(HINTS[domain], '', 1)
    if domain == 'mathir':
        result = result.replace('Match those operations', 'Match the operations', 1)
    return result


def neutral_messages(level, domain, messages):
    if (level, domain) not in CELLS:
        raise ValueError('Only the six predeclared Level 2/3 cells are ablated')
    if (len(messages) != 2 or [m.get('role') for m in messages] != ['system', 'user']
            or any(set(m) != {'role', 'content'} or not isinstance(m['content'], str)
                   for m in messages)):
        raise ValueError('Expected exactly the frozen system and user messages')
    result = deepcopy(messages)
    result[0]['content'] = neutral_system(domain, messages[0]['content'])
    return result


def verify_snapshot(directory, manifest):
    directory = Path(directory)
    for name, digest in manifest['artifact_sha256'].items():
        if file_sha(directory / name) != digest:
            raise ValueError('Frozen artifact changed: ' + name)
    for name, digest in manifest['code_sha256'].items():
        if file_sha(directory / 'code' / name) != digest:
            raise ValueError('Frozen code changed: ' + name)


def select_rows(rows):
    cells = defaultdict(list)
    seen = set()
    for row in rows:
        key = identity(row)
        if key in seen:
            raise ValueError('Duplicate source row identity')
        seen.add(key)
        if key[:2] in CELLS:
            cells[key[:2]].append(row)
    if set(cells) != set(CELLS) or any(len(rows) != 128 for rows in cells.values()):
        raise ValueError('Expected exactly 128 rows in each of the six affected cells')
    if any({r['row_index'] for r in items} != set(range(128)) for items in cells.values()):
        raise ValueError('Expected original source row indices 0 through 127')
    selected, ledger = [], []
    for cell in CELLS:
        ranked = sorted(cells[cell], key=lambda r: (selection_hash(r), identity(r)))
        for rank, row in enumerate(ranked, 1):
            chosen = rank <= PROMPTS_PER_CELL
            ledger.append({'level': row['level'], 'domain': row['domain'],
                           'row_index': row['row_index'], 'pair_id': pair_id(row),
                           'selection_sha256': selection_hash(row), 'rank_in_cell': rank,
                           'selected': chosen, 'row_sha256': sha(row),
                           'problem_sha256': text_sha(row['problem']),
                           'answer_spec_sha256': sha(row['answer'])})
            if chosen:
                selected.append(row)
    return sorted(selected, key=identity), ledger


def validate_reference(rows, requests):
    lookup = {identity(row): row for row in rows}
    expected_cells = {(level, domain) for level in (1, 2, 3)
                      for domain in ('graph_coloring', 'countdown', *DOMAINS)}
    counts = Counter(key[:2] for key in lookup)
    if (len(rows) != 1920 or len(lookup) != 1920 or set(counts) != expected_cells
            or set(counts.values()) != {128}):
        raise ValueError('Expected the complete frozen 1,920-row reference input')
    groups = defaultdict(list)
    seen_samples = set()
    for request in requests:
        key = identity(request)
        if key not in lookup or request['row_sha256'] != sha(lookup[key]):
            raise ValueError('Source request row identity/hash mismatch')
        if request['request_sha256'] != sha(request['request']):
            raise ValueError('Source request payload hash mismatch')
        if request['sample_id'] in seen_samples:
            raise ValueError('Duplicate source sample identity')
        seen_samples.add(request['sample_id'])
        if request['sample_id'] != pair_id(request) + '_' + str(request['sample_index']):
            raise ValueError('Unexpected source sample ID contract')
        messages = request['request']['input']
        if key[:2] in CELLS:
            neutral_messages(*key[:2], messages)
            if messages[1]['content'] != lookup[key]['problem']:
                raise ValueError('Source user message differs from untouched problem')
        groups[key].append(request)
    if len(requests) != 15360 or set(groups) != set(lookup):
        raise ValueError('Expected 15,360 complete source requests')
    for group in groups.values():
        if len(group) != SAMPLE_COUNT or {r['sample_index'] for r in group} != set(range(SAMPLE_COUNT)):
            raise ValueError('Source draws must be exactly 0 through 7 for every row')
        if len({sha(r['request']) for r in group}) != 1:
            raise ValueError('Source draw requests do not have identical settings and messages')
    return groups


def transformations():
    result = {}
    for domain in DOMAINS:
        original = ORIGINAL_SYSTEMS[domain]
        neutral = neutral_system(domain, original)
        result[domain] = {
            'levels': [2, 3], 'original_system': original, 'neutral_system': neutral,
            'original_system_sha256': text_sha(original), 'neutral_system_sha256': text_sha(neutral),
            'edits': [{'operation': 'delete', 'exact_text': HINTS[domain]}] +
                     ([{'operation': 'replace', 'old': 'Match those operations',
                        'new': 'Match the operations',
                        'reason': 'Remove dangling reference after strategy deletion; preserve menu-ID instruction.'}]
                      if domain == 'mathir' else []),
            'unified_diff': ''.join(difflib.unified_diff([original + '\n'], [neutral + '\n'],
                                                       fromfile='original/system', tofile='neutral/system')),
        }
    return result


def prepare(output=OUTPUT, reference=REFERENCE):
    output, reference = Path(output), Path(reference)
    if output.resolve() == reference.resolve() or reference.resolve() in output.resolve().parents:
        raise ValueError('Preparation must use a separate output directory')
    if (output / 'manifest.json').exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['schema'] != SCHEMA or manifest['reference_run'] != str(reference.resolve()):
            raise ValueError('Existing preparation has a different protocol or reference')
        verify_snapshot(output, manifest)
        if file_sha(reference / 'manifest.json') != manifest['reference_manifest_sha256']:
            raise ValueError('Source manifest changed since preparation')
        for arm in ARMS:
            verify_snapshot(output / arm, json.loads((output / arm / 'manifest.json').read_text()))
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite nonempty unfrozen preparation directory')
    source_manifest = json.loads((reference / 'manifest.json').read_text())
    # Only declared input artifacts and code are read; no samples, summaries, or results.
    if set(source_manifest['artifact_sha256']) != {'rows.jsonl', 'requests.jsonl', 'datasets.json'}:
        raise ValueError('Unexpected source input artifact allowlist')
    verify_snapshot(reference, source_manifest)
    rows = read_jsonl(reference / 'rows.jsonl')
    requests = read_jsonl(reference / 'requests.jsonl')
    grouped = validate_reference(rows, requests)
    selected, ledger = select_rows(rows)
    output.mkdir(parents=True, exist_ok=True)
    code_hashes = dict(source_manifest['code_sha256'])
    for name in code_hashes:
        destination = output / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / 'code' / name, destination)
    for name, source in {
        'ops/prepare_modebench_prompt_ablation.py': Path(__file__),
        'ops/frontier_modebench_normalization.py': ROOT / 'ops/frontier_modebench_normalization.py',
    }.items():
        shutil.copyfile(source, output / 'code' / name)
        code_hashes[name] = file_sha(output / 'code' / name)
    shutil.copyfile(reference / 'manifest.json', output / 'reference_manifest.json')
    shutil.copyfile(reference / 'datasets.json', output / 'datasets.json')
    write_jsonl(output / 'rows.jsonl', selected)
    write_json(output / 'selection.json', {
        'seed': SELECTION_SEED, 'prompts_per_cell': PROMPTS_PER_CELL,
        'population_per_cell': 128, 'hash_encoding': 'UTF-8',
        'hash_input': 'str(seed) + NUL + str(level) + NUL + domain + NUL + str(row_index)',
        'ranking': 'ascending SHA256 hex, ties broken by (level, domain, row_index); first 32/cell',
        'selection_fields': ['seed', 'level', 'domain', 'row_index'], 'outcomes_read': False,
        'source_rows_path': str((reference / 'rows.jsonl').resolve()),
        'source_rows_sha256': file_sha(reference / 'rows.jsonl'),
        'candidate_count': len(ledger), 'selected_count': len(selected), 'candidates': ledger,
    })
    write_json(output / 'transformations.json', transformations())
    profiles = {f'{level}/{domain}': source_manifest['profiles'][f'{level}/{domain}']
                for level, domain in CELLS}
    prompt_records, arm_manifests = [], {}
    for arm in ARMS:
        directory = output / arm
        directory.mkdir()
        for name in code_hashes:
            target = directory / 'code' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output / 'code' / name, target)
        shutil.copyfile(output / 'rows.jsonl', directory / 'rows.jsonl')
        shutil.copyfile(output / 'datasets.json', directory / 'datasets.json')
        items = []
        for row in selected:
            group = sorted(grouped[identity(row)], key=lambda r: r['sample_index'])
            original = group[0]['request']['input']
            messages = original if arm == 'original' else neutral_messages(row['level'], row['domain'], original)
            prompt_records.append({
                'pair_id': pair_id(row), 'arm': arm, 'level': row['level'], 'domain': row['domain'],
                'row_index': row['row_index'], 'row_sha256': sha(row),
                'problem_sha256': text_sha(row['problem']), 'answer_spec_sha256': sha(row['answer']),
                'messages': messages, 'messages_sha256': sha(messages),
                'source_sample_id': group[0]['sample_id'],
                'source_request_sha256': group[0]['request_sha256'],
            })
            for source in group:
                item = deepcopy(source)
                item['request']['input'] = deepcopy(messages)
                item.update(request_sha256=sha(item['request']), arm=arm, pair_id=pair_id(row),
                            reference_request_sha256=source['request_sha256'],
                            messages_sha256=sha(messages))
                items.append(item)
        write_jsonl(directory / 'requests.jsonl', items)
        arm_manifest = {
            **{key: source_manifest[key] for key in ('model', 'endpoint', 'max_output_tokens',
                'reasoning_effort', 'temperature', 'top_p', 'seed', 'training', 'tools', 'conversation_state')},
            'schema': SCHEMA + '-arm', 'arm': arm, 'fresh_generation_required': True,
            'profiles': profiles, 'sample_count': SAMPLE_COUNT, 'prompt_count': len(selected),
            'request_count': len(items), 'reference_run': str(reference.resolve()),
            'reference_manifest_sha256': file_sha(reference / 'manifest.json'),
            'code_sha256': code_hashes,
            'artifact_sha256': {name: file_sha(directory / name)
                                for name in ('rows.jsonl', 'datasets.json', 'requests.jsonl')},
            'notes': ['Request profile is the original GPT-5.6 Sol template, not a deployment decision.',
                      'sample_id is scoped by arm and actual deployment run directory.',
                      'No existing model responses may be reused for either arm.',
                      'datasets.json retains full source provenance; rows.jsonl contains only the selected subset.'],
        }
        write_json(directory / 'manifest.json', arm_manifest)
        arm_manifests[arm] = file_sha(directory / 'manifest.json')
    write_jsonl(output / 'prompts.jsonl', prompt_records)
    schedule = []
    for row in sorted(selected, key=lambda r: selection_hash(r)):
        first = int(selection_hash(row), 16) % 2
        for draw in range(SAMPLE_COUNT):
            for arm in (ARMS if (first + draw) % 2 == 0 else tuple(reversed(ARMS))):
                schedule.append({'order_index': len(schedule), 'arm': arm, 'pair_id': pair_id(row),
                                 'sample_id': pair_id(row) + '_' + str(draw), 'sample_index': draw,
                                 'level': row['level'], 'domain': row['domain'], 'row_index': row['row_index']})
    write_jsonl(output / 'execution_order.jsonl', schedule)
    protocol = {
        'schema': SCHEMA, 'condition': 'prompt_hint_ablation_v1',
        'selection_seed': SELECTION_SEED, 'prompts_per_cell': PROMPTS_PER_CELL,
        'sample_count': SAMPLE_COUNT, 'arms': list(ARMS), 'paired_prompt_count': len(selected),
        'request_count_per_model': len(schedule),
        'affected_cells': [{'level': level, 'domain': domain} for level, domain in CELLS],
        'frontier_targets': ['gpt-5.6-sol', 'gpt-5.4', 'grok-4.3'],
        'local_targets': 'Separate frozen run manifests select checkpoints before generation.',
        'primary_comparison': 'Within-model paired original versus neutral prompt means, with eight fresh draws per arm.',
        'sampling_controls': 'Use identical decoding, token budgets, constraints, verifier, and checkpoint within each model pair.',
        'run_order': 'execution_order.jsonl interleaves arms and counterbalances first arm four times per prompt.',
        'fresh_both_arms': True, 'outcome_based_selection': False,
        'source_scope': 'Six affected Level 2/3 domain cells; 32 of 128 source prompts per cell; no Level 1, Countdown, or Graph claim.',
        'unchanged': ['row record', 'user message', 'answer spec', 'metadata', 'format and correctness requirements'],
        'only_prompt_edits': 'transformations.json contains strategy-span deletion and MathIR those-to-the grammar repair.',
        'limitations': ['The original benchmark has already been evaluated; this is a prospectively frozen follow-up, not an untouched preregistration.',
                       'No claim of identical provider RNG draws or matched compute across different models.',
                       'This prompt intervention does not identify training-induced collapse or full unseen mode support.',
                       'All selected cells and runs must be reported, including failures and incomplete runs.'],
        'hash_contract': {'object_sha256': 'SHA256 of json.dumps(sort_keys=True,separators=(comma,colon),allow_nan=False), UTF-8',
                          'problem_system_file_sha256': 'SHA256 of exact UTF-8 string bytes or file bytes'},
    }
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'design.json', {
        **protocol, 'reference_run': str(reference.resolve()),
        'reference_manifest_sha256': file_sha(reference / 'manifest.json'),
        'rows_sha256': file_sha(output / 'rows.jsonl'), 'prompts_sha256': file_sha(output / 'prompts.jsonl'),
        'selection_sha256': file_sha(output / 'selection.json'),
        'transformations_sha256': file_sha(output / 'transformations.json'),
        'arm_manifest_sha256': arm_manifests,
        'arm_rows_sha256': {arm: file_sha(output / arm / 'rows.jsonl') for arm in ARMS},
        'arm_requests_sha256': {arm: file_sha(output / arm / 'requests.jsonl') for arm in ARMS},
        'contract_path': 'code/ops/frontier_modebench_contract.py',
        'contract_sha256': code_hashes['ops/frontier_modebench_contract.py'],
        'normalizer_path': 'code/ops/frontier_modebench_normalization.py',
        'normalizer_sha256': code_hashes['ops/frontier_modebench_normalization.py'],
        'verifier_sha256': code_hashes['src/oat_drgrpo/math_grader.py'],
        'source_template_sha256': code_hashes['src/oat_drgrpo/templates.py'],
    })
    (output / 'README.md').write_text(
        '# Paired ModeBench strategy-hint ablation\n\n'
        'Frozen preparation only: 192 identical problems, six Level 2/3 cells, '
        '32 outcome-independent hash-selected prompts per cell, two fresh arms, eight draws per arm.\n\n'
        'The neutral arm deletes the Python divisor/dispatch suggestions, MathIR ordered isolation '
        'suggestion, and Pantry ingredient preference. MathIR changes “those” to “the” to retain '
        'the menu-ID instruction. All user text, answer specs, and remaining instructions are unchanged. '
        '`transformations.json` records exact text and diffs.\n\n'
        '`design.json` and `manifest.json` freeze identities; `selection.json` includes all candidate '
        'ranks. `prompts.jsonl` contains both message pairs. `original/` and `neutral/` contain '
        'the source request schema and code snapshots; these are input templates, with no outcomes. '
        'Sample IDs are scoped by arm and deployment. Both arms require new model generations.\n\n'
        'Use `execution_order.jsonl` to interleave arms with counterbalanced order where the runner '
        'supports it. Actual model/checkpoint and decoding profiles must be frozen per run, '
        'identically within each model pair. See `protocol.json` for scope and limitations.\n'
    )
    artifact_names = ['reference_manifest.json', 'datasets.json', 'rows.jsonl', 'prompts.jsonl',
                      'selection.json', 'transformations.json', 'protocol.json', 'design.json',
                      'execution_order.jsonl', 'README.md']
    artifact_names += [f'{arm}/{name}' for arm in ARMS
                       for name in ('manifest.json', 'rows.jsonl', 'requests.jsonl', 'datasets.json')]
    manifest = {
        'schema': SCHEMA, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'reference_run': str(reference.resolve()),
        'reference_manifest_sha256': file_sha(reference / 'manifest.json'),
        'reference_artifact_sha256': source_manifest['artifact_sha256'],
        'selection_seed': SELECTION_SEED, 'sample_count': SAMPLE_COUNT,
        'prompt_count': len(selected), 'arm_count': len(ARMS), 'request_count_per_model': len(schedule),
        'profiles': profiles, 'fresh_both_arms': True, 'outcomes_read': False,
        'code_sha256': code_hashes,
        'artifact_sha256': {name: file_sha(output / name) for name in artifact_names},
    }
    write_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.output, args.reference)
    print(json.dumps({key: manifest[key] for key in ('schema', 'prompt_count', 'arm_count',
                                                    'sample_count', 'request_count_per_model')}, indent=2))


if __name__ == '__main__':
    main()
