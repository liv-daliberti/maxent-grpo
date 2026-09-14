#!/usr/bin/env python3
"""Add certified references for SHA ranks17--32, preserving all240 old objects.

Only benchmark inputs and frozen executable verifiers are read. Higher-level
Pantry witnesses are allocations checked on the actual boxed-answer interface.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'artifacts/modebench_discovery_all_levels_sol512_20260912'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
LEVELS, SEED = (1, 2, 3), 20260911
LEGACY_SHA256 = '4ea33166250d056b466785457d98d1462164ef416ce69fd462995992bfdf4674'
PRIOR_SHA256 = '54eac7ed94822927abb926f267b35b98a24f581088b8ea102dda41196ba58e62'
SURFACE_HELPER = PRIOR / 'support_code/gpt56_all_levels_discovery_support.py'
SURFACE_SHA256 = 'b14f04a2b3c5a20f2345a79ed6138c3583e3fc7b8f1973a66da780fafa1e1348'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identity(row):
    return row['level'], row['domain'], row['row_index']


def pair_id(row):
    return f"L{row['level']}_{row['domain']}_{row['row_index']:03d}"


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_selection(rows, all_rows, source_rows):
    """Authenticate all480 exact source rows and disjoint next240 identities."""
    require(len(rows) == 240 and len({identity(r) for r in rows}) == 240, 'Expected240 unique new rows')
    require(len(all_rows) == 480 and len({identity(r) for r in all_rows}) == 480, 'Expected480 unique combined rows')
    expected_all, expected_new = {}, {}
    for level in LEVELS:
        for domain in DOMAINS:
            candidates = [r for r in source_rows if identity(r)[:2] == (level, domain)]
            require(len(candidates) == 128 and {r['row_index'] for r in candidates} == set(range(128)),
                    'Expected128 distinct native-source candidates per cell')
            ranked = sorted(candidates, key=lambda r: (
                hashlib.sha256(f"{SEED}\0{r['level']}\0{r['domain']}\0{r['row_index']}".encode()).hexdigest(), identity(r)))
            expected_all.update({identity(r): r for r in ranked[:32]})
            expected_new.update({identity(r): r for r in ranked[16:32]})
    require({identity(r) for r in all_rows} == set(expected_all), 'Combined identities differ from first32 SHA ranking')
    require({identity(r) for r in rows} == set(expected_new), 'New identities differ from SHA ranks17 through32')
    for row in [*all_rows, *rows]:
        require(row == expected_all[identity(row)], 'Selected row differs from frozen native source bytes')


def certify_pantry_allocation(row, grade, project_support):
    """Construct independent valid allocations, never send masks to L2/L3."""
    require(row['level'] in (2, 3) and row['domain'] == 'pantry_plan', 'Expected higher-level Pantry row')
    spec = json.loads(row['answer']) if isinstance(row['answer'], str) else row['answer']
    ingredient_ids = sorted(ingredient['id'] for ingredient in spec['ingredients'])
    require(len(set(ingredient_ids)) == len(ingredient_ids) and 1 <= len(ingredient_ids) <= 12,
            'Invalid or excessive Pantry ingredient inventory')
    minimum, maximum = spec['min_ingredients'], spec['max_ingredients']
    require(1 <= minimum <= maximum <= len(ingredient_ids), 'Invalid Pantry support-size bounds')
    witnesses = {}
    for size in range(minimum, maximum + 1):
        for selected in combinations(ingredient_ids, size):
            projected = project_support(selected, spec)
            if projected is None:
                continue
            allocation = ';'.join(f'{ingredient}={grams}' for ingredient, grams in projected.allocations_g)
            text = '\\boxed{' + allocation + '}'
            outcome = grade(row['level'], row['domain'], row, text)
            require(outcome['verified'], 'Hosted adapter rejects projected Pantry allocation')
            key = outcome['canonical_key']
            require(key not in witnesses, 'Two Pantry supports share a canonical mode')
            witnesses[key] = {'text': text, 'graded_text': outcome['graded_text'], 'canonical_key': key}
    keys = sorted(witnesses)
    require(len(keys) == spec['certified_mode_count'] and keys, 'Pantry witness count differs from source certificate')
    digest = hashlib.sha256('\n'.join(sorted(key.removeprefix('pantry_plan:pantry-v1:') for key in keys)).encode()).hexdigest()
    require(digest == spec['certified_support_sha256'], 'Pantry witness digest differs from source certificate')
    for key, witness in witnesses.items():
        outcome = grade(row['level'], row['domain'], row, witness['text'])
        require(outcome['verified'] and outcome['canonical_key'] == key, 'Frozen hosted verifier rejects Pantry witness')
    return {'pair_id': pair_id(row), 'level': row['level'], 'domain': row['domain'], 'row_index': row['row_index'],
            'row_sha256': object_sha(row), 'support_count': len(keys), 'support_kind': 'certified_lower_bound',
            'source_field': 'answer.certified_mode_count', 'declared_answer_mode_count': spec['certified_mode_count'],
            'canonical_keys': keys, 'key_sha256': hashlib.sha256(json.dumps(keys, separators=(',', ':')).encode()).hexdigest(),
            'witnesses': witnesses, 'actual_hosted_surface_verified': True,
            'rationale': 'Independent deterministic search over every permitted ingredient subset of the complete prompt-local pantry constructs a feasible allocation; every witness is checked through the frozen original L2/L3 boxed-allocation adapter. Ingredient-set modes retain the conservative certified-lower-bound convention.'}


def build_reference(rows_path, all_rows_path, source_rows_path, legacy_path, code_root, enumerator_path, prior_report_path):
    require(sha(legacy_path) == LEGACY_SHA256, 'Changed frozen240-reference certificate')
    require(sha(prior_report_path) == PRIOR_SHA256, 'Changed frozen prior analysis')
    require(sha(SURFACE_HELPER) == SURFACE_SHA256, 'Changed frozen support surface helper')
    source_manifest_path = Path(source_rows_path).parent / 'manifest.json'
    source_manifest = json.loads(source_manifest_path.read_text())
    legacy = json.loads(Path(legacy_path).read_text())
    require(source_manifest['model'] == 'gpt-5.6-sol', 'Native source has a different model')
    require(sha(source_rows_path) == source_manifest['artifact_sha256'][Path(source_rows_path).name], 'Native source rows changed')
    for name in ('native_source_rows', 'native_source_manifest'):
        old = legacy['sources'][name]
        actual = source_rows_path if name == 'native_source_rows' else source_manifest_path
        require(sha(actual) == old['sha256'], 'Native source differs from prior certificate')
    rows, all_rows = read_rows(rows_path), read_rows(all_rows_path)
    audit_selection(rows, all_rows, read_rows(source_rows_path))
    by_id, new_ids = {pair_id(r): r for r in all_rows}, {pair_id(r) for r in rows}
    refs = dict(legacy['references'])
    require(len(refs) == 240 and set(refs).isdisjoint(new_ids) and set(refs) | new_ids == set(by_id),
            'Retained reference identities differ from first16 cohort')
    old_rows_binding = legacy['sources']['all_rows']
    require(sha(old_rows_binding['path']) == old_rows_binding['sha256'], 'Changed frozen prior rows')
    old_rows = {pair_id(r): r for r in read_rows(old_rows_binding['path'])}
    require(set(old_rows) == set(refs), 'Prior row inventory differs from prior support')
    for key, ref in refs.items():
        require(old_rows[key] == by_id[key], 'Retained row differs from prior cohort')
        if 'row_sha256' in ref:
            require(ref['row_sha256'] == object_sha(by_id[key]), 'Retained support row hash changed')
    require(not any(name.startswith('oat_drgrpo') for name in sys.modules), 'Use a fresh process for frozen verifier imports')
    code_root = Path(code_root).resolve()
    code_manifest_path = code_root.parent / 'manifest.json'
    code_manifest = json.loads(code_manifest_path.read_text())
    for name, expected in code_manifest['code_sha256'].items():
        require(sha(code_root / name) == expected, 'Frozen source changed: ' + name)
    for name, expected in source_manifest['code_sha256'].items():
        if name.startswith('src/oat_drgrpo/') or name == 'ops/frontier_modebench_contract.py':
            require(code_manifest['code_sha256'].get(name) == expected, 'Verifier differs from original native contract: ' + name)
    require(sha(enumerator_path) == legacy['sources']['graph_countdown_helper']['sha256'], 'Graph/Countdown helper changed')
    sys.path.insert(0, str(code_root / 'src'))
    contract = load_file(code_root / 'ops/frontier_modebench_contract.py', '_sol32_frozen_contract')
    mathir = importlib.import_module('oat_drgrpo.mathir')
    grader = importlib.import_module('oat_drgrpo.math_grader')
    pantry = importlib.import_module('oat_drgrpo.pantry_support_action')
    helper = load_file(enumerator_path, '_sol32_frozen_enumerator')
    surface = load_file(SURFACE_HELPER, '_sol32_frozen_surface_support')
    for row in sorted(rows, key=identity):
        if row['domain'] in ('graph_coloring', 'countdown'):
            ref = helper.certify_row(row, grader.validated_modebench_outcome_key)
            # Check each known witness on the precise hosted adapter as well.
            for key, expression in ref['witnesses'].items():
                outcome = contract.grade_response(row['level'], row['domain'], row, '\\boxed{' + expression + '}')
                require(outcome['verified'] and outcome['canonical_key'] == key, 'Hosted adapter rejects enumeration witness')
            ref.update(row_sha256=object_sha(row), actual_hosted_surface_verified=True)
        elif row['domain'] == 'pantry_plan' and row['level'] > 1:
            ref = certify_pantry_allocation(row, contract.grade_response, pantry.project_pantry_support)
        else:
            ref = surface.certify_surface_row(row, contract.grade_response, mathir.enumerate_mathir_action_menu_validations)
        require(ref['pair_id'] not in refs, 'Duplicate support identity')
        refs[ref['pair_id']] = ref
        print(json.dumps({'certified': ref['pair_id'], 'count': ref['support_count'], 'kind': ref['support_kind']}), flush=True)
    for name, module in list(sys.modules.items()):
        if name.startswith('oat_drgrpo') and getattr(module, '__file__', None):
            require(Path(module.__file__).resolve().is_relative_to(code_root), 'Verifier import escaped frozen source: ' + name)
    cells = defaultdict(list)
    for ref in refs.values():
        cells[f"level{ref['level']}/{ref['domain']}"].append(ref)
    require(len(cells) == 15 and {len(rs) for rs in cells.values()} == {32}, 'Missing support cell or unequal cohort size')
    summaries = {cell: {'prompts': len(rs), 'mean_support_count': sum(r['support_count'] for r in rs) / len(rs),
                       'min_support_count': min(r['support_count'] for r in rs), 'max_support_count': max(r['support_count'] for r in rs),
                       'support_kinds': sorted({r['support_kind'] for r in rs})} for cell, rs in sorted(cells.items())}
    return {'schema': 'gpt56-all-levels32-discovery-support-v1', 'status': 'complete',
            'sources': {'new_rows': binding(rows_path), 'all_rows': binding(all_rows_path),
                        'native_source_rows': binding(source_rows_path), 'native_source_manifest': binding(source_manifest_path),
                        'legacy_support': binding(legacy_path), 'prior_rows': binding(old_rows_binding['path']),
                        'prior_analysis_report': binding(prior_report_path), 'frozen_manifest': binding(code_manifest_path),
                        'frozen_grader': binding(grader.__file__), 'frozen_hosted_contract': binding(contract.__file__),
                        'frozen_pantry_projection': binding(pantry.__file__), 'frozen_surface_helper': binding(SURFACE_HELPER),
                        'graph_countdown_helper': binding(enumerator_path), 'support_helper': binding(__file__)},
            'outcomes_read': False, 'selection_from_outcomes': False, 'selected_rows': 480, 'selection_seed': SEED,
            'new_rows': 240, 'new_ranks_per_cell': [17, 32], 'legacy_references_preserved_unchanged': 240,
            'references': refs, 'cells': summaries, 'support_kind_definitions': legacy['support_kind_definitions'],
            'extension_analysis_contract': legacy['extension_analysis_contract']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rows', 'all-rows', 'source-rows', 'legacy-support', 'code-root', 'enumerator-helper', 'prior-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite existing support certificate')
    result = build_reference(args.rows, args.all_rows, args.source_rows, args.legacy_support, args.code_root, args.enumerator_helper, args.prior_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    copies = args.output.parent / 'support_code'
    copies.mkdir(exist_ok=False)
    for path in (Path(__file__), args.enumerator_helper, SURFACE_HELPER, ROOT / 'tests/test_gpt56_all_levels32_discovery_support.py'):
        shutil.copy2(path, copies / path.name)
    new_ids = {pair_id(r) for r in read_rows(args.rows)}
    manifest = {'schema': 'gpt56-all-levels32-support-certificate-manifest-v1', 'status': 'complete',
                'certificate': binding(args.output), 'sources': result['sources'],
                'code_copies': [binding(p) for p in sorted(copies.iterdir())],
                'verified_new_witness_count': sum(r['support_count'] for key, r in result['references'].items() if key in new_ids),
                'countdown_extra_unary_witnesses': 48, 'legacy_references_preserved_unchanged': 240, 'outcomes_read': False}
    (args.output.parent / 'support_certificate_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(args.output), 'cells': result['cells']}, indent=2))


if __name__ == '__main__':
    main()
