#!/usr/bin/env python3
"""Publish/audit fixed MathIR v3 development pools without model evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'ops', ROOT / 'ops/exp_scaling', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import modebench_level3_mathir_v3 as v3
from materialize_modebench_harder_v2 import identity_set, modes, row_hash
from materialize_modebench_level3 import historical_ids, reference_rows, SEEDS, verify_rows
from evaluate_modebench_level3 import atomic_new, file_sha, frozen_interface, sha
from evaluate_modebench_level2_viability import prompt_messages
from oat_drgrpo.math_grader import validated_modebench_outcome_key
from oat_drgrpo.mathir import enumerate_mathir_action_menu_keys

POOL_ROOT = ROOT / 'var/data/modebench_level3_calibration_v7/pools/mathir'
AUDIT_PATH = ROOT / 'var/artifacts/modebench_level3_mathir_v3/structural_audit.json'
POOL_SEEDS = {d: SEEDS['mathir'] + 600_000 + 1000 * d for d in range(4)}
CAPACITY_SIZES = {'train': 384, 'dev': 128, 'eval': 128}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pool_paths():
    return sorted((ROOT / 'var/data').glob('modebench_level3_calibration*/pools/mathir/*.jsonl'))


def exclusions():
    blocked = historical_ids('mathir')
    for path in pool_paths():
        blocked |= identity_set('mathir', read_rows(path))
    return blocked


def inventory(blocked):
    return {str(d): {'universe': len(v3.finite_inventory(d)),
                     'blocked': len(v3.finite_inventory(d) & blocked),
                     'available': len(v3.finite_inventory(d) - blocked)}
            for d in range(3)}


def protected_hashes():
    paths = [ROOT / 'ops/exp_scaling/modebench_level3_mathir_v2.py',
             ROOT / 'ops/exp_scaling/modebench_level3_constraints.py',
             ROOT / 'ops/make_mathir_action_menu_data.py',
             ROOT / 'ops/evaluate_modebench_level3.py',
             ROOT / 'ops/evaluate_modebench_level2_viability.py']
    paths += sorted((ROOT / 'src/oat_drgrpo').glob('*.py'))
    paths += [p for p in pool_paths() if p.parent != POOL_ROOT]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def verify_published_pool(rows, reference, excluded):
    """Apply full structural checks on reload, including within-pool uniqueness."""
    return verify_rows('mathir', rows, reference, modes(reference), excluded)


def check_rows(rows, difficulty, tokenizer, *, all_routes):
    family = v3.family_for_difficulty(difficulty)
    witnesses, maximum_prompt_tokens = 0, 0
    interface = frozen_interface('mathir', 'level2_qwen_r5')
    for row in rows:
        spec = json.loads(row['answer'])
        assert spec['family'] == row['mathir_family'] == family.name
        assert spec['initial_lhs'] == family.initial_lhs and spec['initial_rhs'] == family.initial_rhs
        assert list(spec['actions']) == list('ABCDEF') and spec['max_steps'] == 4
        assert set(spec['actions'].values()) == set(family.commands)
        assert spec['verifier'] == 'mathir_action_menu' and spec['mathir_version'] == 'linear-menu-v1'
        assert not spec['support_is_open']
        assert row['level3_generation_profile'] == v3.PROFILE
        assert row['level3_difficulty'] == difficulty
        assert row['answer_mode_count'] == spec['num_completions'] == spec['valid_mode_count'] == 5
        assert spec['valid_mode_key_sha256'] == v3.template_support(difficulty)[1]
        assert row['problem'] == v3._prompt(family, spec['bindings'], spec['actions'])
        if difficulty < 3:
            assert v3.semantic_identity(family, spec['bindings']) in v3.finite_inventory(difficulty)
        else:
            b = spec['bindings']
            assert all(1 <= abs(value) <= 29 for value in b.values())
            assert b['a'] * b['f'] + b['d'] * b['e']
        by_command = {command: key for key, command in spec['actions'].items()}
        routes = family.certified_routes if all_routes else family.certified_routes[:1]
        for route in routes:
            program = ';'.join(by_command[command] for command in route)
            assert validated_modebench_outcome_key('\\boxed{' + program + '}', row['answer']) is not None
            witnesses += 1
        messages = prompt_messages('mathir', row['problem'], interface['prompt_profile'])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        maximum_prompt_tokens = max(maximum_prompt_tokens, tokens)
        assert tokens + interface['max_tokens'] <= interface['max_model_len']
    # Enumerate the original verifier independently for this concrete menu,
    # rather than trusting the declared support count alone.
    keys = enumerate_mathir_action_menu_keys(json.loads(rows[0]['answer']))
    assert len(keys) == 5
    assert hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest() == v3.template_support(difficulty)[1]
    return {'rows': len(rows), 'original_grader_witnesses': witnesses,
            'exact_symbolic_mode_count': len(keys), 'maximum_prompt_tokens': maximum_prompt_tokens,
            'output_budget': interface['max_tokens'], 'context_budget': interface['max_model_len']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialize-development', action='store_true')
    parser.add_argument('--output', type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.time()
    protected_before = protected_hashes()
    source_sha = file_sha(Path(v3.__file__))
    blocked_before = exclusions()
    pre_inventory = inventory(blocked_before)
    if pre_inventory['1']['available'] < 640:
        raise RuntimeError(f'|c| <= 6 lacks 640 unused identities: {pre_inventory}; no bounds changed')
    reference = reference_rows('mathir', 'dev')
    assert modes(reference) == Counter({5: 128})
    new_rows = {}
    proposed_blocked = set(blocked_before)
    if args.materialize_development:
        # Prepare every pool and check post-publication capacity before writing
        # any pool. A smaller request never switches to another proposal law.
        for d in range(4):
            path = POOL_ROOT / f'difficulty_{d}.jsonl'
            if path.exists() or path.with_suffix('.identity.json').exists():
                raise FileExistsError(path)
            rows = v3.build_pool('mathir', modes(reference), proposed_blocked,
                                 POOL_SEEDS[d], 'level3_development_pool', d, 1)
            verify_rows('mathir', rows, reference, modes(reference), proposed_blocked)
            proposed_blocked |= identity_set('mathir', rows)
            new_rows[d] = rows
        post_inventory = inventory(proposed_blocked)
        if post_inventory['1']['available'] < 640:
            raise RuntimeError(f'new pilot exclusions leave insufficient |c| <= 6 capacity: {post_inventory}')
    else:
        for d in range(4):
            new_rows[d] = read_rows(POOL_ROOT / f'difficulty_{d}.jsonl')
        post_inventory = inventory(blocked_before)
    print(json.dumps({'event': 'inventories_checked', 'before_new_pools': pre_inventory,
                      'after_new_pools': post_inventory}), flush=True)
    from transformers import AutoTokenizer
    prior = json.loads((ROOT / 'var/results/modebench_level3_v1/calibrated_v3_3b_mathir_d0.json').read_text())
    model_path = prior['identity']['model']['path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    pool_checks = {}
    for d, rows in new_rows.items():
        checks = check_rows(rows, d, tokenizer, all_routes=True)
        assert len(rows) == 128 and modes(rows) == modes(reference)
        pool_checks[str(d)] = checks
        print(json.dumps({'event': 'pool_verified', 'difficulty': d, **checks}), flush=True)
    if args.materialize_development:
        POOL_ROOT.mkdir(parents=True, exist_ok=True)
        blocked = set(blocked_before)
        for d, rows in new_rows.items():
            path = POOL_ROOT / f'difficulty_{d}.jsonl'
            checks = verify_rows('mathir', rows, reference, modes(reference), blocked)
            with path.open('x') as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + '\n')
            atomic_new(path.with_suffix('.identity.json'), {
                'schema': 'modebench_level3_development_pool_v1', 'domain': 'mathir',
                'difficulty': d, 'seed': POOL_SEEDS[d], 'rows': len(rows),
                'rows_sha256': row_hash(rows), 'checks': checks,
                'support_histogram': dict(modes(rows)), 'source_sha256': source_sha,
                'generation_profile': v3.PROFILE,
                'proposal': v3.DIFFICULTY_DESCRIPTIONS[d],
                'rng_policy': 'fixed_per_row_per_support_cell_separate_binding_and_menu_streams',
                'information_boundary': 'development candidates only; no confirmation model outcomes used',
            })
            blocked |= identity_set('mathir', rows)
    all_blocked = exclusions()
    final_inventory = inventory(all_blocked)
    assert final_inventory['1']['available'] >= 640
    for d, rows in new_rows.items():
        path = POOL_ROOT / f'difficulty_{d}.jsonl'
        manifest = json.loads(path.with_suffix('.identity.json').read_text())
        assert manifest['source_sha256'] == source_sha
        assert manifest['rows_sha256'] == row_hash(rows)
        other = historical_ids('mathir')
        for candidate in pool_paths():
            if candidate != path:
                other |= identity_set('mathir', read_rows(candidate))
        verify_published_pool(rows, reference, other)
    capacities, capacity_ids = {}, {}
    for d in range(4):
        within_preset = set(all_blocked)
        capacities[str(d)] = {}
        capacity_ids[d] = set()
        for i, (split, count) in enumerate(CAPACITY_SIZES.items()):
            reference_split = reference_rows('mathir', split)
            assert modes(reference_split) == Counter({5: count})
            rows = v3.build_pool('mathir', modes(reference_split), within_preset,
                                 POOL_SEEDS[d] + 100_000 + i * 10_000,
                                 f'capacity_{split}', d, 1)
            checks = verify_rows('mathir', rows, reference_split, modes(reference_split), within_preset)
            checked = check_rows(rows, d, tokenizer, all_routes=False)
            ids = identity_set('mathir', rows)
            within_preset |= ids
            capacity_ids[d] |= ids
            capacities[str(d)][split] = {'checks': checks, 'rows_sha256': row_hash(rows), **checked}
            print(json.dumps({'event': 'capacity_verified', 'difficulty': d, 'split': split, **checked}), flush=True)
        small = v3.build_pool('mathir', Counter({5: 7}), all_blocked, POOL_SEEDS[d], 'prefix', d, 1)
        large = v3.build_pool('mathir', Counter({5: 13}), all_blocked, POOL_SEEDS[d], 'prefix', d, 1)
        assert small == large[:7]
    assert protected_hashes() == protected_before
    assert file_sha(Path(v3.__file__)) == source_sha
    atomic_new(args.output, {
        'schema': 'modebench_level3_mathir_v3_structural_audit_v1',
        'generator_sha256': source_sha, 'audit_source_sha256': file_sha(Path(__file__)),
        'protected_source_and_old_pool_sha256': protected_before,
        'pool_seeds': POOL_SEEDS, 'before_new_pool_inventory': pre_inventory,
        'after_all_candidate_pool_inventory': final_inventory,
        'all_excluded_identities': len(all_blocked),
        'all_excluded_identities_sha256': sha(sorted(all_blocked)),
        'published_pool_checks': pool_checks, 'capacity_checks': capacities,
        'capacity_cross_preset_overlap': {f'{a},{b}': len(capacity_ids[a] & capacity_ids[b])
                                          for a in range(4) for b in range(a + 1, 4)},
        'capacity_policy': '640 fresh rows per preset, disjoint within preset and from every historical/published pool; cross-preset capacity rows may overlap',
        'per_support_quota_prefix_checks': True, 'original_sources_and_pools_preserved': True,
        'pool_files_sha256': {str((POOL_ROOT / f'difficulty_{d}.jsonl').relative_to(ROOT)):
                              file_sha(POOL_ROOT / f'difficulty_{d}.jsonl') for d in range(4)},
        'tokenizer_model_path': model_path,
        'information_boundary': 'structural CPU checks only; no model inference, fitting, confirmation scoring, or treatment training',
        'elapsed_seconds': time.time() - started,
    })
    print(json.dumps({'event': 'audit_complete', 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
