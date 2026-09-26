"""Prospective MathIR binding laws after the failed original Level-5 fit.

All four profiles preserve the original tier-1 equation, six commands, prompt,
five canonical modes and verifier. Profile zero is the original binding law;
three wider coprime laws are hypotheses, not measured or monotonic difficulty.
Bindings affect numeric guards but not symbolic transitions/canonical states.
Nonzero a/e/f and a != d*e preserve the original five-mode template. Each row
has independent binding/menu streams; quotas only stop the support stream.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
from math import gcd
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

import modebench_scale_candidates as original
import modebench_level3_constraints as constraints
from oat_drgrpo.mathir import validate_mathir_action_menu

SCHEMA = 'modebench_scale_l5_mathir_numeric_candidate_laws_v1'
DOMAINS = ('mathir',)
ORIGIN_TIER = 1
BANDS = ((1, 13), (14, 29), (30, 59), (100, 199))
MAX_PROPOSALS_PER_ROW = 100_000
FAMILY = constraints._mathir_family(ORIGIN_TIER)
PROFILES = {'mathir': [
    {'origin_module': constraints.__name__, 'origin_tier': ORIGIN_TIER,
     'family': FAMILY.name, 'actions': 6, 'max_steps': 4, 'canonical_modes': 5,
     'binding_symbols': list('abcdef'), 'absolute_binding_band': [lower, upper],
     'independent_uniform_signs': True, 'nonzero_effective_coefficient': 'a != d*e',
     'coprime_pairs': [] if tier == 0 else [['a', 'e'], ['b', 'f']],
     'sampling': ('independent_uniform_signed_integers_conditioned_on_a_ne_d_times_e'
                  if tier == 0 else
                  'uniform_ordered_coprime_magnitude_pairs_independent_uniform_c_d_and_signs'),
     'difficulty_ordering': 'prospective_hypotheses_no_monotonicity_claim'}
    for tier, (lower, upper) in enumerate(BANDS)
]}
identity = original.identity


def source_paths():
    return sorted({Path(__file__).resolve(), *original.source_paths(),
                   Path(sys.modules['oat_drgrpo.mathir'].__file__).resolve()})


def _seed(*parts):
    payload = json.dumps([SCHEMA, *parts], sort_keys=True, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest(), 'big')


def _row_rng(seed, tier, index, stream):
    return random.Random(_seed(seed, tier, 5, index, stream))


def _valid_tier(tier):
    return type(tier) is int and tier in range(4)


def law_holds(tier, bindings):
    if (not _valid_tier(tier) or not isinstance(bindings, Mapping)
            or set(bindings) != set('abcdef')
            or any(type(value) is not int for value in bindings.values())):
        return False
    lower, upper = BANDS[tier]
    if not all(lower <= abs(value) <= upper for value in bindings.values()):
        return False
    if bindings['a'] == bindings['d'] * bindings['e']:
        return False
    return tier == 0 or (gcd(abs(bindings['a']), abs(bindings['e'])) == 1
                         and gcd(abs(bindings['b']), abs(bindings['f'])) == 1)


@lru_cache(maxsize=3)
def coprime_pairs(tier):
    if not _valid_tier(tier) or tier == 0:
        raise ValueError('coprime catalogs require integer tier1..3')
    lower, upper = BANDS[tier]
    return tuple((a, e) for a in range(lower, upper + 1)
                 for e in range(lower, upper + 1) if gcd(a, e) == 1)


def _proposal(tier, rng):
    """One draw from the fixed proposal; anchor rejection happens outside."""
    lower, upper = BANDS[tier]
    if tier == 0:
        values = tuple(value for value in range(-upper, upper + 1) if value)
        return {key: rng.choice(values) for key in 'abcdef'}
    a, e = rng.choice(coprime_pairs(tier))
    b, f = rng.choice(coprime_pairs(tier))
    magnitudes = dict(a=a, b=b, c=rng.randint(lower, upper),
                      d=rng.randint(lower, upper), e=e, f=f)
    return {key: magnitudes[key] * rng.choice((-1, 1)) for key in 'abcdef'}


@lru_cache(maxsize=1)
def _certificate():
    keys, paths = original._mathir_certificate(ORIGIN_TIER)
    if len(keys) != 5 or len(paths) != 5:
        raise RuntimeError('original MathIR tier1 five-mode certificate changed')
    digest = hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()
    return keys, paths, digest


def _spec(bindings, actions, tag, seed, tier, index):
    _, _, digest = _certificate()
    spec = constraints._mathir_spec(FAMILY, bindings, actions, f'{tag}-t{tier}-m5', seed, index)
    spec.update(source=SCHEMA, num_completions=5, valid_mode_count=5,
                valid_mode_key_sha256=digest)
    return spec


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    if domain != 'mathir' or not _valid_tier(tier):
        raise ValueError('MathIR domain and integer tier0..3 required')
    if type(seed) is not int or seed < 0 or type(multiplier) is not int or multiplier < 1:
        raise ValueError('nonnegative integer seed and positive integer multiplier required')
    if not isinstance(tag, str) or not tag:
        raise ValueError('nonempty split tag required')
    if joint_target is not None:
        raise ValueError('MathIR does not accept joint histograms')
    if not isinstance(target, Mapping) or any(type(support) is not int or support != 5
            or type(count) is not int or count < 0 for support, count in target.items()):
        raise ValueError('MathIR requires exact support5 and nonnegative integer quotas')
    if not isinstance(excluded, (set, frozenset)):
        raise ValueError('semantic exclusions must be a set or frozenset')
    required = target.get(5, 0) * multiplier
    blocked, rows = set(excluded), []
    for index in range(required):
        rng = _row_rng(seed, tier, index, 'bindings')
        for _ in range(MAX_PROPOSALS_PER_ROW):
            bindings = _proposal(tier, rng)
            key = ('mathir', FAMILY.name, tuple(sorted(bindings.items())))
            if law_holds(tier, bindings) and key not in blocked:
                break
        else:
            raise RuntimeError(f'MathIR tier{tier}/support5 exhausted fixed proposal budget')
        blocked.add(key)
        commands = list(FAMILY.commands)
        _row_rng(seed, tier, index, 'menu').shuffle(commands)
        actions = dict(zip('ABCDEF', commands))
        spec = _spec(bindings, actions, tag, seed, tier, index)
        rows.append({
            'problem': constraints.mathir_prompt(FAMILY, bindings, actions),
            'answer': json.dumps(spec, sort_keys=True, separators=(',', ':')),
            'modebench_task': constraints.MATHIR_MENU_VERIFIER,
            'answer_mode_count': 5, 'answer_mode_split': tag, 'mathir_family': FAMILY.name,
            'scale_candidate_generator': SCHEMA, 'scale_candidate_tier': tier,
            'scale_candidate_profile': json.dumps(PROFILES['mathir'][tier], sort_keys=True, separators=(',', ':')),
            'scale_origin_metadata': json.dumps({'level3_difficulty': ORIGIN_TIER}, sort_keys=True, separators=(',', ':')),
            'scale_cell_index': index, 'scale_generation_seed': seed,
            'scale_numeric_law_version': SCHEMA,
        })
    # Display order is quota independent; generation prefixes use cell_index.
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    ids = {identity(domain, row) for row in rows}
    if len(ids) != len(rows) or ids & excluded or Counter(row['answer_mode_count'] for row in rows) != Counter({5: required}):
        raise RuntimeError('MathIR candidate semantic identity or exact quota drift')
    return rows


def verify_rows(domain, rows):
    """Recheck every row against origin tier1 and its five original witnesses."""
    if domain != 'mathir':
        raise ValueError('MathIR domain required')
    expected_keys, paths, _ = _certificate()
    identities, prompts = set(), set()
    witnesses = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError('MathIR row metadata audit failed')
        tier, seed, index = (row.get(key) for key in
                             ('scale_candidate_tier', 'scale_generation_seed', 'scale_cell_index'))
        tag = row.get('answer_mode_split')
        if (not _valid_tier(tier) or type(seed) is not int or seed < 0
                or type(index) is not int or index < 0 or not isinstance(tag, str) or not tag):
            raise RuntimeError('MathIR row metadata audit failed')
        expected_metadata = {
            'modebench_task': constraints.MATHIR_MENU_VERIFIER, 'mathir_family': FAMILY.name,
            'scale_candidate_generator': SCHEMA, 'scale_numeric_law_version': SCHEMA,
            'scale_candidate_profile': json.dumps(PROFILES['mathir'][tier], sort_keys=True, separators=(',', ':')),
            'scale_origin_metadata': json.dumps({'level3_difficulty': ORIGIN_TIER}, sort_keys=True, separators=(',', ':')),
        }
        if (type(row.get('answer_mode_count')) is not int or row['answer_mode_count'] != 5
                or any(row.get(key) != value for key, value in expected_metadata.items())):
            raise RuntimeError('MathIR row metadata audit failed')
        try:
            spec = json.loads(row['answer'])
            bindings, actions = spec['bindings'], spec['actions']
            valid = (law_holds(tier, bindings) and isinstance(actions, dict)
                     and tuple(actions) == tuple('ABCDEF')
                     and Counter(actions.values()) == Counter(FAMILY.commands)
                     and type(spec.get('num_completions')) is int
                     and type(spec.get('valid_mode_count')) is int
                     and type(spec.get('max_steps')) is int
                     and type(spec.get('support_is_open')) is bool
                     and spec == _spec(bindings, actions, tag, seed, tier, index)
                     and row.get('problem') == constraints.mathir_prompt(FAMILY, bindings, actions))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise RuntimeError('MathIR structural law, original prompt or exact-support audit failed')
        key = identity(domain, row)
        if key in identities or row['problem'] in prompts:
            raise RuntimeError('MathIR duplicate semantic identity or prompt audit failed')
        identities.add(key)
        prompts.add(row['problem'])
        command_ids = {command: action for action, command in actions.items()}
        validated = [validate_mathir_action_menu(';'.join(command_ids[c] for c in path), spec)
                     for path in paths]
        if (not all(validated)
                or {result.canonical_key for result in validated if result is not None} != expected_keys):
            raise RuntimeError('MathIR original-verifier witness audit failed')
        witnesses += len(validated)
    return {'exact_canonical_support': True, 'unchanged_original_prompt': True,
            'original_verifier_witnesses': True, 'mathir_numeric_profile': True,
            'origin_tier': ORIGIN_TIER, 'rows_verified': len(rows),
            'witnesses_verified': witnesses}
