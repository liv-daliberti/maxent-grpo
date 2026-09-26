"""Prospective split-denominator MathIR laws after the original Level-5 fit.

Tier zero preserves the original split-linear equation. Three structural
hypotheses change only the equation denominators and the six derived actions,
retaining the original prompt renderer, verifier, six IDs and four-step limit.
All bindings have the original independent signed nonzero [-13,13] proposal,
conditioned on nonzero denominators and a nonzero effective x coefficient.
No measured or monotonic difficulty ordering is claimed.

For each AST, symbolic transitions, repeated-state rejection and canonical keys
are binding independent. The only numeric guards are nonzero denominators and
nonzero divisors A and A-d. The fixed binding predicates preserve these guards;
an exhaustive original-verifier template certificate and every row's five
original-verifier witnesses establish the exact five-mode support.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

import modebench_scale_candidates as original
import modebench_level3_constraints as constraints
from oat_drgrpo.mathir import (
    validate_mathir_action_menu, enumerate_mathir_action_menu_validations,
    parse_mathir_program, _expr_node_count,
)

SCHEMA = 'modebench_scale_l5_mathir_structural_candidate_laws_v1'
DOMAINS = ('mathir',)
ORIGIN_TIER = 1
MAGNITUDE = 13
MAX_PROPOSALS_PER_ROW = 100_000
DENOMINATORS = (('e', 'f'), ('add(a,e)', 'f'),
                ('e', 'sub(b,f)'), ('add(a,e)', 'sub(b,f)'))
FAMILY_NAMES = (
    constraints._mathir_family(ORIGIN_TIER).name,
    'scale_split_linear_sum_coefficient_denominator_v1',
    'scale_split_linear_difference_offset_denominator_v1',
    'scale_split_linear_sum_difference_denominators_v1',
)


def _valid_tier(tier):
    return type(tier) is int and tier in range(4)


def family_for_tier(tier):
    if not _valid_tier(tier):
        raise ValueError('integer tier0..3 required')
    if tier == 0:
        return constraints._mathir_family(ORIGIN_TIER)
    d1, d2 = DENOMINATORS[tier]
    coefficient, offset = f'div(a,{d1})', f'div(b,{d2})'
    subtract = f'sub({offset})'
    remove_variable = 'sub(mul(d,x))'
    divide = f'div(sub({coefficient},d))'
    combined = f'sub(add(mul(d,x),{offset}))'
    display_d1 = '(a+e)' if tier in (1, 3) else 'e'
    display_d2 = '(b-f)' if tier in (2, 3) else 'f'
    return constraints.Family(
        name=FAMILY_NAMES[tier],
        initial_lhs=f'add(div(mul(a,x),{d1}),{offset})',
        initial_rhs='add(mul(d,x),c)',
        display_equation=f'a*x/{display_d1} + b/{display_d2} = d*x + c',
        commands=(subtract, remove_variable, divide, combined,
                  f'add({offset})', f'div({coefficient})'),
        certified_routes=((subtract, remove_variable, divide),
                          (remove_variable, subtract, divide), (combined, divide)),
    )


PROFILES = {'mathir': [
    {'origin_module': constraints.__name__, 'origin_tier': ORIGIN_TIER,
     'family': FAMILY_NAMES[tier], 'actions': 6, 'max_steps': 4, 'canonical_modes': 5,
     'binding_symbols': list('abcdef'), 'absolute_binding_band': [1, MAGNITUDE],
     'coefficient_denominator_ast': d1, 'offset_denominator_ast': d2,
     'initial_lhs': family_for_tier(tier).initial_lhs,
     'initial_rhs': family_for_tier(tier).initial_rhs,
     'commands': list(family_for_tier(tier).commands),
     'independent_uniform_signs': True,
     'guards': ['D1 != 0', 'D2 != 0', 'a != d*D1'],
     'sampling': 'independent_uniform_signed_nonzero_integers_conditioned_on_denominator_and_coefficient_guards',
     'difficulty_ordering': 'prospective_hypotheses_no_monotonicity_claim'}
    for tier, (d1, d2) in enumerate(DENOMINATORS)
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


def denominator_values(tier, bindings):
    if not _valid_tier(tier):
        raise ValueError('integer tier0..3 required')
    a, b, e, f = (bindings[name] for name in 'abef')
    return (a + e if tier in (1, 3) else e,
            b - f if tier in (2, 3) else f)


def law_holds(tier, bindings):
    if (not _valid_tier(tier) or not isinstance(bindings, Mapping)
            or set(bindings) != set('abcdef')
            or any(type(value) is not int or not 1 <= abs(value) <= MAGNITUDE
                   for value in bindings.values())):
        return False
    d1, d2 = denominator_values(tier, bindings)
    return d1 != 0 and d2 != 0 and bindings['a'] != bindings['d'] * d1


def _proposal(tier, rng):
    """The original independent binding proposal; fixed guards reject below."""
    values = tuple(value for value in range(-MAGNITUDE, MAGNITUDE + 1) if value)
    return {key: rng.choice(values) for key in 'abcdef'}


def _base_spec(bindings, actions, tag, seed, tier, index):
    spec = constraints._mathir_spec(family_for_tier(tier), bindings, actions,
                                   f'{tag}-t{tier}-m5', seed, index)
    spec['source'] = SCHEMA
    return spec


@lru_cache(maxsize=4)
def _certificate(tier):
    family = family_for_tier(tier)
    bindings = dict(a=7, b=11, c=13, d=-3, e=5, f=2)
    if not law_holds(tier, bindings):
        raise RuntimeError('fixed MathIR structural certificate violates its law')
    actions = dict(zip('ABCDEF', family.commands))
    # The original parser enforces the eleven-node and six-depth bounds.
    counts = [
        _expr_node_count(parse_mathir_program(command, allowed_symbols='abcdefx', max_steps=1)[0].argument)
        for command in family.commands
    ]
    if max(counts) > 11:
        raise RuntimeError('MathIR structural action exceeds original node limit')
    spec = _base_spec(bindings, actions, 'TEMPLATE_CERTIFICATE_ONLY', 0, tier, 0)
    validations = enumerate_mathir_action_menu_validations(spec)
    keys = frozenset(result.canonical_key for result in validations)
    paths = tuple(tuple(actions[action] for action in result.action_ids) for result in validations)
    if len(keys) != 5 or len(paths) != 5:
        raise RuntimeError(f'MathIR structural tier{tier} does not have exactly five canonical modes')
    if tier == 0 and keys != original._mathir_certificate(ORIGIN_TIER)[0]:
        raise RuntimeError('MathIR structural anchor differs from original tier1 support')
    digest = hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()
    return keys, paths, digest


def _spec(bindings, actions, tag, seed, tier, index):
    _, _, digest = _certificate(tier)
    spec = _base_spec(bindings, actions, tag, seed, tier, index)
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
    family = family_for_tier(tier)
    required = target.get(5, 0) * multiplier
    blocked, rows = set(excluded), []
    for index in range(required):
        rng = _row_rng(seed, tier, index, 'bindings')
        for _ in range(MAX_PROPOSALS_PER_ROW):
            bindings = _proposal(tier, rng)
            key = ('mathir', family.name, tuple(sorted(bindings.items())))
            if law_holds(tier, bindings) and key not in blocked:
                break
        else:
            raise RuntimeError(f'MathIR tier{tier}/support5 exhausted fixed proposal budget')
        blocked.add(key)
        commands = list(family.commands)
        _row_rng(seed, tier, index, 'menu').shuffle(commands)
        actions = dict(zip('ABCDEF', commands))
        spec = _spec(bindings, actions, tag, seed, tier, index)
        rows.append({
            'problem': constraints.mathir_prompt(family, bindings, actions),
            'answer': json.dumps(spec, sort_keys=True, separators=(',', ':')),
            'modebench_task': constraints.MATHIR_MENU_VERIFIER,
            'answer_mode_count': 5, 'answer_mode_split': tag, 'mathir_family': family.name,
            'scale_candidate_generator': SCHEMA, 'scale_candidate_tier': tier,
            'scale_candidate_profile': json.dumps(PROFILES['mathir'][tier], sort_keys=True, separators=(',', ':')),
            'scale_origin_metadata': json.dumps({'level3_difficulty': ORIGIN_TIER}, sort_keys=True, separators=(',', ':')),
            'scale_cell_index': index, 'scale_generation_seed': seed,
            'scale_structural_law_version': SCHEMA,
        })
    # Display order is quota independent; generation prefixes use cell_index.
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    ids = {identity(domain, row) for row in rows}
    if len(ids) != len(rows) or ids & excluded or Counter(row['answer_mode_count'] for row in rows) != Counter({5: required}):
        raise RuntimeError('MathIR candidate semantic identity or exact quota drift')
    return rows


def verify_rows(domain, rows):
    """Recheck every AST and its five original-verifier canonical witnesses."""
    if domain != 'mathir':
        raise ValueError('MathIR domain required')
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
        family = family_for_tier(tier)
        expected_keys, paths, _ = _certificate(tier)
        expected_metadata = {
            'modebench_task': constraints.MATHIR_MENU_VERIFIER, 'mathir_family': family.name,
            'scale_candidate_generator': SCHEMA, 'scale_structural_law_version': SCHEMA,
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
                     and Counter(actions.values()) == Counter(family.commands)
                     and type(spec.get('num_completions')) is int
                     and type(spec.get('valid_mode_count')) is int
                     and type(spec.get('max_steps')) is int
                     and type(spec.get('support_is_open')) is bool
                     and spec == _spec(bindings, actions, tag, seed, tier, index)
                     and row.get('problem') == constraints.mathir_prompt(family, bindings, actions))
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
            'original_verifier_witnesses': True, 'mathir_structural_profile': True,
            'origin_tier': ORIGIN_TIER, 'rows_verified': len(rows),
            'witnesses_verified': witnesses}
