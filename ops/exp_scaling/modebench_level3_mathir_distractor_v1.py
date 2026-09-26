"""Isolated fixed MathIR distractor laws; no experiment is registered here.

All presets retain the original one-sided family, first five commands, six
A--F IDs, and four-step verifier. Only the last command changes. Uniform
sampling without replacement uses an exact finite binding inventory and
independent per-row binding/menu streams. Model outcomes are never inputs.

Support argument: normalization, repeated-state checks, isolation, and keys
operate on symbolic expressions without substituting bindings. The only
numeric transition guards in these menus are a != 0 and, for div(b), b != 0.
Both hold throughout the inventories. Consequently each menu has the same
symbolic transition graph for every allowed binding. Exhaustive representative
certificates compare its complete canonical key set to the original menu's
five keys; five original-verifier witnesses are also checked for every row.
Coincident numeric values or c == 0 do not collapse symbolic states.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'ops', ROOT / 'ops/exp_scaling', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_mathir_action_menu_data import FAMILIES, Family, _prompt
from modebench_level3_constraints import _mathir_spec
from oat_drgrpo.mathir import (
    MATHIR_MENU_VERIFIER, enumerate_mathir_action_menu_validations,
    validate_mathir_action_menu,
)

PROFILE = 'mathir_fixed_distractor_laws_v1'
REPLACEMENTS = {0: 'sub(a)', 1: 'div(b)', 2: 'sub(a)', 3: 'div(b)'}
CONSTANT_BOUNDS = {0: None, 1: None, 2: 12, 3: 12}
DIFFICULTY_DESCRIPTIONS = {
    0: 'Original nonzero a, solution and b law; replace sub(c) with sub(a)',
    1: 'Original nonzero a, solution and b law; replace sub(c) with div(b)',
    2: 'Nonzero original law conditioned on |c| <= 12; replace sub(c) with sub(a)',
    3: 'Nonzero original law conditioned on |c| <= 12; replace sub(c) with div(b)',
}


def family_for_difficulty(difficulty: int) -> Family:
    if (not isinstance(difficulty, int) or isinstance(difficulty, bool)
            or difficulty not in REPLACEMENTS):
        raise ValueError('difficulty must be an integer in 0..3')
    original = FAMILIES[0]
    return replace(original, commands=original.commands[:5] + (REPLACEMENTS[difficulty],))


def semantic_identity(family: Family, bindings: dict[str, int]) -> tuple:
    # A changed distractor/profile/menu never makes an old equation fresh.
    return ('mathir', family.name, tuple(sorted(bindings.items())))


def law_holds(difficulty: int, bindings: dict[str, int]) -> bool:
    family_for_difficulty(difficulty)
    if (set(bindings) != set('abc')
            or any(not isinstance(v, int) or isinstance(v, bool) for v in bindings.values())):
        return False
    a, b, c = (bindings[name] for name in 'abc')
    if not (1 <= abs(a) <= 9 and 1 <= abs(b) <= 12):
        return False
    solution = Fraction(c - b, a)
    bound = CONSTANT_BOUNDS[difficulty]
    return (solution.denominator == 1 and 1 <= abs(solution) <= 9
            and (bound is None or abs(c) <= bound))


@lru_cache(maxsize=4)
def finite_inventory(difficulty: int) -> frozenset[tuple]:
    """Exact distinct semantic identities, not a quota-sized proposal sample."""
    family = family_for_difficulty(difficulty)
    bound = CONSTANT_BOUNDS[difficulty]
    return frozenset(
        semantic_identity(family, {'a': a, 'b': b, 'c': a * solution + b})
        for a in range(-9, 10) if a
        for solution in range(-9, 10) if solution
        for b in range(-12, 13) if b
        if bound is None or abs(a * solution + b) <= bound
    )


@lru_cache(maxsize=4)
def _ordered_inventory(difficulty: int) -> tuple[tuple, ...]:
    return tuple(sorted(finite_inventory(difficulty)))


def available_capacity(difficulty: int, excluded: set) -> int:
    return len(finite_inventory(difficulty) - excluded)


def sample_bindings(difficulty: int, rng: random.Random) -> dict[str, int]:
    """One exactly uniform draw from the full fixed law, before exclusions."""
    return dict(rng.choice(_ordered_inventory(difficulty))[2])


def _row_rng(seed: int, difficulty: int, index: int, stream: str) -> random.Random:
    # Neither the quota nor the number of excluded bindings enters a stream.
    payload = json.dumps([PROFILE, int(seed), difficulty, 5, index, stream],
                         separators=(',', ':')).encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), 'big'))


def _spec(family, bindings, actions, seed, tag, index):
    spec = _mathir_spec(family, bindings, actions, tag, seed, index)
    spec['source'] = 'synthetic_mathir_level3_fixed_distractor_candidates_v1'
    return spec


def _enumerated_certificate(family: Family):
    # This fixture satisfies every preset and all possible menu guards.
    bindings = {'a': 2, 'b': -3, 'c': 11}
    actions = dict(zip('ABCDEF', family.commands))
    validations = enumerate_mathir_action_menu_validations(
        _spec(family, bindings, actions, 0, 'support_certificate', 0))
    keys = tuple(sorted(v.canonical_key for v in validations))
    paths = tuple(tuple(actions[action] for action in v.action_ids) for v in validations)
    if len(keys) != 5 or len(set(keys)) != 5:
        raise RuntimeError(f'{family.name} changed semantic support: {len(keys)}')
    return keys, paths


@lru_cache(maxsize=1)
def _original_certificate() -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    return _enumerated_certificate(FAMILIES[0])


@lru_cache(maxsize=2)
def _template_certificate(replacement: str) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    if replacement not in {'sub(a)', 'div(b)'}:
        raise ValueError('unsupported fixed distractor')
    original = FAMILIES[0]
    family = replace(original, commands=original.commands[:5] + (replacement,))
    expected, paths = _original_certificate()
    actual, _ = _enumerated_certificate(family)
    if actual != expected:
        raise RuntimeError('replacement changed the original five canonical keys')
    if any(command not in original.commands[:5] for path in paths for command in path):
        raise RuntimeError('an original certified path uses a replaced distractor')
    return expected, paths


def template_support(difficulty: int) -> tuple[int, str]:
    family_for_difficulty(difficulty)
    keys, _ = _template_certificate(REPLACEMENTS[difficulty])
    return len(keys), hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def verify_witnesses(spec: dict, difficulty: int) -> None:
    """Check the fixed law, menu contract and all five original canonical paths."""
    family = family_for_difficulty(difficulty)
    if (not law_holds(difficulty, spec['bindings'])
            or spec.get('family') != family.name
            or spec.get('initial_lhs') != family.initial_lhs
            or spec.get('initial_rhs') != family.initial_rhs
            or spec.get('max_steps') != 4
            or list(spec['actions']) != list('ABCDEF')
            or sorted(spec['actions'].values()) != sorted(family.commands)):
        raise RuntimeError('row violates the fixed binding or six-action contract')
    expected, paths = _template_certificate(REPLACEMENTS[difficulty])
    command_ids = {command: action for action, command in spec['actions'].items()}
    actual = set()
    for path in paths:
        validation = validate_mathir_action_menu(';'.join(command_ids[c] for c in path), spec)
        if validation is None:
            raise RuntimeError('original verifier rejected a certified semantic path')
        actual.add(validation.canonical_key)
    if actual != set(expected):
        raise RuntimeError('original verifier canonical support changed')


def build_pool(
    domain: str, target: Counter, excluded: set, seed: int, tag: str,
    difficulty: int, multiplier: int = 4, *, joint_target: Counter | None = None,
) -> list[dict[str, Any]]:
    if domain != 'mathir':
        raise ValueError('this optional generator supports MathIR only')
    if not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier < 1:
        raise ValueError('multiplier must be a positive integer')
    family = family_for_difficulty(difficulty)
    if joint_target is not None:
        raise ValueError('MathIR does not use a separate joint support target')
    if any(isinstance(key, bool) or not isinstance(key, int)
           or isinstance(count, bool) or not isinstance(count, int) or count < 0
           for key, count in target.items()):
        raise ValueError('support keys and row counts must be integers; counts nonnegative')
    target = Counter({key: count for key, count in target.items() if count})
    if set(target) != {5} or target[5] < 1:
        raise ValueError('MathIR requires positive counts in the five-mode support cell')
    desired = target[5] * multiplier
    remaining = sorted(finite_inventory(difficulty) - set(excluded))
    if desired > len(remaining):
        raise RuntimeError(f'preset {difficulty} has {len(remaining)} unused identities; '
                           f'{desired} requested; the fixed binding law is not widened')
    mode_count, digest = template_support(difficulty)
    rows = []
    for index in range(desired):
        # Every remaining binding is equally likely, including near depletion.
        # Sorting is deterministic; it is never based on model outcomes.
        rng = _row_rng(seed, difficulty, index, 'bindings')
        identity = remaining.pop(rng.randrange(len(remaining)))
        bindings = dict(identity[2])
        commands = list(family.commands)
        _row_rng(seed, difficulty, index, 'menu').shuffle(commands)
        actions = dict(zip('ABCDEF', commands))
        spec = _spec(family, bindings, actions, seed, tag, index)
        verify_witnesses(spec, difficulty)
        spec.update(num_completions=mode_count, valid_mode_count=mode_count,
                    valid_mode_key_sha256=digest)
        rows.append({
            'problem': _prompt(family, bindings, actions),
            'answer': json.dumps(spec, sort_keys=True, separators=(',', ':')),
            'modebench_task': MATHIR_MENU_VERIFIER,
            'answer_mode_count': mode_count, 'answer_mode_split': tag,
            'mathir_family': family.name, 'level3_difficulty': difficulty,
            'level3_generation_profile': PROFILE, 'level3_cell_index': index,
        })
    return rows

