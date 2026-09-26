"""Prospective sign-conditioned MathIR laws, preserving the six-action contract.

These broad fixed laws are exploratory, not claims of matched difficulty. No
model outcomes are inputs. Original family names preserve semantic exclusions.
Each row has independent binding and menu streams; quota only stops the stream.
"""
from __future__ import annotations

from collections import Counter
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
from modebench_level3_constraints import _mathir_family, _mathir_spec
from oat_drgrpo.mathir import (
    MATHIR_MENU_VERIFIER, enumerate_mathir_action_menu_validations,
    validate_mathir_action_menu,
)

PROFILE = 'mathir_fixed_sign_laws_v1'
PILOT_SEEDS = (7157100, 7158100, 7159100, 7160100)
DIFFICULTY_DESCRIPTIONS = {
    0: 'Original a*x + b = c conditioned on a > 0 and b < 0',
    1: 'Original a*x + b = c conditioned on a < 0 and b > 0',
    2: 'Original a*x + b = c conditioned on a*b < 0',
    3: 'Original v2 a*(x + b)/e = (c - d*x)/f conditioned on e > 0 and f > 0',
}


def family_for_difficulty(difficulty: int) -> Family:
    if isinstance(difficulty, bool) or difficulty not in DIFFICULTY_DESCRIPTIONS:
        raise ValueError('difficulty must be 0..3')
    return _mathir_family(3) if difficulty == 3 else FAMILIES[0]


def semantic_identity(family: Family, bindings: dict[str, int]) -> tuple:
    return ('mathir', family.name, tuple(sorted(bindings.items())))


def _row_rng(seed: int, difficulty: int, index: int, stream: str) -> random.Random:
    payload = json.dumps([PROFILE, int(seed), difficulty, 5, index, stream],
                         separators=(',', ':')).encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), 'big'))


def law_holds(difficulty: int, bindings: dict[str, int]) -> bool:
    family_for_difficulty(difficulty)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in bindings.values()):
        return False
    if difficulty == 3:
        return (set(bindings) == set('abcdef')
                and all(1 <= abs(bindings[k]) <= 29 for k in 'abcd')
                and all(1 <= bindings[k] <= 29 for k in 'ef')
                and bindings['a'] * bindings['f'] + bindings['d'] * bindings['e'] != 0)
    if set(bindings) != set('abc'):
        return False
    a, b, c = (bindings[k] for k in 'abc')
    if not (1 <= abs(a) <= 9 and 1 <= abs(b) <= 12):
        return False
    solution = Fraction(c - b, a)
    sign = (a > 0 and b < 0) if difficulty == 0 else (
        (a < 0 and b > 0) if difficulty == 1 else a * b < 0)
    return sign and solution.denominator == 1 and 1 <= abs(solution) <= 9


@lru_cache(maxsize=3)
def finite_inventory(difficulty: int) -> frozenset[tuple]:
    family = family_for_difficulty(difficulty)
    if difficulty == 3:
        raise ValueError('rational inventory is counted exactly without enumeration')
    return frozenset(
        semantic_identity(family, {'a': a, 'b': b, 'c': a * solution + b})
        for a in range(-9, 10) if a
        for b in range(-12, 13) if b and a * b < 0
        if difficulty == 2 or (a > 0 if difficulty == 0 else a < 0)
        for solution in range(-9, 10) if solution
    )


def rational_capacity(bound: int = 29) -> int:
    """Count a,b,c,d in nonzero +/-[1,B], e,f in [1,B], af+de != 0."""
    if bound < 1:
        raise ValueError('bound must be positive')
    ratios = Counter(Fraction(a, e) for a in range(1, bound + 1)
                     for e in range(1, bound + 1))
    coefficient_choices = (2 * bound) ** 2 * bound ** 2
    cancelling_choices = 2 * sum(count ** 2 for count in ratios.values())
    return (coefficient_choices - cancelling_choices) * (2 * bound) ** 2


def available_capacity(difficulty: int, excluded: set) -> int:
    family = family_for_difficulty(difficulty)
    if difficulty < 3:
        return len(finite_inventory(difficulty) - excluded)
    blocked = 0
    for identity in excluded:
        if (isinstance(identity, tuple) and len(identity) == 3
                and identity[:2] == ('mathir', family.name)):
            try:
                blocked += bool(law_holds(3, dict(identity[2])))
            except (TypeError, ValueError):
                continue
    return rational_capacity() - blocked


def sample_bindings(difficulty: int, rng: random.Random) -> dict[str, int]:
    family_for_difficulty(difficulty)
    if difficulty < 3:
        return dict(rng.choice(sorted(finite_inventory(difficulty)))[2])
    values = tuple(v for v in range(-29, 30) if v)
    for _ in range(100_000):
        bindings = {name: rng.choice(values) for name in 'abcd'}
        bindings.update(e=rng.randint(1, 29), f=rng.randint(1, 29))
        if law_holds(3, bindings):
            return bindings
    raise RuntimeError('fixed rational law exhausted its proposal budget')


def _spec(family, bindings, actions, seed, tag, index):
    spec = _mathir_spec(family, bindings, actions, tag, seed, index)
    spec['source'] = 'synthetic_mathir_level3_fixed_sign_candidates_v1'
    return spec


@lru_cache(maxsize=2)
def _template_certificate(rational: bool) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    difficulty = 3 if rational else 0
    family = family_for_difficulty(difficulty)
    bindings = sample_bindings(difficulty, _row_rng(0, difficulty, 0, 'certificate'))
    actions = dict(zip('ABCDEF', family.commands))
    validations = enumerate_mathir_action_menu_validations(
        _spec(family, bindings, actions, 0, 'support_certificate', 0))
    keys = tuple(sorted(v.canonical_key for v in validations))
    paths = tuple(tuple(actions[action] for action in v.action_ids) for v in validations)
    if len(keys) != 5 or len(set(keys)) != 5:
        raise RuntimeError(f'{family.name} changed semantic support: {len(keys)}')
    return keys, paths


def template_support(difficulty: int) -> tuple[int, str]:
    family_for_difficulty(difficulty)
    keys, _ = _template_certificate(difficulty == 3)
    return len(keys), hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def verify_witnesses(spec: dict, difficulty: int) -> None:
    """Check all five semantic paths with the original verifier for every row."""
    expected, paths = _template_certificate(difficulty == 3)
    command_ids = {command: action for action, command in spec['actions'].items()}
    keys = set()
    for path in paths:
        result = validate_mathir_action_menu(';'.join(command_ids[c] for c in path), spec)
        if result is None:
            raise RuntimeError('original verifier rejected a certified semantic path')
        keys.add(result.canonical_key)
    if keys != set(expected):
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
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0
           for count in target.values()):
        raise ValueError('row counts must be nonnegative integers')
    target = Counter({int(key): count for key, count in target.items() if count})
    if set(target) != {5} or target[5] < 1:
        raise ValueError('MathIR requires positive counts in the five-mode support cell')
    desired, blocked = target[5] * multiplier, set(excluded)
    available = available_capacity(difficulty, blocked)
    if desired > available:
        raise RuntimeError(f'preset {difficulty} has {available} unused identities; '
                           f'{desired} requested; the fixed binding law is not widened')
    remaining = sorted(finite_inventory(difficulty) - blocked) if difficulty < 3 else None
    mode_count, digest = template_support(difficulty)
    rows = []
    for index in range(desired):
        rng = _row_rng(seed, difficulty, index, 'bindings')
        if remaining is not None:
            # Uniform sampling without replacement from the fixed finite law.
            identity = remaining.pop(rng.randrange(len(remaining)))
            bindings = dict(identity[2])
        else:
            for _ in range(100_000):
                bindings = sample_bindings(difficulty, rng)
                identity = semantic_identity(family, bindings)
                if identity not in blocked:
                    break
            else:
                raise RuntimeError('fixed rational law exhausted its row proposal budget')
        blocked.add(identity)
        commands = list(family.commands)
        _row_rng(seed, difficulty, index, 'menu').shuffle(commands)
        actions = dict(zip('ABCDEF', commands))
        spec = _spec(family, bindings, actions, seed, tag, index)
        # Symbolic transitions never substitute bindings; the fixed laws exclude
        # every zero divisor. Thus the exhaustive template gives an upper bound
        # of five paths, and the five per-row witnesses attain that bound.
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
