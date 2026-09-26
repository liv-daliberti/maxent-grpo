"""Optional broader MathIR candidate range; existing Level 3 pools are unchanged.

Every tier has six action IDs, at most four steps, and exactly five canonical
valid state paths under the original linear-menu-v1 verifier. Tier names describe
structural candidates; empirical model measurements must determine their order
and suitability for capability matching.
"""
from __future__ import annotations

from collections import Counter
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

from make_mathir_action_menu_data import FAMILIES, Family, _prompt, _sample_bindings
from modebench_level3_constraints import _mathir_family, _mathir_spec
from oat_drgrpo.mathir import MATHIR_MENU_VERIFIER, enumerate_mathir_action_menu_keys

PROFILE = 'mathir_broad_v2'
DIFFICULTY_DESCRIPTIONS = {
    0: 'Original one-sided a*x + b = c with Level 1 coefficient/solution bounds',
    1: 'One-sided factored a*(x + b) = c with the same coefficient/solution bounds',
    2: 'Original variable-on-both-sides a*x + b = d*x + c',
    3: 'Distributed numerator and rational expressions a*(x + b)/e = (c - d*x)/f',
}


def family_for_difficulty(difficulty: int) -> Family:
    if difficulty == 0:
        return FAMILIES[0]
    if difficulty == 1:
        return Family(
            name='a_times_x_plus_b_eq_c',
            initial_lhs='mul(a,add(x,b))',
            initial_rhs='c',
            display_equation='a*(x + b) = c',
            commands=('sub(mul(a,b))', 'div(a)', 'sub(b)',
                      'add(b)', 'mul(a)', 'sub(c)'),
            certified_routes=(('div(a)', 'sub(b)'),
                              ('sub(mul(a,b))', 'div(a)')),
        )
    if difficulty == 2:
        return FAMILIES[2]
    if difficulty == 3:
        return _mathir_family(3)
    raise ValueError('difficulty must be 0..3')


def sample_bindings(difficulty: int, rng: random.Random) -> dict[str, int]:
    family = family_for_difficulty(difficulty)
    if difficulty in (0, 2):
        return _sample_bindings(family, rng)
    if difficulty == 1:
        nonzero = [value for value in range(-9, 10) if value]
        solution = rng.choice(nonzero)
        a, b = rng.choice(nonzero), rng.randrange(-12, 13)
        return {'a': a, 'b': b, 'c': a * (solution + b)}
    values = [value for value in range(-29, 30) if value]
    while True:
        bindings = {name: rng.choice(values) for name in 'abcdef'}
        if bindings['a'] * bindings['f'] + bindings['d'] * bindings['e']:
            return bindings


def _spec(family, bindings, actions, seed, tag, index):
    spec = _mathir_spec(family, bindings, actions, tag, seed, index)
    spec['source'] = 'synthetic_mathir_level3_broad_candidates_v2'
    return spec


@lru_cache(maxsize=4)
def template_support(difficulty: int) -> tuple[int, str]:
    family = family_for_difficulty(difficulty)
    bindings = sample_bindings(difficulty, random.Random(6451900 + difficulty))
    actions = dict(zip('ABCDEF', family.commands))
    keys = enumerate_mathir_action_menu_keys(
        _spec(family, bindings, actions, 0, 'support_certificate', 0)
    )
    if len(keys) != 5:
        raise RuntimeError(f'{family.name} changed semantic support: {len(keys)}')
    return len(keys), hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()


def build_pool(
    domain: str,
    target: Counter,
    excluded: set,
    seed: int,
    tag: str,
    difficulty: int,
    multiplier: int = 4,
    *,
    joint_target: Counter | None = None,
) -> list[dict[str, Any]]:
    """Generate exact support-cell counts, with ordinary semantic exclusions.

    Original family names are retained for reused equation templates, including
    the previous rational tier, so changing profile metadata cannot bypass the
    existing family-and-bindings identity rule. No model outcome is an input.
    """
    if domain != 'mathir':
        raise ValueError('this optional generator supports MathIR only')
    if not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier < 1:
        raise ValueError('multiplier must be a positive integer')
    family = family_for_difficulty(difficulty)
    target = Counter({int(key): int(count) for key, count in target.items() if count})
    if set(target) != {5} or target[5] < 1:
        raise ValueError('MathIR requires positive row counts in the five-mode support cell')
    mode_count, digest = template_support(difficulty)
    rng, blocked = random.Random(seed), set(excluded)
    rows = []
    desired = target[5] * multiplier
    for attempt in range(max(20_000, desired * 300)):
        if len(rows) == desired:
            rng.shuffle(rows)
            return rows
        bindings = sample_bindings(difficulty, rng)
        identity = ('mathir', family.name, tuple(sorted(bindings.items())))
        if identity in blocked:
            continue
        blocked.add(identity)
        commands = list(family.commands)
        rng.shuffle(commands)
        actions = dict(zip('ABCDEF', commands))
        spec = _spec(family, bindings, actions, seed, tag, len(rows))
        # All actions have the template's symbolic state transitions. The
        # sampler excludes every possible zero multiplicative denominator, so
        # bindings cannot remove paths from the exhaustively certified support.
        spec.update(num_completions=mode_count, valid_mode_count=mode_count,
                    valid_mode_key_sha256=digest)
        rows.append({
            'problem': _prompt(family, bindings, actions),
            'answer': json.dumps(spec, sort_keys=True, separators=(',', ':')),
            'modebench_task': MATHIR_MENU_VERIFIER,
            'answer_mode_count': mode_count,
            'answer_mode_split': tag,
            'mathir_family': family.name,
            'level3_difficulty': difficulty,
            'level3_generation_profile': PROFILE,
        })
    raise RuntimeError(f'could not build {desired} unique {family.name} rows')
