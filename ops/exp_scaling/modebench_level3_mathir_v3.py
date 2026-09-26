"""Fixed numeric-law MathIR proposals under the unchanged six-action contract.

The first three presets share the original one-sided equation and its exact
five symbolic state paths. Their binding laws condition the original sampler
on a bounded right-hand constant, or retain it unchanged. The last preset
retains the v2 rational equation. No model outcomes are generator inputs.
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

from make_mathir_action_menu_data import Family, _prompt
from modebench_level3_constraints import _mathir_spec
from modebench_level3_mathir_v2 import (
    family_for_difficulty as _v2_family,
    sample_bindings as _v2_bindings,
)
from oat_drgrpo.mathir import MATHIR_MENU_VERIFIER, enumerate_mathir_action_menu_keys

PROFILE = 'mathir_numeric_laws_v3'
CONSTANT_BOUNDS = {0: 12, 1: 6, 2: None, 3: None}
DIFFICULTY_DESCRIPTIONS = {
    0: 'Original a*x + b = c binding law conditioned on |c| <= 12',
    1: 'Original a*x + b = c binding law conditioned on |c| <= 6',
    2: 'Original unconditioned a*x + b = c binding law',
    3: 'Original v2 a*(x + b)/e = (c - d*x)/f binding law',
}


def family_for_difficulty(difficulty: int) -> Family:
    if isinstance(difficulty, bool) or difficulty not in CONSTANT_BOUNDS:
        raise ValueError('difficulty must be 0..3')
    return _v2_family(3 if difficulty == 3 else 0)


def _row_rng(seed: int, difficulty: int, row: int, stream: str) -> random.Random:
    # The requested quota and binding-rejection count are deliberately absent.
    # Binding and menu streams are separate even when conditioning is costly.
    payload = json.dumps([PROFILE, int(seed), difficulty, 5, row, stream],
                         separators=(',', ':')).encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), 'big'))


def sample_bindings(difficulty: int, rng: random.Random) -> dict[str, int]:
    family_for_difficulty(difficulty)
    bound = CONSTANT_BOUNDS[difficulty]
    while True:
        bindings = _v2_bindings(3 if difficulty == 3 else 0, rng)
        if bound is None or abs(bindings['c']) <= bound:
            return bindings


def semantic_identity(family: Family, bindings: dict[str, int]) -> tuple:
    # Reused family names prevent a new profile from bypassing old exclusions.
    return ('mathir', family.name, tuple(sorted(bindings.items())))


@lru_cache(maxsize=3)
def finite_inventory(difficulty: int) -> frozenset[tuple]:
    """Exact identities for the finite one-sided law, before exclusions."""
    family = family_for_difficulty(difficulty)
    if difficulty == 3:
        raise ValueError('the rational preset has no enumerated finite inventory')
    bound = CONSTANT_BOUNDS[difficulty]
    return frozenset(
        semantic_identity(family, {'a': a, 'b': b, 'c': a * solution + b})
        for a in range(-9, 10) if a
        for solution in range(-9, 10) if solution
        for b in range(-12, 13)
        if bound is None or abs(a * solution + b) <= bound
    )


def _spec(family, bindings, actions, seed, tag, index):
    spec = _mathir_spec(family, bindings, actions, tag, seed, index)
    spec['source'] = 'synthetic_mathir_level3_numeric_candidates_v3'
    return spec


@lru_cache(maxsize=4)
def template_support(difficulty: int) -> tuple[int, str]:
    family = family_for_difficulty(difficulty)
    bindings = sample_bindings(difficulty, _row_rng(0, difficulty, 0, 'certificate'))
    actions = dict(zip('ABCDEF', family.commands))
    keys = enumerate_mathir_action_menu_keys(
        _spec(family, bindings, actions, 0, 'support_certificate', 0))
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
    if domain != 'mathir':
        raise ValueError('this optional generator supports MathIR only')
    if not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier < 1:
        raise ValueError('multiplier must be a positive integer')
    family = family_for_difficulty(difficulty)
    target = Counter({int(key): int(count) for key, count in target.items() if count})
    if set(target) != {5} or target[5] < 1:
        raise ValueError('MathIR requires positive row counts in the five-mode support cell')
    desired, blocked = target[5] * multiplier, set(excluded)
    if difficulty != 3:
        available = len(finite_inventory(difficulty) - blocked)
        if desired > available:
            raise RuntimeError(f'preset {difficulty} has {available} unused identities; '
                               f'{desired} requested; the fixed binding law is not widened')
    mode_count, digest = template_support(difficulty)
    rows = []
    for index in range(desired):
        rng = _row_rng(seed, difficulty, index, 'bindings')
        for attempt in range(100_000):
            bindings = sample_bindings(difficulty, rng)
            identity = semantic_identity(family, bindings)
            if identity not in blocked:
                break
        else:
            raise RuntimeError(f'fixed preset {difficulty} exhausted its row proposal budget')
        blocked.add(identity)
        commands = list(family.commands)
        _row_rng(seed, difficulty, index, 'menu').shuffle(commands)
        actions = dict(zip('ABCDEF', commands))
        spec = _spec(family, bindings, actions, seed, tag, index)
        # Symbolic normalization does not substitute binding values. Nonzero a
        # (and the rational preset's excluded zero denominators) preserves all
        # transitions in the exhaustive template certificate for every row.
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
    return rows
