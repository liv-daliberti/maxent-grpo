"""Stable-distribution Countdown presets for matched Level 3 calibration.

The smallest preset restores the shallow, small-target four-operand region
missing from the initial Level 3 pools. Every preset keeps exact canonical
support and the existing prompt/verifier. All proposal weights are fixed before
sampling and independent of requested row counts or model outcomes.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from modebench_level3_discrete import _countdown_prompt, countdown_modes, row_identity

SCHEMA = "modebench_level3_countdown_candidate_v2"
PRESETS = ((2, 14, 56), (2, 18, 72), (3, 32, None), (5, 64, None))
SAMPLING_LAW = "independent_support_stream_uniform_operand_proposals_fixed_target_weights_without_replacement"


@lru_cache(maxsize=2048)
def _target_statistics(numbers: tuple[int, ...]) -> tuple[tuple[int, int, int, int, int], ...]:
    """Exact support and intrinsic expression statistics; no quota dependence."""
    paired, one_product = set(), set()
    for a, b, c, d in permutations(numbers):
        paired.update((a * b + c * d, a * b - c * d, (a + b) * (c + d), (a + b) * (c - d)))
        one_product.update((a * b + c + d, a * b + c - d, a * b - c + d, a * b - c - d))
    statistics = []
    for value, expressions in countdown_modes(numbers).items():
        if value <= 0 or value in numbers:
            continue
        support = len(expressions)
        if support not in (2, 3, 4, 5, 6, 7, 8):
            continue
        family = 0 if value in paired else 1 if value in one_product else 2
        min_divisions = min(key.count("div(") for key in expressions)
        min_products = min(key.count("mul(") for key in expressions)
        statistics.append((value, support, family, min_divisions, min_products))
    return tuple(sorted(statistics))


def proposal_weight(numbers: tuple[int, ...], statistic: tuple[int, int, int, int, int],
                    difficulty: int) -> float:
    """One frozen positive weight per target; rare cells never get rank cutoffs."""
    value, _support, family, min_divisions, min_products = statistic
    if difficulty < 2:
        family_weight = ((12.0, 6.0, 1.0), (6.0, 3.0, 1.0))[difficulty][family]
        return family_weight / (1.0 + abs(value - sum(numbers)) / PRESETS[difficulty][1])
    return (1.0 + 3.0 * min_divisions + min_products) * (1.0 + math.log1p(value))


def _stream_seed(seed: int, difficulty: int, support: int) -> int:
    payload = f"{SCHEMA}:{seed}:{difficulty}:{support}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_pool(domain: str, target: Counter, excluded: set, seed: int, tag: str,
               difficulty: int, multiplier: int = 4) -> list[dict[str, Any]]:
    """Produce exactly the requested support histogram times ``multiplier``.

    Each support cell has its own RNG, so larger requests extend the same row
    sequence within that cell. Exclusions reject semantic identities without
    changing proposal weights. Full-pool order is shuffled separately.
    """
    if domain != "countdown":
        raise ValueError("Countdown v2 only supports countdown")
    if difficulty not in range(4) or multiplier < 1:
        raise ValueError("difficulty must be 0..3 and multiplier must be positive")
    required = Counter({int(support): int(count) * multiplier
                        for support, count in target.items() if count})
    if any(support not in range(2, 9) or count < 0 for support, count in required.items()):
        raise ValueError("Countdown v2 requires positive counts and frozen support cells 2..8")
    if not required:
        return []
    lower, upper, target_cap = PRESETS[difficulty]
    blocked = set(excluded)
    rows: list[dict[str, Any]] = []
    for support, needed in sorted(required.items()):
        rng = random.Random(_stream_seed(seed, difficulty, support))
        accepted = 0
        for _ in range(max(100_000, needed * 2000)):
            numbers = tuple(sorted(rng.sample(range(lower, upper + 1), 4)))
            choices = [statistic for statistic in _target_statistics(numbers)
                       if statistic[1] == support and
                       (target_cap is None or statistic[0] <= target_cap)]
            if not choices:
                continue
            selected = rng.choices(choices, weights=[proposal_weight(numbers, item, difficulty)
                                                     for item in choices])[0]
            value, _count, family, _divisions, _products = selected
            identity = (domain, numbers, value)
            if identity in blocked:
                continue
            spec = {
                "verifier": domain, "numbers": list(numbers), "target": value,
                "source": SCHEMA,
                "instance_id": f"{tag}-{seed}-m{support}-{accepted}",
                "num_completions": support, "num_expressions": support,
            }
            rows.append({
                "problem": _countdown_prompt(list(numbers), value),
                "answer": json.dumps(spec, sort_keys=True),
                "modebench_task": domain, "answer_mode_count": support,
                "answer_mode_split": tag, "level3_difficulty": difficulty,
                "level3_generator": SCHEMA, "level3_sampling_law": SAMPLING_LAW,
                "level3_support_sampling_index": accepted,
                "level3_countdown_target_family": ("paired_products", "one_product", "other")[family],
            })
            blocked.add(identity)
            accepted += 1
            if accepted == needed:
                break
        if accepted != needed:
            raise RuntimeError(f"Countdown v2 preset {difficulty}, support {support}: "
                               f"only {accepted}/{needed} fresh proposals accepted")
    random.Random(seed).shuffle(rows)
    identities = {row_identity(domain, row) for row in rows}
    if len(identities) != len(rows) or identities & excluded:
        raise RuntimeError("Countdown v2 semantic identity disjointness failed")
    if Counter(row["answer_mode_count"] for row in rows) != required:
        raise RuntimeError("Countdown v2 support histogram differs from target")
    return rows
