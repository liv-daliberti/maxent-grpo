"""Exact-support candidate pools for empirical Level 3 calibration.

These knobs propose harder instances; they do not claim a model difficulty
match. That requires the paired, frozen-budget development evaluation. All
identities are semantic instance identities, matching the Level 1/2 tools.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement
import json
from math import prod
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "ops", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_modebench_data import _countdown_prompt, _graph_prompt
from make_python_factor_mode_data import _row as _python_row
from oat_drgrpo.python_modebench import proper_divisors

Row = dict[str, Any]
DOMAINS = ("countdown", "graph_coloring", "python_factors")


def row_identity(domain: str, row: Row) -> tuple:
    spec = json.loads(row["answer"])
    if domain == "countdown":
        return (domain, tuple(sorted(spec["numbers"])), int(spec["target"]))
    if domain == "graph_coloring":
        return (domain, int(spec["n"]), tuple(sorted(tuple(edge) for edge in spec["edges"])),
                "".join("?" if value is None else str(value) for value in spec["partial_colors"]))
    if domain == "python_factors":
        return (domain, tuple(sorted(spec["cases"])))
    raise ValueError(domain)


def countdown_modes(numbers: tuple[int, ...]) -> dict[int, dict[str, str]]:
    """Enumerate every binary expression, quotienting only commutative children.

    The key construction is exactly ``_canonical_countdown_ast``: association
    remains significant and subtraction/division retain operand order. A subset
    dynamic program avoids repeatedly parsing equivalent expression strings.
    Representative expressions are retained for independent verifier audits.
    """
    @lru_cache(None)
    def build(mask: int) -> dict[Fraction, dict[str, str]]:
        if mask.bit_count() == 1:
            value = numbers[mask.bit_length() - 1]
            return {Fraction(value): {str(value): str(value)}}
        out: dict[Fraction, dict[str, str]] = defaultdict(dict)
        left_mask = (mask - 1) & mask
        while left_mask:
            right_mask = mask ^ left_mask
            if left_mask < right_mask:
                for a, left in build(left_mask).items():
                    for b, right in build(right_mask).items():
                        for ka, ea in left.items():
                            for kb, eb in right.items():
                                lo, hi = sorted((ka, kb))
                                out[a + b][f"add({lo},{hi})"] = f"({ea}+{eb})"
                                out[a * b][f"mul({lo},{hi})"] = f"({ea}*{eb})"
                                out[a - b][f"sub({ka},{kb})"] = f"({ea}-{eb})"
                                out[b - a][f"sub({kb},{ka})"] = f"({eb}-{ea})"
                                if b:
                                    out[a / b][f"div({ka},{kb})"] = f"({ea}/{eb})"
                                if a:
                                    out[b / a][f"div({kb},{ka})"] = f"({eb}/{ea})"
            left_mask = (left_mask - 1) & mask
        return dict(out)

    return {int(value): expressions for value, expressions in build((1 << len(numbers)) - 1).items()
            if value.denominator == 1}


def _countdown_pool(target: Counter, excluded: set, seed: int, tag: str, difficulty: int) -> list[Row]:
    rng = random.Random(seed)
    # Four operands keep exact enumeration practical while increasing arithmetic
    # magnitude and expression search difficulty far beyond Level 2's <=14.
    lower, upper = ((2, 18), (3, 32), (5, 64), (8, 99))[difficulty]
    seen = set(excluded)
    seen_numbers: set[tuple[int, ...]] = set()
    got: Counter = Counter()
    rows: list[Row] = []
    for attempt in range(max(2000, sum(target.values()) * 50)):
        numbers = tuple(sorted(rng.sample(range(lower, upper + 1), 4)))
        if numbers in seen_numbers:
            continue
        seen_numbers.add(numbers)
        candidates = []
        for value, expressions in countdown_modes(numbers).items():
            count = len(expressions)
            if value <= 0 or value in numbers or count not in target or got[count] >= target[count]:
                continue
            identity = ("countdown", numbers, value)
            if identity in seen:
                continue
            # Avoid an accidental pure magnitude ranking: a candidate's easiest
            # valid expression determines its score, including division need.
            min_divisions = min(key.count("div(") for key in expressions)
            min_multiplications = min(key.count("mul(") for key in expressions)
            score = (min_divisions, min_multiplications, len(str(value)), value)
            candidates.append((score, value, count))
        rng.shuffle(candidates)
        if difficulty:
            candidates.sort(reverse=True)
        # Keep at most one target per support cell for this operand set.
        used_support = set()
        for _, value, count in candidates:
            if count in used_support or got[count] >= target[count]:
                continue
            spec = {"verifier": "countdown", "numbers": list(numbers), "target": value,
                    "source": "synthetic_exact_countdown_level3", "instance_id": f"{tag}-{seed}-{len(rows)}",
                    "num_completions": count, "num_expressions": count}
            rows.append({"problem": _countdown_prompt(list(numbers), value),
                         "answer": json.dumps(spec, sort_keys=True), "modebench_task": "countdown",
                         "answer_mode_count": count, "answer_mode_split": tag,
                         "level3_difficulty": difficulty})
            seen.add(("countdown", numbers, value))
            used_support.add(count)
            got[count] += 1
        if got == target:
            rng.shuffle(rows)
            return rows
    raise RuntimeError(f"Countdown difficulty {difficulty} misses exact support {target - got}")


def graph_completion_count(n: int, edges: list[list[int]], partial: list[int | None], cap: int | None = None) -> int:
    """Count colorings by constraint propagation; an optional cap is rejection-only."""
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u - 1].add(v - 1)
        neighbors[v - 1].add(u - 1)
    colors = [0 if value is None else int(value) for value in partial]
    if any(colors[u - 1] and colors[u - 1] == colors[v - 1] for u, v in edges):
        return 0
    remaining = {index for index, value in enumerate(colors) if not value}
    count = 0

    def visit() -> None:
        nonlocal count
        if cap is not None and count > cap:
            return
        if not remaining:
            count += 1
            return
        options = {index: {1, 2, 3} - {colors[neighbor] for neighbor in neighbors[index]}
                   for index in remaining}
        index = min(remaining, key=lambda item: (len(options[item]), -len(neighbors[item]), item))
        remaining.remove(index)
        for color in sorted(options[index]):
            colors[index] = color
            visit()
        colors[index] = 0
        remaining.add(index)

    visit()
    return count


def _graph_pool(target: Counter, excluded: set, seed: int, tag: str, difficulty: int) -> list[Row]:
    rng = random.Random(seed)
    seen = set(excluded)
    got: Counter = Counter()
    rows: list[Row] = []
    hidden_count = (4, 5, 6, 7)[difficulty]
    n = (6, 8, 10, 12)[difficulty]
    for attempt in range(max(20000, sum(target.values()) * 1000)):
        planted = [rng.randint(1, 3) for _ in range(n)]
        hidden = set(rng.sample(range(n), hidden_count))
        partial = [None if index in hidden else color for index, color in enumerate(planted)]
        probability = rng.uniform(.32, .65)
        edges = [[u + 1, v + 1] for u in range(n) for v in range(u + 1, n)
                 if planted[u] != planted[v] and rng.random() < probability]
        if any(not any(index + 1 in edge for edge in edges) for index in hidden):
            continue
        count = graph_completion_count(n, edges, partial, cap=max(target))
        if count not in target or got[count] >= target[count]:
            continue
        spec = {"verifier": "graph_coloring", "n": n, "edges": edges,
                "partial_colors": partial, "source": "exact_answer_mode_graph_coloring_level3",
                "instance_id": f"{tag}-{seed}-{len(rows)}", "num_completions": count,
                "num_solutions": graph_completion_count(n, edges, [None] * n)}
        row = {"problem": _graph_prompt(n, edges, partial), "answer": json.dumps(spec, sort_keys=True),
               "modebench_task": "graph_coloring", "answer_mode_count": count,
               "answer_mode_split": tag, "level3_difficulty": difficulty}
        identity = row_identity("graph_coloring", row)
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(row)
        got[count] += 1
        if got == target:
            rng.shuffle(rows)
            return rows
    raise RuntimeError(f"Graph difficulty {difficulty} misses exact support {target - got}")


def _python_pool(target: Counter, excluded: set, seed: int, tag: str, difficulty: int) -> list[Row]:
    rng = random.Random(seed)
    upper = (192, 384, 640, 1000)[difficulty]
    # Keep rare prime-power divisor counts (e.g. seven proper divisors at 256).
    # A hard lower bound would silently make some exact support cells impossible.
    by_count: dict[int, list[int]] = defaultdict(list)
    divisors: dict[int, tuple[int, ...]] = {}
    for value in range(6, upper + 1):
        ds = proper_divisors(value)
        if len(ds) >= 2:
            divisors[value] = ds
            by_count[len(ds)].append(value)
    profiles: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for profile in combinations_with_replacement(sorted(by_count), 4):
        support = prod(profile)
        if support in target and all(len(by_count[count]) >= repeats for count, repeats in Counter(profile).items()):
            profiles[support].append(profile)
    missing = set(target) - set(profiles)
    if missing:
        raise RuntimeError(f"Python bound {upper} cannot realize support {sorted(missing)}")
    seen = set(excluded)
    rows: list[Row] = []
    for support, needed in sorted(target.items()):
        candidates: dict[tuple[int, ...], tuple] = {}
        # A small reservoir ranks intrinsic factorization complexity while
        # preserving all requested support cells and the unchanged four inputs.
        reservoir_size = max(needed * 5, 24)
        for attempt in range(max(10000, reservoir_size * 100)):
            profile = rng.choice(profiles[support])
            cases_list = []
            for count, repeats in Counter(profile).items():
                values = by_count[count]
                # Weight each input by magnitude and its smallest proper divisor;
                # this increases long arithmetic and reduces universal 2/3 rules.
                weights = [(value / upper + .15) ** difficulty *
                           (divisors[value][0] ** (.5 * difficulty)) for value in values]
                available = list(values)
                available_weights = list(weights)
                for _ in range(repeats):
                    value = rng.choices(available, weights=available_weights)[0]
                    index = available.index(value)
                    available.pop(index)
                    available_weights.pop(index)
                    cases_list.append(value)
            cases = tuple(sorted(cases_list))
            if max(cases) <= 96 or ("python_factors", cases) in seen or cases in candidates:
                continue
            common = set(divisors[cases[0]])
            for value in cases[1:]:
                common.intersection_update(divisors[value])
            score = (not common, min(divisors[value][0] for value in cases),
                     sum(divisors[value][0] for value in cases), sum(cases))
            candidates[cases] = score
            if len(candidates) >= reservoir_size:
                break
        if len(candidates) < needed:
            raise RuntimeError(f"Python support {support} only has {len(candidates)} candidates for {needed}")
        ranked = list(candidates)
        rng.shuffle(ranked)
        if difficulty:
            ranked.sort(key=candidates.__getitem__, reverse=True)
        for cases in ranked[:needed]:
            row = _python_row(cases=cases, split_tag=tag, seed=seed, index=len(rows))
            row["level3_difficulty"] = difficulty
            rows.append(row)
            seen.add(("python_factors", cases))
    rng.shuffle(rows)
    return rows


def build_pool(domain: str, target: Counter, excluded: set, seed: int, tag: str,
               difficulty: int, multiplier: int = 4) -> list[Row]:
    """Build exactly ``multiplier * target[support]`` fresh rows per support cell.

    ``excluded`` uses Level 1/2 identity tuples with their leading domain name.
    Callers must include all historical data and earlier generated split IDs.
    Inputs, verifiers and canonical outcome definitions remain unchanged.
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported discrete domain: {domain}")
    if difficulty not in range(4) or multiplier < 1:
        raise ValueError("difficulty must be 0..3 and multiplier must be positive")
    required = Counter({int(support): int(count) * multiplier for support, count in target.items() if count})
    if not required:
        return []
    if any(support < 2 or count < 0 for support, count in required.items()):
        raise ValueError("support must be >=2 and target counts must be nonnegative")
    builder = {"countdown": _countdown_pool, "graph_coloring": _graph_pool,
               "python_factors": _python_pool}[domain]
    rows = builder(required, excluded, seed, tag, difficulty)
    if Counter(int(row["answer_mode_count"]) for row in rows) != required:
        raise RuntimeError("Generated support histogram differs from requested histogram")
    identities = {row_identity(domain, row) for row in rows}
    if len(identities) != len(rows) or identities & excluded:
        raise RuntimeError("Generated identities overlap within pool or exclusions")
    return rows
