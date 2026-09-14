"""Outcome-independent controlled rounding of Level 3 mixture cell counts.

Every cell keeps its exact row count and every tier keeps the global Hamilton
count. Each cell/tier allocation is either floor(n*w/20) or ceil(n*w/20).
The binary transportation problem selects only fractional-cell extras; it never
changes weights or silently relaxes either margin when a solve fails.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
from numbers import Integral
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

DENOMINATOR = 20
TIERS = 4
SCHEMA = 'modebench_level3_controlled_matrix_rounding_v1'


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _weights(weights) -> tuple[int, int, int, int]:
    if not isinstance(weights, (tuple, list)) or len(weights) != TIERS:
        raise ValueError('exactly four integer mixture weights are required')
    values = tuple(_integer(value, 'weight') for value in weights)
    if sum(values) != DENOMINATOR:
        raise ValueError('mixture weights must sum to 20')
    return values


def hamilton_totals(count: int, weights, seed: int) -> tuple[int, int, int, int]:
    """Global tier margins, matching fitter.hamilton(count, weights, (), seed)."""
    count, seed, weights = _integer(count, 'count'), _integer(seed, 'seed'), _weights(weights)
    result = [count * weight // DENOMINATOR for weight in weights]
    order = sorted(range(TIERS), key=lambda tier: (-(count * weights[tier] % DENOMINATOR),
                                                  _sha([seed, [], tier])))
    for tier in order[:count - sum(result)]:
        result[tier] += 1
    return tuple(result)


@lru_cache(maxsize=4096)
def _allocate_cached(cells: tuple[tuple[tuple, int], ...], weights: tuple[int, ...],
                     seed: int) -> tuple[tuple[int, int, int, int], ...]:
    if not cells:
        return ()
    counts = [count for _, count in cells]
    floors = [[count * weight // DENOMINATOR for weight in weights] for count in counts]
    remainders = [[count * weight % DENOMINATOR for weight in weights] for count in counts]
    totals = hamilton_totals(sum(counts), weights, seed)
    row_demand = [count - sum(row) for count, row in zip(counts, floors)]
    column_demand = [totals[tier] - sum(row[tier] for row in floors) for tier in range(TIERS)]
    variables = [(row, tier) for row in range(len(cells)) for tier in range(TIERS)
                 if remainders[row][tier]]
    if not variables:
        if any(row_demand) or any(column_demand):
            raise RuntimeError('no fractional cells can meet the prescribed global Hamilton margins')
        return tuple(tuple(row) for row in floors)
    rows, columns, values, costs = [], [], [], []
    # One integer unit of remainder preference dominates the sum of every hash
    # tie-break cost. Hashes select reproducibly among equally good roundings.
    preference_scale = len(variables) + 1
    for variable, (row, tier) in enumerate(variables):
        rows.extend((row, len(cells) + tier))
        columns.extend((variable, variable))
        values.extend((1., 1.))
        hash_tie = int(_sha([SCHEMA, seed, list(cells[row][0]), tier])[:13], 16) / (16 ** 13)
        costs.append(-preference_scale * remainders[row][tier] + hash_tie)
    matrix = coo_matrix((values, (rows, columns)),
                        shape=(len(cells) + TIERS, len(variables))).tocsc()
    demand = np.asarray(row_demand + column_demand, dtype=float)
    result = milp(c=np.asarray(costs), integrality=np.ones(len(variables)), bounds=Bounds(0., 1.),
                  constraints=LinearConstraint(matrix, demand, demand),
                  options={'presolve': True, 'mip_rel_gap': 0.0, 'time_limit': 30.0})
    if not result.success or result.x is None:
        raise RuntimeError('controlled rounding cannot certify the prescribed global Hamilton margins '
                           f'with per-cell floor/ceil bounds: status={result.status}; {result.message}; '
                           f'weights={weights}; global_totals={totals}; cells={cells}')
    for (row, tier), value in zip(variables, result.x):
        rounded = int(round(float(value)))
        if rounded not in (0, 1) or abs(value - rounded) > 1e-7:
            raise RuntimeError('allocation solver returned a nonbinary extra')
        floors[row][tier] += rounded
    for (key, count), allocated in zip(cells, floors):
        if sum(allocated) != count:
            raise RuntimeError(f'allocation does not preserve cell {key}')
        if any(value not in (count * weight // DENOMINATOR,
                             (count * weight + DENOMINATOR - 1) // DENOMINATOR)
               for value, weight in zip(allocated, weights)):
            raise RuntimeError(f'allocation exceeds floor/ceil bounds for cell {key}')
    if tuple(sum(row[tier] for row in floors) for tier in range(TIERS)) != totals:
        raise RuntimeError('allocation does not preserve global tier margins')
    return tuple(tuple(row) for row in floors)


def allocate_cells(target: Counter[tuple, int], weights, seed: int) -> dict[tuple, tuple[int, int, int, int]]:
    """Allocate exact cell and global tier margins, without model outcomes.

    Zero-count cells are retained as zero vectors. Cached values are immutable;
    callers receive a fresh dictionary. Cache keys include every target count,
    all four integer weights, and the deterministic tie-break seed.
    """
    weights, seed = _weights(weights), _integer(seed, 'seed')
    if not hasattr(target, 'items'):
        raise ValueError('target must map tuple cell keys to nonnegative integer counts')
    entries = []
    for key, count in target.items():
        if not isinstance(key, tuple):
            raise ValueError('cell keys must be tuples')
        try:
            serialized = json.dumps(list(key), sort_keys=True, separators=(',', ':'), allow_nan=False)
            hash(key)
        except (TypeError, ValueError) as error:
            raise ValueError('cell keys must be hashable and JSON serializable') from error
        entries.append((serialized, key, _integer(count, 'cell count')))
    entries.sort(key=lambda entry: entry[0])
    cells = tuple((key, count) for _, key, count in entries)
    allocated = _allocate_cached(cells, weights, seed)
    return {key: counts for (key, _), counts in zip(cells, allocated)}


def allocation_cache_info():
    return _allocate_cached.cache_info()


def clear_allocation_cache() -> None:
    _allocate_cached.cache_clear()
