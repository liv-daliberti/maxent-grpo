"""Fresh small-graph presets for two-metric Level 3 difficulty calibration.

Level 1 includes 4-, 5-, and 6-vertex graphs. Forcing six vertices removed its
high-success component, even with three missing colors. These presets vary
small-graph size and whether hidden vertices must be solved jointly. They never
inspect model outcomes. All retain the original prompt and exact verifier.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from modebench_level3_discrete import _graph_prompt, graph_completion_count, row_identity
from make_modebench_data import _valid_graph_colorings

SCHEMA = "modebench_level3_graph_candidate_v3"
PRESETS = {
    0: "five_vertices_independent_hidden_except_prime_five_support",
    1: "twenty_percent_four_vertices_otherwise_five",
    2: "five_vertices_original_topology_distribution",
    3: "twenty_percent_four_thirty_percent_five_otherwise_six",
}


def size_targets(target: Counter, difficulty: int) -> dict[int, Counter]:
    """Stable per-support size quotas, leaving enough scarce n=4 identities.

    The nine-completion cell requires at least two differently colored shown
    vertices, hence n>=5 for three hidden vertices. Five modes also require
    n>=5; eighteen modes need n>=6 under the original minimum-edge rule.
    Quotas use integer floors;
    their deviations from stated fractions are at most one row per cell.
    """
    sizes: dict[int, Counter] = {4: Counter(), 5: Counter(), 6: Counter()}
    for support, count in sorted(target.items()):
        if support == 18 and difficulty != 0:
            sizes[6][support] = count
            continue
        four = count // 5 if difficulty in (1, 3) and support in (4, 6, 8, 12) else 0
        five = count * 3 // 10 if difficulty == 3 else count - four
        six = count - four - five
        sizes[4][support], sizes[5][support], sizes[6][support] = four, five, six
    return {n: Counter({support: count for support, count in counts.items() if count})
            for n, counts in sizes.items() if any(counts.values())}


def _candidate(n: int, rng: random.Random, local: bool) -> tuple[list[list[int]], list[int | None]] | None:
    hidden = set(rng.sample(range(n), 3))
    if local:
        partial = [None if index in hidden else rng.randint(1, 3) for index in range(n)]
        edges = []
        for u, v in combinations(range(n), 2):
            if u in hidden and v in hidden:
                continue
            if partial[u] is not None and partial[u] == partial[v]:
                continue
            if rng.random() < .5:
                edges.append([u + 1, v + 1])
        return edges, partial
    possible = list(combinations(range(1, n + 1), 2))
    edge_count = rng.randint(n - 2, min(8, len(possible)))
    edges = [list(edge) for edge in sorted(rng.sample(possible, edge_count))]
    colorings = _valid_graph_colorings(n, edges)
    if not colorings:
        return None
    coloring = rng.choice(colorings)
    partial = [None if index in hidden else color for index, color in enumerate(coloring)]
    return edges, partial


def build_pool(domain: str, target: Counter, excluded: set, seed: int, tag: str,
               difficulty: int, multiplier: int = 4) -> list[dict[str, Any]]:
    """Return exact support counts times ``multiplier``, excluding all prior IDs.

    Difficulty denotes a structural preset, not an empirical performance order.
    Caller-provided exclusions must include history and earlier candidate pools.
    """
    if domain != "graph_coloring":
        raise ValueError("graph v3 only supports graph_coloring")
    if difficulty not in PRESETS or multiplier < 1:
        raise ValueError("difficulty must be 0..3 and multiplier must be positive")
    required = Counter({int(support): int(count) * multiplier
                        for support, count in target.items() if count})
    if any(support < 2 or count < 0 for support, count in required.items()):
        raise ValueError("support must be >=2 and counts must be nonnegative")
    if not required:
        return []
    if set(required) - {4, 5, 6, 8, 9, 12, 18}:
        raise ValueError("graph v3 presets implement the frozen Level 1 support cells 4,5,6,8,9,12,18")
    rng = random.Random(seed)
    blocked = set(excluded)
    rows: list[dict[str, Any]] = []
    for n, wanted in size_targets(required, difficulty).items():
        got: Counter = Counter()
        for _ in range(max(30_000, sum(wanted.values()) * 1000)):
            # Five is prime and cannot be a product of independent color
            # domains of sizes 1,2,3. Its rare frozen train/eval cell
            # therefore retains the original coupled graph distribution.
            need_five = difficulty == 0 and got[5] < wanted[5]
            candidate = _candidate(n, rng, local=difficulty == 0 and not need_five)
            if candidate is None:
                continue
            edges, partial = candidate
            count = graph_completion_count(n, edges, partial, cap=max(wanted))
            if (count not in wanted or got[count] >= wanted[count]
                    or need_five and count != 5):
                continue
            identity = (domain, n, tuple(tuple(edge) for edge in edges),
                        "".join("?" if color is None else str(color) for color in partial))
            if identity in blocked:
                continue
            spec = {
                "verifier": "graph_coloring", "n": n, "edges": edges,
                "partial_colors": partial, "source": SCHEMA,
                "instance_id": f"{tag}-{seed}-{len(rows)}",
                "num_completions": count,
                "num_solutions": graph_completion_count(n, edges, [None] * n),
            }
            rows.append({
                "problem": _graph_prompt(n, edges, partial),
                "answer": json.dumps(spec, sort_keys=True),
                "modebench_task": domain, "answer_mode_count": count,
                "answer_mode_split": tag, "level3_difficulty": difficulty,
                "level3_generator": SCHEMA, "level3_graph_preset": PRESETS[difficulty],
            })
            blocked.add(identity)
            got[count] += 1
            if got == wanted:
                break
        if got != wanted:
            raise RuntimeError(f"graph v3 preset {difficulty}, n={n} misses support {wanted - got}; "
                               "fresh small-graph identities are finite; no reuse or size fallback is permitted")
    rng.shuffle(rows)
    identities = {row_identity(domain, row) for row in rows}
    if len(identities) != len(rows) or identities & excluded:
        raise RuntimeError("graph v3 semantic identity disjointness failed")
    if Counter(row["answer_mode_count"] for row in rows) != required:
        raise RuntimeError("graph v3 support histogram differs from target")
    return rows
