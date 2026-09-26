"""Refined graph pools spanning easy and hard relative-difficulty regimes.

The initial 3B pilot was already too hard in pass@1 at four hidden vertices.
These presets include fresh Level-1-sized instances so development calibration
can match both pass@1 and pass@8. Preset numbers are proposals, not guarantees
of monotone empirical difficulty. The original discrete generator is immutable.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from modebench_level3_discrete import (
    _graph_prompt,
    graph_completion_count,
    row_identity,
)

SCHEMA = "modebench_level3_graph_candidate_v2"
PRESETS = ((6, 3), (8, 3), (8, 4), (10, 6))


def build_pool(domain: str, target: Counter, excluded: set, seed: int, tag: str,
               difficulty: int, multiplier: int = 4) -> list[dict[str, Any]]:
    """Return exactly ``target[support] * multiplier`` fresh rows per cell.

    Exclusions use the existing graph semantic identity tuples, with their
    leading ``graph_coloring`` domain. Callers supply all historical and prior
    candidate identities. Three-hidden-vertex presets permit isolated vertices:
    excluding these would make the inherited nine-completion cell impossible.
    """
    if domain != "graph_coloring":
        raise ValueError("graph v2 only supports graph_coloring")
    if difficulty not in range(len(PRESETS)) or multiplier < 1:
        raise ValueError("difficulty must be 0..3 and multiplier must be positive")
    required = Counter({int(support): int(count) * multiplier
                        for support, count in target.items() if count})
    if any(support < 2 or count < 0 for support, count in required.items()):
        raise ValueError("support must be >=2 and counts must be nonnegative")
    if not required:
        return []
    n, hidden_count = PRESETS[difficulty]
    if max(required) > 3 ** hidden_count:
        raise ValueError("requested support exceeds the preset's full coloring space")
    rng = random.Random(seed)
    blocked = set(excluded)
    got: Counter = Counter()
    rows: list[dict[str, Any]] = []
    for _ in range(max(20_000, sum(required.values()) * 1000)):
        planted = [rng.randint(1, 3) for _ in range(n)]
        hidden = set(rng.sample(range(n), hidden_count))
        partial = [None if index in hidden else color for index, color in enumerate(planted)]
        probability = rng.uniform(.22, .60) if hidden_count == 3 else rng.uniform(.32, .65)
        edges = [[u + 1, v + 1] for u in range(n) for v in range(u + 1, n)
                 if planted[u] != planted[v] and rng.random() < probability]
        if len(edges) < 2:
            continue
        count = graph_completion_count(n, edges, partial, cap=max(required))
        if count not in required or got[count] >= required[count]:
            continue
        spec = {
            "verifier": "graph_coloring", "n": n, "edges": edges,
            "partial_colors": partial, "source": SCHEMA,
            "instance_id": f"{tag}-{seed}-{len(rows)}",
            "num_completions": count,
            "num_solutions": graph_completion_count(n, edges, [None] * n),
        }
        row = {
            "problem": _graph_prompt(n, edges, partial),
            "answer": json.dumps(spec, sort_keys=True),
            "modebench_task": domain, "answer_mode_count": count,
            "answer_mode_split": tag, "level3_difficulty": difficulty,
            "level3_generator": SCHEMA,
        }
        identity = row_identity(domain, row)
        if identity in blocked:
            continue
        blocked.add(identity)
        rows.append(row)
        got[count] += 1
        if got == required:
            rng.shuffle(rows)
            identities = {row_identity(domain, row) for row in rows}
            if len(identities) != len(rows) or identities & excluded:
                raise RuntimeError("graph v2 identity disjointness failed")
            return rows
    raise RuntimeError(f"graph v2 preset {difficulty} misses support {required - got}")
