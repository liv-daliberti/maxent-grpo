#!/usr/bin/env python3
"""Fail-closed decisive-operation signatures for finite MATH route menus.

This module does not infer an open-ended semantic partition.  It recognizes
only a small set of explicit mathematical engines and calls two routes
distinct only for a frozen allowlist of incompatible engine pairs.  Missing,
mixed, or merely representational routes remain unclassified.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


_BRACKETED_LABEL = re.compile(
    r"\[(?:KERNEL|EQUIVALENT RENDERING):[^\]]*\]",
    flags=re.IGNORECASE,
)

_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "crt_construction": (
        re.compile(r"\bchinese remainder\b", re.I),
        re.compile(r"\bcrt\b", re.I),
    ),
    "euclidean_algorithm": (
        re.compile(r"\beuclidean algorithm\b", re.I),
        re.compile(r"\beuclidean (?:remainder|division)\b", re.I),
    ),
    "prime_factorization": (
        re.compile(r"\bprime factori[sz]", re.I),
        re.compile(r"\bprime exponent", re.I),
    ),
    "calculus": (
        re.compile(r"\bderivat", re.I),
        re.compile(r"\bdifferentiat", re.I),
        re.compile(r"\bcritical point", re.I),
        re.compile(r"\b(?:log[- ]?)?gradient", re.I),
    ),
    "sharp_inequality": (
        re.compile(r"\bam[\s-]*gm\b", re.I),
        re.compile(r"\barithmetic[\s-]*geometric mean\b", re.I),
        re.compile(r"\bcauchy(?:[\s-]*schwarz)?\b", re.I),
        re.compile(r"\bjensen", re.I),
        re.compile(r"\bsharp (?:bound|inequality)\b", re.I),
        re.compile(r"\bpositive reciprocal pair\b", re.I),
    ),
    "quadratic_formula": (
        re.compile(r"\bquadratic formula\b", re.I),
        re.compile(r"\bdiscriminant\b", re.I),
        re.compile(r"\broot formula\b", re.I),
    ),
    "vieta_relations": (
        re.compile(r"\bvieta", re.I),
        re.compile(r"\bsum (?:of )?(?:the )?roots\b", re.I),
        re.compile(r"\bproduct (?:of )?(?:the )?roots\b", re.I),
    ),
    "polynomial_expansion": (
        re.compile(r"(?<!without )\bexpand(?:ed|ing|s)?\b", re.I),
        re.compile(r"\bpolynomial expansion\b", re.I),
        re.compile(r"\bexplicit coefficients\b", re.I),
    ),
    "strategic_value": (
        re.compile(r"\bcoefficient sum (?:as|is)\b", re.I),
        re.compile(r"\bfunction value\b", re.I),
        re.compile(r"\bvalue invariant\b", re.I),
        re.compile(r"\binverse translation\b", re.I),
        re.compile(r"\bevaluat\w* (?:the polynomial )?at [+-]?\d", re.I),
    ),
    "finite_constraint_search": (
        re.compile(r"\benumerat", re.I),
        re.compile(r"\bexhaust", re.I),
        re.compile(r"\btest each\b", re.I),
        re.compile(r"\bcheck each\b", re.I),
        re.compile(r"\bfilter\b", re.I),
        re.compile(r"\bgenerate (?:all |every )?(?:candidate|integer)", re.I),
    ),
    "symbolic_interval": (
        re.compile(r"\bsolve\w* (?:symbolically )?for (?:the )?interval\b", re.I),
        re.compile(r"\binterval endpoint", re.I),
        re.compile(r"\binteger lattice point", re.I),
        re.compile(r"\bceiling/floor\b", re.I),
        re.compile(r"\bceiling\b.*\bfloor\b", re.I | re.S),
    ),
    "factor_localization": (
        re.compile(r"\bsquare[\s-]*root bound", re.I),
        re.compile(r"\b(?:consecutive|adjacent) factor", re.I),
        re.compile(r"\blocali[sz]\w*.*\bfactor", re.I | re.S),
    ),
    "extremal_balance": (
        re.compile(r"\bbalanced integer", re.I),
        re.compile(r"\bconvexity identity\b", re.I),
        re.compile(r"\bminimi[sz]\w* (?:the )?(?:side )?difference\b", re.I),
        re.compile(r"\bfixed (?:odd )?sum\b.*\bparity\b", re.I | re.S),
    ),
    "vector_geometry": (
        re.compile(r"\bcross[\s-]*product\b", re.I),
        re.compile(r"\bvector geometry\b", re.I),
        re.compile(r"\bdeterminant\b.*\barea\b", re.I | re.S),
    ),
    "synthetic_geometry": (
        re.compile(r"\bisosceles\b.*\baltitude\b", re.I | re.S),
        re.compile(r"\bside[\s-]*length\b.*\baltitude\b", re.I | re.S),
        re.compile(r"\bbase (?:times|and) (?:the )?altitude\b", re.I),
    ),
    "dynamic_programming": (
        re.compile(r"\bdynamic programming\b", re.I),
        re.compile(r"\bdp (?:table|state|recurrence)\b", re.I),
        re.compile(r"\brecurrence\b.*\bstate", re.I | re.S),
    ),
    "closed_form_count": (
        re.compile(r"\bclosed[\s-]*form (?:count|probability|formula)\b", re.I),
        re.compile(r"\bbinomial (?:coefficient|formula|probability)\b", re.I),
    ),
    "complementary_counting": (
        re.compile(r"\bcomplement(?:ary)? (?:count|probability|event|method)\b", re.I),
        re.compile(r"\bprobability of (?:neither|no )", re.I),
        re.compile(r"\bone minus\b.*\bprobability\b", re.I | re.S),
    ),
    "inclusion_exclusion": (
        re.compile(r"\binclusion[\s-]*exclusion\b", re.I),
        re.compile(r"\bdirect union\b", re.I),
        re.compile(r"\bsum\b.*\bsubtract\b.*\bintersection\b", re.I | re.S),
    ),
    "incidence_double_count": (
        re.compile(r"\bdouble count", re.I),
        re.compile(r"\bincidence", re.I),
        re.compile(r"\bdegree sum\b", re.I),
    ),
    "unordered_pair_count": (
        re.compile(r"\bunordered (?:player )?pairs?\b", re.I),
        re.compile(r"\b(?:choose|select) (?:any )?2\b", re.I),
        re.compile(r"\\binom\{[^}]+\}\{2\}", re.I),
    ),
}


# These are the only distinctions the automatic gate may create.  Every pair
# names incompatible decisive engines, not alternative surface renderings of
# one calculation.
SAFE_DISTINCT_PAIRS = frozenset(
    {
        frozenset(("crt_construction", "finite_constraint_search")),
        frozenset(("calculus", "sharp_inequality")),
        frozenset(("euclidean_algorithm", "prime_factorization")),
        frozenset(("quadratic_formula", "vieta_relations")),
        frozenset(("polynomial_expansion", "strategic_value")),
        frozenset(("finite_constraint_search", "symbolic_interval")),
        frozenset(("finite_constraint_search", "extremal_balance")),
        frozenset(("factor_localization", "quadratic_formula")),
        frozenset(("vector_geometry", "synthetic_geometry")),
        frozenset(("dynamic_programming", "closed_form_count")),
        frozenset(("complementary_counting", "inclusion_exclusion")),
        frozenset(("incidence_double_count", "unordered_pair_count")),
    }
)


def _route_text(route: Mapping[str, Any]) -> str:
    """Render only the declared method, never the problem or final answer."""

    parts = [str(route.get("label") or ""), str(route.get("plan") or "")]
    actions = route.get("actions") or ()
    for action in actions:
        if isinstance(action, Mapping):
            parts.append(str(action.get("operation") or ""))
        else:
            parts.append(str(action))
    return _BRACKETED_LABEL.sub("", "\n".join(parts))


def signature_hits(route: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every explicit decisive-operation signature found."""

    text = _route_text(route)
    hits = {
        name
        for name, patterns in _PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    }

    # Listing final CRT representatives and comparing interval endpoints are
    # subordinate execution steps, not independent exhaustive searches.
    if "crt_construction" in hits:
        hits.discard("finite_constraint_search")
    if "symbolic_interval" in hits:
        hits.discard("finite_constraint_search")
    if "factor_localization" in hits:
        hits.discard("finite_constraint_search")
    if "calculus" in hits:
        hits.discard("finite_constraint_search")

    return tuple(sorted(hits))


def decisive_signature(route: Mapping[str, Any]) -> str | None:
    """Return one unambiguous signature, otherwise fail closed."""

    hits = signature_hits(route)
    return hits[0] if len(hits) == 1 else None


def safe_distinct_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[bool, str | None, str | None]:
    """Decide distinctness only through the frozen incompatible-pair list."""

    left_signature = decisive_signature(left)
    right_signature = decisive_signature(right)
    pair = (
        frozenset((left_signature, right_signature))
        if left_signature is not None and right_signature is not None
        else frozenset()
    )
    return (
        len(pair) == 2 and pair in SAFE_DISTINCT_PAIRS,
        left_signature,
        right_signature,
    )


def strategy_route(
    menu: Mapping[str, Any],
    strategy_id: str,
) -> dict[str, Any]:
    """Extract one strategy and its referenced operations from a menu."""

    actions = {
        str(row["action_id"]): row
        for row in menu.get("actions") or ()
        if isinstance(row, Mapping) and "action_id" in row
    }
    strategy = next(
        (
            row
            for row in menu.get("strategies") or ()
            if isinstance(row, Mapping)
            and str(row.get("strategy_id")) == str(strategy_id)
        ),
        None,
    )
    if strategy is None:
        raise ValueError(f"menu has no strategy {strategy_id}")
    action_ids: Iterable[Any] = strategy.get("action_ids") or ()
    selected = []
    for raw_action_id in action_ids:
        action_id = str(raw_action_id)
        if action_id not in actions:
            raise ValueError(f"strategy references unknown action {action_id}")
        selected.append(actions[action_id])
    return {
        "plan": str(strategy.get("plan") or ""),
        "actions": selected,
    }


def safe_menu_pair(
    menu: Mapping[str, Any],
) -> tuple[bool, str | None, str | None]:
    """Apply the safe pair gate to an exact two-strategy finite menu."""

    strategies = menu.get("strategies") or ()
    ids = [
        str(row.get("strategy_id"))
        for row in strategies
        if isinstance(row, Mapping)
    ]
    if ids != ["S1", "S2"]:
        return False, None, None
    return safe_distinct_pair(
        strategy_route(menu, "S1"),
        strategy_route(menu, "S2"),
    )
