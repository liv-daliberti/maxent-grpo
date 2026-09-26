"""Support-matched Level 3 candidate pools for MathIR and Pantry.

These are candidate difficulty settings, not claims of empirical calibration.
Both domains retain their existing executable verifiers and answer grammars.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "ops", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_mathir_action_menu_data import Family, _prompt as mathir_prompt
from make_pantry_plan_mode_data import (
    FAMILY_POOLS,
    SODIUM_CAPS,
    _canonical_sha256,
    _prompt as pantry_prompt,
)
from oat_drgrpo.mathir import (
    MATHIR_MENU_VERIFIER,
    MATHIR_MENU_VERSION,
    enumerate_mathir_action_menu_keys,
)
from oat_drgrpo.pantry_plan import (
    PANTRY_PLAN_VERIFIER,
    PANTRY_PLAN_VERSION,
    parse_pantry_plan_spec,
    validate_pantry_plan,
)


def _mathir_family(difficulty: int) -> Family:
    """Six familiar action IDs; difficulty comes from equation structure."""
    if difficulty == 0:
        lhs, rhs = "add(mul(a,x),b)", "add(mul(d,x),c)"
        display = "a*x + b = d*x + c"
        coefficient, offset, other = "a", "b", "d"
    elif difficulty == 1:
        lhs, rhs = "add(div(mul(a,x),e),div(b,f))", "add(mul(d,x),c)"
        display = "a*x/e + b/f = d*x + c"
        coefficient, offset, other = "div(a,e)", "div(b,f)", "d"
    elif difficulty == 2:
        lhs, rhs = "div(add(mul(a,x),b),e)", "div(add(mul(d,x),c),f)"
        display = "(a*x + b)/e = (d*x + c)/f"
        coefficient, offset, other = "div(a,e)", "div(b,e)", "div(d,f)"
    else:
        lhs, rhs = "div(mul(a,add(x,b)),e)", "div(sub(c,mul(d,x)),f)"
        display = "a*(x + b)/e = (c - d*x)/f"
        coefficient, offset, other = "div(a,e)", "div(mul(a,b),e)", "neg(div(d,f))"
    variable = f"mul({other},x)"
    # The final template needs a distributed negation to stay below the
    # unchanged verifier's eleven-node action-argument limit.
    if difficulty == 3:
        remove_variable = "add(div(mul(d,x),f))"
        combined = f"sub(sub({offset},div(mul(d,x),f)))"
        divide = f"div(add({coefficient},div(d,f)))"
    else:
        remove_variable = f"sub({variable})"
        combined = f"sub(add({variable},{offset}))"
        divide = f"div(sub({coefficient},{other}))"
    subtract = f"sub({offset})"
    return Family(
        name="ax_plus_b_eq_dx_plus_c" if difficulty == 0 else f"level3_rational_equation_{difficulty}",
        initial_lhs=lhs,
        initial_rhs=rhs,
        display_equation=display,
        commands=(subtract, remove_variable, divide, combined,
                  f"add({offset})", f"div({coefficient})"),
        certified_routes=((subtract, remove_variable, divide),
                          (remove_variable, subtract, divide),
                          (combined, divide)),
    )


def _mathir_spec(family, bindings, actions, tag, seed, index):
    return {
        "verifier": MATHIR_MENU_VERIFIER,
        "mathir_version": MATHIR_MENU_VERSION,
        "bindings": bindings,
        "initial_lhs": family.initial_lhs,
        "initial_rhs": family.initial_rhs,
        "max_steps": 4,
        "actions": actions,
        "support_is_open": False,
        "source": "synthetic_mathir_level3_candidates_v1",
        "family": family.name,
        "instance_id": f"{tag}-{seed}-{index}",
    }


@lru_cache(maxsize=4)
def _mathir_template_support(difficulty: int) -> tuple[int, str]:
    family = _mathir_family(difficulty)
    bindings = dict(a=7, b=11, c=13, d=-3)
    if difficulty:
        bindings.update(e=5, f=2)
    actions = dict(zip("ABCDEF", family.commands))
    keys = enumerate_mathir_action_menu_keys(
        _mathir_spec(family, bindings, actions, "certificate", 0, 0)
    )
    digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    return len(keys), digest


def _mathir_pool(target, excluded, seed, tag, difficulty, multiplier):
    family = _mathir_family(difficulty)
    count, digest = _mathir_template_support(difficulty)
    if set(target) != {count}:
        raise ValueError(f"MathIR template has {count} modes; requested {dict(target)}")
    rng = random.Random(seed)
    blocked = set(excluded)
    rows = []
    magnitude = (9, 13, 19, 29)[difficulty]
    values = [v for v in range(-magnitude, magnitude + 1) if v]
    while len(rows) < sum(target.values()) * multiplier:
        bindings = {key: rng.choice(values) for key in ("abcd" if not difficulty else "abcdef")}
        if difficulty == 0:
            nonzero_coefficient = bindings["a"] != bindings["d"]
        elif difficulty == 1:
            nonzero_coefficient = bindings["a"] != bindings["d"] * bindings["e"]
        else:
            right = bindings["d"] * bindings["e"] * (-1 if difficulty == 3 else 1)
            nonzero_coefficient = bindings["a"] * bindings["f"] != right
        identity = ("mathir", family.name, tuple(sorted(bindings.items())))
        if not nonzero_coefficient or identity in blocked:
            continue
        blocked.add(identity)
        commands = list(family.commands)
        rng.shuffle(commands)
        actions = dict(zip("ABCDEF", commands))
        spec = _mathir_spec(family, bindings, actions, tag, seed, len(rows))
        # Symbolic normalization, repeated-state rejection and terminal tests
        # do not substitute bindings. All possible multiplicative arguments
        # are nonzero here, so the enumerated template support is invariant.
        spec.update(num_completions=count, valid_mode_count=count,
                    valid_mode_key_sha256=digest)
        rows.append({
            "problem": mathir_prompt(family, bindings, actions),
            "answer": json.dumps(spec, sort_keys=True, separators=(",", ":")),
            "modebench_task": MATHIR_MENU_VERIFIER,
            "answer_mode_count": count,
            "answer_mode_split": tag,
            "mathir_family": family.name,
            "level3_difficulty": difficulty,
        })
    rng.shuffle(rows)
    return rows


ATTRIBUTES = ("mass_g", "energy_kcal", "protein_g", "fiber_g", "sodium_mg")
# Curated attributes have at most three decimals. Nutrition totals are exact
# integers in units of 1/100000; no floating-point arithmetic certifies support.
TOTAL_SCALE = 100_000


@lru_cache(maxsize=512)
def _quantity_product(grids: tuple[tuple[int, ...], ...]) -> np.ndarray:
    return np.asarray(list(itertools.product(*grids)), dtype=np.int64)


def _pantry_allocations(spec):
    """Enumerate every legal quantity tuple once, with exact integer totals."""
    parsed = parse_pantry_plan_spec(spec)
    ingredients = parsed.ingredients
    all_totals, all_quantities, all_supports = [], [], []
    for size in range(parsed.min_ingredients, parsed.max_ingredients + 1):
        for indexes in itertools.combinations(range(len(ingredients)), size):
            selected = [ingredients[i] for i in indexes]
            if any(row.tags & parsed.forbidden_tags for row in selected):
                continue
            quantities = _quantity_product(tuple(
                tuple(range(row.min_if_used_g, row.available_g + 1, row.step_g))
                for row in selected
            ))
            attribute_rows = []
            for ingredient in selected:
                attrs = ingredient.attributes
                converted = [Decimal(TOTAL_SCALE)] + [
                    attrs[name] * (TOTAL_SCALE // 100) for name in ATTRIBUTES[1:]
                ]
                if any(x != int(x) for x in converted):
                    raise ValueError("curated ingredient precision exceeds exact total scale")
                attribute_rows.append([int(x) for x in converted])
            totals = quantities @ np.asarray(attribute_rows, dtype=np.int64)
            padded = np.zeros((len(quantities), len(ingredients)), dtype=np.int64)
            padded[:, indexes] = quantities
            all_totals.append(totals)
            all_quantities.append(padded)
            all_supports.append(np.full(len(quantities), sum(1 << i for i in indexes), dtype=np.int64))
    return np.concatenate(all_totals), np.concatenate(all_quantities), np.concatenate(all_supports)


def exact_pantry_supports(spec: dict[str, Any]) -> dict[tuple[str, ...], str]:
    """Exact exhaustive certification, accelerated by integer array arithmetic."""
    totals, quantities, masks = _pantry_allocations(spec)
    admitted = _pantry_admitted(spec, totals)
    accepted = np.flatnonzero(admitted)
    _, first = np.unique(masks[accepted], return_index=True)
    result = {}
    ingredients = spec["ingredients"]
    for index in accepted[first]:
        pairs = sorted((row["id"], int(grams)) for row, grams in zip(ingredients, quantities[index]) if grams)
        key = tuple(name for name, _ in pairs)
        result[key] = ";".join(f"{name}={grams}" for name, grams in pairs)
    return result


def _pantry_admitted(spec, totals):
    admitted = np.ones(len(totals), dtype=bool)
    for attr, bounds in spec["targets"].items():
        column = ATTRIBUTES.index(attr)
        for limit, raw in bounds.items():
            scaled = Decimal(str(raw)) * TOTAL_SCALE
            if scaled != int(scaled):
                raise ValueError("target precision exceeds exact total scale")
            admitted &= totals[:, column] >= int(scaled) if limit == "min" else totals[:, column] <= int(scaled)
    return admitted


def _decimal_units(value):
    # Keep the existing prompt precision; exhaustively recheck rounded bounds.
    rounded = (Decimal(int(value)) / TOTAL_SCALE).quantize(Decimal("0.001"))
    return format(rounded.normalize(), "f")


@lru_cache(maxsize=1)
def _pantry_source():
    curation = json.loads((ROOT / "var/data/pantry_plan_v1/ingredients.json").read_text())
    return {row["id"]: row for row in curation["ingredients"]}, curation["ingredient_table_sha256"]


def _pantry_candidate(family, support_count, seed, tag, index, difficulty, rng):
    source, table_hash = _pantry_source()
    menu_size = (6, 7, 8, 8)[difficulty]
    ids = sorted(rng.sample(FAMILY_POOLS[family], menu_size))
    ingredients = []
    for ingredient_id in ids:
        step = 25 if difficulty == 0 else rng.choice((20, 25)) if difficulty == 1 else rng.choice((10, 15, 20))
        minimum_steps = 2 if difficulty < 2 else rng.choice((2, 3, 4))
        ingredient = source[ingredient_id]
        ingredients.append({
            "id": ingredient_id,
            "available_g": step * (minimum_steps + rng.choice((2, 3, 4))),
            "step_g": step,
            "min_if_used_g": step * minimum_steps,
            "attributes_per_100g": dict(ingredient["attributes_per_100g"]),
            "tags": list(ingredient["tags"]),
        })
    forbidden = []
    possible = sorted({tag for row in ingredients for tag in row["tags"] if tag in {"peanut", "tree_nut"}})
    if possible and rng.random() < (0.25, 0.5, 0.6, 0.75)[difficulty]:
        forbidden = [rng.choice(possible)]
    spec = {
        "verifier": PANTRY_PLAN_VERIFIER,
        "pantry_version": PANTRY_PLAN_VERSION,
        "family": family,
        "instance_id": f"{tag}-{seed}-{family}-{index}",
        "source_ingredient_table_sha256": table_hash,
        "ingredients": ingredients,
        "targets": {"mass_g": {"min": "1"}},
        "min_ingredients": 2,
        "max_ingredients": 4,
        "forbidden_tags": forbidden,
        "certified_mode_count": support_count,
    }
    totals, quantities, masks = _pantry_allocations(spec)
    if len(np.unique(masks)) < support_count:
        return None
    sodium_cap = int(SODIUM_CAPS[family] * TOTAL_SCALE)
    seeds = np.flatnonzero((totals[:, 4] <= sodium_cap) & (totals[:, 0] >= 100 * TOTAL_SCALE))
    if not len(seeds):
        return None
    center = totals[rng.choice(seeds.tolist())]
    # Expand simultaneous mass/nutrition intervals around one feasible center.
    # Each support enters at its best quantity tuple's required width. Selecting
    # a gap between support thresholds targets a count without dropping modes.
    weights = np.asarray((0.45, 0.75, 1.1, 1.1, 1.8))
    if difficulty == 3:
        weights = np.asarray((0.3, 0.5, 0.7, 0.7, 1.4))
    scales = np.maximum(center, np.asarray((100, 100, 1, 1, 25)) * TOTAL_SCALE) * weights
    lower_columns = (0, 1, 2, 3)
    upper_columns = (0, 1, 4) if difficulty < 3 else (0, 1, 2, 3, 4)
    required_width = np.zeros(len(totals))
    for column in lower_columns:
        required_width = np.maximum(required_width, (center[column] - totals[:, column]) / scales[column])
    for column in upper_columns:
        required_width = np.maximum(required_width, (totals[:, column] - center[column]) / scales[column])
    required_width[totals[:, 4] > sodium_cap] = np.inf
    per_support = np.full(1 << len(ingredients), np.inf)
    np.minimum.at(per_support, masks, required_width)
    thresholds = np.sort(per_support[np.isfinite(per_support)])
    if len(thresholds) < support_count:
        return None
    lo = thresholds[support_count - 1]
    hi = thresholds[support_count] if len(thresholds) > support_count else lo + 0.1
    if hi - lo < 1e-6:
        return None
    width = lo + (hi - lo) * rng.uniform(0.25, 0.75)
    targets = {}
    for column, attr in enumerate(ATTRIBUTES):
        bounds = {}
        if column in lower_columns:
            bounds["min"] = _decimal_units(max(0, int(np.ceil(center[column] - scales[column] * width))))
        if column in upper_columns:
            maximum = int(np.floor(center[column] + scales[column] * width))
            if column == 4:
                maximum = min(maximum, sodium_cap)
            bounds["max"] = _decimal_units(maximum)
        targets[attr] = bounds
    spec["targets"] = targets
    admitted = _pantry_admitted(spec, totals)
    accepted = np.flatnonzero(admitted)
    valid_masks, first = np.unique(masks[accepted], return_index=True)
    if len(valid_masks) != support_count:
        return None
    support_keys = []
    for mask, allocation_index in zip(valid_masks, accepted[first]):
        support_keys.append("+".join(row["id"] for i, row in enumerate(ingredients) if int(mask) & (1 << i)))
        # Every claimed support has an independently accepted witness through
        # the unchanged Decimal verifier; negatives are ruled out exhaustively.
        witness = ";".join(f"{row['id']}={int(grams)}" for row, grams in zip(ingredients, quantities[allocation_index]) if grams)
        if validate_pantry_plan(witness, spec) is None:
            raise RuntimeError("integer enumeration disagrees with Pantry verifier")
    spec["certified_support_sha256"] = hashlib.sha256("\n".join(sorted(support_keys)).encode()).hexdigest()
    fingerprint_spec = dict(spec)
    fingerprint_spec.pop("instance_id")
    return {
        "problem": pantry_prompt(family, spec),
        "answer": json.dumps(spec, sort_keys=True, separators=(",", ":")),
        "modebench_task": PANTRY_PLAN_VERIFIER,
        "answer_mode_family": family,
        "answer_mode_count": support_count,
        "answer_mode_split": tag,
        "instance_fingerprint": _canonical_sha256(fingerprint_spec),
        "level3_difficulty": difficulty,
        "level3_feasible_allocation_count": int(admitted.sum()),
        "level3_legal_allocation_count": len(totals),
    }


def _pantry_pool(target, excluded, seed, tag, difficulty, multiplier, joint_target):
    rng = random.Random(seed)
    if joint_target is None:
        joint_target = Counter()
        families = list(FAMILY_POOLS)
        for support, count in sorted(target.items()):
            for index in range(count):
                joint_target[support, families[index % len(families)]] += 1
    joint_support = Counter()
    for (support, family), count in joint_target.items():
        if family not in FAMILY_POOLS:
            raise ValueError(f"unknown Pantry family: {family}")
        joint_support[support] += count
    if joint_support != target:
        raise ValueError("Pantry joint target does not match requested support histogram")
    tasks = [(support, family) for (support, family), count in sorted(joint_target.items()) for _ in range(count * multiplier)]
    rng.shuffle(tasks)
    rows, blocked = [], {x[1] if isinstance(x, tuple) else x for x in excluded}
    for support, family in tasks:
        for attempt in range(1000):
            row = _pantry_candidate(family, support, seed, tag, len(rows), difficulty, rng)
            if row is not None and row["instance_fingerprint"] not in blocked:
                blocked.add(row["instance_fingerprint"])
                rows.append(row)
                break
        else:
            raise RuntimeError(f"could not produce Pantry support={support} family={family} difficulty={difficulty}")
        if len(rows) % 64 == 0:
            print(f"[level3] {tag} pantry difficulty={difficulty} {len(rows)}/{len(tasks)}", flush=True)
    rng.shuffle(rows)
    return rows


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
    """Produce exactly ``target * multiplier`` candidates in each support cell.

    Pantry's optional ``joint_target`` maps ``(support_count, family)`` to row
    counts, preserving both marginals and their joint distribution exactly.
    Callers must exclude prior candidate pools as well as frozen dataset IDs.
    """
    if difficulty not in range(4) or multiplier < 1:
        raise ValueError("difficulty must be 0..3 and multiplier must be positive")
    target = Counter({int(k): int(v) for k, v in target.items() if v})
    if not target or any(k < 2 or v < 0 for k, v in target.items()):
        raise ValueError("target must contain positive counts of multimode supports")
    if domain == "mathir":
        return _mathir_pool(target, excluded, seed, tag, difficulty, multiplier)
    if domain == "pantry":
        return _pantry_pool(target, excluded, seed, tag, difficulty, multiplier, joint_target)
    raise ValueError(f"unsupported constraint domain: {domain}")
