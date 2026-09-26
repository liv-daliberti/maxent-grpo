"""Pantry r4 candidate laws: the r3 ladder re-cut to spend every rung in the live region.

The r2 provider hardcoded difficulty = 0, which pinned step_g to 25 and left
menu_size, the quantity-step granularity, the minimum-usage steps, the
forbidden-tag probability and the tighter interval-width law unreachable. Only
available_g varied with tier, so the measured ladder was monotone but shallow:
its hardest tier reached pass1 0.148 / pass8 0.336 against a target of
0.059 / 0.260, and no mixture could reach the target.

This provider grades difficulty with four knobs that move together, and keeps
availability aligned to the step by construction rather than by a fixed table:

    available_g = step_g * (minimum_steps + headroom)

Tier 0 reproduces the r2 tier-3 configuration exactly, so it is a calibration
control with a known measured point. Tiers 1..3 extend beyond it.

Canonical modes still count ingredient supports, never quantity variants. The
sodium-cap, prompt and verifier laws are unchanged. No difficulty match and no
monotonic ordering are claimed here; the ordering is a hypothesis for a pilot
to measure.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
import modebench_scale_candidates as original
from modebench_level3_constraints import (
    ATTRIBUTES, TOTAL_SCALE, FAMILY_POOLS, SODIUM_CAPS,
    PANTRY_PLAN_VERIFIER, PANTRY_PLAN_VERSION, validate_pantry_plan,
    _pantry_source, _pantry_allocations, _pantry_admitted, _decimal_units,
    _canonical_sha256, pantry_prompt,
)

SCHEMA = 'modebench_scale_l5_pantry_r4_candidate_laws_v1'
DOMAINS = ('pantry',)
MAX_PROPOSALS_PER_ROW = 1000
STEP_G = 25
MINIMUM_STEPS = 2
# breakfast_formulation holds only 8 ingredients, so 8 is a hard ceiling on menu_size.
# The r2 law's (6, 7, 8, 8) was capped for this reason, not arbitrarily.
# Re-cut from r3 after its pilot. r3 moved all four knobs together between tiers 1
# and 2, and pass@1 fell from 0.0742 to 0.0048 across that single rung with the
# 0.0586 target inside the gap; tiers 2 and 3 were dead at dead fractions 0.90 and
# 0.92 and inverted against each other. Tiers 0 and 1 are kept at exactly their r3
# settings, whose measured points are 0.1896/0.3898 and 0.0742/0.2161, and the two
# dead rungs are replaced by milder steps that hold the tight-interval law off and
# the forbidden-tag probability fixed, so only headroom and menu size move below
# tier 1. No ordering is claimed; the pilot measures it.
MENU_SIZES = (6, 7, 7, 8)
# available_g = STEP_G * (MINIMUM_STEPS + headroom); more headroom means more legal
# quantities per ingredient and a larger search. Tier 0 reproduces r2 tier 3.
HEADROOM = ((2, 3, 4), (4, 5, 6), (6, 7, 8), (6, 7, 8))
FORBIDDEN_PROBABILITY = (0.25, 0.45, 0.45, 0.45)
TIGHT_INTERVALS = (False, False, False, False)
AVAILABLE_GRAMS = tuple(tuple(STEP_G * (MINIMUM_STEPS + slack) for slack in tier)
                        for tier in HEADROOM)
ORIGIN_CONSTRAINTS_SHA256 = '331cd3d8eb0b3651153c66af3c8dc07c52872f3adfd664963056a7b8cea48225'
PROFILES = {'pantry': [
    {'menu_size': MENU_SIZES[tier], 'minimum_if_used_g': STEP_G * MINIMUM_STEPS,
     'quantity_step_g': STEP_G,
     'available_g_choices': list(available), 'min_ingredients': 2, 'max_ingredients': 4,
     'forbidden_tag_proposal_probability': FORBIDDEN_PROBABILITY[tier],
     'interval_width_law': 'tight' if TIGHT_INTERVALS[tier] else 'original_tier0',
     'forbidden_tag_candidates': ['peanut', 'tree_nut'],
     'family_sodium_caps_mg': {family: str(cap) for family, cap in SODIUM_CAPS.items()},
     'constraint_width_law': 'unchanged_original_tier0_simultaneous_intervals',
     'canonical_mode': 'ingredient_support',
     'sampling': 'independent_support_family_streams_conditioned_on_exact_support'}
    for tier, available in enumerate(AVAILABLE_GRAMS)
]}
identity = original.identity


def source_paths():
    return sorted({Path(__file__).resolve(), *original.source_paths(),
                   Path(sys.modules['oat_drgrpo.pantry_plan'].__file__).resolve(),
                   ROOT / 'var/data/pantry_plan_v1/ingredients.json'})


def _seed(*parts):
    payload = json.dumps([SCHEMA, *parts], sort_keys=True, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest(), 'big')


def _annotate(row, tier, seed, index):
    row = dict(row)
    inherited = {key: row.pop(key) for key in list(row) if key.startswith('level3_')}
    row.update(scale_candidate_generator=SCHEMA, scale_candidate_tier=tier,
               scale_candidate_profile=json.dumps(PROFILES['pantry'][tier], sort_keys=True, separators=(',', ':')),
               scale_origin_metadata=json.dumps(inherited, sort_keys=True, separators=(',', ':')),
               scale_cell_index=index, scale_generation_seed=seed,
               scale_bridge_law_version=SCHEMA)
    return row


def _candidate(family, support_count, seed, tag, index, tier, rng):
    difficulty = tier
    source, table_hash = _pantry_source()
    menu_size = MENU_SIZES[tier]
    ids = sorted(rng.sample(FAMILY_POOLS[family], menu_size))
    ingredients = []
    for ingredient_id in ids:
        step = STEP_G
        minimum_steps = MINIMUM_STEPS
        ingredient = source[ingredient_id]
        ingredients.append({
            "id": ingredient_id,
            "available_g": step * (minimum_steps + rng.choice(HEADROOM[tier])),
            "step_g": step,
            "min_if_used_g": step * minimum_steps,
            "attributes_per_100g": dict(ingredient["attributes_per_100g"]),
            "tags": list(ingredient["tags"]),
        })
    forbidden = []
    possible = sorted({tag for row in ingredients for tag in row["tags"] if tag in {"peanut", "tree_nut"}})
    if possible and rng.random() < FORBIDDEN_PROBABILITY[tier]:
        forbidden = [rng.choice(possible)]
    spec = {
        "verifier": PANTRY_PLAN_VERIFIER,
        "pantry_version": PANTRY_PLAN_VERSION,
        "family": family,
        "instance_id": f"{tag}-{seed}-{family}-t{tier}-m{support_count}-{index}",
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
    if TIGHT_INTERVALS[tier]:
        weights = np.asarray((0.3, 0.5, 0.7, 0.7, 1.4))
    scales = np.maximum(center, np.asarray((100, 100, 1, 1, 25)) * TOTAL_SCALE) * weights
    lower_columns = (0, 1, 2, 3)
    upper_columns = (0, 1, 2, 3, 4) if TIGHT_INTERVALS[tier] else (0, 1, 4)
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


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    """Produce exact joint family/support quotas from independent fresh streams."""
    if domain != 'pantry' or type(tier) is not int or tier not in range(4):
        raise ValueError('Pantry domain and integer tier0..3 required')
    if type(seed) is not int or seed < 0 or type(multiplier) is not int or multiplier < 1:
        raise ValueError('nonnegative integer seed and positive integer multiplier required')
    if not isinstance(tag, str) or not tag:
        raise ValueError('nonempty split tag required')
    if not isinstance(target, Mapping) or any(type(support) is not int or support not in range(8, 46)
            or type(count) is not int or count < 0 for support, count in target.items()):
        raise ValueError('registered Pantry supports8..45 and nonnegative integer quotas required')
    if not isinstance(joint_target, Mapping):
        raise ValueError('explicit exact Pantry family/support joint_target required')
    marginal = Counter()
    for cell, count in joint_target.items():
        if (not isinstance(cell, tuple) or len(cell) != 2 or type(cell[0]) is not int
                or cell[0] not in range(8, 46) or cell[1] not in FAMILY_POOLS
                or type(count) is not int or count < 0):
            raise ValueError('invalid exact Pantry family/support joint_target')
        marginal[cell[0]] += count
    if +marginal != +Counter(target):
        raise ValueError('Pantry joint_target marginal differs from requested support quotas')
    required = Counter({cell: count * multiplier for cell, count in joint_target.items() if count})
    blocked, rows = set(excluded), []
    for (support, family), count in sorted(required.items()):
        rng = random.Random(_seed(seed, tier, support, family, 'pantry'))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                row = _candidate(family, support, seed, tag, index, tier, rng)
                if row is None or identity(domain, row) in blocked:
                    continue
                blocked.add(identity(domain, row))
                rows.append(_annotate(row, tier, seed, index))
                break
            else:
                raise RuntimeError(f'Pantry bridge exhausted fixed proposal budget: tier{tier}/cell{(support, family)}')
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    ids = {identity(domain, row) for row in rows}
    if len(ids) != len(rows) or ids & set(excluded):
        raise RuntimeError('Pantry bridge identities overlap')
    if Counter((row['answer_mode_count'], row['answer_mode_family']) for row in rows) != required:
        raise RuntimeError('Pantry bridge exact joint quota drift')
    return rows


def verify_rows(domain, rows):
    """Validate the structural law, metadata and every original executable witness."""
    if domain != 'pantry':
        raise ValueError(domain)
    source, table_hash = _pantry_source()
    for row in rows:
        tier = row.get('scale_candidate_tier')
        if type(tier) is not int or tier not in range(4):
            raise RuntimeError('Pantry bridge profile audit failed: invalid tier')
        profile = json.dumps(PROFILES[domain][tier], sort_keys=True, separators=(',', ':'))
        if (row.get('scale_candidate_generator') != SCHEMA or row.get('scale_bridge_law_version') != SCHEMA
                or row.get('scale_candidate_profile') != profile):
            raise RuntimeError('Pantry bridge profile audit failed: metadata changed')
        spec = json.loads(row['answer'])
        family = row.get('answer_mode_family')
        ingredients = spec['ingredients']
        ids = [item['id'] for item in ingredients]
        good = (family in FAMILY_POOLS and spec.get('family') == family
                and len(ids) == len(set(ids)) == MENU_SIZES[tier]
                and ids == sorted(ids) and set(ids) <= set(FAMILY_POOLS[family])
                and spec.get('verifier') == PANTRY_PLAN_VERIFIER and spec.get('pantry_version') == PANTRY_PLAN_VERSION
                and spec.get('min_ingredients') == 2 and spec.get('max_ingredients') == 4
                and spec.get('source_ingredient_table_sha256') == table_hash
                and row.get('modebench_task') == PANTRY_PLAN_VERIFIER
                and type(row.get('answer_mode_count')) is int and row['answer_mode_count'] in range(8, 46)
                and spec.get('certified_mode_count') == row['answer_mode_count'])
        if not good:
            raise RuntimeError('Pantry bridge structural law audit failed')
        for item in ingredients:
            expected = source[item['id']]
            good = good and item['step_g'] == STEP_G and item['min_if_used_g'] == STEP_G * MINIMUM_STEPS
            good = good and item['available_g'] in AVAILABLE_GRAMS[tier]
            good = good and item['attributes_per_100g'] == expected['attributes_per_100g'] and item['tags'] == expected['tags']
        possible = {tag for item in ingredients for tag in item['tags'] if tag in {'peanut', 'tree_nut'}}
        forbidden = spec['forbidden_tags']
        good = good and isinstance(forbidden, list) and len(forbidden) <= 1 and set(forbidden) <= possible
        # The tight interval law bounds every attribute from above, so the audited
        # bound shape follows the tier exactly as the generator's upper_columns does.
        bounds = {'mass_g': {'min', 'max'}, 'energy_kcal': {'min', 'max'},
                  'protein_g': {'min'}, 'fiber_g': {'min'}, 'sodium_mg': {'max'}}
        if TIGHT_INTERVALS[tier]:
            bounds['protein_g'] = {'min', 'max'}
            bounds['fiber_g'] = {'min', 'max'}
        good = good and set(spec['targets']) == set(bounds) and all(
            set(spec['targets'][attribute]) == keys for attribute, keys in bounds.items())
        good = good and Decimal(spec['targets']['sodium_mg']['max']) <= SODIUM_CAPS[family]
        good = good and type(row.get('scale_generation_seed')) is int and row['scale_generation_seed'] >= 0
        good = good and type(row.get('scale_cell_index')) is int and row['scale_cell_index'] >= 0
        expected_id = (f"{row['answer_mode_split']}-{row['scale_generation_seed']}-{family}"
                       f"-t{tier}-m{row['answer_mode_count']}-{row['scale_cell_index']}")
        good = good and spec.get('instance_id') == expected_id
        if not good:
            raise RuntimeError('Pantry bridge structural law or row metadata audit failed')
        totals, _, masks = _pantry_allocations(spec)
        admitted = _pantry_admitted(spec, totals)
        valid_masks = np.unique(masks[admitted])
        support_keys = sorted('+'.join(item['id'] for i, item in enumerate(ingredients)
                                      if int(mask) & (1 << i)) for mask in valid_masks)
        digest = hashlib.sha256('\n'.join(support_keys).encode()).hexdigest()
        expected_origin = {'level3_difficulty': tier, 'level3_feasible_allocation_count': int(admitted.sum()),
                           'level3_legal_allocation_count': len(totals)}
        if (len(valid_masks) != row['answer_mode_count'] or spec.get('certified_support_sha256') != digest
                or json.loads(row['scale_origin_metadata']) != expected_origin):
            raise RuntimeError('Pantry bridge exact-support certificate or allocation metadata audit failed')
    return {**original.verify_rows(domain, rows), 'pantry_bridge_structural_profile': True,
            'original_family_ingredient_table': True, 'canonical_modes_are_ingredient_supports': True}
