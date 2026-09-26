"""Prospective Pantry r3 menu laws; no measured difficulty or registration claim.

The four fixed laws are the exact native-qualified seven/eight-ingredient and
fixed100/mixed100,125,150g constructors. All other original tier0 rules remain
unchanged. Every scratch qualification/reference identity is always excluded;
a later native registration must additionally snapshot all current history.
The finite scratch witness does not guarantee remaining fresh capacity after
its own fixtures are excluded; future generation must satisfy every quota.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
import modebench_scale_pantry_bridge_candidates as bridge
import modebench_scale_candidates as original
from modebench_level3_constraints import (
    ATTRIBUTES, TOTAL_SCALE, FAMILY_POOLS, SODIUM_CAPS,
    PANTRY_PLAN_VERIFIER, PANTRY_PLAN_VERSION, validate_pantry_plan,
    _pantry_source, _pantry_allocations, _pantry_admitted, _decimal_units,
    _canonical_sha256, pantry_prompt,
)

SCHEMA = 'modebench_scale_pantry_menu_revision3_candidate_laws_v1'
DOMAINS = ('pantry',)
MAX_PROPOSALS_PER_ROW = 1000
BASE_SOURCE = ROOT / 'ops/exp_scaling/modebench_scale_pantry_bridge_candidates.py'
BASE_SHA256 = '73b4b4f6ee9b521ea6e50330c3337fc861c139a719f47bc183d12d50a368ef52'
QUALIFICATION_ROOT = ROOT / 'artifacts/modebench_scale_pantry_menu_feasibility_20260912'
QUALIFICATION_PATH = QUALIFICATION_ROOT / 'manifest.json'
QUALIFICATION_SHA256 = 'd83ccb0ddb4662fbafd4a9d6495b68fb4c5a1fb5ed1445e87c758f3f88859ff2'
EXCLUSIONS_PATH = QUALIFICATION_ROOT / 'scratch_exclusions.json'
EXCLUSIONS_SHA256 = '299816c273d789704df5042875931712e163257dd79f54285ad743eaa2656f5f'
_MENU_SIZES = (7, 7, 8, 8)
_AVAILABILITY = ((100,), (100, 125, 150), (100,), (100, 125, 150))
PROFILES = {'pantry': [
    {'menu_size': menu, 'minimum_if_used_g': 50, 'quantity_step_g': 25,
     'available_g_choices': list(available), 'min_ingredients': 2, 'max_ingredients': 4,
     'forbidden_tag_proposal_probability': .25,
     'forbidden_tag_candidates': ['peanut', 'tree_nut'],
     'family_sodium_caps_mg': {family: str(cap) for family, cap in SODIUM_CAPS.items()},
     'constraint_width_law': 'unchanged_original_tier0_simultaneous_intervals',
     'canonical_mode': 'ingredient_support',
     'sampling': 'independent_support_family_streams_conditioned_on_exact_support',
     'menu_sampling': 'uniform_subset_then_sorted_ids',
     'availability_sampling': 'fixed_or_independent_uniform_three_choices',
     'difficulty_ordering': 'prospective_no_monotonicity_claim'}
    for menu, available in zip(_MENU_SIZES, _AVAILABILITY)
]}
identity = original.identity


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _qualification():
    if _file_sha(BASE_SOURCE) != BASE_SHA256 or _file_sha(QUALIFICATION_PATH) != QUALIFICATION_SHA256:
        raise ValueError('pinned native constructor or scratch qualification changed')
    manifest = json.loads(QUALIFICATION_PATH.read_text())
    if (manifest.get('schema') != 'modebench_scale_pantry_menu_feasibility_manifest_v1'
            or manifest.get('status') != 'native_scratch_capacity_qualified_no_production_adoption'
            or manifest.get('full_capacity_rows') != 3488
            or manifest.get('scratch_exclusions_path') != str(EXCLUSIONS_PATH)
            or manifest.get('scratch_exclusions_sha256') != EXCLUSIONS_SHA256):
        raise ValueError('wrong scratch capacity qualification')
    pins = manifest.get('files_sha256')
    if not isinstance(pins, dict) or not pins or pins.get(str(EXCLUSIONS_PATH)) != EXCLUSIONS_SHA256:
        raise ValueError('missing exact scratch exclusion closure')
    for path, expected in pins.items():
        if not Path(path).resolve().is_relative_to(ROOT) or _file_sha(path) != expected:
            raise ValueError('scratch qualification input changed: ' + path)
    if _file_sha(EXCLUSIONS_PATH) != EXCLUSIONS_SHA256:
        raise ValueError('scratch exclusions changed')
    value = json.loads(EXCLUSIONS_PATH.read_text())
    raw = value.get('identities', [])
    prompts = value.get('prompt_sha256', [])
    if (value.get('schema') != 'modebench_scale_pantry_menu_scratch_exclusions_v1'
            or value.get('status') != 'all_materialized_qualification_and_reference_identities'
            or value.get('production_reuse_allowed') is not False
            or value.get('identity_count') != 3945 or value.get('prompt_count') != 3945
            or len(raw) != 3945 or len(prompts) != 3945
            or any(not isinstance(x, list) or len(x) != 2 or x[0] != 'pantry'
                or not isinstance(x[1], str) or re.fullmatch('[0-9a-f]{64}', x[1]) is None for x in raw)
            or any(not isinstance(x, str) or re.fullmatch('[0-9a-f]{64}', x) is None for x in prompts)):
        raise ValueError('incomplete scratch identity or prompt exclusion set')
    ids, prompt_ids = {tuple(x) for x in raw}, set(prompts)
    if len(ids) != 3945 or len(prompt_ids) != 3945:
        raise ValueError('duplicate scratch exclusions')
    return ids, prompt_ids, pins


def source_paths():
    _, _, pins = _qualification()
    return sorted({Path(__file__).resolve(), BASE_SOURCE, QUALIFICATION_PATH, EXCLUSIONS_PATH,
                   *original.source_paths(), *(Path(p) for p in pins),
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
               scale_pantry_menu_law_version=SCHEMA,
               scale_scratch_qualification_sha256=QUALIFICATION_SHA256,
               scale_scratch_exclusions_sha256=EXCLUSIONS_SHA256)
    return row


def _candidate(family, support_count, seed, tag, index, tier, rng):
    difficulty = 0
    source, table_hash = _pantry_source()
    menu_size = _MENU_SIZES[tier]
    ids = sorted(rng.sample(FAMILY_POOLS[family], menu_size))
    ingredients = []
    for ingredient_id in ids:
        step = 25 if difficulty == 0 else rng.choice((20, 25)) if difficulty == 1 else rng.choice((10, 15, 20))
        minimum_steps = 2 if difficulty < 2 else rng.choice((2, 3, 4))
        ingredient = source[ingredient_id]
        ingredients.append({
            "id": ingredient_id,
            "available_g": _AVAILABILITY[tier][0] if len(_AVAILABILITY[tier]) == 1 else step * (minimum_steps + rng.choice((2, 3, 4))),
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


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    """Exact joint quotas, prospective streams, fixed whole-proposal rejection."""
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
    scratch_ids, scratch_prompts, _ = _qualification()
    required = Counter({cell: count * multiplier for cell, count in joint_target.items() if count})
    blocked, texts, rows = set(excluded) | scratch_ids, set(scratch_prompts), []
    for (support, family), count in sorted(required.items()):
        rng = random.Random(_seed(seed, tier, support, family, 'pantry'))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                row = _candidate(family, support, seed, tag, index, tier, rng)
                if row is None:
                    continue
                ident, prompt = identity(domain, row), _canonical_sha256(row['problem'])
                if ident in blocked or prompt in texts:
                    continue
                blocked.add(ident); texts.add(prompt)
                rows.append(_annotate(row, tier, seed, index))
                break
            else:
                raise RuntimeError(f'Pantry menu r3 exhausted fixed proposal budget: tier{tier}/cell{(support, family)}')
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    ids = {identity(domain, row) for row in rows}
    if len(ids) != len(rows) or ids & (set(excluded) | scratch_ids):
        raise RuntimeError('Pantry menu r3 identities overlap')
    if Counter((row['answer_mode_count'], row['answer_mode_family']) for row in rows) != required:
        raise RuntimeError('Pantry menu r3 exact joint quota drift')
    return rows


def verify_rows(domain, rows):
    """Preserve every native bridge read-back gate plus the exact menu law and scratch exclusions."""
    if domain != 'pantry':
        raise ValueError(domain)
    scratch_ids, scratch_prompts, _ = _qualification()
    seen_ids, seen_prompts = set(), set()
    source, table_hash = _pantry_source()
    for row in rows:
        tier = row.get('scale_candidate_tier')
        if type(tier) is not int or tier not in range(4):
            raise ValueError('Pantry menu r3 profile audit failed: invalid tier')
        profile = json.dumps(PROFILES[domain][tier], sort_keys=True, separators=(',', ':'))
        if (row.get('scale_candidate_generator') != SCHEMA or row.get('scale_pantry_menu_law_version') != SCHEMA
                or row.get('scale_scratch_qualification_sha256') != QUALIFICATION_SHA256
                or row.get('scale_scratch_exclusions_sha256') != EXCLUSIONS_SHA256
                or row.get('scale_candidate_profile') != profile):
            raise ValueError('Pantry menu r3 profile audit failed: metadata changed')
        spec = json.loads(row['answer'])
        family = row.get('answer_mode_family')
        ingredients = spec['ingredients']
        ids = [item['id'] for item in ingredients]
        good = (family in FAMILY_POOLS and spec.get('family') == family and len(ids) == len(set(ids)) == _MENU_SIZES[tier]
                and ids == sorted(ids) and set(ids) <= set(FAMILY_POOLS[family])
                and spec.get('verifier') == PANTRY_PLAN_VERIFIER and spec.get('pantry_version') == PANTRY_PLAN_VERSION
                and spec.get('min_ingredients') == 2 and spec.get('max_ingredients') == 4
                and spec.get('source_ingredient_table_sha256') == table_hash
                and row.get('modebench_task') == PANTRY_PLAN_VERIFIER
                and type(row.get('answer_mode_count')) is int and row['answer_mode_count'] in range(8, 46)
                and spec.get('certified_mode_count') == row['answer_mode_count'])
        if not good:
            raise ValueError('Pantry menu r3 structural law audit failed')
        for item in ingredients:
            expected = source[item['id']]
            good = good and item['step_g'] == 25 and item['min_if_used_g'] == 50 and item['available_g'] in _AVAILABILITY[tier]
            good = good and item['attributes_per_100g'] == expected['attributes_per_100g'] and item['tags'] == expected['tags']
        possible = {tag for item in ingredients for tag in item['tags'] if tag in {'peanut', 'tree_nut'}}
        forbidden = spec['forbidden_tags']
        good = good and isinstance(forbidden, list) and len(forbidden) <= 1 and set(forbidden) <= possible
        bounds = {'mass_g': {'min', 'max'}, 'energy_kcal': {'min', 'max'},
                  'protein_g': {'min'}, 'fiber_g': {'min'}, 'sodium_mg': {'max'}}
        good = good and set(spec['targets']) == set(bounds) and all(
            set(spec['targets'][attribute]) == keys for attribute, keys in bounds.items())
        good = good and Decimal(spec['targets']['sodium_mg']['max']) <= SODIUM_CAPS[family]
        good = good and type(row.get('scale_generation_seed')) is int and row['scale_generation_seed'] >= 0
        good = good and type(row.get('scale_cell_index')) is int and row['scale_cell_index'] >= 0
        expected_id = (f"{row['answer_mode_split']}-{row['scale_generation_seed']}-{family}"
                       f"-t{tier}-m{row['answer_mode_count']}-{row['scale_cell_index']}")
        good = good and spec.get('instance_id') == expected_id
        if not good:
            raise ValueError('Pantry menu r3 structural law or row metadata audit failed')
        ident, prompt = identity(domain, row), _canonical_sha256(row['problem'])
        if ident in scratch_ids or prompt in scratch_prompts or ident in seen_ids or prompt in seen_prompts:
            raise ValueError('Pantry menu r3 reuses scratch or duplicate input')
        seen_ids.add(ident); seen_prompts.add(prompt)
        totals, _, masks = _pantry_allocations(spec)
        admitted = _pantry_admitted(spec, totals)
        valid_masks = np.unique(masks[admitted])
        support_keys = sorted('+'.join(item['id'] for i, item in enumerate(ingredients)
                                      if int(mask) & (1 << i)) for mask in valid_masks)
        digest = hashlib.sha256('\n'.join(support_keys).encode()).hexdigest()
        expected_origin = {'level3_difficulty': 0, 'level3_feasible_allocation_count': int(admitted.sum()),
                           'level3_legal_allocation_count': len(totals)}
        if (len(valid_masks) != row['answer_mode_count'] or spec.get('certified_support_sha256') != digest
                or json.loads(row['scale_origin_metadata']) != expected_origin):
            raise ValueError('Pantry menu r3 exact-support certificate or allocation metadata audit failed')
    return {**original.verify_rows(domain, rows), 'pantry_menu_revision3_profile': True, 'complete_scratch_exclusions': True,
            'original_family_ingredient_table': True, 'canonical_modes_are_ingredient_supports': True}
