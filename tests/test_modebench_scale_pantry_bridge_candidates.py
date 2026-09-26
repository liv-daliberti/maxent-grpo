"""Structural-only tests: no model inference or production held-out rows."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_scale_pantry_bridge_candidates as bridge
import modebench_level3_constraints as constraints


# Frozen histogram metadata only, including training-only high-support cells.
# This fixture remains useful when local production artifacts are unavailable.
UNION_BY_FAMILY = {
    'breakfast_formulation': {8:2,9:3,10:4,11:3,12:3,13:6,14:6,15:4,16:3,17:2,18:4,19:2,20:5,21:2,23:1,24:1,40:1,42:1,45:1},
    'high_fiber_snack': {8:2,9:6,10:6,11:5,12:4,13:3,14:2,15:1,16:1,17:2,18:3,19:2,20:2,21:1,22:1,23:1,24:1,28:2,29:1,30:1,32:1,36:1,39:1},
    'low_sodium_pantry_meal': {8:2,9:2,10:3,11:1,12:1,13:2,14:2,15:2,16:1,17:2,18:1,19:2,20:4,21:3,22:4,23:2,24:3,25:2,26:4,27:1,28:1,29:2,30:1,31:1,32:2,33:1,34:1,35:2,36:2,37:2,38:1,39:1,40:1,44:1},
    'plant_protein_bowl': {8:1,9:1,10:1,11:1,12:1,13:1,14:2,15:2,16:4,17:4,18:2,19:2,20:2,21:3,22:4,23:2,24:4,25:3,26:3,27:1,28:1,29:3,30:1,31:2,32:4,33:1,34:1,36:1,37:2,38:1,39:2,40:1,42:1},
}
JOINT_UNION = Counter({(support, family): count for family, cells in UNION_BY_FAMILY.items()
                       for support, count in cells.items()})


def marginal(joint):
    result = Counter()
    for (support, _), count in joint.items():
        result[support] += count
    return result


def pool(joint, tier=0, seed=962912001, excluded=(), tag='STRUCTURAL_TEST_ONLY', **kwargs):
    return bridge.build_pool('pantry', marginal(joint), set(excluded), seed, tag, tier,
                             joint_target=joint, **kwargs)


@pytest.fixture(scope='module')
def complete_pools():
    result, excluded, prompts = {}, set(), set()
    for tier in range(4):
        rows = pool(JOINT_UNION, tier=tier, seed=962912100 + tier, excluded=excluded)
        ids = {bridge.identity('pantry', row) for row in rows}
        row_prompts = {row['problem'] for row in rows}
        assert len(ids) == len(row_prompts) == len(rows)
        assert not ids & excluded and not row_prompts & prompts
        excluded |= ids
        prompts |= row_prompts
        result[tier] = rows
    return result


def test_portable_fixture_matches_registered_histogram_when_present():
    assert len(JOINT_UNION) == 109 and sum(JOINT_UNION.values()) == 232
    path = ROOT / 'var/data/modebench_scale_v1/protocol.json'
    if path.is_file():
        from materialize_modebench_scale import deserialize_cells, union_histogram
        protocol = json.loads(path.read_text())
        actual = union_histogram({split: deserialize_cells(hist) for split, hist
                                  in protocol['histograms']['pantry'].items()})
        assert JOINT_UNION == actual


@pytest.mark.parametrize('tier', range(4))
def test_all_109_cells_exact_full_support_and_original_witnesses(complete_pools, tier):
    rows = complete_pools[tier]
    assert Counter((row['answer_mode_count'], row['answer_mode_family']) for row in rows) == JOINT_UNION
    assert len({json.loads(row['answer'])['instance_id'] for row in rows}) == len(rows)
    audit = bridge.verify_rows('pantry', rows)
    assert audit['rows_verified'] == 232
    assert audit['witnesses_verified'] == sum(support * count for (support, _), count in JOINT_UNION.items())
    assert audit['exact_canonical_support'] and audit['unchanged_original_prompt']
    assert audit['pantry_bridge_structural_profile'] and audit['original_family_ingredient_table']
    assert audit['canonical_modes_are_ingredient_supports']
    for row in rows:
        spec = json.loads(row['answer'])
        assert len(spec['ingredients']) == 6
        for item in spec['ingredients']:
            legal = list(range(item['min_if_used_g'], item['available_g'] + 1, item['step_g']))
            if tier < 3:
                assert legal == ([50], [50, 75], [50, 75, 100])[tier]
            else:
                assert legal in ([50,75,100], [50,75,100,125], [50,75,100,125,150])
        if row['answer_mode_count'] > 25:
            assert all(not set(item['tags']) & set(spec['forbidden_tags']) for item in spec['ingredients'])


@pytest.mark.parametrize('tier', range(4))
def test_deterministic_independent_cells_prefix_and_semantic_exclusion(tier):
    family, other = 'breakfast_formulation', 'low_sodium_pantry_meal'
    one = pool({(45, family): 1}, tier=tier)
    two = pool({(45, family): 2}, tier=tier)
    assert one == pool({(45, family): 1}, tier=tier)
    assert one == sorted(two, key=lambda row: row['scale_cell_index'])[:1]
    wider = pool({(45, family): 2, (44, other): 1}, tier=tier)
    assert [row for row in wider if row['answer_mode_family'] == family] == two
    blocked = {bridge.identity('pantry', row) for row in two}
    fresh = pool({(45, family): 2}, tier=tier, excluded=blocked, tag='INDEPENDENT_SPLIT')
    assert not blocked & {bridge.identity('pantry', row) for row in fresh}
    assert not {row['problem'] for row in two} & {row['problem'] for row in fresh}
    bridge.verify_rows('pantry', fresh)


def test_multiplier_preserves_per_cell_proposal_law():
    joint = {(40, 'breakfast_formulation'): 1, (39, 'high_fiber_snack'): 1}
    assert pool(joint, multiplier=2) == pool({cell: count * 2 for cell, count in joint.items()})


def test_anchor_is_exact_original_tier0_proposal_law():
    accepted = rejected = 0
    for family, support in [('breakfast_formulation', 45), ('plant_protein_bowl', 12),
                            ('high_fiber_snack', 39), ('low_sodium_pantry_meal', 44)]:
        for seed in range(962912300, 962912310):
            args = (family, support, seed, 'ANCHOR_LAW_TEST', 0)
            new = bridge._candidate(*args, 3, random.Random(seed))
            old = constraints._pantry_candidate(*args, 0, random.Random(seed))
            if old is not None and new is not None:
                # Only the nonsemantic cell ID changes; the anchor law is exact.
                old_spec = json.loads(old['answer'])
                old_spec['instance_id'] = json.loads(new['answer'])['instance_id']
                old['answer'] = json.dumps(old_spec, sort_keys=True, separators=(',', ':'))
            assert new == old
            accepted += new is not None
            rejected += new is None
    assert accepted and rejected


def reseal_native_row(row, spec):
    """Make downstream edits self-consistent so tests isolate structural checks."""
    spec['targets'] = {attribute: spec['targets'][attribute] for attribute in constraints.ATTRIBUTES}
    row['answer'] = json.dumps(spec, sort_keys=True, separators=(',', ':'))
    row['problem'] = constraints.pantry_prompt(spec['family'], spec)
    fingerprint = dict(spec)
    fingerprint.pop('instance_id')
    row['instance_fingerprint'] = constraints._canonical_sha256(fingerprint)


@pytest.mark.parametrize('mutation', ['quantities', 'nutrition', 'tags', 'menu', 'sodium_cap', 'target_shape'])
def test_self_consistent_native_row_cannot_escape_registered_structural_law(complete_pools, mutation):
    row = copy.deepcopy(complete_pools[0][0])
    spec = json.loads(row['answer'])
    if mutation == 'quantities':
        spec['ingredients'][0]['available_g'] = 75
    elif mutation == 'nutrition':
        spec['ingredients'][0]['attributes_per_100g']['protein_g'] = '999'
    elif mutation == 'tags':
        spec['ingredients'][0]['tags'] += ['unregistered_tag']
    elif mutation == 'menu':
        spec['ingredients'] = spec['ingredients'][:-1]
    elif mutation == 'sodium_cap':
        spec['targets']['sodium_mg']['max'] = str(bridge.SODIUM_CAPS[spec['family']] + 1)
    else:
        spec['targets']['fiber_g']['max'] = '9999'
    reseal_native_row(row, spec)
    with pytest.raises(RuntimeError, match='structural law'):
        bridge.verify_rows('pantry', [row])


def test_exact_support_digest_is_verified_even_when_native_fingerprint_is_resealed(complete_pools):
    row = copy.deepcopy(complete_pools[0][0])
    spec = json.loads(row['answer'])
    spec['certified_support_sha256'] = '0' * 64
    reseal_native_row(row, spec)
    assert bridge.original.verify_rows('pantry', [row])['rows_verified'] == 1
    with pytest.raises(RuntimeError, match='exact-support certificate'):
        bridge.verify_rows('pantry', [row])


def test_instance_id_binds_support_and_tier_even_with_the_same_generation_seed():
    rows = [row for tier in range(4) for row in pool(
        {(8, 'breakfast_formulation'): 2, (9, 'breakfast_formulation'): 2}, tier=tier)]
    assert len({json.loads(row['answer'])['instance_id'] for row in rows}) == len(rows)
    bridge.verify_rows('pantry', rows)
    row = copy.deepcopy(rows[0])
    spec = json.loads(row['answer'])
    spec['instance_id'] = json.loads(rows[1]['answer'])['instance_id']
    reseal_native_row(row, spec)
    with pytest.raises(RuntimeError, match='row metadata audit'):
        bridge.verify_rows('pantry', [row])


@pytest.mark.parametrize('field,value', [('problem', 'changed'), ('instance_fingerprint', '0' * 64),
    ('scale_candidate_profile', '{}'), ('scale_bridge_law_version', 'wrong'),
    ('scale_candidate_generator', 'wrong'), ('scale_candidate_tier', True),
    ('scale_generation_seed', 1), ('scale_cell_index', -1), ('answer_mode_count', 46)])
def test_prompt_identity_profile_and_generation_metadata_tampering_rejected(complete_pools, field, value):
    row = copy.deepcopy(complete_pools[0][0])
    row[field] = value
    with pytest.raises(RuntimeError, match='audit failed'):
        bridge.verify_rows('pantry', [row])


@pytest.mark.parametrize('key', ['level3_difficulty', 'level3_feasible_allocation_count', 'level3_legal_allocation_count'])
def test_original_allocation_metadata_is_recomputed(complete_pools, key):
    row = copy.deepcopy(complete_pools[0][0])
    metadata = json.loads(row['scale_origin_metadata'])
    metadata[key] += 1
    row['scale_origin_metadata'] = json.dumps(metadata)
    with pytest.raises(RuntimeError, match='allocation metadata'):
        bridge.verify_rows('pantry', [row])


def test_arrow_roundtrip_preserves_all_four_profiles(complete_pools, tmp_path):
    from datasets import Dataset, DatasetDict, load_from_disk
    rows = [complete_pools[tier][0] for tier in range(4)]
    path = tmp_path / 'synthetic-pantry'
    DatasetDict({'multi_answer': Dataset.from_list(rows)}).save_to_disk(str(path))
    loaded = [dict(row) for row in load_from_disk(str(path))['multi_answer']]
    assert loaded == rows
    bridge.verify_rows('pantry', loaded)


@pytest.mark.parametrize('changes', [{'tier': True}, {'tier': 4}, {'seed': -1}, {'seed': 1.2},
    {'multiplier': False}, {'tag': ''}, {'target': {8: True}}, {'target': {8.0: 1}},
    {'target': {8: -1}}, {'target': {46: 1}}, {'joint_target': None},
    {'joint_target': {(8, 'unknown'): 1}}, {'joint_target': {(9, 'breakfast_formulation'): 1}},
    {'joint_target': {(8, 'breakfast_formulation'): True}}, {'joint_target': {8: 1}},
    {'domain': 'graph_coloring'}])
def test_invalid_inputs_rejected(changes):
    args = dict(domain='pantry', target=Counter({8: 1}), excluded=set(), seed=1, tag='INVALID_TEST', tier=0,
                joint_target=Counter({(8, 'breakfast_formulation'): 1}))
    args.update(changes)
    with pytest.raises(ValueError):
        bridge.build_pool(**args)


def test_capacity_exhaustion_is_explicit_not_a_partial_histogram(monkeypatch):
    monkeypatch.setattr(bridge, '_candidate', lambda *args: None)
    monkeypatch.setattr(bridge, 'MAX_PROPOSALS_PER_ROW', 2)
    with pytest.raises(RuntimeError, match='exhausted fixed proposal budget'):
        pool({(45, 'breakfast_formulation'): 1})


def test_source_inventory_pins_original_ingredient_and_verifier_dependencies():
    paths = bridge.source_paths()
    assert Path(bridge.__file__).resolve() in paths
    assert Path(bridge.original.__file__).resolve() in paths
    assert ROOT / 'var/data/pantry_plan_v1/ingredients.json' in paths
    assert Path(sys.modules['oat_drgrpo.pantry_plan'].__file__).resolve() in paths
    assert len(paths) == len(set(paths)) and all(path.is_file() for path in paths)
    assert hashlib.sha256(Path(constraints.__file__).read_bytes()).hexdigest() == bridge.ORIGIN_CONSTRAINTS_SHA256
