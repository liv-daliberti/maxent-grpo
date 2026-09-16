from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
from modebench_level3_constraints import _mathir_family, build_pool, exact_pantry_supports
from make_pantry_plan_mode_data import enumerate_pantry_supports
from oat_drgrpo.mathir import enumerate_mathir_action_menu_keys, validate_mathir_action_menu


@pytest.mark.parametrize("difficulty", range(4))
def test_mathir_support_grammar_and_bindings_invariance(difficulty):
    rows = build_pool("mathir", Counter({5: 2}), set(), 709 + difficulty,
                      "level3_test", difficulty, multiplier=1)
    assert len(rows) == 2
    excluded = {
        ("mathir", spec["family"], tuple(sorted(spec["bindings"].items())))
        for row in rows for spec in (json.loads(row["answer"]),)
    }
    fresh = build_pool("mathir", Counter({5: 2}), excluded, 709 + difficulty,
                       "level3_test", difficulty, multiplier=1)
    fresh_ids = {
        ("mathir", spec["family"], tuple(sorted(spec["bindings"].items())))
        for row in fresh for spec in (json.loads(row["answer"]),)
    }
    assert not excluded & fresh_ids
    family = _mathir_family(difficulty)
    for row in rows:
        spec = json.loads(row["answer"])
        assert list(spec["actions"]) == list("ABCDEF")
        assert spec["max_steps"] == 4
        inverse = {command: key for key, command in spec["actions"].items()}
        for route in family.certified_routes:
            assert validate_mathir_action_menu(";".join(inverse[c] for c in route), spec)
    # Independent bindings and randomized IDs have the complete template support.
    keys = enumerate_mathir_action_menu_keys(json.loads(rows[1]["answer"]))
    assert len(keys) == rows[1]["answer_mode_count"] == 5
    digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    assert digest == json.loads(rows[1]["answer"])["valid_mode_key_sha256"]


@pytest.mark.parametrize("difficulty", range(4))
def test_pantry_exact_joint_histogram_reproducibility_and_exclusion(difficulty):
    joint = Counter({(8, "breakfast_formulation"): 1,
                     (20, "low_sodium_pantry_meal"): 1,
                     (32, "high_fiber_snack"): 1})
    target = Counter({8: 1, 20: 1, 32: 1})
    kwargs = dict(domain="pantry", target=target, seed=918 + difficulty,
                  tag="level3_test", difficulty=difficulty, multiplier=1,
                  joint_target=joint)
    rows = build_pool(excluded=set(), **kwargs)
    assert build_pool(excluded=set(), **kwargs) == rows
    assert Counter((row["answer_mode_count"], row["answer_mode_family"]) for row in rows) == joint
    excluded = {("pantry", row["instance_fingerprint"]) for row in rows}
    fresh = build_pool(excluded=excluded, **kwargs)
    assert not excluded & {("pantry", row["instance_fingerprint"]) for row in fresh}
    for row in rows:
        spec = json.loads(row["answer"])
        assert (spec["min_ingredients"], spec["max_ingredients"]) == (2, 4)
        assert len(exact_pantry_supports(spec)) == row["answer_mode_count"]


def test_pantry_integer_enumerator_agrees_with_original_decimal_verifier():
    row = build_pool("pantry", Counter({12: 1}), set(), 20,
                     "level3_test", 3, multiplier=1)[0]
    spec = json.loads(row["answer"])
    # Compare all positive and negative allocations in a smaller legal grid.
    spec["ingredients"] = copy.deepcopy(spec["ingredients"][:4])
    for ingredient in spec["ingredients"]:
        ingredient["available_g"] = ingredient["min_if_used_g"] + ingredient["step_g"]
    for targets in (spec["targets"], {"mass_g": {"min": "90", "max": "225"},
                                      "protein_g": {"min": "2.001", "max": "12.357"}}):
        spec["targets"] = targets
        fast = exact_pantry_supports(spec)
        slow = enumerate_pantry_supports(spec)
        assert set(fast) == set(slow)


def test_rejects_inconsistent_joint_target():
    with pytest.raises(ValueError, match="joint target"):
        build_pool("pantry", Counter({8: 1}), set(), 1, "test", 0,
                   joint_target=Counter({(9, "plant_protein_bowl"): 1}))
