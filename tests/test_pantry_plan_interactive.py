from __future__ import annotations

from oat_drgrpo.pantry_plan import PANTRY_PLAN_VERIFIER, PANTRY_PLAN_VERSION
from oat_drgrpo.pantry_plan_interactive import (
    PantryInteractiveState,
    STOP_ACTION,
    legal_pantry_actions,
    pantry_running_totals,
    render_pantry_policy_state,
    step_pantry_plan,
)


def _ingredient(ingredient_id: str, protein: str, fiber: str) -> dict:
    return {
        "id": ingredient_id,
        "available_g": 300,
        "step_g": 25,
        "min_if_used_g": 50,
        "attributes_per_100g": {
            "energy_kcal": "120",
            "protein_g": protein,
            "fiber_g": fiber,
            "sodium_mg": "4",
        },
        "tags": [],
    }


def _spec() -> dict:
    return {
        "verifier": PANTRY_PLAN_VERIFIER,
        "pantry_version": PANTRY_PLAN_VERSION,
        "ingredients": [
            _ingredient("lentils", "9", "8"),
            _ingredient("brown_rice", "3", "2"),
            _ingredient("chickpeas", "9", "7"),
        ],
        "targets": {
            "mass_g": {"min": "250", "max": "400"},
            "protein_g": {"min": "20"},
            "fiber_g": {"min": "10"},
        },
        "min_ingredients": 2,
        "max_ingredients": 2,
        "forbidden_tags": [],
        "certified_mode_count": 2,
    }


def _take(state: PantryInteractiveState, action: str) -> PantryInteractiveState:
    return step_pantry_plan(state, action, _spec()).state


def test_interactive_actions_express_the_same_verified_endpoint():
    state = PantryInteractiveState()
    state = _take(state, "lentils")
    state = _take(state, "200")
    state = _take(state, "brown_rice")
    state = _take(state, "100")
    result = step_pantry_plan(state, STOP_ACTION, _spec())

    assert result.terminal
    assert result.reward == 1.0
    assert result.validation is not None
    assert result.validation.canonical_key.endswith("brown_rice+lentils")


def test_action_mask_enforces_only_public_local_constraints():
    state = PantryInteractiveState()
    assert legal_pantry_actions(state, _spec()) == (
        "lentils",
        "brown_rice",
        "chickpeas",
    )

    state = _take(state, "lentils")
    assert legal_pantry_actions(state, _spec()) == tuple(
        str(value) for value in range(50, 301, 25)
    )
    state = _take(state, "50")
    assert STOP_ACTION not in legal_pantry_actions(state, _spec())

    state = _take(state, "brown_rice")
    state = _take(state, "50")
    assert legal_pantry_actions(state, _spec()) == (STOP_ACTION,)
    result = step_pantry_plan(state, STOP_ACTION, _spec())
    assert result.terminal and result.reward == 0.0


def test_running_totals_and_rendered_feedback_use_no_solution_catalogue():
    state = _take(_take(PantryInteractiveState(), "lentils"), "100")
    totals = dict(pantry_running_totals(state, _spec()))
    rendered = render_pantry_policy_state(state, _spec())

    assert totals["mass_g"] == 100
    assert totals["protein_g"] == 9
    assert "current=9" in rendered
    assert "certified" not in rendered.lower()
