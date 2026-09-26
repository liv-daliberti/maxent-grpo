"""Prospective finite-action interface for PantryPlan language policies.

The original PantryPlan benchmark asks a model to emit a complete allocation
string in one completion.  This module exposes the same endpoint verifier as a
small state machine:

1. choose an unused, locally permitted ingredient;
2. choose one inventory-aligned quantity for it; and
3. stop when ready.

The legal-action mask contains only information already present in the public
problem specification.  It does not expose certified supports or endpoint
feasibility.  A policy must still discover an allocation satisfying every
joint target, and semantic identity still comes exclusively from
``validate_pantry_plan`` at STOP.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from .pantry_plan import (
    PantryPlanError,
    PantryPlanSpec,
    PantryPlanValidation,
    parse_pantry_plan_spec,
    validate_pantry_plan,
)


STOP_ACTION = "STOP"


@dataclass(frozen=True)
class PantryInteractiveState:
    """One locally valid partial allocation."""

    allocations_g: tuple[tuple[str, int], ...] = ()
    pending_ingredient: str | None = None
    terminal: bool = False


@dataclass(frozen=True)
class PantryInteractiveTransition:
    """Result of one finite action."""

    state: PantryInteractiveState
    terminal: bool
    reward: float
    validation: PantryPlanValidation | None


def _parsed(spec: Mapping[str, Any] | PantryPlanSpec) -> PantryPlanSpec:
    if isinstance(spec, PantryPlanSpec):
        return spec
    return parse_pantry_plan_spec(spec)


def _allocation_map(state: PantryInteractiveState) -> dict[str, int]:
    return dict(state.allocations_g)


def legal_pantry_actions(
    state: PantryInteractiveState,
    spec: Mapping[str, Any] | PantryPlanSpec,
) -> tuple[str, ...]:
    """Return the exact public-information action menu for ``state``."""

    parsed = _parsed(spec)
    if state.terminal:
        return ()
    allocations = _allocation_map(state)
    if state.pending_ingredient is not None:
        if state.pending_ingredient in allocations:
            raise PantryPlanError("pending ingredient is already allocated")
        ingredient = parsed.ingredient_by_id.get(state.pending_ingredient)
        if ingredient is None:
            raise PantryPlanError("pending ingredient is absent from the pantry")
        return tuple(
            str(grams)
            for grams in range(
                ingredient.min_if_used_g,
                ingredient.available_g + 1,
                ingredient.step_g,
            )
        )

    actions: list[str] = []
    if len(allocations) < parsed.max_ingredients:
        actions.extend(
            ingredient.ingredient_id
            for ingredient in parsed.ingredients
            if ingredient.ingredient_id not in allocations
            and not (ingredient.tags & parsed.forbidden_tags)
        )
    if len(allocations) >= parsed.min_ingredients:
        actions.append(STOP_ACTION)
    if not actions:
        raise PantryPlanError("interactive state has no legal action")
    return tuple(actions)


def _candidate(state: PantryInteractiveState) -> str:
    return ";".join(
        f"{ingredient_id}={grams}"
        for ingredient_id, grams in state.allocations_g
    )


def step_pantry_plan(
    state: PantryInteractiveState,
    action: str,
    spec: Mapping[str, Any],
) -> PantryInteractiveTransition:
    """Apply one masked action and verify only when the policy chooses STOP."""

    parsed = parse_pantry_plan_spec(spec)
    normalized = str(action).strip()
    if normalized not in legal_pantry_actions(state, parsed):
        raise PantryPlanError(f"action {normalized!r} is not legal in this state")

    if state.pending_ingredient is None:
        if normalized == STOP_ACTION:
            terminal_state = PantryInteractiveState(
                allocations_g=state.allocations_g,
                pending_ingredient=None,
                terminal=True,
            )
            validation = validate_pantry_plan(_candidate(state), spec)
            return PantryInteractiveTransition(
                state=terminal_state,
                terminal=True,
                reward=1.0 if validation is not None else 0.0,
                validation=validation,
            )
        return PantryInteractiveTransition(
            state=PantryInteractiveState(
                allocations_g=state.allocations_g,
                pending_ingredient=normalized,
            ),
            terminal=False,
            reward=0.0,
            validation=None,
        )

    allocations = {
        **_allocation_map(state),
        state.pending_ingredient: int(normalized),
    }
    return PantryInteractiveTransition(
        state=PantryInteractiveState(
            allocations_g=tuple(sorted(allocations.items())),
            pending_ingredient=None,
        ),
        terminal=False,
        reward=0.0,
        validation=None,
    )


def pantry_running_totals(
    state: PantryInteractiveState,
    spec: Mapping[str, Any] | PantryPlanSpec,
) -> tuple[tuple[str, Decimal], ...]:
    """Compute public running totals for a partial allocation."""

    parsed = _parsed(spec)
    totals: dict[str, Decimal] = {"mass_g": Decimal(0)}
    for ingredient_id, grams in state.allocations_g:
        ingredient = parsed.ingredient_by_id[ingredient_id]
        quantity = Decimal(grams)
        totals["mass_g"] += quantity
        for name, per_100g in ingredient.attributes_per_100g:
            totals[name] = (
                totals.get(name, Decimal(0))
                + per_100g * quantity / Decimal(100)
            )
    return tuple(sorted(totals.items()))


def render_pantry_policy_state(
    state: PantryInteractiveState,
    spec: Mapping[str, Any],
) -> str:
    """Render public state feedback without any certified-answer leakage."""

    parsed = parse_pantry_plan_spec(spec)
    allocations = (
        ";".join(f"{name}={grams}" for name, grams in state.allocations_g)
        or "(none)"
    )
    totals = dict(pantry_running_totals(state, parsed))
    target_lines = []
    for target in parsed.targets:
        value = totals.get(target.attribute, Decimal(0))
        bounds = []
        if target.minimum is not None:
            bounds.append(f"min={target.minimum}")
        if target.maximum is not None:
            bounds.append(f"max={target.maximum}")
        target_lines.append(
            f"- {target.attribute}: current={value}; " + "; ".join(bounds)
        )
    phase = (
        f"choose grams for {state.pending_ingredient}"
        if state.pending_ingredient is not None
        else "choose an unused ingredient or STOP"
    )
    return "\n".join(
        [
            f"Partial allocation: {allocations}",
            "Running public totals:",
            *target_lines,
            f"Decision phase: {phase}.",
        ]
    )
