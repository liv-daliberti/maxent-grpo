#!/usr/bin/env python3
"""Submit E130: Re:Dr with a one-slot bank, on the E128-controlled panel.

Re:Dr changes two things at once relative to its control. It adds a supervised
likelihood term on the policy's own past verified outputs, and it keys that
term by canonical mode, rehearsing the distinct modes a prompt has discovered
uniformly. E130 keeps the first and removes the second by setting the bank
capacity to one: the bank retains each prompt's first discovered mode and no
other, so the replay loss is a length-normalized likelihood term on exactly one
stored success per prompt and mode identity never enters the objective.

Everything else is Re:Dr. Same ``verified_likelihood_per_rollout`` objective,
same ``alpha = 0.10``, same one scheduled bank per optimizer update, same
admission rule, same horizon, same evaluation. Three keys move relative to the
E128 control objective and the launcher refuses a fourth.

This is not E120-R1. That cohort kept one exemplar per discovered mode and
changed the within-bank weights from uniform to fresh-observation frequency.
E130 changes what the bank holds, so there is nothing within it to weight.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_comparator_launch as shared  # noqa: E402

#: One slot per prompt. The bank then holds the first canonical mode the prompt
#: discovers, and admission of any later mode has nowhere to go.
CAPACITY = 1


def objective_overrides(root: Path) -> dict[str, str]:
    """Re:Dr's objective, with the bank reduced to a single slot.

    The shared builder starts from the E128 control objective, so the replay
    variant and the live replay derivative are both stated here rather than
    inherited. Every other replay key --- objective, coefficient, mass
    coefficient, cross-prompt scheduling, bootstrap --- is already the
    treatment's value in that baseline and is deliberately not restated: the
    drift guard would reject a restatement that happened to differ, and
    silence is the stronger assertion that it does not.
    """

    return {
        "OAT_ZERO_VARIANT": shared.e78.VARIANTS["replay"],
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY": str(CAPACITY),
    }


def common_exports(root: Path) -> tuple[str, ...]:
    return (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        # The whole point of the arm: the replay gradient is live, and the bank
        # it is computed over holds one mode.
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        f"OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY={CAPACITY}",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE="
        "verified_likelihood_per_rollout",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={shared.PASSES}",
        f"OAT_ZERO_MAX_PROMPT_EPOCHS={shared.PASSES}",
    )


def expected_exports(root: Path) -> tuple[str, ...]:
    return (
        f"OAT_ZERO_VARIANT={shared.e78.VARIANTS['replay']}",
        "OAT_ZERO_GAPO_ENABLED=0",
        "OAT_ZERO_SETPO_COEFFICIENT=0.0",
    )


def ledger_extras(root: Path) -> dict[str, object]:
    return {
        "control_cohort": "e128",
        "treatment_reference": "e78 replay arm (Re:Dr), capacity 16",
        "replay_bank_capacity": CAPACITY,
        "ablation": "mode_identity_removed_from_the_replay_objective",
        "distinct_from_e120r1": (
            "E120-R1 held one exemplar per discovered mode and reweighted "
            "within the bank; E130 reduces the bank to one slot, so mode "
            "identity cannot enter the objective at all"
        ),
        "declared_source_change": (
            "validate_zero_math_args capacity floor made objective-dependent: "
            "two for bank_balance, whose KL term is undefined on a one-mode "
            "group, and one for the mass objectives, which already score "
            "singleton groups in ordinary training"
        ),
    }


COHORT = shared.Cohort(
    tag="e130",
    arm="replay_singleton",
    variant=shared.e78.VARIANTS["replay"],
    title="E130  Qwen-0.5B  mode-agnostic replay, bank capacity one",
    ledger="var/artifacts/e130_mode_agnostic_replay_05b_jobs.json",
    protocol="paper/preregistration/e130_mode_agnostic_replay_05b_20260922.md",
    objective_summary="DrGRPO_with_single_slot_verified_likelihood_replay",
    scientific_difference=(
        "against the matched E128 control: the verified-likelihood replay "
        "derivative is live over a one-slot bank, so a past success is "
        "rehearsed without the mode keying Re:Dr adds; no other live-gradient "
        "difference"
    ),
    objective_overrides=objective_overrides,
    snapshot_requirements=(
        ("src/oat_drgrpo/args.py", "online_canonical_replay_capacity:"),
        ("src/oat_drgrpo/args.py", "minimum_capacity"),
        # Both halves of the capacity-1 patch, so a snapshot carrying only the
        # args.py half cannot pass this gate again.
        ("src/oat_drgrpo/online_canonical_bank.py", "at least one"),
        (
            "src/oat_drgrpo/canonical_replay.py",
            "canonical_replay_uniform_verified_likelihood_loss",
        ),
        ("ops/run_experiment.sh", "verified_first_replay_rehearsal_only)"),
        ("ops/train.sh", "--online-canonical-replay-capacity"),
        ("src/oat_drgrpo/learner/run.py", "eval_mode_coverage_disjoint_draws"),
    ),
    expected_exports=expected_exports,
    ledger_extras=ledger_extras,
    common_exports=common_exports,
)


if __name__ == "__main__":
    raise SystemExit(shared.main(COHORT))
