#!/usr/bin/env python3
"""Submit E132: Re:Dr itself, on the E128-controlled panel.

E130 and E131 are both measured against the E128 control, but the Re:Dr arm
they are compared against is the published E78 replay arm, whose cells span
node302 (A100) and node105 (A5000). Every "recovers N% of the Re:Dr effect"
statement therefore pairs a within-cohort numerator against a cross-cohort
denominator. For PCMD the seam is small --- the recorded E128-minus-E78 control
shift is +.0028 over 20 paired cells --- but for correctness it is not: the two
controls differ by more than ten points of pass@8 on Countdown and Python.

E132 removes the seam by running Re:Dr on this panel, so the comparison becomes
within-cohort on both axes.

Exactly two keys move from the E128 control objective: the variant, and the
live replay derivative. Bank capacity is deliberately **not** restated. The
default is 16, which is Re:Dr's capacity, and this launcher family treats
silence as the stronger assertion that a key is untouched --- restating a value
that happened to differ is what the drift guard exists to catch. E132 is
therefore E130 with one difference: E130 forces the bank to one slot, E132
leaves it at Re:Dr's capacity.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_comparator_launch as shared  # noqa: E402


def objective_overrides(root: Path) -> dict[str, str]:
    """Re:Dr's objective, stated as the two keys that move.

    The shared builder starts from the E128 control objective, so the replay
    variant and the live replay derivative are stated here rather than
    inherited. Capacity, objective, coefficient, mass coefficient, cross-prompt
    scheduling and bootstrap are already the treatment's values in that
    baseline and are deliberately not restated.
    """

    return {
        "OAT_ZERO_VARIANT": shared.e78.VARIANTS["replay"],
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0",
    }


def common_exports(root: Path) -> tuple[str, ...]:
    return (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
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
        "reference_cohort": "e78 replay arm (Re:Dr), the cross-cohort arm this "
                            "cohort exists to replace",
        "replay_bank_capacity": "default (16); not restated by this launcher",
        "purpose": "within_cohort_redr_for_the_e130_and_e131_comparisons",
        "seam_being_closed": (
            "E130/E131 are measured against the E128 control while Re:Dr came "
            "from E78, whose cells span node302 (A100) and node105 (A5000); "
            "the recorded PCMD control shift is +.0028 over 20 paired cells "
            "but the pass@8 controls differ by over ten points on Countdown "
            "and Python"
        ),
        "distinct_from_e130": (
            "same cohort, same two keys moved from the control; E130 additionally "
            "forces the bank to one slot, E132 leaves capacity at Re:Dr's default"
        ),
        "declared_source_change": None,
    }


COHORT = shared.Cohort(
    tag="e132",
    arm="redr_matched",
    variant=shared.e78.VARIANTS["replay"],
    title="E132  Qwen-0.5B  Re:Dr on the E128-controlled panel",
    ledger="var/artifacts/e132_matched_redr_05b_jobs.json",
    protocol="paper/preregistration/e132_matched_redr_05b_20260923.md",
    objective_summary="DrGRPO_with_verified_likelihood_replay_at_default_capacity",
    scientific_difference=(
        "against the matched E128 control: the verified-likelihood replay "
        "derivative is live over a bank at Re:Dr's default capacity, so modes "
        "are retained and rehearsed uniformly; no other live-gradient difference"
    ),
    objective_overrides=objective_overrides,
    snapshot_requirements=(
        ("src/oat_drgrpo/canonical_replay.py",
         "canonical_replay_uniform_verified_likelihood_loss"),
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
