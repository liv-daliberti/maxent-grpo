#!/usr/bin/env python3
"""Submit E131: the E128 Dr.GRPO control, trained for twelve passes.

The factorial matches optimizer updates and fresh rollouts between replay and
control, and switches the control's replay traversal off at the derivative
rather than spending it on more learning. E131 spends it. The objective is the
E128 control's, unmodified; only the horizon moves, from eight prompt passes to
twelve --- 4,608 optimizer updates against 3,072, with 50% more fresh rollouts.

Twelve is generous rather than matched. The measured replay likelihood-pass
subtotal is a fraction of one reference forward per visit, so 50% more of every
update is strictly more additional computation than replay consumes. The
comparison is therefore one-sided by construction and is reported as one-sided:
if the control still does not reach the replay arm after being given more than
replay costs, the accuracy result is not a compute artifact.

The horizon is expressed as an objective override rather than a module
constant so that the drift guard sees it, the scheduler record carries it, and
the audit refuses a cell whose held record says eight.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_comparator_launch as shared  # noqa: E402

#: Twelve prompt passes: 4,608 updates at 384 training rows.
PASSES = 12


def objective_overrides(root: Path) -> dict[str, str]:
    """The E128 control objective, with only the horizon moved.

    Nothing about the objective changes; both keys are schedule. They are
    routed through the override set so the drift guard registers them as
    deliberate and the held-job audit can require them.
    """

    return {
        "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
        "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
    }


def common_exports(root: Path) -> tuple[str, ...]:
    return (
        # Unchanged from the control: the replay traversal runs and its
        # derivative stays exactly zero. This arm buys extra task gradient,
        # not extra supervision.
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={PASSES}",
        f"OAT_ZERO_MAX_PROMPT_EPOCHS={PASSES}",
    )


def expected_exports(root: Path) -> tuple[str, ...]:
    return (
        f"OAT_ZERO_VARIANT={shared.e78.VARIANTS['control']}",
        "OAT_ZERO_GAPO_ENABLED=0",
        "OAT_ZERO_SETPO_COEFFICIENT=0.0",
    )


def ledger_extras(root: Path) -> dict[str, object]:
    return {
        "control_cohort": "e128",
        "horizon_passes": PASSES,
        "baseline_horizon_passes": shared.PASSES,
        "additional_updates": shared.TRAIN_ROWS * (PASSES - shared.PASSES),
        "comparison": "one_sided_generous_to_the_control",
        "why_not_equal_flops": (
            "total training FLOPs were never instrumented end to end; the "
            "measured replay likelihood-pass subtotal is 0.19 to 1.64 times "
            "one reference forward per visit, so 50% more of every update "
            "overshoots replay's cost in the control's favour"
        ),
    }


COHORT = shared.Cohort(
    tag="e131",
    arm="control_12pass",
    variant=shared.e78.VARIANTS["control"],
    title="E131  Qwen-0.5B  extended-horizon Dr.GRPO control (12 passes)",
    ledger="var/artifacts/e131_extended_horizon_control_05b_jobs.json",
    protocol=(
        "paper/preregistration/e131_extended_horizon_control_05b_20260922.md"
    ),
    objective_summary=(
        "DrGRPO_control_with_compute_only_canonical_replay_traversal_12_passes"
    ),
    scientific_difference=(
        "against the matched E128 control: the same objective trained for "
        "twelve prompt passes rather than eight; no objective difference of "
        "any kind"
    ),
    objective_overrides=objective_overrides,
    snapshot_requirements=(
        # The control must run the same runtime as the arms it is read with,
        # so the objective code is asserted present even though this arm never
        # enables it.
        ("src/oat_drgrpo/args.py", "gapo_enabled:"),
        ("src/oat_drgrpo/args.py", "setpo_coefficient:"),
        ("src/oat_drgrpo/learner/grpo.py", "gapo_group_rewards("),
        ("src/oat_drgrpo/learner/grpo.py", "shape_setpo_advantages("),
        ("src/oat_drgrpo/learner/run.py", "eval_mode_coverage_disjoint_draws"),
    ),
    expected_exports=expected_exports,
    ledger_extras=ledger_extras,
    common_exports=common_exports,
    passes=PASSES,
)


if __name__ == "__main__":
    raise SystemExit(shared.main(COHORT))
