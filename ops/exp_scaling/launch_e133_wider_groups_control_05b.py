#!/usr/bin/env python3
"""Submit E133: the control with wider fresh groups, on the E128 panel.

App. "The control at a longer horizon" names three ways a control could spend
replay's share of the budget on learning: more task-gradient passes over
existing groups, larger fresh-rollout groups, and more fresh-sampling updates.
E131 measured the third and closed neither gap. E133 measures the second.

The distinction matters for the diversity claim specifically. Extra passes give
more updates on groups of the same width; wider groups change what a single
group can contain. If drawing 24 rollouts per prompt instead of 16 recovers
Re:Dr's breadth, then retention is not the mechanism and "sample wider" is the
simpler explanation. That is the sharpest untested alternative to the paper's
mechanism claim.

E133 gets the *same* extra budget E131 received, spent differently: group width
up 50% at the unchanged eight-pass horizon, so the update count stays at 3,072.
E131 held width and raised count; E133 holds count and raises width. Both land
near 1.5x the control's training compute, against Re:Dr's measured +2.4%.

Three coupled keys move. Group width alone would leave the learner consuming
sixteen of the twenty-four rollouts per update, so the train batch and the
per-device buffer move with it and one update still consumes exactly one
prompt's group. They are declared through ``objective_overrides`` because that
is what the drift guard audits: anything that differs from the E78 control
objective and is not declared there is refused, and these are exactly the
differences this arm intends.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_comparator_launch as shared  # noqa: E402

#: Fresh rollouts per prompt. The control panel draws 16; this is 1.5x, the
#: same multiple of training compute E131 was given as extra passes.
GROUP = 24
#: The control's width, for the ledger to record what the multiple is against.
BASELINE_GROUP = 16


def objective_overrides(root: Path) -> dict[str, str]:
    """Group width, and the two keys that must move with it.

    ``TRAIN_BATCH_SIZE`` keeps one optimizer update consuming exactly one
    prompt's group, which is what holds the update count at 3,072 and makes
    this arm a reallocation of E131's budget rather than an addition to it.
    ``PI_BUFFER_MAXLEN_PER_DEVICE`` sizes the rollout buffer to the group.
    """

    return {
        "OAT_ZERO_NUM_SAMPLES": str(GROUP),
        "OAT_ZERO_TRAIN_BATCH_SIZE": str(GROUP),
        "OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE": str(GROUP),
    }


def common_exports(root: Path) -> tuple[str, ...]:
    return (
        # The replay traversal stays inert: this is a control arm.
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
        f"OAT_ZERO_NUM_SAMPLES={GROUP}",
        f"OAT_ZERO_TRAIN_BATCH_SIZE={GROUP}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={shared.PASSES}",
        f"OAT_ZERO_MAX_PROMPT_EPOCHS={shared.PASSES}",
    )


def expected_exports(root: Path) -> tuple[str, ...]:
    return (
        f"OAT_ZERO_NUM_SAMPLES={GROUP}",
        "OAT_ZERO_GAPO_ENABLED=0",
        "OAT_ZERO_SETPO_COEFFICIENT=0.0",
    )


def ledger_extras(root: Path) -> dict[str, object]:
    return {
        "control_cohort": "e128",
        "treatment_reference": "e132 (within-cohort Re:Dr)",
        "budget_sibling": "e131, which received the same ~1.5x training compute "
                          "as extra passes rather than as wider groups",
        "fresh_group_size": GROUP,
        "baseline_fresh_group_size": BASELINE_GROUP,
        "update_count_held": True,
        "reallocation": "larger fresh-rollout groups, the second of the three "
                        "alternatives named in the compute-budget appendix",
        "why_this_one": (
            "extra passes give more updates on groups of the same width; wider "
            "groups change what a group can contain, so this is the "
            "reallocation that tests whether breadth is a sampling artifact "
            "rather than a retention effect"
        ),
        "measured_replay_overhead_being_exceeded": (
            "+2.4% of total learner compute, +1.1% wall clock, paired on "
            "matched hardware against this same control"
        ),
        "declared_source_change": None,
    }


COHORT = shared.Cohort(
    tag="e133",
    arm="wide_groups",
    variant=shared.e78.VARIANTS["control"],
    title=f"E133  Qwen-0.5B  control with {GROUP}-rollout fresh groups",
    ledger="var/artifacts/e133_wider_groups_control_05b_jobs.json",
    protocol="paper/preregistration/e133_wider_groups_control_05b_20260924.md",
    objective_summary="DrGRPO_control_with_wider_fresh_groups",
    scientific_difference=(
        "against the matched E128 control: fresh groups carry "
        f"{GROUP} rollouts per prompt rather than {BASELINE_GROUP}, with the "
        "update count and every other quantity held; the replay derivative "
        "stays inert"
    ),
    objective_overrides=objective_overrides,
    snapshot_requirements=(
        ("ops/run_experiment.sh", "verified_first_replay_rehearsal_only)"),
        ("src/oat_drgrpo/learner/run.py", "eval_mode_coverage_disjoint_draws"),
    ),
    expected_exports=expected_exports,
    ledger_extras=ledger_extras,
    common_exports=common_exports,
)


if __name__ == "__main__":
    raise SystemExit(shared.main(COHORT))
