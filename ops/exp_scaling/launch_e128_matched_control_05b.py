#!/usr/bin/env python3
"""Submit E128 matched Dr.GRPO controls for the E126/E127 comparison.

E97 and E115 could difference their comparator against the completed E78
control and honestly say only the objective differed. E126 and E127 cannot:
the E78 runtime was retired by the irreversible 2026-09-04 cleanup and is not
reconstructible from any commit, and these cohorts run on ``cs`` A5000 nodes
rather than the mltheory nodes the E78 cells were pinned to. Read against E78,
a GAPO or SetPO effect would carry a GPU-model term and an unmeasurable
runtime term.

E128 removes both by re-running the control arm itself: same five domains,
same five seeds, same schedule, same snapshot, same A5000 placement as E126 and
E127. The objective is exactly E78's ``control`` --- ordinary Dr.GRPO with the
passive, zero-derivative canonical replay traversal --- so the three cohorts
form a self-contained paired comparison.

This does not restate or replace the E78 record. The comparators already
trained against E78 keep their own pairings; only the two new rows are read
against E128.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_comparator_launch as shared  # noqa: E402


def objective_overrides(root: Path) -> dict[str, str]:
    """No overrides: the arm *is* the E78 control objective.

    Returning nothing is the point. The shared builder starts from
    ``e78.fixed_objective("control")`` and refuses any key that moves away from
    it, so an empty override set is a machine-checked assertion that this arm
    is the control rather than a restatement of one.
    """

    return {}


def expected_exports(root: Path) -> tuple[str, ...]:
    return (
        f"OAT_ZERO_VARIANT={shared.e78.VARIANTS['control']}",
        "OAT_ZERO_GAPO_ENABLED=0",
        "OAT_ZERO_SETPO_COEFFICIENT=0.0",
    )


def ledger_extras(root: Path) -> dict[str, object]:
    return {
        "control_for": ["e126", "e127"],
        "control_objective_source": "launch_e78_verified_replay_only_05b.fixed_objective('control')",
        "why_not_e78": (
            "the E78 runtime e76_tuned_scale_96f68ebb47757af8 was retired by "
            "the irreversible 2026-09-04 cleanup and no commit reproduces it; "
            "E78 cells are also pinned to mltheory node302/node105 while these "
            "run on cs A5000"
        ),
    }


COHORT = shared.Cohort(
    tag="e128",
    arm="control",
    variant=shared.e78.VARIANTS["control"],
    title="E128  Qwen-0.5B  matched Dr.GRPO control on cs A5000",
    ledger="var/artifacts/e128_matched_control_05b_jobs.json",
    protocol="paper/preregistration/e128_matched_control_05b_20260918.md",
    objective_summary="DrGRPO_control_with_compute_only_canonical_replay_traversal",
    scientific_difference=(
        "none by construction: this is the E78 control objective re-run under "
        "the E126/E127 runtime and placement, so the two comparators have a "
        "GPU-matched and runtime-matched paired control"
    ),
    objective_overrides=objective_overrides,
    snapshot_requirements=(
        # The control must run the same runtime as the arms it serves, so the
        # objective code is asserted present even though this arm never enables
        # it. A control built from a snapshot lacking these files would be a
        # different runtime wearing the same name.
        ("src/oat_drgrpo/args.py", "gapo_enabled:"),
        ("src/oat_drgrpo/args.py", "setpo_coefficient:"),
        ("src/oat_drgrpo/learner/grpo.py", "gapo_group_rewards("),
        ("src/oat_drgrpo/learner/grpo.py", "shape_setpo_advantages("),
        ("src/oat_drgrpo/learner/run.py", "eval_mode_coverage_disjoint_draws"),
    ),
    expected_exports=expected_exports,
    ledger_extras=ledger_extras,
)


if __name__ == "__main__":
    raise SystemExit(shared.main(COHORT))
