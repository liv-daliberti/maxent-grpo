#!/usr/bin/env python3
"""Submit E135: RLEP-Dr's update with its pool built online, on the E128 panel.

App. "What RLEP-Dr replays" measures why the offline comparator cannot move
breadth: its pool is harvested, per RLEP's own protocol, from a policy RL has
already trained --- here the paired terminal Dr.GRPO control --- and on
essentially every replay-eligible prompt that pool holds a single canonical
mode. A reviewer can accept that and still ask the sharper question: is the
gap between RLEP-Dr and Re:Dr a matter of *how* successes are rehearsed (a
positive-advantage row inside the group versus a fixed-coefficient likelihood
term) or of *what* is rehearsed and *when* it was collected?

E135 isolates the second factor. It keeps every part of RLEP-Dr's update ---
sixteen fresh rows plus two replayed successes under one common baseline,
frequency-preserving sampling, the at-least-two-successes eligibility rule, the
unchanged 16-row Dr.GRPO update on ineligible prompts --- and changes only the
pool: it starts empty and is filled from the learner's own verified fresh
rollouts as training proceeds, so a prompt becomes replay-eligible on the pass
after it was first solved at least twice. No canonicalization, deduplication or
balancing is applied; eight copies of one mode are eight rows.

If online RLEP recovers a substantial share of Re:Dr's breadth, the loss form
is not what separates the methods and the offline pool is. If it does not, the
fixed-coefficient likelihood term (whose single-slot form is E130) carries a
mechanism the self-extinguishing advantage row lacks.

Four keys move from the E128 control objective: the variant becomes ``rlep``,
the replay dose is two rows, the pool is online, and ineligible prompts fall
back to the plain update. The ``rlep`` variant branch of ``run_experiment.sh``
also switches the inert canonical-replay traversal off, exactly as it did for
the offline RLEP-Dr cells (E98-R1/E116), so E135 differs from those cells in
the pool alone and from the E128 control in the RLEP update alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_comparator_launch as shared  # noqa: E402

#: RLEP's published dose: sixteen fresh rollouts plus two replayed successes.
REPLAY_ROWS = 2


def objective_overrides(root: Path) -> dict[str, str]:
    return {
        "OAT_ZERO_VARIANT": "rlep",
        "OAT_ZERO_RLEP_REPLAY_COUNT": str(REPLAY_ROWS),
        "OAT_ZERO_RLEP_ONLINE_POOL": "1",
        "OAT_ZERO_RLEP_SPARSE_FALLBACK": "1",
    }


def common_exports(root: Path) -> tuple[str, ...]:
    return (
        f"OAT_ZERO_RLEP_REPLAY_COUNT={REPLAY_ROWS}",
        "OAT_ZERO_RLEP_ONLINE_POOL=1",
        "OAT_ZERO_RLEP_SPARSE_FALLBACK=1",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={shared.PASSES}",
        f"OAT_ZERO_MAX_PROMPT_EPOCHS={shared.PASSES}",
    )


def expected_exports(root: Path) -> tuple[str, ...]:
    return (
        "OAT_ZERO_VARIANT=rlep",
        "OAT_ZERO_RLEP_ONLINE_POOL=1",
        "OAT_ZERO_GAPO_ENABLED=0",
        "OAT_ZERO_SETPO_COEFFICIENT=0.0",
    )


def ledger_extras(root: Path) -> dict[str, object]:
    return {
        "control_cohort": "e128",
        "treatment_reference": "e132 (within-cohort Re:Dr)",
        "offline_sibling": (
            "e98r1/e116 sparse RLEP-Dr, whose pools were harvested from the "
            "paired terminal Dr.GRPO control and hold one canonical mode per "
            "eligible prompt (paper/results/rlep_pool_composition_20260924.json)"
        ),
        "replay_rows": REPLAY_ROWS,
        "pool": (
            "online: starts empty, every validator-positive fresh response is "
            "appended to its prompt after the group is scored; sampled two "
            "without replacement, frequency preserved, once a prompt holds at "
            "least two"
        ),
        "what_is_held_from_rlep_dr": (
            "16+2 mixed group under one common baseline, replay-row advantage "
            "1 - mean mixed reward, at-least-two eligibility, plain 16-row "
            "update on ineligible prompts, decoded-text replay rows"
        ),
        "what_moves_from_rlep_dr": "the pool's source policy and collection time",
        "pool_is_checkpointed": True,
        "declared_source_change": (
            "src/oat_drgrpo/rlep.py gains OnlineRLEPExperiencePool; args.py the "
            "rlep_online_pool flag; learner/grpo.py commits verified fresh rows "
            "after the replay draw; learner/run.py checkpoints the pool; "
            "train.sh and run_experiment.sh pass the flag through. Every hunk "
            "is unreachable unless rlep_online_pool is set."
        ),
    }


COHORT = shared.Cohort(
    tag="e135",
    arm="online_rlep",
    variant="rlep",
    title="E135  Qwen-0.5B  RLEP-Dr with an online experience pool",
    ledger="var/artifacts/e135_online_rlep_05b_jobs.json",
    protocol="paper/preregistration/e135_online_rlep_05b_20260924.md",
    objective_summary="RLEP_16_plus_2_with_online_frequency_preserving_pool",
    scientific_difference=(
        "against the matched E128 control: RLEP-Dr's 16+2 mixed update with "
        "the experience pool filled online from the learner's own verified "
        "rollouts; against E98-R1/E116 RLEP-Dr: the pool alone moves"
    ),
    objective_overrides=objective_overrides,
    snapshot_requirements=(
        ("src/oat_drgrpo/rlep.py", "class OnlineRLEPExperiencePool"),
        ("src/oat_drgrpo/args.py", "rlep_online_pool:"),
        ("src/oat_drgrpo/learner/grpo.py", "OnlineRLEPExperiencePool(minimum=rlep_count)"),
        ("src/oat_drgrpo/learner/run.py", "rlep_online_pool_state"),
        ("ops/train.sh", "--rlep-online-pool"),
        ("ops/run_experiment.sh", "OAT_ZERO_RLEP_ONLINE_POOL=\"$RLEP_ONLINE_POOL\""),
        ("src/oat_drgrpo/learner/run.py", "eval_mode_coverage_disjoint_draws"),
    ),
    expected_exports=expected_exports,
    ledger_extras=ledger_extras,
    common_exports=common_exports,
    patched_files=shared.PATCHED_FILES + ("src/oat_drgrpo/rlep.py",),
    # The pool only engages when a prompt is revisited, so the smoke makes two
    # passes over its 32 prompts: the second pass is where replay rows appear.
    smoke_overrides={
        "OAT_ZERO_MAX_QUERIES": "64",
        "OAT_ZERO_NUM_PROMPT_EPOCH": "2",
        "OAT_ZERO_MAX_PROMPT_EPOCHS": "2",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL": "64",
        "OAT_ZERO_SAVE_STEPS": "64",
        "OAT_ZERO_SAVE_FROM": "64",
    },
)


if __name__ == "__main__":
    raise SystemExit(shared.main(COHORT))
