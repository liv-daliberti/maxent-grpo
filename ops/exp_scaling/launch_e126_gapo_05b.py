#!/usr/bin/env python3
"""Submit E126 GAPO cells behind a real 32-query learner smoke.

GAPO (Anschel et al., EMNLP 2025) replaces the binary task reward with a
group-level frequency-aware reward over the prompt's enumerated valid set. It
is the closest published analogue to this campaign's own pressure --- both
refuse to let an already-produced mode keep its full weight --- and the
difference the comparison isolates is *where* that refusal acts: GAPO flattens
frequency inside the fresh rollout group, Re:Max equalizes rehearsal across a
bank of verified modes over time.

The 25 scientific cells inherit their data, schedule, and completed E78 control
comparator cell by cell; placement is the declared exception, documented in
``diversity_comparator_launch``. Jobs are submitted held, audited through
Slurm, recorded atomically, and released only after the ledger exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_comparator_launch as shared  # noqa: E402


SUPPORT_INDEX = "var/artifacts/gapo_support_index.json"
REWARD_SCALE = "unit"


def objective_overrides(root: Path) -> dict[str, str]:
    index = root / SUPPORT_INDEX
    if not index.is_file():
        raise SystemExit(
            f"GAPO support index is absent: {index}\n"
            "build it with ops/build_gapo_support_index.py before submitting"
        )
    return {
        "OAT_ZERO_VARIANT": "gapo",
        "OAT_ZERO_GAPO_ENABLED": "1",
        "OAT_ZERO_GAPO_SUPPORT_INDEX": str(index),
        "OAT_ZERO_GAPO_REWARD_SCALE": REWARD_SCALE,
    }


def expected_exports(root: Path) -> tuple[str, ...]:
    return (
        "OAT_ZERO_VARIANT=gapo",
        "OAT_ZERO_GAPO_ENABLED=1",
        f"OAT_ZERO_GAPO_REWARD_SCALE={REWARD_SCALE}",
        f"OAT_ZERO_GAPO_SUPPORT_INDEX={root / SUPPORT_INDEX}",
    )


def ledger_extras(root: Path) -> dict[str, object]:
    """Pin the support index by digest: L is part of the objective."""

    return {
        "gapo_support_index": str(root / SUPPORT_INDEX),
        "gapo_support_index_sha256": shared.e81.digest(root / SUPPORT_INDEX),
        "gapo_reward_scale": REWARD_SCALE,
        "gapo_support_source": "modebench_answer_mode_count",
    }


COHORT = shared.Cohort(
    tag="e126",
    arm="gapo",
    variant="gapo",
    title="E126  Qwen-0.5B  GAPO frequency-aware group reward",
    ledger="var/artifacts/e126_gapo_05b_jobs.json",
    protocol="paper/preregistration/e126_gapo_05b_20260918.md",
    objective_summary="DrGRPO_with_group_frequency_aware_reward_over_enumerated_support",
    scientific_difference=(
        "against E78 control: the binary task reward is replaced by GAPO's "
        "group frequency-aware reward with L read from ModeBench's enumerated "
        "support; no other live-gradient difference"
    ),
    objective_overrides=objective_overrides,
    snapshot_requirements=(
        ("src/oat_drgrpo/args.py", "gapo_enabled:"),
        ("src/oat_drgrpo/gapo.py", "gapo_group_rewards"),
        ("src/oat_drgrpo/learner/grpo.py", "gapo_group_rewards("),
        ("ops/run_experiment.sh", "gapo)"),
        ("ops/train.sh", "--gapo-enabled"),
    ),
    expected_exports=expected_exports,
    ledger_extras=ledger_extras,
)


if __name__ == "__main__":
    raise SystemExit(shared.main(COHORT))
