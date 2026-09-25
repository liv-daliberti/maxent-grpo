#!/usr/bin/env python3
"""Submit E127 SetPO cells behind a real 32-query learner smoke.

SetPO (Li et al., 2026) adds each rollout's leave-one-out marginal contribution
to its group's kernelized set diversity onto the Dr.GRPO advantage. It is the
sharpest available test of one thing this campaign asserts implicitly: that a
verified outcome key is a better index of a mode than a sentence embedding is.
SetPO measures diversity in embedding space over generated text; Re:Max and
Semantic-MaxEnt measure it over certified answer identity. Same panel, same
control, a different notion of "different".

The 25 scientific cells inherit E78's registered data and schedule, and are
differenced against the matched E128 control rather than E78's own, for the
runtime, placement and evaluator reasons documented in
``diversity_comparator_launch``. Jobs are submitted held, audited through
Slurm, recorded atomically, and released only after the ledger exists.

This cohort replaces a lambda=0.1 submission withdrawn the same day, before any
scientific cell started; see ``COEFFICIENT`` below.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity_comparator_launch as shared  # noqa: E402


#: The paper names no embedder. MiniLM is the family DQO reports for the same
#: purpose, which keeps the two embedding-kernel methods in the comparison
#: commensurable, and it is staged locally so compute nodes need no network.
EMBEDDER = (
    "var/cache/huggingface/transformers/"
    "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/"
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
#: SetPO's lambda. The paper reports no value for a 0.5B verifier-reward
#: setting, so it is registered rather than tuned --- but it is registered
#: against a measured scale rather than guessed.
#:
#: The withdrawn lambda=0.1 submission's smoke measured
#: ``marginal_abs_mean = 0.01279`` against ``base_advantage_rms = 0.24206``,
#: putting SetPO's contribution at 0.53% of the Dr.GRPO advantage --- small
#: enough that the arm would most likely have been indistinguishable from its
#: control for reasons having nothing to do with the method. lambda=1.0 puts
#: the term at roughly 5.3% of the task advantage, matching the ratio this
#: campaign's own adaptive semantic controller targets
#: (``semantic_rms_target_ratio = 0.05``).
#:
#: This is a scale calibration read off an operational smoke, not a tuning
#: pass: no scientific cell had run, and no outcome metric was consulted. See
#: var/artifacts/withdrawn/README_e127_lambda0p1_withdrawal.md.
COEFFICIENT = "1.0"
EMBED_BATCH_SIZE = "64"


def objective_overrides(root: Path) -> dict[str, str]:
    embedder = root / EMBEDDER
    if not (embedder / "config.json").is_file():
        raise SystemExit(
            f"SetPO embedder snapshot is absent or incomplete: {embedder}"
        )
    return {
        "OAT_ZERO_VARIANT": "setpo",
        "OAT_ZERO_SETPO_COEFFICIENT": COEFFICIENT,
        "OAT_ZERO_SETPO_EMBEDDER_PATH": str(embedder),
        "OAT_ZERO_SETPO_EMBED_BATCH_SIZE": EMBED_BATCH_SIZE,
    }


def expected_exports(root: Path) -> tuple[str, ...]:
    return (
        "OAT_ZERO_VARIANT=setpo",
        f"OAT_ZERO_SETPO_COEFFICIENT={COEFFICIENT}",
        f"OAT_ZERO_SETPO_EMBEDDER_PATH={root / EMBEDDER}",
    )


def ledger_extras(root: Path) -> dict[str, object]:
    """Pin the kernel: a different embedder is a different objective."""

    embedder = root / EMBEDDER
    weights = sorted(embedder.glob("*.safetensors"))
    if not weights:
        raise SystemExit(f"SetPO embedder has no weights: {embedder}")
    return {
        "setpo_coefficient": float(COEFFICIENT),
        "setpo_embedder_path": str(embedder),
        "setpo_embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "setpo_embedder_weights_sha256": {
            path.name: shared.e81.digest(path) for path in weights
        },
        "setpo_kernel": "mean_pooled_cosine_similarity_clamped_to_unit",
    }


COHORT = shared.Cohort(
    tag="e127",
    arm="setpo",
    variant="setpo",
    title="E127  Qwen-0.5B  SetPO set-level diversity shaping",
    ledger="var/artifacts/e127_setpo_05b_jobs.json",
    protocol="paper/preregistration/e127_setpo_05b_20260918.md",
    objective_summary="DrGRPO_with_set_level_leave_one_out_diversity_advantage",
    scientific_difference=(
        f"against the matched E128 control: lambda={COEFFICIENT} times each "
        "row's leave-one-out marginal contribution to its group's kernelized "
        "set diversity is added to the Dr.GRPO advantage; no other "
        "live-gradient difference"
    ),
    objective_overrides=objective_overrides,
    snapshot_requirements=(
        ("src/oat_drgrpo/args.py", "setpo_coefficient:"),
        ("src/oat_drgrpo/setpo.py", "setpo_marginal_contributions"),
        ("src/oat_drgrpo/setpo_embedder.py", "SetPOEmbedder"),
        ("src/oat_drgrpo/learner/grpo.py", "shape_setpo_advantages("),
        ("ops/run_experiment.sh", "setpo)"),
        ("ops/train.sh", "--setpo-coefficient"),
    ),
    expected_exports=expected_exports,
    ledger_extras=ledger_extras,
)


if __name__ == "__main__":
    raise SystemExit(shared.main(COHORT))
