#!/usr/bin/env python3
"""Refresh the newest free-form cohorts and their live charts."""

from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "var/artifacts"
PARSER = ROOT / "ops/exp_scaling/parse_scaling_curve.py"
LOCK = ARTIFACTS / "refresh_latest_freeform_05b.lock"
CELLS = (
    ("cde32_freeform_05b_ema_10ep_v4_preemptsafe", 384),
    ("gce32_freeform_05b_ema_10ep_v5", 192),
    ("cde33_freeform_3b_ema_10ep_v3_a100", 384),
    ("gce33_freeform_3b_ema_10ep_v3_a100", 192),
    ("cde36_answer_option_mi_dual_eval_05b_v1", 384),
    ("gce36_answer_option_mi_dual_eval_05b_v1", 192),
    ("cde37_outcome_collision_05b_v1", 384),
    ("gce37_outcome_collision_05b_v1", 192),
    ("cde38_semantic_shannon_05b_v1", 384),
    ("gce38_semantic_shannon_05b_v1", 192),
    (
        "mte39_math12k_384_semantic_entropy_05b_v1",
        384,
        ("math",),
    ),
    (
        "cde41_semantic_shannon_advantage_05b_v1",
        384,
        ("multi_answer", "unique_answer"),
    ),
    (
        "gce41_semantic_shannon_advantage_05b_v1",
        192,
        ("multi_answer", "unique_answer"),
    ),
    (
        "mte41_math12k_384_semantic_shannon_advantage_05b_v1",
        384,
        ("math",),
    ),
    (
        "cde43_success_conditioned_signed_semantic_shannon_05b_v1",
        384,
        ("multi_answer", "unique_answer"),
    ),
    (
        "gce43_success_conditioned_signed_semantic_shannon_05b_v1",
        192,
        ("multi_answer", "unique_answer"),
    ),
    (
        "mte43_math12k_384_success_conditioned_signed_semantic_shannon_05b_v1",
        384,
        ("math",),
    ),
)
E37_CELLS = (
    (
        "cde37_outcome_collision_05b_v1",
        384,
        ("multi_answer", "unique_answer"),
    ),
    (
        "gce37_outcome_collision_05b_v1",
        192,
        ("multi_answer", "unique_answer"),
    ),
    (
        "cde38_semantic_shannon_05b_v1",
        384,
        ("multi_answer", "unique_answer"),
    ),
    (
        "gce38_semantic_shannon_05b_v1",
        192,
        ("multi_answer", "unique_answer"),
    ),
    (
        "mte39_math12k_384_semantic_entropy_05b_v1",
        384,
        ("math",),
    ),
    (
        "cde41_semantic_shannon_advantage_05b_v1",
        384,
        ("multi_answer", "unique_answer"),
    ),
    (
        "gce41_semantic_shannon_advantage_05b_v1",
        192,
        ("multi_answer", "unique_answer"),
    ),
    (
        "mte41_math12k_384_semantic_shannon_advantage_05b_v1",
        384,
        ("math",),
    ),
    (
        "cde43_success_conditioned_signed_semantic_shannon_05b_v1",
        384,
        ("multi_answer", "unique_answer"),
    ),
    (
        "gce43_success_conditioned_signed_semantic_shannon_05b_v1",
        192,
        ("multi_answer", "unique_answer"),
    ),
    (
        "mte43_math12k_384_success_conditioned_signed_semantic_shannon_05b_v1",
        384,
        ("math",),
    ),
)
CURRENT_CANONICAL_CELLS = (
    (
        "cde52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "gce52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2",
        192,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "pye52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
)
STAGE_A_CURRENT_CANONICAL_CELLS = (
    (
        "cde52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "gce52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1",
        192,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "pye52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
)
STAGE_A_IDENTITY = (
    ARTIFACTS
    / "e52_direct_inverse_entropy_canonical_05b_stage_a_v1_identity.json"
)
E53_CURRENT_CANONICAL_CELLS = (
    (
        "cde53_verified_replay_05b_50ep_sentinel_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "gce53_verified_replay_05b_50ep_sentinel",
        192,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "pye53_verified_replay_05b_50ep_sentinel_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
)
E53_SENTINEL_IDENTITY = (
    ARTIFACTS / "e53_verified_replay_05b_sentinel_identity.json"
)
E55_CURRENT_CANONICAL_CELLS = (
    (
        "cde55_per_rollout_verified_anchor_05b_50ep_sentinel_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "gce55_per_rollout_verified_anchor_05b_50ep_sentinel",
        192,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "pye55_per_rollout_verified_anchor_05b_50ep_sentinel_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
)
E55_SENTINEL_IDENTITY = (
    ARTIFACTS / "e55_per_rollout_verified_anchor_identity.json"
)
E56_CURRENT_CANONICAL_CELLS = (
    (
        "cde56_open_set_split_canonical_05b_50ep_sentinel_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "gce56_open_set_split_canonical_05b_50ep_sentinel",
        192,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "pye56_open_set_split_canonical_05b_50ep_sentinel_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
)
E56_SENTINEL_IDENTITY = (
    ARTIFACTS / "e56_open_set_split_canonical_05b_sentinel_identity.json"
)
E57_CURRENT_CANONICAL_CELLS = (
    (
        "cde57_verified_first_split_canonical_05b_50ep_sentinel_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "gce57_verified_first_split_canonical_05b_50ep_sentinel",
        192,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "pye57_verified_first_split_canonical_05b_50ep_sentinel_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
)
E57_SENTINEL_IDENTITY = (
    ARTIFACTS
    / "e57_verified_first_split_canonical_05b_sentinel_identity.json"
)
E58_CURRENT_CANONICAL_CELLS = (
    (
        "cde58_global_verified_replay_canonical_05b_50ep_sentinel_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "gce58_global_verified_replay_canonical_05b_50ep_sentinel",
        192,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "pye58_global_verified_replay_canonical_05b_50ep_sentinel_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
)
E58_SENTINEL_IDENTITY = (
    ARTIFACTS
    / "e58_global_verified_replay_canonical_05b_sentinel_identity.json"
)
E59_MATHIR_CELLS = (
    (
        "mie59_mathir_global_verified_replay_05b_50ep",
        384,
        ("multi_answer",),
        50,
    ),
)
E59_MATHIR_IDENTITY = (
    ARTIFACTS / "e59_mathir_global_verified_replay_matched_identity.json"
)
E53_STAGE_A_CURRENT_CANONICAL_CELLS = (
    (
        "cde53_verified_replay_05b_50ep_stage_a_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "gce53_verified_replay_05b_50ep_stage_a",
        192,
        ("multi_answer", "unique_answer"),
        50,
    ),
    (
        "pye53_verified_replay_05b_50ep_stage_a_allcs",
        384,
        ("multi_answer", "unique_answer"),
        50,
    ),
)
E53_STAGE_A_IDENTITY = (
    ARTIFACTS / "e53_verified_replay_05b_stage_a_identity.json"
)
# Kept for import compatibility with older local tooling. The live mode itself
# is current-only and does not render the cancelled fixed E44 ModeBench arms.
E44_OGS_CELLS = CURRENT_CANONICAL_CELLS


def current_canonical_cells() -> tuple[tuple, ...]:
    """Prefer the fresh three-seed cohort once its identity is published."""

    if E59_MATHIR_IDENTITY.is_file():
        # E59 is the separately identified, smoke-gated fourth-domain
        # extension of E58. Keep E58's three original domains frozen and add
        # only E59's fresh matched MathIR control/treatment prefix.
        return (
            E53_CURRENT_CANONICAL_CELLS
            + E58_CURRENT_CANONICAL_CELLS
            + E59_MATHIR_CELLS
        )
    if E58_SENTINEL_IDENTITY.is_file():
        # E58 reuses E53's still-matched seed-9010 Dr.GRPO controls.
        return E53_CURRENT_CANONICAL_CELLS + E58_CURRENT_CANONICAL_CELLS
    if E57_SENTINEL_IDENTITY.is_file():
        # E57 reuses E53's still-matched seed-9010 Dr.GRPO controls.
        return E53_CURRENT_CANONICAL_CELLS + E57_CURRENT_CANONICAL_CELLS
    if E56_SENTINEL_IDENTITY.is_file():
        # E56 reuses E53's still-matched seed-9010 Dr.GRPO controls.
        return E53_CURRENT_CANONICAL_CELLS + E56_CURRENT_CANONICAL_CELLS
    if E55_SENTINEL_IDENTITY.is_file():
        # E55 submitted only fresh treatments. Keep refreshing E53 because
        # those frozen seed-9010 Dr.GRPO rows are E55's preregistered controls.
        return E53_CURRENT_CANONICAL_CELLS + E55_CURRENT_CANONICAL_CELLS
    if E53_STAGE_A_IDENTITY.is_file():
        return E53_STAGE_A_CURRENT_CANONICAL_CELLS
    if E53_SENTINEL_IDENTITY.is_file():
        return E53_CURRENT_CANONICAL_CELLS
    if STAGE_A_IDENTITY.is_file():
        return STAGE_A_CURRENT_CANONICAL_CELLS
    return CURRENT_CANONICAL_CELLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e37-only",
        action="store_true",
        help=(
            "refresh only the retained E37 comparators, E38 Shannon extension, "
            "E39 MATH extension, E41 separately-centered Shannon-advantage "
            "extension, E43 success-conditioned signed-Shannon extension, and "
            "their dedicated three-domain head-to-head figure"
        ),
    )
    parser.add_argument(
        "--current-canonical-only",
        action="store_true",
        help=(
            "refresh only the active canonical ModeBench sentinel or Stage A "
            "cohort and its matched controls"
        ),
    )
    parser.add_argument(
        "--e44-ogs-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--e46-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    current_aliases = sum(
        bool(value)
        for value in (
            args.current_canonical_only,
            args.e44_ogs_only,
            args.e46_only,
        )
    )
    if current_aliases > 1:
        parser.error(
            "--current-canonical-only was provided more than once through "
            "a legacy alias"
        )
    args.current_canonical_only = bool(current_aliases)
    if args.e37_only and args.current_canonical_only:
        parser.error(
            "--e37-only and --current-canonical-only are mutually exclusive"
        )
    return args


def main() -> int:
    args = parse_args()
    if args.current_canonical_only:
        cells = current_canonical_cells()
    elif args.e37_only:
        cells = E37_CELLS
    else:
        cells = CELLS
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[latest-05b-refresh] skipped: another refresh is running")
            return 0
        for cell in cells:
            stamp, pool_size = cell[:2]
            eval_splits = cell[2] if len(cell) >= 3 else ()
            max_training_passes = (
                None
                if args.current_canonical_only
                else cell[3] if len(cell) >= 4 else 10
            )
            command = [
                sys.executable,
                str(PARSER),
                "--stamp-prefix",
                stamp,
                "--run-data-root",
                str(ROOT / "var/data"),
                "--out",
                str(ARTIFACTS / f"{stamp}_scaling_curve.json"),
                "--prompt-pool-size",
                str(pool_size),
                "--num-samples",
                "16",
            ]
            if max_training_passes is not None:
                command.extend(
                    (
                        "--max-training-passes",
                        str(max_training_passes),
                    )
                )
            if args.current_canonical_only:
                jobs_manifest = (
                    ARTIFACTS / f"{stamp}_comparative_jobs.tsv"
                )
                if not jobs_manifest.is_file():
                    raise FileNotFoundError(
                        f"current canonical manifest missing: {jobs_manifest}"
                    )
                command.extend(
                    ("--jobs-manifest", str(jobs_manifest))
                )
            if eval_splits:
                command.extend(("--eval-splits", *eval_splits))
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
            )
        try:
            if args.current_canonical_only:
                from plot_divergence import (
                    render_online_canonical_maxent_05b as render_figures,
                )
            elif args.e37_only:
                from plot_divergence import (
                    render_outcome_collision_05b as render_figures,
                )
            else:
                # E32 feeds the latest-only diagnostic, the maintained 0.5B
                # panel, both task splits, the free-form grid, and the combined
                # divergence grid. Republish every consumer atomically.
                from plot_divergence import main as render_figures
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("matplotlib"):
                raise
            print(
                "[latest-05b-refresh] matplotlib unavailable; curve JSONs "
                "were published and figures were left unchanged"
            )
        else:
            render_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
