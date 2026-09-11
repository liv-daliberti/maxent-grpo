#!/usr/bin/env python3
"""Plot MaxEnt checkpoint dynamics by environment and model scale.

The standard MaxEnt treatments and their matched canonical Dr.GRPO controls
show pass@1, mean@8, pass@8, coverage@8, and distinct@8. Rows are aligned at 0.5B,
3B, and 7B, and every axis uses the standardized ten-pass horizon. Probability
metrics share a [0, 1] y-axis across the full grid; distinct@8 uses one shared
campaign-wide count scale.

Reads the tidy curve JSONs written by parse_scaling_curve.py. Thin lines are
training seeds; heavy lines are seed means. When all three seeds have four
repeated K=8 evaluations, a light band shows a 95% t-confidence interval for
Monte Carlo evaluation uncertainty in the three-seed mean. Every raw draw
remains in the curve artifact. The script writes a combined figure plus
separate Countdown and graph-coloring figures. The 0.5B rows also show E32's
conditional-token Haarnoja-dual sidecar as a distinct policy-space line.
Regenerate with:

  python ops/exp_scaling/plot_divergence.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "paper/figures/compute_divergence"
OUT_BY_ENVIRONMENT = {
    "Countdown": ROOT / "paper/figures/compute_divergence_countdown",
    "Graph coloring": ROOT / "paper/figures/compute_divergence_graph_coloring",
}
LATEST_PREVIEW = ROOT / "var/artifacts/divergence_grid_latest.png"
OUT_BY_PANEL = {
    "on_policy": ROOT / "paper/figures/compute_divergence_canonical_maxent",
    "freeform": ROOT / "paper/figures/compute_divergence_freeform_modebench",
}
LATEST_PREVIEW_BY_PANEL = {
    "on_policy": ROOT / "var/artifacts/divergence_canonical_maxent_latest.png",
    "freeform": ROOT / "var/artifacts/divergence_freeform_modebench_latest.png",
}
OUT_FREEFORM_05B = ROOT / "paper/figures/freeform_conditional_token_05b"
OUT_LATEST_FREEFORM_05B = ROOT / "paper/figures/freeform_05b_latest"
OUT_ANSWER_OPTION_MI_05B = (
    ROOT / "var/artifacts/e36_answer_option_mi_dual_eval_05b_live"
)
OUT_ANSWER_OPTION_BINDING_05B = (
    ROOT / "var/artifacts/e36_answer_option_binding_05b_live"
)
OUT_OUTCOME_COLLISION_05B = (
    ROOT / "paper/figures/e37_outcome_collision_05b_live"
)
OUT_ONLINE_CANONICAL_MAXENT_05B = (
    ROOT
    / "paper/figures/e52_current_canonical_05b_live"
)
OUT_VERIFIED_REPLAY_05B = (
    ROOT
    / "paper/figures/e53_verified_replay_05b_live"
)
OUT_PER_ROLLOUT_VERIFIED_ANCHOR_05B = (
    ROOT
    / "paper/figures/e55_per_rollout_verified_anchor_05b_live"
)
OUT_OPEN_SET_SPLIT_CANONICAL_05B = (
    ROOT / "paper/figures/e56_open_set_split_canonical_05b_live"
)
OUT_VERIFIED_FIRST_SPLIT_CANONICAL_05B = (
    ROOT / "paper/figures/e57_verified_first_split_canonical_05b_live"
)
OUT_GLOBAL_VERIFIED_REPLAY_CANONICAL_05B = (
    ROOT / "paper/figures/e58_global_verified_replay_canonical_05b_live"
)
OUT_MATHIR_GLOBAL_VERIFIED_REPLAY_05B = (
    ROOT / "paper/figures/e59_mathir_global_verified_replay_05b_live"
)
E59_MATHIR_PREFIX = "mie59_mathir_global_verified_replay_05b_50ep"
LEGACY_OUT_ONLINE_CANONICAL_MAXENT_05B = (
    ROOT
    / "paper/figures/e52_direct_inverse_entropy_canonical_05b_live"
)
OUT_FREEFORM_DIAGNOSTIC = (
    ROOT / "var/artifacts/compute_divergence_freeform_modebench_corrected_diagnostic"
)

HUE_MAXENT = "#00876c"
HUE_MAXENT_FEEDBACK = "#7cae00"
HUE_MAXENT_SAC_DUAL = "#cc4778"
HUE_BASE = "#767676"
HUE_FREEFORM_DUAL = "#6A3D9A"
HUE_FREEFORM_BASE = "#3f3f3f"
HUE_ANSWER_OPTION_CONTROL = "#0072B2"
HUE_ANSWER_OPTION_MI = "#D55E00"
HUE_OUTCOME_COLLISION_CONTROL = "#3f3f3f"
HUE_OUTCOME_COLLISION = "#009E73"
HUE_SEMANTIC_SHANNON = "#D55E00"
HUE_SEMANTIC_SHANNON_ADVANTAGE = "#0072B2"
HUE_SUCCESS_CONDITIONED_SIGNED_SHANNON = "#CC0077"
HUE_ONLINE_CANONICAL_CONTROL = "#3f3f3f"
HUE_ONLINE_CANONICAL_MAXENT = "#0072B2"
# Keep the live canonical treatment unmistakable against both the neutral
# control and the light panel grid. The former Okabe-Ito pink was too pale at
# dashboard scale, especially when a treatment overlapped its control.
HUE_ONLINE_CANONICAL_HAARNOJA = "#C21875"
INK = "#1a1a1a"
MAX_TRAINING_EPOCHS = 10.0
T_975_DF3 = 3.182446305284263
METRIC_BOUNDS = {
    "mean8": (0.0, 1.0),
    "pass8": (0.0, 1.0),
    "coverage8": (0.0, 1.0),
    "distinct8": (0.0, 8.0),
}
PROPORTION_METRICS = frozenset({"greedy", "mean8", "pass8", "coverage8"})
PROPORTION_YLIM = (0.0, 1.0)
DISTINCT_TICK_STEP = 0.5
FIGURE_WIDTH_PER_ENVIRONMENT_PANEL = 8.4
FIGURE_HEIGHT_SINGLE_ROW = 4.5
FIGURE_HEIGHT_BASE = 2.5
FIGURE_HEIGHT_PER_ROW = 1.8
OUTER_MARGIN_INCHES = 0.65


def _e16_canonical_lifecycle_note() -> str:
    """Describe the real E16 lifecycle without inventing scheduler state."""

    protocol = ROOT / "paper/preregistration/e16_canonical_maxent_replication.md"
    try:
        status_line = next(
            line for line in protocol.read_text(encoding="utf-8").splitlines()
            if line.startswith("**Status:")
        )
    except (FileNotFoundError, StopIteration):
        return "E16 CANONICAL PROTOCOL UNAVAILABLE"
    if "DESIGN DRAFT" in status_line:
        return "E16 CANONICAL PROTOCOL IN DESIGN"

    artifact_root = ROOT / "var/artifacts"
    full_manifests = (
        artifact_root / "cde16_canonical_maxent_05b_v2_comparative_jobs.tsv",
        artifact_root / "gce16_canonical_maxent_05b_v2_comparative_jobs.tsv",
    )
    if all(path.is_file() for path in full_manifests):
        return "E16 FULL CANONICAL GRID SUBMITTED\nSEE LIVE MONITOR"

    approvals = []
    for path in artifact_root.glob("*e16*smoke*approval*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema") == "e16_canonical_smoke_approval_v1"
            and payload.get("status") == "approved"
            and payload.get("all_six_passed") is True
        ):
            approvals.append(path)
    if approvals:
        return "E16 CANONICAL SMOKE PASSED\nFULL GRID NOT SUBMITTED"

    smoke_manifests = (
        artifact_root
        / "cde16_canonical_maxent_joint_smoke_v3_comparative_jobs.tsv",
        artifact_root
        / "gce16_canonical_maxent_joint_smoke_v3_comparative_jobs.tsv",
    )
    submitted = sum(path.is_file() for path in smoke_manifests)
    if submitted == 2:
        return "E16 CANONICAL SMOKE SUBMITTED\nSEE LIVE MONITOR"
    if submitted == 1:
        return "E16 CANONICAL SMOKE PARTIAL\nCOHORT REMAINS HELD"
    return "E16 CANONICAL SMOKE NOT SUBMITTED"


def _atomic_savefig(fig: plt.Figure, path: Path, **kwargs: object) -> None:
    """Publish a complete figure so the live monitor cannot expose truncation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.tmp{path.suffix}"
    )
    try:
        fig.savefig(temporary, **kwargs)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    """Publish a compatibility alias without exposing a partial image."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}"
    )
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

ENVIRONMENTS = [
    (
        "Countdown",
        [
            (
                "0.5B",
                (
                    "var/artifacts/cde16_canonical_maxent_05b_v2_scaling_curve.json",
                    "var/artifacts/cde19_canonical_drgrpo_05b_v1_scaling_curve.json",
                    "var/artifacts/cde32_freeform_05b_ema_10ep_v4_preemptsafe_scaling_curve.json",
                ),
                384,
                1.5,
                None,
            ),
            (
                "3B",
                (
                    "var/artifacts/cde17_canonical_maxent_3b_v5_scaling_curve.json",
                    "var/artifacts/cde18_canonical_drgrpo_3b_v1_scaling_curve.json",
                    "var/artifacts/cde33_freeform_3b_ema_10ep_v3_a100_scaling_curve.json",
                ),
                384,
                None,
                "NO CHECKPOINT CURVE YET",
            ),
            (
                "7B",
                (
                    "var/artifacts/cde23_canonical_maxent_7b_v6_4xa100_evalsync_fix_scaling_curve.json",
                    "var/artifacts/cde29_freeform_7b_4gpu_v5_buffer_restore_scaling_curve.json",
                ),
                384,
                16,
                "7B STANDARD MAXENT E23 QUEUED\nNO ELIGIBLE OUTCOME YET",
            ),
        ],
    ),
    (
        "Graph coloring",
        [
            (
                "0.5B",
                (
                    "var/artifacts/gce16_canonical_maxent_05b_v2_scaling_curve.json",
                    "var/artifacts/gce19_canonical_drgrpo_05b_v1_scaling_curve.json",
                    "var/artifacts/gce32_freeform_05b_ema_10ep_v5_scaling_curve.json",
                ),
                192,
                3,
                None,
            ),
            (
                "3B",
                (
                    "var/artifacts/gce17_canonical_maxent_3b_v5_scaling_curve.json",
                    "var/artifacts/gce18_canonical_drgrpo_3b_v1_scaling_curve.json",
                    "var/artifacts/gce33_freeform_3b_ema_10ep_v3_a100_scaling_curve.json",
                ),
                192,
                1,
                None,
            ),
            (
                "7B",
                (
                    "var/artifacts/gce24_canonical_maxent_7b_v4_4xa100_evalsync_fix_scaling_curve.json",
                    "var/artifacts/gce29_freeform_7b_4gpu_v5_buffer_restore_scaling_curve.json",
                ),
                192,
                16,
                "7B STANDARD MAXENT E24 QUEUED\nNO ELIGIBLE OUTCOME YET",
            ),
        ],
    ),
]
METRICS = [
    ("greedy", "pass@1"),
    ("mean8", "mean@8"),
    ("pass8", "pass@8"),
    ("coverage8", "coverage@8"),
    ("distinct8", "distinct@8"),
]
ARMS = [
    (
        "canonical_drgrpo",
        "Dr.GRPO (canonical matched control)",
        HUE_BASE,
        (0, (4, 2)),
    ),
    (
        "maxent",
        r"Standard MaxEnt fixed",
        HUE_MAXENT,
        "-",
    ),
    (
        "maxent_control",
        r"Standard MaxEnt proportional",
        HUE_MAXENT_FEEDBACK,
        (0, (1, 1)),
    ),
    (
        "maxent_dual",
        r"Standard MaxEnt Haarnoja dual",
        HUE_MAXENT_SAC_DUAL,
        (0, (5, 1.5, 1, 1.5)),
    ),
    (
        "freeform_drgrpo",
        r"Dr.GRPO (free-form reference)",
        HUE_FREEFORM_BASE,
        (0, (4, 2)),
    ),
    (
        "freeform_maxent_dual",
        r"Free-form conditional-token MaxEnt base-preserving Haarnoja dual",
        HUE_FREEFORM_DUAL,
        (0, (7, 2)),
    ),
]
ARM_BY_KEY = {arm[0]: arm for arm in ARMS}
CANONICAL_PANEL = (
    "on_policy",
    "Canonical-action MaxEnt",
    tuple(
        ARM_BY_KEY[key]
        for key in ("canonical_drgrpo", "maxent", "maxent_control", "maxent_dual")
    ),
)
FREEFORM_PANEL = (
    "freeform",
    "Free-form token-policy MaxEnt (0.5B E32 and 3B E33 matched EMA reruns; 7B E29)",
    tuple(
        ARM_BY_KEY[key]
        for key in ("freeform_drgrpo", "freeform_maxent_dual")
    ),
)
FREEFORM_05B_PANEL = (
    "freeform",
    "E32 matched 0.5B free-form: Dr.GRPO vs EMA-Haarnoja MaxEnt",
    FREEFORM_PANEL[2],
)
LATEST_FREEFORM_05B_PANEL = (
    "latest_freeform",
    "E32 rerun, E36 answer-option MI, and E37/E38/E41/E43 semantic-diversity evaluation",
    (
        ARM_BY_KEY["freeform_drgrpo"],
        ARM_BY_KEY["freeform_maxent_dual"],
        (
            "freeform_answer_option_drgrpo",
            r"E36 matched Dr.GRPO (neutral; 3-seed mean)",
            HUE_ANSWER_OPTION_CONTROL,
            (0, (4, 2)),
        ),
        (
            "freeform_answer_option_mi",
            r"E36 answer-option MI (neutral; 3-seed mean)",
            HUE_ANSWER_OPTION_MI,
            "-",
        ),
        (
            "freeform_outcome_collision_drgrpo",
            r"E37 matched Dr.GRPO (3-seed mean)",
            HUE_OUTCOME_COLLISION_CONTROL,
            (0, (4, 2)),
        ),
        (
            "freeform_outcome_collision",
            r"E37 semantic collision entropy (3-seed mean)",
            HUE_OUTCOME_COLLISION,
            "-",
        ),
        (
            "freeform_semantic_shannon",
            r"E38 predictive semantic Shannon entropy (3-seed mean)",
            HUE_SEMANTIC_SHANNON,
            (0, (6, 1.5, 1.5, 1.5)),
        ),
        (
            "freeform_semantic_shannon_advantage",
            r"E41 separately-centered semantic Shannon advantage (3-seed mean)",
            HUE_SEMANTIC_SHANNON_ADVANTAGE,
            (0, (2, 1.2)),
        ),
        (
            "freeform_success_conditioned_signed_semantic_shannon",
            r"E43 success-conditioned signed Shannon advantage (3-seed mean)",
            HUE_SUCCESS_CONDITIONED_SIGNED_SHANNON,
            "-",
        ),
    ),
)
OUTCOME_COLLISION_05B_ARMS = (
    (
        "freeform_outcome_collision_drgrpo",
        r"Matched Dr.GRPO",
        HUE_OUTCOME_COLLISION_CONTROL,
        (0, (4, 2)),
    ),
    (
        "freeform_outcome_collision",
        r"Semantic collision entropy",
        HUE_OUTCOME_COLLISION,
        "-",
    ),
    (
        "freeform_semantic_shannon",
        r"Predictive semantic Shannon entropy",
        HUE_SEMANTIC_SHANNON,
        (0, (6, 1.5, 1.5, 1.5)),
    ),
    (
        "freeform_semantic_shannon_advantage",
        r"Separately-centered semantic Shannon advantage (E41)",
        HUE_SEMANTIC_SHANNON_ADVANTAGE,
        (0, (2, 1.2)),
    ),
    (
        "freeform_success_conditioned_signed_semantic_shannon",
        r"success-conditioned signed Shannon advantage (E43)",
        HUE_SUCCESS_CONDITIONED_SIGNED_SHANNON,
        "-",
    ),
)
OUTCOME_COLLISION_DIAGNOSTIC_METRICS = (
    ("greedy", "neutral pass@1"),
    ("mean8", "neutral mean@8"),
    ("pass8", "neutral pass@8"),
    ("coverage8", "valid coverage@8"),
    ("distinct8", "mean # distinct correct@8"),
    ("outcome_collision_rate", "train outcome collision"),
    (
        "semantic_shannon_normalized_surprisal_mean",
        "train normalized surprise",
    ),
    (
        "semantic_shannon_separate_semantic_advantage_rms",
        r"train RMS $A_{\rm semantic}$",
    ),
    ("mechanism_parseable_fraction", "train parseable fraction"),
    (
        "semantic_shannon_success_conditioned_signed_effective_advantage_rms",
        r"E43 train RMS applied signed $A_{\rm semantic}$",
    ),
    (
        "semantic_shannon_success_conditioned_signed_eligible_fraction",
        "E43 eligible success fraction",
    ),
    (
        "semantic_shannon_success_conditioned_signed_effective_advantage_positive_fraction",
        "E43 applied positive fraction",
    ),
    (
        "semantic_shannon_success_conditioned_signed_effective_advantage_negative_fraction",
        "E43 applied negative fraction",
    ),
    (
        "semantic_shannon_success_conditioned_signed_cap_fraction",
        "E43 fraction capped (|cap|=0.05)",
    ),
)
ONLINE_CANONICAL_MAXENT_05B_ARMS = (
    (
        "grpo",
        r"Matched Dr.GRPO",
        HUE_ONLINE_CANONICAL_CONTROL,
        (0, (4, 2)),
    ),
    (
        "online_canonical_maxent",
        r"E45 MathIR fixed canonical MaxEnt",
        HUE_ONLINE_CANONICAL_MAXENT,
        "-",
    ),
    (
        "maxent_inverse",
        r"Direct inverse conditional entropy",
        HUE_ONLINE_CANONICAL_MAXENT,
        (0, (4, 2)),
    ),
    (
        "maxent_inverse_canonical",
        r"Direct inverse entropy + fixed canonical",
        HUE_ONLINE_CANONICAL_HAARNOJA,
        "-",
    ),
    (
        "maxent_inverse_canonical_replay",
        r"E53 inverse entropy + verified replay",
        HUE_ONLINE_CANONICAL_HAARNOJA,
        "-",
    ),
    (
        "verified_first_split_canonical",
        r"E57 verified-first split canonical",
        HUE_ONLINE_CANONICAL_HAARNOJA,
        "-",
    ),
    (
        "verified_first_global_replay_canonical",
        r"E58 global verified replay",
        HUE_ONLINE_CANONICAL_HAARNOJA,
        "-",
    ),
    (
        "verified_first_bootstrap_local_canonical",
        r"E60 finite global bootstrap $\rightarrow$ prompt-local replay",
        HUE_ONLINE_CANONICAL_HAARNOJA,
        "-",
    ),
)
ONLINE_CANONICAL_ADVANTAGE_COMPONENTS = (
    (
        "online_canonical_entropy_advantage_rms",
        r"$A_{\rm entropy}$",
        "#56B4E9",
        (0, (4, 2)),
    ),
    (
        "online_canonical_novelty_advantage_rms",
        r"$A_{\rm new}$",
        "#009E73",
        (0, (1, 1)),
    ),
    (
        "online_canonical_combined_advantage_rms",
        r"combined",
        HUE_ONLINE_CANONICAL_HAARNOJA,
        "-",
    ),
)
ONLINE_CANONICAL_DIAGNOSTIC_METRICS = (
    ("greedy", "neutral pass@1", (0.0, 1.0), False),
    ("mean8", "neutral mean@8", (0.0, 1.0), False),
    ("pass8", "neutral pass@8", (0.0, 1.0), False),
    ("coverage8", "valid coverage@8", (0.0, 1.0), False),
    ("distinct8", "mean # distinct correct@8", (0.0, 8.0), True),
    (
        "online_canonical_normalized_entropy_ratio_mean",
        r"post-update $H(q_x)/\log |B_x^+|$",
        (0.0, 1.0),
        False,
    ),
    (
        "online_canonical_entropy_alpha_used",
        r"canonical entropy coefficient $\alpha$",
        (0.08, 0.52),
        False,
    ),
    (
        "online_canonical_advantage_rms_components",
        "train exploration advantage RMS",
        None,
        False,
    ),
    (
        "online_canonical_exploration_to_task_rms_ratio",
        r"exploration RMS / task RMS",
        None,
        False,
    ),
    (
        "online_canonical_eligible_fraction",
        "verified eligible fraction",
        (0.0, 1.0),
        False,
    ),
    (
        "online_canonical_mean_support_per_prompt",
        r"mean verified support per prompt",
        None,
        False,
    ),
    (
        "online_canonical_tracked_outcomes",
        "cumulative verified discoveries",
        None,
        True,
    ),
)
ANSWER_OPTION_MI_05B_PANEL = (
    "answer_option_mi",
    "E36 matched 0.5B neutral quality: Dr.GRPO vs answer-option MI",
    (
        (
            "freeform_answer_option_drgrpo",
            r"Matched Dr.GRPO (neutral prompt)",
            HUE_ANSWER_OPTION_CONTROL,
            (0, (4, 2)),
        ),
        (
            "freeform_answer_option_mi",
            r"Answer-option MI policy (neutral prompt)",
            HUE_ANSWER_OPTION_MI,
            "-",
        ),
    ),
)
METHOD_PANELS = (
    CANONICAL_PANEL,
    FREEFORM_PANEL,
)
PUBLISHED_METHOD_PANELS = (CANONICAL_PANEL, FREEFORM_PANEL)

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.7,
    }
)


def paths_in(path: str | tuple[str, ...]) -> tuple[str, ...]:
    return (path,) if isinstance(path, str) else path


def load_series(
    path: str | tuple[str, ...],
    steps_per_epoch: int,
    split: str = "multi_answer",
    *,
    max_training_epochs: float | None = MAX_TRAINING_EPOCHS,
):
    """Return arm -> seed -> [(epoch, metric row)] for one eval split."""
    series = defaultdict(lambda: defaultdict(list))
    points = {}
    for item in paths_in(path):
        try:
            with (ROOT / item).open() as handle:
                rows = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            if row["split"] != split:
                continue
            # `global_step / pool_rows` only measures passes when each step
            # consumes one prompt. The two-GPU campaigns consume two prompts
            # per step, so use rollout-normalized prompt consumption whenever
            # the parser has it. Keep the old conversion for legacy artifacts.
            epoch = row.get("training_passes")
            if epoch is None:
                epoch = row["step"] / steps_per_epoch
            if (
                max_training_epochs is not None
                and epoch > float(max_training_epochs) + 1e-9
            ):
                continue
            # A later extension can repeat the first few passes of an earlier
            # run. Keep one observation per arm/seed/pass (later configured
            # sources win) so seed means do not double-weight those reruns.
            arm = row["arm"]
            if (
                "e22_freeform_conditional_dual_05b_v2" in item
                and arm == "maxent_dual"
            ):
                continue
            if "_canonical_drgrpo_" in item and arm == "grpo":
                arm = "canonical_drgrpo"
            freeform_cohort = "_freeform_" in item
            outcome_collision_cohort = "_outcome_collision_" in item
            semantic_shannon_cohort = "_semantic_shannon_" in item
            semantic_shannon_advantage_cohort = (
                "_semantic_shannon_advantage_" in item
            )
            success_conditioned_signed_semantic_shannon_cohort = (
                "_success_conditioned_signed_semantic_shannon_" in item
            )
            e39_semantic_entropy_cohort = (
                "mte39_math12k_384_semantic_entropy_05b_v1" in item
            )
            if (
                outcome_collision_cohort or e39_semantic_entropy_cohort
            ) and arm == "grpo":
                arm = "freeform_outcome_collision_drgrpo"
            elif (
                outcome_collision_cohort or e39_semantic_entropy_cohort
            ) and arm == "outcome_collision":
                arm = "freeform_outcome_collision"
            elif (
                semantic_shannon_cohort or e39_semantic_entropy_cohort
            ) and arm == "semantic_shannon":
                arm = "freeform_semantic_shannon"
            elif (
                semantic_shannon_advantage_cohort
                and arm == "semantic_shannon_advantage"
            ):
                arm = "freeform_semantic_shannon_advantage"
            elif (
                success_conditioned_signed_semantic_shannon_cohort
                and arm == "success_conditioned_signed_semantic_shannon"
            ):
                arm = "freeform_success_conditioned_signed_semantic_shannon"
            if freeform_cohort and arm == "maxent_dual":
                arm = "freeform_maxent_dual"
            if "_answer_option_mi_" in item and arm == "grpo":
                arm = "freeform_answer_option_drgrpo"
            elif freeform_cohort and arm == "grpo":
                arm = "freeform_drgrpo"
            if "_answer_option_mi_" in item and arm == "diayn":
                arm = "freeform_answer_option_mi"
            if "_freeform_drgrpo_" in item and arm == "grpo":
                arm = "freeform_drgrpo"
            points[(arm, row["seed"], epoch)] = row
    for (arm, seed, epoch), row in points.items():
        series[arm][seed].append((epoch, row))
    for arm in series:
        for seed in series[arm]:
            series[arm][seed].sort()
    return series


def seed_mean(per_seed, metric: str, required_seed_count: int = 3):
    """Return means only where the required number of seeds reached an epoch."""
    if required_seed_count <= 0:
        raise ValueError("required_seed_count must be positive")
    by_epoch = defaultdict(list)
    for points in per_seed.values():
        for epoch, row in points:
            if row[metric] is not None:
                by_epoch[epoch].append(row[metric])
    epochs = sorted(
        epoch
        for epoch, values in by_epoch.items()
        if len(values) >= required_seed_count
    )
    return epochs, [sum(by_epoch[epoch]) / len(by_epoch[epoch]) for epoch in epochs]


def seed_mean_eval_ci(per_seed, metric: str, required_seed_count: int = 3):
    """95% t-CI over four fixed evaluation draws of the all-seed mean.

    Training seeds are first averaged within each fixed evaluation draw. The
    four resulting draw-level means are the observations for the interval.
    This isolates K=8 Monte Carlo evaluation uncertainty; it is deliberately
    not presented as uncertainty over the population of training seeds.
    """

    if required_seed_count <= 0:
        raise ValueError("required_seed_count must be positive")
    rows_by_epoch = defaultdict(list)
    for points in per_seed.values():
        for epoch, row in points:
            if row[metric] is not None:
                rows_by_epoch[epoch].append(row)

    epochs = []
    lower = []
    upper = []
    metric_lower, metric_upper = METRIC_BOUNDS.get(
        metric,
        (-math.inf, math.inf),
    )
    for epoch in sorted(rows_by_epoch):
        rows = rows_by_epoch[epoch]
        if len(rows) < required_seed_count:
            continue
        draws = [row.get(f"{metric}_draws") or [] for row in rows]
        if not draws or {len(values) for values in draws} != {4}:
            continue
        draw_means = [
            statistics.fmean(values[index] for values in draws)
            for index in range(4)
        ]
        center = statistics.fmean(draw_means)
        se = statistics.stdev(draw_means) / math.sqrt(len(draw_means))
        margin = T_975_DF3 * se
        epochs.append(epoch)
        lower.append(max(metric_lower, center - margin))
        upper.append(min(metric_upper, center + margin))
    return epochs, lower, upper


def step_count(path: str | tuple[str, ...] | None) -> int:
    if path is None:
        return 0
    counts = []
    for item in paths_in(path):
        try:
            with (ROOT / item).open() as handle:
                rows = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        counts.append(len({row["step"] for row in rows}))
    return max(counts, default=0)


def shared_distinct_ylim() -> tuple[float, float]:
    """Return one readable distinct@8 scale for every published grid.

    The upper limit is derived from all configured environments, scales, and
    method panels—not from an individual facet. Round upward to a half mode
    and retain a little headroom, while respecting distinct@8's [0, 8]
    mathematical support.
    """

    observed_max = 0.0
    for _environment, scale_rows in ENVIRONMENTS:
        for _scale, path, steps_per_epoch, _budget, _empty_note in scale_rows:
            series = load_series(path, steps_per_epoch)
            for per_seed in series.values():
                for points in per_seed.values():
                    observed_max = max(
                        observed_max,
                        *(float(row["distinct8"]) for _epoch, row in points
                          if row["distinct8"] is not None),
                    )
    padded_max = observed_max * 1.05
    rounded_max = (
        math.ceil(padded_max / DISTINCT_TICK_STEP) * DISTINCT_TICK_STEP
    )
    return (0.0, min(8.0, max(1.0, rounded_max)))


def draw_placeholder(ax, message: str | None) -> None:
    if message:
        ax.text(
            0.5,
            0.5,
            message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#777777",
            fontsize=7.7,
        )
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def draw_curve(
    ax,
    series,
    metric: str,
    budget: float | None,
    arms,
    *,
    mean_seed_count: int = 3,
    mean_seed_count_by_arm: dict[str, int] | None = None,
    x_max: float = MAX_TRAINING_EPOCHS,
) -> None:
    if budget is not None and budget < MAX_TRAINING_EPOCHS * 0.98:
        ax.axvline(
            budget,
            color="#999999",
            lw=0.7,
            ls=(0, (1.5, 2)),
            zorder=1,
        )
    for arm, _label, hue, line_style in arms:
        per_seed = series.get(arm, {})
        required_seed_count = (
            mean_seed_count_by_arm.get(arm, mean_seed_count)
            if mean_seed_count_by_arm is not None
            else mean_seed_count
        )
        for points in per_seed.values():
            epochs = [epoch for epoch, row in points if row[metric] is not None]
            values = [row[metric] for _epoch, row in points if row[metric] is not None]
            sparse_seed = len(epochs) <= 3
            ax.plot(
                epochs,
                values,
                color=hue,
                lw=0.8 if sparse_seed else 0.55,
                alpha=0.6 if sparse_seed else 0.3,
                marker="o" if sparse_seed else None,
                ms=2.3,
                zorder=2,
            )
        ci_epochs, ci_lower, ci_upper = seed_mean_eval_ci(
            per_seed,
            metric,
            required_seed_count=required_seed_count,
        )
        if len(ci_epochs) == 1:
            ax.vlines(
                ci_epochs,
                ci_lower,
                ci_upper,
                color=hue,
                lw=3.0,
                alpha=0.16,
                zorder=2.5,
            )
        elif ci_epochs:
            ax.fill_between(
                ci_epochs,
                ci_lower,
                ci_upper,
                color=hue,
                alpha=0.13,
                linewidth=0,
                zorder=2.5,
            )
        epochs, values = seed_mean(
            per_seed,
            metric,
            required_seed_count=required_seed_count,
        )
        if epochs:
            sparse = len(epochs) <= 25
            ax.plot(
                epochs,
                values,
                color=hue,
                lw=1.65,
                ls=line_style,
                zorder=4,
                marker="o" if sparse else None,
                ms=2.5,
            )
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.7)


def render_figure(
    environments,
    out: Path,
    latest_preview: Path | None = None,
    method_panels=METHOD_PANELS,
    *,
    mean_seed_count: int = 3,
    mean_seed_count_by_arm: dict[str, int] | None = None,
    x_max_by_scale: dict[str, float] | None = None,
    diagnostic_note: str | None = (
        "Heavy lines: three-seed means; thin lines: unsmoothed seed traces. "
        "Light bands: 95% t-CI over four fixed K=8 evaluation draws after "
        "averaging across three seeds. Pass@1 is deterministic greedy."
    ),
) -> None:
    environment_count = len(environments)
    panel_count = len(method_panels)
    row_count = max(len(scale_rows) for _environment, scale_rows in environments)
    columns_per_environment = panel_count * len(METRICS)
    column_count = environment_count * columns_per_environment
    figure_width = (
        FIGURE_WIDTH_PER_ENVIRONMENT_PANEL * environment_count * panel_count
    )
    figure_height = (
        FIGURE_HEIGHT_SINGLE_ROW
        if row_count == 1
        else FIGURE_HEIGHT_BASE + FIGURE_HEIGHT_PER_ROW * row_count
    )
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )

    for environment_index, (_environment, scale_rows) in enumerate(environments):
        for row_index, (scale, path, steps_per_epoch, budget, empty_note) in enumerate(
            scale_rows
        ):
            landed_steps = step_count(path)
            live = landed_steps > 0
            series = load_series(path, steps_per_epoch) if live else None
            for panel_index, (panel_key, _panel_label, panel_arms) in enumerate(
                method_panels
            ):
                panel_eligible = panel_key != "freeform" or scale in {
                    "0.5B",
                    "3B",
                    "7B",
                }
                panel_live = (
                    panel_eligible
                    and live
                    and any(arm in series for arm, *_ in panel_arms)
                )
                column_offset = (
                    environment_index * columns_per_environment
                    + panel_index * len(METRICS)
                )
                for metric_index, (metric, metric_label) in enumerate(METRICS):
                    ax = axes[row_index][column_offset + metric_index]
                    if row_index == 0:
                        ax.set_title(metric_label, loc="left", fontsize=8.2)
                    if metric_index == 0:
                        ax.set_ylabel(scale, fontsize=8.2)
                    if panel_live:
                        draw_curve(
                            ax,
                            series,
                            metric,
                            budget,
                            panel_arms,
                            mean_seed_count=mean_seed_count,
                            mean_seed_count_by_arm=mean_seed_count_by_arm,
                            x_max=(
                                x_max_by_scale.get(scale, MAX_TRAINING_EPOCHS)
                                if x_max_by_scale is not None
                                else MAX_TRAINING_EPOCHS
                            ),
                        )
                        if (
                            panel_key == "freeform"
                            and metric_index == 1
                            and "freeform_maxent_dual" not in series
                        ):
                            # The historical E22 Dr.GRPO reference can make the
                            # panel live before the active E27/E25 treatment has
                            # produced an evaluation. Make that asymmetry
                            # explicit instead of letting the missing treatment
                            # look like an old/cohort-routing plotting bug.
                            cohort = {
                                "0.5B": "E32",
                                "3B": "E25-v2",
                                "7B": "E29",
                            }[scale]
                            ax.text(
                                0.5,
                                0.88,
                                f"{cohort} 125% TARGET\nPENDING FIRST EVALUATION",
                                transform=ax.transAxes,
                                ha="center",
                                va="top",
                                color=HUE_FREEFORM_DUAL,
                                fontsize=6.4,
                                bbox={
                                    "facecolor": "white",
                                    "edgecolor": "none",
                                    "alpha": 0.82,
                                    "pad": 0.8,
                                },
                            )
                        if (
                            panel_key == "on_policy"
                            and metric_index == 1
                            and not any(
                                arm in series
                                for arm in ("maxent", "maxent_control", "maxent_dual")
                            )
                        ):
                            gate_note = (
                                _e16_canonical_lifecycle_note()
                                if scale == "0.5B"
                                else (
                                    "3B STANDARD MAXENT E17 REGISTERED\n"
                                    "NO ELIGIBLE OUTCOME"
                                    if scale == "3B"
                                    else "7B STANDARD MAXENT QUEUED\nNO ELIGIBLE OUTCOME YET"
                                )
                            )
                            ax.text(
                                0.97,
                                0.96,
                                gate_note,
                                transform=ax.transAxes,
                                ha="right",
                                va="top",
                                color=HUE_MAXENT,
                                fontsize=5.5,
                                bbox={
                                    "facecolor": "white",
                                    "edgecolor": "none",
                                    "alpha": 0.78,
                                    "pad": 0.8,
                                },
                            )
                        if landed_steps == 1:
                            if metric_index == 1:
                                ax.text(
                                    0.5,
                                    0.88,
                                    "INITIAL EVAL ONLY",
                                    transform=ax.transAxes,
                                    ha="center",
                                    va="top",
                                    color="#777777",
                                    fontsize=7.3,
                                )
                    else:
                        if panel_key == "freeform" and metric_index == 1:
                            message = (
                                "E32 MATCHED RERUN\nPENDING FIRST EVALUATION"
                                if scale == "0.5B"
                                else (
                                    "E25-v2 125% TARGET\nPENDING FIRST EVALUATION"
                                    if scale == "3B"
                                    else "E29 125% TARGET\nPENDING FIRST EVALUATION"
                                )
                            )
                        else:
                            message = empty_note if metric_index == 1 else None
                        draw_placeholder(ax, message)

    # Apply scales by metric across every environment, model scale, and method
    # panel. Facet-specific autoscaling can otherwise make equal-sized changes
    # look different. Probability metrics use their common mathematical
    # support; distinct@8 uses one count scale derived from the full campaign.
    distinct_ylim = shared_distinct_ylim()
    for environment_index in range(environment_count):
        environment_offset = environment_index * columns_per_environment
        for panel_index in range(panel_count):
            panel_offset = environment_offset + panel_index * len(METRICS)
            for metric_index, (metric, _metric_label) in enumerate(METRICS):
                ylim = (
                    PROPORTION_YLIM
                    if metric in PROPORTION_METRICS
                    else distinct_ylim
                )
                for row_index in range(row_count):
                    axes[row_index][panel_offset + metric_index].set_ylim(*ylim)

    for environment_index, (environment, _scale_rows) in enumerate(environments):
        center = (environment_index + 0.5) / environment_count
        fig.text(
            center,
            0.905,
            environment,
            ha="center",
            fontsize=10,
            weight="bold",
        )
        for panel_index, (_key, panel_label, _arms) in enumerate(method_panels):
            panel_center = (
                environment_index * panel_count + panel_index + 0.5
            ) / (environment_count * panel_count)
            fig.text(
                panel_center,
                0.855,
                panel_label,
                ha="center",
                fontsize=8.5,
                weight="bold",
                color=INK,
            )
        for divider_index in range(1, panel_count):
            panel_divider = (
                environment_index * panel_count + divider_index
            ) / (environment_count * panel_count)
            fig.add_artist(
                Line2D(
                    [panel_divider, panel_divider],
                    [0.07, 0.84],
                    transform=fig.transFigure,
                    color="#d0d0d0",
                    lw=0.65,
                    ls=(0, (2, 2)),
                )
            )
    for environment_index in range(1, environment_count):
        divider = environment_index / environment_count
        fig.add_artist(
            Line2D(
                [divider, divider],
                [0.07, 0.89],
                transform=fig.transFigure,
                color="#bbbbbb",
                lw=0.7,
            )
        )
    legend_arms = ARMS if method_panels == METHOD_PANELS else method_panels[0][2]
    legend_handles = [
        Line2D([], [], color=hue, lw=1.8, ls=line_style, label=label)
        for _arm, label, hue, line_style in legend_arms
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=7.2 if environment_count > 1 else 6.8,
        ncol=len(legend_arms) if environment_count > 1 else min(3, len(legend_arms)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        handlelength=2.0 if environment_count > 1 else 1.7,
    )
    fig.supxlabel(
        "training passes over the prompt pool", fontsize=8.2, y=0.015
    )
    if diagnostic_note is not None:
        fig.text(
            0.5,
            0.075 if row_count == 1 else 0.047,
            diagnostic_note,
            ha="center",
            va="center",
            fontsize=7.2,
            color="#555555",
        )
    # ``tight_layout`` overreacts to the dense grid's repeated tick labels and
    # annotations. Reserve a real physical margin so labels do not get clipped
    # as the number of environments and method panels changes.
    outer_margin = OUTER_MARGIN_INCHES / figure_width
    fig.subplots_adjust(
        left=outer_margin,
        right=1.0 - outer_margin,
        bottom=0.18 if row_count == 1 else 0.09,
        top=0.70 if row_count == 1 else 0.81,
        wspace=0.48,
        hspace=0.28,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savefig(fig, out.with_suffix(".pdf"))
    _atomic_savefig(fig, out.with_suffix(".png"), dpi=200)
    if latest_preview is not None:
        latest_preview.parent.mkdir(parents=True, exist_ok=True)
        _atomic_savefig(fig, latest_preview, dpi=200)
        print(f"wrote {out}.pdf/.png and {latest_preview}")
    else:
        print(f"wrote {out}.pdf/.png")
    plt.close(fig)


def render_freeform_diagnostic() -> None:
    diagnostic_environments = [
        (
            environment,
            [row for row in scale_rows if row[0] in {"0.5B", "3B", "7B"}],
        )
        for environment, scale_rows in ENVIRONMENTS
    ]
    render_figure(
        diagnostic_environments,
        OUT_FREEFORM_DIAGNOSTIC,
        method_panels=(FREEFORM_PANEL,),
        mean_seed_count=3,
        diagnostic_note=(
            "Diagnostic: heavy lines are means only where all three seeds are "
            "present; thin lines are unsmoothed seed trajectories; light bands "
            "are 95% t-CIs over the four fixed K=8 evaluation draws."
        ),
    )


def render_latest_freeform_05b() -> None:
    """Render E32, E36, and the active E37/E38/E41/E43 head-to-head."""

    environments = [
        (
            "Countdown",
            [
                (
                    "E32 · 3 seeds",
                    "var/artifacts/cde32_freeform_05b_ema_10ep_v4_preemptsafe_scaling_curve.json",
                    384,
                    None,
                    "E32 AWAITING FIRST EVALUATION",
                ),
                (
                    "E36 · 3 seeds",
                    "var/artifacts/cde36_answer_option_mi_dual_eval_05b_v1_scaling_curve.json",
                    384,
                    10.0,
                    "E36 AWAITING FIRST EVALUATION",
                ),
                (
                    "E37/E38/E41/E43 · 3 seeds",
                    (
                        "var/artifacts/cde37_outcome_collision_05b_v1_scaling_curve.json",
                        "var/artifacts/cde38_semantic_shannon_05b_v1_scaling_curve.json",
                        "var/artifacts/cde41_semantic_shannon_advantage_05b_v1_scaling_curve.json",
                        "var/artifacts/cde43_success_conditioned_signed_semantic_shannon_05b_v1_scaling_curve.json",
                    ),
                    384,
                    10.0,
                    "E37/E38/E41/E43 AWAITING FIRST EVALUATION",
                ),
            ],
        ),
        (
            "Graph coloring",
            [
                (
                    "E32 · 3 seeds",
                    "var/artifacts/gce32_freeform_05b_ema_10ep_v5_scaling_curve.json",
                    192,
                    None,
                    "E32 AWAITING FIRST EVALUATION",
                ),
                (
                    "E36 · 3 seeds",
                    "var/artifacts/gce36_answer_option_mi_dual_eval_05b_v1_scaling_curve.json",
                    192,
                    10.0,
                    "E36 AWAITING FIRST EVALUATION",
                ),
                (
                    "E37/E38/E41/E43 · 3 seeds",
                    (
                        "var/artifacts/gce37_outcome_collision_05b_v1_scaling_curve.json",
                        "var/artifacts/gce38_semantic_shannon_05b_v1_scaling_curve.json",
                        "var/artifacts/gce41_semantic_shannon_advantage_05b_v1_scaling_curve.json",
                        "var/artifacts/gce43_success_conditioned_signed_semantic_shannon_05b_v1_scaling_curve.json",
                    ),
                    192,
                    10.0,
                    "E37/E38/E41/E43 AWAITING FIRST EVALUATION",
                ),
            ],
        ),
    ]
    render_figure(
        environments,
        OUT_LATEST_FREEFORM_05B,
        ROOT / "var/artifacts/freeform_05b_latest.png",
        method_panels=(LATEST_FREEFORM_05B_PANEL,),
        mean_seed_count=3,
        diagnostic_note=(
            "Rows: E32 EMA rerun, E36 neutral-prompt MI evaluation, and fresh "
            "E37 comparators with the E38 predictive-Shannon and E41 "
            "separately-centered Shannon-advantage extensions plus E43 "
            "success-conditioned signed Shannon advantage. Heavy lines require all three "
            "seeds; light bands show K=8 draw uncertainty."
        ),
    )


def render_answer_option_mi_05b() -> None:
    """Render E36's neutral-quality comparison separately from E32."""

    environments = [
        (
            "Countdown",
            [
                (
                    "0.5B",
                    "var/artifacts/cde36_answer_option_mi_dual_eval_05b_v1_scaling_curve.json",
                    384,
                    10.0,
                    "E36 AWAITING FIRST EVALUATION",
                )
            ],
        ),
        (
            "Graph coloring",
            [
                (
                    "0.5B",
                    "var/artifacts/gce36_answer_option_mi_dual_eval_05b_v1_scaling_curve.json",
                    192,
                    10.0,
                    "E36 AWAITING FIRST EVALUATION",
                )
            ],
        ),
    ]
    render_figure(
        environments,
        OUT_ANSWER_OPTION_MI_05B,
        method_panels=(ANSWER_OPTION_MI_05B_PANEL,),
        mean_seed_count=3,
        diagnostic_note=(
            "E36 neutral quality: heavy lines require all three seeds; thin lines are seed "
            "trajectories. Light bands are 95% t-CIs over four fixed K=8 "
            "evaluation draws, not training-seed uncertainty."
        ),
    )


def _mechanism_metric_value(row: dict, metric: str):
    """Read active semantic-entropy telemetry across logger conventions."""

    value = row.get(metric)
    if value is not None:
        return value
    if (
        metric
        == "semantic_shannon_success_conditioned_signed_cap_fraction"
    ):
        positive = row.get(
            "semantic_shannon_success_conditioned_signed_positive_cap_fraction"
        )
        negative = row.get(
            "semantic_shannon_success_conditioned_signed_negative_cap_fraction"
        )
        if positive is not None or negative is not None:
            return float(positive or 0.0) + float(negative or 0.0)
    if metric == "mechanism_parseable_fraction":
        signed_parseable = row.get(
            "semantic_shannon_success_conditioned_signed_parseable_fraction"
        )
        if signed_parseable is not None:
            return signed_parseable
        for prefix in ("outcome_collision", "semantic_shannon"):
            value = row.get(f"{prefix}_parseable_fraction")
            if value is not None:
                return value
            invalid = row.get(f"{prefix}_invalid_fraction")
            if invalid is not None:
                return 1.0 - float(invalid)
    if metric == "outcome_collision_parseable_fraction":
        invalid = row.get("outcome_collision_invalid_fraction")
        if invalid is not None:
            return 1.0 - float(invalid)
    return None


def _outcome_collision_metric_value(row: dict, metric: str):
    """Backward-compatible E37 telemetry reader used by focused tests."""

    return _mechanism_metric_value(row, metric)


def _outcome_collision_seed_mean(per_seed, metric: str):
    """All-three-seed mean for an optional mechanism metric."""

    by_epoch = defaultdict(list)
    for points in per_seed.values():
        for epoch, row in points:
            value = _mechanism_metric_value(row, metric)
            if value is not None:
                by_epoch[epoch].append(float(value))
    epochs = [
        epoch
        for epoch in sorted(by_epoch)
        if len(by_epoch[epoch]) >= 3
    ]
    return epochs, [statistics.fmean(by_epoch[epoch]) for epoch in epochs]


def render_outcome_collision_05b() -> None:
    """Render E37/E38/E39/E41/E43 quality, breadth, and diagnostics."""

    environment_paths = (
        (
            "Countdown",
            (
                "var/artifacts/cde37_outcome_collision_05b_v1_scaling_curve.json",
                "var/artifacts/cde38_semantic_shannon_05b_v1_scaling_curve.json",
                "var/artifacts/cde41_semantic_shannon_advantage_05b_v1_scaling_curve.json",
                "var/artifacts/cde43_success_conditioned_signed_semantic_shannon_05b_v1_scaling_curve.json",
            ),
            384,
            "multi_answer",
        ),
        (
            "Graph coloring",
            (
                "var/artifacts/gce37_outcome_collision_05b_v1_scaling_curve.json",
                "var/artifacts/gce38_semantic_shannon_05b_v1_scaling_curve.json",
                "var/artifacts/gce41_semantic_shannon_advantage_05b_v1_scaling_curve.json",
                "var/artifacts/gce43_success_conditioned_signed_semantic_shannon_05b_v1_scaling_curve.json",
            ),
            192,
            "multi_answer",
        ),
        (
            "MATH-500 (train MATH12K-384)",
            (
                "var/artifacts/mte39_math12k_384_semantic_entropy_05b_v1_scaling_curve.json",
                "var/artifacts/mte41_math12k_384_semantic_shannon_advantage_05b_v1_scaling_curve.json",
                "var/artifacts/mte43_math12k_384_success_conditioned_signed_semantic_shannon_05b_v1_scaling_curve.json",
            ),
            384,
            "math",
        ),
    )
    fig, axes = plt.subplots(
        len(environment_paths),
        len(OUTCOME_COLLISION_DIAGNOSTIC_METRICS),
        figsize=(29.8, 7.8),
        squeeze=False,
    )
    for row_index, (environment, path, steps_per_epoch, eval_split) in enumerate(
        environment_paths
    ):
        series = load_series(path, steps_per_epoch, split=eval_split)
        has_any_evaluation = any(
            points
            for per_seed in series.values()
            for points in per_seed.values()
        )
        has_shannon_evaluation = any(
            points
            for points in series.get("freeform_semantic_shannon", {}).values()
        )
        has_e41_evaluation = any(
            points
            for points in series.get(
                "freeform_semantic_shannon_advantage",
                {},
            ).values()
        )
        has_e43_evaluation = any(
            points
            for points in series.get(
                "freeform_success_conditioned_signed_semantic_shannon",
                {},
            ).values()
        )
        for column_index, (metric, label) in enumerate(
            OUTCOME_COLLISION_DIAGNOSTIC_METRICS
        ):
            ax = axes[row_index][column_index]
            single_answer_coverage_na = (
                eval_split == "math" and metric == "coverage8"
            )
            metric_is_live = False
            if not single_answer_coverage_na:
                for (
                    arm,
                    _arm_label,
                    hue,
                    line_style,
                ) in OUTCOME_COLLISION_05B_ARMS:
                    is_e43 = (
                        arm
                        == "freeform_success_conditioned_signed_semantic_shannon"
                    )
                    per_seed = series.get(arm, {})
                    for points in per_seed.values():
                        values = [
                            (
                                epoch,
                                _mechanism_metric_value(row, metric),
                            )
                            for epoch, row in points
                        ]
                        values = [
                            (epoch, float(value))
                            for epoch, value in values
                            if value is not None
                        ]
                        if not values:
                            continue
                        metric_is_live = True
                        ax.plot(
                            [epoch for epoch, _value in values],
                            [value for _epoch, value in values],
                            color=hue,
                            alpha=0.52 if is_e43 else 0.30,
                            lw=1.0 if is_e43 else 0.65,
                            marker="D" if is_e43 else "o",
                            markersize=2.9 if is_e43 else 1.8,
                            markeredgecolor="white" if is_e43 else hue,
                            markeredgewidth=0.35 if is_e43 else 0.0,
                            zorder=6 if is_e43 else 2,
                        )
                    mean_epochs, mean_values = _outcome_collision_seed_mean(
                        per_seed,
                        metric,
                    )
                    if mean_epochs:
                        ax.plot(
                            mean_epochs,
                            mean_values,
                            color=hue,
                            ls=line_style,
                            lw=2.9 if is_e43 else 1.9,
                            marker="D" if is_e43 else "o",
                            markersize=4.3 if is_e43 else 2.6,
                            markeredgecolor="white" if is_e43 else hue,
                            markeredgewidth=0.55 if is_e43 else 0.0,
                            zorder=8 if is_e43 else 4,
                        )
            if single_answer_coverage_na:
                ax.text(
                    0.5,
                    0.5,
                    "N/A\nsingle-answer task",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="#666666",
                )
                ax.set_yticks([])
            elif not metric_is_live:
                cohort = (
                    "E39/E41/E43"
                    if eval_split == "math"
                    else "E37/E38/E41/E43"
                )
                if has_any_evaluation and metric.startswith(
                    "semantic_shannon_success_conditioned_signed_"
                ):
                    message = "AWAITING E43 SIGNED\nTELEMETRY"
                elif has_any_evaluation and metric.startswith(
                    "semantic_shannon_separate_"
                ):
                    message = "AWAITING E41 ADVANTAGE\nTELEMETRY"
                elif has_any_evaluation and metric.startswith(
                    "semantic_shannon_"
                ):
                    message = "AWAITING SHANNON\nTELEMETRY"
                elif has_any_evaluation and metric.startswith(
                    "outcome_collision_"
                ):
                    message = "AWAITING COLLISION\nTELEMETRY"
                elif (
                    has_any_evaluation
                    and metric == "mechanism_parseable_fraction"
                ):
                    message = "AWAITING MECHANISM\nTELEMETRY"
                else:
                    message = f"AWAITING FIRST\n{cohort} EVALUATION"
                ax.text(
                    0.5,
                    0.5,
                    message,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="#777777",
                )
            elif (
                not has_shannon_evaluation
                and metric
                in {"greedy", "mean8", "pass8", "coverage8", "distinct8"}
            ):
                ax.text(
                    0.98,
                    0.04,
                    "SHANNON PENDING",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=5.8,
                    color=HUE_SEMANTIC_SHANNON,
                )
            if (
                metric_is_live
                and not has_e41_evaluation
                and metric
                in {"greedy", "mean8", "pass8", "coverage8", "distinct8"}
            ):
                ax.text(
                    0.98,
                    0.12,
                    "E41 PENDING",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=5.8,
                    color=HUE_SEMANTIC_SHANNON_ADVANTAGE,
                )
            if (
                metric_is_live
                and not has_e43_evaluation
                and metric
                in {"greedy", "mean8", "pass8", "coverage8", "distinct8"}
            ):
                ax.text(
                    0.98,
                    0.20,
                    "E43 PENDING",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=5.8,
                    color=HUE_SUCCESS_CONDITIONED_SIGNED_SHANNON,
                )
            ax.set_xlim(0.0, 10.0)
            if metric == "distinct8":
                ax.set_ylim(0.0, 8.0)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            elif (
                metric
                == "semantic_shannon_success_conditioned_signed_effective_advantage_rms"
            ):
                ax.set_ylim(0.0, 0.055)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            else:
                ax.set_ylim(0.0, 1.0)
            ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
            ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2.5, width=0.7)
            ax.set_title(label, fontsize=7.8)
            if column_index == 0:
                ax.set_ylabel(environment, fontsize=8.5, weight="bold")
            if row_index == len(environment_paths) - 1:
                ax.set_xlabel("training passes", fontsize=7.5)

    legend_handles = []
    for arm, label, hue, line_style in OUTCOME_COLLISION_05B_ARMS:
        is_e43 = arm == "freeform_success_conditioned_signed_semantic_shannon"
        legend_handles.append(
            Line2D(
                [],
                [],
                color=hue,
                lw=2.9 if is_e43 else 1.9,
                ls=line_style,
                marker="D" if is_e43 else None,
                markersize=4.3 if is_e43 else 0.0,
                markeredgecolor="white" if is_e43 else hue,
                markeredgewidth=0.55 if is_e43 else 0.0,
                label=label,
            )
        )
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        fontsize=8.0,
    )
    fig.suptitle(
        "E37/E38/E39/E41/E43 free-form semantic diversity — matched head-to-head",
        fontsize=10.2,
        weight="bold",
        y=0.93,
    )
    fig.text(
        0.5,
        0.018,
        "Countdown/Graph use identical neutral K=8 evaluation prompts. MATH-500 "
        "uses one fixed K=8 draw (seed 390100) every two passes; coverage@8 is "
        "N/A for this single-answer task, and distinct@8 counts distinct normalized "
        "correct representations, not reasoning paths.\nCollision, predictive "
        "Shannon surprise, E41 separately-centered semantic advantage, and E43 "
        "success-conditioned signed Shannon advantage are on-policy diagnostics. "
        "E43 learns semantic support from verified successes and retains bounded "
        "positive and negative pressure on eligible successes; its applied RMS, "
        "eligibility, signed fractions, and cap-hit fraction are shown. Thin lines are seeds "
        "and heavy lines require all three paired seeds.",
        ha="center",
        fontsize=7.2,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.15,
        top=0.85,
        wspace=0.40,
        hspace=0.46,
    )
    _atomic_savefig(fig, OUT_OUTCOME_COLLISION_05B.with_suffix(".pdf"))
    _atomic_savefig(
        fig,
        OUT_OUTCOME_COLLISION_05B.with_suffix(".png"),
        dpi=200,
    )
    plt.close(fig)
    print(f"wrote {OUT_OUTCOME_COLLISION_05B}.pdf/.png")


def _online_canonical_seed_mean(per_seed, metric: str):
    """All-three-seed mean for an optional verified-bank metric."""

    by_epoch = defaultdict(list)
    for points in per_seed.values():
        for epoch, row in points:
            value = row.get(metric)
            if value is not None:
                by_epoch[epoch].append(float(value))
    epochs = [
        epoch
        for epoch in sorted(by_epoch)
        if len(by_epoch[epoch]) >= 3
    ]
    return epochs, [statistics.fmean(by_epoch[epoch]) for epoch in epochs]


def _live_x_upper(epochs, *, empty_frontier: float = 0.25) -> float:
    """Return a small-padded live frontier instead of the full run budget."""

    finite = [float(value) for value in epochs if math.isfinite(float(value))]
    if not finite or max(finite) <= 0.0:
        return float(empty_frontier)
    frontier = max(finite)
    return frontier + max(0.02 * frontier, 0.02)


def _tight_nonnegative_limits(
    values,
    *,
    empty_limits: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Tightly frame observed nonnegative data with modest visual padding."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return empty_limits
    lower = min(finite)
    upper = max(finite)
    span = upper - lower
    if span <= 1e-12:
        padding = max(abs(upper) * 0.06, 0.01)
    else:
        padding = max(span * 0.08, 0.005)
    return max(0.0, lower - padding), upper + padding


def render_online_canonical_maxent_05b() -> None:
    """Render the active canonical campaign at its live frontier."""

    e59_active = (
        ROOT
        / "var/artifacts/"
        "e59_mathir_global_verified_replay_matched_identity.json"
    ).is_file()
    e58_active = (
        ROOT
        / "var/artifacts/"
        "e58_global_verified_replay_canonical_05b_sentinel_identity.json"
    ).is_file()
    e57_active = (
        ROOT
        / "var/artifacts/"
        "e57_verified_first_split_canonical_05b_sentinel_identity.json"
    ).is_file()
    e56_active = (
        ROOT
        / "var/artifacts/"
        "e56_open_set_split_canonical_05b_sentinel_identity.json"
    ).is_file()
    e55_active = (
        ROOT
        / "var/artifacts/e55_per_rollout_verified_anchor_identity.json"
    ).is_file()
    e53_stage_a_active = (
        ROOT / "var/artifacts/e53_verified_replay_05b_stage_a_identity.json"
    ).is_file()
    e53_active = e53_stage_a_active or (
        ROOT
        / "var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
    ).is_file()
    stage_a_active = (
        ROOT
        / "var/artifacts/"
        "e52_direct_inverse_entropy_canonical_05b_stage_a_v1_identity.json"
    ).is_file() and not any(
        (
            e53_active,
            e55_active,
            e56_active,
            e57_active,
            e58_active,
            e59_active,
        )
    )
    if e59_active and not e58_active:
        raise RuntimeError("E59 MathIR display requires the frozen E58 cohort")
    control_prefixes: dict[str, str] | None = None
    if e58_active:
        countdown_prefix = (
            "cde58_global_verified_replay_canonical_05b_50ep_"
            "sentinel_allcs"
        )
        graph_prefix = (
            "gce58_global_verified_replay_canonical_05b_50ep_sentinel"
        )
        python_prefix = (
            "pye58_global_verified_replay_canonical_05b_50ep_"
            "sentinel_allcs"
        )
        control_prefixes = {
            "Countdown": "cde53_verified_replay_05b_50ep_sentinel_allcs",
            "Graph coloring": "gce53_verified_replay_05b_50ep_sentinel",
            "Python factors": (
                "pye53_verified_replay_05b_50ep_sentinel_allcs"
            ),
        }
        phase_label = (
            "global verified-replay single-seed sentinel "
            "(seed 9010; three-seed replication pending)"
        )
        line_description = (
            "Solid lines show the fresh E58 seed 9010; dashed controls are "
            "the matched E53 Dr.GRPO seed 9010. This is one sentinel seed per "
            "arm, not a three-seed mean; replication remains gated on the "
            "sentinel."
        )
        campaign = "E58"
        treatments = (
            (
                "verified_first_global_replay_canonical",
                HUE_ONLINE_CANONICAL_HAARNOJA,
                "-",
            ),
        )
        primary_treatment_arm = "verified_first_global_replay_canonical"
        entropy_panel = (
            (
                "semantic_shannon_success_conditioned_signed_"
                "open_set_entropy_ema"
            ),
            "open-set predictive entropy EMA",
            None,
            False,
        )
        alpha_panel = (
            (
                "semantic_shannon_success_conditioned_signed_"
                "open_set_next_coefficient"
            ),
            r"next semantic coefficient $\beta$",
            None,
            False,
        )
    elif e57_active:
        countdown_prefix = (
            "cde57_verified_first_split_canonical_05b_50ep_sentinel_allcs"
        )
        graph_prefix = (
            "gce57_verified_first_split_canonical_05b_50ep_sentinel"
        )
        python_prefix = (
            "pye57_verified_first_split_canonical_05b_50ep_sentinel_allcs"
        )
        control_prefixes = {
            "Countdown": "cde53_verified_replay_05b_50ep_sentinel_allcs",
            "Graph coloring": "gce53_verified_replay_05b_50ep_sentinel",
            "Python factors": (
                "pye53_verified_replay_05b_50ep_sentinel_allcs"
            ),
        }
        phase_label = "verified-first split-controller sentinel"
        line_description = (
            "Solid lines show the fresh E57 seed 9010; dashed controls are "
            "the matched E53 Dr.GRPO seed 9010."
        )
        campaign = "E57"
        treatments = (
            (
                "verified_first_split_canonical",
                HUE_ONLINE_CANONICAL_HAARNOJA,
                "-",
            ),
        )
        primary_treatment_arm = "verified_first_split_canonical"
        entropy_panel = (
            (
                "semantic_shannon_success_conditioned_signed_"
                "open_set_entropy_ema"
            ),
            "open-set predictive entropy EMA",
            None,
            False,
        )
        alpha_panel = (
            (
                "semantic_shannon_success_conditioned_signed_"
                "open_set_next_coefficient"
            ),
            r"next semantic coefficient $\beta$",
            None,
            False,
        )
    elif e56_active:
        countdown_prefix = (
            "cde56_open_set_split_canonical_05b_50ep_sentinel_allcs"
        )
        graph_prefix = "gce56_open_set_split_canonical_05b_50ep_sentinel"
        python_prefix = (
            "pye56_open_set_split_canonical_05b_50ep_sentinel_allcs"
        )
        control_prefixes = {
            "Countdown": "cde53_verified_replay_05b_50ep_sentinel_allcs",
            "Graph coloring": "gce53_verified_replay_05b_50ep_sentinel",
            "Python factors": (
                "pye53_verified_replay_05b_50ep_sentinel_allcs"
            ),
        }
        phase_label = "open-set split-controller sentinel"
        line_description = (
            "Solid lines show the fresh E56 seed 9010; dashed controls are "
            "the matched E53 Dr.GRPO seed 9010."
        )
        campaign = "E56"
        treatments = (
            (
                "open_set_split_canonical",
                HUE_ONLINE_CANONICAL_HAARNOJA,
                "-",
            ),
        )
        primary_treatment_arm = "open_set_split_canonical"
        entropy_panel = (
            (
                "semantic_shannon_success_conditioned_signed_"
                "open_set_entropy_ema"
            ),
            "open-set predictive entropy EMA",
            None,
            False,
        )
        alpha_panel = (
            (
                "semantic_shannon_success_conditioned_signed_"
                "open_set_next_coefficient"
            ),
            r"next semantic coefficient $\beta$",
            None,
            False,
        )
    elif e55_active:
        countdown_prefix = (
            "cde55_per_rollout_verified_anchor_05b_50ep_sentinel_allcs"
        )
        graph_prefix = (
            "gce55_per_rollout_verified_anchor_05b_50ep_sentinel"
        )
        python_prefix = (
            "pye55_per_rollout_verified_anchor_05b_50ep_sentinel_allcs"
        )
        control_prefixes = {
            "Countdown": "cde53_verified_replay_05b_50ep_sentinel_allcs",
            "Graph coloring": "gce53_verified_replay_05b_50ep_sentinel",
            "Python factors": (
                "pye53_verified_replay_05b_50ep_sentinel_allcs"
            ),
        }
        phase_label = "per-rollout verified-anchor sentinel"
        line_description = (
            "Solid lines show the fresh E55 seed 9010; dashed controls are "
            "the frozen E53 matched Dr.GRPO seed 9010."
        )
        campaign = "E55"
        treatments = (
            (
                "maxent_inverse_canonical_replay",
                HUE_ONLINE_CANONICAL_HAARNOJA,
                "-",
            ),
        )
        primary_treatment_arm = "maxent_inverse_canonical_replay"
        entropy_panel = (
            "canonical_replay_normalized_model_entropy",
            "model entropy over verified replay bank",
            None,
            False,
        )
        alpha_panel = (
            "canonical_replay_next_alpha",
            r"next verified-anchor coefficient $\alpha$",
            None,
            False,
        )
    elif e53_active:
        if e53_stage_a_active:
            countdown_prefix = "cde53_verified_replay_05b_50ep_stage_a_allcs"
            graph_prefix = "gce53_verified_replay_05b_50ep_stage_a"
            python_prefix = "pye53_verified_replay_05b_50ep_stage_a_allcs"
            phase_label = "Stage A — three fresh seeds"
            line_description = (
                "Heavy lines show the fresh seed-43/44/45 mean; faint lines "
                "show individual seeds."
            )
        else:
            countdown_prefix = "cde53_verified_replay_05b_50ep_sentinel_allcs"
            graph_prefix = "gce53_verified_replay_05b_50ep_sentinel"
            python_prefix = "pye53_verified_replay_05b_50ep_sentinel_allcs"
            phase_label = "verified-replay sentinel"
            line_description = "Lines show the fresh engineering seed 9010."
        campaign = "E53"
        treatments = (
            (
                "maxent_inverse",
                HUE_ONLINE_CANONICAL_MAXENT,
                (0, (4, 2)),
            ),
            (
                "maxent_inverse_canonical_replay",
                HUE_ONLINE_CANONICAL_HAARNOJA,
                "-",
            ),
        )
        primary_treatment_arm = "maxent_inverse_canonical_replay"
        entropy_panel = (
            "canonical_replay_normalized_model_entropy",
            "model entropy over verified replay bank",
            None,
            False,
        )
        alpha_panel = (
            "canonical_replay_next_alpha",
            r"next verified-replay coefficient $\alpha$",
            None,
            False,
        )
    elif stage_a_active:
        countdown_prefix = (
            "cde52_direct_inverse_entropy_canonical_05b_50ep_"
            "stage_a_v1_allcs"
        )
        graph_prefix = (
            "gce52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1"
        )
        python_prefix = (
            "pye52_direct_inverse_entropy_canonical_05b_50ep_"
            "stage_a_v1_allcs"
        )
        phase_label = "Stage A — three fresh seeds"
        line_description = (
            "Heavy lines show the fresh seed-43/44/45 mean; faint lines show "
            "individual seeds."
        )
        campaign = "E52"
    else:
        countdown_prefix = (
            "cde52_direct_inverse_entropy_canonical_05b_50ep_"
            "sentinel_v2_allcs"
        )
        graph_prefix = (
            "gce52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2"
        )
        python_prefix = (
            "pye52_direct_inverse_entropy_canonical_05b_50ep_"
            "sentinel_v2_allcs"
        )
        phase_label = "engineering sentinel"
        line_description = "Lines show the engineering sentinel seed 9009."
        campaign = "E52"
    if not any((e53_active, e55_active, e56_active, e57_active, e58_active)):
        treatments = (
            (
                "maxent_inverse",
                HUE_ONLINE_CANONICAL_MAXENT,
                (0, (4, 2)),
            ),
            (
                "maxent_inverse_canonical",
                HUE_ONLINE_CANONICAL_HAARNOJA,
                "-",
            ),
        )
        primary_treatment_arm = "maxent_inverse_canonical"
        entropy_panel = (
            "maxent_inverse_multiplier",
            r"warmup mean / conditional entropy EMA",
            None,
            False,
        )
        alpha_panel = (
            "maxent_inverse_next_alpha",
            r"next direct entropy coefficient $\lambda$",
            None,
            False,
        )
    if e59_active:
        phase_label = "three-domain sentinel + matched MathIR extension"
        line_description = (
            "The first three rows retain the frozen E58 seed-9010 sentinel "
            "and its matched E53 control. The MathIR row shows fresh E59 "
            "seed-43/44/45 means with faint individual-seed trajectories."
        )
    quality = (
        ("greedy", "neutral pass@1", None, False),
        ("mean8", "neutral mean@8", None, False),
        ("pass8", "neutral pass@8", None, False),
        ("distinct8", "mean # distinct correct@8", None, True),
    )
    environment_paths = (
        {
            "environment": f"Countdown — {campaign} (live frontier)",
            "control_path": (
                "var/artifacts/"
                f"{(control_prefixes or {}).get('Countdown', countdown_prefix)}"
                "_scaling_curve.json"
            ),
            "treatment_path": f"var/artifacts/{countdown_prefix}_scaling_curve.json",
            "steps_per_epoch": 384,
            "max_passes": 50.0,
            "treatments": treatments,
            "primary_treatment_arm": primary_treatment_arm,
            "primary_treatment_hue": HUE_ONLINE_CANONICAL_HAARNOJA,
            "quality": quality,
            "entropy": entropy_panel,
            "alpha": alpha_panel,
            "experiment": campaign,
        },
        {
            "environment": f"Graph coloring — {campaign} (live frontier)",
            "control_path": (
                "var/artifacts/"
                f"{(control_prefixes or {}).get('Graph coloring', graph_prefix)}"
                "_scaling_curve.json"
            ),
            "treatment_path": f"var/artifacts/{graph_prefix}_scaling_curve.json",
            "steps_per_epoch": 192,
            "max_passes": 50.0,
            "treatments": treatments,
            "primary_treatment_arm": primary_treatment_arm,
            "primary_treatment_hue": HUE_ONLINE_CANONICAL_HAARNOJA,
            "quality": quality,
            "entropy": entropy_panel,
            "alpha": alpha_panel,
            "experiment": campaign,
        },
        {
            "environment": f"Python factors — {campaign} (live frontier)",
            "control_path": (
                "var/artifacts/"
                f"{(control_prefixes or {}).get('Python factors', python_prefix)}"
                "_scaling_curve.json"
            ),
            "treatment_path": f"var/artifacts/{python_prefix}_scaling_curve.json",
            "steps_per_epoch": 384,
            "max_passes": 50.0,
            "treatments": treatments,
            "primary_treatment_arm": primary_treatment_arm,
            "primary_treatment_hue": HUE_ONLINE_CANONICAL_HAARNOJA,
            "quality": quality,
            "entropy": entropy_panel,
            "alpha": alpha_panel,
            "experiment": campaign,
        },
    )
    if e59_active:
        mathir_curve = (
            f"var/artifacts/{E59_MATHIR_PREFIX}_scaling_curve.json"
        )
        environment_paths += (
            {
                "environment": (
                    "Executable MathIR — E59 action-menu eval "
                    "(not MATH-500; live frontier)"
                ),
                "control_path": mathir_curve,
                "treatment_path": mathir_curve,
                "steps_per_epoch": 384,
                "max_passes": 50.0,
                "treatments": (
                    (
                        "verified_first_global_replay_canonical",
                        HUE_ONLINE_CANONICAL_HAARNOJA,
                        "-",
                    ),
                ),
                "primary_treatment_arm": (
                    "verified_first_global_replay_canonical"
                ),
                "primary_treatment_hue": HUE_ONLINE_CANONICAL_HAARNOJA,
                "quality": quality,
                "entropy": entropy_panel,
                "alpha": alpha_panel,
                "experiment": "E59",
            },
        )
    mechanism_tail = (
        (
            "canonical_replay_balance_loss",
            "verified replay KL",
            None,
            False,
        ),
        (
            "canonical_replay_available_modes",
            "replayed verified modes",
            None,
            True,
        ),
        (
            "online_canonical_new_outcome_row_fraction",
            "new verified outcome fraction",
            None,
            False,
        ),
        (
            "online_canonical_mean_support_per_prompt",
            "mean verified support per prompt",
            None,
            False,
        ),
        (
            "online_canonical_tracked_outcomes",
            "cumulative verified discoveries",
            None,
            True,
        ),
    ) if (
        e53_active
        or e55_active
        or e56_active
        or e57_active
        or e58_active
        or e59_active
    ) else (
        (
            "online_canonical_advantage_rms_components",
            "train exploration advantage RMS",
            None,
            False,
        ),
        (
            "online_canonical_new_outcome_row_fraction",
            "new verified outcome fraction",
            None,
            False,
        ),
        (
            "online_canonical_eligible_fraction",
            "verified eligible fraction",
            None,
            False,
        ),
        (
            "online_canonical_mean_support_per_prompt",
            "mean verified support per prompt",
            None,
            False,
        ),
        (
            "online_canonical_tracked_outcomes",
            "cumulative verified discoveries",
            None,
            True,
        ),
    )
    num_panels = len(quality) + 2 + len(mechanism_tail)
    fig, axes = plt.subplots(
        len(environment_paths),
        num_panels,
        figsize=(25.0, 3.0 + 1.4 * len(environment_paths)),
        squeeze=False,
    )
    for row_index, row_spec in enumerate(environment_paths):
        environment = row_spec["environment"]
        if environment.startswith("Graph coloring"):
            environment_label = "Graph coloring"
        elif environment.startswith("Countdown"):
            environment_label = "Countdown"
        elif environment.startswith("Python factors"):
            environment_label = "Python factors"
        elif environment.startswith("Executable MathIR"):
            environment_label = "MathIR (not MATH-500)"
        else:
            environment_label = environment
        treatments = row_spec["treatments"]
        treatment_arm = row_spec["primary_treatment_arm"]
        treatment_hue = row_spec["primary_treatment_hue"]
        experiment = row_spec["experiment"]
        steps_per_epoch = row_spec["steps_per_epoch"]
        series = load_series(
            row_spec["control_path"],
            steps_per_epoch,
            split="multi_answer",
            max_training_epochs=None,
        )
        if row_spec["treatment_path"] != row_spec["control_path"]:
            # Cross-campaign controls can share the treatment arm name. Keep
            # only the preregistered Dr.GRPO rows from the frozen control
            # artifact so an older treatment can never masquerade as E55
            # while the fresh job is still pending.
            series = {"grpo": series.get("grpo", {})}
            treatment_series = load_series(
                row_spec["treatment_path"],
                steps_per_epoch,
                split="multi_answer",
                max_training_epochs=None,
            )
            for arm, per_seed in treatment_series.items():
                series.setdefault(arm, {}).update(per_seed)
        treatment_live_epochs = [
            epoch
            for arm, _hue, _style in treatments
            for points in series.get(arm, {}).values()
            for epoch, _row in points
        ]
        live_x_upper = _live_x_upper(treatment_live_epochs)
        treatment_live = any(
            points
            for arm, _hue, _style in treatments
            for points in series.get(arm, {}).values()
        )
        panel_specs = (
            *row_spec["quality"],
            row_spec["entropy"],
            row_spec["alpha"],
            *mechanism_tail,
        )
        quality_metrics = {
            metric for metric, _label, _limits, _integer in row_spec["quality"]
        }
        passive_control_metrics = {
            "online_canonical_mean_support_per_prompt",
            "online_canonical_tracked_outcomes",
        }
        for column_index, (
            metric,
            label,
            _legacy_y_limits,
            integer_ticks,
        ) in enumerate(panel_specs):
            ax = axes[row_index][column_index]
            metric_is_live = False
            plotted_y_values: list[float] = []
            component_panel = (
                metric == "online_canonical_advantage_rms_components"
            )
            fixed_alpha_panel = metric == "__fixed_alpha__"
            if fixed_alpha_panel:
                metric_is_live = treatment_live
                metric_series = []
                ax.axhline(
                    0.10,
                    color=treatment_hue,
                    lw=2.0,
                    zorder=3,
                )
                plotted_y_values.append(0.10)
                ax.text(
                    0.98,
                    0.10,
                    "fixed by protocol",
                    transform=ax.get_xaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=5.8,
                    color=treatment_hue,
                )
            elif component_panel:
                metric_series = [
                    (
                        component_metric,
                        component_label,
                        (
                            treatment_hue
                            if component_metric
                            == "online_canonical_combined_advantage_rms"
                            else component_hue
                        ),
                        component_line_style,
                        series.get(treatment_arm, {}),
                    )
                    for (
                        component_metric,
                        component_label,
                        component_hue,
                        component_line_style,
                    ) in ONLINE_CANONICAL_ADVANTAGE_COMPONENTS
                ]
            else:
                metric_series = [
                    (
                        metric,
                        None,
                        HUE_ONLINE_CANONICAL_CONTROL,
                        (0, (4, 2)),
                        series.get("grpo", {}),
                    ),
                    *(
                        (
                            metric,
                            None,
                            hue,
                            line_style,
                            series.get(arm, {}),
                        )
                        for arm, hue, line_style in treatments
                    ),
                ]
                if (
                    metric not in quality_metrics
                    and metric not in passive_control_metrics
                ):
                    metric_series = [
                        (
                            metric,
                            None,
                            hue,
                            line_style,
                            series.get(arm, {}),
                        )
                        for arm, hue, line_style in treatments
                    ]
            for (
                plotted_metric,
                _component_label,
                hue,
                line_style,
                per_seed,
            ) in metric_series:
                is_control_series = hue == HUE_ONLINE_CANONICAL_CONTROL
                is_primary_treatment_series = (
                    not is_control_series and hue == treatment_hue
                )
                point_marker = (
                    "D" if is_primary_treatment_series else "o"
                )
                observed_seed_count = sum(
                    bool(seed_points) for seed_points in per_seed.values()
                )
                # Three-seed campaigns use translucent individual traces plus
                # the opaque mean below. A single-seed sentinel has no mean
                # trace by construction, so fading its only trace makes both
                # the result and especially the dashed control look absent.
                # Render the raw sentinel lines strongly until replication
                # provides the preregistered three-seed aggregate.
                sentinel_trace = observed_seed_count < 3
                raw_alpha = (
                    0.90
                    if sentinel_trace and is_primary_treatment_series
                    else 0.82
                    if sentinel_trace
                    else 0.48
                    if is_primary_treatment_series
                    else 0.24
                )
                raw_line_width = (
                    2.2
                    if sentinel_trace and is_primary_treatment_series
                    else 1.45
                    if sentinel_trace
                    else 1.0
                    if is_primary_treatment_series
                    else 0.55
                )
                raw_marker_size = (
                    4.0
                    if sentinel_trace and is_primary_treatment_series
                    else 3.0
                    if sentinel_trace
                    else 3.2
                    if is_primary_treatment_series
                    else 1.7
                )
                for points in per_seed.values():
                    values = [
                        (epoch, row.get(plotted_metric))
                        for epoch, row in points
                        if row.get(plotted_metric) is not None
                    ]
                    if not values:
                        continue
                    metric_is_live = True
                    plotted_y_values.extend(
                        float(value) for _epoch, value in values
                    )
                    ax.plot(
                        [epoch for epoch, _value in values],
                        [float(value) for _epoch, value in values],
                        color=hue,
                        alpha=raw_alpha,
                        lw=raw_line_width,
                        marker=point_marker,
                        markersize=raw_marker_size,
                        markeredgecolor=(
                            "white"
                            if is_primary_treatment_series
                            else hue
                        ),
                        markeredgewidth=(
                            0.45 if is_primary_treatment_series else 0.0
                        ),
                        zorder=(
                            5 if is_primary_treatment_series else 2
                        ),
                    )
                mean_epochs, mean_values = _online_canonical_seed_mean(
                    per_seed,
                    plotted_metric,
                )
                if mean_epochs:
                    ax.plot(
                        mean_epochs,
                        mean_values,
                        color=hue,
                        ls=line_style,
                        lw=2.8 if is_primary_treatment_series else 1.55,
                        marker=point_marker,
                        markersize=(
                            4.2 if is_primary_treatment_series else 2.5
                        ),
                        markeredgecolor=(
                            "white"
                            if is_primary_treatment_series
                            else hue
                        ),
                        markeredgewidth=(
                            0.6 if is_primary_treatment_series else 0.0
                        ),
                        zorder=(
                            6 if is_primary_treatment_series else 3
                        ),
                    )
                    if (
                        experiment in {"E57", "E58", "E59"}
                        and plotted_metric in quality_metrics
                        and is_primary_treatment_series
                    ):
                        ax.annotate(
                            experiment,
                            (mean_epochs[-1], mean_values[-1]),
                            xytext=(-2, 4),
                            textcoords="offset points",
                            ha="right",
                            va="bottom",
                            color=hue,
                            fontsize=5.8,
                            weight="bold",
                            zorder=7,
                        )
            if component_panel and metric_is_live:
                ax.legend(
                    handles=[
                        Line2D(
                            [],
                            [],
                            color=hue,
                            lw=1.6,
                            ls=line_style,
                            label=component_label,
                        )
                        for (
                            component_metric,
                            component_label,
                            hue,
                            line_style,
                        ) in ONLINE_CANONICAL_ADVANTAGE_COMPONENTS
                        for hue in (
                            treatment_hue
                            if component_metric
                            == "online_canonical_combined_advantage_rms"
                            else hue,
                        )
                    ],
                    frameon=False,
                    fontsize=5.8,
                    loc="upper left",
                    handlelength=2.2,
                    borderaxespad=0.2,
                )
            if not metric_is_live:
                if metric in quality_metrics:
                    message = f"AWAITING FIRST\n{experiment} EVALUATION"
                else:
                    message = "AWAITING VERIFIED-BANK\nTELEMETRY"
                ax.text(
                    0.5,
                    0.5,
                    message,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="#777777",
                )
            elif (
                metric in quality_metrics
                and not treatment_live
            ):
                ax.text(
                    0.98,
                    0.04,
                    "TREATMENT PENDING",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=5.8,
                    color=treatment_hue,
                )
            elif (
                experiment in {"E57", "E58", "E59"}
                and metric in quality_metrics
                and treatment_live
            ):
                # Keep the live arm explicit even when its values coincide
                # with the dashed control or sit exactly on the zero axis.
                ax.text(
                    0.98,
                    0.94,
                    f"{experiment} TREATMENT",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=5.3,
                    weight="bold",
                    color=treatment_hue,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": treatment_hue,
                        "linewidth": 0.55,
                        "alpha": 0.92,
                    },
                    zorder=8,
                )
            if metric == "maxent_inverse_multiplier":
                ax.axhline(
                    1.0,
                    color=HUE_ONLINE_CANONICAL_HAARNOJA,
                    lw=0.9,
                    ls=(0, (2, 2)),
                    alpha=0.8,
                    zorder=1,
                )
                plotted_y_values.append(0.8)
            if metric == "maxent_inverse_next_alpha":
                ax.axhline(
                    0.000075,
                    color=HUE_ONLINE_CANONICAL_CONTROL,
                    lw=0.9,
                    ls=(0, (2, 2)),
                    alpha=0.8,
                    zorder=1,
                )
                plotted_y_values.append(0.1)
            ax.set_xlim(0.0, live_x_upper)
            ax.set_ylim(*_tight_nonnegative_limits(plotted_y_values))
            if integer_ticks:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
            ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["bottom"].set_zorder(1)
            ax.tick_params(length=2.5, width=0.7)
            ax.set_title(label, fontsize=7.8)
            if column_index == 0:
                ax.set_ylabel(environment_label, fontsize=8.5, weight="bold")
            if row_index == len(environment_paths) - 1:
                ax.set_xlabel("training passes", fontsize=7.5)

    active_arms = {
        "grpo",
        *(arm for arm, _hue, _line_style in treatments),
    }
    legend_handles = [
        Line2D(
            [],
            [],
            color=hue,
            lw=2.0,
            ls=line_style,
            label=(
                r"E58 verified-first + global split mass/balance replay"
                r" (seed 9010 sentinel)"
                if (
                    e58_active
                    and not e59_active
                    and arm == "verified_first_global_replay_canonical"
                )
                else
                r"E58/E59 verified-first + global split mass/balance replay"
                if (
                    e59_active
                    and arm == "verified_first_global_replay_canonical"
                )
                else
                r"E57 verified-first discovery + split mass/balance replay"
                if (
                    e57_active
                    and arm == "verified_first_split_canonical"
                )
                else
                r"E56 open-set discovery + split mass/balance replay"
                if e56_active and arm == "open_set_split_canonical"
                else
                r"Matched Dr.GRPO (seed 9010 sentinel)"
                if (
                    e58_active
                    and not e59_active
                    and arm == "grpo"
                )
                else
                r"E55 inverse discovery + per-rollout verified anchor"
                if (
                    e55_active
                    and arm == "maxent_inverse_canonical_replay"
                )
                else label
            ),
        )
        for arm, label, hue, line_style in ONLINE_CANONICAL_MAXENT_05B_ARMS
        if arm in active_arms
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        fontsize=8.2,
    )
    if e59_active:
        campaign = "E58 + E59"
    fig.suptitle(
        f"Current canonical experiment — {campaign} ModeBench "
        f"{phase_label} — 0.5B",
        fontsize=10.2,
        weight="bold",
        y=0.935,
    )
    fig.text(
        0.5,
        0.012,
        "Each row uses its own matched Dr.GRPO and preregistered evaluation. "
        "Every x-axis expands with the latest treatment checkpoint; panels in "
        "a row share that live frontier, while metric ranges tighten to the "
        "values observed so far. "
        f"{campaign} uses K=8 exact-answer metrics with a 50-pass sentinel "
        "budget, but live-chart ingestion remains uncapped and will display "
        "later checkpoints. "
        "Gold-support-normalized coverage is deliberately omitted from this "
        "decision view and never enters the controller or terminal gate. "
        "Cancelled predecessor treatments are omitted.\n"
        "Bank admission is validator-bound in every treatment: Countdown uses the "
        "executed exact AST, graph coloring the constraint-checked vector, and "
        "Python factors the externally executed return vector"
        + (
            ", while MathIR uses the exact rational equation-state trajectory "
            "from the submitted and executed finite action sequence. Its "
            "held-out action-menu evaluation is not MATH-500"
            if e59_active
            else ""
        )
        + ". The model cannot supply its own canonical key.\n"
        + (
            "E58 keeps E57's zero-gradient verified-first cold start, then "
            "schedules exactly one model-discovered validator-positive bank "
            "per optimizer update in checkpointed prompt-hash round-robin "
            "order. Verified surprisal controls common mass, verified-bank "
            "entropy controls relative known-mode balance, and success-only "
            "open-set pressure explores newly verified outcomes. All three "
            "coefficients use only their own 64-observation warmup/EMA state "
            "and remain unbounded. The one-bank schedule is a fixed compute "
            "measure, not a desired entropy or mode count; training never "
            "reads gold support or evaluation. "
            if e58_active
            else
            "E57 removes direct token entropy. Before the first "
            "validator-positive model sample, all-zero groups have zero "
            "policy gradient and preserve the pretrained proposal "
            "distribution. After discovery, success-only open-set semantic "
            "pressure explores verified outcomes, verified surprisal controls "
            "common mass, and verified-bank entropy controls relative "
            "known-mode balance. All three coefficients use only their own "
            "64-observation warmup/EMA state and remain unbounded. The replay "
            "losses share one fixed 1/16 pseudo-rollout measure; training "
            "never reads gold support or evaluation. "
            if e57_active
            else
            "E56 separates four causal roles. Conditional-content entropy "
            "maintains reachability; a success-only open-set predictor gives "
            "negative pressure to repeated known outcomes and positive "
            "pressure to newly validator-confirmed outcomes; verified "
            "surprisal controls common-mass likelihood; and verified-bank "
            "entropy controls relative known-mode balance. All four "
            "coefficients use only their own 64-observation warmup/EMA state "
            "and remain unbounded. The two replay losses share one fixed "
            "1/16 pseudo-rollout measure; training never reads gold support "
            "or evaluation. "
            if e56_active
            else
            "E55 separates discovery, anchoring, and multi-mode retention. "
            "Direct conditional-content entropy remains active before any "
            "verified mode exists; uniform verified likelihood anchors the "
            "first observed correct exemplar; and only banks with at least "
            "two modes feed the canonical-entropy inverse controller. The "
            "anchor counts as one pseudo-rollout in the 16-sample group "
            "(fixed measure 1/16). Both inverse coefficients are unbounded, "
            "and training never reads gold support or evaluation. "
            if e55_active
            else
            "E53 matches passive-tracking Dr.GRPO against direct inverse "
            "conditional-content entropy and the same discovery bridge plus "
            "teacher-forced verified-exemplar replay. Both inverse coefficients "
            "use only their own model-entropy warmup/EMA and have no coefficient "
            "bound. Replay capacity 16 is a compute bound; full verified "
            "discovery counts continue beyond it. "
            if e53_active
            else
            "E52 matches passive-tracking Dr.GRPO against direct inverse "
            "conditional-content entropy, with and without a fixed canonical "
            "bank. After 64 warmup observations, lambda is 0.000075 times the "
            "run's warmup mean divided by its conditional-entropy EMA; it has "
            "no coefficient bound. The hybrid canonical coefficient remains "
            "fixed at 0.10. "
        )
        +
        f"{line_description}",
        ha="center",
        fontsize=7.2,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.17,
        top=0.85,
        wspace=0.40,
        hspace=0.55,
    )
    output_base = (
        OUT_MATHIR_GLOBAL_VERIFIED_REPLAY_05B if e59_active
        else OUT_GLOBAL_VERIFIED_REPLAY_CANONICAL_05B if e58_active
        else OUT_VERIFIED_FIRST_SPLIT_CANONICAL_05B if e57_active
        else OUT_OPEN_SET_SPLIT_CANONICAL_05B if e56_active
        else OUT_PER_ROLLOUT_VERIFIED_ANCHOR_05B if e55_active
        else OUT_VERIFIED_REPLAY_05B if e53_active
        else OUT_ONLINE_CANONICAL_MAXENT_05B
    )
    _atomic_savefig(
        fig,
        output_base.with_suffix(".pdf"),
    )
    _atomic_savefig(
        fig,
        output_base.with_suffix(".png"),
        dpi=200,
    )
    compatibility_bases = (
        OUT_ONLINE_CANONICAL_MAXENT_05B,
        LEGACY_OUT_ONLINE_CANONICAL_MAXENT_05B,
    )
    for compatibility_base in compatibility_bases:
        if compatibility_base == output_base:
            continue
        for suffix in (".pdf", ".png"):
            _atomic_copy(
                output_base.with_suffix(suffix),
                compatibility_base.with_suffix(suffix),
            )
    plt.close(fig)
    print(
        f"wrote {output_base}.pdf/.png "
        f"(compatibility aliases: {', '.join(map(str, compatibility_bases))})"
    )


def render_answer_option_binding_05b() -> None:
    """Render E36's DIAYN-only latent-conditioned binding diagnostics."""

    environment_paths = (
        (
            "Countdown",
            "var/artifacts/cde36_answer_option_mi_dual_eval_05b_v1_scaling_curve.json",
            384,
        ),
        (
            "Graph coloring",
            "var/artifacts/gce36_answer_option_mi_dual_eval_05b_v1_scaling_curve.json",
            192,
        ),
    )
    metric_specs = (
        ("latent_mean8", "latent-conditioned mean@8", (0.0, 1.0), None),
        (
            "latent_coverage8",
            "latent-conditioned coverage@8",
            (0.0, 1.0),
            None,
        ),
        (
            "option_mi",
            r"held-draw $I(Z;A\mid X,\mathrm{correct})$ lower bound",
            (-0.1, math.log(4.0)),
            0.0,
        ),
        (
            "option_classifier",
            "held-draw option-classifier accuracy",
            (0.0, 1.0),
            0.25,
        ),
    )
    fig, axes = plt.subplots(
        len(environment_paths),
        len(metric_specs),
        figsize=(12.0, 5.2),
        squeeze=False,
    )
    for row_index, (environment, path, steps_per_epoch) in enumerate(
        environment_paths
    ):
        series = load_series(path, steps_per_epoch)
        per_seed = series.get("freeform_answer_option_mi", {})
        for column_index, (metric, label, ylim, baseline) in enumerate(
            metric_specs
        ):
            ax = axes[row_index][column_index]
            for seed, points in sorted(per_seed.items()):
                values = [
                    (epoch, row.get(metric))
                    for epoch, row in points
                    if row.get(metric) is not None
                ]
                if values:
                    ax.plot(
                        [value[0] for value in values],
                        [value[1] for value in values],
                        color=HUE_ANSWER_OPTION_MI,
                        alpha=0.28,
                        lw=0.8,
                    )
            mean_epochs, mean_values = seed_mean(
                per_seed,
                metric,
                required_seed_count=3,
            )
            if mean_epochs:
                ax.plot(
                    mean_epochs,
                    mean_values,
                    color=HUE_ANSWER_OPTION_MI,
                    lw=2.0,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "AWAITING 3-SEED E36 EVALUATION",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="#777777",
                )
            if baseline is not None:
                ax.axhline(
                    baseline,
                    color="#888888",
                    lw=0.7,
                    ls=(0, (3, 2)),
                )
            ax.set_xlim(0.0, 10.0)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.18, lw=0.5)
            ax.set_title(label, fontsize=8.0)
            if column_index == 0:
                ax.set_ylabel(environment, fontsize=8.5, weight="bold")
            if row_index == len(environment_paths) - 1:
                ax.set_xlabel("training passes", fontsize=8.0)
    fig.suptitle(
        "E36 latent-conditioned binding — independent deterministic seed per "
        "(draw, prompt, z)",
        fontsize=10.0,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Thin lines are individual training seeds; heavy lines appear only "
        "when all three seeds have landed. These are not neutral-quality curves.",
        ha="center",
        fontsize=7.3,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.13,
        top=0.86,
        wspace=0.34,
        hspace=0.42,
    )
    _atomic_savefig(fig, OUT_ANSWER_OPTION_BINDING_05B.with_suffix(".pdf"))
    _atomic_savefig(
        fig,
        OUT_ANSWER_OPTION_BINDING_05B.with_suffix(".png"),
        dpi=200,
    )
    plt.close(fig)
    print(f"wrote {OUT_ANSWER_OPTION_BINDING_05B}.pdf/.png")


def main() -> None:
    render_figure(ENVIRONMENTS, OUT, LATEST_PREVIEW)
    for panel in PUBLISHED_METHOD_PANELS:
        panel_key = panel[0]
        render_figure(
            ENVIRONMENTS,
            OUT_BY_PANEL[panel_key],
            LATEST_PREVIEW_BY_PANEL[panel_key],
            method_panels=(panel,),
        )
    for environment in ENVIRONMENTS:
        render_figure([environment], OUT_BY_ENVIRONMENT[environment[0]])
    freeform_05b_environments = []
    for environment, scale_rows in ENVIRONMENTS:
        scale, paths, steps_per_epoch, _budget, empty_note = scale_rows[0]
        freeform_05b_environments.append(
            (environment, [(scale, paths, steps_per_epoch, None, empty_note)])
        )
    render_figure(
        freeform_05b_environments,
        OUT_FREEFORM_05B,
        method_panels=(FREEFORM_05B_PANEL,),
    )
    render_latest_freeform_05b()
    render_answer_option_mi_05b()
    render_answer_option_binding_05b()
    render_outcome_collision_05b()
    render_online_canonical_maxent_05b()


if __name__ == "__main__":
    main()
