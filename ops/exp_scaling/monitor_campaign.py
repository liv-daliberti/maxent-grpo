#!/usr/bin/env python3
"""Live terminal dashboard for the latest 0.5B and 3B free-form reruns.

The dashboard intentionally separates durable evaluation progress ("landed")
from the optimizer step in the currently allocated attempt ("live"). That
distinction matters when a replacement starts below a crashed attempt's
frontier.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = ROOT / "var/data"
DEFAULT_ARTIFACT_ROOT = ROOT / "var/artifacts"
FIGURE_REFRESH_SCRIPT = ROOT / "ops/exp_scaling/refresh_latest_freeform_05b.py"
DEFAULT_FIGURE_REFRESH_LOG = (
    DEFAULT_ARTIFACT_ROOT / "logs/monitor_figure_refresh.log"
)
E16_PROTOCOL = ROOT / "paper/preregistration/e16_canonical_maxent_replication.md"
SEEDS = (43, 44, 45)
SMOKE_SEED = 9006
EVAL_KEY = "eval/multi_answer/sampled_mode_coverage_at_8"
MATH_EVAL_KEY = "eval/math/sampled_any_correct_at_8"
MAX_PASSES = 10.0
LATEST_05B_PREFIXES = {
    "Countdown": "cde32_freeform_05b_ema_10ep_v4_preemptsafe",
    "Graph coloring": "gce32_freeform_05b_ema_10ep_v5",
}
LATEST_3B_PREFIXES = {
    "Countdown": "cde33_freeform_3b_ema_10ep_v3_a100",
    "Graph coloring": "gce33_freeform_3b_ema_10ep_v3_a100",
}
E39_PREFIX = "mte39_math12k_384_semantic_entropy_05b_v1"
E39_ENVIRONMENT = "MATH-500 (train MATH12K-384)"
E41_ARM = "semantic_shannon_advantage"
E41_PREFIXES = {
    "Countdown": "cde41_semantic_shannon_advantage_05b_v1",
    "Graph coloring": "gce41_semantic_shannon_advantage_05b_v1",
}
E41_MATH_PREFIX = "mte41_math12k_384_semantic_shannon_advantage_05b_v1"
E43_ARM = "success_conditioned_signed_semantic_shannon"
E43_PREFIXES = {
    "Countdown": "cde43_success_conditioned_signed_semantic_shannon_05b_v1",
    "Graph coloring": (
        "gce43_success_conditioned_signed_semantic_shannon_05b_v1"
    ),
}
E43_MATH_PREFIX = (
    "mte43_math12k_384_success_conditioned_signed_semantic_shannon_05b_v1"
)
E44_OGS_PREFIXES = {
    "Countdown": "cde44_ogs_canonical_maxent_05b_v1",
    "Graph coloring": "gce44_ogs_canonical_maxent_05b_v1",
}
E52_DIRECT_INVERSE_PREFIXES = {
    "Countdown": (
        "cde52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2_allcs"
    ),
    "Graph coloring": (
        "gce52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2"
    ),
    "Python factors": (
        "pye52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2_allcs"
    ),
}
E52_STAGE_A_PREFIXES = {
    "Countdown": (
        "cde52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1_allcs"
    ),
    "Graph coloring": (
        "gce52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1"
    ),
    "Python factors": (
        "pye52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1_allcs"
    ),
}
E52_STAGE_A_IDENTITY = (
    "e52_direct_inverse_entropy_canonical_05b_stage_a_v1_identity.json"
)
E53_SENTINEL_PREFIXES = {
    "Countdown": "cde53_verified_replay_05b_50ep_sentinel_allcs",
    "Graph coloring": "gce53_verified_replay_05b_50ep_sentinel",
    "Python factors": "pye53_verified_replay_05b_50ep_sentinel_allcs",
}
E53_SENTINEL_IDENTITY = "e53_verified_replay_05b_sentinel_identity.json"
E55_SENTINEL_PREFIXES = {
    "Countdown": (
        "cde55_per_rollout_verified_anchor_05b_50ep_sentinel_allcs"
    ),
    "Graph coloring": (
        "gce55_per_rollout_verified_anchor_05b_50ep_sentinel"
    ),
    "Python factors": (
        "pye55_per_rollout_verified_anchor_05b_50ep_sentinel_allcs"
    ),
}
E55_SENTINEL_IDENTITY = "e55_per_rollout_verified_anchor_identity.json"
E56_SENTINEL_PREFIXES = {
    "Countdown": "cde56_open_set_split_canonical_05b_50ep_sentinel_allcs",
    "Graph coloring": "gce56_open_set_split_canonical_05b_50ep_sentinel",
    "Python factors": (
        "pye56_open_set_split_canonical_05b_50ep_sentinel_allcs"
    ),
}
E56_SENTINEL_IDENTITY = (
    "e56_open_set_split_canonical_05b_sentinel_identity.json"
)
E57_SENTINEL_PREFIXES = {
    "Countdown": (
        "cde57_verified_first_split_canonical_05b_50ep_sentinel_allcs"
    ),
    "Graph coloring": (
        "gce57_verified_first_split_canonical_05b_50ep_sentinel"
    ),
    "Python factors": (
        "pye57_verified_first_split_canonical_05b_50ep_sentinel_allcs"
    ),
}
E57_SENTINEL_IDENTITY = (
    "e57_verified_first_split_canonical_05b_sentinel_identity.json"
)
E58_SENTINEL_PREFIXES = {
    "Countdown": (
        "cde58_global_verified_replay_canonical_05b_50ep_sentinel_allcs"
    ),
    "Graph coloring": (
        "gce58_global_verified_replay_canonical_05b_50ep_sentinel"
    ),
    "Python factors": (
        "pye58_global_verified_replay_canonical_05b_50ep_sentinel_allcs"
    ),
}
E58_SENTINEL_IDENTITY = (
    "e58_global_verified_replay_canonical_05b_sentinel_identity.json"
)
E59_MATHIR_PREFIX = "mie59_mathir_global_verified_replay_05b_50ep"
E59_MATHIR_IDENTITY = (
    "e59_mathir_global_verified_replay_matched_identity.json"
)
E53_STAGE_A_PREFIXES = {
    "Countdown": "cde53_verified_replay_05b_50ep_stage_a_allcs",
    "Graph coloring": "gce53_verified_replay_05b_50ep_stage_a",
    "Python factors": "pye53_verified_replay_05b_50ep_stage_a_allcs",
}
E53_STAGE_A_IDENTITY = "e53_verified_replay_05b_stage_a_identity.json"
STEP_ZERO_PAIRING_FIELDS = tuple(
    field
    for metric in ("greedy", "mean8", "pass8", "coverage8", "distinct8")
    for field in (
        metric,
        f"{metric}_draws",
        f"{metric}_draw_std",
        f"{metric}_draw_se",
        f"{metric}_draw_min",
        f"{metric}_draw_max",
    )
)


@dataclass(frozen=True)
class RunSpec:
    environment: str
    scale: str
    method: str
    arm: str
    stamp_prefix: str
    prompt_pool_size: int
    num_samples: int
    settled_when_idle: bool = False
    metric_prefixes: tuple[str, ...] = ()
    empty_status: str | None = None
    eval_key: str = EVAL_KEY
    max_passes: float = MAX_PASSES
    seeds: tuple[int, ...] = SEEDS

    def run_stamp(self, seed: int) -> str:
        return f"{self.stamp_prefix}_{self.arm}_s{seed}"

    def metric_run_stamps(self, seed: int) -> tuple[str, ...]:
        prefixes = (self.stamp_prefix, *self.metric_prefixes)
        return tuple(f"{prefix}_{self.arm}_s{seed}" for prefix in prefixes)


@dataclass
class MetricState:
    offset: int = 0
    pending: bytes = b""
    landed_passes: float | None = None
    latest_step: int | None = None
    latest_passes: float | None = None
    mtime: float = 0.0


@dataclass(frozen=True)
class RunMetrics:
    landed_passes: float | None
    latest_step: int | None
    latest_passes: float | None
    latest_mtime: float | None
    furthest_passes: float | None = None


@dataclass(frozen=True)
class JobInfo:
    job_id: str
    state: str
    reason: str = ""
    start_time: float | None = None


def campaign_specs() -> list[RunSpec]:
    """Return table rows in their intended display order."""
    specs: list[RunSpec] = []

    def add_cell(
        environment: str,
        scale: str,
        maxent_prefix: str,
        maxent_pool: int,
        maxent_samples: int,
        canonical_drgrpo_prefix: str | None = None,
    ) -> None:
        # E16 is being rebuilt as an E15-derived canonical-policy extension.
        # Keep its requested 0.5B cells visible, but do not count an unfinished
        # protocol design as queued compute. Larger scales remain out of scope.
        maxent_empty_status = "DESIGN" if scale == "0.5B" else "HELD"
        cell_specs: list[RunSpec] = []
        if canonical_drgrpo_prefix is not None:
            cell_specs.append(
                RunSpec(
                    environment,
                    scale,
                    "matched canonical Dr.GRPO",
                    "grpo",
                    canonical_drgrpo_prefix,
                    maxent_pool,
                    maxent_samples,
                )
            )
        cell_specs.extend(
            [
                RunSpec(
                    environment,
                    scale,
                    "Standard MaxEnt fixed",
                    "maxent",
                    maxent_prefix,
                    maxent_pool,
                    maxent_samples,
                    empty_status=maxent_empty_status,
                ),
                RunSpec(
                    environment,
                    scale,
                    "Standard MaxEnt proportional",
                    "maxent_control",
                    maxent_prefix,
                    maxent_pool,
                    maxent_samples,
                    empty_status=maxent_empty_status,
                ),
                RunSpec(
                    environment,
                    scale,
                    "Standard MaxEnt Haarnoja dual",
                    "maxent_dual",
                    maxent_prefix,
                    maxent_pool,
                    maxent_samples,
                    empty_status=maxent_empty_status,
                ),
            ]
        )
        if scale == "0.5B":
            freeform_control_prefix = {
                "Countdown": "cde22_freeform_conditional_dual_05b_v2",
                "Graph coloring": "gce22_freeform_conditional_dual_05b_v2",
            }.get(environment)
            freeform_treatment_prefix = {
                "Countdown": "cde27_freeform_conditional_dual_05b_v1",
                "Graph coloring": "gce27_freeform_conditional_dual_05b_v1",
            }.get(environment)
            if freeform_control_prefix is not None and freeform_treatment_prefix is not None:
                cell_specs.extend(
                    [
                        RunSpec(
                            environment,
                            scale,
                            "matched free-form Dr.GRPO",
                            "grpo",
                            freeform_control_prefix,
                            maxent_pool,
                            maxent_samples,
                        ),
                        RunSpec(
                            environment,
                            scale,
                            "Free-form conditional-token MaxEnt base-preserving Haarnoja dual (125% entropy target)",
                            "maxent_dual",
                            freeform_treatment_prefix,
                            maxent_pool,
                            maxent_samples,
                        ),
                    ]
                )
        elif scale == "3B":
            freeform_control_prefix = {
                "Countdown": "cde28_freeform_drgrpo_3b_repair_v2",
                "Graph coloring": "gce28_freeform_drgrpo_3b_repair_v2",
            }.get(environment)
            freeform_treatment_prefix = {
                "Countdown": "cde25_freeform_conditional_dual_3b_repair_v2",
                "Graph coloring": "gce25_freeform_conditional_dual_3b_repair_v2",
            }.get(environment)
            if freeform_control_prefix is not None and freeform_treatment_prefix is not None:
                cell_specs.extend(
                    [
                        RunSpec(
                            environment,
                            scale,
                            "matched free-form Dr.GRPO",
                            "grpo",
                            freeform_control_prefix,
                            maxent_pool,
                            16,
                        ),
                        RunSpec(
                            environment,
                            scale,
                            "Free-form conditional-token MaxEnt base-preserving Haarnoja dual (125% entropy target)",
                            "maxent_dual",
                            freeform_treatment_prefix,
                            maxent_pool,
                            16,
                        ),
                    ]
                )
        elif scale == "7B":
            freeform_prefix = {
                "Countdown": "cde29_freeform_7b_4gpu_v5_buffer_restore",
                "Graph coloring": "gce29_freeform_7b_4gpu_v5_buffer_restore",
            }.get(environment)
            if freeform_prefix is not None:
                cell_specs.extend(
                    [
                        RunSpec(
                            environment,
                            scale,
                            "matched free-form Dr.GRPO",
                            "grpo",
                            freeform_prefix,
                            maxent_pool,
                            16,
                        ),
                        RunSpec(
                            environment,
                            scale,
                            "Free-form conditional-token MaxEnt base-preserving Haarnoja dual (125% entropy target)",
                            "maxent_dual",
                            freeform_prefix,
                            maxent_pool,
                            16,
                        ),
                    ]
                )
        specs.extend(cell_specs)

    add_cell(
        "Countdown",
        "0.5B",
        "cde16_canonical_maxent_05b_v2",
        384,
        16,
        canonical_drgrpo_prefix="cde19_canonical_drgrpo_05b_v1",
    )
    add_cell(
        "Countdown",
        "3B",
        "cde17_canonical_maxent_3b_v5",
        384,
        16,
        canonical_drgrpo_prefix="cde18_canonical_drgrpo_3b_v1",
    )
    add_cell(
        "Countdown",
        "7B",
        "cde23_canonical_maxent_7b_v6_4xa100_evalsync_fix",
        384,
        16,
    )
    add_cell(
        "Graph coloring",
        "0.5B",
        "gce16_canonical_maxent_05b_v2",
        192,
        16,
        canonical_drgrpo_prefix="gce19_canonical_drgrpo_05b_v1",
    )
    add_cell(
        "Graph coloring",
        "3B",
        "gce17_canonical_maxent_3b_v5",
        192,
        16,
        canonical_drgrpo_prefix="gce18_canonical_drgrpo_3b_v1",
    )
    add_cell(
        "Graph coloring",
        "7B",
        "gce24_canonical_maxent_7b_v4_4xa100_evalsync_fix",
        192,
        16,
    )
    for method, arm in (
        ("Dr.GRPO", "grpo"),
        ("Standard MaxEnt fixed", "maxent"),
        ("Standard MaxEnt proportional", "maxent_control"),
        ("Standard MaxEnt Haarnoja dual", "maxent_dual"),
    ):
        specs.append(
            RunSpec(
                "MATH-500 (free-form)",
                "0.5B",
                method,
                arm,
                "mte21_math_conditional_token_05b_v4",
                8515,
                16,
                eval_key=MATH_EVAL_KEY,
                max_passes=1.0,
            )
        )
    specs.append(
        RunSpec(
            "MATH-500 (free-form)",
            "0.5B",
            "Free-form conditional-token MaxEnt base-preserving Haarnoja dual (125% entropy target)",
            "maxent_dual",
            "mte26_math_freeform_conditional_dual_high_entropy_05b_v2",
            8515,
            16,
            eval_key=MATH_EVAL_KEY,
            max_passes=1.0,
        )
    )
    return specs


def latest_05b_specs() -> list[RunSpec]:
    """Return maintained reruns plus the current clean 0.5B experiments."""

    specs = []
    for scale, prefixes in (("0.5B", LATEST_05B_PREFIXES), ("3B", LATEST_3B_PREFIXES)):
        for environment, prompt_pool_size in (("Countdown", 384), ("Graph coloring", 192)):
            prefix = prefixes[environment]
            specs.extend(
                [
                    RunSpec(environment, scale, "free-form Dr.GRPO", "grpo", prefix, prompt_pool_size, 16, max_passes=10.0),
                    RunSpec(environment, scale, "free-form EMA-Haarnoja MaxEnt", "maxent_dual", prefix, prompt_pool_size, 16, max_passes=10.0),
                ]
            )
            if scale == "0.5B":
                pilot_prefix = {
                    "Countdown": "cde36_answer_option_mi_dual_eval_05b_v1",
                    "Graph coloring": "gce36_answer_option_mi_dual_eval_05b_v1",
                }[environment]
                specs.extend(
                    [
                        RunSpec(
                            environment,
                            scale,
                            "free-form Dr.GRPO (E36 neutral)",
                            "grpo",
                            pilot_prefix,
                            prompt_pool_size,
                            16,
                            max_passes=10.0,
                            seeds=(3601, 3602, 3603),
                        ),
                        RunSpec(
                            environment,
                            scale,
                            "free-form answer-option MI (E36 neutral)",
                            "diayn",
                            pilot_prefix,
                            prompt_pool_size,
                            16,
                            max_passes=10.0,
                            seeds=(3601, 3602, 3603),
                        ),
                    ]
                )
                outcome_prefix = {
                    "Countdown": "cde37_outcome_collision_05b_v1",
                    "Graph coloring": "gce37_outcome_collision_05b_v1",
                }[environment]
                specs.extend(
                    [
                        RunSpec(
                            environment,
                            scale,
                            "free-form Dr.GRPO (E37 matched)",
                            "grpo",
                            outcome_prefix,
                            prompt_pool_size,
                            16,
                            max_passes=10.0,
                        ),
                        RunSpec(
                            environment,
                            scale,
                            "free-form semantic collision entropy (E37)",
                            "outcome_collision",
                            outcome_prefix,
                            prompt_pool_size,
                            16,
                            max_passes=10.0,
                        ),
                    ]
                )
                shannon_prefix = {
                    "Countdown": "cde38_semantic_shannon_05b_v1",
                    "Graph coloring": "gce38_semantic_shannon_05b_v1",
                }[environment]
                specs.append(
                    RunSpec(
                        environment,
                        scale,
                        "free-form predictive semantic Shannon entropy (E38)",
                        "semantic_shannon",
                        shannon_prefix,
                        prompt_pool_size,
                        16,
                        max_passes=10.0,
                    )
                )
                specs.append(
                    RunSpec(
                        environment,
                        scale,
                        "separately-centered semantic Shannon advantage (E41)",
                        E41_ARM,
                        E41_PREFIXES[environment],
                        prompt_pool_size,
                        16,
                        max_passes=10.0,
                    )
                )
                specs.append(
                    RunSpec(
                        environment,
                        scale,
                        "success-conditioned signed Shannon advantage (E43)",
                        E43_ARM,
                        E43_PREFIXES[environment],
                        prompt_pool_size,
                        16,
                        max_passes=10.0,
                    )
                )
    for method, arm in (
        ("free-form Dr.GRPO (E39 matched)", "grpo"),
        ("free-form semantic collision entropy (E39)", "outcome_collision"),
        (
            "free-form predictive semantic Shannon entropy (E39)",
            "semantic_shannon",
        ),
    ):
        specs.append(
            RunSpec(
                E39_ENVIRONMENT,
                "0.5B",
                method,
                arm,
                E39_PREFIX,
                384,
                16,
                eval_key=MATH_EVAL_KEY,
                max_passes=10.0,
            )
        )
    specs.append(
        RunSpec(
            E39_ENVIRONMENT,
            "0.5B",
            "separately-centered semantic Shannon advantage (E41)",
            E41_ARM,
            E41_MATH_PREFIX,
            384,
            16,
            eval_key=MATH_EVAL_KEY,
            max_passes=10.0,
        )
    )
    specs.append(
        RunSpec(
            E39_ENVIRONMENT,
            "0.5B",
            "success-conditioned signed Shannon advantage (E43)",
            E43_ARM,
            E43_MATH_PREFIX,
            384,
            16,
            eval_key=MATH_EVAL_KEY,
            max_passes=10.0,
        )
    )
    return specs


def e37_05b_specs() -> list[RunSpec]:
    """Return retained comparators plus the active E41/E43 extensions."""

    return [
        spec
        for spec in latest_05b_specs()
        if (
            "e37_outcome_collision_05b" in spec.stamp_prefix
            or "e38_semantic_shannon_05b" in spec.stamp_prefix
            or spec.stamp_prefix == E39_PREFIX
            or spec.stamp_prefix in {*E41_PREFIXES.values(), E41_MATH_PREFIX}
            or spec.stamp_prefix in {*E43_PREFIXES.values(), E43_MATH_PREFIX}
        )
    ]


def e46_specs(
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> list[RunSpec]:
    """Return the active canonical campaign, including E59 MathIR."""

    specs: list[RunSpec] = []
    e59_active = (artifact_root / E59_MATHIR_IDENTITY).is_file()
    e58_active = (artifact_root / E58_SENTINEL_IDENTITY).is_file()
    e57_active = (artifact_root / E57_SENTINEL_IDENTITY).is_file()
    e56_active = (artifact_root / E56_SENTINEL_IDENTITY).is_file()
    e55_active = (artifact_root / E55_SENTINEL_IDENTITY).is_file()
    e53_stage_a_active = (artifact_root / E53_STAGE_A_IDENTITY).is_file()
    e53_active = e53_stage_a_active or (
        artifact_root / E53_SENTINEL_IDENTITY
    ).is_file()
    stage_a_active = (
        artifact_root / E52_STAGE_A_IDENTITY
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
    if e58_active:
        prefixes = E58_SENTINEL_PREFIXES
        control_prefixes = E53_SENTINEL_PREFIXES
        seeds = (9010,)
        treatment_method = "global verified-replay canonical control"
        treatment_arm = "verified_first_global_replay_canonical"
    elif e57_active:
        prefixes = E57_SENTINEL_PREFIXES
        control_prefixes = E53_SENTINEL_PREFIXES
        seeds = (9010,)
        treatment_method = "verified-first split canonical control"
        treatment_arm = "verified_first_split_canonical"
    elif e56_active:
        prefixes = E56_SENTINEL_PREFIXES
        control_prefixes = E53_SENTINEL_PREFIXES
        seeds = (9010,)
        treatment_method = "open-set split canonical control"
        treatment_arm = "open_set_split_canonical"
    elif e55_active:
        prefixes = E55_SENTINEL_PREFIXES
        control_prefixes = E53_SENTINEL_PREFIXES
        seeds = (9010,)
        treatment_method = "inverse discovery + per-rollout verified anchor"
        treatment_arm = "maxent_inverse_canonical_replay"
    elif e53_active:
        prefixes = (
            E53_STAGE_A_PREFIXES
            if e53_stage_a_active
            else E53_SENTINEL_PREFIXES
        )
        control_prefixes = prefixes
        seeds = (43, 44, 45) if e53_stage_a_active else (9010,)
        treatment_method = "inverse entropy + verified replay"
        treatment_arm = "maxent_inverse_canonical_replay"
    else:
        prefixes = (
            E52_STAGE_A_PREFIXES
            if stage_a_active
            else E52_DIRECT_INVERSE_PREFIXES
        )
        control_prefixes = prefixes
        seeds = (43, 44, 45) if stage_a_active else (9009,)
        treatment_method = "inverse entropy + fixed canonical"
        treatment_arm = "maxent_inverse_canonical"
    for environment, prompt_pool_size in (
        ("Countdown", 384),
        ("Graph coloring", 192),
        ("Python factors", 384),
    ):
        prefix = prefixes[environment]
        control_prefix = control_prefixes[environment]
        specs.append(
            RunSpec(
                environment,
                "0.5B",
                "matched Dr.GRPO",
                "grpo",
                control_prefix,
                prompt_pool_size,
                16,
                empty_status="READY",
                max_passes=50.0,
                seeds=seeds,
            )
        )
        if not e55_active and not e56_active and not e57_active and not e58_active:
            specs.append(
                RunSpec(
                    environment,
                    "0.5B",
                    "unbounded inverse conditional entropy",
                    "maxent_inverse",
                    prefix,
                    prompt_pool_size,
                    16,
                    empty_status="READY",
                    max_passes=50.0,
                    seeds=seeds,
                )
            )
        specs.append(
            RunSpec(
                environment,
                "0.5B",
                treatment_method,
                treatment_arm,
                prefix,
                prompt_pool_size,
                16,
                empty_status="READY",
                max_passes=50.0,
                seeds=seeds,
            )
        )
    if e59_active:
        for method, arm in (
            ("matched Dr.GRPO", "grpo"),
            (
                "E59 global verified-replay canonical",
                "verified_first_global_replay_canonical",
            ),
        ):
            specs.append(
                RunSpec(
                    "Executable MathIR (action-menu eval; not MATH-500)",
                    "0.5B",
                    method,
                    arm,
                    E59_MATHIR_PREFIX,
                    384,
                    16,
                    empty_status="READY",
                    max_passes=50.0,
                    seeds=(43, 44, 45),
                )
            )
    return specs


def e44_ogs_specs(
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> list[RunSpec]:
    """Backward-compatible alias for the current canonical live view."""

    return e46_specs(artifact_root)


def semantic_entropy_step_zero_pairing_status(artifact_root: Path) -> str:
    """Summarize exact step-zero agreement for every active five-arm cohort."""

    def load_rows(
        prefix: str,
        split: str,
    ) -> dict[tuple[str, int], dict]:
        path = artifact_root / f"{prefix}_scaling_curve.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, list):
            return {}
        rows: dict[tuple[str, int], dict] = {}
        for row in payload:
            if (
                isinstance(row, dict)
                and row.get("split") == split
                and row.get("step") == 0
            ):
                try:
                    seed = int(row["seed"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows[(str(row.get("arm")), seed)] = row
        return rows

    task_cohorts = (
        (
            "Countdown",
            "E37/E38/E41/E43",
            "multi_answer",
            {
                "grpo": "cde37_outcome_collision_05b_v1",
                "outcome_collision": "cde37_outcome_collision_05b_v1",
                "semantic_shannon": "cde38_semantic_shannon_05b_v1",
                E41_ARM: E41_PREFIXES["Countdown"],
                E43_ARM: E43_PREFIXES["Countdown"],
            },
        ),
        (
            "Graph",
            "E37/E38/E41/E43",
            "multi_answer",
            {
                "grpo": "gce37_outcome_collision_05b_v1",
                "outcome_collision": "gce37_outcome_collision_05b_v1",
                "semantic_shannon": "gce38_semantic_shannon_05b_v1",
                E41_ARM: E41_PREFIXES["Graph coloring"],
                E43_ARM: E43_PREFIXES["Graph coloring"],
            },
        ),
        (
            "MATH",
            "E39/E41/E43",
            "math",
            {
                "grpo": E39_PREFIX,
                "outcome_collision": E39_PREFIX,
                "semantic_shannon": E39_PREFIX,
                E41_ARM: E41_MATH_PREFIX,
                E43_ARM: E43_MATH_PREFIX,
            },
        ),
    )
    completed = {"E37/E38/E41/E43": 0, "E39/E41/E43": 0}
    mismatches: list[str] = []
    for task, cohort, split, arm_prefixes in task_cohorts:
        rows_by_prefix = {
            prefix: load_rows(prefix, split)
            for prefix in set(arm_prefixes.values())
        }
        for seed in SEEDS:
            rows = tuple(
                rows_by_prefix[prefix].get((arm, seed))
                for arm, prefix in arm_prefixes.items()
            )
            if any(row is None for row in rows):
                continue
            completed[cohort] += 1
            signatures = tuple(
                tuple(row.get(field) for field in STEP_ZERO_PAIRING_FIELDS)
                for row in rows
                if row is not None
            )
            if any(signature != signatures[0] for signature in signatures[1:]):
                mismatches.append(f"{task}/s{seed}")

    if mismatches:
        return "FAIL exact mismatch: " + ", ".join(mismatches)
    legacy_completed = completed["E37/E38/E41/E43"]
    math_completed = completed["E39/E41/E43"]
    if legacy_completed == 6 and math_completed == 3:
        return "PASS (9/9 exact across all matched five-arm cohorts)"
    if legacy_completed == 6:
        return (
            "PASS E37/E38/E41/E43 (6/6 exact); "
            f"E39/E41/E43 pending ({math_completed}/3 complete five-arm seeds)"
        )
    if math_completed == 3:
        return (
            f"E37/E38/E41/E43 pending ({legacy_completed}/6); "
            "E39/E41/E43 PASS (3/3 exact five-arm seeds)"
        )
    return (
        f"pending (E37/E38/E41/E43 {legacy_completed}/6; "
        f"E39/E41/E43 {math_completed}/3 complete five-arm seeds)"
    )


def e38_step_zero_pairing_status(artifact_root: Path) -> str:
    """Backward-compatible name for the active semantic-entropy pairing."""

    return semantic_entropy_step_zero_pairing_status(artifact_root)


def e16_smoke_specs() -> list[RunSpec]:
    """Return the corrected E16 canonical six-cell engineering smoke."""
    specs: list[RunSpec] = []
    for environment, prefix, pool in (
        (
            "Countdown",
            "cde16_canonical_maxent_joint_smoke_v3",
            384,
        ),
        (
            "Graph coloring",
            "gce16_canonical_maxent_joint_smoke_v3",
            192,
        ),
    ):
        for method, arm in (
            ("fixed", "maxent"),
            ("proportional", "maxent_control"),
            ("Haarnoja dual", "maxent_dual"),
        ):
            specs.append(
                RunSpec(
                    environment,
                    "0.5B smoke",
                    method,
                    arm,
                    prefix,
                    pool,
                    16,
                )
            )
    return specs


def e16_protocol_is_frozen(path: Path = E16_PROTOCOL) -> bool:
    """Read E16's prospective status without inferring submission from design."""
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("**Status:"):
                    return line.startswith("**Status: FROZEN")
    except OSError:
        pass
    return False


class MetricCache:
    """Incrementally read metrics files instead of rescanning every tick."""

    def __init__(self) -> None:
        self._files: dict[Path, MetricState] = {}

    def _read_file(self, path: Path, spec: RunSpec) -> MetricState:
        state = self._files.setdefault(path, MetricState())
        try:
            stat = path.stat()
        except FileNotFoundError:
            return state
        if stat.st_size < state.offset:
            state = MetricState()
            self._files[path] = state

        with path.open("rb") as handle:
            handle.seek(state.offset)
            chunk = handle.read()
            state.offset = handle.tell()
        if chunk:
            records = (state.pending + chunk).split(b"\n")
            state.pending = records.pop()
            eval_key_bytes = spec.eval_key.encode()
            # Only evaluation records affect the durable frontier, and only
            # the final record in this new chunk affects live progress. This
            # keeps a cold dashboard start fast even with long histories.
            for index, raw in enumerate(records):
                if not raw.strip():
                    continue
                if eval_key_bytes not in raw and index != len(records) - 1:
                    continue
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                step = record.get("misc/global_step", record.get("trainer/global_step"))
                consumed = record.get("misc/prompt_consumed")
                if step is not None:
                    state.latest_step = int(step)
                if consumed is not None:
                    passes = float(consumed) / (
                        spec.num_samples * spec.prompt_pool_size
                    )
                    state.latest_passes = passes
                    if (
                        spec.eval_key in record
                        and passes <= spec.max_passes + 1e-9
                    ):
                        if (
                            state.landed_passes is None
                            or passes > state.landed_passes
                        ):
                            state.landed_passes = passes
        state.mtime = stat.st_mtime
        return state

    def run_metrics(
        self, run_dirs: Path | list[Path] | tuple[Path, ...] | None, spec: RunSpec
    ) -> RunMetrics:
        if run_dirs is None:
            return RunMetrics(None, None, None, None)
        if isinstance(run_dirs, Path):
            run_dirs = [run_dirs]
        states: list[MetricState] = []
        for run_dir in run_dirs:
            for path in sorted(run_dir.glob("debug_*/train_metrics.jsonl")):
                states.append(self._read_file(path, spec))
        if not states:
            return RunMetrics(None, None, None, None)
        landed = [s.landed_passes for s in states if s.landed_passes is not None]
        furthest = [s.latest_passes for s in states if s.latest_passes is not None]
        current = max(states, key=lambda state: state.mtime)
        return RunMetrics(
            max(landed) if landed else None,
            current.latest_step,
            current.latest_passes,
            current.mtime or None,
            max(furthest) if furthest else None,
        )


def find_run_dirs(data_root: Path, stamps: set[str]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    if not data_root.exists():
        return found
    for path in data_root.iterdir():
        if not path.is_dir():
            continue
        for stamp in stamps:
            if path.name.endswith(f"_{stamp}"):
                found.setdefault(stamp, []).append(path)
                break
    return found


def command_output(command: list[str]) -> tuple[str, str | None]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return "", str(exc)
    if result.returncode != 0:
        return "", result.stderr.strip() or f"exit status {result.returncode}"
    return result.stdout, None


def parse_slurm_time(value: str) -> float | None:
    if not value or value in {"N/A", "Unknown"}:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def query_queue(user: str) -> tuple[dict[str, JobInfo], str | None]:
    output, error = command_output(
        ["squeue", "-u", user, "-h", "-o", "%i|%T|%R|%S|%j"]
    )
    jobs: dict[str, JobInfo] = {}
    for line in output.splitlines():
        fields = line.split("|", 4)
        if len(fields) != 5 or fields[4] != "xdr_train":
            continue
        job_id, state, reason, start, _ = fields
        jobs[job_id] = JobInfo(job_id, state.upper(), reason, parse_slurm_time(start))
    return jobs, error


def parse_run_stamp_from_scontrol(output: str) -> str | None:
    match = re.search(r"(?:^|[, ])RUN_STAMP=([^, ]+)", output)
    return match.group(1) if match else None


def load_job_stamp_map(
    artifact_root: Path, target_stamps: set[str]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for manifest in artifact_root.glob("*_comparative_jobs.tsv"):
        try:
            with manifest.open(newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    stamp = row.get("run_stamp", "")
                    job_id = row.get("job_id", "").split(";", 1)[0]
                    if stamp in target_stamps and job_id:
                        mapping[job_id] = stamp
        except (OSError, csv.Error):
            continue

    save_path_pattern = re.compile(rb"\[experiment\] save_path=([^\r\n]+)")
    log_root = artifact_root / "logs"
    for log in log_root.glob("xdr_train-*.out"):
        match_id = re.search(r"-(\d+)\.out$", log.name)
        if not match_id:
            continue
        try:
            with log.open("rb") as handle:
                head = handle.read(131072)
        except OSError:
            continue
        match = save_path_pattern.search(head)
        if not match:
            continue
        save_path = match.group(1).decode(errors="replace")
        stamp = next(
            (
                target
                for target in target_stamps
                if save_path.endswith(f"_{target}")
            ),
            None,
        )
        if stamp is not None:
            mapping[match_id.group(1)] = stamp
    return mapping


def fill_active_job_stamps(
    queue: dict[str, JobInfo], job_stamps: dict[str, str], target_stamps: set[str]
) -> None:
    """Resolve not-yet-started replacement jobs from Slurm's SubmitLine."""
    for job_id in queue:
        if job_id in job_stamps:
            continue
        output, _ = command_output(["scontrol", "show", "job", "-o", job_id])
        stamp = parse_run_stamp_from_scontrol(output)
        if stamp in target_stamps:
            job_stamps[job_id] = stamp


def query_terminal_jobs(job_ids: set[str]) -> dict[str, JobInfo]:
    if not job_ids:
        return {}
    output, _ = command_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(sorted(job_ids, key=lambda value: int(value))),
            "-o",
            "JobIDRaw,State,End",
        ]
    )
    jobs: dict[str, JobInfo] = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 3:
            continue
        job_id, state, end = fields[:3]
        if job_id not in job_ids:
            continue
        jobs[job_id] = JobInfo(
            job_id,
            state.split()[0].upper(),
            "",
            parse_slurm_time(end),
        )
    return jobs


def select_active_job(jobs: list[JobInfo]) -> JobInfo | None:
    if not jobs:
        return None
    priority = {
        "RUNNING": 5,
        "COMPLETING": 4,
        "CONFIGURING": 3,
        "PENDING": 2,
        "SUSPENDED": 1,
    }
    return max(jobs, key=lambda job: (priority.get(job.state, 0), int(job.job_id)))


def terminal_label(state: str) -> str:
    if state == "COMPLETED":
        return "done"
    if state in {"FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}:
        return "FAIL"
    if state == "TIMEOUT":
        return "TIME"
    if state == "CANCELLED":
        return "cancel"
    if state == "PREEMPTED":
        return "preempt"
    return state.lower() if state else "—"


def live_label(
    active: JobInfo | None,
    terminal: JobInfo | None,
    metrics: RunMetrics,
    max_passes: float = MAX_PASSES,
) -> str:
    if active is not None:
        if active.state == "PENDING":
            return "P"
        if active.state == "CONFIGURING":
            return "CF"
        if active.state == "COMPLETING":
            return "CG"
        if active.state != "RUNNING":
            return active.state[:4]
        # A replacement in startup must not inherit the old attempt's live step.
        if (
            metrics.latest_mtime is None
            or (
                active.start_time is not None
                and metrics.latest_mtime < active.start_time - 120
            )
            or metrics.latest_step is None
            or metrics.latest_passes is None
        ):
            return "R init"
        return f"R{metrics.latest_step}@{metrics.latest_passes:.2f}"
    if metrics.landed_passes is not None and metrics.landed_passes >= max_passes - 1e-9:
        return "done"
    if terminal is not None:
        return terminal_label(terminal.state)
    return "—"


def format_landed(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def completion_credit(
    label: str,
    metrics: RunMetrics,
    max_passes: float = MAX_PASSES,
) -> float:
    """Endpoint-normalized credit that cannot regress across restarts."""
    if label == "done":
        return max_passes
    candidates = [
        value
        for value in (metrics.landed_passes, metrics.furthest_passes)
        if value is not None
    ]
    return min(max_passes, max(0.0, max(candidates, default=0.0)))


def progress_bar(percent: float, width: int = 30) -> str:
    filled = round(width * min(100.0, max(0.0, percent)) / 100.0)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def render_table(rows: list[list[str]]) -> str:
    header = [
        "Environment",
        "Scale",
        "Method",
        "Landed passes",
        "Live status",
    ]
    all_rows = [header, *rows]
    widths = [max(len(row[index]) for row in all_rows) for index in range(len(header))]

    def render(row: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("─" * width for width in widths)
    return "\n".join([render(header), separator, *(render(row) for row in rows)])


class Dashboard:
    def __init__(
        self,
        data_root: Path,
        artifact_root: Path,
        stale_minutes: int,
        *,
        latest_05b_only: bool = False,
        e37_only: bool = False,
        e44_ogs_only: bool = False,
        e46_only: bool = False,
    ):
        self.e37_only = e37_only
        self.e46_only = e46_only or e44_ogs_only
        self.latest_05b_only = (
            latest_05b_only or e37_only or self.e46_only
        )
        if self.e46_only:
            self.specs = e46_specs(artifact_root)
        elif e37_only:
            self.specs = e37_05b_specs()
        elif latest_05b_only:
            self.specs = latest_05b_specs()
        else:
            self.specs = campaign_specs()
        self.smoke_specs = [] if self.latest_05b_only else e16_smoke_specs()
        self.e16_protocol_frozen = (
            False if self.latest_05b_only else e16_protocol_is_frozen()
        )
        self.data_root = data_root
        self.artifact_root = artifact_root
        self.stale_seconds = stale_minutes * 60
        self.metric_cache = MetricCache()
        self.target_stamps = {
            stamp
            for spec in self.specs
            for seed in spec.seeds
            for stamp in spec.metric_run_stamps(seed)
        }
        self.target_stamps.update(
            spec.run_stamp(SMOKE_SEED) for spec in self.smoke_specs
        )
        self.job_stamps = load_job_stamp_map(artifact_root, self.target_stamps)
        self.previous_completion_percent: float | None = None

    def snapshot(self) -> tuple[str, list[str]]:
        queue, queue_error = query_queue(getpass.getuser())
        fill_active_job_stamps(queue, self.job_stamps, self.target_stamps)
        # Started replacements become discoverable from their output logs.
        self.job_stamps.update(
            load_job_stamp_map(self.artifact_root, self.target_stamps)
        )
        known_job_stamps = set(self.job_stamps.values())
        terminal_jobs = query_terminal_jobs(set(self.job_stamps))
        run_dirs = find_run_dirs(self.data_root, self.target_stamps)

        active_by_stamp: dict[str, list[JobInfo]] = {}
        terminal_by_stamp: dict[str, list[JobInfo]] = {}
        for job_id, stamp in self.job_stamps.items():
            if job_id in queue:
                active_by_stamp.setdefault(stamp, []).append(queue[job_id])
            elif job_id in terminal_jobs:
                terminal_by_stamp.setdefault(stamp, []).append(terminal_jobs[job_id])

        rows: list[list[str]] = []
        warnings: list[str] = []
        active_counts = {
            "RUNNING": 0,
            "PENDING": 0,
            "TRANSITIONING": 0,
            "PROBLEM": 0,
            "STOPPED": 0,
        }
        fully_landed = 0
        displayed_runs = 0
        gate_held_runs = 0
        gate_waiting_runs = 0
        completion_passes = 0.0
        completion_capacity = 0.0
        settled_runs = 0
        now = time.time()

        # The engineering smoke is operational gating evidence, not one of
        # the 18 analytical runs.  Surface its real scheduler/metric state in
        # a compact line while keeping it out of rows and completion credit.
        smoke_statuses: dict[str, list[tuple[str, str]]] = {}
        smoke_has_activity = False
        for spec in self.smoke_specs:
            stamp = spec.run_stamp(SMOKE_SEED)
            metric_dirs = run_dirs.get(stamp, [])
            metrics = self.metric_cache.run_metrics(metric_dirs, spec)
            active = select_active_job(active_by_stamp.get(stamp, []))
            terminals = terminal_by_stamp.get(stamp, [])
            terminal = max(
                terminals,
                key=lambda job: (job.start_time or 0.0, int(job.job_id)),
                default=None,
            )
            label = live_label(active, terminal, metrics, spec.max_passes)
            if label == "—":
                known_job = stamp in known_job_stamps
                if known_job:
                    # A manifest proves submission, but only Slurm can prove
                    # P/R/done.  Keep the unknown state explicit when Slurm is
                    # unavailable instead of presenting it as pending.
                    label = "submitted"
                elif metrics.latest_step is not None:
                    label = f"metrics@S{metrics.latest_step}"
                else:
                    label = "not-submitted"
            if label != "not-submitted":
                smoke_has_activity = True
            smoke_statuses.setdefault(spec.environment, []).append(
                (spec.method, label)
            )
            if active is not None and active.state == "RUNNING":
                active_counts["RUNNING"] += 1
            elif active is not None and active.state == "PENDING":
                active_counts["PENDING"] += 1
            elif active is not None:
                active_counts["TRANSITIONING"] += 1
            if label in {"FAIL", "TIME"}:
                active_counts["PROBLEM"] += 1
            elif label in {"cancel", "preempt"}:
                active_counts["STOPPED"] += 1

        if self.e16_protocol_frozen or smoke_has_activity:
            smoke_parts = []
            for environment in ("Countdown", "Graph coloring"):
                cells = "; ".join(
                    f"{method}: {label}"
                    for method, label in smoke_statuses.get(environment, [])
                )
                smoke_parts.append(f"{environment} [{cells}]")
            smoke_line = f"E16 smoke s{SMOKE_SEED} — " + " | ".join(smoke_parts)
        else:
            smoke_line = (
                "E16 canonical extension — DESIGN: E15-derived Countdown policy "
                "and exact-entropy controller preflight; no jobs submitted"
            )

        for spec in self.specs:
            landed_cells: list[str] = []
            live_cells: list[str] = []
            row_completion: list[float] = []
            row_settled = 0
            for seed in spec.seeds:
                stamp = spec.run_stamp(seed)
                metric_dirs = [
                    run_dir
                    for metric_stamp in spec.metric_run_stamps(seed)
                    for run_dir in run_dirs.get(metric_stamp, [])
                ]
                metrics = self.metric_cache.run_metrics(metric_dirs, spec)
                active = select_active_job(active_by_stamp.get(stamp, []))
                terminals = terminal_by_stamp.get(stamp, [])
                terminal = max(
                    terminals,
                    key=lambda job: (job.start_time or 0.0, int(job.job_id)),
                    default=None,
                )
                landed_cells.append(format_landed(metrics.landed_passes))
                label = live_label(active, terminal, metrics, spec.max_passes)
                known_job = stamp in known_job_stamps
                if label == "—" and known_job:
                    # The normal P/R/done labels take precedence whenever
                    # Slurm resolves the job.  A manifest-only job is known to
                    # have been submitted, but is not falsely called pending.
                    label = "submitted"
                elif label == "—" and metrics.latest_step is not None:
                    label = f"metrics@S{metrics.latest_step}"
                if (
                    label == "—"
                    and spec.empty_status is not None
                    and metrics.latest_step is None
                    and metrics.furthest_passes is None
                ):
                    # Keep smoke-gated analytical cells visible without
                    # pretending that they are queued or counting them as
                    # unfinished work in the active campaign denominator.
                    label = spec.empty_status
                if (
                    active is None
                    and spec.settled_when_idle
                    and metrics.landed_passes is not None
                ):
                    # The 0.5B/3B controls are intentionally reused landed
                    # curves; their old allocations were often cancelled
                    # during cleanup rather than being live campaign failures.
                    label = "done"
                live_cells.append(label)
                row_completion.append(
                    completion_credit(label, metrics, spec.max_passes)
                )
                if label == "HELD":
                    gate_held_runs += 1
                elif label in {"GATE", "DESIGN"}:
                    gate_waiting_runs += 1
                if label == "done":
                    row_settled += 1
                if (
                    metrics.landed_passes is not None
                    and metrics.landed_passes >= spec.max_passes - 1e-9
                ):
                    fully_landed += 1
                if active is not None and active.state == "RUNNING":
                    active_counts["RUNNING"] += 1
                    current_attempt_has_metrics = (
                        metrics.latest_mtime is not None
                        and (
                            active.start_time is None
                            or metrics.latest_mtime >= active.start_time - 120
                        )
                    )
                    age = now - metrics.latest_mtime if current_attempt_has_metrics else None
                    if age is not None and age >= self.stale_seconds:
                        warnings.append(
                            f"{stamp}: no metrics write for {age / 60:.0f}m"
                        )
                    elif (
                        not current_attempt_has_metrics
                        and active.start_time is not None
                        and now - active.start_time >= 3600
                    ):
                        warnings.append(f"{stamp}: startup has no metrics after 60m")
                elif active is not None and active.state == "PENDING":
                    active_counts["PENDING"] += 1
                elif active is not None:
                    # COMPLETING/CONFIGURING/SUSPENDED remain visible in the
                    # row and must also reconcile in the header's active sum.
                    active_counts["TRANSITIONING"] += 1
                if label in {"FAIL", "TIME"}:
                    active_counts["PROBLEM"] += 1
                elif label in {"cancel", "preempt"}:
                    active_counts["STOPPED"] += 1
            if any(value != "—" for value in landed_cells + live_cells):
                included = [label not in {"HELD", "DESIGN"} for label in live_cells]
                displayed_runs += sum(included)
                completion_capacity += sum(
                    spec.max_passes for include in included if include
                )
                completion_passes += sum(
                    credit
                    for credit, include in zip(row_completion, included)
                    if include
                )
                settled_runs += row_settled
                rows.append(
                    [
                        spec.environment,
                        spec.scale,
                        spec.method,
                        "/".join(landed_cells),
                        " / ".join(live_cells),
                    ]
                )

        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        completion_percent = (
            100.0 * completion_passes / completion_capacity
            if completion_capacity
            else 0.0
        )
        if self.previous_completion_percent is None:
            completion_delta = "baseline"
        else:
            delta = completion_percent - self.previous_completion_percent
            completion_delta = f"{delta:+.2f} pp since last refresh"
        self.previous_completion_percent = completion_percent
        if self.e46_only:
            e59_active = any(
                spec.stamp_prefix == E59_MATHIR_PREFIX
                for spec in self.specs
            )
            e58_active = any(
                "e58_global_verified_replay_canonical" in spec.stamp_prefix
                for spec in self.specs
            )
            e57_active = any(
                "e57_verified_first_split_canonical" in spec.stamp_prefix
                for spec in self.specs
            )
            e56_active = any(
                "e56_open_set_split_canonical" in spec.stamp_prefix
                for spec in self.specs
            )
            replay_active = any(
                spec.arm == "maxent_inverse_canonical_replay"
                for spec in self.specs
            )
            e55_active = any(
                "e55_per_rollout_verified_anchor" in spec.stamp_prefix
                for spec in self.specs
            )
            e53_active = (
                replay_active
                and not e55_active
                and not e56_active
                and not e57_active
                and not e58_active
                and not e59_active
            )
            stage_a_active = not e59_active and any(
                spec.seeds == (43, 44, 45) for spec in self.specs
            )
            phase_label = (
                "three-domain sentinel + matched MathIR extension"
                if e59_active
                else "Stage A (three fresh seeds)"
                if stage_a_active
                else "global verified-replay sentinel"
                if e58_active
                else "verified-first split-controller sentinel"
                if e57_active
                else "open-set split-controller sentinel"
                if e56_active
                else "per-rollout verified-anchor sentinel"
                if e55_active
                else "verified-replay sentinel"
                if e53_active
                else "engineering sentinel"
            )
            campaign = (
                "E58 + E59"
                if e59_active
                else "E58"
                if e58_active
                else "E57"
                if e57_active
                else "E56"
                if e56_active
                else "E55"
                if e55_active
                else "E53"
                if e53_active
                else "E52"
            )
            heading = (
                f"Current canonical experiments: {campaign} ModeBench "
                f"{phase_label} 0.5B "
                f"— refreshed {timestamp}\n"
                f"Overall completion: {progress_bar(completion_percent)} "
                f"{completion_percent:.2f}%  ({completion_delta}; "
                f"{settled_runs}/{displayed_runs} settled)\n"
                f"Fully landed evals: {fully_landed}/{displayed_runs}  "
                f"Running: {active_counts['RUNNING']}  "
                f"Pending: {active_counts['PENDING']}  "
                f"Transitioning: {active_counts['TRANSITIONING']}  "
                f"Terminal failures: {active_counts['PROBLEM']}  "
                f"Stopped/cancelled: {active_counts['STOPPED']}"
            )
            footer = [
                (
                    "Landed = furthest persisted matched evaluation at or below "
                    f"each 50-pass {campaign} endpoint (K=8); Rstep@pass is the "
                    "current attempt."
                ),
                (
                    f"This tracker contains only the active {campaign} "
                    f"{phase_label} across Countdown, graph coloring, "
                    "Python factors"
                    + (", and executable MathIR" if e59_active else "")
                    + ": matched Dr.GRPO"
                    + (
                        " with a zero-gradient verified-first cold start, "
                        "checkpointed global verified replay, success-only "
                        "open-set semantic discovery, verified-mass retention, "
                        "and known-mode balance. "
                        if e58_active or e59_active
                        else
                        " with a zero-gradient verified-first cold start, "
                        "success-only open-set semantic discovery, "
                        "verified-mass retention, and known-mode balance. "
                        if e57_active
                        else
                        ", direct inverse token entropy, success-only open-set "
                        "semantic discovery, verified-mass retention, and "
                        "known-mode balance. "
                        if e56_active
                        else
                        " and the same support-independent inverse-entropy "
                        "discovery actuator plus a per-rollout verified "
                        "likelihood anchor. "
                        if e55_active
                        else
                        ", direct inverse conditional entropy, and the same "
                        "direct actuator plus verified-exemplar replay. "
                        if e53_active
                        else
                        ", direct inverse conditional entropy, and the same "
                        "direct actuator plus a fixed canonical bank. "
                    )
                    + "Superseded experiments are "
                    "omitted."
                ),
                "READY = configuration exists but no matching Slurm manifest, allocation, or metrics are currently discoverable.",
                "Hybrid bank admission is fail-closed: the learner revalidates the exact sampled output and aborts if validator admission disagrees with the actor task reward.",
                "Countdown keys come from the executed AST; graph keys from the constraint-checked vector; Python-factor keys from the return vector produced by the isolated worker.",
                (
                    "MathIR keys come only from the exact rational equation-state "
                    "trajectory produced by executing the submitted finite action "
                    "sequence; labels and prose cannot supply a key."
                    if e59_active
                    else ""
                ),
                (
                    (
                        "E58 has no direct token-entropy objective: all-zero "
                        "groups preserve the pretrained proposal distribution "
                        "with zero policy gradient until the model discovers a "
                        "validator-positive outcome. It then schedules exactly "
                        "one model-discovered verified bank per optimizer "
                        "update."
                        if e58_active or e59_active
                        else
                        "E57 has no direct token-entropy objective: all-zero "
                        "groups preserve the pretrained proposal distribution "
                        "with zero policy gradient until the model discovers a "
                        "validator-positive outcome. It then separates open-set "
                        "semantic discovery, verified mass, and known-mode "
                        "balance."
                        if e57_active
                        else
                        "The direct actuator differentiates conditional "
                        "content-token entropy at visited prefixes, excludes "
                        "EOS, and remains live before any verified canonical "
                        "support exists. "
                    )
                    + (
                        ""
                        if e57_active or e58_active or e59_active
                        else
                        "E56 gives each failure mode its own sensor--actuator "
                        "pair: conditional-token entropy for reachability, an "
                        "observed-support-plus-unseen predictive model for "
                        "discovery, verified surprisal for common mass, and "
                        "verified-bank entropy for relative balance."
                        if e56_active
                        else
                        "E55 anchors even a singleton validator-confirmed "
                        "exemplar at one pseudo-rollout in a 16-sample group; "
                        "only banks with two or more modes feed the separate "
                        "canonical-entropy controller."
                        if e55_active
                        else
                        "E53 then teacher-forces validator-confirmed exemplars, "
                        "so a previously observed mode retains a restorative "
                        "gradient even when absent from the current rollout."
                        if e53_active
                        else
                        "The hybrid independently adds fixed canonical entropy "
                        "and novelty advantages after ordinary Dr.GRPO task "
                        "centering."
                    )
                ),
                (
                    "E58 self-calibrates its three verified-outcome "
                    "coefficients during their own first 64 eligible "
                    "observations. Every coefficient is unbounded; exactly "
                    "one verified bank is replayed per update as a fixed "
                    "compute measure. Training never reads gold support, a "
                    "desired entropy/mode count, or evaluation."
                    if e58_active or e59_active
                    else
                    "E57 self-calibrates its three verified-outcome "
                    "coefficients during their own first 64 eligible "
                    "observations. Every coefficient is unbounded; the replay "
                    "pair shares one fixed 1/16 pseudo-rollout measure. "
                    "Training never reads gold support or evaluation."
                    if e57_active
                    else
                    "E56 self-calibrates all four coefficients during their "
                    "own first 64 eligible observations. Every coefficient is "
                    "unbounded; the replay pair shares one fixed 1/16 "
                    "pseudo-rollout measure. Training never reads gold support "
                    "or evaluation."
                    if e56_active
                    else
                    "E55 holds each coefficient at its preregistered reference "
                    "during its own 64 eligible observations, then applies the "
                    "run's warmup-entropy/EMA ratio. Replay alpha is unbounded; "
                    "the fixed 1/16 factor is a sampling-measure correction, "
                    "not a coefficient bound. Training never reads gold "
                    "support or evaluation."
                    if e55_active
                    else
                    "E53 holds each coefficient at its preregistered reference "
                    "during its own 64 eligible observations, then applies the "
                    "run's warmup-entropy/EMA ratio. Neither coefficient is "
                    "bounded; training never reads gold support or evaluation."
                    if e53_active
                    else
                    "E52 holds direct alpha at 0.000075 for 64 observations, "
                    "then applies alpha = 0.000075 * warmup_mean / entropy_EMA "
                    "to the next optimizer round. It has no Haarnoja optimizer "
                    "and no lower or upper projection; the canonical coefficient "
                    "remains fixed at 0.10."
                ),
                (
                    "Chart: paper/figures/"
                    "e59_mathir_global_verified_replay_05b_live.png."
                    if e59_active
                    else
                    "Chart: paper/figures/"
                    "e58_global_verified_replay_canonical_05b_live.png."
                    if e58_active
                    else
                    "Chart: paper/figures/"
                    "e57_verified_first_split_canonical_05b_live.png."
                    if e57_active
                    else
                    "Chart: paper/figures/"
                    "e56_open_set_split_canonical_05b_live.png."
                    if e56_active
                    else
                    "Chart: paper/figures/"
                    "e55_per_rollout_verified_anchor_05b_live.png."
                    if e55_active
                    else
                    "Chart: paper/figures/e53_verified_replay_05b_live.png."
                    if e53_active
                    else
                    "Chart: paper/figures/e52_current_canonical_05b_live.png."
                ),
            ]
            if queue_error:
                footer.append(f"Scheduler warning: {queue_error}")
            if warnings:
                footer.append("Possible stalls: " + "; ".join(warnings))
            return (
                f"{heading}\n\n{render_table(rows)}\n\n"
                + "\n".join(footer),
                warnings,
            )

        if self.e37_only:
            heading = (
                "E37/E38/E39/E41/E43 semantic-diversity head-to-head 0.5B — refreshed "
                f"{timestamp}\n"
                f"Overall completion: {progress_bar(completion_percent)} "
                f"{completion_percent:.2f}%  ({completion_delta}; "
                f"{settled_runs}/{displayed_runs} settled)\n"
                f"Fully landed evals: {fully_landed}/{displayed_runs}  "
                f"Running: {active_counts['RUNNING']}  Pending: {active_counts['PENDING']}  "
                f"Transitioning: {active_counts['TRANSITIONING']}  "
                f"Terminal failures: {active_counts['PROBLEM']}  "
                f"Stopped/cancelled: {active_counts['STOPPED']}"
            )
            footer = [
                "Landed = furthest persisted matched K=8 evaluation at or below the ten-pass endpoint; Rstep@pass is the current attempt.",
                "All rows use paired seeds 43/44/45. Countdown/Graph use identical neutral prompts and draw seeds 370100-370103.",
                "Retained E37 arms = matched Dr.GRPO and a duplicate-outcome penalty over observed canonical answer keys (coefficient 0.10).",
                "E38 = bounded predictive semantic Shannon surprise over the same observed answer keys (coefficient 0.10); no latent instructions or valid-mode catalogue.",
                "E39 = MATH-500 evaluation every two passes after training on MATH12K-384, using one fixed K=8 draw (seed 390100).",
                "E41 = separately-centered semantic Shannon advantage, preserving the ordinary task advantage while centering the predictive semantic term against its frozen baseline.",
                "E43 = success-conditioned signed Shannon advantage: its semantic predictor is fit only to verified successes, while eligible successes retain both positive and negative bounded semantic pressure.",
                "MATH is single-answer: coverage@8 is N/A; distinct@8 counts distinct normalized correct representations, not reasoning paths.",
                "Step-zero matched pairing: "
                + semantic_entropy_step_zero_pairing_status(self.artifact_root)
                + ".",
                "Chart: paper/figures/e37_outcome_collision_05b_live.png.",
            ]
            if queue_error:
                footer.append(f"Scheduler warning: {queue_error}")
            if warnings:
                footer.append("Possible stalls: " + "; ".join(warnings))
            return f"{heading}\n\n{render_table(rows)}\n\n" + "\n".join(footer), warnings

        if self.latest_05b_only:
            heading = (
                "Free-form reruns + E36 MI + E37/E38/E39/E41/E43 semantic-diversity evaluation — refreshed "
                f"{timestamp}\n"
                f"Overall completion: {progress_bar(completion_percent)} "
                f"{completion_percent:.2f}%  ({completion_delta}; "
                f"{settled_runs}/{displayed_runs} settled)\n"
                f"Fully landed evals: {fully_landed}/{displayed_runs}  "
                f"Running: {active_counts['RUNNING']}  Pending: {active_counts['PENDING']}  "
                f"Transitioning: {active_counts['TRANSITIONING']}  "
                f"Terminal failures: {active_counts['PROBLEM']}  "
                f"Stopped/cancelled: {active_counts['STOPPED']}"
            )
            footer = [
                "Landed = furthest persisted K=8 evaluation at or below the row's configured endpoint.",
                "E32/E33 and E37/E38/E39/E41/E43 rows use paired seeds 43/44/45; E36 rows use seeds 3601/3602/3603; every current row has a ten-pass endpoint. Rstep@pass is the current attempt.",
                "E36 quality = identical neutral prompts and draw seeds 360100-360103 in both arms; the plotted quality curves never contain latent instructions.",
                "MaxEnt = conditional-token Haarnoja dual, 125% entropy target, entropy EMA decay 0.7.",
                "E36 binding = DIAYN-only z-conditioned generation with independent deterministic seeds per (draw,prompt,z) and held-draw conditional I(Z;A|X).",
                "E37 = matched free-form Dr.GRPO versus semantic collision entropy over observed canonical answer keys.",
                "E38 = bounded predictive semantic Shannon surprise over the same observed answer keys; neither treatment uses latent instructions or an enumerated valid-mode catalogue.",
                "E39 adds MATH-500 every two passes with one fixed K=8 draw (seed 390100); coverage@8 is N/A for single-answer math and distinct@8 counts normalized correct representations, not reasoning paths.",
                "E41 adds a separately-centered semantic Shannon advantage on Countdown, Graph coloring, and MATH.",
                "E43 adds a success-conditioned signed Shannon advantage on the same three domains, retaining bounded positive and negative pressure on eligible successes.",
                "Charts: var/artifacts/freeform_05b_latest.png includes E32, E36, and the retained semantic-diversity comparison; the dedicated three-domain E37/E38/E39/E41/E43 quality/breadth diagnostic is paper/figures/e37_outcome_collision_05b_live.png.",
            ]
            if queue_error:
                footer.append(f"Scheduler warning: {queue_error}")
            if warnings:
                footer.append("Possible stalls: " + "; ".join(warnings))
            return f"{heading}\n\n{render_table(rows)}\n\n" + "\n".join(footer), warnings

        heading = (
            "Canonical MaxEnt + matched controls "
            f"+ E21/E22/E25/E26/E27/E28/E29 free-form MaxEnt — refreshed {timestamp}\n"
            f"Overall completion: {progress_bar(completion_percent)} "
            f"{completion_percent:.2f}%  ({completion_delta}; "
            f"{settled_runs}/{displayed_runs} settled)\n"
            f"Fully landed evals: {fully_landed}/{displayed_runs}  "
            f"Running: {active_counts['RUNNING']}  Pending: {active_counts['PENDING']}  "
            f"Transitioning: {active_counts['TRANSITIONING']}  "
            f"Terminal failures: {active_counts['PROBLEM']}  "
            f"Stopped/cancelled: {active_counts['STOPPED']}  "
            f"Design-waiting: {gate_waiting_runs}  "
            f"Scale-held: {gate_held_runs}\n"
            f"{smoke_line}\n"
            "E16 full 0.5B cells stay DESIGN until their analytical jobs "
            "exist; the six-cell smoke is an operational gate only"
        )
        footer = [
            "Landed = furthest persisted primary evaluation at or below each run's configured endpoint.",
            "Rstep@pass = current attempt; P = queued; R init = allocated/startup.",
            "Restarts keep landed progress while live progress follows the replacement.",
            "Overall % gives settled runs full credit; active runs use their furthest persisted pass against their own endpoint.",
            "E16 0.5B rows stay visible before launch and then use the normal P/R/done lifecycle.",
            "DESIGN = no full E16 analytical job exists for this cell; excluded from completion totals.",
            "HELD = deliberately withheld from normal scheduling; active 7B Countdown and graph-coloring runs follow the E23/E24 canonical protocols.",
            "E21 MATH-500 is free-form token-policy MaxEnt with a one-pass endpoint and sampled pass@8 as its landed marker.",
            "E22 adds free-form conditional-token Haarnoja-dual runs on 0.5B Countdown and graph coloring; they remain separate from canonical-action MaxEnt.",
            "The active 3B free-form rows are clean repair-v2 only: abandoned E25/E28 attempts and their contaminated metrics are excluded. E25 and E28 graph replay from initialization; E28 Countdown starts from audited clean steps 96/96/864.",
            "E25 is the 3B 125%-target dual and E28 its matched free-form Dr.GRPO control. E27/E22 are the corresponding clean 0.5B treatment/control cohorts.",
            "E29 is the matched 7B free-form Dr.GRPO/Haarnoja-dual cohort; each run uses four GPUs on one node.",
            "E26-v2 is the treatment-only 125%-target MATH dual; cancelled E21 treatments remain visible as historical partial runs.",
        ]
        if queue_error:
            footer.append(f"Scheduler warning: {queue_error}")
        if warnings:
            footer.append("Possible stalls: " + "; ".join(warnings))
        return f"{heading}\n\n{render_table(rows)}\n\n" + "\n".join(footer), warnings


class FigureRefresher:
    """Launch compact, non-overlapping figure refreshes in the background."""

    def __init__(
        self,
        interval_seconds: float,
        log_path: Path,
        refresh_args: tuple[str, ...] = (),
    ):
        self.interval_seconds = interval_seconds
        self.log_path = log_path
        self.refresh_args = refresh_args
        self.process: subprocess.Popen | None = None
        self.next_start = 0.0
        self.last_returncode: int | None = None
        self.last_finished_at: datetime | None = None

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.process is not None:
            returncode = self.process.poll()
            if returncode is not None:
                self.last_returncode = returncode
                self.last_finished_at = datetime.now().astimezone()
                self.process = None
        if self.process is not None or now < self.next_start:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab") as log_handle:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(FIGURE_REFRESH_SCRIPT),
                    *self.refresh_args,
                ],
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.next_start = now + self.interval_seconds

    def status(self) -> str:
        cadence = f"every {self.interval_seconds:g}s"
        if self.process is not None:
            state = f"running ({cadence})"
        elif self.last_finished_at is None:
            state = f"scheduled ({cadence})"
        else:
            outcome = "ok" if self.last_returncode == 0 else f"exit {self.last_returncode}"
            state = (
                f"last {outcome} at {self.last_finished_at:%H:%M:%S %Z} "
                f"({cadence})"
            )
        return f"Figure refresh: {state}; log={self.log_path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="seconds between refreshes (default: 30)",
    )
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="append snapshots instead of clearing an interactive terminal",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=45,
        help="warn after this many minutes without a metrics write (default: 45)",
    )
    parser.add_argument(
        "--figure-refresh-seconds",
        type=float,
        default=0.0,
        help="rebuild all campaign figures at this cadence; 0 disables it",
    )
    parser.add_argument(
        "--figure-refresh-log",
        type=Path,
        default=DEFAULT_FIGURE_REFRESH_LOG,
        help="background figure-refresh log path",
    )
    parser.add_argument(
        "--all-campaigns",
        action="store_true",
        help="show the historical all-scale dashboard instead of latest E32/E33 reruns",
    )
    parser.add_argument(
        "--e37-only",
        action="store_true",
        help=(
            "show only the 0.5B E37 comparators and matched E38 Shannon "
            "plus the E39 MATH, E41 separately-centered Shannon, and E43 "
            "success-conditioned signed-Shannon extensions"
        ),
    )
    parser.add_argument(
        "--current-canonical-only",
        action="store_true",
        help=(
            "show only the current E52 ModeBench sentinel with matched "
            "Dr.GRPO and both inverse-entropy arms"
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
    exclusive_modes = sum(
        bool(value)
        for value in (
            args.e37_only,
            args.current_canonical_only,
            args.all_campaigns,
        )
    )
    if exclusive_modes > 1:
        parser.error(
            "--e37-only, --current-canonical-only, and --all-campaigns are "
            "mutually exclusive"
        )
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.stale_minutes <= 0:
        parser.error("--stale-minutes must be positive")
    if args.figure_refresh_seconds < 0:
        parser.error("--figure-refresh-seconds must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    dashboard = Dashboard(
        args.data_root,
        args.artifact_root,
        args.stale_minutes,
        latest_05b_only=(
            not args.all_campaigns
            and not args.e37_only
            and not args.current_canonical_only
        ),
        e37_only=args.e37_only,
        e46_only=args.current_canonical_only,
    )
    interactive_clear = sys.stdout.isatty() and not args.no_clear and not args.once
    figure_refresher = None
    if args.figure_refresh_seconds > 0 and not args.once:
        figure_refresher = FigureRefresher(
            args.figure_refresh_seconds,
            args.figure_refresh_log,
            (
                ("--current-canonical-only",)
                if args.current_canonical_only
                else ("--e37-only",)
                if args.e37_only
                else ()
            ),
        )
    try:
        while True:
            text, _ = dashboard.snapshot()
            if figure_refresher is not None:
                figure_refresher.tick()
                text += f"\n{figure_refresher.status()}"
            if interactive_clear:
                print("\033[2J\033[H", end="")
            print(text, flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        suffix = ""
        if figure_refresher is not None and figure_refresher.process is not None:
            suffix = " The in-flight figure refresh will finish in the background."
        print(f"\nMonitor stopped; the Slurm jobs continue running.{suffix}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
