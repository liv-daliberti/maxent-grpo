"""Source-contract tests that do not import optional plotting dependencies."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PLOT_PATH = ROOT / "ops/exp_scaling/plot_divergence.py"
REFRESH_PATH = ROOT / "ops/exp_scaling/refresh_campaign_curves.py"
LATEST_REFRESH_PATH = ROOT / "ops/exp_scaling/refresh_latest_freeform_05b.py"
MATH_PLOT_PATH = ROOT / "ops/plot_e21_math_token_maxent_live.py"
MAKEFILE_PATH = ROOT / "Makefile"


def _assignment(path: Path, name: str) -> tuple[ast.AST, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
            return statement.value, source
    raise AssertionError(f"{name} assignment not found in {path}")


def _strings(node: ast.AST) -> set[str]:
    return {
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def test_current_canonical_refresh_contains_sentinel_and_stage_a_cells():
    source = LATEST_REFRESH_PATH.read_text(encoding="utf-8")

    for required in (
        "CURRENT_CANONICAL_CELLS",
        "STAGE_A_CURRENT_CANONICAL_CELLS",
        "e52_direct_inverse_entropy_canonical_05b_stage_a_v1_identity.json",
        "def current_canonical_cells()",
    ):
        assert required in source
    for domain_prefix in ("cde52", "gce52", "pye52"):
        assert f'"{domain_prefix}_direct_inverse_entropy_canonical_05b_50ep_' in source
    assert "sentinel_v2" in source
    assert "stage_a_v1" in source
    assert "None\n                if args.current_canonical_only" in source
    assert "if max_training_passes is not None:" in source
    assert "--jobs-manifest" in source
    assert "current canonical manifest missing" in source


def test_standard_maxent_05b_figures_consume_e16_and_e19_curves():
    environments, _ = _assignment(PLOT_PATH, "ENVIRONMENTS")
    cells, _ = _assignment(REFRESH_PATH, "CELLS")
    figure_strings = _strings(environments)
    refresh_cells = ast.literal_eval(cells)
    refresh_stamps = {stamp for stamp, _pool, _samples in refresh_cells}

    for prefix in (
        "cde16_canonical_maxent_05b_v2",
        "gce16_canonical_maxent_05b_v2",
        "cde19_canonical_drgrpo_05b_v1",
        "gce19_canonical_drgrpo_05b_v1",
        "cde32_freeform_05b_ema_10ep_v4_preemptsafe",
        "gce32_freeform_05b_ema_10ep_v5",
        "cde23_canonical_maxent_7b_v6_4xa100_evalsync_fix",
        "gce24_canonical_maxent_7b_v4_4xa100_evalsync_fix",
        "cde33_freeform_3b_ema_10ep_v3_a100",
        "gce33_freeform_3b_ema_10ep_v3_a100",
        "cde29_freeform_7b_4gpu_v5_buffer_restore",
        "gce29_freeform_7b_4gpu_v5_buffer_restore",
    ):
        assert f"var/artifacts/{prefix}_scaling_curve.json" in figure_strings
        assert prefix in refresh_stamps

    assert not any("e11_standard_maxent" in value for value in figure_strings)
    assert not any("e11_standard_maxent" in stamp for stamp in refresh_stamps)


def test_e36_live_figure_consumes_both_three_seed_dual_eval_cells():
    cells, _ = _assignment(LATEST_REFRESH_PATH, "CELLS")
    refresh_stamps = {cell[0] for cell in ast.literal_eval(cells)}
    source = PLOT_PATH.read_text(encoding="utf-8")

    for prefix in (
        "cde36_answer_option_mi_dual_eval_05b_v1",
        "gce36_answer_option_mi_dual_eval_05b_v1",
    ):
        assert prefix in refresh_stamps
        assert f"var/artifacts/{prefix}_scaling_curve.json" in source
    assert "e36_answer_option_mi_dual_eval_05b_live" in source
    assert '"freeform_answer_option_mi"' in source
    assert "mean_seed_count=3" in source
    assert "LATEST_FREEFORM_05B_PANEL" in source
    assert '"E32 · 3 seeds"' in source
    assert '"E36 · 3 seeds"' in source


def test_e37_live_figures_consume_retained_comparators_and_e41_e43_cells():
    cells, _ = _assignment(LATEST_REFRESH_PATH, "CELLS")
    e37_cells, refresh_source = _assignment(LATEST_REFRESH_PATH, "E37_CELLS")
    refresh_stamps = {cell[0] for cell in ast.literal_eval(cells)}
    e37_refresh_cells = ast.literal_eval(e37_cells)
    e37_refresh_stamps = {cell[0] for cell in e37_refresh_cells}
    source = PLOT_PATH.read_text(encoding="utf-8")

    for prefix in (
        "cde37_outcome_collision_05b_v1",
        "gce37_outcome_collision_05b_v1",
        "cde38_semantic_shannon_05b_v1",
        "gce38_semantic_shannon_05b_v1",
    ):
        assert prefix in refresh_stamps
        assert prefix in e37_refresh_stamps
        assert f"var/artifacts/{prefix}_scaling_curve.json" in source
    e39_prefix = "mte39_math12k_384_semantic_entropy_05b_v1"
    assert e39_prefix in e37_refresh_stamps
    assert next(
        cell for cell in e37_refresh_cells if cell[0] == e39_prefix
    ) == (e39_prefix, 384, ("math",))
    assert f"var/artifacts/{e39_prefix}_scaling_curve.json" in source
    e41_cells = (
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
    )
    for cell in e41_cells:
        assert cell in e37_refresh_cells
        assert cell[0] in refresh_stamps
        assert f"var/artifacts/{cell[0]}_scaling_curve.json" in source
    e43_cells = (
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
    for cell in e43_cells:
        assert cell in e37_refresh_cells
        assert cell[0] in refresh_stamps
        assert f"var/artifacts/{cell[0]}_scaling_curve.json" in source
    assert not any("e40_" in stamp for stamp in refresh_stamps)
    assert not any("e40_" in stamp for stamp in e37_refresh_stamps)
    assert "e40_semantic_collision_pg" not in source
    assert not any("e42_" in stamp for stamp in refresh_stamps)
    assert not any("e42_" in stamp for stamp in e37_refresh_stamps)
    assert "e42_quality_gated_semantic_novelty" not in source
    assert len(e37_refresh_stamps) == 11
    assert "--e37-only" in refresh_source
    assert "--eval-splits" in refresh_source
    assert 'ROOT / "paper/figures/e37_outcome_collision_05b_live"' in source
    assert '"freeform_outcome_collision"' in source
    assert '"freeform_outcome_collision_drgrpo"' in source
    assert '"freeform_semantic_shannon"' in source
    assert '"freeform_semantic_shannon_advantage"' in source
    assert '"freeform_success_conditioned_signed_semantic_shannon"' in source
    assert "render_outcome_collision_05b()" in source
    assert '("distinct8", "mean # distinct correct@8")' in source
    assert '"semantic_shannon_normalized_surprisal_mean"' in source
    assert (
        '"semantic_shannon_separate_semantic_advantage_rms"' in source
    )
    assert (
        '"semantic_shannon_success_conditioned_signed_effective_advantage_rms"'
        in source
    )
    assert (
        '"semantic_shannon_success_conditioned_signed_eligible_fraction"'
        in source
    )
    assert (
        '"semantic_shannon_success_conditioned_signed_effective_advantage_positive_fraction"'
        in source
    )
    assert (
        '"semantic_shannon_success_conditioned_signed_effective_advantage_negative_fraction"'
        in source
    )
    assert (
        '"semantic_shannon_success_conditioned_signed_cap_fraction"' in source
    )
    assert "success-conditioned signed Shannon advantage (E43)" in source
    assert 'marker="D" if is_e43 else "o"' in source
    assert 'marker="o"' in source
    assert "ax.set_ylim(0.0, 8.0)" in source
    assert "MATH-500 (train MATH12K-384)" in source
    assert "N/A\\nsingle-answer task" in source
    assert "distinct normalized " in source
    assert "correct representations, not reasoning paths" in source


def test_contaminated_3b_freeform_branches_are_clipped_at_clean_boundaries():
    limits, _ = _assignment(REFRESH_PATH, "MAX_CLEAN_STEP_BY_STAMP")

    assert ast.literal_eval(limits) == {
        "cde25_freeform_conditional_dual_3b_v2": {43: 576, 44: 576, 45: 288},
        "gce25_freeform_conditional_dual_3b_v2": {43: 576, 44: 480, 45: 576},
        "cde28_freeform_drgrpo_3b_v1": {43: 96, 44: 96, 45: 864},
        "gce28_freeform_drgrpo_3b_v1": {43: 144, 44: 48, 45: 48},
    }


def test_standard_maxent_legend_and_panel_drop_legacy_held_wording():
    arms, _ = _assignment(PLOT_PATH, "ARMS")
    panels, _ = _assignment(PLOT_PATH, "METHOD_PANELS")
    source = PLOT_PATH.read_text(encoding="utf-8")
    arm_strings = _strings(arms)
    panel_strings = _strings(panels)

    assert {
        "Standard MaxEnt fixed",
        "Standard MaxEnt proportional",
        "Standard MaxEnt Haarnoja dual",
    }.issubset(arm_strings)
    assert "Free-form conditional-token MaxEnt base-preserving Haarnoja dual" in arm_strings
    assert "Dr.GRPO (free-form reference)" in arm_strings
    assert "Free-form token-policy sidecar vs canonical Dr.GRPO (0.5B)" not in panel_strings
    assert "Free-form token-policy sidecar vs canonical Dr.GRPO (0.5B)" not in source
    assert "Free-form token-policy MaxEnt (0.5B E32 and 3B E33 matched EMA reruns; 7B E29)" in source
    assert '"e22_freeform_conditional_dual_05b_v2" in item' in source
    assert "Dr.GRPO (canonical matched control)" in arm_strings
    assert not any("legacy" in label.lower() for label in arm_strings | panel_strings)


def test_e16_figure_note_tracks_real_protocol_lifecycle():
    source = PLOT_PATH.read_text(encoding="utf-8")

    assert "def _e16_canonical_lifecycle_note" in source
    assert "E16 CANONICAL PROTOCOL IN DESIGN" in source
    assert "E16 CANONICAL SMOKE NOT SUBMITTED" in source
    assert "E16 CANONICAL SMOKE SUBMITTED" in source
    assert "E16 FULL CANONICAL GRID SUBMITTED" in source
    assert "_e16_canonical_lifecycle_note()" in source


def test_compute_divergence_publishes_only_maxent_figures_and_previews():
    outputs, source = _assignment(PLOT_PATH, "OUT_BY_PANEL")
    previews, _ = _assignment(PLOT_PATH, "LATEST_PREVIEW_BY_PANEL")

    assert {
        "on_policy",
        "paper/figures/compute_divergence_canonical_maxent",
        "freeform",
        "paper/figures/compute_divergence_freeform_modebench",
    }.issubset(_strings(outputs))
    assert {
        "var/artifacts/divergence_canonical_maxent_latest.png",
        "var/artifacts/divergence_freeform_modebench_latest.png",
    }.issubset(_strings(previews))
    assert "rescaling" not in _strings(outputs) | _strings(previews)
    assert "aggregation" not in source.lower()
    assert "method_panels=(panel,)" in source


def test_compute_divergence_shows_pass1_and_all_four_sampled_k8_metrics():
    metrics, source = _assignment(PLOT_PATH, "METRICS")

    assert ast.literal_eval(metrics) == [
        ("greedy", "pass@1"),
        ("mean8", "mean@8"),
        ("pass8", "pass@8"),
        ("coverage8", "coverage@8"),
        ("distinct8", "distinct@8"),
    ]
    assert "Pass@1 is deterministic greedy" in source
    assert "light band shows a 95% t-confidence interval" in source
    assert "Every raw draw" in source


def test_fixed_draw_terminal_backfills_cover_clean_05b_freeform_cells():
    backfills, _ = _assignment(REFRESH_PATH, "REPEATED_EVAL_BACKFILLS")

    assert ast.literal_eval(backfills) == (
        (
            "cde22_freeform_conditional_dual_05b_v2",
            "cde30_freeform_05b_fixed_k8x4_v1",
            "grpo",
        ),
        (
            "cde27_freeform_conditional_dual_05b_v1",
            "cde30_freeform_05b_fixed_k8x4_v1",
            "maxent_dual",
        ),
        (
            "gce22_freeform_conditional_dual_05b_v2",
            "gce30_freeform_05b_fixed_k8x4_v1",
            "grpo",
        ),
        (
            "gce27_freeform_conditional_dual_05b_v1",
            "gce30_freeform_05b_fixed_k8x4_v1",
            "maxent_dual",
        ),
    )


def test_e22_curve_is_distinct_from_canonical_dual_and_make_refreshes_it():
    source = PLOT_PATH.read_text(encoding="utf-8")
    makefile = MAKEFILE_PATH.read_text(encoding="utf-8")

    assert 'arm = "freeform_maxent_dual"' in source
    assert "_freeform_conditional_dual_" in source
    assert 'for key in ("freeform_drgrpo", "freeform_maxent_dual")' in source
    assert 'arm = "freeform_drgrpo"' in source
    assert '"_freeform_drgrpo_" in item' in source
    assert 'freeform_cohort = "_freeform_" in item' in source
    assert '"7B": "E29"' in source
    assert "e22-figures:" in makefile
    assert "ops/exp_scaling/refresh_campaign_curves.py" in makefile
    assert "E25-v2 125% TARGET" in source
    assert '"0.5B": "E32"' in source
    assert '"freeform_maxent_dual" not in series' in source


def test_math_publishes_compute_divergence_style_companion_from_make():
    source = MATH_PLOT_PATH.read_text(encoding="utf-8")
    makefile = MAKEFILE_PATH.read_text(encoding="utf-8")

    assert "paper/figures/compute_divergence_math_maxent" in source
    assert "var/artifacts/divergence_math_maxent_latest.png" in source
    assert "Free-form token-policy MaxEnt" in source
    assert "not canonical-action MaxEnt" in source
    assert "minimum_seeds=2" in source
    assert "mte26_math_freeform_conditional_dual_high_entropy_05b_v2" in source
    assert "mte21_math_conditional_token_05b_v4" in source
    assert 'arm_prefix = control_prefix if arm == "grpo" else prefix' in source
    assert "E26-v2 aggressive Haarnoja dual" in source
    assert "Historical E21 Dr.GRPO" in source
    assert "ops/exp_scaling/plot_divergence.py" in makefile
    assert "ops/plot_e21_math_token_maxent_live.py" in makefile
    assert "math-divergence-figure:" in makefile
