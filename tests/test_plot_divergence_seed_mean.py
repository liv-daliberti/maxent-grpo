import importlib.util
import json
import statistics
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "ops" / "exp_scaling" / "plot_divergence.py"
_SPEC = importlib.util.spec_from_file_location("plot_divergence", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
seed_mean = _MODULE.seed_mean
seed_mean_eval_ci = _MODULE.seed_mean_eval_ci
outcome_collision_seed_mean = _MODULE._outcome_collision_seed_mean
outcome_collision_metric_value = _MODULE._outcome_collision_metric_value
mechanism_metric_value = _MODULE._mechanism_metric_value
live_x_upper = _MODULE._live_x_upper
tight_nonnegative_limits = _MODULE._tight_nonnegative_limits


def test_seed_mean_requires_all_three_seeds_when_requested():
    per_seed = {
        43: [(0.0, {"score": 0.3}), (0.25, {"score": 0.6})],
        44: [(0.0, {"score": 0.6}), (0.25, {"score": 0.9})],
        45: [(0.0, {"score": 0.9})],
    }

    epochs, values = seed_mean(per_seed, "score")

    assert epochs == [0.0]
    assert values == pytest.approx([0.6])


def test_seed_mean_rejects_nonpositive_required_seed_count():
    with pytest.raises(ValueError, match="must be positive"):
        seed_mean({}, "score", required_seed_count=0)


def test_eval_ci_uses_four_draw_means_after_averaging_training_seeds():
    per_seed = {
        seed: [
            (
                0.5,
                {
                    "mean8": 0.5,
                    "mean8_draws": [0.2, 0.4, 0.6, 0.8],
                },
            )
        ]
        for seed in (43, 44, 45)
    }

    epochs, lower, upper = seed_mean_eval_ci(per_seed, "mean8")

    se = statistics.stdev([0.2, 0.4, 0.6, 0.8]) / 2
    margin = _MODULE.T_975_DF3 * se
    assert epochs == [0.5]
    assert lower == pytest.approx([0.5 - margin])
    assert upper == pytest.approx([0.5 + margin])


def test_eval_ci_requires_all_seeds_and_all_four_draws():
    per_seed = {
        43: [(0.0, {"pass8": 0.5, "pass8_draws": [0.4, 0.5, 0.6, 0.5]})],
        44: [(0.0, {"pass8": 0.5, "pass8_draws": [0.4, 0.5, 0.6, 0.5]})],
        45: [(0.0, {"pass8": 0.5, "pass8_draws": []})],
    }

    assert seed_mean_eval_ci(per_seed, "pass8") == ([], [], [])


def test_probability_metrics_share_full_unit_interval():
    assert _MODULE.PROPORTION_METRICS == {
        "greedy",
        "mean8",
        "pass8",
        "coverage8",
    }
    assert _MODULE.PROPORTION_YLIM == (0.0, 1.0)


def test_distinct_scale_uses_campaign_wide_rounded_upper_limit(monkeypatch):
    monkeypatch.setattr(
        _MODULE,
        "ENVIRONMENTS",
        [
            ("Countdown", [("0.5B", "countdown.json", 1, None, None)]),
            ("Graph coloring", [("3B", "graph.json", 1, None, None)]),
        ],
    )
    values = {
        "countdown.json": 1.56,
        "graph.json": 2.51,
    }

    def fake_load_series(path, _steps_per_epoch):
        return {
            "maxent": {
                43: [(0.0, {"distinct8": values[path]})],
            }
        }

    monkeypatch.setattr(_MODULE, "load_series", fake_load_series)

    assert _MODULE.shared_distinct_ylim() == (0.0, 3.0)


def test_current_canonical_live_limits_zoom_to_observed_data():
    assert live_x_upper([]) == pytest.approx(0.25)
    assert 0.6 < live_x_upper([0.0, 0.6]) < 0.7

    lower, upper = tight_nonnegative_limits([0.34, 0.36])
    assert 0.33 < lower < 0.34
    assert 0.36 < upper < 0.37
    assert tight_nonnegative_limits([]) == (0.0, 1.0)


def test_live_series_ingestion_can_extend_beyond_fifty_passes(
    monkeypatch, tmp_path
):
    artifact = tmp_path / "live.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "split": "multi_answer",
                    "training_passes": training_passes,
                    "step": int(training_passes),
                    "arm": "maxent_inverse_canonical_replay",
                    "seed": 9010,
                }
                for training_passes in (49.0, 51.0)
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)

    bounded = _MODULE.load_series(
        "live.json", 1, max_training_epochs=50.0
    )
    unbounded = _MODULE.load_series(
        "live.json", 1, max_training_epochs=None
    )

    assert [point[0] for point in bounded["maxent_inverse_canonical_replay"][9010]] == [
        49.0
    ]
    assert [point[0] for point in unbounded["maxent_inverse_canonical_replay"][9010]] == [
        49.0,
        51.0,
    ]


def test_current_canonical_renderer_expands_with_treatment_frontier():
    source = _SCRIPT.read_text(encoding="utf-8")

    assert "live_x_upper = _live_x_upper(treatment_live_epochs)" in source
    assert "ax.set_xlim(0.0, live_x_upper)" in source
    assert source.count("max_training_epochs=None") >= 2
    assert "ax.set_ylim(*_tight_nonnegative_limits(plotted_y_values))" in source
    assert (
        "Every x-axis expands with the latest treatment checkpoint"
        in source
    )
    assert "live-chart ingestion remains uncapped" in source
    assert (
        "Gold-support-normalized coverage is deliberately omitted"
        in source
    )
    renderer = source[
        source.index("def render_online_canonical_maxent_05b()") :
        source.index("\ndef render_", source.index(
            "def render_online_canonical_maxent_05b()"
        ) + 1)
    ]
    assert '("coverage8", "valid coverage@8"' not in renderer
    assert '"environment": "MathIR algebra — E45"' not in source
    assert (
        '"environment": f"Python factors — {campaign} (live frontier)"'
        in source
    )
    assert (
        "Current canonical experiment — {campaign} ModeBench "
        in source
    )
    assert "e53_verified_replay_05b_live" in source
    assert "e56_open_set_split_canonical_05b_live" in source
    assert '"open_set_split_canonical"' in source
    assert "E56 open-set discovery + split mass/balance replay" in source
    assert "Stage A — three fresh seeds" in source
    assert "Heavy lines show the fresh seed-43/44/45 mean" in source


def test_load_series_accepts_e51_fifty_pass_budget(monkeypatch, tmp_path):
    artifact = tmp_path / "e51_curve.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "arm": "online_canonical_policy_entropy",
                    "seed": 43,
                    "step": int(training_passes * 384),
                    "split": "multi_answer",
                    "training_passes": training_passes,
                    "mean8": 0.25,
                }
                for training_passes in (10.0, 50.0, 50.25)
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)

    historical = _MODULE.load_series(artifact.name, 384)
    e51 = _MODULE.load_series(
        artifact.name,
        384,
        max_training_epochs=50.0,
    )

    assert [epoch for epoch, _row in historical[
        "online_canonical_policy_entropy"
    ][43]] == [10.0]
    assert [epoch for epoch, _row in e51[
        "online_canonical_policy_entropy"
    ][43]] == [10.0, 50.0]


def test_outcome_collision_diagnostic_requires_three_seeds_and_derives_parseable():
    per_seed = {
        43: [
            (
                0.0,
                {
                    "outcome_collision_invalid_fraction": 0.25,
                },
            )
        ],
        44: [
            (
                0.0,
                {
                    "outcome_collision_parseable_fraction": 0.8,
                },
            )
        ],
        45: [
            (
                0.0,
                {
                    "outcome_collision_invalid_fraction": 0.15,
                },
            )
        ],
    }

    assert outcome_collision_metric_value(
        per_seed[43][0][1],
        "outcome_collision_parseable_fraction",
    ) == pytest.approx(0.75)
    epochs, values = outcome_collision_seed_mean(
        per_seed,
        "outcome_collision_parseable_fraction",
    )
    assert epochs == [0.0]
    assert values == pytest.approx([(0.75 + 0.8 + 0.85) / 3])


def test_e43_cap_hit_fraction_sums_positive_and_negative_caps():
    row = {
        "semantic_shannon_success_conditioned_signed_positive_cap_fraction": 0.125,
        "semantic_shannon_success_conditioned_signed_negative_cap_fraction": 0.0625,
    }

    assert mechanism_metric_value(
        row,
        "semantic_shannon_success_conditioned_signed_cap_fraction",
    ) == pytest.approx(0.1875)


def test_outcome_collision_renderer_handles_pending_empty_artifacts(
    monkeypatch,
    tmp_path,
):
    artifact_root = tmp_path / "var" / "artifacts"
    artifact_root.mkdir(parents=True)
    for prefix in (
        "cde37_outcome_collision_05b_v1",
        "gce37_outcome_collision_05b_v1",
        "cde38_semantic_shannon_05b_v1",
        "gce38_semantic_shannon_05b_v1",
        "mte39_math12k_384_semantic_entropy_05b_v1",
    ):
        (artifact_root / f"{prefix}_scaling_curve.json").write_text(
            "[]",
            encoding="utf-8",
        )
    out = artifact_root / "e37_pending"
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(_MODULE, "OUT_OUTCOME_COLLISION_05B", out)

    _MODULE.render_outcome_collision_05b()

    assert out.with_suffix(".png").stat().st_size > 0
    assert out.with_suffix(".pdf").stat().st_size > 0


def test_online_canonical_renderer_handles_pending_empty_artifacts(
    monkeypatch,
    tmp_path,
):
    artifact_root = tmp_path / "var" / "artifacts"
    artifact_root.mkdir(parents=True)
    for prefix in (
        "cde51_policy_entropy_adaptive_canonical_05b_50ep_v2_allcs",
        "gce51_policy_entropy_adaptive_canonical_05b_50ep_v2",
        "pye51_policy_entropy_adaptive_canonical_05b_50ep_v2_allcs",
    ):
        (artifact_root / f"{prefix}_scaling_curve.json").write_text(
            "[]",
            encoding="utf-8",
        )
    out = artifact_root / "e44_ogs_pending"
    legacy_out = artifact_root / "e45_e51_compatibility"
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(
        _MODULE,
        "OUT_ONLINE_CANONICAL_MAXENT_05B",
        out,
    )
    monkeypatch.setattr(
        _MODULE,
        "LEGACY_OUT_ONLINE_CANONICAL_MAXENT_05B",
        legacy_out,
    )

    _MODULE.render_online_canonical_maxent_05b()

    assert out.with_suffix(".png").stat().st_size > 0
    assert out.with_suffix(".pdf").stat().st_size > 0
    assert legacy_out.with_suffix(".png").read_bytes() == (
        out.with_suffix(".png").read_bytes()
    )
    assert legacy_out.with_suffix(".pdf").read_bytes() == (
        out.with_suffix(".pdf").read_bytes()
    )


def test_load_series_routes_e37_control_and_treatment_arms(
    monkeypatch,
    tmp_path,
):
    artifact = tmp_path / "e37_outcome_collision_curve.json"
    rows = []
    for arm in ("grpo", "outcome_collision"):
        rows.append(
            {
                "arm": arm,
                "seed": 43,
                "step": 0,
                "split": "multi_answer",
                "training_passes": 0.0,
                "mean8": 0.25,
                "outcome_collision_rate": (
                    None if arm == "grpo" else 0.5
                ),
            }
        )
    artifact.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)

    series = _MODULE.load_series(artifact.name, 384)

    assert set(series) == {
        "freeform_outcome_collision_drgrpo",
        "freeform_outcome_collision",
    }


def test_load_series_routes_e38_semantic_shannon_arm(
    monkeypatch,
    tmp_path,
):
    artifact = tmp_path / "e38_semantic_shannon_curve.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "arm": "semantic_shannon",
                    "seed": 43,
                    "step": 0,
                    "split": "multi_answer",
                    "training_passes": 0.0,
                    "mean8": 0.25,
                    "semantic_shannon_normalized_surprisal_mean": 0.5,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)

    series = _MODULE.load_series(artifact.name, 384)

    assert set(series) == {"freeform_semantic_shannon"}


def test_load_series_routes_e41_separate_shannon_advantage_arm(
    monkeypatch,
    tmp_path,
):
    artifact = tmp_path / "cde41_semantic_shannon_advantage_curve.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "arm": "semantic_shannon_advantage",
                    "seed": 43,
                    "step": 0,
                    "split": "multi_answer",
                    "training_passes": 0.0,
                    "mean8": 0.25,
                    "semantic_shannon_separate_semantic_advantage_rms": 0.5,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)

    series = _MODULE.load_series(artifact.name, 384)

    assert set(series) == {"freeform_semantic_shannon_advantage"}


def test_load_series_routes_e43_success_conditioned_signed_shannon_arm(
    monkeypatch,
    tmp_path,
):
    artifact = (
        tmp_path / "cde43_success_conditioned_signed_semantic_shannon_curve.json"
    )
    artifact.write_text(
        json.dumps(
            [
                {
                    "arm": "success_conditioned_signed_semantic_shannon",
                    "seed": 43,
                    "step": 0,
                    "split": "multi_answer",
                    "training_passes": 0.0,
                    "mean8": 0.25,
                    "semantic_shannon_success_conditioned_signed_effective_advantage_rms": 0.02,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)

    series = _MODULE.load_series(artifact.name, 384)

    assert set(series) == {
        "freeform_success_conditioned_signed_semantic_shannon"
    }


def test_load_series_routes_all_e39_arms_from_math_split(
    monkeypatch,
    tmp_path,
):
    artifact = (
        tmp_path
        / "mte39_math12k_384_semantic_entropy_05b_v1_scaling_curve.json"
    )
    rows = [
        {
            "arm": arm,
            "seed": 43,
            "step": 0,
            "split": "math",
            "training_passes": 0.0,
            "mean8": 0.25,
        }
        for arm in ("grpo", "outcome_collision", "semantic_shannon")
    ]
    rows.append(
        {
            "arm": "grpo",
            "seed": 43,
            "step": 0,
            "split": "multi_answer",
            "training_passes": 0.0,
            "mean8": 0.99,
        }
    )
    artifact.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)

    series = _MODULE.load_series(artifact.name, 384, split="math")

    assert set(series) == {
        "freeform_outcome_collision_drgrpo",
        "freeform_outcome_collision",
        "freeform_semantic_shannon",
    }
    assert {
        points[0][1]["mean8"]
        for per_seed in series.values()
        for points in per_seed.values()
    } == {0.25}


def test_outcome_collision_renderer_marks_math_coverage_not_applicable(
    monkeypatch,
    tmp_path,
):
    artifact_root = tmp_path / "var" / "artifacts"
    artifact_root.mkdir(parents=True)
    for prefix in (
        "cde37_outcome_collision_05b_v1",
        "gce37_outcome_collision_05b_v1",
        "cde38_semantic_shannon_05b_v1",
        "gce38_semantic_shannon_05b_v1",
        "mte39_math12k_384_semantic_entropy_05b_v1",
        "cde41_semantic_shannon_advantage_05b_v1",
        "gce41_semantic_shannon_advantage_05b_v1",
        "mte41_math12k_384_semantic_shannon_advantage_05b_v1",
        "cde43_success_conditioned_signed_semantic_shannon_05b_v1",
        "gce43_success_conditioned_signed_semantic_shannon_05b_v1",
        "mte43_math12k_384_success_conditioned_signed_semantic_shannon_05b_v1",
    ):
        (artifact_root / f"{prefix}_scaling_curve.json").write_text(
            "[]",
            encoding="utf-8",
        )
    figures = []
    monkeypatch.setattr(_MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(
        _MODULE,
        "_atomic_savefig",
        lambda fig, *_args, **_kwargs: figures.append(fig),
    )

    _MODULE.render_outcome_collision_05b()

    assert figures
    math_coverage_axis = figures[0].axes[
        2 * len(_MODULE.OUTCOME_COLLISION_DIAGNOSTIC_METRICS) + 3
    ]
    assert any(
        "N/A" in text.get_text() and "single-answer" in text.get_text()
        for text in math_coverage_axis.texts
    )
