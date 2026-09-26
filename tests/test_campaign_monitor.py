import json
import os

import exp_scaling.monitor_campaign as monitor_campaign
from exp_scaling.monitor_campaign import (
    Dashboard,
    FigureRefresher,
    JobInfo,
    MetricCache,
    RunMetrics,
    RunSpec,
    campaign_specs,
    completion_credit,
    e37_05b_specs,
    e38_step_zero_pairing_status,
    e44_ogs_specs,
    e16_smoke_specs,
    e16_protocol_is_frozen,
    find_run_dirs,
    live_label,
    latest_05b_specs,
    parse_run_stamp_from_scontrol,
    semantic_entropy_step_zero_pairing_status,
)


def test_latest_dashboard_includes_e36_three_seed_dual_eval_rows():
    specs = [spec for spec in latest_05b_specs() if "e36" in spec.stamp_prefix]

    assert len(specs) == 4
    assert {spec.environment for spec in specs} == {"Countdown", "Graph coloring"}
    assert {spec.arm for spec in specs} == {"grpo", "diayn"}
    assert {spec.seeds for spec in specs} == {(3601, 3602, 3603)}
    assert {spec.max_passes for spec in specs} == {10.0}


def test_latest_dashboard_includes_e37_paired_outcome_collision_rows():
    specs = [
        spec
        for spec in latest_05b_specs()
        if "e37_outcome_collision" in spec.stamp_prefix
    ]

    assert len(specs) == 4
    assert {spec.environment for spec in specs} == {
        "Countdown",
        "Graph coloring",
    }
    assert {spec.arm for spec in specs} == {"grpo", "outcome_collision"}
    assert {spec.seeds for spec in specs} == {(43, 44, 45)}
    assert {spec.max_passes for spec in specs} == {10.0}
    assert {spec.num_samples for spec in specs} == {16}


def test_latest_dashboard_includes_e38_matched_shannon_rows():
    specs = [
        spec
        for spec in latest_05b_specs()
        if "e38_semantic_shannon" in spec.stamp_prefix
    ]

    assert len(specs) == 2
    assert {spec.environment for spec in specs} == {
        "Countdown",
        "Graph coloring",
    }
    assert {spec.arm for spec in specs} == {"semantic_shannon"}
    assert {spec.seeds for spec in specs} == {(43, 44, 45)}
    assert {spec.max_passes for spec in specs} == {10.0}
    assert {spec.num_samples for spec in specs} == {16}


def test_latest_dashboard_includes_e39_matched_math_three_arm_rows():
    prefix = "mte39_math12k_384_semantic_entropy_05b_v1"
    specs = [
        spec
        for spec in latest_05b_specs()
        if spec.stamp_prefix == prefix
    ]

    assert len(specs) == 3
    assert {spec.environment for spec in specs} == {
        "MATH-500 (train MATH12K-384)"
    }
    assert {spec.arm for spec in specs} == {
        "grpo",
        "outcome_collision",
        "semantic_shannon",
    }
    assert {spec.seeds for spec in specs} == {(43, 44, 45)}
    assert {spec.prompt_pool_size for spec in specs} == {384}
    assert {spec.num_samples for spec in specs} == {16}
    assert {spec.max_passes for spec in specs} == {10.0}
    assert {spec.eval_key for spec in specs} == {
        "eval/math/sampled_any_correct_at_8"
    }


def test_latest_dashboard_includes_e41_separate_shannon_rows_in_three_domains():
    specs = [
        spec
        for spec in latest_05b_specs()
        if "e41" in spec.stamp_prefix
    ]

    assert len(specs) == 3
    assert {spec.environment for spec in specs} == {
        "Countdown",
        "Graph coloring",
        "MATH-500 (train MATH12K-384)",
    }
    assert {spec.arm for spec in specs} == {"semantic_shannon_advantage"}
    assert {spec.method for spec in specs} == {
        "separately-centered semantic Shannon advantage (E41)"
    }
    assert {spec.seeds for spec in specs} == {(43, 44, 45)}
    assert {spec.max_passes for spec in specs} == {10.0}
    assert {spec.num_samples for spec in specs} == {16}


def test_latest_dashboard_includes_e43_signed_shannon_rows_in_three_domains():
    specs = [
        spec
        for spec in latest_05b_specs()
        if "e43" in spec.stamp_prefix
    ]

    assert len(specs) == 3
    assert {spec.environment for spec in specs} == {
        "Countdown",
        "Graph coloring",
        "MATH-500 (train MATH12K-384)",
    }
    assert {spec.arm for spec in specs} == {
        "success_conditioned_signed_semantic_shannon"
    }
    assert {spec.method for spec in specs} == {
        "success-conditioned signed Shannon advantage (E43)"
    }
    assert {spec.seeds for spec in specs} == {(43, 44, 45)}
    assert {spec.max_passes for spec in specs} == {10.0}
    assert {spec.num_samples for spec in specs} == {16}


def test_e37_only_specs_include_e41_and_e43_extensions_only():
    specs = e37_05b_specs()

    assert len(specs) == 15
    assert {spec.scale for spec in specs} == {"0.5B"}
    assert {spec.environment for spec in specs} == {
        "Countdown",
        "Graph coloring",
        "MATH-500 (train MATH12K-384)",
    }
    assert {spec.arm for spec in specs} == {
        "grpo",
        "outcome_collision",
        "semantic_shannon",
        "semantic_shannon_advantage",
        "success_conditioned_signed_semantic_shannon",
    }
    assert all(
        "e37_outcome_collision_05b" in spec.stamp_prefix
        or "e38_semantic_shannon_05b" in spec.stamp_prefix
        or spec.stamp_prefix
        == "mte39_math12k_384_semantic_entropy_05b_v1"
        or "e41_semantic_shannon_advantage_05b" in spec.stamp_prefix
        or "e41_math12k_384_semantic_shannon_advantage_05b"
        in spec.stamp_prefix
        or "e43_success_conditioned_signed_semantic_shannon_05b"
        in spec.stamp_prefix
        or "e43_math12k_384_success_conditioned_signed_semantic_shannon_05b"
        in spec.stamp_prefix
        for spec in specs
    )
    assert all("e42_" not in spec.stamp_prefix for spec in specs)


def test_current_canonical_specs_contain_only_e52_sentinel_arms(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    specs = e44_ogs_specs(artifact_root)

    assert len(specs) == 9
    assert {spec.environment for spec in specs} == {
        "Countdown",
        "Graph coloring",
        "Python factors",
    }
    assert {spec.arm for spec in specs} == {
        "grpo",
        "maxent_inverse",
        "maxent_inverse_canonical",
    }
    assert {spec.seeds for spec in specs} == {(9009,)}
    assert {spec.max_passes for spec in specs} == {50.0}
    assert {spec.num_samples for spec in specs} == {16}
    assert {spec.empty_status for spec in specs} == {"READY"}
    assert all(
        "e52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2"
        in spec.stamp_prefix
        or "e52_direct_inverse_entropy_canonical_05b_50ep_sentinel_v2_allcs"
        in spec.stamp_prefix
        for spec in specs
    )


def test_current_canonical_specs_switch_to_e56_open_set_sentinel(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / monitor_campaign.E56_SENTINEL_IDENTITY).write_text(
        "{}\n",
        encoding="utf-8",
    )

    specs = e44_ogs_specs(artifact_root)

    assert len(specs) == 6
    assert {spec.arm for spec in specs} == {
        "grpo",
        "open_set_split_canonical",
    }
    assert {spec.seeds for spec in specs} == {(9010,)}
    assert {
        spec.stamp_prefix
        for spec in specs
        if spec.arm == "open_set_split_canonical"
    } == set(monitor_campaign.E56_SENTINEL_PREFIXES.values())
    assert {
        spec.stamp_prefix for spec in specs if spec.arm == "grpo"
    } == set(monitor_campaign.E53_SENTINEL_PREFIXES.values())


def test_current_canonical_specs_switch_to_stage_a_identity(
    monkeypatch,
    tmp_path,
):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / monitor_campaign.E52_STAGE_A_IDENTITY).write_text(
        "{}\n",
        encoding="utf-8",
    )
    specs = e44_ogs_specs(artifact_root)

    assert len(specs) == 9
    assert {spec.seeds for spec in specs} == {(43, 44, 45)}
    assert all(
        "e52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1"
        in spec.stamp_prefix
        or "e52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1_allcs"
        in spec.stamp_prefix
        for spec in specs
    )


def test_current_canonical_dashboard_includes_only_e52(
    monkeypatch,
    tmp_path,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue",
        lambda _user: ({}, None),
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs",
        lambda _ids: {},
    )

    text, _ = Dashboard(
        tmp_path / "data",
        artifacts,
        45,
        e46_only=True,
    ).snapshot()

    assert "E45" not in text
    assert "MathIR" not in text
    assert text.count("unbounded inverse conditional entropy") == 3
    assert text.count("inverse entropy + fixed canonical") == 3
    assert text.count("matched Dr.GRPO") >= 3
    assert "Python factors" in text
    assert "no lower or upper projection" in text
    assert text.count("READY") >= 9
    assert (
        "Current canonical experiments: E52 ModeBench engineering sentinel"
        in text
    )
    assert (
        "paper/figures/"
        "e52_current_canonical_05b_live.png"
        in text
    )
    assert "E37" not in text
    assert "E38" not in text
    assert "E44" not in text


def test_latest_dashboard_shows_pending_e37_before_first_artifact(
    monkeypatch,
    tmp_path,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    prefix = "cde37_outcome_collision_05b_v1"
    stamp = f"{prefix}_outcome_collision_s43"
    (artifacts / f"{prefix}_comparative_jobs.tsv").write_text(
        "job_id\trun_stamp\n" f"3701\t{stamp}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue",
        lambda _user: (
            {"3701": JobInfo("3701", "PENDING", "Priority")},
            None,
        ),
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs",
        lambda _ids: {},
    )

    text, _ = Dashboard(
        tmp_path / "data",
        artifacts,
        45,
        e37_only=True,
    ).snapshot()

    row = next(
        line
        for line in text.splitlines()
        if line.startswith("Countdown")
        and "semantic collision entropy (E37)" in line
    )
    assert "—/—/—" in row
    assert "P / — / —" in row
    assert "paper/figures/e37_outcome_collision_05b_live.png" in text
    assert "E36" not in text
    assert "EMA-Haarnoja" not in text
    assert "3B" not in text


def test_latest_dashboard_shows_pending_e38_shannon_before_first_artifact(
    monkeypatch,
    tmp_path,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    prefix = "cde38_semantic_shannon_05b_v1"
    stamp = f"{prefix}_semantic_shannon_s43"
    (artifacts / f"{prefix}_comparative_jobs.tsv").write_text(
        "job_id\trun_stamp\n" f"3801\t{stamp}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue",
        lambda _user: (
            {"3801": JobInfo("3801", "PENDING", "Priority")},
            None,
        ),
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs",
        lambda _ids: {},
    )

    text, _ = Dashboard(
        tmp_path / "data",
        artifacts,
        45,
        e37_only=True,
    ).snapshot()

    row = next(
        line
        for line in text.splitlines()
        if line.startswith("Countdown")
        and "predictive semantic Shannon entropy (E38)" in line
    )
    assert "—/—/—" in row
    assert "P / — / —" in row
    assert "E37/E38/E39/E41/E43 semantic-diversity head-to-head" in text
    assert "Step-zero matched pairing: pending" in text


def test_latest_dashboard_shows_pending_e41_before_first_artifact(
    monkeypatch,
    tmp_path,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    prefix = "cde41_semantic_shannon_advantage_05b_v1"
    stamp = f"{prefix}_semantic_shannon_advantage_s43"
    (artifacts / f"{prefix}_comparative_jobs.tsv").write_text(
        "job_id\trun_stamp\n" f"4101\t{stamp}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue",
        lambda _user: (
            {"4101": JobInfo("4101", "PENDING", "Priority")},
            None,
        ),
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs",
        lambda _ids: {},
    )

    text, _ = Dashboard(
        tmp_path / "data",
        artifacts,
        45,
        e37_only=True,
    ).snapshot()

    row = next(
        line
        for line in text.splitlines()
        if line.startswith("Countdown")
        and "separately-centered semantic Shannon advantage (E41)" in line
    )
    assert "—/—/—" in row
    assert "P / — / —" in row
    assert "preserving the ordinary task advantage" in text


def test_latest_dashboard_shows_pending_e43_before_first_artifact(
    monkeypatch,
    tmp_path,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    prefix = "cde43_success_conditioned_signed_semantic_shannon_05b_v1"
    stamp = f"{prefix}_success_conditioned_signed_semantic_shannon_s43"
    (artifacts / f"{prefix}_comparative_jobs.tsv").write_text(
        "job_id\trun_stamp\n" f"4301\t{stamp}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue",
        lambda _user: (
            {"4301": JobInfo("4301", "PENDING", "Priority")},
            None,
        ),
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs",
        lambda _ids: {},
    )

    text, _ = Dashboard(
        tmp_path / "data",
        artifacts,
        45,
        e37_only=True,
    ).snapshot()

    row = next(
        line
        for line in text.splitlines()
        if line.startswith("Countdown")
        and "success-conditioned signed Shannon advantage (E43)" in line
    )
    assert "—/—/—" in row
    assert "P / — / —" in row
    assert "retain both positive and negative bounded semantic pressure" in text


def test_e43_step_zero_pairing_requires_exact_match_to_retained_four_arms(
    tmp_path,
):
    fields = {
        "step": 0,
        "split": "multi_answer",
        "mean8": 0.25,
        "mean8_draws": [0.2, 0.3, 0.25, 0.25],
        "pass8": 0.5,
        "coverage8": 0.125,
        "distinct8": 0.75,
        "greedy": 0.1,
    }
    for e37_prefix, e38_prefix, e41_prefix, e43_prefix in (
        (
            "cde37_outcome_collision_05b_v1",
            "cde38_semantic_shannon_05b_v1",
            "cde41_semantic_shannon_advantage_05b_v1",
            "cde43_success_conditioned_signed_semantic_shannon_05b_v1",
        ),
        (
            "gce37_outcome_collision_05b_v1",
            "gce38_semantic_shannon_05b_v1",
            "gce41_semantic_shannon_advantage_05b_v1",
            "gce43_success_conditioned_signed_semantic_shannon_05b_v1",
        ),
    ):
        e37_rows = []
        e38_rows = []
        e41_rows = []
        e43_rows = []
        for seed in (43, 44, 45):
            for arm in ("grpo", "outcome_collision"):
                e37_rows.append({"arm": arm, "seed": seed, **fields})
            e38_rows.append(
                {"arm": "semantic_shannon", "seed": seed, **fields}
            )
            e41_rows.append(
                {
                    "arm": "semantic_shannon_advantage",
                    "seed": seed,
                    **fields,
                }
            )
            e43_rows.append(
                {
                    "arm": "success_conditioned_signed_semantic_shannon",
                    "seed": seed,
                    **fields,
                }
            )
        (tmp_path / f"{e37_prefix}_scaling_curve.json").write_text(
            json.dumps(e37_rows),
            encoding="utf-8",
        )
        (tmp_path / f"{e38_prefix}_scaling_curve.json").write_text(
            json.dumps(e38_rows),
            encoding="utf-8",
        )
        (tmp_path / f"{e41_prefix}_scaling_curve.json").write_text(
            json.dumps(e41_rows),
            encoding="utf-8",
        )
        (tmp_path / f"{e43_prefix}_scaling_curve.json").write_text(
            json.dumps(e43_rows),
            encoding="utf-8",
        )

    assert e38_step_zero_pairing_status(tmp_path).startswith(
        "PASS E37/E38/E41/E43"
    )

    graph_path = (
        tmp_path / "gce38_semantic_shannon_05b_v1_scaling_curve.json"
    )
    graph_rows = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_rows[0]["mean8"] = 0.251
    graph_path.write_text(json.dumps(graph_rows), encoding="utf-8")

    assert "Graph/s43" in e38_step_zero_pairing_status(tmp_path)


def test_e39_e41_e43_step_zero_pairing_requires_exact_match_across_all_five_arms(
    tmp_path,
):
    prefix = "mte39_math12k_384_semantic_entropy_05b_v1"
    fields = {
        "step": 0,
        "split": "math",
        "mean8": 0.25,
        "mean8_draws": [0.25],
        "pass8": 0.5,
        "pass8_draws": [0.5],
        "coverage8": None,
        "coverage8_draws": [],
        "distinct8": 0.75,
        "distinct8_draws": [0.75],
        "greedy": 0.1,
        "greedy_draws": [],
    }
    rows = [
        {"arm": arm, "seed": seed, **fields}
        for seed in (43, 44, 45)
        for arm in ("grpo", "outcome_collision", "semantic_shannon")
    ]
    path = tmp_path / f"{prefix}_scaling_curve.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    e41_path = (
        tmp_path
        / "mte41_math12k_384_semantic_shannon_advantage_05b_v1_scaling_curve.json"
    )
    e41_path.write_text(
        json.dumps(
            [
                {
                    "arm": "semantic_shannon_advantage",
                    "seed": seed,
                    **fields,
                }
                for seed in (43, 44, 45)
            ]
        ),
        encoding="utf-8",
    )
    e43_path = (
        tmp_path
        / "mte43_math12k_384_success_conditioned_signed_semantic_shannon_05b_v1_scaling_curve.json"
    )
    e43_path.write_text(
        json.dumps(
            [
                {
                    "arm": "success_conditioned_signed_semantic_shannon",
                    "seed": seed,
                    **fields,
                }
                for seed in (43, 44, 45)
            ]
        ),
        encoding="utf-8",
    )

    status = semantic_entropy_step_zero_pairing_status(tmp_path)

    assert "E39/E41/E43 PASS (3/3 exact five-arm seeds)" in status

    rows[4]["distinct8"] = 0.751
    path.write_text(json.dumps(rows), encoding="utf-8")

    assert "MATH/s44" in semantic_entropy_step_zero_pairing_status(tmp_path)


def test_e37_only_dashboard_shows_pending_e39_math_and_disclosure(
    monkeypatch,
    tmp_path,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    prefix = "mte39_math12k_384_semantic_entropy_05b_v1"
    stamp = f"{prefix}_outcome_collision_s43"
    (artifacts / f"{prefix}_comparative_jobs.tsv").write_text(
        "job_id\trun_stamp\n" f"3901\t{stamp}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue",
        lambda _user: (
            {"3901": JobInfo("3901", "PENDING", "Priority")},
            None,
        ),
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs",
        lambda _ids: {},
    )

    text, _ = Dashboard(
        tmp_path / "data",
        artifacts,
        45,
        e37_only=True,
    ).snapshot()

    row = next(
        line
        for line in text.splitlines()
        if line.startswith("MATH-500")
        and "semantic collision entropy (E39)" in line
    )
    assert "—/—/—" in row
    assert "P / — / —" in row
    assert "one fixed K=8 draw (seed 390100)" in text
    assert "coverage@8 is N/A" in text
    assert "not reasoning paths" in text


def _record(step, consumed, *, evaluation=False):
    record = {"misc/global_step": step, "misc/prompt_consumed": consumed}
    if evaluation:
        record["eval/multi_answer/sampled_mode_coverage_at_8"] = 0.25
    return json.dumps(record) + "\n"


def test_metric_cache_separates_landed_and_live_progress(tmp_path):
    run_dir = tmp_path / "run"
    debug_dir = run_dir / "debug_1"
    debug_dir.mkdir(parents=True)
    metrics_path = debug_dir / "train_metrics.jsonl"
    metrics_path.write_text(
        _record(2, 20, evaluation=True) + _record(3, 30), encoding="utf-8"
    )
    spec = RunSpec("env", "scale", "method", "arm", "stamp", 10, 2)
    cache = MetricCache()

    metrics = cache.run_metrics(run_dir, spec)

    assert metrics.landed_passes == 1.0
    assert metrics.latest_step == 3
    assert metrics.latest_passes == 1.5
    assert metrics.furthest_passes == 1.5

    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(_record(12, 120, evaluation=True))
    os.utime(metrics_path, None)
    metrics = cache.run_metrics(run_dir, spec)

    assert metrics.landed_passes == 6.0
    assert metrics.latest_step == 12
    assert metrics.latest_passes == 6.0
    assert metrics.furthest_passes == 6.0


def test_metric_cache_merges_duplicate_run_directories(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "debug_1").mkdir(parents=True)
    (second / "debug_2").mkdir(parents=True)
    (first / "debug_1" / "train_metrics.jsonl").write_text(
        _record(8, 80, evaluation=True), encoding="utf-8"
    )
    (second / "debug_2" / "train_metrics.jsonl").write_text(
        _record(10, 100, evaluation=True), encoding="utf-8"
    )
    spec = RunSpec("env", "scale", "method", "arm", "stamp", 10, 2)

    metrics = MetricCache().run_metrics([first, second], spec)

    assert metrics.landed_passes == 5.0


def test_metric_cache_supports_one_pass_math_eval_key(tmp_path):
    run_dir = tmp_path / "run"
    debug_dir = run_dir / "debug_1"
    debug_dir.mkdir(parents=True)
    metrics_path = debug_dir / "train_metrics.jsonl"
    math_eval_key = "eval/math/sampled_any_correct_at_8"
    metrics_path.write_text(
        json.dumps(
            {
                "misc/global_step": 2129,
                "misc/prompt_consumed": 2129 * 16,
                math_eval_key: 0.6,
            }
        )
        + "\n"
        + json.dumps(
            {
                "misc/global_step": 2200,
                "misc/prompt_consumed": 2200 * 16,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = RunSpec(
        "MATH-500 (free-form)",
        "0.5B",
        "Dr.GRPO",
        "grpo",
        "mte21_math_conditional_token_05b_v4",
        8515,
        16,
        eval_key=math_eval_key,
        max_passes=1.0,
    )

    metrics = MetricCache().run_metrics(run_dir, spec)

    assert metrics.landed_passes == 2129 / 8515
    assert metrics.latest_passes == 2200 / 8515


def test_e21_free_form_math_specs_are_one_pass_rows():
    specs = [
        spec
        for spec in campaign_specs()
        if spec.stamp_prefix == "mte21_math_conditional_token_05b_v4"
    ]

    assert len(specs) == 4
    assert {spec.arm for spec in specs} == {
        "grpo",
        "maxent",
        "maxent_control",
        "maxent_dual",
    }
    assert {spec.max_passes for spec in specs} == {1.0}
    assert {spec.eval_key for spec in specs} == {
        "eval/math/sampled_any_correct_at_8"
    }
    assert {spec.prompt_pool_size for spec in specs} == {8515}


def test_e26_is_a_separate_treatment_only_high_entropy_math_row():
    specs = [
        spec
        for spec in campaign_specs()
        if spec.stamp_prefix
        == "mte26_math_freeform_conditional_dual_high_entropy_05b_v2"
    ]

    assert len(specs) == 1
    assert specs[0].environment == "MATH-500 (free-form)"
    assert specs[0].arm == "maxent_dual"
    assert specs[0].max_passes == 1.0
    assert "base-preserving Haarnoja dual" in specs[0].method


def test_e22_control_and_e27_treatment_are_the_active_05b_rows():
    control_specs = [
        spec
        for spec in campaign_specs()
        if spec.stamp_prefix.endswith("e22_freeform_conditional_dual_05b_v2")
    ]
    treatment_specs = [
        spec
        for spec in campaign_specs()
        if spec.stamp_prefix.endswith("e27_freeform_conditional_dual_05b_v1")
    ]

    assert len(control_specs) == 2
    assert len(treatment_specs) == 2
    assert {spec.environment for spec in control_specs + treatment_specs} == {
        "Countdown",
        "Graph coloring",
    }
    assert {spec.scale for spec in control_specs + treatment_specs} == {"0.5B"}
    assert {spec.arm for spec in control_specs} == {"grpo"}
    assert {spec.arm for spec in treatment_specs} == {"maxent_dual"}
    assert {spec.prompt_pool_size for spec in control_specs + treatment_specs} == {192, 384}
    assert {spec.max_passes for spec in control_specs + treatment_specs} == {10.0}

    countdown_methods = [
        spec.method
        for spec in campaign_specs()
        if spec.environment == "Countdown" and spec.scale == "0.5B"
    ]
    graph_methods = [
        spec.method
        for spec in campaign_specs()
        if spec.environment == "Graph coloring" and spec.scale == "0.5B"
    ]
    assert countdown_methods[-2:] == [
        "matched free-form Dr.GRPO",
        "Free-form conditional-token MaxEnt base-preserving Haarnoja dual (125% entropy target)",
    ]
    assert graph_methods[-2:] == [
        "matched free-form Dr.GRPO",
        "Free-form conditional-token MaxEnt base-preserving Haarnoja dual (125% entropy target)",
    ]


def test_e25_and_e28_repair_v2_are_the_only_active_free_form_3b_rows():
    specs = campaign_specs()
    treatment_specs = [
        spec
        for spec in specs
        if spec.stamp_prefix.endswith("e25_freeform_conditional_dual_3b_repair_v2")
    ]
    control_specs = [
        spec
        for spec in specs
        if spec.stamp_prefix.endswith("e28_freeform_drgrpo_3b_repair_v2")
    ]

    assert len(treatment_specs) == 2
    assert len(control_specs) == 2
    assert {spec.environment for spec in treatment_specs + control_specs} == {
        "Countdown",
        "Graph coloring",
    }
    assert {spec.scale for spec in treatment_specs + control_specs} == {"3B"}
    assert {spec.arm for spec in treatment_specs} == {"maxent_dual"}
    assert {spec.arm for spec in control_specs} == {"grpo"}
    assert {spec.prompt_pool_size for spec in treatment_specs + control_specs} == {192, 384}
    assert {spec.num_samples for spec in treatment_specs + control_specs} == {16}
    assert {spec.metric_prefixes for spec in treatment_specs + control_specs} == {()}
    assert not any(
        spec.stamp_prefix.endswith("e25_freeform_conditional_dual_3b_v2")
        or spec.stamp_prefix.endswith("e28_freeform_drgrpo_3b_v1")
        for spec in specs
    )

    for environment in ("Countdown", "Graph coloring"):
        methods = [
            spec.method
            for spec in specs
            if spec.environment == environment and spec.scale == "3B"
        ]
        assert methods[-2:] == [
            "matched free-form Dr.GRPO",
            "Free-form conditional-token MaxEnt base-preserving Haarnoja dual (125% entropy target)",
        ]


def test_dashboard_uses_repair_v2_job_and_ignores_abandoned_3b_metrics(
    monkeypatch, tmp_path
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    old_prefix = "cde25_freeform_conditional_dual_3b_v2"
    repair_prefix = "cde25_freeform_conditional_dual_3b_repair_v2"
    old_stamp = f"{old_prefix}_maxent_dual_s43"
    repair_stamp = f"{repair_prefix}_maxent_dual_s43"
    (artifacts / f"{old_prefix}_comparative_jobs.tsv").write_text(
        "job_id\trun_stamp\n" f"900\t{old_stamp}\n", encoding="utf-8"
    )
    (artifacts / f"{repair_prefix}_comparative_jobs.tsv").write_text(
        "job_id\trun_stamp\n" f"901\t{repair_stamp}\n", encoding="utf-8"
    )

    data = tmp_path / "data"
    old_debug = data / f"attempt_{old_stamp}" / "debug_1"
    repair_debug = data / f"attempt_{repair_stamp}" / "debug_1"
    old_debug.mkdir(parents=True)
    repair_debug.mkdir(parents=True)
    # Countdown has 384 prompts and K=16. The abandoned attempt reached four
    # passes; the clean replacement has reached one quarter pass.
    (old_debug / "train_metrics.jsonl").write_text(
        _record(1536, 4 * 384 * 16, evaluation=True), encoding="utf-8"
    )
    (repair_debug / "train_metrics.jsonl").write_text(
        _record(96, 96 * 16, evaluation=True), encoding="utf-8"
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue",
        lambda _user: ({"901": JobInfo("901", "RUNNING")}, None),
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs", lambda _ids: {}
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.e16_protocol_is_frozen", lambda: False
    )

    dashboard = Dashboard(data, artifacts, 45)
    text, _ = dashboard.snapshot()

    repair_line = next(
        line
        for line in text.splitlines()
        if line.startswith("Countdown")
        and "3B" in line
        and "125% entropy target" in line
    )
    assert dashboard.job_stamps == {"901": repair_stamp}
    assert "0.25/—/—" in repair_line
    assert "R96@0.25 / — / —" in repair_line
    assert "4.00" not in repair_line


def test_countdown_7b_uses_canonical_e23_and_freeform_e29_namespaces():
    specs = [
        spec
        for spec in campaign_specs()
        if spec.environment == "Countdown" and spec.scale == "7B"
    ]

    assert len(specs) == 5
    assert {spec.stamp_prefix for spec in specs} == {
        "cde23_canonical_maxent_7b_v6_4xa100_evalsync_fix",
        "cde29_freeform_7b_4gpu_v5_buffer_restore",
    }
    assert {spec.num_samples for spec in specs} == {16}
    assert [spec.method for spec in specs[-2:]] == [
        "matched free-form Dr.GRPO",
        "Free-form conditional-token MaxEnt base-preserving Haarnoja dual (125% entropy target)",
    ]


def test_graph_7b_uses_canonical_e24_and_freeform_e29_namespaces():
    specs = [
        spec
        for spec in campaign_specs()
        if spec.environment == "Graph coloring" and spec.scale == "7B"
    ]

    assert len(specs) == 5
    assert {spec.stamp_prefix for spec in specs} == {
        "gce24_canonical_maxent_7b_v4_4xa100_evalsync_fix",
        "gce29_freeform_7b_4gpu_v5_buffer_restore",
    }
    assert {spec.prompt_pool_size for spec in specs} == {192}
    assert {spec.num_samples for spec in specs} == {16}


def test_find_run_dirs_keeps_duplicate_stamp_directories(tmp_path):
    stamp = "campaign_arm_s43"
    (tmp_path / f"old_{stamp}").mkdir()
    (tmp_path / f"new_{stamp}").mkdir()

    found = find_run_dirs(tmp_path, {stamp})

    assert {path.name for path in found[stamp]} == {f"old_{stamp}", f"new_{stamp}"}


def test_parse_replacement_run_stamp_from_slurm_submit_line():
    output = (
        "JobId=123 JobState=PENDING SubmitLine=sbatch --export=ALL,"
        "RUN_STAMP=cde5_haarnoja_7b_v1_xdr_sac_dual_s43,"
        "OAT_ZERO_SEED=43 script.slurm"
    )

    assert (
        parse_run_stamp_from_scontrol(output)
        == "cde5_haarnoja_7b_v1_xdr_sac_dual_s43"
    )


def test_live_label_does_not_reuse_metrics_from_an_old_attempt():
    active = JobInfo("123", "RUNNING", start_time=1_000)
    old_metrics = RunMetrics(3.0, 400, 2.0, 500)

    assert live_label(active, None, old_metrics) == "R init"


def test_completion_credit_is_monotonic_across_attempts():
    metrics = RunMetrics(1.0, 3, 0.5, 1_000, furthest_passes=2.5)

    assert completion_credit("R3@0.50", metrics) == 2.5
    assert completion_credit("done", metrics) == 10.0


def test_e16_specs_show_canonical_design_while_larger_scales_remain_held(
    monkeypatch, tmp_path
):
    maxent_specs = [
        spec
        for spec in campaign_specs()
        if spec.arm.startswith("maxent")
        and spec.stamp_prefix != "mte21_math_conditional_token_05b_v4"
        and "freeform" not in spec.stamp_prefix
    ]
    e16_specs = [spec for spec in maxent_specs if spec.scale == "0.5B"]
    held_specs = [spec for spec in maxent_specs if spec.scale != "0.5B"]

    assert len(maxent_specs) == 18
    assert len(e16_specs) == 6
    assert {spec.empty_status for spec in e16_specs} == {"DESIGN"}
    assert {
        spec.stamp_prefix for spec in e16_specs if spec.environment == "Countdown"
    } == {"cde16_canonical_maxent_05b_v2"}
    assert {
        spec.stamp_prefix
        for spec in e16_specs
        if spec.environment == "Graph coloring"
    } == {"gce16_canonical_maxent_05b_v2"}
    assert len(held_specs) == 12
    assert {spec.empty_status for spec in held_specs} == {"HELD"}
    smoke_specs = e16_smoke_specs()
    assert len(smoke_specs) == 6
    assert {spec.environment for spec in smoke_specs} == {
        "Countdown",
        "Graph coloring",
    }
    assert {spec.arm for spec in smoke_specs} == {
        "maxent",
        "maxent_control",
        "maxent_dual",
    }
    assert {spec.run_stamp(9006) for spec in smoke_specs} == {
        "cde16_canonical_maxent_joint_smoke_v3_maxent_s9006",
        "cde16_canonical_maxent_joint_smoke_v3_maxent_control_s9006",
        "cde16_canonical_maxent_joint_smoke_v3_maxent_dual_s9006",
        "gce16_canonical_maxent_joint_smoke_v3_maxent_s9006",
        "gce16_canonical_maxent_joint_smoke_v3_maxent_control_s9006",
        "gce16_canonical_maxent_joint_smoke_v3_maxent_dual_s9006",
    }

    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue", lambda _user: ({}, None)
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs", lambda _ids: {}
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.e16_protocol_is_frozen", lambda: False
    )
    text, _ = Dashboard(tmp_path / "data", tmp_path / "artifacts", 45).snapshot()

    assert text.count("Standard MaxEnt fixed") == 6
    assert text.count("Standard MaxEnt proportional") == 6
    assert text.count("Standard MaxEnt Haarnoja dual") == 6
    assert "E16 full 0.5B cells" in text
    assert "Design-waiting: 18" in text
    assert "Scale-held: 36" in text
    assert "DESIGN / DESIGN / DESIGN" in text
    assert "HELD / HELD / HELD" in text
    assert "0/0 settled" in text
    assert "no jobs submitted" in text


def test_canonical_maxent_cells_use_matched_canonical_drgrpo_controls():
    specs = campaign_specs()
    matched = [spec for spec in specs if spec.method == "matched canonical Dr.GRPO"]

    assert {
        (spec.environment, spec.scale, spec.stamp_prefix)
        for spec in matched
    } == {
        ("Countdown", "0.5B", "cde19_canonical_drgrpo_05b_v1"),
        ("Graph coloring", "0.5B", "gce19_canonical_drgrpo_05b_v1"),
        ("Countdown", "3B", "cde18_canonical_drgrpo_3b_v1"),
        ("Graph coloring", "3B", "gce18_canonical_drgrpo_3b_v1"),
    }
    assert {spec.num_samples for spec in matched} == {16}

    assert not any(
        spec.arm in {"xdr_tau0p05", "xdr_tau_control", "xdr_sac_dual"}
        for spec in specs
    )
    assert not any(
        spec.method == "Dr.GRPO"
        and spec.stamp_prefix != "mte21_math_conditional_token_05b_v4"
        for spec in specs
    )


def test_e16_frozen_protocol_surfaces_truthful_not_submitted_smoke(
    monkeypatch, tmp_path
):
    protocol = tmp_path / "e16.md"
    protocol.write_text("**Status: FROZEN — prospective**\n", encoding="utf-8")
    assert e16_protocol_is_frozen(protocol)
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.e16_protocol_is_frozen", lambda: True
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue", lambda _user: ({}, None)
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs", lambda _ids: {}
    )

    text, _ = Dashboard(tmp_path / "data", tmp_path / "artifacts", 45).snapshot()

    smoke_line = next(
        line for line in text.splitlines() if line.startswith("E16 smoke s9006")
    )
    assert smoke_line.count("not-submitted") == 6
    assert "Pending: 0" in text
    assert "0/0 settled" in text


def test_retired_free_form_e16_smoke_manifests_are_not_reported_as_canonical(
    monkeypatch, tmp_path
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    prefix = "cde16_standard_maxent_joint_smoke_v1"
    (artifacts / f"{prefix}_comparative_jobs.tsv").write_text(
        "job_id\trun_stamp\n"
        f"201\t{prefix}_maxent_s9006\n"
        f"202\t{prefix}_maxent_control_s9006\n"
        f"203\t{prefix}_maxent_dual_s9006\n",
        encoding="utf-8",
    )
    data = tmp_path / "data"
    debug = data / f"attempt_{prefix}_maxent_control_s9006" / "debug_1"
    debug.mkdir(parents=True)
    (debug / "train_metrics.jsonl").write_text(
        _record(5, 80), encoding="utf-8"
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue",
        lambda _user: (
            {
                "201": JobInfo("201", "PENDING", "Priority"),
                "202": JobInfo("202", "RUNNING"),
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs",
        lambda _ids: {"203": JobInfo("203", "COMPLETED")},
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.fill_active_job_stamps",
        lambda _queue, _job_stamps, _target_stamps: None,
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.e16_protocol_is_frozen", lambda: False
    )

    text, _ = Dashboard(data, artifacts, 45).snapshot()

    assert "E16 smoke s9006" not in text
    assert "no jobs submitted" in text
    assert "Running: 0  Pending: 0" in text
    assert "0/0 settled" in text


def test_e16_canonical_manifest_replaces_design_status_with_pending_state(
    monkeypatch, tmp_path
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    manifest = artifacts / "cde16_canonical_maxent_05b_v2_comparative_jobs.tsv"
    manifest.write_text(
        "job_id\trun_stamp\n"
        "123\tcde16_canonical_maxent_05b_v2_maxent_s43\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue",
        lambda _user: ({"123": JobInfo("123", "PENDING", "Priority")}, None),
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs", lambda _ids: {}
    )

    text, _ = Dashboard(tmp_path / "data", artifacts, 45).snapshot()

    countdown_fixed = next(
        line
        for line in text.splitlines()
        if line.startswith("Countdown") and "Standard MaxEnt fixed" in line
    )
    assert "P / DESIGN / DESIGN" in countdown_fixed
    assert "Pending: 1" in text
    assert "Design-waiting: 17" in text


def test_e16_canonical_smoke_line_uses_real_scheduler_and_metric_states(
    monkeypatch, tmp_path
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    countdown_prefix = "cde16_canonical_maxent_joint_smoke_v3"
    graph_prefix = "gce16_canonical_maxent_joint_smoke_v3"
    manifest = artifacts / "e16_canonical_smoke_comparative_jobs.tsv"
    manifest.write_text(
        "job_id\trun_stamp\n"
        f"301\t{countdown_prefix}_maxent_s9006\n"
        f"302\t{countdown_prefix}_maxent_control_s9006\n"
        f"303\t{countdown_prefix}_maxent_dual_s9006\n"
        f"304\t{graph_prefix}_maxent_s9006\n",
        encoding="utf-8",
    )
    debug = (
        tmp_path
        / "data"
        / f"attempt_{countdown_prefix}_maxent_control_s9006"
        / "debug_1"
    )
    debug.mkdir(parents=True)
    (debug / "train_metrics.jsonl").write_text(
        _record(12, 192), encoding="utf-8"
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue",
        lambda _user: (
            {
                "301": JobInfo("301", "PENDING", "Priority"),
                "302": JobInfo("302", "RUNNING"),
            },
            None,
        ),
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs",
        lambda _ids: {
            "303": JobInfo("303", "COMPLETED"),
            "304": JobInfo("304", "FAILED"),
        },
    )

    text, _ = Dashboard(tmp_path / "data", artifacts, 45).snapshot()

    smoke_line = next(
        line for line in text.splitlines() if line.startswith("E16 smoke s9006")
    )
    assert "Countdown [fixed: P; proportional: R12@0.03; Haarnoja dual: done]" in smoke_line
    assert "Graph coloring [fixed: FAIL" in smoke_line
    assert smoke_line.count("not-submitted") == 2
    assert "Running: 1  Pending: 1" in text
    assert "Terminal failures: 1" in text
    # Smoke jobs are operational gates and never enter the analytical
    # completion denominator.  Full rows remain design-stage until submitted.
    assert "0/0 settled" in text
    assert "DESIGN / DESIGN / DESIGN" in text


def test_manifest_without_scheduler_state_is_submitted_not_pending(
    monkeypatch, tmp_path
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    prefix = "cde16_canonical_maxent_joint_smoke_v3"
    (artifacts / "smoke_comparative_jobs.tsv").write_text(
        "job_id\trun_stamp\n" f"401\t{prefix}_maxent_s9006\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_queue", lambda _user: ({}, "down")
    )
    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.query_terminal_jobs", lambda _ids: {}
    )

    text, _ = Dashboard(tmp_path / "data", artifacts, 45).snapshot()

    smoke_line = next(
        line for line in text.splitlines() if line.startswith("E16 smoke s9006")
    )
    assert "fixed: submitted" in smoke_line
    assert "fixed: P" not in smoke_line
    assert "Pending: 0" in text


def test_figure_refresher_starts_immediately_without_overlap(monkeypatch, tmp_path):
    launches = []

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

    def fake_popen(*args, **kwargs):
        process = FakeProcess()
        launches.append((args, kwargs, process))
        return process

    monkeypatch.setattr("exp_scaling.monitor_campaign.subprocess.Popen", fake_popen)
    refresher = FigureRefresher(60, tmp_path / "refresh.log")

    refresher.tick(now=0)
    refresher.tick(now=30)
    assert len(launches) == 1
    assert "running" in refresher.status()

    launches[0][2].returncode = 0
    refresher.tick(now=61)
    assert len(launches) == 2


def test_figure_refresher_forwards_e37_only_mode(monkeypatch, tmp_path):
    launches = []

    class FakeProcess:
        def poll(self):
            return None

    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.subprocess.Popen",
        lambda *args, **kwargs: launches.append((args, kwargs)) or FakeProcess(),
    )
    refresher = FigureRefresher(
        60,
        tmp_path / "refresh.log",
        ("--e37-only",),
    )

    refresher.tick(now=0)

    assert launches[0][0][0][-1] == "--e37-only"


def test_figure_refresher_forwards_current_canonical_only_mode(
    monkeypatch,
    tmp_path,
):
    launches = []

    class FakeProcess:
        def poll(self):
            return None

    monkeypatch.setattr(
        "exp_scaling.monitor_campaign.subprocess.Popen",
        lambda *args, **kwargs: launches.append((args, kwargs)) or FakeProcess(),
    )
    refresher = FigureRefresher(
        60,
        tmp_path / "refresh.log",
        ("--current-canonical-only",),
    )

    refresher.tick(now=0)

    assert launches[0][0][0][-1] == "--current-canonical-only"
