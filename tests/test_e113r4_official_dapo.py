"""Static and verifier gates for the pinned official-verl DAPO successor."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ops/exp_scaling"
sys.path.insert(0, str(TOOLS))
import e113r4_official_dapo_common as common  # noqa: E402


def test_scaled_batch_geometry_preserves_official_ratios() -> None:
    assert common.GEN_PROMPT_BATCH == 3 * common.TRAIN_PROMPT_BATCH
    assert common.GEN_PROMPT_BATCH == 384
    assert common.RESPONSES_PER_PROMPT == 16
    assert common.PPO_MINI_BATCH == 32
    assert common.TRAIN_PROMPT_BATCH % common.PPO_MINI_BATCH == 0
    assert common.TOTAL_TRAINING_STEPS == 24
    assert common.TOTAL_TRAINING_STEPS * common.TRAIN_PROMPT_BATCH == 3_072
    assert common.MAX_SAMPLED_RESPONSES == 1_474_560


def test_upstream_sampler_is_the_real_pooled_dapo_path() -> None:
    source = (common.VERL_ROOT / "recipe/dapo/src/dapo_ray_trainer.py").read_text(
        encoding="utf-8"
    )
    required = (
        "prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)",
        "if std > 0 or len(prompt_uid2metric_vals[uid]) == 1",
        "batch = DataProto.concat([batch, new_batch])",
        "if num_prompt_in_batch < prompt_bsz:",
        "max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches",
        "batch = batch[:traj_bsz]",
        'metrics["train/num_gen_batches"] = num_gen_batches',
    )
    for contract in required:
        assert contract in source


def test_runner_retains_published_dapo_objective() -> None:
    source = (ROOT / "ops/run_e113r4_official_dapo.sh").read_text(encoding="utf-8")
    required = (
        "python3 -m recipe.dapo.src.main_dapo",
        "data.gen_batch_size=384",
        "data.train_batch_size=128",
        "actor_rollout_ref.rollout.n=16",
        "algorithm.adv_estimator=grpo",
        "algorithm.filter_groups.enable=True",
        "algorithm.filter_groups.metric=acc",
        'max_num_gen_batches="${E113R4_MAX_NUM_GEN_BATCHES:-10}"',
        '"algorithm.filter_groups.max_num_gen_batches=$max_num_gen_batches"',
        "actor_rollout_ref.actor.clip_ratio_low=0.20",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "reward_model.reward_manager=dapo",
    )
    for contract in required:
        assert contract in source


def test_runner_uses_short_node_local_ray_socket_root() -> None:
    source = (ROOT / "ops/run_e113r4_official_dapo.sh").read_text(encoding="utf-8")
    assert 'mktemp -d "/tmp/e113r4-${SLURM_JOB_ID:-manual}-XXXXXX"' in source
    assert 'export TMPDIR="$job_tmp"' in source
    assert 'export RAY_TMPDIR="$job_tmp"' in source
    assert '--bind "$job_tmp:$job_tmp"' in source
    assert 'RAY_TMPDIR="$E113R4_OUTPUT/tmp/ray"' not in source


def test_r4r2_release_survives_completed_smoke_controller_expiry() -> None:
    source = (
        ROOT / "ops/exp_scaling/release_e113r4r2_science_after_smokes.py"
    ).read_text(encoding="utf-8")
    amendment = ROOT / (
        "paper/preregistration/"
        "e113r4r2s1_completed_smoke_dependency_expiry_20260824.md"
    )
    assert amendment.is_file()
    assert 'parser.add_argument("--launch", action="store_true")' in source
    assert "gate = smoke_gate(payload)" in source
    assert 'if not gate["passed"]' in source
    assert '"--hold"' in source
    assert '"--dependency=" in submit_line' in source
    assert 'f"--dependency=afterok:' not in source


def test_data_manifest_has_exact_ten_split_contract() -> None:
    payload = json.loads(common.DATA_MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] == "e113r4_official_verl_dapo_data_v1"
    assert payload["train_rows"] == 384
    assert payload["prompt_rendering_exactly_matches_frozen_comparators"] is True
    observed = {
        (row["domain"], row["split"], row["rows"]) for row in payload["outputs"]
    }
    expected = {
        (domain, split, 384 if split == "train" else 128)
        for domain in common.DOMAINS
        for split in ("train", "eval")
    }
    assert observed == expected


def test_reward_adapter_exposes_raw_binary_accuracy() -> None:
    path = TOOLS / "e113r4_verl_reward.py"
    spec = importlib.util.spec_from_file_location("e113r4_verl_reward_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ground_truth = json.dumps(
        {
            "verifier": "graph_coloring",
            "n": 5,
            "edges": [[1, 2], [1, 4], [1, 5], [2, 3], [2, 5], [3, 4]],
            "partial_colors": [None, 3, None, 3, None],
        }
    )
    correct = module.compute_score(
        "modebench/graph_coloring", "\\boxed{13132}", ground_truth
    )
    wrong = module.compute_score(
        "modebench/graph_coloring", "\\boxed{11111}", ground_truth
    )
    assert correct["score"] == correct["acc"] == 1.0
    assert wrong["score"] == wrong["acc"] == 0.0


def test_filter_exhaustion_recovery_is_exact_cell_and_bounded() -> None:
    launcher = (TOOLS / "recover_e113r4_filter_group_exhaustion.py").read_text(
        encoding="utf-8"
    )
    protocol = ROOT / (
        "paper/preregistration/"
        "e113r4r2s5_filter_group_exhaustion_recovery_20260829.md"
    )
    required = (
        "FAILED_JOB_ID = 30869122",
        "REJECTED_HELD_JOB_IDS = (30971093, 30971112)",
        "CAP_BEFORE = 10",
        "CAP_AFTER = 20",
        "MAX_EPOCHS_AFTER = 480",
        '"family": "qwen05b"',
        '"domain": "python_factors"',
        '"seed": 44',
        '"arm": "dapo"',
        "num_prompt_in_batch=122 < prompt_bsz=128",
        '"E113R4_MAX_NUM_GEN_BATCHES": str(CAP_AFTER)',
        '"E113R4_MAX_EPOCHS": str(MAX_EPOCHS_AFTER)',
        '"Account": "mltheory"',
        '"Partition": "all"',
        '"--qos=long"',
        '"efficacy_outcomes_inspected": False',
        '"accepted_batch_definition_changed": False',
        '"restart_from_initial_model": True',
    )
    assert protocol.is_file()
    assert all(contract in launcher for contract in required)


def test_common_python_filter_recovery_is_bounded_and_resumable() -> None:
    launcher = (TOOLS / "recover_e113r4_common_python_filter_exhaustion.py").read_text(
        encoding="utf-8"
    )
    protocol = ROOT / (
        "paper/preregistration/"
        "e113r4r2s6_common_python_filter_exhaustion_recovery_20260829.md"
    )
    required = (
        "43: 30869121",
        "45: 30869123",
        "46: 30869124",
        "47: 30869125",
        "PENDING_BY_SEED = {44: 30971120}",
        "TERMINAL_ACCEPTED = {43: 50, 45: 44, 46: 55, 47: 64}",
        "CAP_AFTER = 40",
        "MAX_EPOCHS_AFTER = 960",
        "CHECKPOINT_STEP = 10",
        "DISCARDED_STEP = 11",
        '"E113R4_MAX_NUM_GEN_BATCHES": str(CAP_AFTER)',
        '"E113R4_MAX_EPOCHS": str(MAX_EPOCHS_AFTER)',
        '"resume_checkpoint_step": CHECKPOINT_STEP',
        '"restart_from_initial_model": False',
        '"restart_from_initial_model": True',
        '"accepted_batch_definition_changed": False',
        '"common_recovery_source": True',
        '"efficacy_outcomes_used_for_repair": False',
        '"QOS=long"',
    )
    assert protocol.is_file()
    assert all(contract in launcher for contract in required)


def test_signal53_wave_recovery_is_exact_and_checkpoint_preserving() -> None:
    launcher = (TOOLS / "recover_e113r4_e117r2_signal53_wave.py").read_text(
        encoding="utf-8"
    )
    protocol = ROOT / (
        "paper/preregistration/" "e113r4_e117r2_signal53_wave_recovery_20260830.md"
    )
    required = (
        "E113_PROCESS_FAILURE_IDS",
        "E113_SIGNAL53_IDS",
        "E113_FAILED_IDS",
        "E113_RESUME_STEPS",
        '"scientific_environment_changed": False',
        '"source_snapshot_changed": False',
        '"run_directories_changed": False',
        '"efficacy_outcomes_inspected": False',
        '"Account=mltheory"',
        '"Partition=all"',
        '"QOS=long"',
        '"TimeLimit=12:00:00"',
        "validate_checkpoint",
        "all_replacements_held_and_audited_before_release",
        "collateral_job_id_excluded",
    )
    assert protocol.is_file()
    assert all(contract in launcher for contract in required)


def test_released_ledger_is_exact_official_r4_graph() -> None:
    payload = json.loads(common.LEDGER.read_text(encoding="utf-8"))
    assert payload["schema"] == "e113r4_official_verl_dapo_jobs_v1"
    assert payload["released"] is True
    assert payload["scientific_cells"] == 50
    assert payload["official_upstream_unmodified"] is True
    smoke_ids = {row["job_id"] for row in payload["smokes"].values()}
    assert len(payload["runs"]) == 50
    assert len({row["job_id"] for row in payload["runs"]}) == 50
    dependencies = {
        frozenset(row["dependency_smoke_job_ids"]) for row in payload["runs"]
    }
    recovery = payload.get("ray_socket_recovery")
    if recovery is None:
        assert dependencies == {frozenset(smoke_ids)}
        assert smoke_ids == {30800804, 30800805}
    else:
        assert recovery["schema"] == "e113r4_r4r1_ray_socket_recovery_v1"
        assert recovery["failed_smoke_job_ids"] == [30800804, 30800805]
        assert recovery["canceled_science_job_count"] == 50
        assert recovery["scientific_parameters_changed"] is False
        assert recovery["official_upstream_changed"] is False
        scheduler_recovery = payload.get("vllm_scheduler_recovery")
        if scheduler_recovery is None:
            assert {row["replaces_job_id"] for row in payload["runs"]} == set(
                recovery["canceled_science_job_ids"]
            )
            assert smoke_ids == set(recovery["replacement_smoke_job_ids"])
            assert dependencies == {frozenset(smoke_ids)}
        else:
            failed_scheduler_smokes = set(scheduler_recovery["failed_smoke_job_ids"])
            assert failed_scheduler_smokes == set(recovery["replacement_smoke_job_ids"])
            assert smoke_ids == set(scheduler_recovery["replacement_smoke_job_ids"])
            if not scheduler_recovery["science_replacements_submitted"]:
                assert {row["replaces_job_id"] for row in payload["runs"]} == set(
                    recovery["canceled_science_job_ids"]
                )
                assert dependencies == {frozenset(failed_scheduler_smokes)}
                assert scheduler_recovery["replacement_science_job_ids"] == []
                assert (
                    payload["runs"]
                    == scheduler_recovery["superseded_runs_pending_replacement"]
                )
            else:
                expected_replaced = set(
                    scheduler_recovery["zero_runtime_science_job_ids"]
                )
                observed_replaced = {row["replaces_job_id"] for row in payload["runs"]}
                falcon_filter_recovery = payload.get(
                    "falcon_python_filter_group_exhaustion_recovery"
                )
                if falcon_filter_recovery is not None:
                    assert falcon_filter_recovery["schema"] == (
                        "e113r4_r4r2s8_falcon_python_"
                        "filter_exhaustion_recovery_v1"
                    )
                    assert falcon_filter_recovery["released"] is True
                    assert falcon_filter_recovery["max_num_gen_batches_after"] == 80
                    for original in falcon_filter_recovery[
                        "original_jobs_by_seed"
                    ].values():
                        observed_replaced.discard(original["job_id"])
                        observed_replaced.add(original["replaces_job_id"])
                common_filter_recovery = payload.get(
                    "common_python_filter_group_exhaustion_recovery"
                )
                if common_filter_recovery is not None:
                    assert common_filter_recovery["schema"] == (
                        "e113r4_r4r2s6_common_python_" "filter_exhaustion_recovery_v1"
                    )
                    for original in common_filter_recovery[
                        "original_jobs_by_seed"
                    ].values():
                        observed_replaced.discard(original["job_id"])
                        observed_replaced.add(original["replaces_job_id"])
                filter_recovery = payload.get("filter_group_exhaustion_recovery")
                if filter_recovery is not None:
                    assert filter_recovery["schema"] == (
                        "e113r4_r4r2s5_" "filter_group_exhaustion_recovery_v1"
                    )
                    original = filter_recovery["original_job"]
                    observed_replaced.discard(original["job_id"])
                    observed_replaced.add(original["replaces_job_id"])
                assert observed_replaced == expected_replaced
                assert dependencies == {frozenset(smoke_ids)}
                assert len(scheduler_recovery["replacement_science_job_ids"]) == 50
                expiry = scheduler_recovery["science_scheduler_dependency_amendment"]
                assert expiry["schema"] == ("e113r4_r4r2s1_dependency_expiry_v1")
                assert expiry["scientific_parameters_changed"] is False
                assert all(
                    not any(
                        str(arg).startswith("--dependency=") for arg in row["command"]
                    )
                    for row in payload["runs"]
                )
    assert payload["placement_amendment"]["schema"] == "e113r4_r4p1_placement_only_v1"
    assert payload["placement_amendment"]["scientific_parameters_changed"] is False
    scheduling = payload["scheduling_amendment"]
    assert scheduling["schema"] == "e113r4_r4p2_scheduling_only_v1"
    assert scheduling["authoritative_job_count"] == 52
    assert scheduling["science_time_limit_after"] == "7-00:00:00"
    assert scheduling["smoke_time_limit_after"] == "1-00:00:00"
    assert scheduling["scientific_parameters_changed"] is False
    original_retirement = payload["original_e113_retirement"]
    assert original_retirement["registered_job_count"] == 50
    retirement = json.loads(Path(original_retirement["artifact"]).read_text())
    assert (
        retirement["state_after_cancellation"]["all_registered_jobs_canceled"] is True
    )
    assert retirement["scientific_cells_ever_started"] == 0
    assert retirement["eligible_for_named_dapo_efficacy"] is False
    assert common.VERIFIER_SITE.is_dir()
    assert "E113R4_VERIFIER_SITE" in (
        ROOT / "ops/run_e113r4_official_dapo.sh"
    ).read_text(encoding="utf-8")
