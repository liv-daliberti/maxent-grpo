from __future__ import annotations

import json
from pathlib import Path

from ops.exp_scaling import audit_e113r1_dapo_recovery_smokes as recovery_audit
from ops.exp_scaling import launch_e113_dapo_direct_baseline as launch
from ops.exp_scaling import launch_e113r1_dapo_recovery_smokes as recovery
from ops.exp_scaling import launch_e113r3_dapo_full_relaunch as relaunch
from ops.exp_scaling import launch_e113r1m1_qwen_memory_recovery as memory_recovery


ROOT = Path(__file__).resolve().parents[1]


def test_registered_grid_and_terminal_control_pairs_are_complete():
    expected = {
        (family, domain, seed)
        for family, seeds in launch.FAMILY_SEEDS.items()
        for domain in launch.DOMAINS
        for seed in seeds
    }
    observed = set()
    for family in launch.FAMILY_SEEDS:
        references = launch.references(ROOT, family)
        controls = launch.control_index(ROOT, family)
        assert len(references) == 25
        assert len(controls) == 25
        for run in references:
            key = (str(run["domain"]), int(run["seed"]))
            assert key in controls
            assert (
                Path(str(controls[key]["run_dir"])) / "TRAINING_COMPLETE.json"
            ).is_file()
            observed.add((family, *key))
    assert observed == expected


def test_objective_is_full_dapo_and_isolated_from_other_interventions():
    objective = launch.objective()
    assert objective["OAT_ZERO_VARIANT"] == "dapo"
    assert objective["OAT_ZERO_CRITIC_TYPE"] == "grpo"
    assert objective["OAT_ZERO_DAPO_ENABLED"] == "1"
    assert objective["OAT_ZERO_DAPO_CLIP_LOW"] == "0.20"
    assert objective["OAT_ZERO_DAPO_CLIP_HIGH"] == "0.28"
    assert objective["OAT_ZERO_DAPO_MAX_NUM_GEN_BATCHES"] == "10"
    assert objective["OAT_ZERO_DAPO_OVERLONG_BUFFER_RATIO"] == "0.20"
    assert objective["OAT_ZERO_DAPO_OVERLONG_PENALTY_FACTOR"] == "1.0"
    assert objective["OAT_ZERO_XDR_TAU"] == "inf"
    for key in (
        "OAT_ZERO_UCPO_TAU",
        "OAT_ZERO_POLICY_ENTROPY_COEF",
        "OAT_ZERO_SEED_ENTROPY_ALPHA",
        "OAT_ZERO_MAXENT_ALPHA",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF",
        "OAT_ZERO_OUTCOME_COLLISION_COEF",
        "OAT_ZERO_DIAYN_MI_BETA",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA",
    ):
        assert float(objective[key]) == 0.0
    assert objective["OAT_ZERO_RLEP_REPLAY_COUNT"] == "0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "0"
    assert objective["OAT_ZERO_VERIFIED_DISCOVERY_TRACKING"] == "0"


def test_query_ceiling_and_accepted_update_schedule_are_frozen():
    assert launch.TRAIN_ROWS == 384
    assert launch.PASSES == 8
    assert launch.TARGET_STEPS == 3072
    assert launch.CHECKPOINT_INTERVAL == 192
    assert launch.MAX_GENERATION_BATCHES == 10
    assert launch.MAX_QUERIES == 3072 * 16 * 10 == 491_520


def test_environments_inherit_both_model_surfaces_but_replace_objective():
    falcon_root = launch.e79.model_root(ROOT)
    for family in launch.FAMILY_SEEDS:
        run = launch.references(ROOT, family)[0]
        env, target = launch.build_env(ROOT, family, run, ROOT, falcon_root)
        assert target.name.startswith(f"xdr_{launch.MODEL_TAGS[family]}_dapo_")
        assert env["OAT_ZERO_VARIANT"] == "dapo"
        assert env["OAT_ZERO_MAX_QUERIES"] == "491520"
        assert env["OAT_ZERO_NUM_PROMPT_EPOCH"] == "8"
        assert env["OAT_ZERO_MAX_PROMPT_EPOCHS"] == "8"
        assert env["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "0"
        assert env["OAT_ZERO_SOURCE_ROOT"] == str(ROOT / "src")
        if family == "falcon1b":
            assert env["OAT_ZERO_PRETRAIN"] == str(falcon_root)


def test_scientific_jobs_depend_on_their_family_smoke():
    for family in launch.FAMILY_SEEDS:
        run = launch.references(ROOT, family)[0]
        env = launch.objective()
        command = launch.sbatch_command(
            ROOT,
            family,
            run,
            env,
            dependency="12345",
        )
        assert "--hold" in command
        assert "--dependency=afterok:12345" in command
        assert "--nice=100" in command
        expected_name = launch.job_name(family, str(run["domain"]), int(run["seed"]))
        assert f"--job-name={expected_name}" in command


def test_paper_registers_method_but_makes_no_dapo_outcome_claim():
    manuscript = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    readme = (ROOT / "paper/README.md").read_text(encoding="utf-8")
    manuscript_words = " ".join(manuscript.split())
    readme_words = " ".join(readme.split())
    assert "\\citep{yu2025dapo}" in manuscript
    assert "R3 excluded; R4-R2 gate passed; 4/50 science cells terminal; 46 nonterminal; 0 failed; 0 standardized endpoints" in manuscript_words
    assert "No matched pass@8/breadth DAPO point or numerical efficacy conclusion is reported" in manuscript_words
    assert "The analogous DAPO-minus-control estimand is" in manuscript_words
    assert "four terminal Qwen Graph science cells" in manuscript_words
    assert "Custom R3 excluded; official-verl R4-R2 gate passed; 4/50 science cells terminal; 46 nonterminal; 0 failed; 0 standardized endpoints" in readme_words
    assert "DAPO licenses no efficacy claim" in readme_words
    assert (ROOT / "paper/results/e113r4_dapo_progress.json").is_file()
    assert (ROOT / "paper/preregistration/e113r4r2_vllm_scheduler_recovery_20260824.md").is_file()
    assert (ROOT / "var/artifacts/e113_dapo_direct_baseline_jobs.json").is_file()
    assert (ROOT / "var/artifacts/e113r4_official_verl_dapo_jobs.json").is_file()


def test_recovery_smokes_change_only_operational_domain_and_budget():
    assert recovery.SMOKE_DOMAIN == "graph_coloring"
    assert recovery.SMOKE_MAX_TRAIN == 32
    assert recovery.SMOKE_MAX_QUERIES == 32 * 16 * 10 == 5120
    assert recovery.ORIGINAL_LEDGER == launch.LEDGER
    assert (ROOT / recovery.AMENDMENT).is_file()

    falcon_root = launch.e79.model_root(ROOT)
    for family, run in recovery.smoke_runs(ROOT).items():
        assert run["domain"] == "graph_coloring"
        assert int(run["seed"]) == int(launch.FAMILY_SEEDS[family][0])
        env = recovery.recovery_env(ROOT, family, run, ROOT, falcon_root)
        assert env["OAT_ZERO_VARIANT"] == "dapo"
        assert env["OAT_ZERO_DAPO_MAX_NUM_GEN_BATCHES"] == "10"
        assert env["OAT_ZERO_MAX_QUERIES"] == "5120"
        assert "graph" in env["OAT_ZERO_PROMPT_DATA"]
        assert env["SAVE_PATH"] == str(recovery.recovery_target(ROOT, family))
        command = recovery.recovery_command(ROOT, family, run, env)
        assert "--hold" in command
        assert f"--job-name={recovery.recovery_job_name(family)}" in command
        assert not any(token.startswith("--dependency=") for token in command)


def test_recovery_auditor_requires_two_clean_32_update_receipts(tmp_path):
    smokes = {}
    for offset, family in enumerate(sorted(recovery_audit.EXPECTED_FAMILIES)):
        run_dir = tmp_path / family
        attempt = run_dir / f"debug_job{100 + offset}"
        attempt.mkdir(parents=True)
        rows = [
            {
                "actor/dapo_accepted_groups": 1.0,
                "actor/dapo_dynamic_sampling_enabled": 1.0,
                "actor/dapo_generation_batches": 1.0,
                "train/dapo_enabled": 1.0,
                "train/dapo_clip_low": 0.2,
                "train/dapo_clip_high": 0.28,
                "train/dapo_token_level_active_tokens": 64.0,
                "train/pg_loss": 0.0,
                "train/policy_grad_norm": 0.1,
                "misc/query_step": float(16 * step),
                "trainer/global_step": float(step),
            }
            for step in range(1, 33)
        ]
        (attempt / "train_metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        (run_dir / "TRAINING_COMPLETE.json").write_text(
            json.dumps(
                {
                    "schema": "oat_zero_training_complete_v1",
                    "terminal_step": 32,
                    "terminal_attempt": str(attempt),
                }
            ),
            encoding="utf-8",
        )
        smokes[family] = {
            "job_id": 100 + offset,
            "run_dir": str(run_dir),
            "max_train": 32,
            "max_queries": 5120,
        }
    ledger = tmp_path / "jobs.json"
    ledger.write_text(
        json.dumps(
            {
                "schema": recovery_audit.EXPECTED_SCHEMA,
                "released": True,
                "scientific_cells": 0,
                "runs": [],
                "smoke_domain": "graph_coloring",
                "smokes": smokes,
            }
        ),
        encoding="utf-8",
    )
    report = recovery_audit.audit(ledger)
    assert report["passed"] is True
    assert all(row["accepted_updates"] == 32 for row in report["smokes"])


def test_effective_auditor_accepts_runner_step_33_only_with_32_policy_steps(
    tmp_path,
):
    run_dir = tmp_path / "falcon1b"
    attempt = run_dir / "debug_job30790112"
    attempt.mkdir(parents=True)
    rows = [
        {
            "actor/dapo_accepted_groups": 1.0,
            "actor/dapo_dynamic_sampling_enabled": 1.0,
            "actor/dapo_generation_batches": 1.0,
            "train/dapo_enabled": 1.0,
            "train/dapo_clip_low": 0.2,
            "train/dapo_clip_high": 0.28,
            "train/dapo_token_level_active_tokens": 64.0,
            "train/pg_loss": 0.0,
            "train/policy_grad_norm": 0.1,
            "misc/query_step": float(16 * step),
            "trainer/global_step": float(step),
        }
        for step in range(1, 33)
    ]
    rows.append({**rows[-1], "trainer/step": 33.0})
    (attempt / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "oat_zero_training_complete_v1",
                "terminal_step": 33,
                "terminal_attempt": str(attempt),
            }
        ),
        encoding="utf-8",
    )
    smoke = {
        "job_id": 30790112,
        "run_dir": str(run_dir),
        "max_train": 32,
        "max_queries": 5120,
    }
    strict = recovery_audit.audit_smoke("falcon1b", smoke)
    effective = recovery_audit.audit_smoke(
        "falcon1b", smoke, expected_receipt_step=33
    )
    assert strict["passed"] is False
    assert effective["passed"] is True
    assert effective["accepted_updates"] == 32
    assert effective["raw_accepted_records"] == 33
    assert effective["duplicate_accepted_records"] == 1


def test_full_relaunch_restores_all_50_cells_without_changing_dapo():
    assert relaunch.SCIENTIFIC_CELLS == 50
    assert relaunch.R1_LEDGER == recovery.LEDGER
    assert relaunch.M1_LEDGER == memory_recovery.LEDGER
    assert (ROOT / relaunch.PROTOCOL).is_file()
    original = relaunch.load_original(ROOT)
    assert relaunch.validate_m1_ledger(ROOT, original)["scientific_cells"] == 0
    s2, s3 = relaunch.validate_scheduler_amendments(ROOT)
    assert s2["job_id"] == s3["job_id"] == 30790590
    falcon_root = launch.e79.model_root(ROOT)
    cells = relaunch.prepare_cells(
        ROOT,
        ROOT,
        falcon_root,
        require_fresh=False,
    )
    keys = {
        (cell["family"], cell["run"]["domain"], int(cell["run"]["seed"]))
        for cell in cells
    }
    expected = {
        (family, domain, int(seed))
        for family, seeds in launch.FAMILY_SEEDS.items()
        for domain in launch.DOMAINS
        for seed in seeds
    }
    assert keys == expected
    assert len(cells) == 50
    for cell in cells:
        env = cell["env"]
        command = cell["command"]
        assert env["OAT_ZERO_VARIANT"] == "dapo"
        assert env["OAT_ZERO_DAPO_MAX_NUM_GEN_BATCHES"] == "10"
        assert env["OAT_ZERO_MAX_QUERIES"] == "491520"
        assert env["OAT_ZERO_WATCHDOG_REQUEUE"] == "0"
        assert env["OAT_ZERO_WATCHDOG_MAX_RESTARTS"] == "0"
        assert "e113r3" in env["SAVE_PATH"]
        assert "e113r3" in env["RUN_STAMP"]
        assert "--hold" in command
        assert not any(token.startswith("--dependency=") for token in command)
        if cell["family"] == "qwen05b":
            assert env["OAT_ZERO_ADAM_OFFLOAD"] == "0"
            assert env["OAT_ZERO_ACTIVATION_OFFLOADING"] == "0"
            assert f"--partition={relaunch.QWEN_PARTITION}" in command
            assert f"--account={memory_recovery.ACCOUNT}" in command
            assert f"--nodelist={memory_recovery.NODELIST}" in command
            assert f"--gres={memory_recovery.GRES}" in command


def test_full_relaunch_gate_fails_closed_before_both_smokes_pass(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(relaunch, "load_original", lambda _root: {})
    monkeypatch.setattr(relaunch, "validate_m1_ledger", lambda _root, _original: {})
    monkeypatch.setattr(relaunch, "validate_scheduler_amendments", lambda _root: ({}, {}))
    monkeypatch.setattr(
        relaunch.gate_audit,
        "audit",
        lambda _r1_path, _m1_path: {
            "passed": False,
            "violations": ["gate pending"],
            "smokes": [],
        },
    )
    try:
        relaunch.require_effective_gate(tmp_path)
    except SystemExit as error:
        assert "gate pending" in str(error)
    else:
        raise AssertionError("E113-R3 gate accepted a pending smoke audit")


def test_full_relaunch_gate_accepts_only_a_complete_audit(tmp_path, monkeypatch):
    expected = {"passed": True, "violations": [], "smokes": [{}, {}]}
    monkeypatch.setattr(relaunch, "load_original", lambda _root: {})
    monkeypatch.setattr(relaunch, "validate_m1_ledger", lambda _root, _original: {})
    monkeypatch.setattr(relaunch, "validate_scheduler_amendments", lambda _root: ({}, {}))
    monkeypatch.setattr(
        relaunch.gate_audit,
        "audit",
        lambda _r1_path, _m1_path: expected,
    )
    assert relaunch.require_effective_gate(tmp_path) is expected


def test_qwen_memory_recovery_changes_only_capacity_and_fresh_paths():
    run = memory_recovery.source_run(ROOT)
    env = memory_recovery.build_env(ROOT, run, ROOT)
    command = memory_recovery.command(ROOT, run, env)
    assert env["OAT_ZERO_VARIANT"] == "dapo"
    assert env["OAT_ZERO_MAX_QUERIES"] == "5120"
    assert env["OAT_ZERO_TRAIN_BATCH_SIZE"] == "16"
    assert env["OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE"] == "16"
    assert env["OAT_ZERO_ADAM_OFFLOAD"] == "0"
    assert env["OAT_ZERO_ACTIVATION_OFFLOADING"] == "0"
    assert "e113r1m1" in env["SAVE_PATH"]
    assert f"--partition={memory_recovery.PARTITION}" in command
    assert f"--account={memory_recovery.ACCOUNT}" in command
    assert f"--nodelist={memory_recovery.NODELIST}" in command
    assert f"--gres={memory_recovery.GRES}" in command
    assert "--hold" in command
    assert not any(token.startswith("--dependency=") for token in command)


def test_r3_qwen_partition_normalization_is_held_stage_only(monkeypatch):
    calls = []

    def fake_run(command, *, check):
        calls.append((command, check))

    monkeypatch.setattr(relaunch.subprocess, "run", fake_run)
    relaunch.normalize_qwen_partition("123")
    assert calls == [
        (["scontrol", "update", "JobId=123", "Partition=all"], True)
    ]


def test_r3_terminal_failure_cleanup_is_exact_and_never_requeues():
    wrapper = (ROOT / "ops/slurm/train_node302.slurm").read_text(encoding="utf-8")
    assert "BEGIN E113R3_TERMINAL_FAILURE_CLEANUP_AMENDMENT" in wrapper
    assert "SLURM_JOB_ID >= 30790925" in wrapper
    assert "SLURM_JOB_ID <= 30790974" in wrapper
    assert "OAT_ZERO_WATCHDOG_STALE_SECONDS=300" in wrapper
    assert "DAPO dynamic sampling could not produce a non-constant reward group" in wrapper
    assert '[[ "${OAT_ZERO_WATCHDOG_REQUEUE:-}" != "0" ]]' in wrapper


def test_r3_external_reaper_is_exact_cpu_only_and_never_requeues():
    monitor = (
        ROOT / "ops/exp_scaling/monitor_e113r3_failed_allocations.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "ops/exp_scaling/launch_e113r3_failure_reaper.py"
    ).read_text(encoding="utf-8")
    assert "list(range(30790925, 30790975))" in monitor
    assert "cleanup utility changed after reaper release" in monitor
    assert "TRAINING_COMPLETE.json" in monitor
    assert 'state != "RUNNING"' in monitor
    assert '"--no-requeue"' in launcher
    assert '"--cpus-per-task=1"' in launcher
    assert '"--mem=1G"' in launcher
    assert "--gres" not in launcher
