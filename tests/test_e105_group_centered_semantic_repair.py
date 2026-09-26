from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops" / "exp_scaling" / "launch_e105_group_centered_semantic_repair_full_three_scale.py"
PROTOCOL = ROOT / "paper" / "preregistration" / "e105_group_centered_semantic_repair_full_three_scale_20260817.md"
AMENDMENT = ROOT / "paper" / "preregistration" / "e105_python_lambda_normalization_amendment_20260817.md"
ANALYSIS_PROTOCOL = ROOT / "paper" / "preregistration" / "e105_paired_analysis_specification_20260817.md"
SPARSE_THEORY = ROOT / "var" / "artifacts" / "e105_sparse_success_theory_tests.json"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("e105_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_grid_is_three_scales_five_static_domains_and_seventy_five_cells() -> None:
    module = _load_launcher()

    assert module.TARGET_STEPS == 3_072
    assert module.CHECKPOINT_INTERVAL == 192
    assert tuple(module.SCALE_SEEDS) == ("qwen05b", "falcon1b", "qwen3b")
    assert tuple(module.DOMAINS) == (
        "graph_coloring",
        "countdown",
        "python_factors",
        "mathir",
        "pantry_plan",
    )
    assert all("pointmaze" not in domain.lower() for domain in module.DOMAINS)
    assert module.E106_SUPPLEMENTAL_LEDGER_KEYS[-3:] == (
        "qwen3_clean_restart_all_partition_amendment",
        "e110_falcon_python_admission_horizon_replacement",
        "e110_two_hour_backfill_amendment",
    )
    assert len(module.E106_SUPPLEMENTAL_LEDGER_KEYS) == 17
    assert module.POST_CAMPAIGN_ANALYSIS_PROTOCOL.endswith(
        "e105_post_campaign_analysis_20260818.md"
    )
    assert module.POST_CAMPAIGN_ANALYSIS_SCRIPT.endswith(
        "e105_post_campaign_analysis.slurm"
    )

    root = module.repo_root()
    total_cells = 0
    for scale in module.SCALE_SEEDS:
        references = module.references(root, scale)
        comparators = module.comparator_index(
            root, scale, allow_legacy_python=True
        )
        assert len(references) == 25
        assert len(comparators) == 25
        total_cells += len(references)
    assert total_cells == 75


def test_e105_ledger_records_are_standard_campaign_static_rows() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert '"arm": ARM' in source
    assert '"arm",' in source


def test_post_campaign_analysis_depends_on_exactly_75_treatments_and_15_controls(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_launcher()
    ledger = tmp_path / module.REPAIRED_PYTHON_COMPARATOR_LEDGER
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps({"runs": [{"job_id": job_id} for job_id in range(100, 115)]}),
        encoding="utf-8",
    )
    script = tmp_path / module.POST_CAMPAIGN_ANALYSIS_SCRIPT
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\nset -euo pipefail\n", encoding="utf-8")
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="999\n", stderr="")
        assert command[:4] == ["scontrol", "show", "job", "-o"]
        record = (
            "JobId=999 JobState=PENDING Reason=Dependency RunTime=00:00:00 "
            "Partition=cs Account=allcs ReqTRES=cpu=2,mem=16G,node=1 "
            "SubmitLine=sbatch --partition=all "
            f"TimeLimit=02:00:00 Command={script}"
        )
        return SimpleNamespace(returncode=0, stdout=record, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module.submit_post_campaign_analysis(
        tmp_path,
        e105_job_ids=[str(job_id) for job_id in range(200, 275)],
    )
    assert payload["job_id"] == 999
    assert payload["dependency_cells"] == 90
    assert payload["cpu_only"] is True
    assert payload["dependency_job_ids"] == [
        *range(100, 115),
        *range(200, 275),
    ]
    submit = calls[0]
    dependency = next(token for token in submit if token.startswith("--dependency="))
    assert dependency == "--dependency=afterany:" + ":".join(
        str(job_id) for job_id in [*range(100, 115), *range(200, 275)]
    )
    assert not any("gpu" in token.lower() for token in submit)


def test_full_objective_is_exactly_the_e104_repaired_objective() -> None:
    module = _load_launcher()
    objective = module.e104.fixed_objective()

    assert objective["OAT_ZERO_VARIANT"] == "verified_replay_semantic_maxent_group_centered"
    assert objective["OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE"] == "0"
    assert objective["OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"] == "1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "verified_likelihood_per_rollout"
    )
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
    assert objective["OAT_ZERO_SEMANTIC_RMS_CONTROL"] == "0"


def test_python_pairs_use_only_repaired_replay_comparators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_launcher()
    root = module.repo_root()
    repaired_runs = []
    expected_jobs = {}
    for scale, seeds in module.SCALE_SEEDS.items():
        legacy = json.loads(
            (root / module.COMPARATOR_LEDGERS[scale]).read_text(encoding="utf-8")
        )
        python_replay = {
            int(run["seed"]): run
            for run in legacy["runs"]
            if run.get("arm") == "replay"
            and run.get("domain") == "python_factors"
        }
        for seed in seeds:
            source = python_replay[seed]
            job_id = int(source["job_id"]) + 10_000_000
            expected_jobs[(scale, seed)] = job_id
            repaired_runs.append(
                dict(
                    source,
                    scale=scale,
                    job_id=job_id,
                    run_stamp=f"e109_{scale}_python_repaired_replay_s{seed}",
                    run_dir=str(tmp_path / f"{scale}-s{seed}"),
                )
            )
    payload = {
        "schema": "e109_repaired_python_replay_comparators_jobs_v1",
        "released": True,
        "snapshot_sha256": module.e106.SNAPSHOT_SHA256,
        "parser_surface_version": module.e106.SURFACE_VERSION,
        "semantic_coefficient": 0.0,
        "pointmaze": "excluded",
        "runs": repaired_runs,
    }
    ledger = tmp_path / "e109.json"
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        module, "REPAIRED_PYTHON_COMPARATOR_LEDGER", str(ledger)
    )

    for scale, seeds in module.SCALE_SEEDS.items():
        index = module.comparator_index(root, scale)
        for seed in seeds:
            assert index[("python_factors", seed)]["job_id"] == expected_jobs[
                (scale, seed)
            ]
        legacy_graph = next(
            run
            for run in json.loads(
                (root / module.COMPARATOR_LEDGERS[scale]).read_text(
                    encoding="utf-8"
                )
            )["runs"]
            if run.get("arm") == "replay"
            and run.get("domain") == "graph_coloring"
            and int(run["seed"]) == seeds[0]
        )
        assert index[("graph_coloring", seeds[0])]["job_id"] == legacy_graph[
            "job_id"
        ]

    ledger.write_text(
        json.dumps(dict(payload, semantic_coefficient=0.1)),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="activates semantics"):
        module.comparator_index(root, "qwen05b")


def test_all_seventy_five_launch_commands_bind_the_frozen_three_scale_design() -> None:
    module = _load_launcher()
    root = module.repo_root()
    snapshot = (root / module.e106.SNAPSHOT).resolve()
    fixed_objective = module.e104.fixed_objective()
    cells = []

    for scale, seeds in module.SCALE_SEEDS.items():
        comparators = module.comparator_index(
            root, scale, allow_legacy_python=True
        )
        for run in module.references(root, scale):
            domain = str(run["domain"])
            seed = int(run["seed"])
            env, target = module.build_env(root, scale, run, snapshot)
            command = module.sbatch_command(root, scale, run, env)
            exported = next(
                token for token in command if token.startswith("--export=ALL,")
            )

            assert domain in module.DOMAINS
            assert seed in seeds
            assert env["OAT_ZERO_SEED"] == str(seed)
            assert env["OAT_ZERO_SOURCE_ROOT"] == str(snapshot / "src")
            assert env["OAT_ZERO_OPS_SNAPSHOT_ROOT"] == str(snapshot / "ops")
            assert env["OAT_ZERO_MAX_TRAIN"] == str(module.TRAIN_ROWS)
            assert env["OAT_ZERO_NUM_PROMPT_EPOCH"] == str(module.PASSES)
            assert env["OAT_ZERO_EVAL_PROMPT_INTERVAL"] == str(
                module.CHECKPOINT_INTERVAL
            )
            assert env["SAVE_PATH"] == str(target)
            assert env["RUN_STAMP"] == module.run_stamp(scale, domain, seed)
            assert all(env[key] == value for key, value in fixed_objective.items())
            for key in (
                "OAT_ZERO_SOURCE_ROOT",
                "OAT_ZERO_OPS_SNAPSHOT_ROOT",
                "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE",
                "OAT_ZERO_ONLINE_CANONICAL_REPLAY",
                "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE",
                "OAT_ZERO_SEED",
                "SAVE_PATH",
                "RUN_STAMP",
            ):
                assert f"{key}={env[key]}" in exported
            assert command[:3] == ["sbatch", "--parsable", "--hold"]
            assert "--nice=100" in command
            assert f"--job-name={module.job_name(scale, domain, seed)}" in command
            assert "pointmaze" not in " ".join(command).lower()
            comparator = comparators[(domain, seed)]
            assert comparator["arm"] == "replay"
            cells.append((scale, domain, seed, str(target), env["RUN_STAMP"]))

    assert len(cells) == 75
    assert len(set(cells)) == 75
    assert len({cell[3] for cell in cells}) == 75
    assert len({cell[4] for cell in cells}) == 75
    assert {cell[:3] for cell in cells} == {
        (scale, domain, seed)
        for scale, seeds in module.SCALE_SEEDS.items()
        for domain in module.DOMAINS
        for seed in seeds
    }


def test_protocol_freezes_outcome_blind_gate_and_excludes_pointmaze() -> None:
    protocol = PROTOCOL.read_text() + AMENDMENT.read_text()
    launcher = LAUNCHER.read_text()

    assert "Frozen with E104" in protocol
    assert "The 75 cells" in protocol
    assert "PointMaze is excluded" in protocol
    assert 'e104_audit.get("outcome_metrics_inspected") is not True' in launcher
    assert 'audit.get("post_update_outcome_metrics_inspected") is not False' in launcher
    assert 'audit.get("mechanism_gate_used_outcome_metrics") is not False' in launcher
    assert "combined gate lacks a live nonzero semantic update at every scale" in launcher
    assert "one aggregate correctness field at step 0" in launcher
    assert "E104_THEORY_CLARIFICATION" in launcher
    assert "theory_clarification_sha256" in launcher
    assert "E104_PLACEMENT_AMENDMENT" in launcher
    assert "placement_amendment_sha256" in launcher
    assert "qwen3_a6000_placement_amendment" in launcher
    assert "Qwen-3B placement amendment digest mismatch" in launcher
    assert "E104_UNIT_EVIDENCE" in launcher
    assert "unit-test evidence digest mismatch" in launcher
    assert 'audit.get("passed") is not True' in launcher
    assert 'audit.get("complete") is not True' in launcher
    assert "E106_AUDIT" in launcher
    assert "e106_python_lambda_normalization_combined_gate_v1" in launcher
    assert "e106.SNAPSHOT_SHA256" in launcher
    assert "E106_UNIT_EVIDENCE" in launcher
    assert "require_e106_supplemental_evidence" in launcher
    assert "superseded_e104_falcon_python_cancellation" in launcher
    assert "E106 ledger provenance digest mismatch" in launcher
    assert "E106 diagnosis violated outcome blinding" in launcher
    assert "parser_to_semantic_integration_evidence" in launcher
    assert "cross_scale_python_surface_evidence" in launcher
    assert "prior_falcon_python_bootstrap_evidence" in launcher
    assert "extended_frozen_estimator_evidence" in launcher
    assert "python_smoke_time_limit_amendment" in launcher
    assert "falcon_python_a6000_pool_amendment" in launcher
    assert "falcon_one_hour_backfill_amendment" in launcher
    assert "falcon_same_gpu_pool_widening_amendment" in launcher
    assert "falcon_a6000_all_partition_pool_amendment" in launcher
    assert "frozen_policy_gradient_direction_evidence" in launcher
    assert "semantic_replay_identity_contract_evidence" in launcher
    assert "repair_amendment_sha256" in launcher
    assert "PYTHON_COMPARATOR_AMENDMENT" in launcher
    assert "python_comparator_amendment_sha256" in launcher
    assert "REPAIRED_PYTHON_COMPARATOR_LEDGER" in launcher
    assert "repaired_python_comparator_ledger_sha256" in launcher
    assert "historical_comparator_ledgers" in launcher
    assert "python_comparator_repaired" in launcher
    assert "semantic_replay_identity_contract_evidence" in launcher
    assert "analysis_protocol_sha256" in launcher
    assert "analysis_builder_sha256" in launcher
    assert "analysis_plotter_sha256" in launcher
    assert "e105_group_centered_semantic_auc_effects" in (
        ROOT
        / "ops/exp_scaling/plot_e105_group_centered_semantic_endpoint_effects.py"
    ).read_text()
    assert "SPARSE_THEORY_EVIDENCE" in launcher
    assert "sparse-success theory test did not pass" in launcher
    assert "PointMaze" in ANALYSIS_PROTOCOL.read_text()
    assert "separate registered trajectory-AUC forest" in ANALYSIS_PROTOCOL.read_text()
    assert "e106_python_lambda_b853595e3b158046" in protocol


def test_sparse_success_theory_evidence_is_snapshot_bound_and_blinded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_launcher()
    evidence = json.loads(SPARSE_THEORY.read_text(encoding="utf-8"))
    snapshot = (module.repo_root() / module.e106.SNAPSHOT).resolve()

    module.require_sparse_success_theory_evidence(module.repo_root(), snapshot)
    assert evidence["passed"] is True
    assert evidence["returncode"] == 0
    assert evidence["snapshot_sha256"] == module.e106.SNAPSHOT_SHA256
    assert evidence["group_size"] == 16
    assert evidence["post_e104_or_e106_update_outcomes_inspected"] is False
    assert evidence["pointmaze"] == "excluded"
    assert "2 passed" in evidence["stdout"]

    tampered = dict(evidence, passed=False)
    tampered_path = tmp_path / "tampered_sparse_theory.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(module, "SPARSE_THEORY_EVIDENCE", str(tampered_path))
    with pytest.raises(SystemExit, match="sparse-success theory test did not pass"):
        module.require_sparse_success_theory_evidence(module.repo_root(), snapshot)


def test_release_gate_freshly_revalidates_e106_supplemental_evidence() -> None:
    module = _load_launcher()
    root = module.repo_root()
    audit = json.loads((root / module.E106_AUDIT).read_text(encoding="utf-8"))
    ledger = json.loads((root / module.E106_LEDGER).read_text(encoding="utf-8"))
    snapshot = Path(ledger["snapshot_root"]).resolve()

    module.require_e106_supplemental_evidence(root, audit, snapshot)

    stale = dict(audit)
    stale["parser_to_semantic_integration_evidence"] = dict(
        audit["parser_to_semantic_integration_evidence"], sha256="stale"
    )
    with pytest.raises(SystemExit, match="parser_to_semantic_integration_evidence digest mismatch"):
        module.require_e106_supplemental_evidence(root, stale, snapshot)
