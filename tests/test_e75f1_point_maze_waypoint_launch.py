from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e75f1_point_maze_waypoint_05b_12pass.py"
QUALIFIER = ROOT / "ops/qualify_e75f1_point_maze_waypoint_dev.py"
PROTOCOL = ROOT / "paper/preregistration/e75f1_point_maze_waypoint_05b_12pass_20260804.md"
MONITOR = ROOT / "ops/exp_scaling/status_e72.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation(*, pass_maps: int = 48, multimode_maps: int = 32, mean8: float = 0.4):
    rows = []
    for index in range(64):
        family = ("barrier_3mode", "barrier_4mode", "barrier_5mode")[index % 3]
        passed = index < pass_maps
        rows.append({
            "map_id": f"map-{index}",
            "family": family,
            "pass8": float(passed),
            "distinct8": 2.0 if index < multimode_maps else float(passed),
        })
    return {
        "schema": "point-maze-waypoint-pilot-evaluation-v1",
        "split": "dev",
        "evaluation_prompt_count": 64,
        "evaluation_trajectory_count": 512,
        "mean8": mean8,
        "per_map": rows,
    }


def receipt():
    return {
        "status": "complete",
        "evaluation_only": True,
        "optimizer_updates": 0,
        "evaluation_split": "dev",
        "evaluation_prompt_count": 64,
        "evaluation_coordinates": 1,
        "data_identity_sha256": "data",
        "metrics_sha256": "metrics",
    }


def test_protocol_is_paper_shaped_and_reports_the_null_pilot():
    text = PROTOCOL.read_text()
    assert "one-pass, one-seed" in text
    assert "-0.03125 distinct routes@8" in text
    assert "384 train maps" in text
    assert "128 evaluation maps" in text
    assert "Paired seeds: 43, 44, 45, 46, and 47" in text
    assert "exactly 4,608 optimizer" in text
    assert "No efficacy threshold" in text


def test_fullscale_jobs_use_frozen_r3_warmstart_and_exact_horizon():
    prepare = (ROOT / "ops/slurm/e75f1_point_maze_waypoint_prepare.slurm").read_text()
    dev = (ROOT / "ops/slurm/e75f1_point_maze_waypoint_dev.slurm").read_text()
    train = (ROOT / "ops/slurm/e75f1_point_maze_waypoint_train.slurm").read_text()
    assert "--train-count 384 --dev-count 64 --eval-count 128" in prepare
    assert prepare.count("--exclude-identity") == 4
    assert "point_maze_waypoint_warmstart_e75r3" in dev
    assert "--evaluation-prompts 64 --evaluation-split dev" in dev
    assert "point_maze_waypoint_warmstart_e75r3" in train
    assert "--passes 12 --evaluation-interval 96" in train
    assert "--evaluation-prompts 128 --evaluation-split eval" in train
    assert 'SEED="${OAT_ZERO_SEED:?seed is required}"' in train


def test_dev_gate_is_proportional_and_fail_closed():
    module = load_module(QUALIFIER, "e75f1_qualifier")
    passed = module.qualify(
        receipt=receipt(),
        evaluation=evaluation(),
        data_identity_sha256="data",
        metrics_sha256="metrics",
    )
    assert passed["status"] == "pass"
    assert passed["eligible_for_full_cohort"] is True
    failed = module.qualify(
        receipt=receipt(),
        evaluation=evaluation(pass_maps=47, multimode_maps=31),
        data_identity_sha256="data",
        metrics_sha256="metrics",
    )
    assert failed["status"] == "fail"
    assert failed["eligible_for_full_cohort"] is False


def test_launcher_freezes_two_by_five_cells_and_terminal_audit():
    module = load_module(LAUNCHER, "e75f1_launcher")
    assert module.ARMS == ("grpo", "verified_first_global_replay_canonical")
    assert module.SEEDS == (43, 44, 45, 46, 47)
    source = LAUNCHER.read_text()
    assert 'time_limit="3-00:00:00"' in source
    assert '"all_online_to_audit": "afterok"' in source
    assert '"optimizer_updates_per_cell": 4608' in source
    assert "e75f1_point_maze_waypoint_05b_12pass_audit.json" in source


def test_campaign_monitor_wires_pointmaze_and_ant_records_into_snapshot():
    monitor = MONITOR.read_text()
    assert '"waypoint_f1": waypoint_fullscale_progress(root)' in monitor
    assert '"ant_maze": ant_maze_progress(root)' in monitor
    assert "AntMaze 0.5B paper-scale record" in monitor
    assert "no LM cohort launched" in monitor
    assert "v19 repair: controller=" in monitor
    assert "v19 route admission:" in monitor
    assert "v19 0.5B viability:" in monitor


def test_ant_monitor_collects_live_v19_dependency_chain(tmp_path, monkeypatch):
    module = load_module(MONITOR, "e72_status_ant_v19")
    artifacts = tmp_path / "var/artifacts"
    logs = artifacts / "logs"
    logs.mkdir(parents=True)
    (artifacts / "ant_continuing_waypoint_controller_v19_submission.json").write_text(
        '{"job_id": 30259111, "released": true}\n'
    )
    (artifacts / "ant_continuing_waypoint_controller_v19_identity.json").write_text(
        '{"job_id": 30259111, "timesteps": 6000000}\n'
    )
    (logs / "ant-cont-v19-30259111.out").write_text(
        "|    total_timesteps      | 1005568      |\n"
    )
    (
        artifacts
        / "ant_maze_v15_controller_v19_dependent_launcher_identity.json"
    ).write_text('{"dependent_launcher_job_id": 30259182}\n')
    (
        artifacts / "ant_maze_v15_v19_viability_preparer_identity.json"
    ).write_text('{"preparer_job_id": 30259233}\n')
    states = {
        30259111: "RUNNING",
        30259182: "PENDING",
        30259233: "PENDING",
    }
    monkeypatch.setattr(module, "scheduler_job_state", states.__getitem__)

    progress = module.ant_v19_progress(tmp_path)

    assert progress["controller"] == {
        "job_id": 30259111,
        "job_state": "RUNNING",
        "timesteps": 1_005_568,
        "expected_timesteps": 6_000_000,
        "evaluation": {},
    }
    assert progress["route"]["launcher_job_id"] == 30259182
    assert progress["route"]["job_state"] == "PENDING"
    assert progress["viability"]["preparer_job_id"] == 30259233
    assert progress["viability"]["job_state"] == "PENDING"
    assert progress["viability"]["lm_job_launched"] is False
    assert progress["viability"]["lm_sampled"] is False


def test_ant_monitor_renders_failed_controller_as_clean_gate_stop():
    module = load_module(MONITOR, "e72_status_ant_v19_gate_stop")
    current = {
        "unix": 1.0,
        "cohorts": {},
        "frontier": {
            "done": 0,
            "total": 1,
            "per_stage": {},
        },
        "falcon": {},
        "waypoint_f1": {},
        "e76": {},
        "queue": {},
        "ant_maze": {
            "launched": True,
            "complete": 10,
            "updates": 480,
            "expected_updates": 480,
            "audit": {
                "status": "pass",
                "decision": "ant_maze_terminal_eligible",
            },
            "v19": {
                "launched": True,
                "controller": {
                    "job_id": 30259111,
                    "evaluation": {
                        "status": "fail",
                        "decision": "ant_waypoint_v19_ineligible",
                        "evaluation": {
                            "summary": {
                                "success_rate": 0.3229166666666667,
                                "minimum_heading_success_rate": 0.0,
                            }
                        },
                    },
                },
                "route": {
                    "launcher_job_id": 30259182,
                    "job_state": "FAILED",
                    "audit": {},
                },
                "viability": {
                    "preparer_job_id": 30259233,
                    "job_state": "FAILED",
                    "qualification": {},
                    "lm_job_launched": False,
                },
            },
        },
    }

    rendered = module.render(current, None, {})

    assert (
        "v19 route admission: stopped at controller gate "
        "(not launched; launcher=30259182); no route sampled"
    ) in rendered
    assert (
        "v19 0.5B viability: stopped at controller gate "
        "(not launched); LM samples=none"
    ) in rendered
    assert "waiting on controller (FAILED" not in rendered
    assert "waiting on route (FAILED" not in rendered
