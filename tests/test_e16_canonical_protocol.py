from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from ops.exp_scaling import check_e16_canonical_smoke as smoke
from ops.exp_scaling import e16_canonical_plan as plan
from ops.exp_scaling import verify_e16_antecedent as antecedent
from ops.exp_scaling import verify_e16_canonical_datasets as datasets_gate


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_e16_plan_is_e15_derived_and_exactly_budgeted():
    smoke_plan = plan.campaign_plan("smoke")
    full = plan.campaign_plan("full")
    assert smoke_plan["seeds"] == [9006]
    assert full["seeds"] == [43, 44, 45]
    assert tuple(full["arms"]) == (
        "maxent",
        "maxent_control",
        "maxent_dual",
    )
    assert full["arms"]["maxent"]["alpha"] == 0.10
    assert full["arms"]["maxent_control"] == {
        "alpha": 0.075,
        "controller": "proportional",
        "ema_decay": 0.9,
        "gain": 4.0,
        "max_alpha": 0.10,
        "target_ratio_interface": 1.0,
        "warmup_steps_interface": 1,
    }
    assert full["arms"]["maxent_dual"]["min_alpha"] == 0.05
    assert full["arms"]["maxent_dual"]["max_alpha"] == 0.10
    assert full["arms"]["maxent_dual"]["alpha_lr"] == 0.005
    assert full["common"]["prompt_epochs"] == 5
    assert full["common"]["maxent_length_target"] == 0
    graph = full["tasks"]["graph_coloring"]
    countdown = full["tasks"]["countdown"]
    assert graph["prefix"] == "gce16_canonical_maxent_05b_v2"
    assert countdown["prefix"] == "cde16_canonical_maxent_05b_v2"
    assert graph["target_optimizer_updates"] == 960
    assert graph["trajectory_query_budget"] == 15344
    assert countdown["target_optimizer_updates"] == 1920
    assert countdown["trajectory_query_budget"] == 30704
    assert graph["target_entropy"] == 2.7974740052946436
    assert countdown["target_entropy"] == 3.9741470618167156
    assert smoke_plan["tasks"]["graph_coloring"]["target_optimizer_updates"] == 32
    assert smoke_plan["tasks"]["countdown"]["trajectory_query_budget"] == 496


def test_countdown_codec_static_gate_rejects_label_access(monkeypatch):
    def label_leaking_decoder(code, reference):
        return reference["target"] + code

    monkeypatch.setattr(
        datasets_gate, "decode_countdown_action_code", label_leaking_decoder
    )
    with pytest.raises(datasets_gate.DatasetCodecError, match="may read only"):
        datasets_gate.validate_countdown_codec_static()


def test_e16_frozen_datasets_pass_exhaustive_codec_gate():
    identity = datasets_gate.verify_e16_canonical_datasets(
        graph_data_root=ROOT / "var/data/exact_answer_mode_probe",
        countdown_data_root=ROOT / "var/data/exact_countdown_easy3_probe",
    )
    assert identity["codec"]["code_count"] == 108
    assert identity["codec"]["target_or_solution_access"] is False
    assert (
        identity["countdown"]["combined_content_hash"]
        == datasets_gate.EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH
    )
    assert identity["countdown"]["train_rows"] == 384
    assert identity["countdown"]["eval_rows"] == 128


def test_e16_antecedent_binds_selected_e15_m10(tmp_path):
    outcome = ROOT / "var/artifacts/e15_canonical_dose_decision_20260718_2229.json"
    summary = antecedent.verify_e16_antecedent(outcome)
    assert summary["selected_arm"] == "M10"
    assert summary["fixed_alpha"] == 0.10
    changed = tmp_path / "changed.json"
    payload = json.loads(outcome.read_text())
    payload["selected_arm"] = "M075"
    changed.write_text(json.dumps(payload))
    with pytest.raises(antecedent.AntecedentError, match="hash drifted"):
        antecedent.verify_e16_antecedent(changed)


def _controller_rows(task: str, arm: str) -> list[dict[str, float]]:
    target = smoke.TARGETS[task]
    if arm == "maxent_control":
        controller = smoke.MaxEntProportionalController(
            base_alpha=0.075,
            max_alpha=0.10,
            target_ratio=1.0,
            warmup_steps=1,
            ema_decay=0.9,
            gain=4.0,
            configured_target_entropy=target,
            entropy_units="canonical_action_nats_exact_v1",
            observation_metric_key="canonical_exact_sequence_entropy",
        )
        prefix = "maxent_control"
    else:
        controller = smoke.MaxEntDualController(
            base_alpha=0.075,
            min_alpha=0.05,
            max_alpha=0.10,
            target_ratio=1.0,
            warmup_steps=1,
            alpha_lr=0.005,
            configured_target_entropy=target,
            entropy_units="canonical_action_nats_exact_v1",
            observation_metric_key="canonical_exact_sequence_entropy",
        )
        prefix = "maxent_dual"
    rows = []
    for step in range(1, 33):
        used = float(controller.current_alpha)
        entropy = target - 0.01 * math.sin(step)
        diagnostics = controller.observe(entropy)
        rows.append(
            {
                "train/maxent_alpha_used": used,
                "train/canonical_exact_sequence_entropy": entropy,
                f"train/{prefix}_next_alpha": float(controller.current_alpha),
                f"train/{prefix}_target_entropy": target,
                f"train/{prefix}_observations": diagnostics[
                    f"{prefix}_observations"
                ],
            }
        )
    return rows


def _framed_smoke_metrics() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = [
        {
            "trainer/step": 0,
            "trainer/global_step": 0,
            "trainer/policy_sgd_step": 0,
        }
    ]
    rows.extend(
        {
            "trainer/step": step,
            "trainer/global_step": step,
            "trainer/policy_sgd_step": step,
            "train/learning_round": step,
        }
        for step in range(1, 33)
    )
    rows.append(
        {
            "trainer/step": 33,
            "trainer/global_step": 32,
            "trainer/policy_sgd_step": 32,
            "train/learning_round": 32,
        }
    )
    return rows


def test_smoke_metric_stream_is_ordered_unique_and_restart_invalid(tmp_path):
    metrics = tmp_path / "train_metrics.jsonl"
    rows = _framed_smoke_metrics()
    metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    observed = smoke._metric_rows(metrics)
    assert [row["train/learning_round"] for row in observed] == list(range(1, 33))

    duplicated = copy.deepcopy(rows)
    duplicated.insert(10, copy.deepcopy(duplicated[10]))
    metrics.write_text("\n".join(json.dumps(row) for row in duplicated) + "\n")
    with pytest.raises(smoke.SmokeGateError, match="exactly 32 ordered update"):
        smoke._metric_rows(metrics)

    reordered = copy.deepcopy(rows)
    reordered[8], reordered[9] = reordered[9], reordered[8]
    metrics.write_text("\n".join(json.dumps(row) for row in reordered) + "\n")
    with pytest.raises(smoke.SmokeGateError, match="exact ordered sequence"):
        smoke._metric_rows(metrics)


def test_smoke_run_rejects_multiple_debug_metric_streams(tmp_path):
    stamp = f"{smoke.PREFIXES['graph_coloring']}_maxent_s9006"
    run_dir = tmp_path / f"run_{stamp}"
    first = run_dir / "debug_first" / "train_metrics.jsonl"
    second = run_dir / "debug_restart" / "train_metrics.jsonl"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}\n")
    second.write_text("{}\n")
    with pytest.raises(smoke.SmokeGateError, match="exactly one debug metrics"):
        smoke._validate_run_and_metrics(
            run_dir=run_dir, run_stamp=stamp, metrics_path=first
        )


@pytest.mark.parametrize("arm", ["maxent_control", "maxent_dual"])
def test_smoke_gate_replays_exact_controller_handoff(arm):
    rows = _controller_rows("graph_coloring", arm)
    smoke._validate_controller(rows, task="graph_coloring", arm=arm)
    broken = copy.deepcopy(rows)
    broken[8][f"train/{arm}_next_alpha"] += 0.002
    with pytest.raises(smoke.SmokeGateError, match="next alpha"):
        smoke._validate_controller(broken, task="graph_coloring", arm=arm)


def test_launcher_is_fail_closed_and_uses_one_shared_snapshot():
    launcher = (
        ROOT / "ops/exp_scaling/launch_e16_canonical_maxent_replication.sh"
    ).read_text()
    assert "{smoke-config|smoke|full-config|full}" in launcher
    assert "E16 protocol is not FROZEN" in launcher
    assert 'cell["campaign_identity"]["approval_verifier"]["path"]' in launcher
    assert '$PYTHON_BIN "$approval_verifier"' in launcher
    assert "OAT_ZERO_E16_SMOKE_APPROVAL_SHA256" in launcher
    assert "gce16_canonical_maxent_joint_smoke_v1" not in launcher
    assert "OAT_ZERO_MAXENT_FIXED_ALPHA=0.10" in launcher
    assert "OAT_ZERO_MAXENT_CONTROL_BASE_ALPHA=0.075" in launcher
    assert "OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.075" in launcher
    assert "OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=\"$target\"" in launcher
    assert "OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=\"$target\"" in launcher
    assert "OAT_ZERO_MAXENT_LENGTH_TARGET=0" in launcher
    first_preflight = launcher.index("export E16_CONFIG_PREFLIGHT=1")
    snapshot = launcher.index('snapshot_parent="$ROOT_DIR/var/artifacts/source_snapshots')
    first_identity = launcher.index('write_identity "$task"')
    first_submission = launcher.index("export E16_CONFIG_PREFLIGHT=0")
    assert first_preflight < snapshot < first_identity < first_submission
    assert 'export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$snapshot_root"' in launcher


def test_shared_submitter_forwards_generic_task_and_per_arm_alphas():
    submitter = (ROOT / "ops/submit_countdown_comparative.sh").read_text()
    trainer = (ROOT / "ops/train.sh").read_text()
    assert "OAT_ZERO_CANONICAL_ACTION_TASK" in submitter
    assert "MAXENT_FIXED_ALPHA" in submitter
    assert "MAXENT_CONTROL_BASE_ALPHA" in submitter
    assert "MAXENT_DUAL_BASE_ALPHA" in submitter
    assert '--canonical-action-task "$CANONICAL_ACTION_TASK"' in trainer


def test_approval_builder_requires_complete_grid_and_replays_results(
    tmp_path, monkeypatch
):
    identity = {
        "dataset_identity_sha256": "d" * 64,
        "e15_outcome_sha256": "e" * 64,
        "execution_surface_hash": "a" * 64,
        "full_config_digest": "b" * 64,
        "protocol_sha256": "c" * 64,
        "runtime_identity_sha256": "f" * 64,
        "smoke_config_digest": "c" * 64,
        "source_hash": "1" * 64,
    }
    paths = []
    replayed_by_path = {}
    shared_bindings = {}
    for index, (task, arm) in enumerate(
        (task, arm) for task in smoke.TASKS for arm in smoke.ARMS
    ):
        run_stamp = f"{smoke.PREFIXES[task]}_{arm}_s9006"
        run_dir = tmp_path / f"run_{run_stamp}"
        checkpoint = run_dir / f"debug_{index}" / "saved_models" / "step_00032"
        checkpoint.mkdir(parents=True)
        checkpoint_config = checkpoint / "config.json"
        checkpoint_config.write_text(f'{{"cell": {index}}}')
        bindings = {}
        for label in ("endpoint_audit", "identity", "manifest", "metrics", "stdout"):
            if label in {"identity", "manifest"}:
                shared_key = (task, label)
                if shared_key not in shared_bindings:
                    evidence_file = tmp_path / f"{task}-{label}.json"
                    evidence_file.write_text(
                        json.dumps({"task": task, "label": label})
                    )
                    shared_bindings[shared_key] = {
                        "path": str(evidence_file),
                        "sha256": _sha256(evidence_file),
                    }
                bindings[label] = shared_bindings[shared_key]
            else:
                evidence_file = tmp_path / f"{label}-{index}.json"
                evidence_file.write_text(
                    json.dumps({"cell": index, "label": label})
                )
                bindings[label] = {
                    "path": str(evidence_file),
                    "sha256": _sha256(evidence_file),
                }
        bindings["checkpoint_config"] = {
            "path": str(checkpoint_config),
            "sha256": _sha256(checkpoint_config),
        }
        campaign_identity = {
            "path": bindings["identity"]["path"],
            "sha256": bindings["identity"]["sha256"],
        }
        result = {
            "arm": arm,
            "campaign_identity": campaign_identity,
            "checkpoint": {
                "path": str(checkpoint),
                "weights_manifest_sha256": f"{index + 1:064x}",
            },
            "checks": {key: True for key in smoke.REQUIRED_CHECKS},
            "endpoint_audit": bindings["endpoint_audit"],
            "evidence": bindings,
            "identity": identity,
            "job_id": str(index + 1),
            "protocol": "E16",
            "run_dir": str(run_dir),
            "run_stamp": run_stamp,
            "schema": "e16_canonical_smoke_cell_result_v1",
            "seed": 9006,
            "slurm": {"exit_code": "0:0", "state": "COMPLETED"},
            "stage": "smoke",
            "status": "pass",
            "summary": {},
            "task": task,
        }
        path = tmp_path / f"result-{index}.json"
        path.write_text(json.dumps(result, sort_keys=True))
        paths.append(path)
        replayed_by_path[(task, arm)] = result

    monkeypatch.setattr(
        smoke,
        "validate_cell",
        lambda **kwargs: replayed_by_path[(kwargs["task"], kwargs["arm"])],
    )
    approval = smoke.build_approval(paths)
    assert approval["all_six_passed"] is True
    assert len(approval["cells"]) == 6
    with pytest.raises(smoke.SmokeGateError, match="exactly six"):
        smoke.build_approval(paths[:-1])
    reused = json.loads(paths[-1].read_text())
    reused["evidence"]["metrics"] = json.loads(paths[0].read_text())["evidence"][
        "metrics"
    ]
    paths[-1].write_text(json.dumps(reused, sort_keys=True))
    with pytest.raises(smoke.SmokeGateError, match="metrics .*reused"):
        smoke.build_approval(paths)
