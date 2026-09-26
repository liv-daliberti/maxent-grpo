from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e112r1_verified_support_discovery_full_three_scale.py"
PROTOCOL = ROOT / "paper/preregistration/e112_verified_support_discovery_full_three_scale_20260818.md"
REPAIR_PROTOCOL = ROOT / "paper/preregistration/e112r1_sampler_contract_repair_and_e112_retirement_20260819.md"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e112_is_exactly_three_scales_five_domains_five_seeds():
    launch = _load("e112_grid")
    assert launch.TRAIN_ROWS == 384
    assert launch.PASSES == 8
    assert launch.TARGET_STEPS == 3072
    assert launch.CHECKPOINT_INTERVAL == 192
    assert launch.QWEN3_A6000_CHECKPOINT_INTERVAL == 64
    cells = []
    for scale in launch.SCALE_SEEDS:
        for run in launch.references(ROOT, scale):
            cells.append((scale, run["domain"], run["seed"]))
    assert len(cells) == 75
    assert len(set(cells)) == 75
    assert "point" not in " ".join(str(cell) for cell in cells).lower()


def test_e112_uses_exact_e111_treatment_at_full_horizon(tmp_path):
    launch = _load("e112_env")
    _, a6000_cells = launch.e105.require_qwen3_paired_placement(ROOT)
    for scale in launch.SCALE_SEEDS:
        for run in launch.references(ROOT, scale):
            env, target = launch.build_env(
                ROOT,
                scale,
                run,
                tmp_path,
                a6000_cells,
            )
            assert env["OAT_ZERO_VARIANT"] == launch.e111.VARIANT
            assert env["OAT_ZERO_MAX_TRAIN"] == "384"
            assert env["OAT_ZERO_NUM_PROMPT_EPOCH"] == "8"
            assert env["OAT_ZERO_EVAL_PROMPT_INTERVAL"] == "192"
            assert env["OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE"] == "1"
            assert env["OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK"] == "1"
            assert env["OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"] == "0"
            assert env["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "1"
            assert env["OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS"] == "0"
            assert env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
            canonical = run["domain"] == "pantry_plan"
            assert env["OAT_ZERO_REPLICATED_FREEFORM_SAMPLING"] == (
                "0" if canonical else "1"
            )
            assert env["OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC"] == (
                "0" if canonical else "1"
            )
            assert env["OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING"] == (
                "1" if canonical else "0"
            )
            assert "e112r1" in env["RUN_STAMP"]
            assert "e112r1" in str(target)


def test_e112_preserves_frozen_qwen3_paired_hardware(tmp_path):
    launch = _load("e112_placement")
    _, a6000_cells = launch.e105.require_qwen3_paired_placement(ROOT)
    runs = launch.references(ROOT, "qwen3b")
    for run in runs:
        env, _ = launch.build_env(
            ROOT,
            "qwen3b",
            run,
            tmp_path,
            a6000_cells,
        )
        command = launch.sbatch_command(ROOT, "qwen3b", run, env, a6000_cells)
        cell = (str(run["domain"]), int(run["seed"]))
        assert any(token.startswith("--job-name=e112r1-q3-") for token in command)
        if cell in a6000_cells:
            assert "--partition=lowprio" in command
            assert f"--nodelist={launch.QWEN3_A6000_NODE_LIST}" in command
            assert "--gres=gpu:a6000:1" in command
            assert env["OAT_ZERO_SAVE_STEPS"] == "64"
            assert env["OAT_ZERO_SAVE_FROM"] == "64"
            assert env["OAT_ZERO_RESUME_STEPS"] == "64"
        else:
            assert "--partition=mltheory" in command
            assert "--nodelist=node302" in command
            assert "--gres=gpu:a100:1" in command
            assert env["OAT_ZERO_SAVE_STEPS"] == "192"
            assert env["OAT_ZERO_SAVE_FROM"] == "192"
            assert env["OAT_ZERO_RESUME_STEPS"] == "192"


def test_e112r1_held_audit_rejects_missing_sampler_contract(monkeypatch, tmp_path):
    launch = _load("e112r1_held_sampler_audit")
    _, a6000_cells = launch.e105.require_qwen3_paired_placement(ROOT)
    run = next(
        row
        for row in launch.references(ROOT, "qwen05b")
        if row["domain"] == "countdown"
    )
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="JobState=PENDING Reason=JobHeldUser",
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="REPLICATED_FREEFORM_SAMPLING=1"):
        launch.held_job_audit(
            "123", "qwen05b", run, tmp_path, a6000_cells
        )


def test_e112_release_fails_closed_when_e111_audit_is_not_terminal(monkeypatch):
    launch = _load("e112_gate")
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="pending"),
    )
    with pytest.raises(SystemExit, match="did not pass"):
        launch.require_terminal_e111_gate(ROOT)


def test_e112_release_interlock_rejects_active_e105(tmp_path, monkeypatch):
    launch = _load("e112_e105_interlock")
    ledger = tmp_path / launch.E105_LEDGER
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        __import__("json").dumps(
            {"runs": [{"job_id": value} for value in range(100, 175)]}
        )
        + "\n",
        encoding="utf-8",
    )
    note = tmp_path / launch.E105_SUPERSESSION
    note.parent.mkdir(parents=True)
    note.write_text("frozen\n", encoding="utf-8")
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="100\n", stderr=""
        ),
    )
    with pytest.raises(SystemExit, match="remains active"):
        launch.require_e105_retired(tmp_path)


def test_e112_protocol_freezes_pairing_blinding_and_no_pointmaze():
    text = PROTOCOL.read_text(encoding="utf-8")
    repair = REPAIR_PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "all 60 non-Pantry cells omitted",
        "replicated free-form sampling and local actor weight",
        "fresh run directories, run stamps, job names",
        "No efficacy endpoint was inspected",
    ):
        assert required in repair
    for required in (
        "3 model scales x 5 natural domains x 5 seeds = 75 treatment cells",
        "3,072 optimizer updates",
        "parser-repaired E109 comparator",
        "fail closed",
        "No E112 or E109 task outcome is inspected",
        "E105 artifacts remain audit-only",
        "PointMaze is excluded",
        "the exact 75 E105 jobs are no longer active",
        "64 optimizer updates",
        "checkpoint-deserialization recovery evidence passes",
        "auto-resume selects only complete model+optimizer ZIP checkpoints",
        "post-freeze E111 training-reward exposure",
    ):
        assert required in text
