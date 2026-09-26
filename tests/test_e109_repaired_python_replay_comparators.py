from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e109_repaired_python_replay_comparators.py"
PROTOCOL = ROOT / (
    "paper/preregistration/e109_repaired_python_replay_comparators_20260817.md"
)


def _load():
    spec = importlib.util.spec_from_file_location("e109_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e109_is_exactly_fifteen_python_replay_cells() -> None:
    module = _load()
    root = module.repo_root()
    snapshot = (root / module.e106.SNAPSHOT).resolve()
    cells = []
    for scale, seeds in module.SCALE_SEEDS.items():
        runs = module.references(root, scale)
        assert len(runs) == 5
        for run in runs:
            seed = int(run["seed"])
            env, target = module.build_env(root, scale, run, snapshot)
            cells.append((scale, seed, target))
            assert seed in seeds
            assert str(run["domain"]) == "python_factors"
            assert env["OAT_ZERO_SOURCE_ROOT"] == str(snapshot / "src")
            assert env["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
            assert env[
                "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"
            ] == "0"
            assert env["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"
            assert env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
            assert env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
    assert len(cells) == 15
    assert len(set(cells)) == 15


def test_e109_commands_preserve_scale_resources_and_pair_qwen3_a6000() -> None:
    module = _load()
    root = module.repo_root()
    snapshot = (root / module.e106.SNAPSHOT).resolve()
    for scale in module.SCALE_SEEDS:
        for run in module.references(root, scale):
            seed = int(run["seed"])
            env, _ = module.build_env(root, scale, run, snapshot)
            command = module.sbatch_command(
                root,
                scale,
                run,
                env,
                {73, 74},
            )
            joined = " ".join(command)
            assert command[:3] == ["sbatch", "--parsable", "--hold"]
            assert f"--job-name={module.job_name(scale, seed)}" in command
            assert "pointmaze" not in joined.lower()
            if scale == "qwen3b" and seed in {73, 74}:
                assert "--partition=lowprio" in command
                assert f"--nodelist={module.QWEN3_A6000_NODE_LIST}" in command
                assert "--gres=gpu:a6000:1" in command
            elif scale == "qwen3b":
                assert "--partition=mltheory" in command
                assert "--nodelist=node302" in command
                assert "--gres=gpu:a100:1" in command


def test_e109_reads_prospective_python_pair_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    monkeypatch.setattr(
        module.e105,
        "require_qwen3_paired_placement",
        lambda _root: (
            {},
            {
                ("python_factors", 73),
                ("python_factors", 74),
                ("mathir", 71),
            },
        ),
    )

    assert module.qwen3_a6000_seeds(tmp_path) == {73, 74}


def test_e109_held_audit_enforces_prospective_qwen3_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    snapshot = tmp_path / "snapshot"
    record = " ".join(
        (
            "JobState=PENDING",
            "Reason=JobHeldUser",
            "JobName=e109-q3-python-s73",
            "OAT_ZERO_SEED=73",
            f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
            f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
            "OAT_ZERO_MAX_TRAIN=384",
            "OAT_ZERO_NUM_PROMPT_EPOCH=8",
            "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
            "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=0",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
            "Partition=lowprio",
            f"ReqNodeList={module.QWEN3_A6000_NODE_LIST}",
            "TresPerNode=gres/gpu:a6000:1",
        )
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=record, stderr=""
        ),
    )

    module.held_job_audit("123", "qwen3b", {"seed": 73}, snapshot, {73})
    with pytest.raises(RuntimeError, match="lacks"):
        module.held_job_audit("123", "qwen3b", {"seed": 73}, snapshot, set())


def test_generic_status_reader_infers_e109_single_arm_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_path = ROOT / "ops/exp_scaling/status_e78.py"
    spec = importlib.util.spec_from_file_location("e109_status_reader", status_path)
    assert spec is not None and spec.loader is not None
    status = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(status)
    ledger = tmp_path / "e109.json"
    ledger.write_text(
        json.dumps(
            {
                "target_steps": 3072,
                "train_rows": 384,
                "checkpoint_interval_steps": 192,
                "passes": 8,
                "runs": [
                    {
                        "arm": "replay",
                        "domain": "python_factors",
                        "job_id": 1,
                        "seed": 43,
                        "run_dir": str(tmp_path / "run"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(status, "scheduler_states", lambda _ids: {1: "PENDING"})
    monkeypatch.setattr(status, "run_step", lambda _path: 0)
    monkeypatch.setattr(status, "checkpoint_step", lambda _path: 0)
    monkeypatch.setattr(status, "is_complete", lambda *_args: False)

    snapshot = status.load_snapshot(ledger)

    assert snapshot["arms"] == ["replay"]
    assert snapshot["domains"] == ["python_factors"]


def test_e109_protocol_removes_parser_confound_before_submission() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "confound" in text
    assert "exactly 15 cells" in text
    assert "PointMaze" in text
    assert "before E105 or E109 submission" in text
    assert "supersedes" in text
    assert "post_e104_or_e106_update_outcomes_inspected" in launcher
    assert "e105.check_gate(root)" in launcher
    assert "e105.require_qwen3_paired_placement(root)" in launcher
    assert "SEMANTIC_SHANNON_COEF\": \"0.0" in launcher
    assert "prospective Python placement drifted" in launcher
    assert "qwen3_paired_placement_artifact_sha256" in launcher
    assert "TresPerNode=gres/gpu:a6000:1" in launcher
    assert "TresPerNode=gres/gpu:a100:1" in launcher
    assert "qwen3_a6000," in launcher
    assert '"domains": [DOMAIN]' in launcher
    assert '"arms": [ARM]' in launcher
