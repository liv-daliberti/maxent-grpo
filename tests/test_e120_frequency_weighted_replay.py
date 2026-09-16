import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_e120_key_weighting_is_forwarded_end_to_end():
    args = (ROOT / "src/oat_drgrpo/args.py").read_text(encoding="utf-8")
    train = (ROOT / "ops/train.sh").read_text(encoding="utf-8")
    launcher = (
        ROOT / "ops/exp_scaling/launch_e120_frequency_weighted_replay.py"
    ).read_text(encoding="utf-8")

    assert "online_canonical_replay_key_weighting:" in args
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING:-uniform" in train
    assert "--online-canonical-replay-key-weighting" in train
    assert '"$ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING"' in train
    assert "Frozen source lacks requested canonical replay key weighting" in train
    assert (
        '"OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING": KEY_WEIGHTING'
    ) in launcher


def test_e120_resource_paths_and_scheduler_requests_exclude_pvl():
    launcher = (
        ROOT / "ops/exp_scaling/launch_e120_frequency_weighted_replay.py"
    ).read_text(encoding="utf-8")
    wrapper = (
        ROOT / "ops/slurm/e120_resource_fenced_train.slurm"
    ).read_text(encoding="utf-8")

    assert '"partition": "cs"' in launcher
    assert '"partition": "mltheory"' in launcher
    assert '"account": "allcs"' in launcher
    assert '"account": "mltheory"' in launcher
    assert "refuse_pvl(record" in launcher
    assert '"${record,,}" == *pvl*' in wrapper
    assert "e120_non_pvl_train" not in launcher


def load_smoke_auditor():
    path = ROOT / "ops/exp_scaling/audit_e120_frequency_smoke.py"
    spec = importlib.util.spec_from_file_location("e120_smoke_auditor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e120_smoke_auditor_follows_terminal_attempt_receipt(tmp_path):
    auditor = load_smoke_auditor()
    run_dir = tmp_path / "run"
    attempt = run_dir / "debug_job123"
    attempt.mkdir(parents=True)
    metrics = attempt / "train_metrics.jsonl"
    metrics.write_text("{}\n", encoding="utf-8")
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps({"terminal_attempt": str(attempt)}) + "\n", encoding="utf-8"
    )
    assert auditor.resolve_metrics(run_dir) == metrics


def test_e120_smoke_auditor_rejects_terminal_attempt_escape(tmp_path):
    auditor = load_smoke_auditor()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps({"terminal_attempt": str(outside)}) + "\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="invalid terminal attempt receipt"):
        auditor.resolve_metrics(run_dir)

