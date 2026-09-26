from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "ops/validate_deepspeed_checkpoint.py"
RUNNER = ROOT / "ops/run_experiment.sh"
LIVE_ROOT = ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"ok")


def _checkpoint(path: Path, *, corrupt_optimizer: bool = False) -> None:
    path.mkdir(parents=True)
    _zip(path / "mp_rank_00_model_states.pt")
    optimizer = path / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
    if corrupt_optimizer:
        optimizer.write_bytes(b"partial archive")
    else:
        _zip(optimizer)


def test_checkpoint_validator_accepts_complete_and_requires_optimizer(tmp_path):
    module = _load("checkpoint_validate_contract")
    complete = tmp_path / "step_00002"
    _checkpoint(complete)
    assert module.validate_checkpoint(complete) == []
    (complete / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").unlink()
    assert "checkpoint has no optimizer-state archive" in module.validate_checkpoint(complete)


def test_latest_selector_falls_back_from_higher_partial_checkpoint(tmp_path):
    module = _load("checkpoint_select_contract")
    checkpoints = tmp_path / "debug_job1/checkpoints"
    _checkpoint(checkpoints / "step_00002")
    _checkpoint(checkpoints / "step_00004", corrupt_optimizer=True)
    selected, rejected = module.select_latest_checkpoint(tmp_path)
    assert selected == checkpoints / "step_00002"
    assert str(checkpoints / "step_00004") in rejected
    assert any("unreadable ZIP directory" in value for value in rejected[str(checkpoints / "step_00004")])


def test_root_and_live_e111_auto_resume_use_identical_validated_selector():
    root_helper = HELPER.read_bytes()
    live_helper = (LIVE_ROOT / HELPER.name).read_bytes()
    assert live_helper == root_helper
    for runner in (RUNNER, LIVE_ROOT / RUNNER.name):
        text = runner.read_text(encoding="utf-8")
        assert 'validate_deepspeed_checkpoint.py' in text
        assert '"$PYTHON_BIN" "$checkpoint_validator" --select-under "$SAVE_PATH"' in text
