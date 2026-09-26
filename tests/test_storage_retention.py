import json
from pathlib import Path
from types import SimpleNamespace

from oat_drgrpo.learner.run import ZeroMathRunMixin


ROOT = Path(__file__).resolve().parent.parent


class _Strategy:
    def __init__(self):
        self.checkpoint_calls = []

    def is_rank_0(self):
        return True

    def save_ckpt(self, _model, save_dir, *, tag, **kwargs):
        self.checkpoint_calls.append((save_dir, tag, kwargs))
        checkpoint = Path(save_dir) / tag
        checkpoint.mkdir(parents=True)
        (checkpoint / "state.bin").write_bytes(b"state")


class _StorageHarness(ZeroMathRunMixin):
    def __init__(self, *, run_root: Path, attempt: str = "debug_job123"):
        self.args = SimpleNamespace(
            save_path=str(run_root),
            save_ckpt=True,
            export_steps=0,
            export_from=0,
            resume_steps=20,
            resume_from=20,
            max_export_num=1,
            max_resume_num=1,
            max_export_mem=64,
            max_resume_mem=256,
            prune_resume_on_success=True,
        )
        self.save_path = str(run_root / attempt)
        Path(self.save_path).mkdir(parents=True)
        self.steps = 20
        self.strategy = _Strategy()
        self.model = SimpleNamespace(model=object())

    def _should_do(self, interval):
        return self.steps % interval == 0

    def _checkpoint_client_state(self):
        return {"steps": self.steps}


def test_default_policy_decouples_eval_resume_and_export(tmp_path):
    learner = _StorageHarness(run_root=tmp_path / "run")

    assert learner._storage_actions(
        terminal=False, allow_scheduled_save=True
    ) == (False, True)
    assert learner._storage_actions(
        terminal=True, allow_scheduled_save=True
    ) == (True, False)

    learner.steps = 10
    assert learner._storage_actions(
        terminal=False, allow_scheduled_save=True
    ) == (False, False)


def test_resume_rotation_keeps_previous_until_new_save_then_retains_one(tmp_path):
    learner = _StorageHarness(run_root=tmp_path / "run")
    checkpoint_root = Path(learner.save_path) / "checkpoints"
    old = checkpoint_root / "step_00010"
    old.mkdir(parents=True)
    (old / "state.bin").write_bytes(b"old")

    learner._save_resume_checkpoint()

    assert [path.name for path in checkpoint_root.iterdir() if path.is_dir()] == [
        "step_00020"
    ]
    assert learner.strategy.checkpoint_calls[0][2]["max_num"] == 2


def test_success_prunes_every_attempt_checkpoint_and_writes_marker(tmp_path):
    run_root = tmp_path / "run"
    learner = _StorageHarness(run_root=run_root, attempt="debug_job123")
    for attempt in ("debug_old", "debug_job123"):
        checkpoint = run_root / attempt / "checkpoints" / "step_00020"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "state.bin").write_bytes(b"state")
    terminal = run_root / "debug_job123" / "saved_models" / "step_00020"
    terminal.mkdir(parents=True)
    (terminal / "model.safetensors").write_bytes(b"weights")

    learner._finalize_successful_storage()

    assert not list(run_root.glob("*/checkpoints"))
    assert terminal.is_dir()
    marker = json.loads((run_root / "TRAINING_COMPLETE.json").read_text())
    assert marker["terminal_step"] == 20
    assert marker["resume_checkpoints_pruned"] is True
    assert len(marker["removed_checkpoint_roots"]) == 2
    assert marker["cleanup_errors"] == []


def test_pruning_can_be_explicitly_disabled(tmp_path):
    run_root = tmp_path / "run"
    learner = _StorageHarness(run_root=run_root)
    learner.args.prune_resume_on_success = False
    checkpoint = Path(learner.save_path) / "checkpoints" / "step_00020"
    checkpoint.mkdir(parents=True)

    learner._finalize_successful_storage()

    assert checkpoint.is_dir()
    marker = json.loads((run_root / "TRAINING_COMPLETE.json").read_text())
    assert marker["resume_checkpoints_pruned"] is False


def test_missing_terminal_export_preserves_resume_state(tmp_path):
    run_root = tmp_path / "run"
    learner = _StorageHarness(run_root=run_root)
    checkpoint = Path(learner.save_path) / "checkpoints" / "step_00020"
    checkpoint.mkdir(parents=True)

    learner._finalize_successful_storage()

    assert checkpoint.is_dir()
    marker = json.loads((run_root / "TRAINING_COMPLETE.json").read_text())
    assert marker["resume_checkpoints_pruned"] is False
    assert "terminal export missing" in marker["cleanup_errors"][0]


def test_shared_launcher_defaults_to_lean_storage_and_stable_requeues():
    experiment = (ROOT / "ops/run_experiment.sh").read_text(encoding="utf-8")
    trainer = (ROOT / "ops/train.sh").read_text(encoding="utf-8")

    assert 'OAT_ZERO_FIXED_EXP_SUFFIX="${OAT_ZERO_FIXED_EXP_SUFFIX:-job${SLURM_JOB_ID}}"' in experiment
    assert 'OAT_ZERO_EXPORT_STEPS="${OAT_ZERO_EXPORT_STEPS:-0}"' in experiment
    assert 'OAT_ZERO_RESUME_STEPS="${OAT_ZERO_RESUME_STEPS:-$PROMPT_EPOCH_STEPS}"' in experiment
    assert 'OAT_ZERO_MAX_EXPORT_NUM="${OAT_ZERO_MAX_EXPORT_NUM:-1}"' in experiment
    assert 'OAT_ZERO_MAX_RESUME_NUM="${OAT_ZERO_MAX_RESUME_NUM:-1}"' in experiment
    assert 'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS="${OAT_ZERO_PRUNE_RESUME_ON_SUCCESS:-1}"' in experiment
    assert '--export-steps "${OAT_ZERO_EXPORT_STEPS:-0}"' in trainer
    assert '--resume-steps "${OAT_ZERO_RESUME_STEPS:--1}"' in trainer
    assert "--no-prune-resume-on-success" in trainer
