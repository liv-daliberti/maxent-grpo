"""Static guardrails for objective-branch localization.

These checks keep GRPO-vs-MaxEnt runtime branching in ``training/trl_trainer.py``
so the rest of the training pipeline stays identical across objectives.
"""

from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_train_objective_flag_not_used_in_shared_pipeline_files() -> None:
    repo = Path(__file__).resolve().parents[2]
    shared_files = [
        repo / "src" / "maxent_grpo" / "pipelines" / "training" / "baseline.py",
        repo / "src" / "maxent_grpo" / "core" / "model.py",
    ]
    for path in shared_files:
        text = _read(str(path))
        assert "train_grpo_objective" not in text, (
            f"{path} should stay objective-agnostic; keep GRPO-vs-MaxEnt "
            "branching in training/trl_trainer.py."
        )


def test_train_objective_flag_stays_in_trl_trainer() -> None:
    repo = Path(__file__).resolve().parents[2]
    trainer_text = _read(
        str(repo / "src" / "maxent_grpo" / "training" / "trl_trainer.py")
    )
    assert "train_grpo_objective" in trainer_text
