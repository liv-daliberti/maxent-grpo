from __future__ import annotations

from pathlib import Path

import torch

from oat_drgrpo.vllm_worker import (
    Level2SafeWorkerWrap,
    PinnedWorkerExtensionArgs,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "src/oat_drgrpo/learner/run.py"
LAUNCHERS = ROOT / "ops/exp_scaling"


class _BufferedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([7.0]))
        self.register_buffer(
            "cos_sin_cache",
            torch.tensor([[1.0, 0.0], [0.5, -0.5]]),
            persistent=False,
        )


def _worker(model: torch.nn.Module) -> Level2SafeWorkerWrap:
    worker = object.__new__(Level2SafeWorkerWrap)
    worker.model_runner = type("Runner", (), {"model": model})()
    worker._oat_zero_buffer_backup = None
    return worker


def test_level2_worker_restores_nonpersistent_buffers_exactly():
    model = _BufferedModel()
    original = model.cos_sin_cache.clone()
    worker = _worker(model)

    backup = worker.backup_model_buffers()
    model.cos_sin_cache.fill_(float("nan"))  # stand in for discarded storage
    restored = worker.restore_model_buffers()

    assert backup == restored == {
        "buffer_count": 1,
        "buffer_bytes": original.numel() * original.element_size(),
    }
    assert torch.equal(model.cos_sin_cache, original)
    assert worker._oat_zero_buffer_backup is None


def test_level2_worker_fails_closed_on_missing_or_changed_buffers():
    worker = _worker(_BufferedModel())
    try:
        worker.restore_model_buffers()
    except RuntimeError as exc:
        assert "without a backup" in str(exc)
    else:
        raise AssertionError("restore without backup should fail")

    worker.backup_model_buffers()
    del worker.model_runner.model.cos_sin_cache
    try:
        worker.restore_model_buffers()
    except RuntimeError as exc:
        assert "buffers changed" in str(exc)
    else:
        raise AssertionError("buffer-set drift should fail")


def test_level2_actor_pins_the_safe_worker_extension():
    args = PinnedWorkerExtensionArgs({"tensor_parallel_size": 1})
    args.update({"worker_extension_cls": "oat.utils.distributed.WorkerWrap"})

    assert args["worker_extension_cls"].endswith("Level2SafeWorkerWrap")


def test_level2_lifecycle_backs_up_before_discard_and_restores_after_remap():
    source = RUN.read_text(encoding="utf-8")

    assert source.index("backup_model_buffers") < source.index("actor.futures.sleep(2)")
    assert source.index('actor.futures.wake_up(["weights"])') < source.index(
        "restore_model_buffers"
    )


def test_no_other_scaling_launcher_silently_enables_level2_sleep():
    exposed = []
    for launcher in LAUNCHERS.glob("*.sh"):
        text = launcher.read_text(encoding="utf-8")
        if "OAT_ZERO_VLLM_SLEEP_LEVEL=2" in text:
            exposed.append(launcher.name)

    assert exposed == ["launch_e29_modebench_freeform_7b_4gpu.sh"]
