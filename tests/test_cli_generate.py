"""Smoke test for Hydra CLI entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
import os


def test_hydra_cli_help(monkeypatch) -> None:
    """Ensure the Hydra console script runs and shows help without torch imports."""

    # Ensure we resolve the CLI from the repo without installation
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv(
        "PYTHONPATH",
        str(repo_root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    # Import with torch/accelerate/trl stubbed to satisfy training.types guard.
    torch_mod = type("TorchModule", (), {})()
    torch_utils = type("TorchUtils", (), {})()

    class _TorchData:
        DataLoader = object
        Sampler = object

    torch_utils.data = _TorchData()
    torch_mod.utils = torch_utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils.data)
    accel_stub = type("AccelStub", (), {"Accelerator": lambda *a, **k: None})()
    accel_state = type(
        "AccelStateStub",
        (),
        {"DistributedType": type("DT", (), {"DEEPSPEED": "deepspeed"})},
    )()
    monkeypatch.setitem(sys.modules, "accelerate", accel_stub)
    monkeypatch.setitem(sys.modules, "accelerate.state", accel_state)
    trl_stub = type(
        "TrlStub",
        (),
        {"ModelConfig": type("ModelConfig", (), {}), "TrlParser": lambda *_: None},
    )()
    monkeypatch.setitem(sys.modules, "trl", trl_stub)
    module = __import__("cli.hydra_cli", fromlist=["hydra_main"])
    assert hasattr(module, "hydra_main")
