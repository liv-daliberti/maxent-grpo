"""Tests for the repository Slurm training launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _resolve_train_slurm() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "ops" / "slurm" / "train_dual_4plus4.slurm",
        repo_root / "ops" / "slurm" / "train.slurm",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Slurm training launcher not found in expected locations")


def _resolve_tiny_smoke_slurm() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root / "ops" / "slurm" / "train_tiny_gpu_smoke.slurm"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Tiny smoke Slurm launcher not found in expected location")


def test_slurm_launcher_uses_shared_grpo_entrypoint():
    script = _resolve_train_slurm().read_text()
    assert 'launch_train "grpo-train" "src/maxent_grpo/grpo.py"' in script
    assert 'launch_train "maxent-train" "src/maxent_grpo/grpo.py"' in script
    assert "/grpo/config_" in script
    assert "/maxent-grpo/config_" in script
    assert "/n/fs/similarity/kalshi" not in script
    assert 'EXTRA_SITE_PACKAGES="${EXTRA_SITE_PACKAGES:-}"' in script


def test_slurm_launcher_help_text(tmp_path):
    env = os.environ.copy()
    env["VAR_DIR"] = str(tmp_path / "var")
    env.setdefault("HF_TOKEN", "test")
    script_path = _resolve_train_slurm()
    cmd = ["bash", str(script_path), "--help"]
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout
    assert ("--run-only" in result.stdout) or ("--task" in result.stdout)


def test_tiny_smoke_launcher_keeps_site_packages_repo_local():
    script = _resolve_tiny_smoke_slurm().read_text()
    assert "/n/fs/similarity/kalshi" not in script
    assert 'ENV_ACTIVATE="${ENV_ACTIVATE:-$ROOT_DIR/var/openr1/bin/activate}"' in script
    assert 'VENV_SITE_PACKAGES="${VENV_SITE_PACKAGES:-$ROOT_DIR/var/openr1/lib/python3.11/site-packages}"' in script
    assert 'EXTERNAL_SITE_PACKAGES="${EXTERNAL_SITE_PACKAGES:-}"' in script
