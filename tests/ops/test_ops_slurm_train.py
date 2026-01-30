"""Tests for ops/slurm/train.slurm task routing."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def _task_entries(script: str) -> dict[str, dict[str, str]]:
    match = re.search(r'case "\$TASK_NORMALIZED" in(?P<body>.*?)esac', script, re.S)
    assert match, "could not locate task dispatcher in train.slurm"
    body = match.group("body")
    entries: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"^\s*([^\n)]+)\)\s*(.+?)\n\s*;;", re.M | re.S)
    for case in pattern.finditer(body):
        selectors = [selector.strip() for selector in case.group(1).split("|")]
        block = case.group(2)
        entry_match = re.search(r'TRAIN_ENTRYPOINT="([^"]+)"', block)
        dir_match = re.search(r'TASK_DIR="([^"]+)"', block)
        for selector in selectors:
            entries[selector] = {
                "entrypoint": entry_match.group(1) if entry_match else "",
                "task_dir": dir_match.group(1) if dir_match else "",
            }
    return entries


def test_slurm_task_dispatch_includes_infoseed():
    script = Path("ops/slurm/train.slurm").read_text()
    entries = _task_entries(script)
    assert entries["grpo"]["entrypoint"].endswith("maxent_grpo/grpo.py")
    assert entries["grpo"]["task_dir"] == "grpo"
    assert entries["maxent"]["entrypoint"].endswith("maxent_grpo/maxent_grpo.py")
    assert entries["maxent"]["task_dir"] == "maxent-grpo"
    assert entries["maxent-grpo"]["entrypoint"].endswith("maxent_grpo/maxent_grpo.py")
    assert entries["infoseed"]["entrypoint"].endswith("maxent_grpo/infoseed.py")
    assert entries["infoseed"]["task_dir"] == "infoseed"
    assert entries["info-seed"]["entrypoint"].endswith("maxent_grpo/infoseed.py")


def test_slurm_train_supports_dry_run(tmp_path):
    env = os.environ.copy()
    env["MAXENT_DRY_RUN"] = "1"
    env["VAR_DIR"] = str(tmp_path / "var")
    env.setdefault("HF_TOKEN", "test")
    cmd = [
        "bash",
        "ops/slurm/train.slurm",
        "--model",
        "Qwen2.5-1.5B-Instruct",
        "--task",
        "maxent",
        "--config",
        "math",
        "--accelerator",
        "zero3",
    ]
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "[dry-run]" in result.stdout
