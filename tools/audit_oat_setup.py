#!/usr/bin/env python3
"""Audit the repository for a single canonical OAT runtime/setup."""

from __future__ import annotations

import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PYTHON = REPO_ROOT / "var" / "seed_paper_eval" / "paper310" / "bin" / "python"
CANONICAL_UPSTREAM = REPO_ROOT / "understand-r1-zero"
EXPECTED_VERSIONS = {
    "torch": "2.6.0",
    "transformers": "4.51.3",
    "vllm": "0.8.4",
    "oat-llm": "0.1.3.post1",
    "deepspeed": "0.16.8",
    "math-verify": "0.7.0",
    "fire": "0.7.0",
}
LEGACY_PATHS = [
    REPO_ROOT / "var" / "seed_paper_eval" / "oat072_py310",
    REPO_ROOT / "var" / "seed_paper_eval" / "oat072_py310_exact",
    REPO_ROOT / "var" / "external" / "understand-r1-zero",
    REPO_ROOT / "var" / "external" / "understand-r1-zero-pristine",
    REPO_ROOT
    / "ops"
    / "slurm"
    / "train_understand_r1_zero_qwen2p5_math_1p5b_main_node302.slurm",
    REPO_ROOT
    / "ops"
    / "slurm"
    / "train_understand_r1_zero_qwen2p5_math_1p5b_r1_noflash_node302.slurm",
    REPO_ROOT / "ops" / "slurm" / "train_oat_zero_exact_1p5b_smoke_all.slurm",
    REPO_ROOT / "ops" / "slurm" / "train_oat_zero_exact_1p5b_mltheory_a100.slurm",
    REPO_ROOT / "ops" / "slurm" / "train_oat_zero_exact_1p5b_long_all_a6000.slurm",
]


def _probe_python_versions() -> dict[str, str]:
    if not CANONICAL_PYTHON.exists():
        raise SystemExit(f"Missing canonical python: {CANONICAL_PYTHON}")
    probe = subprocess.run(
        [
            str(CANONICAL_PYTHON),
            "-c",
            (
                "import importlib.metadata as md, json, sys; "
                "pkgs=['torch','transformers','vllm','oat-llm','deepspeed','math-verify','fire']; "
                "data={'python': sys.version.split()[0]}; "
                "data.update({name: md.version(name) for name in pkgs}); "
                "print(json.dumps(data, sort_keys=True))"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(probe.stdout.strip().splitlines()[-1])


def main() -> int:
    failures: list[str] = []

    if not CANONICAL_UPSTREAM.joinpath("train_zero_math.py").exists():
        failures.append(f"missing canonical upstream checkout: {CANONICAL_UPSTREAM}")

    versions = _probe_python_versions()
    python_version = str(versions.get("python", ""))
    if not python_version.startswith("3.10."):
        failures.append(f"expected python 3.10.x, got {python_version}")
    for package_name, expected_version in EXPECTED_VERSIONS.items():
        actual_version = str(versions.get(package_name, "MISSING"))
        if actual_version != expected_version:
            failures.append(
                f"expected {package_name}=={expected_version}, got {actual_version}"
            )

    existing_legacy_paths = [path for path in LEGACY_PATHS if path.exists()]
    if existing_legacy_paths:
        failures.append(
            "legacy OAT paths still present:\n"
            + "\n".join(f"  - {path}" for path in existing_legacy_paths)
        )

    launcher = (REPO_ROOT / "ops" / "run_oat_zero_exact_1p5b_upstream.sh").read_text(
        encoding="utf-8"
    )
    if "understand-r1-zero-pristine" in launcher or "requirements.no_flash.txt" in launcher:
        failures.append("canonical launcher still references legacy OAT bootstrap paths")

    if failures:
        print("OAT setup audit FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("OAT setup audit OK")
    print(json.dumps(versions, sort_keys=True))
    print(f"canonical_upstream={CANONICAL_UPSTREAM}")
    print(f"canonical_python={CANONICAL_PYTHON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
