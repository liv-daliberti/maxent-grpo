from __future__ import annotations

import os
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_e50_route_probe_contract_is_64_unforced_samples():
    source = (
        ROOT
        / "ops/math_strategy_calibration/"
        "score_e50_route_probe.py"
    ).read_text(encoding="utf-8")
    assert "SAMPLES = 64" in source
    assert "natural_support_count >= 8 and all_accepted" in source
    assert "min(route_counts.values()) >= 2 and accepted >= 8" in source
    assert "task_reward_positive=positives" in source
    assert "allow_unstructured_menu_inference=True" in source
    assert '"canonicalizer_sha256"' in source
    assert '"menu_source_sha256"' in source


def test_e50_probe_imports_frozen_route_modules_after_current_grader():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = (
        str(ROOT / "src")
        + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    )
    code = f"""
import importlib.util
import pathlib

root = pathlib.Path({str(ROOT)!r})
script = root / "ops/math_strategy_calibration/score_e50_route_probe.py"
spec = importlib.util.spec_from_file_location("probe_loader", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from oat_drgrpo.math_grader import boxed_reward_fn
canonicalizer, menu = module._load_frozen_modules()
frozen = root / (
    "var/artifacts/source_snapshots/"
    "e49t_natural_menu_"
    "3cf26e564b3cafbdc4d0a819febe6d0a61d443746750931c87250f662fcec0a6/"
    "src/oat_drgrpo"
)
assert pathlib.Path(canonicalizer.__file__) == (
    frozen / "math_strategy_canonicalizer.py"
)
assert pathlib.Path(menu.__file__) == frozen / "math_strategy_menu.py"
"""
    subprocess.run(
        [
            str(ROOT / "var/seed_paper_eval/paper310/bin/python"),
            "-c",
            code,
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
