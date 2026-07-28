from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
OVERLAY = (
    ROOT
    / "var/artifacts/source_snapshots/"
    "e50_math_corrected_overlay_v1/src"
)
FROZEN = (
    ROOT
    / "var/artifacts/source_snapshots/"
    "e49t_natural_menu_"
    "3cf26e564b3cafbdc4d0a819febe6d0a61d443746750931c87250f662fcec0a6/"
    "src/oat_drgrpo"
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        value
        for value in path.rglob("*")
        if value.is_file()
        and "__pycache__" not in value.parts
        and value.suffix != ".pyc"
    ):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def test_e50_overlay_preserves_canonicalizer_and_repairs_scalar_equations():
    assert _tree_sha256(OVERLAY) == (
        "ba032e9ca300c25385be9650582556f6c8f833ce1f4f5a7197c4259ce5e44a1e"
    )
    assert _sha256(FROZEN / "math_strategy_canonicalizer.py") == (
        "1dc83a2cd9da4cb092dbde747a647b416107846f985683104a98cbe9b022282b"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(OVERLAY)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = r"""
import pathlib
import sympy
from math_verify import grader
import oat_drgrpo.math_strategy_canonicalizer as canonicalizer
import oat_drgrpo.math_grader as math_grader
import oat_drgrpo.online_canonical_controller as controller

expected = pathlib.Path(
    "/n/fs/similarity/maxent-grpo/var/artifacts/source_snapshots/"
    "e49t_natural_menu_"
    "3cf26e564b3cafbdc4d0a819febe6d0a61d443746750931c87250f662fcec0a6/"
    "src/oat_drgrpo/math_strategy_canonicalizer.py"
)
assert pathlib.Path(canonicalizer.__file__) == expected
assert pathlib.Path(math_grader.__file__) == expected.with_name("math_grader.py")
assert pathlib.Path(controller.__file__) == expected.with_name(
    "online_canonical_controller.py"
)
x = sympy.Symbol("x")
assert grader.sympy_solve_and_compare(
    sympy.Eq(x, sympy.Rational(1, 2)),
    sympy.Eq(2 * x, 1),
    6,
    15,
)
assert not grader.sympy_solve_and_compare(
    sympy.Eq(x, sympy.Rational(1, 2)),
    sympy.Eq(2 * x, 3),
    6,
    15,
)
assert getattr(
    grader.sympy_solve_and_compare,
    "_e50_scalar_equation_repair",
    False,
)
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


def test_e50_wrappers_bind_overlay_and_frozen_canonicalizer_separately():
    toy = (
        ROOT / "ops/exp_scaling/launch_e50_matched_math_toy_05b.sh"
    ).read_text(encoding="utf-8")
    full = (
        ROOT / "ops/exp_scaling/launch_e50e_exact_oat_math_05b.sh"
    ).read_text(encoding="utf-8")
    base_toy = (
        ROOT
        / "ops/exp_scaling/"
        "launch_e49t_natural_menu_math_toy_05b.sh"
    ).read_text(encoding="utf-8")
    base_full = (
        ROOT
        / "ops/exp_scaling/"
        "launch_e49v_exact_oat_natural_menu_math_05b.sh"
    ).read_text(encoding="utf-8")
    assert "E49T_SOURCE_ROOT_OVERRIDE=\"$source_overlay\"" in toy
    assert "E49T_CANONICALIZER_SOURCE_OVERRIDE" in toy
    assert "E49V_SOURCE_ROOT_OVERRIDE=\"$source_overlay\"" in full
    assert "E49V_CANONICALIZER_SOURCE_OVERRIDE" in full
    assert "E49T_CANONICALIZER_SOURCE_OVERRIDE" in base_toy
    assert "E49V_CANONICALIZER_SOURCE_OVERRIDE" in base_full
