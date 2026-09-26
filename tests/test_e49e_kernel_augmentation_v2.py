from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "augment_e49e_kernel_routes_v2.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49e_kernel_augmentation_v2_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route(kernel: str, combo: tuple[str, str], operation: str):
    return {
        "kernel_id": kernel,
        "actions": [
            {"op_code": combo[0], "operation": operation},
            {"op_code": combo[1], "operation": "Conclude symbolically."},
        ],
        "plan": "Apply the declared operations without precomputing values.",
    }


def test_v2_drops_leaking_and_duplicate_routes_individually():
    module = _load()
    payload = {
        "schema": module.PROPOSAL_SCHEMA,
        "routes": [
            _route(
                "direct_algebra",
                ("NORMALIZE", "CONCLUDE"),
                "Derive the forbidden answer 9.",
            ),
            _route(
                "factorization_roots",
                ("FACTOR", "CONCLUDE"),
                "Factor the symbolic expression.",
            ),
            _route(
                "symmetry_invariant",
                ("FACTOR", "CONCLUDE"),
                "Repeat the same factorization.",
            ),
        ],
    }
    menu = module._payload_to_menu(
        payload,
        "9",
        problem="Solve the symbolic expression in x.",
    )
    assert len(menu.strategies) == 1
    assert "[KERNEL:factorization_roots]" in menu.strategies[0].plan


def test_v2_still_fails_closed_when_every_route_leaks():
    module = _load()
    payload = {
        "schema": module.PROPOSAL_SCHEMA,
        "routes": [
            _route(
                "direct_algebra",
                ("NORMALIZE", "CONCLUDE"),
                "Derive 9.",
            ),
            _route(
                "factorization_roots",
                ("FACTOR", "CONCLUDE"),
                "Factor until the value 9 appears.",
            ),
        ],
    }
    with pytest.raises(
        ValueError,
        match="no individually admissible",
    ):
        module._payload_to_menu(
            payload,
            "9",
            problem="Solve the symbolic expression in x.",
        )


def test_v2_launcher_and_repair_bind_the_iteration():
    launcher = (
        ROOT
        / "ops/math_strategy_calibration/"
        "launch_e49e_kernel_augmentation_job.sh"
    ).read_text(encoding="utf-8")
    repair_launcher = (
        ROOT
        / "ops/math_strategy_calibration/"
        "launch_e49e_singleton_repair_job.sh"
    ).read_text(encoding="utf-8")
    repair = (
        ROOT
        / "ops/math_strategy_calibration/"
        "repair_e49e_singleton_gaps.py"
    ).read_text(encoding="utf-8")
    assert "prior_augmentation_records_sha256" in launcher
    assert "E49E_KERNEL_PRIOR_RECORDS" in launcher
    assert "e49e_kernel_routewise_v2_amendment_20260724.md" in launcher
    assert "e49e_kernel_augmentation_math_toy_v2" in repair_launcher
    assert "e49e_finite_kernel_augmentation_v2" in repair_launcher
    assert "e49e_finite_kernel_augmentation_v2" in repair


def test_v2_carries_forward_only_strictly_larger_v1_support(monkeypatch):
    module = _load()
    one = module.base.pipeline._menu_from_payload(
        {
            "schema": module.base.pipeline.MENU_SCHEMA,
            "actions": [
                {"action_id": "A1", "operation": "Normalize symbolically."}
            ],
            "strategies": [
                {
                    "strategy_id": "S1",
                    "action_ids": ["A1"],
                    "plan": "Use direct normalization.",
                }
            ],
        }
    )
    two = module.base.pipeline._menu_from_payload(
        {
            "schema": module.base.pipeline.MENU_SCHEMA,
            "actions": [
                {"action_id": "A1", "operation": "Normalize symbolically."},
                {"action_id": "A2", "operation": "Factor symbolically."},
            ],
            "strategies": [
                {
                    "strategy_id": "S1",
                    "action_ids": ["A1"],
                    "plan": "Use direct normalization.",
                },
                {
                    "strategy_id": "S2",
                    "action_ids": ["A2"],
                    "plan": "Use factorization.",
                },
            ],
        }
    )
    prior = {
        "origin": "finite_kernel_augmentation",
        "pass": True,
        "menu": module.base.json.loads(two.canonical_json),
    }
    module._prior_records["row"] = prior
    monkeypatch.setattr(
        module,
        "_base_build_one",
        lambda **_: {
            "row_id": "row",
            "menu": module.base.json.loads(one.canonical_json),
            "menu_sha256": one.sha256,
            "proposal": {"pass": True},
            "certification": {"old": False},
            "pass": True,
        },
    )
    monkeypatch.setattr(
        module.base.repair,
        "_augmentation_selected",
        lambda *_, **__: (two, {"old": True}),
    )
    result = module._build_one(
        row_id="row",
        raw_record={},
        input_record={},
        problem="p",
        reference_answer="a",
    )
    assert result["origin"] == "prior_v1_finite_kernel_augmentation"
    assert len(result["menu"]["strategies"]) == 2
    assert result["certification"] == {"old": True}
    assert result["pass"] is True
