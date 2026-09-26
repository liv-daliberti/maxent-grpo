from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "certify_e49h_curated_routes.py"
)
CONTRACTS = (
    ROOT
    / "ops/math_strategy_calibration/"
    "e49h_curated_distinct_routes_toy.json"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49h_curated_distinct_route_expansion_20260724.md"
)
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e49h_e49j_restricted_mathir_gate_amendment_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49h_curated_routes.sh"
)
SLURM = ROOT / "ops/slurm/e49h_curated_routes_node915.slurm"


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49h_curated_routes_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_cohort_is_overcomplete_and_split_balanced():
    payload = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    assert len(payload) == 22
    assert sum(key.startswith("train:") for key in payload) == 12
    assert sum(key.startswith("eval:") for key in payload) == 10
    for menu in payload.values():
        assert menu["schema"] == "math_strategy_action_menu_v1"
        assert [row["strategy_id"] for row in menu["strategies"]] == [
            "S1",
            "S2",
        ]
        used = [
            action_id
            for strategy in menu["strategies"]
            for action_id in strategy["action_ids"]
        ]
        assert sorted(used) == sorted(
            action["action_id"] for action in menu["actions"]
        )
        assert len(used) == len(set(used))


def test_preflight_closes_and_nonleak_checks_every_contract():
    module = _load()
    _, source, menus = module._load_and_validate(
        contracts_path=CONTRACTS,
        source=ROOT / "var/data/e49b_math_strategy_toy",
    )
    assert len(source) == 100
    assert len(menus) == 22


def test_protocol_blocks_training_until_manual_zero_false_new_gate():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "FROZEN BEFORE ANY E49H 72B REQUEST" in text
    assert "manual false-new is exactly zero" in text
    assert "at least 10 train and 10 evaluation" in text
    assert "three-epoch toy pair" in text
    assert "ordinary matched Dr.GRPO" in text


def test_launcher_requires_passing_e49j_and_freezes_zero_requests():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    slurm = SLURM.read_text(encoding="utf-8")
    amendment = AMENDMENT.read_text(encoding="utf-8")
    assert "E49J calibration does not authorize E49H" in launcher
    assert "FROZEN BEFORE ANY E49H 72B REQUEST" in amendment
    assert "four-way unanimity" in amendment
    assert '"requests_at_freeze": 0' in launcher
    assert '"soundness_assessment_count": 4 * (' in launcher
    assert '"maximum_pair_veto_assessment_count": 4 * (' in launcher
    assert "--nodelist=node915" in slurm
    assert "E49H frozen binding failed" in slurm
    assert "--workers 4" in slurm


def test_local_decision_requires_all_sound_and_four_strict_votes():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "audit.get(\"pass\") is True" in source
    assert "len(veto_by_row[row_id]) == 4" in source
    assert "distinct_vote_count == 4" in source
    assert "sound_pass[row_id] and hardened_distinct" in source
