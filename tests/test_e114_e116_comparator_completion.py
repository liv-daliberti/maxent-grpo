"""Contracts for the E114--E116 five-method comparator completion."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"
sys.path.insert(0, str(EXP))

import launch_e114_plain_grpo_qwen3b_extension as e114  # noqa: E402
import launch_e115_ucpo_completion as e115  # noqa: E402

pytest.importorskip("datasets")
import launch_e116_sparse_rlep_completion as e116  # noqa: E402


def test_new_scientific_grid_is_exactly_90_cells():
    grpo = len(e114.DOMAINS) * len(e114.SEEDS)
    ucpo = sum(len(family.domains) * len(family.seeds) for family in e115.FAMILIES)
    rlep = sum(len(family.domains) * len(family.seeds) for family in e116.FAMILIES)

    assert (grpo, ucpo, rlep) == (20, 35, 35)
    assert grpo + ucpo + rlep == 90


def test_completion_targets_only_missing_cells():
    q05_ucpo = next(f for f in e115.FAMILIES if f.key == "qwen05b")
    q3_ucpo = next(f for f in e115.FAMILIES if f.key == "qwen3b")
    q05_rlep = next(f for f in e116.FAMILIES if f.key == "qwen05b")
    q3_rlep = next(f for f in e116.FAMILIES if f.key == "qwen3b")

    assert q05_ucpo.domains == q05_rlep.domains == ("countdown", "mathir")
    assert q05_ucpo.seeds == q05_rlep.seeds == (43, 44, 45, 46, 47)
    assert q3_ucpo.domains == q3_rlep.domains == e114.DOMAINS
    assert q3_ucpo.seeds == q3_rlep.seeds == (70, 71, 72, 73, 74)
    assert e114.SEEDS == (71, 72, 73, 74)


def test_ucpo_and_sparse_rlep_objectives_are_frozen():
    assert e115.objective() == {
        "OAT_ZERO_VARIANT": "ucpo",
        "OAT_ZERO_UCPO_TAU": "0.2",
        "OAT_ZERO_RLEP_EXPERIENCE_ROOT": "",
        "OAT_ZERO_RLEP_REPLAY_COUNT": "0",
    }
    required = dict(e116.SNAPSHOT_REQUIREMENTS)
    assert required["src/oat_drgrpo/pantry_support_action.py"] == (
        "pantry_support_mask_from_allocation"
    )
    assert required["ops/exp_scaling/audit_e98_rlep_pool.py"] == "--allow-sparse"


def test_every_batch_has_a_pre_submission_protocol():
    for protocol in (e114.PROTOCOL, e115.PROTOCOL, e116.PROTOCOL):
        assert protocol.is_file()
        text = protocol.read_text(encoding="utf-8")
        assert "Date frozen: 2026-08-19" in text
        assert "before submission" in text
