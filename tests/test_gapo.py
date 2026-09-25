"""Contract tests for the GAPO frequency-aware group reward."""

from __future__ import annotations

import json

import pytest

from oat_drgrpo.gapo import (
    SCALE_PAPER,
    SUPPORT_INDEX_SCHEMA,
    GAPOSupportIndex,
    gapo_group_rewards,
    reference_key,
)
from oat_drgrpo.outcome_collision import INVALID_OUTCOME_KEY


def test_published_reward_matches_the_paper_formula():
    """A valid row carries 1 - (f_i - 1/L); an invalid row carries -1."""

    keys = ["a", "a", "b", None]
    rewards, _ = gapo_group_rewards(
        keys, [1, 1, 1, 0], [4] * 4, num_samples=4, reward_scale=SCALE_PAPER
    )
    # Three valid rows: f(a) = 2/3, f(b) = 1/3, L = 4.
    assert rewards[0] == pytest.approx(1.0 - (2 / 3 - 0.25))
    assert rewards[1] == pytest.approx(1.0 - (2 / 3 - 0.25))
    assert rewards[2] == pytest.approx(1.0 - (1 / 3 - 0.25))
    assert rewards[3] == pytest.approx(-1.0)


def test_unit_scaling_is_the_affine_image_of_the_published_span():
    """Unit scaling preserves relative structure and returns the control span."""

    keys = ["a", "a", "b", None]
    published, _ = gapo_group_rewards(
        keys, [1, 1, 1, 0], [4] * 4, num_samples=4, reward_scale=SCALE_PAPER
    )
    unit, _ = gapo_group_rewards(keys, [1, 1, 1, 0], [4] * 4, num_samples=4)
    for raw, scaled in zip(published, unit):
        assert scaled == pytest.approx((raw + 1.0) / 2.0)
    # An invalid row lands exactly on the binary control's zero.
    assert unit[3] == pytest.approx(0.0)


def test_a_rarer_mode_is_rewarded_above_a_duplicated_one():
    """The whole point of the objective, stated as a test."""

    rewards, _ = gapo_group_rewards(
        ["a", "a", "a", "b"], [1, 1, 1, 1], [4] * 4, num_samples=4
    )
    assert rewards[3] > rewards[0]


def test_frequency_is_normalized_over_valid_rows_not_the_group():
    """GAPO's f divides by the count of valid rows, as published."""

    rewards, diagnostics = gapo_group_rewards(
        ["a", "a", None, None], [1, 1, 0, 0], [8] * 4, num_samples=4
    )
    # Both valid rows share one key, so f = 2/2 = 1 rather than 2/4.
    assert diagnostics.frequency_max == pytest.approx(1.0)
    assert rewards[0] == pytest.approx((1.0 - (1.0 - 1 / 8) + 1.0) / 2.0)


def test_support_larger_than_the_group_is_reported_not_hidden():
    """When L exceeds G the uniform target is unreachable by construction."""

    _, diagnostics = gapo_group_rewards(
        ["a", "b", "c", "d"], [1, 1, 1, 1], [3600] * 4, num_samples=4
    )
    assert diagnostics.unreachable_support_group_fraction == pytest.approx(1.0)
    _, reachable = gapo_group_rewards(
        ["a", "b", "c", "d"], [1, 1, 1, 1], [3] * 4, num_samples=4
    )
    assert reachable.unreachable_support_group_fraction == pytest.approx(0.0)


def test_a_verified_row_without_an_answer_key_is_refused():
    """A certified-correct row with no identity would understate breadth."""

    for key in (None, INVALID_OUTCOME_KEY):
        with pytest.raises(ValueError, match="no answer key"):
            gapo_group_rewards(
                [key, "b", "c", "d"], [1, 1, 1, 1], [4] * 4, num_samples=4
            )


def test_one_group_must_carry_one_support_size():
    with pytest.raises(ValueError, match="one support size"):
        gapo_group_rewards(
            ["a", "b", "c", "d"], [1, 1, 1, 1], [4, 4, 5, 4], num_samples=4
        )


def test_support_index_refuses_an_uncovered_prompt(tmp_path):
    """A lookup miss fails closed rather than defaulting the support."""

    path = tmp_path / "index.json"
    path.write_text(
        json.dumps(
            {
                "schema": SUPPORT_INDEX_SCHEMA,
                "support_sizes": {reference_key("known"): 7},
            }
        ),
        encoding="utf-8",
    )
    index = GAPOSupportIndex.load(path)
    assert index.lookup(["known", "known"]) == [7, 7]
    with pytest.raises(ValueError, match="does not cover"):
        index.lookup(["known", "unknown"])


def test_support_index_rejects_a_foreign_schema(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps({"schema": "something_else", "support_sizes": {"a": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema"):
        GAPOSupportIndex.load(path)
