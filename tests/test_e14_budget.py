import pytest

from ops.exp_scaling.e14_budget import max_queries_for_exact_updates


@pytest.mark.parametrize(
    ("updates", "group_size", "expected_budget"),
    [(1, 16, 1), (4, 16, 48), (128, 16, 2032)],
)
def test_e14_query_budget_matches_oats_strict_post_update_stop(
    updates, group_size, expected_budget
):
    budget = max_queries_for_exact_updates(
        updates=updates, trajectories_per_update=group_size
    )

    assert budget == expected_budget
    assert budget // group_size + 1 == updates
