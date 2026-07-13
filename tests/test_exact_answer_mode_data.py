from ops.make_exact_answer_mode_data import _synthetic_graph_rows


def test_easy_local_multi_rows_remain_multi_with_balancing_enabled():
    rows = _synthetic_graph_rows(
        24,
        seed=123,
        hidden_count=1,
        min_completions=2,
        max_completions=3,
        max_n=5,
        max_edges=4,
        min_solutions=2,
        split_tag="eval_multi_answer",
        prompt_style="local_neighbors",
        balance_hidden_color=True,
    )

    assert len(rows) == 24
    assert {row["answer_mode_split"] for row in rows} == {"eval_multi_answer"}
    assert min(int(row["answer_mode_count"]) for row in rows) >= 2
    assert max(int(row["answer_mode_count"]) for row in rows) <= 3
