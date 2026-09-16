"""Training-time mode-coverage draws must reserve disjoint child-seed blocks.

vLLM 0.8.4 V0 expands an ``n=K`` request with seed ``s`` into children
``s..s+K-1``. Seeding consecutive draws one apart therefore makes them share
children, and every statistic that pools a prompt's draws -- pairwise modal
diversity above all -- counts those repeats as independent observations.

These tests pin the arithmetic rather than the implementation, so a future
refactor that reintroduces consecutive seeds fails here.
"""
from __future__ import annotations

from collections import Counter


def child_streams(seed_base: int, draw_count: int, k: int, stride: int) -> Counter:
    """Children each draw would receive, counted across the draws."""
    seen: Counter = Counter()
    for draw_index in range(draw_count):
        seed = seed_base + draw_index * stride
        seen.update(range(seed, seed + k))
    return seen


def test_consecutive_seeds_reproduce_the_eleven_stream_defect():
    seen = child_streams(610200, draw_count=4, k=8, stride=1)
    assert len(seen) == 11
    assert sorted(seen.values(), reverse=True)[0] == 4
    assert [seen[s] for s in sorted(seen)] == [1, 2, 3, 4, 4, 4, 4, 4, 3, 2, 1]


def test_striding_by_k_makes_every_draw_disjoint():
    seen = child_streams(610200, draw_count=4, k=8, stride=8)
    assert len(seen) == 32
    assert set(seen.values()) == {1}


def test_default_stride_is_k_for_the_shipped_arguments():
    """Read the declared default from source; importing args pulls in oat/CUDA."""
    import ast
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / 'src/oat_drgrpo/args.py'
    tree = ast.parse(source.read_text())
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                try:
                    defaults[node.target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    assert 'eval_mode_coverage_disjoint_draws' in defaults, (
        'the disjoint-draw switch must remain a declared argument'
    )
    assert defaults['eval_mode_coverage_disjoint_draws'] is True, (
        'new runs must reserve disjoint draw blocks by default'
    )


def test_legacy_flag_still_reproduces_the_old_schedule():
    k, draws, base = 8, 4, 610200
    legacy = [base + i * (1) for i in range(draws)]
    assert legacy == [610200, 610201, 610202, 610203]
    fixed = [base + i * k for i in range(draws)]
    assert fixed == [610200, 610208, 610216, 610224]
    assert not (set(range(fixed[0], fixed[0] + k)) & set(range(fixed[1], fixed[1] + k)))


def test_stride_scales_with_k():
    for k in (2, 4, 8, 16):
        seen = child_streams(1000, draw_count=4, k=k, stride=k)
        assert len(seen) == 4 * k, f'K={k} must yield 4K disjoint streams'
