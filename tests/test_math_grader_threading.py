"""The full MATH verifier must work inside OAT's actor ThreadPool."""

from __future__ import annotations

from multiprocessing.pool import ThreadPool
import time

import sympy
from math_verify import grader as math_verify_grader

from oat_drgrpo.math_grader import (
    collect_threaded_math_rewards,
    is_latex_equal,
)
from oat_drgrpo.math_grader_process import FullMathVerifierProcess


def _slow_correct_reward(response, reference):
    del response, reference
    time.sleep(1.2)
    return {"formatted": True}, 1.0


def test_full_latex_verifier_is_safe_in_actor_worker_thread():
    with ThreadPool(1) as pool:
        result = pool.apply_async(is_latex_equal, (r"\frac{1}{2}", "0.5"))
        assert result.get(timeout=5) is True


def test_math_verify_equation_comparison_handles_scalar_sympy_roots():
    x = sympy.Symbol("x")
    assert math_verify_grader.sympy_solve_and_compare(
        sympy.Eq(2 * x, 1),
        sympy.Eq(x, sympy.Rational(1, 2)),
        6,
        15,
    )
    assert not math_verify_grader.sympy_solve_and_compare(
        sympy.Eq(2 * x, 1),
        sympy.Eq(x, sympy.Rational(2, 3)),
        6,
        15,
    )


def test_full_verifier_outer_wait_covers_compound_bounded_work():
    with ThreadPool(2) as pool:
        rewards, infos = collect_threaded_math_rewards(
            pool,
            _slow_correct_reward,
            responses=["first", "second"],
            references=["first", "second"],
            timeout_seconds=4,
        )

    assert rewards == [1.0, 1.0]
    assert infos == [{"formatted": True}, {"formatted": True}]


def test_full_verifier_runs_in_dedicated_worker_and_recovers_after_exit():
    grader = FullMathVerifierProcess(reward_kind="boxed", timeout_seconds=5)
    try:
        rewards, infos = grader.grade_batch(
            responses=[r"The answer is $\boxed{\frac{1}{2}}$."],
            references=["0.5"],
        )
        assert rewards == [1.0]
        assert infos == [{"formatted": True}]
        assert grader._process is not None
        old_pid = grader._process.pid
        grader._process.kill()
        grader._process.wait(timeout=2)

        rewards, _ = grader.grade_batch([r"$\boxed{2}$."], ["3"])
        assert rewards == [0.0]
        assert grader._process is not None
        assert grader._process.pid != old_pid
    finally:
        grader.close()


def test_full_verifier_timeout_hard_kills_and_can_restart_worker():
    grader = FullMathVerifierProcess(reward_kind="boxed", timeout_seconds=0.001)
    try:
        rewards, infos = grader.grade_batch([r"$\boxed{1}$."], ["1"])
        assert rewards == [0.0]
        assert infos == [{"formatted": False, "verifier_timeout": True}]
        assert grader._process is None

        grader.timeout_seconds = 5
        rewards, _ = grader.grade_batch([r"$\boxed{1}$."], ["1"])
        assert rewards == [1.0]
    finally:
        grader.close()
