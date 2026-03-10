from __future__ import annotations

import json

from maxent_grpo.rewards.basic import (
    get_reward_funcs,
    python_unit_test_reward,
)


def test_python_unit_test_reward_mbpp_pass_fail(monkeypatch):
    monkeypatch.setenv("MAXENT_CODE_REWARD_WORKERS", "1")
    monkeypatch.setenv("MAXENT_CODE_REWARD_TIMEOUT_S", "3")

    completions = [
        "```python\ndef square(x):\n    return x * x\n```",
        "```python\ndef square(x):\n    return x + x\n```",
    ]
    answers = [
        str(["assert square(2) == 4", "assert square(5) == 25"]),
        str(["assert square(2) == 4", "assert square(5) == 25"]),
    ]

    rewards = python_unit_test_reward(completions, answers)
    assert rewards[0] == 1.0
    assert rewards[1] < 1.0


def test_python_unit_test_reward_humaneval_style(monkeypatch):
    monkeypatch.setenv("MAXENT_CODE_REWARD_WORKERS", "1")
    monkeypatch.setenv("MAXENT_CODE_REWARD_TIMEOUT_S", "3")

    completions = ["```python\ndef add(a, b):\n    return a + b\n```"]
    answers = [
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(-1, 5) == 4\n"
    ]
    prompts = ['def add(a, b):\n    """Return a+b."""']

    rewards = python_unit_test_reward(completions, answers, prompts=prompts)
    assert rewards == [1.0]


def test_python_unit_test_reward_apps_style(monkeypatch):
    monkeypatch.setenv("MAXENT_CODE_REWARD_WORKERS", "1")
    monkeypatch.setenv("MAXENT_CODE_REWARD_TIMEOUT_S", "3")

    completions = ["```python\na, b = map(int, input().split())\nprint(a + b)\n```"]
    answers = [
        json.dumps(
            {
                "inputs": ["1 2\n", "10 5\n"],
                "outputs": ["3\n", "15\n"],
            }
        )
    ]

    rewards = python_unit_test_reward(completions, answers)
    assert rewards == [1.0]


def test_get_reward_funcs_supports_python_unit_test_aliases():
    cfg = type(
        "Cfg", (), {"reward_funcs": ["python_unit_tests", "mbpp_python_tests"]}
    )()
    funcs = get_reward_funcs(cfg)
    assert len(funcs) == 2
    assert funcs[0] is python_unit_test_reward
    assert funcs[1] is python_unit_test_reward
