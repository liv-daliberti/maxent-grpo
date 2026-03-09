from __future__ import annotations

from typing import List

from maxent_grpo.training.generation.vllm_utils import init_vllm_client_communicator


class _FlakyClient:
    def __init__(self, failures: List[BaseException]) -> None:
        self.failures = list(failures)
        self.init_calls = 0
        self.close_calls = 0

    def init_communicator(self) -> None:
        self.init_calls += 1
        if self.failures:
            raise self.failures.pop(0)

    def close_communicator(self) -> None:
        self.close_calls += 1


def test_init_vllm_client_communicator_recovers_from_already_initialized(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MAXENT_VLLM_ASYNC_INIT", "0")
    monkeypatch.setenv("MAXENT_VLLM_INIT_RETRIES", "1")
    monkeypatch.setenv("MAXENT_VLLM_INIT_RETRY_BACKOFF_S", "0")

    client = _FlakyClient(
        [RuntimeError("Weight update group already initialized. Call close_communicator first.")]
    )
    init_vllm_client_communicator(client)

    assert client.init_calls == 2
    assert client.close_calls == 1


def test_init_vllm_client_communicator_raises_after_nonrecoverable_failures(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MAXENT_VLLM_ASYNC_INIT", "0")
    monkeypatch.setenv("MAXENT_VLLM_INIT_RETRIES", "2")
    monkeypatch.setenv("MAXENT_VLLM_INIT_RETRY_BACKOFF_S", "0")

    client = _FlakyClient([RuntimeError("boom"), RuntimeError("still boom")])

    try:
        init_vllm_client_communicator(client)
    except RuntimeError as exc:
        assert "still boom" in str(exc)
    else:  # pragma: no cover - explicit failure message
        raise AssertionError("Expected RuntimeError from nonrecoverable init failures")

    assert client.init_calls == 2
    assert client.close_calls == 2
