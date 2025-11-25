"""
Additional coverage for the vLLM patch helpers.
"""

import json

import pytest

import maxent_grpo.patches.vllm as VP


class _Resp:
    def __init__(self, status_code=200, payload=None, lines=None):
        self.status_code = status_code
        self._payload = payload or {}
        self._lines = lines or []
        self.text = str(self._payload)

    def json(self):
        return self._payload

    def iter_lines(self):
        for ln in self._lines:
            yield ln


def test_safe_generate_retry_then_success(monkeypatch):
    """First request fails → retries → succeeds with non-streaming payload."""

    attempts = []
    slept = []

    def fake_post(url, json, timeout, stream, headers):
        attempts.append(json)
        if len(attempts) == 1:
            raise VP.requests.ConnectionError("flaky")
        return _Resp(200, {"results": [{"outputs": [{"text": "ok"}]}]})

    monkeypatch.setattr(VP.requests, "post", fake_post)
    monkeypatch.setattr(VP.time, "sleep", lambda *_args, **_kwargs: slept.append(True))
    texts, meta, latency = VP.safe_generate(
        prompts=["p1"],
        top_k=2,
        frequency_penalty=0.5,
        presence_penalty=0.1,
        guided_regex="^ok",
        backoff=0.0,
        max_retries=2,
    )
    assert slept == [True]  # backoff was invoked
    assert len(attempts) == 2
    first_payload = attempts[0]
    assert first_payload["top_k"] == 2
    assert first_payload["frequency_penalty"] == 0.5
    assert first_payload["presence_penalty"] == 0.1
    assert first_payload["guided_regex"] == "^ok"
    assert texts == [["ok"]]
    assert meta is None
    assert latency >= 0.0


def test_collect_stream_texts_multi_prompt():
    """Streaming helper groups chunks by prompt index and ignores bad rows."""

    resp = _Resp(
        lines=[
            b'{"prompt_index": 0, "text": "a"}',
            b'{"prompt_index": 1, "text": "b"}',
            b'{"prompt_index": 0, "text": "c"}',
            b'{"prompt_index": 99, "text": "skip"}',
        ]
    )
    out = VP._collect_stream_texts(resp, num_prompts=2)
    assert out == [["ac"], ["b"]]


def test_build_vllm_headers_non_dict_extra(monkeypatch):
    """Extra headers that parse but are not a dict are ignored gracefully."""

    monkeypatch.delenv("VLLM_GROUP_REQUEST_ID", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_EXTRA_HEADERS", json.dumps(["not", "a", "dict"]))
    headers = VP._build_vllm_headers(["foo"])
    assert headers["X-VLLM-Group-Request-ID"]  # hashed from prompts
    assert headers.keys() == {"X-VLLM-Group-Request-ID"}
