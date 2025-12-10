"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json

import pytest

import maxent_grpo.patches.vllm as VP


class R:
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


def test_safe_request_ok(monkeypatch):
    def fake_get(url, timeout):
        assert url.endswith("/health")
        return R(200, {"ok": True})

    monkeypatch.setattr(VP.requests, "get", fake_get)
    out = VP.safe_request("http://x/health")
    assert out == {"ok": True}


def test_safe_generate_parses_choices(monkeypatch):
    def fake_post(url, json, timeout, stream, headers):
        assert stream is False
        assert "X-VLLM-Group-Request-ID" in headers
        return R(200, {"choices": [{"text": "A"}, {"text": "B"}]})

    monkeypatch.setattr(VP.requests, "post", fake_post)
    texts, meta, latency = VP.safe_generate(prompts=["p"], stream=False)
    assert texts == [["A", "B"]]
    assert meta is None
    assert latency >= 0.0


def test_safe_generate_decodes_token_ids(monkeypatch):
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(map(str, ids))

    def fake_post(url, json, timeout, stream, headers):
        return R(200, {"completion_ids": [[1, 2, 3]]})

    monkeypatch.setattr(VP.requests, "post", fake_post)
    texts, meta, _ = VP.safe_generate(prompts=["p"], stream=False, tokenizer=Tok())
    assert texts == [["123"]]
    assert meta is None


def test_safe_generate_stream_joins(monkeypatch):
    lines = [
        b'{"prompt_index": 0, "text": "he"}',
        b'{"prompt_index": 0, "text": "llo"}',
    ]

    def fake_post(url, json, timeout, stream, headers):
        assert stream is True
        return R(200, payload={}, lines=lines)

    monkeypatch.setattr(VP.requests, "post", fake_post)
    texts, meta, _ = VP.safe_generate(prompts=["p"], stream=True)
    assert texts == [["hello"]]
    assert meta is None


def test_safe_generate_with_logprobs(monkeypatch):
    payload = {
        "results": [
            {
                "outputs": [
                    {
                        "text": "done",
                        "cumulative_logprob": -2.5,
                        "token_ids": [1, 2, 3],
                        "output_token_logprobs": [-0.5, -1.0, -1.0],
                    }
                ]
            }
        ]
    }

    def fake_post(url, json, timeout, stream, headers):
        return R(200, payload)

    monkeypatch.setattr(VP.requests, "post", fake_post)
    texts, meta, _ = VP.safe_generate(prompts=["p"], stream=False, return_logprobs=True)
    assert texts == [["done"]]
    assert meta is not None and meta[0][0] is not None
    assert meta[0][0].logprob_sum == -2.5
    assert meta[0][0].token_count == 3
    assert meta[0][0].token_logprobs == [-0.5, -1.0, -1.0]
    assert meta[0][0].raw_output["token_ids"] == [1, 2, 3]


def test_safe_generate_filters_client_tag(monkeypatch):
    payload = {
        "results": [
            {"metadata": {"client_tag": "rank-2"}, "outputs": [{"text": "skip"}]},
            {"metadata": {"client_tag": "rank-1"}, "outputs": [{"text": "keep"}]},
        ]
    }

    def fake_post(url, json, timeout, stream, headers):
        assert headers["X-VLLM-Client-Tag"] == "rank-1"
        assert json.get("client_tag") == "rank-1"
        return R(200, payload)

    monkeypatch.setattr(VP.requests, "post", fake_post)
    texts, meta, _ = VP.safe_generate(
        prompts=["p"], stream=False, client_tag="rank-1", return_logprobs=False
    )
    assert texts == [["keep"]]
    assert meta is None


def test_safe_generate_no_matching_client_tag(monkeypatch):
    payload = {
        "results": [
            {"metadata": {"client_tag": "rank-2"}, "outputs": [{"text": "skip"}]},
        ]
    }

    def fake_post(url, json, timeout, stream, headers):
        return R(200, payload)

    monkeypatch.setattr(VP.requests, "post", fake_post)
    monkeypatch.setattr(VP.time, "sleep", lambda *_args, **_kwargs: None)
    with pytest.raises(VP.GenerationServiceError):
        VP.safe_generate(
            prompts=["p"],
            stream=False,
            client_tag="rank-1",
            max_retries=1,
            backoff=0.0,
        )


def test_filter_response_preserves_untagged_entries():
    data = {"results": [{"outputs": [{"text": "loose"}]}]}
    filtered = VP._filter_response_for_client_tag(data, "rank-9")
    assert filtered["results"][0]["outputs"][0]["text"] == "loose"


def test_filter_response_filters_groups_and_outputs():
    data = {
        "results": [
            {
                "metadata": {"client_tag": "rank-1"},
                "outputs": [
                    {"metadata": {"client_tag": "rank-9"}, "text": "drop"},
                    {"metadata": {"client_tag": "rank-1"}, "text": "keep"},
                ],
            },
            {
                "metadata": {"client_tag": "rank-2"},
                "outputs": [{"text": "skip"}],
            },
        ]
    }
    filtered = VP._filter_response_for_client_tag(data, "rank-1")
    assert len(filtered["results"]) == 1
    outputs = filtered["results"][0]["outputs"]
    assert len(outputs) == 1 and outputs[0]["text"] == "keep"


def test_filter_response_raises_when_all_outputs_removed():
    data = {
        "results": [
            {
                "metadata": {"client_tag": "rank-1"},
                "outputs": [{"metadata": {"client_tag": "rank-2"}, "text": "drop"}],
            }
        ]
    }
    with pytest.raises(RuntimeError):
        VP._filter_response_for_client_tag(data, "rank-1")


def test_tokenizer_protocol_raises():
    class BareTokenizer(VP.TokenizerLike):
        pass

    tok = BareTokenizer()
    with pytest.raises(NotImplementedError):
        tok.decode([1, 2, 3])


def test_safe_request_retries(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise VP.requests.ConnectionError("boom")
        return R(200, {"ok": True})

    monkeypatch.setattr(VP.requests, "get", fake_get)
    monkeypatch.setattr(VP.time, "sleep", lambda *_args, **_kwargs: None)
    out = VP.safe_request("http://retry.test/health", max_retries=3, backoff=0.01)
    assert out == {"ok": True}
    assert len(calls) == 3


def test_safe_request_http_error(monkeypatch):
    def fake_get(url, timeout):
        return R(500, {"err": "nope"})

    monkeypatch.setattr(VP.requests, "get", fake_get)
    with pytest.raises(RuntimeError):
        VP.safe_request("http://fail/health", max_retries=1)


def test_safe_request_exhausts_connection_retries(monkeypatch):
    calls = 0

    def fake_get(url, timeout):
        nonlocal calls
        calls += 1
        raise VP.requests.ConnectionError("down")

    monkeypatch.setattr(VP.requests, "get", fake_get)
    slept = {}

    def _sleep(*_args, **_kwargs):
        slept["called"] = True

    monkeypatch.setattr(VP.time, "sleep", _sleep)
    with pytest.raises(VP.requests.ConnectionError):
        VP.safe_request("http://fail/health", max_retries=2, backoff=0.0)
    assert calls == 2
    assert slept.get("called") is True


def test_clean_logprob_seq_variants():
    assert VP._clean_logprob_seq(None) is None
    assert VP._clean_logprob_seq({"token_logprobs": [-1.0, None]}) == [-1.0]
    assert VP._clean_logprob_seq({"logprobs": [-0.1, -0.2]}) == [-0.1, -0.2]
    assert VP._clean_logprob_seq([None]) is None
    assert VP._clean_logprob_seq("junk") is None


def test_infer_token_count_paths():
    entry_seq = {"text": "one two three"}
    assert VP._infer_token_count(entry_seq, [0.1, 0.2]) == 2
    entry_logprobs = {"output_token_logprobs": [-0.1, -0.2, -0.3]}
    assert VP._infer_token_count(entry_logprobs, None) == 3
    entry_text = {"text": "hello world"}
    assert VP._infer_token_count(entry_text, None) == 2
    assert VP._infer_token_count({}, None) == 1


def test_extract_logprob_info_variants():
    assert VP._extract_logprob_info("not a dict") is None
    info = VP._extract_logprob_info(
        {"logprobs": {"token_logprobs": [-0.1, None, -0.2]}}
    )
    assert info is not None
    assert info.logprob_sum == pytest.approx(-0.3)
    assert info.token_count == 2
    assert info.token_logprobs == [-0.1, -0.2]


def test_build_vllm_headers_honors_env(monkeypatch):
    monkeypatch.setenv("VLLM_GROUP_REQUEST_ID", "fixed")
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    monkeypatch.setenv("VLLM_EXTRA_HEADERS", '{"X-Extra":"yep"}')
    headers = VP._build_vllm_headers(["foo"])
    assert headers["X-VLLM-Group-Request-ID"] == "fixed"
    assert headers["Authorization"] == "Bearer secret"
    assert headers["X-Extra"] == "yep"
    monkeypatch.delenv("VLLM_GROUP_REQUEST_ID")
    monkeypatch.delenv("VLLM_EXTRA_HEADERS")


def test_build_vllm_headers_hashes_prompts(monkeypatch):
    monkeypatch.delenv("VLLM_GROUP_REQUEST_ID", raising=False)
    headers = VP._build_vllm_headers(["a", "b"])
    assert headers["X-VLLM-Group-Request-ID"]
    assert len(headers["X-VLLM-Group-Request-ID"]) == 64


def test_extract_logprob_info_falls_back_to_output_token_logprobs():
    entry = {"output_token_logprobs": [-1.0, -2.0, -3.0]}
    info = VP._extract_logprob_info(entry)
    assert info is not None
    assert info.token_count == 3
    assert info.logprob_sum == pytest.approx(-6.0)


def test_extract_logprob_info_returns_none_when_missing():
    assert VP._extract_logprob_info({"text": "no logs"}) is None
    assert (
        VP._extract_logprob_info({"logprobs": {"token_logprobs": [None, None]}}) is None
    )


def test_extract_logprob_info_uses_metadata_block():
    entry = {
        "text": "x",
        "metadata": {"logprob_sum": -1.5, "token_count": 3, "token_logprobs": [-0.5] * 3},
    }
    info = VP._extract_logprob_info(entry)
    assert info is not None
    assert info.logprob_sum == pytest.approx(-1.5)
    assert info.token_count == 3
    assert info.token_logprobs == [-0.5, -0.5, -0.5]


def test_extract_logprob_info_infers_metadata_count():
    entry = {"text": "two tokens", "metadata": {"logprob_sum": -2.0}}
    info = VP._extract_logprob_info(entry)
    assert info is not None
    assert info.logprob_sum == pytest.approx(-2.0)
    assert info.token_count == 2


def test_parse_nonstream_json_choices_with_logs():
    data = {"choices": [{"text": "x", "logprobs": {"token_logprobs": [-0.5]}}]}
    texts, meta = VP._parse_nonstream_json(data, want_logprobs=True)
    assert texts == [["x"]]
    assert meta and meta[0][0].logprob_sum == -0.5


def test_parse_nonstream_json_results_variants(monkeypatch):
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(map(str, ids))

    data = {
        "results": [
            {"text": "direct"},
            {"completion_ids": [[1, 2, 3]]},
            "raw",
        ]
    }
    texts, meta = VP._parse_nonstream_json(data, tokenizer=Tok(), want_logprobs=True)
    assert texts == [["direct"], ["123"], ["raw"]]
    assert meta == [[], [], []]


def test_parse_nonstream_json_text_list():
    texts, meta = VP._parse_nonstream_json({"text": ["a", "b"]})
    assert texts == [["a"], ["b"]]
    assert meta is None


def test_parse_nonstream_json_token_ids_without_tokenizer():
    with pytest.raises(RuntimeError):
        VP._parse_nonstream_json({"completion_ids": [[1], [2]]})


def test_parse_nonstream_json_unknown():
    with pytest.raises(RuntimeError):
        VP._parse_nonstream_json({"unexpected": True})


def test_safe_generate_payload_and_request_id(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout, stream, headers):
        captured["payload"] = json
        return R(200, {"choices": [{"text": "ok"}]})

    monkeypatch.setattr(VP.requests, "post", fake_post)
    texts, meta, _ = VP.safe_generate(
        prompts=["p"],
        top_k=5,
        best_of=2,
        frequency_penalty=0.2,
        presence_penalty=0.4,
        stop=["</s>"],
        logit_bias={"42": -1.0},
        guided_json="{}",
        guided_regex="foo.*",
        request_id_prefix="pref",
        max_retries=1,
    )
    payload = captured["payload"]
    assert payload["top_k"] == 5
    assert payload["sampling_params"]["top_k"] == 5
    assert payload["best_of"] == 2
    assert payload["frequency_penalty"] == 0.2
    assert payload["presence_penalty"] == 0.4
    assert payload["stop"] == ["</s>"]
    assert payload["logit_bias"] == {"42": -1.0}
    assert payload["guided_json"] == "{}"
    assert payload["guided_regex"] == "foo.*"
    assert payload["request_id"].startswith("pref-")
    assert texts == [["ok"]]
    assert meta is None


def test_safe_generate_retries_then_fails(monkeypatch):
    attempts = []

    def fake_post(url, json, timeout, stream, headers):
        attempts.append(1)
        return R(500, {"err": "nope"})

    monkeypatch.setattr(VP.requests, "post", fake_post)
    monkeypatch.setattr(VP.time, "sleep", lambda *_args, **_kwargs: None)
    with pytest.raises(VP.GenerationServiceError) as exc_info:
        VP.safe_generate(prompts=["p"], max_retries=2, backoff=0.0)
    assert len(attempts) == 2
    payload = exc_info.value.payload.to_dict()
    assert payload["prompt_count"] == 1
    assert payload["status_code"] == 500
    assert payload["attempt"] == 2
    extras = payload.get("extra")
    assert extras["backoff_initial"] == pytest.approx(0.0)
    assert extras["backoff_multiplier"] == pytest.approx(2.0)
    assert extras["elapsed_ms"] >= 0.0


def test_safe_generate_failure_carries_metadata(monkeypatch):
    def fake_post(url, json, timeout, stream, headers):
        return R(503, {"err": "nope"})

    monkeypatch.setattr(VP.requests, "post", fake_post)
    monkeypatch.setattr(VP.time, "sleep", lambda *_a, **_k: None)
    with pytest.raises(VP.GenerationServiceError) as exc_info:
        VP.safe_generate(
            prompts=["p"],
            max_retries=1,
            metadata={"dataset": "hf/train", "model_id": "org/model"},
        )
    payload = exc_info.value.payload.to_dict()
    assert payload["extra"]["dataset"] == "hf/train"
    assert payload["extra"]["model_id"] == "org/model"


def test_collect_stream_texts_handles_blanks():
    resp = R(
        lines=[
            b"",
            b'{"prompt_index": 1, "text": "hi"}',
            b'{"prompt_index": 9, "text": "skip"}',
        ]
    )
    out = VP._collect_stream_texts(resp, num_prompts=2)
    assert out == [[""], ["hi"]]


def test_build_headers_env(monkeypatch):
    monkeypatch.setenv("VLLM_GROUP_REQUEST_ID", "group-123")
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    monkeypatch.setenv("VLLM_EXTRA_HEADERS", json.dumps({"X-Test": "1"}))
    headers = VP._build_vllm_headers(["a"])
    assert headers["X-VLLM-Group-Request-ID"] == "group-123"
    assert headers["Authorization"] == "Bearer secret"
    assert headers["X-Test"] == "1"


def test_build_headers_ignores_bad_extra(monkeypatch):
    monkeypatch.delenv("VLLM_GROUP_REQUEST_ID", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_EXTRA_HEADERS", "{not-json")
    headers = VP._build_vllm_headers(["foo"])
    assert headers["X-VLLM-Group-Request-ID"]
    assert len(headers) == 1  # only group id when extras fail to parse
