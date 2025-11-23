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

import utils.vllm_patch as VP


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
