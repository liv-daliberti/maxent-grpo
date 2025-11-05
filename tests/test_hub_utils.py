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

import utils.hub as H


def test_param_count_parsing_single_and_product():
    assert H.get_param_count_from_repo_id("foo-7b") == 7_000_000_000
    assert H.get_param_count_from_repo_id("bar-42m") == 42_000_000
    # 2x3.5b -> 7.0b
    assert H.get_param_count_from_repo_id("baz-2x3.5b") == 7_000_000_000


def test_gpu_count_divisible_by_heads_and_64(monkeypatch):
    # Head count 42 should result in 2 GPUs (8->7->6->5->4->3->2)
    class _Cfg:
        def __init__(self, n):
            self.num_attention_heads = n

    def fake_from_pretrained(*a, **k):
        return _Cfg(42)

    monkeypatch.setattr(H.AutoConfig, "from_pretrained", fake_from_pretrained)
    assert H.get_gpu_count_for_vllm("any") == 2
