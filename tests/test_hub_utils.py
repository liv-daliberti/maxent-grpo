from types import SimpleNamespace

import src.utils.hub as H


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

