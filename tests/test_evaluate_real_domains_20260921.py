"""Checks for scientifically consequential evaluator accounting invariants."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = spec_from_file_location("real_eval", Path(__file__).parents[1] / "ops/evaluate_real_domains_20260921.py")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)


def rows(keys):
    return [{"accepted": key is not None, "canonical_key": key, "hard_violations": []} for key in keys]


def test_correctness_cannot_stand_in_for_diversity():
    collapsed = mod.mode_metrics(rows(["a"] * 32))
    diverse = mod.mode_metrics(rows(["a"] * 16 + ["b"] * 16))
    assert collapsed["pass_at_k"] == diverse["pass_at_k"]
    assert collapsed["pcmd"] == 0
    assert diverse["pcmd"] == pytest.approx(16 / 31)
    assert collapsed["expected_distinct_valid_modes_at_k"]["32"] == 1
    assert diverse["expected_distinct_valid_modes_at_k"]["32"] == 2


def test_invalid_answers_remain_in_unconditional_denominator():
    value = mod.mode_metrics(rows(["a", "b"] + [None] * 30))
    assert value["accuracy"] == 2 / 32
    assert value["expected_distinct_valid_modes_at_k"]["1"] == 2 / 32
    assert value["pcmd"] is None
    assert value["pcmd_eligible"] is False


def test_missing_or_duplicate_requests_are_not_a_complete_run():
    tasks = [SimpleNamespace(task_id="q", family="qa", split="dev", metadata={})]
    complete = [{"task_id": "q", "sample_index": i, **row} for i, row in enumerate(rows(["a", None]))]
    assert mod.summarize(tasks, complete, 2)["summary"]["samples"] == 2
    with pytest.raises(ValueError, match="request identities"):
        mod.summarize(tasks, [complete[0], complete[0]], 2)
    with pytest.raises(ValueError, match="request identities"):
        mod.summarize(tasks, complete[:1], 2)


def test_known_gold_support_is_checked():
    task = SimpleNamespace(task_id="q", family="qa", split="dev", metadata={"known_mode_count": 1})
    attempts = [{"task_id": "q", "sample_index": i, **row} for i, row in enumerate(rows(["a", "b"]))]
    with pytest.raises(ValueError, match="known labeled support"):
        mod.summarize([task], attempts, 2)


@pytest.mark.parametrize("decision", [
    {"accepted": True, "canonical_key": "a", "hard_violations": ["unstable"], "receipt": {}},
    {"accepted": True, "canonical_key": None, "hard_violations": [], "receipt": {}},
    {"accepted": False, "canonical_key": "a", "hard_violations": [], "receipt": {}},
])
def test_invalid_verifier_contract_cannot_receive_reward(decision):
    with pytest.raises(ValueError):
        mod.validate_decision(decision)


def test_worker_exception_is_a_hard_failure_not_a_wrong_answer():
    def fail(text):
        raise RuntimeError("infrastructure failure")
    task = SimpleNamespace(verify=fail)
    response = dict(task_id="q", sample_index=0, request_seed=1, prompt_sha256="p", text_sha256="t", token_count=1, finish_reason="stop", text="A")
    decision = mod.verify_one(task, response)
    assert not decision["accepted"]
    assert "infrastructure failure" in decision["hard_violations"][0]


def test_vllm_support_matches_tokenizer_action_space():
    import torch
    logits = torch.tensor([1., 2., 3., 4.])
    masked = mod.MaskUnrenderableTokens(2)([], logits)
    assert masked[:2].tolist() == [1., 2.]
    assert torch.isneginf(masked[2:]).all()


def test_endpoint_rejects_wrong_base_or_modified_adapter(tmp_path):
    import json
    checkpoint = tmp_path / "checkpoint"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    model = tmp_path / "base"
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(model)}))
    (adapter / "adapter_model.safetensors").write_bytes(b"local weights")
    (checkpoint / "bank.json").write_text("{}")
    (checkpoint / "training.pt").write_bytes(b"local optimizer")
    seal = {"arm": "remax", "completed_updates": 32, "config_sha256": "config",
            "bank_sha256": mod.sha256(checkpoint / "bank.json"),
            "training_state_sha256": mod.sha256(checkpoint / "training.pt"),
            "adapter_files": {p.name: mod.sha256(p) for p in adapter.iterdir()}}
    (checkpoint / "complete.json").write_text(json.dumps(seal))
    config = {"lora_path": str(adapter), "lora_arm": "remax", "lora_completed_updates": 32,
              "lora_checkpoint_config_sha256": "config"}
    assert mod.verify_lora_checkpoint(config, model)["seal"] == seal
    with pytest.raises(ValueError, match="different base"):
        mod.verify_lora_checkpoint(config, tmp_path / "wrong_model")
    (adapter / "adapter_model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="adapter seal"):
        mod.verify_lora_checkpoint(config, model)


def test_vllm_cachetools_compatibility_preserves_touch_and_eviction(monkeypatch):
    import cachetools
    monkeypatch.delattr(cachetools.LRUCache, "_LRUCache__update", raising=False)
    monkeypatch.setattr(cachetools.LRUCache, "_LRUCache__touch", cachetools.LRUCache._LRUCache__touch)
    # Track the new attribute through pytest's monkeypatch undo stack.
    monkeypatch.setattr(cachetools.LRUCache, "_LRUCache__update", None, raising=False)
    monkeypatch.delattr(cachetools.LRUCache, "_LRUCache__update")
    result = mod.configure_vllm_cachetools_compatibility("0.8.4")
    assert result["action"] == "process_local_private_update_alias_to_touch"
    cache = cachetools.LRUCache(maxsize=2)
    cache[1], cache[2] = "one", "two"
    cache._LRUCache__update(1)
    cache[3] = "three"
    assert set(cache) == {1, 3}
    assert cache[1] == "one" and cache[3] == "three"
    assert mod.configure_vllm_cachetools_compatibility("0.8.4")["action"] == "not_required"


def test_vllm_cachetools_compatibility_rejects_unknown_lru_shape(monkeypatch):
    import cachetools
    monkeypatch.delattr(cachetools.LRUCache, "_LRUCache__update", raising=False)
    monkeypatch.delattr(cachetools.LRUCache, "_LRUCache__touch", raising=False)
    with pytest.raises(RuntimeError, match="unsupported cachetools"):
        mod.configure_vllm_cachetools_compatibility("0.8.4")
