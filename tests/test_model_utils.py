from types import SimpleNamespace

import utils.model_utils as MU
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_get_tokenizer_applies_chat_template():
    model_args = SimpleNamespace(model_name_or_path="x", model_revision="main", trust_remote_code=True)
    training_args = SimpleNamespace(chat_template="SYS: {{ system }}\nUSER: {{ input }}")
    tok = MU.get_tokenizer(model_args, training_args)
    assert getattr(tok, "chat_template", None) is not None


def test_get_model_passes_expected_kwargs(monkeypatch):
    # Force quantization_config=None to avoid device_map
    monkeypatch.setattr(MU, "get_quantization_config", lambda *a, **k: None)
    seen = {}
    def fake_from_pretrained(name, **kwargs):
        seen.update({"name": name, **kwargs})
        return SimpleNamespace(config=SimpleNamespace())
    monkeypatch.setattr(MU.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)

    model_args = SimpleNamespace(
        model_name_or_path="org/model",
        model_revision="dev",
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype="float16",
    )
    training_args = SimpleNamespace(gradient_checkpointing=True)
    MU.get_model(model_args, training_args)

    assert seen["name"] == "org/model"
    assert seen["revision"] == "dev"
    assert seen["trust_remote_code"] is True
    assert seen["attn_implementation"] == "sdpa"
    assert seen["use_cache"] is False
