"""Unit tests for helper utilities (prompt templating and softmax)."""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest

torch_stub = types.ModuleType("torch")
torch_stub.__spec__ = SimpleNamespace()
import math


class _Tensor(list):
    def __init__(self, data):
        super().__init__(float(v) for v in data)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return _Tensor([a - other for a in self])
        return _Tensor([a - b for a, b in zip(self, other)])

    def __truediv__(self, other):
        return _Tensor([a / other for a in self])

    def exp(self):
        return _Tensor([math.exp(a) for a in self])

    def clamp(self, min=None, max=None):
        return _Tensor(
            [
                (
                    a
                    if (min is None or a >= min) and (max is None or a <= max)
                    else (min if min is not None and a < min else max)
                )
                for a in self
            ]
        )

    def sum(self):
        return sum(self)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def max(self):
        return max(self)


torch_stub.tensor = lambda data, **kwargs: _Tensor(data)
torch_stub.Tensor = _Tensor
torch_stub.float32 = float
torch_stub.softmax = lambda tensor, dim=0: _Tensor(
    [val / tensor.sum() for val in tensor]
)
torch_stub.mul = lambda tensor, scalar: _Tensor([val * scalar for val in tensor])
torch_stub.add = lambda tensor, scalar: _Tensor([val + scalar for val in tensor])
setattr(
    _Tensor, "__mul__", lambda self, scalar: _Tensor([val * scalar for val in self])
)
setattr(
    _Tensor, "__rmul__", lambda self, scalar: _Tensor([val * scalar for val in self])
)
setattr(
    _Tensor, "__add__", lambda self, scalar: _Tensor([val + scalar for val in self])
)
setattr(_Tensor, "tolist", lambda self: list(self))
sys.modules["torch"] = torch_stub
torch_utils = types.ModuleType("torch.utils")
sys.modules["torch.utils"] = torch_utils
torch_utils_data = types.ModuleType("torch.utils.data")
torch_utils_data.DataLoader = type("DataLoader", (object,), {})
torch_utils_data.Sampler = type("Sampler", (object,), {})
sys.modules["torch.utils.data"] = torch_utils_data
torch_stub.utils = SimpleNamespace(data=torch_utils_data)
torch_optim = types.ModuleType("torch.optim")
torch_optim.Optimizer = type("Optimizer", (object,), {})
sys.modules["torch.optim"] = torch_optim
torch_nn = types.ModuleType("torch.nn")
torch_nn.functional = types.ModuleType("torch.nn.functional")
torch_nn.functional.log_softmax = lambda *args, **kwargs: None
sys.modules["torch.nn"] = torch_nn
sys.modules["torch.nn.functional"] = torch_nn.functional
accelerate_stub = types.ModuleType("accelerate")
accelerate_stub.Accelerator = type("Accelerator", (object,), {})
accelerate_stub.state = SimpleNamespace(
    DistributedType=SimpleNamespace(DEEPSPEED="deepspeed"),
    deepspeed_plugin=None,
)
accelerate_state = types.ModuleType("accelerate.state")
accelerate_state.DistributedType = SimpleNamespace(DEEPSPEED="deepspeed")
accelerate_utils = types.ModuleType("accelerate.utils")
accelerate_utils.is_peft_model = lambda *_args, **_kwargs: False
accelerate_utils.DeepSpeedPlugin = type("DeepSpeedPlugin", (), {})
sys.modules["accelerate"] = accelerate_stub
sys.modules["accelerate.state"] = accelerate_state
sys.modules["accelerate.utils"] = accelerate_utils
transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedModel = type("PreTrainedModel", (), {})
transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
sys.modules["transformers"] = transformers_stub
trl_stub = types.ModuleType("trl")
trl_stub.ScriptArguments = type("ScriptArguments", (object,), {})
trl_stub.GRPOConfig = type("GRPOConfig", (object,), {})
sys.modules["trl"] = trl_stub

# flake8: noqa: E402
from training.run_helpers import (
    MaxEntOptions,
    _group_softmax,
    _to_prompt,
    _truncate_prompt,
    GenerationSamplingConfig,
    VLLMClientConfig,
)


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"{messages[0]['content']}|{messages[1]['content']}"


def test_to_prompt_prefers_chat_template(monkeypatch):
    example = {"instruction": "Hello", "answer": "42"}
    out = _to_prompt(example, _Tokenizer(), "instruction", "SYS", char_limit=100)
    assert "SYS" in out["prompt"]
    assert out["answer"] == "42"


def test_to_prompt_fallback_when_chat_template_missing():
    class _BrokenTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise AttributeError

    example = {"instruction": "Hello"}
    out = _to_prompt(example, _BrokenTokenizer(), "instruction", None, char_limit=64)
    assert out["prompt"].startswith("USER:")
    assert out["prompt"].endswith("ASSISTANT:")


def test_group_softmax_normalizes_values(monkeypatch):
    values = [0.0, 1.0, 2.0]
    from training import run_helpers

    torch_like = SimpleNamespace(
        float32=float,
        tensor=lambda data, dtype=None: _Tensor(data),
        softmax=lambda tensor, dim=0: _Tensor([val / sum(tensor) for val in tensor]),
    )
    original_dep = run_helpers._require_dependency
    monkeypatch.setattr(
        run_helpers,
        "_require_dependency",
        lambda name, hint: torch_like if name == "torch" else original_dep(name, hint),
    )
    probs = _group_softmax(values, temperature=1.0, eps=1e-6)
    assert pytest.approx(sum(probs), rel=1e-6) == 1.0
    assert all(p >= 0 for p in probs)


def test_maxent_options_env_overrides(monkeypatch):
    monkeypatch.setenv("MAXENT_TAU", "0.5")
    monkeypatch.setenv("MAXENT_Q_TEMPERATURE", "0.7")
    monkeypatch.setenv("MAXENT_Q_EPS", "0.123")
    monkeypatch.setenv("MAXENT_LENGTH_NORM_REF", "0")
    opts = MaxEntOptions()
    assert opts.tau == pytest.approx(0.5)
    assert opts.q_temperature == pytest.approx(0.7)
    assert opts.q_epsilon == pytest.approx(0.123)
    assert opts.length_normalize_ref is False


def test_truncate_prompt_warns_once(monkeypatch, caplog):
    from training import run_helpers

    caplog.set_level("WARNING")
    run_helpers._TRUNC_STATE["warned"] = False
    truncated = _truncate_prompt("x" * 10, char_limit=5)
    assert truncated == "xxxxx"
    assert "Prompt length exceeded" in caplog.text

    caplog.clear()
    truncated = _truncate_prompt("y" * 10, char_limit=5)
    assert truncated == "yyyyy"
    assert "Prompt length exceeded" not in caplog.text


def test_prompt_char_limit_from_tokens_respects_env(monkeypatch):
    from importlib import reload
    from training import run_helpers

    monkeypatch.setenv("MAX_PROMPT_CHARS", "99")
    reload(run_helpers)
    limit = run_helpers._prompt_char_limit_from_tokens(10)
    assert limit == max(99, 40)
    monkeypatch.delenv("MAX_PROMPT_CHARS", raising=False)
    reload(run_helpers)
    assert (
        run_helpers._prompt_char_limit_from_tokens(0) == run_helpers.PROMPT_CHAR_LIMIT
    )


def test_generation_sampling_config_proxies_vllm_fields():
    vllm_cfg = VLLMClientConfig(
        url="http://vllm",
        rounds_cfg=3,
        retry_sleep=0.5,
        backfill_local=True,
        request_logprobs=True,
        best_of=4,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        top_k=5,
        stop_sequences=["</s>"],
        timeout=30.0,
        max_retries=2,
        backoff=1.5,
        guided_json=None,
        guided_regex=None,
        logit_bias={"0": -2.0},
        request_id_prefix="eval",
        sync_weights=True,
    )
    cfg = GenerationSamplingConfig(
        max_prompt_len=128,
        max_completion_len=64,
        gen_temperature=0.7,
        gen_top_p=0.95,
        use_vllm=True,
        vllm=vllm_cfg,
    )
    assert cfg.vllm_url == "http://vllm"
    assert cfg.vllm_rounds_cfg == 3
    assert cfg.vllm_retry_sleep == 0.5
    assert cfg.vllm_backfill_local is True
    assert cfg.vllm_request_logprobs is True
    assert cfg.vllm_best_of == 4
    assert cfg.vllm_frequency_penalty == pytest.approx(0.2)
    assert cfg.vllm_presence_penalty == pytest.approx(0.1)
    assert cfg.vllm_top_k == 5
    assert cfg.vllm_stop_sequences == ["</s>"]


def test_maybe_init_wandb_run_honors_report_to(monkeypatch):
    from training import run_helpers

    accelerator = SimpleNamespace(is_main_process=True)
    training_args = SimpleNamespace(report_to=["wandb"], run_name="demo")
    init_called = {}

    monkeypatch.setattr(
        run_helpers,
        "init_wandb_training",
        lambda *_a, **_k: init_called.setdefault("called", True),
    )

    class _FakeRun:
        pass

    class _FakeWandb:
        def __init__(self):
            self.kwargs = None

        def init(self, **kwargs):
            self.kwargs = kwargs
            return _FakeRun()

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        run_helpers,
        "_optional_dependency",
        lambda name: fake_wandb if name == "wandb" else None,
    )
    run = run_helpers._maybe_init_wandb_run(accelerator, training_args, {"foo": "bar"})
    assert isinstance(run, _FakeRun)
    assert fake_wandb.kwargs["config"] == {"foo": "bar"}
    assert init_called["called"] is True


def test_maybe_init_wandb_run_sets_offline_for_non_main(monkeypatch):
    from training import run_helpers

    accelerator = SimpleNamespace(is_main_process=False)
    training_args = SimpleNamespace(report_to=["wandb"])
    monkeypatch.setattr(run_helpers, "init_wandb_training", lambda *_a, **_k: None)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    result = run_helpers._maybe_init_wandb_run(accelerator, training_args, {})
    assert result is None
    assert os.environ["WANDB_MODE"] == "offline"


def test_maybe_create_deepspeed_plugin_reads_config(monkeypatch, tmp_path):
    from training import run_helpers

    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    ds_yaml = tmp_path / "ds.yaml"
    ds_yaml.write_text(
        """
deepspeed_config:
  zero_stage: 2
  offload_param_device: cpu
  zero3_save_16bit_model: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", str(ds_yaml))

    class _FakeDSPlugin:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_mod = SimpleNamespace(DeepSpeedPlugin=_FakeDSPlugin)
    monkeypatch.setattr(
        run_helpers,
        "_require_dependency",
        lambda name, hint: (
            fake_mod
            if name == "accelerate.utils"
            else run_helpers._require_dependency(name, hint)
        ),
    )
    plugin = run_helpers._maybe_create_deepspeed_plugin()
    assert isinstance(plugin, _FakeDSPlugin)
    assert plugin.kwargs["zero_stage"] == 2
    assert plugin.kwargs["offload_param_device"] == "cpu"
    assert plugin.kwargs["zero3_save_16bit_model"] is True
