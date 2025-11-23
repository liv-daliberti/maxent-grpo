"""Unit tests for helper utilities (prompt templating and softmax)."""

from __future__ import annotations

import sys
import types
from importlib import import_module, reload
from types import SimpleNamespace

import pytest


@pytest.fixture
def run_helpers(monkeypatch, training_stubs):
    """Reload training.run_helpers with lightweight dependency stubs installed."""
    # training_stubs fixture installs torch/accelerate/transformers shims
    import training  # noqa: F401
    module = import_module("training.run_helpers")
    return reload(module)


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"{messages[0]['content']}|{messages[1]['content']}"


def test_to_prompt_prefers_chat_template(run_helpers):
    example = {"instruction": "Hello", "answer": "42"}
    out = run_helpers._to_prompt(
        example, _Tokenizer(), "instruction", "SYS", char_limit=100
    )
    assert "SYS" in out["prompt"]
    assert out["answer"] == "42"


def test_to_prompt_fallback_when_chat_template_missing(run_helpers):
    class _BrokenTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise AttributeError

    example = {"instruction": "Hello"}
    out = run_helpers._to_prompt(
        example, _BrokenTokenizer(), "instruction", None, char_limit=64
    )
    assert out["prompt"].startswith("USER:")
    assert out["prompt"].endswith("ASSISTANT:")


def test_group_softmax_normalizes_values(monkeypatch, run_helpers):
    values = [0.0, 1.0, 2.0]

    class _Tensor(list):
        def __init__(self, data):
            super().__init__(float(v) for v in data)

        def __truediv__(self, other):
            return _Tensor([a / other for a in self])

        def __sub__(self, other):
            return _Tensor([a - other for a in self])

        def max(self):
            return max(self)

        def __mul__(self, scalar):
            return _Tensor([a * scalar for a in self])

        __rmul__ = __mul__

        def __add__(self, scalar):
            return _Tensor([a + scalar for a in self])

        def sum(self):
            return sum(self)

        def tolist(self):
            return list(self)

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
    probs = run_helpers._group_softmax(values, temperature=1.0, eps=1e-6)
    assert pytest.approx(sum(probs), rel=1e-6) == 1.0
    assert all(p >= 0 for p in probs)


def test_group_softmax_empty_returns_empty(run_helpers):
    assert run_helpers._group_softmax([]) == []


def test_maxent_options_env_overrides(monkeypatch, run_helpers):
    monkeypatch.setenv("MAXENT_TAU", "0.5")
    monkeypatch.setenv("MAXENT_Q_TEMPERATURE", "0.7")
    monkeypatch.setenv("MAXENT_Q_EPS", "0.123")
    monkeypatch.setenv("MAXENT_LENGTH_NORM_REF", "0")
    opts = run_helpers.MaxEntOptions()
    assert opts.tau == pytest.approx(0.5)
    assert opts.q_temperature == pytest.approx(0.7)
    assert opts.q_epsilon == pytest.approx(0.123)
    assert opts.length_normalize_ref is False


def test_truncate_prompt_warns_once(monkeypatch, caplog, run_helpers):
    caplog.set_level("WARNING")
    run_helpers._TRUNC_STATE["warned"] = False
    truncated = run_helpers._truncate_prompt("x" * 10, char_limit=5)
    assert truncated == "xxxxx"
    assert "Prompt length exceeded" in caplog.text

    caplog.clear()
    truncated = run_helpers._truncate_prompt("y" * 10, char_limit=5)
    assert truncated == "yyyyy"
    assert "Prompt length exceeded" not in caplog.text


def test_prompt_char_limit_from_tokens_respects_env(monkeypatch, run_helpers):
    monkeypatch.setenv("MAX_PROMPT_CHARS", "99")
    module = reload(run_helpers)
    limit = module._prompt_char_limit_from_tokens(10)
    assert limit == max(99, 40)
    monkeypatch.delenv("MAX_PROMPT_CHARS", raising=False)
    module = reload(run_helpers)
    assert module._prompt_char_limit_from_tokens(0) == module.PROMPT_CHAR_LIMIT


def test_truncate_prompt_shared_with_baseline(monkeypatch, caplog):
    caplog.set_level("WARNING")
    import training.run_helpers as run_helpers
    import pipelines.training.baseline as baseline

    run_helpers._TRUNC_STATE["warned"] = False
    monkeypatch.setattr(run_helpers, "PROMPT_CHAR_LIMIT", 5)

    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[-1]["content"]

    out = baseline._to_prompt(
        {"prompt": "z" * 10, "answer": "42"}, _Tok(), "prompt", None
    )
    assert out["prompt"] == "zzzzz"
    assert "Prompt length exceeded" in caplog.text
    caplog.clear()

    # Second truncation through the shared helper should not warn again.
    _ = run_helpers._truncate_prompt("w" * 10)
    assert "Prompt length exceeded" not in caplog.text

    run_helpers._TRUNC_STATE["warned"] = False


def test_generation_sampling_config_proxies_vllm_fields(run_helpers):
    vllm_cfg = run_helpers.VLLMClientConfig(
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
    cfg = run_helpers.GenerationSamplingConfig(
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
    assert cfg.vllm_timeout == 30.0
    assert cfg.vllm_max_retries == 2
    assert cfg.vllm_backoff == pytest.approx(1.5)
    assert cfg.vllm_guided_json is None
    assert cfg.vllm_guided_regex is None
    assert cfg.vllm_logit_bias == {"0": -2.0}
    assert cfg.vllm_request_id_prefix == "eval"
    assert cfg.vllm_sync_weights is True


def test_require_dependency_and_optional_dependency(monkeypatch, run_helpers):
    run_helpers._import_module.cache_clear()
    demo_mod = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "demo_mod", demo_mod)
    assert run_helpers._require_dependency("demo_mod", "hint") is demo_mod
    run_helpers._import_module.cache_clear()
    monkeypatch.delitem(sys.modules, "missing_mod_demo", raising=False)
    assert run_helpers._optional_dependency("missing_mod_demo") is None


def test_wandb_error_types_defaults_and_custom_error(monkeypatch, run_helpers):
    run_helpers._wandb_error_types.cache_clear()
    monkeypatch.delitem(sys.modules, "wandb.errors", raising=False)
    assert run_helpers._wandb_error_types() == (RuntimeError, ValueError)

    run_helpers._wandb_error_types.cache_clear()
    errors_mod = types.ModuleType("wandb.errors")

    class CustomError(RuntimeError):
        pass

    errors_mod.Error = CustomError
    monkeypatch.setitem(sys.modules, "wandb.errors", errors_mod)
    err_types = run_helpers._wandb_error_types()
    assert err_types[0] is CustomError
    assert RuntimeError in err_types and ValueError in err_types


def test_report_to_contains_handles_variants(run_helpers):
    assert run_helpers._report_to_contains(None, "wandb") is False
    assert run_helpers._report_to_contains("WandB", "wandb") is True
    assert run_helpers._report_to_contains(["mlflow", "wandb"], "WANDB") is True


def test_maybe_init_wandb_run_warns_without_package(monkeypatch, caplog, run_helpers):
    accelerator = SimpleNamespace(is_main_process=True)
    training_args = SimpleNamespace(report_to="wandb")
    init_called = {}
    monkeypatch.setattr(
        run_helpers,
        "init_wandb_training",
        lambda *_a, **_k: init_called.setdefault("called", True),
    )
    monkeypatch.setattr(run_helpers, "_optional_dependency", lambda *_a, **_k: None)
    caplog.set_level("WARNING")
    result = run_helpers._maybe_init_wandb_run(accelerator, training_args, {})
    assert result is None
    assert init_called["called"] is True
    assert "wandb package is not installed" in caplog.text


def test_maybe_init_wandb_run_applies_env_overrides(monkeypatch, run_helpers):
    accelerator = SimpleNamespace(is_main_process=True)
    training_args = SimpleNamespace(report_to=["wandb"], run_name=None)
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_RUN_GROUP", "group")

    class _FakeRun:
        pass

    class _FakeWandb:
        def __init__(self):
            self.kwargs = None

        def init(self, **kwargs):
            self.kwargs = kwargs
            return _FakeRun()

    wandb = _FakeWandb()
    monkeypatch.setattr(run_helpers, "init_wandb_training", lambda *_a, **_k: None)
    monkeypatch.setattr(run_helpers, "_optional_dependency", lambda *_a, **_k: wandb)
    run = run_helpers._maybe_init_wandb_run(accelerator, training_args, {"x": 1})
    assert isinstance(run, _FakeRun)
    assert wandb.kwargs["project"] == "proj"
    assert wandb.kwargs["entity"] == "ent"
    assert wandb.kwargs["group"] == "group"


def test_log_wandb_handles_first_and_error_runs(monkeypatch, caplog, run_helpers):
    run_helpers._FIRST_WANDB_LOGGED_RUNS.clear()
    caplog.set_level("INFO")
    run_helpers._log_wandb(None, {"a": 1}, step=1)

    class _Run:
        def __init__(self):
            self.logged = []
            self.id = "run1"

        def log(self, metrics, step):
            self.logged.append((metrics, step))

    good_run = _Run()
    run_helpers._log_wandb(good_run, {"x": 2}, step=3)
    assert good_run.logged == [({"x": 2}, 3)]
    assert "Logging first metrics" in caplog.text

    class _BadRun:
        def __init__(self):
            self.id = "run2"

        def log(self, *_a, **_k):
            raise ValueError("boom")

    caplog.set_level("WARNING")
    run_helpers._log_wandb(_BadRun(), {"y": 5}, step=4)
    assert "Failed to log metrics" in caplog.text


def test_maybe_create_deepspeed_plugin_returns_none_when_disabled(
    run_helpers, monkeypatch
):
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "false")
    assert run_helpers._maybe_create_deepspeed_plugin() is None


def test_maybe_create_deepspeed_plugin_reads_config(monkeypatch, tmp_path, run_helpers):
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
    orig_require = run_helpers._require_dependency
    monkeypatch.setattr(
        run_helpers,
        "_require_dependency",
        lambda name, hint: (
            fake_mod if name == "accelerate.utils" else orig_require(name, hint)
        ),
    )
    plugin = run_helpers._maybe_create_deepspeed_plugin()
    assert isinstance(plugin, _FakeDSPlugin)
    assert plugin.kwargs["zero_stage"] == 2
    assert plugin.kwargs["offload_param_device"] == "cpu"
    assert plugin.kwargs["zero3_save_16bit_model"] is True


def test_maybe_create_deepspeed_plugin_handles_yaml_errors(
    monkeypatch, tmp_path, run_helpers
):
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    cfg_path = tmp_path / "ds_err.yaml"
    cfg_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", str(cfg_path))

    def _raise(*_a, **_k):
        raise ValueError("bad yaml")

    monkeypatch.setattr(run_helpers.yaml, "safe_load", _raise)
    fake_mod = SimpleNamespace(
        DeepSpeedPlugin=lambda **kwargs: SimpleNamespace(kwargs=kwargs)
    )
    orig_require = run_helpers._require_dependency
    monkeypatch.setattr(
        run_helpers,
        "_require_dependency",
        lambda name, hint: (
            fake_mod if name == "accelerate.utils" else orig_require(name, hint)
        ),
    )
    plugin = run_helpers._maybe_create_deepspeed_plugin()
    assert isinstance(plugin, SimpleNamespace)
    assert plugin.kwargs["zero_stage"] == 3


def test_maybe_create_deepspeed_plugin_returns_none_when_no_config_values(
    monkeypatch, tmp_path, run_helpers
):
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    cfg_path = tmp_path / "ds_empty.yaml"
    cfg_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", str(cfg_path))
    fake_mod = SimpleNamespace(DeepSpeedPlugin=lambda **kwargs: kwargs)
    orig_require = run_helpers._require_dependency
    monkeypatch.setattr(
        run_helpers,
        "_require_dependency",
        lambda name, hint: (
            fake_mod if name == "accelerate.utils" else orig_require(name, hint)
        ),
    )
    monkeypatch.setattr(
        run_helpers.yaml,
        "safe_load",
        lambda *_a, **_k: {"deepspeed_config": {"zero_stage": None}},
    )
    assert run_helpers._maybe_create_deepspeed_plugin() is None


def test_require_deepspeed_wraps_import_errors(monkeypatch, run_helpers):
    def _raise(*_a, **_k):
        raise ImportError("x")

    monkeypatch.setattr(run_helpers, "_require_dependency", _raise)
    with pytest.raises(RuntimeError):
        run_helpers.require_deepspeed("ctx")


def test_get_trl_prepare_deepspeed(monkeypatch, run_helpers):
    monkeypatch.delitem(sys.modules, "trl.trainer.utils", raising=False)
    assert run_helpers.get_trl_prepare_deepspeed() is None

    utils_mod = types.ModuleType("trl.trainer.utils")

    def _prep():
        return "ok"

    utils_mod.prepare_deepspeed = _prep
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils_mod)
    assert run_helpers.get_trl_prepare_deepspeed() is _prep


def test_chat_tokenizer_protocol_methods_raise(run_helpers):
    class _Dummy(run_helpers.ChatTokenizer):
        pass

    dummy = _Dummy()
    with pytest.raises(NotImplementedError):
        dummy.apply_chat_template([])
    with pytest.raises(NotImplementedError):
        _ = dummy.eos_token_id
    with pytest.raises(NotImplementedError):
        dummy("x")


def test_prepare_labels_for_ce_masks_prompt_tokens(run_helpers):
    class _Tensor2D:
        def __init__(self, data):
            self.data = [list(row) for row in data]

        def clone(self):
            return _Tensor2D(self.data)

        def __getitem__(self, key):
            if isinstance(key, tuple):
                row_idx, slc = key
                return self.data[row_idx][slc]
            return self.data[key]

        def __setitem__(self, key, value):
            if isinstance(key, tuple):
                row_idx, slc = key
                row = self.data[row_idx]
                if isinstance(slc, slice):
                    start = slc.start or 0
                    stop = slc.stop if slc.stop is not None else len(row)
                    row[start:stop] = [value] * (stop - start)
                    self.data[row_idx] = row
            else:
                self.data[key] = value

    tensor = _Tensor2D([[1, 2, 3], [4, 5]])
    labels = run_helpers._prepare_labels_for_ce(tensor, [2, 1])
    assert labels.data[0][:2] == [-100, -100]
    assert labels.data[1][0] == -100


def test_batch_tokenize_pairs_returns_lengths(run_helpers):
    class _FakeTensor:
        def __init__(self, data):
            self.data = data

        def ne(self, value):
            masked = [[1 if v != value else 0 for v in row] for row in self.data]
            return _FakeTensor(masked)

        def sum(self, dim):
            assert dim == 1
            return _FakeTensor1D([sum(row) for row in self.data])

        def tolist(self):
            return self.data

    class _FakeTensor1D(_FakeTensor):
        def sum(self, dim=None):
            return sum(self.data)

    class _TokenizerStub:
        pad_token_id = 0

        def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
            encoded = [[1] * len(txt) for txt in texts]
            tensor = _FakeTensor(encoded)
            return {"input_ids": tensor, "attention_mask": tensor}

    prompts = ["hi", "ok"]
    completions = [" there", "!"]
    input_ids, attn, prompt_lengths = run_helpers._batch_tokenize_pairs(
        _TokenizerStub(), prompts, completions
    )
    assert isinstance(input_ids, _FakeTensor)
    assert isinstance(attn, _FakeTensor)
    assert prompt_lengths == [len(p) for p in prompts]
