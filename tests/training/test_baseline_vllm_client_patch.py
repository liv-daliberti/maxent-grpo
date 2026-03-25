from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

from maxent_grpo.training import baseline


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise RuntimeError(f"status={self.status_code}")


class _FakeSession:
    def __init__(self, events: list[tuple[str, str]] | None = None):
        self.get_calls: list[tuple[str, float | None]] = []
        self.post_calls: list[tuple[str, dict, float | None]] = []
        self.closed = False
        self._events = events if events is not None else []
        self.generate_response_payload: dict = {"completion_ids": [[1, 2, 3]]}

    def get(self, url: str, timeout: float | None = None) -> _FakeResponse:
        self.get_calls.append((url, timeout))
        self._events.append(("get", url))
        return _FakeResponse(payload={"world_size": 1})

    def post(
        self,
        url: str,
        json: dict | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.post_calls.append((url, json or {}, timeout))
        self._events.append(("post", url))
        if url.endswith("/generate/"):
            return _FakeResponse(payload=dict(self.generate_response_payload))
        return _FakeResponse()

    def close(self) -> None:
        self.closed = True


class _FakeTensor:
    def __init__(self, dtype: str, shape: tuple[int, ...], device: str = "cuda:0"):
        self.dtype = dtype
        self.shape = shape
        self.device = device

    def detach(self) -> "_FakeTensor":
        return self

    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def element_size(self) -> int:
        return 2


class _FakeClient:
    def __init__(self, base_url: str | None = None, group_port: int = 0, **_kwargs):
        self.base_url = base_url or "http://127.0.0.1:8000"
        self.host = "127.0.0.1"
        self.group_port = group_port
        self.events: list[tuple[str, str]] = []
        self.session = _FakeSession(self.events)

    def init_communicator(self) -> None:
        raise AssertionError("patched method should replace this")

    def update_named_param(self, _name: str, _weights) -> None:
        raise AssertionError("patched method should replace this")

    def update_model_params(self, model) -> None:
        for name, param in model.named_parameters():
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self) -> dict[str, str]:
        return {"message": "reset"}

    def close_communicator(self) -> None:
        raise AssertionError("patched method should replace this")


def test_patch_trl_vllm_client_batches_builtin_weight_transfer(monkeypatch):
    trl_root = ModuleType("trl")
    trl_extras = ModuleType("trl.extras")
    trl_vllm_client = ModuleType("trl.extras.vllm_client")
    trl_vllm_client.VLLMClient = _FakeClient
    trl_trainer = ModuleType("trl.trainer")
    trl_grpo = ModuleType("trl.trainer.grpo_trainer")
    trl_grpo.VLLMClient = _FakeClient

    send_calls: list[tuple[list[tuple[str, _FakeTensor]], object, int, bool]] = []
    init_calls: list[dict] = []

    class _FakeEngine:
        @staticmethod
        def trainer_init(init_info: dict) -> object:
            init_calls.append(dict(init_info))
            client.events.append(("trainer_init", "trainer"))
            return SimpleNamespace(name="comm")

        @staticmethod
        def trainer_send_weights(
            iterator,
            group,
            src: int = 0,
            packed: bool = False,
            **_kwargs,
        ) -> None:
            send_calls.append((list(iterator), group, src, packed))

    fake_nccl_mod = ModuleType("vllm.distributed.weight_transfer.nccl_engine")
    fake_nccl_mod.NCCLWeightTransferEngine = _FakeEngine
    fake_gpu_worker_mod = ModuleType("vllm.v1.worker.gpu_worker")

    class _FakeGPUWorker:
        def init_weight_transfer_engine(self, init_info: dict) -> None:
            return None

        def update_weights(self, update_info: dict) -> None:
            return None

    fake_gpu_worker_mod.GPUWorker = _FakeGPUWorker

    for name, module in (
        ("trl", trl_root),
        ("trl.extras", trl_extras),
        ("trl.extras.vllm_client", trl_vllm_client),
        ("trl.trainer", trl_trainer),
        ("trl.trainer.grpo_trainer", trl_grpo),
        ("vllm.distributed.weight_transfer.nccl_engine", fake_nccl_mod),
        ("vllm.v1.worker.gpu_worker", fake_gpu_worker_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    fake_stream = SimpleNamespace(synchronize=lambda: None)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(current_stream=lambda device=None: fake_stream)
    )
    monkeypatch.setattr(baseline, "require_torch", lambda _ctx: fake_torch)
    monkeypatch.setattr(baseline.atexit, "register", lambda _fn: None)

    baseline._patch_trl_vllm_client_init()

    client = _FakeClient(base_url="http://127.0.0.1:8000", group_port=51216)
    client.init_communicator()
    client.update_named_param("embed.weight", _FakeTensor("torch.bfloat16", (4, 8)))
    client.update_named_param("lm_head.weight", _FakeTensor("torch.bfloat16", (8, 4)))

    assert len(client.session.post_calls) == 1
    assert client.session.post_calls[0][0].endswith("/init_communicator/")
    assert client.events[:3] == [
        ("get", "http://127.0.0.1:8000/get_world_size/"),
        ("post", "http://127.0.0.1:8000/init_communicator/"),
        ("trainer_init", "trainer"),
    ]

    reset_result = client.reset_prefix_cache()

    assert init_calls == [
        {
            "master_address": "127.0.0.1",
            "master_port": 51216,
            "rank_offset": 1,
            "world_size": 2,
        }
    ]
    assert len(send_calls) == 1
    assert [name for name, _tensor in send_calls[0][0]] == [
        "embed.weight",
        "lm_head.weight",
    ]
    assert send_calls[0][2] == 0
    assert send_calls[0][3] is True
    assert len(client.session.post_calls) == 2
    assert client.session.post_calls[1][0].endswith("/update_named_param/")
    update_payload = client.session.post_calls[1][1]
    assert update_payload["dtype"] == "bfloat16"
    assert update_payload["shape"] == [4, 8]
    assert update_payload["name"].startswith(baseline._VLLM_BATCH_UPDATE_PREFIX)
    encoded = json.loads(
        update_payload["name"][len(baseline._VLLM_BATCH_UPDATE_PREFIX) :]
    )
    assert encoded == {
        "names": ["embed.weight", "lm_head.weight"],
        "dtypes": ["bfloat16", "bfloat16"],
        "shapes": [[4, 8], [8, 4]],
    }
    assert reset_result == {"message": "reset"}


def test_patch_vllm_guided_decoding_compat_remaps_structured_outputs(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_sampling_params = ModuleType("vllm.sampling_params")
    trl_trainer = ModuleType("trl.trainer")
    trl_grpo = ModuleType("trl.trainer.grpo_trainer")

    class _StructuredOutputsParams:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    sampling_calls: list[dict] = []

    def _sampling_params_factory(*args, **kwargs):
        sampling_calls.append(dict(kwargs))
        return {"args": args, "kwargs": kwargs}

    fake_sampling_params.StructuredOutputsParams = _StructuredOutputsParams
    fake_vllm.SamplingParams = _sampling_params_factory
    trl_grpo.SamplingParams = _sampling_params_factory

    for name, module in (
        ("vllm", fake_vllm),
        ("vllm.sampling_params", fake_sampling_params),
        ("trl.trainer", trl_trainer),
        ("trl.trainer.grpo_trainer", trl_grpo),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    baseline._patch_vllm_guided_decoding_compat()

    guided = fake_sampling_params.GuidedDecodingParams(
        backend="outlines",
        regex="a+",
    )
    result = trl_grpo.SamplingParams(n=2, guided_decoding=guided)

    assert "guided_decoding" not in sampling_calls[0]
    assert sampling_calls[0]["n"] == 2
    structured_outputs = sampling_calls[0]["structured_outputs"]
    assert isinstance(structured_outputs, _StructuredOutputsParams)
    assert structured_outputs.regex == "a+"
    assert structured_outputs._backend == "outlines"
    assert result["kwargs"]["structured_outputs"] is structured_outputs


def test_patch_trl_vllm_client_generate_attaches_boundary_fields(monkeypatch):
    trl_root = ModuleType("trl")
    trl_extras = ModuleType("trl.extras")
    trl_vllm_client = ModuleType("trl.extras.vllm_client")
    trl_vllm_client.VLLMClient = _FakeClient
    trl_trainer = ModuleType("trl.trainer")
    trl_grpo = ModuleType("trl.trainer.grpo_trainer")
    trl_grpo.VLLMClient = _FakeClient
    transformers_mod = ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            class _Tok:
                vocab_size = 3

                def __len__(self):
                    return 5

            return _Tok()

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return SimpleNamespace(vocab_size=10)

    transformers_mod.AutoTokenizer = _FakeAutoTokenizer
    transformers_mod.AutoConfig = _FakeAutoConfig

    for name, module in (
        ("trl", trl_root),
        ("trl.extras", trl_extras),
        ("trl.extras.vllm_client", trl_vllm_client),
        ("trl.trainer", trl_trainer),
        ("trl.trainer.grpo_trainer", trl_grpo),
        ("transformers", transformers_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("MAXENT_VLLM_SERVER_MODEL_NAME", "Qwen/test")

    baseline._patch_trl_vllm_client_init()

    client = _FakeClient(base_url="http://127.0.0.1:8000")
    completions = client.generate(["hello"], n=2, max_tokens=7)

    assert completions == [[1, 2, 3]]
    generate_url, payload, _timeout = client.session.post_calls[-1]
    assert generate_url == "http://127.0.0.1:8000/generate/"
    assert payload["prompts"] == ["hello"]
    assert payload["n"] == 2
    assert payload["max_tokens"] == 7
    assert payload["blocked_token_ids"] == [5, 6, 7, 8, 9]
    assert "logit_bias" not in payload


def test_patch_trl_vllm_client_generate_rejects_invalid_completion_ids(monkeypatch):
    trl_root = ModuleType("trl")
    trl_extras = ModuleType("trl.extras")
    trl_vllm_client = ModuleType("trl.extras.vllm_client")
    trl_vllm_client.VLLMClient = _FakeClient
    trl_trainer = ModuleType("trl.trainer")
    trl_grpo = ModuleType("trl.trainer.grpo_trainer")
    trl_grpo.VLLMClient = _FakeClient
    transformers_mod = ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            class _Tok:
                vocab_size = 3

                def __len__(self):
                    return 5

            return _Tok()

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return SimpleNamespace(vocab_size=10)

    transformers_mod.AutoTokenizer = _FakeAutoTokenizer
    transformers_mod.AutoConfig = _FakeAutoConfig

    for name, module in (
        ("trl", trl_root),
        ("trl.extras", trl_extras),
        ("trl.extras.vllm_client", trl_vllm_client),
        ("trl.trainer", trl_trainer),
        ("trl.trainer.grpo_trainer", trl_grpo),
        ("transformers", transformers_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("MAXENT_VLLM_SERVER_MODEL_NAME", "Qwen/test")

    baseline._patch_trl_vllm_client_init()

    client = _FakeClient(base_url="http://127.0.0.1:8000")
    client.session.generate_response_payload = {"completion_ids": [[7]]}

    with pytest.raises(RuntimeError, match="tokenizer-addressable range"):
        client.generate(["hello"])
