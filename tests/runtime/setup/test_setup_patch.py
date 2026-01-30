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

Smoke tests for the TRL vllm_serve patcher in tools/patch_trl_vllm_serve.py.
"""

from __future__ import annotations

import importlib
import importlib.util
import importlib.metadata as importlib_metadata
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_patch_module(module_name: str = "patcher_under_test"):
    """Import tools/patch_trl_vllm_serve.py as a module."""

    patch_path = Path(__file__).resolve().parents[3] / "tools" / "patch_trl_vllm_serve.py"
    spec = importlib.util.spec_from_file_location(module_name, patch_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _install_fake_trl(monkeypatch, target_path: Path) -> None:
    """Stub importlib lookups to point at a fake trl.scripts.vllm_serve."""

    fake_mod = ModuleType("trl.scripts.vllm_serve")
    fake_mod.__file__ = str(target_path)
    real_import = importlib.import_module
    real_version = importlib_metadata.version

    def _fake_import(name: str, *args, **kwargs):
        if name == "trl.scripts.vllm_serve":
            return fake_mod
        return real_import(name, *args, **kwargs)

    def _fake_version(pkg: str):
        if pkg == "trl":
            return "0.18.0"
        return real_version(pkg)

    monkeypatch.setattr("importlib.import_module", _fake_import)
    monkeypatch.setattr("importlib.metadata.version", _fake_version)


def test_trl_patch_idempotent(tmp_path, monkeypatch):
    """Applying the patch twice should only insert the hooks once."""

    target = tmp_path / "vllm_serve.py"
    stub_text = """from typing import Optional
from itertools import chain

class LLM:
    pass


class BaseModel:
    pass


def chunk_list(lst, _n):
    return [lst]


script_args = type("Args", (), {"data_parallel_size": 1, "enforce_eager": False, "dtype": "float16"})()
connections = []


def main(_script_args=None):
    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        # Sampling parameters
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
        )
        # Evenly distribute prompts across DP ranks
        chunked_prompts = chunk_list(request.prompts, script_args.data_parallel_size)

        # Send the prompts to each worker
        for connection, prompts in zip(connections, chunked_prompts):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

        # Receive results
        all_outputs = [connection.recv() for connection in connections]

        # Handle empty prompts (see above)
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts) if prompts]

        # Flatten and combine all results
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        return {"completion_ids": completion_ids}


llm = LLM(
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
)
"""
    target.write_text(stub_text, encoding="utf-8")
    _install_fake_trl(monkeypatch, target)
    patch_mod = _load_patch_module(module_name="patch_trl_idempotent")

    patch_mod.patch_trl_vllm_serve()
    patch_mod.patch_trl_vllm_serve()

    patched = target.read_text(encoding="utf-8")
    assert patched.count("use_tqdm_on_load=True") == 1
    assert "return_logprobs: bool = False" in patched
    assert patched.count("logprobs=logprobs") == 1
    assert patched.count("output_token_logprobs") == 2
    assert '"completion_ids": token_ids' in patched


def test_trl_patch_raises_when_anchor_missing(tmp_path, monkeypatch):
    """Fail loudly when expected anchors are absent."""

    target = tmp_path / "vllm_serve.py"
    target.write_text("def noop():\n    return None\n", encoding="utf-8")
    _install_fake_trl(monkeypatch, target)
    patch_mod = _load_patch_module(module_name="patch_missing_anchor")

    with pytest.raises(RuntimeError) as excinfo:
        patch_mod.patch_trl_vllm_serve()
    assert "insertion point" in str(excinfo.value)
