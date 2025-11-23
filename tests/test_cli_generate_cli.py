from __future__ import annotations

import importlib
import types
from argparse import Namespace
def test_run_cli_invokes_pipeline(monkeypatch):
    module = importlib.import_module("src.cli.generate")
    called = {}

    class _Cfg:
        @classmethod
        def from_namespace(cls, ns):
            called["cfg_args"] = ns
            return cls()

    monkeypatch.setattr(module, "DistilabelGenerationConfig", _Cfg)
    monkeypatch.setattr(
        module, "run_generation_job", lambda cfg: called.setdefault("ran_with", cfg)
    )

    ns = Namespace(hf_dataset="dset", model="m", vllm_server_url="u")
    module.run_cli(ns)
    assert called["cfg_args"] is ns
    assert isinstance(called["ran_with"], _Cfg)


def test_typer_entrypoint_and_app(monkeypatch):
    # Reload with a stub Typer module to ensure typer branch is active.
    typer_stub = types.SimpleNamespace()
    typer_stub.Option = lambda default, *args, **_kwargs: default
    run_calls = {}

    def _run(fn):
        run_calls["fn"] = fn
        return fn

    typer_stub.run = _run
    monkeypatch.setitem(importlib.import_module("sys").modules, "typer", typer_stub)
    module = importlib.reload(importlib.import_module("src.cli.generate"))
    called = {}
    monkeypatch.setattr(module, "run_cli", lambda ns: called.setdefault("ns", ns))

    module._typer_entrypoint(
        hf_dataset="dataset",
        hf_dataset_config=None,
        hf_dataset_split="train",
        prompt_column="prompt",
        prompt_template="{{ instruction }}",
        model="model",
        vllm_server_url="http://localhost",
        temperature=0.1,
        top_p=0.9,
        max_new_tokens=10,
        num_generations=2,
        input_batch_size=3,
        client_replicas=4,
        timeout=5,
        retries=1,
        hf_output_dataset=None,
        private=True,
    )

    ns = called["ns"]
    assert ns.hf_dataset == "dataset"
    assert ns.private is True

    module.app()
    assert run_calls["fn"].__name__ == "_typer_entrypoint"
