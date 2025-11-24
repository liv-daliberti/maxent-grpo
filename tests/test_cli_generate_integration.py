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

Integration tests for the CLI layer around distilabel generation.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
import sys


import maxent_grpo.cli.generate as cli_generate


def _install_distilabel_and_datasets(monkeypatch):
    distilabel_mod = SimpleNamespace()

    class _Pipeline:
        def __init__(self):
            self.run_calls = []

        class _Ctx:
            def __init__(self, pipe):
                self._pipe = pipe

            def __enter__(self):
                return self._pipe

            def __exit__(self, exc_type, exc, tb):
                return False

        def ray(self):
            return self._Ctx(self)

        def run(self, **kwargs):
            self.run_calls.append(kwargs)
            return SimpleNamespace(push_to_hub=lambda repo, private: (repo, private))

    distilabel_mod.pipeline = SimpleNamespace(Pipeline=_Pipeline)
    distilabel_mod.steps = SimpleNamespace(
        StepResources=lambda replicas: SimpleNamespace(replicas=replicas),
        tasks=SimpleNamespace(TextGeneration=lambda **kwargs: kwargs),
    )
    distilabel_mod.llms = SimpleNamespace(OpenAILLM=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "distilabel", distilabel_mod)

    datasets_mod = SimpleNamespace()
    datasets_mod.load_dataset = lambda name, config, split: {
        "rows": [(name, config, split)]
    }
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    return distilabel_mod


def test_build_generate_parser_smoke():
    parser = cli_generate.build_generate_parser()
    args = parser.parse_args(["--hf-dataset", "org/repo", "--model", "m"])
    assert args.hf_dataset == "org/repo"
    assert args.model == "m"
    assert args.input_batch_size == 64


def test_run_cli_invokes_pipeline(monkeypatch):
    _install_distilabel_and_datasets(monkeypatch)
    calls = {}

    monkeypatch.setattr(
        cli_generate, "run_generation_job", lambda cfg: calls.setdefault("cfg", cfg)
    )
    args = Namespace(
        hf_dataset="org/repo",
        hf_dataset_config=None,
        hf_dataset_split="train",
        prompt_column="prompt",
        prompt_template="{{ instruction }}",
        model="demo",
        vllm_server_url="http://localhost",
        temperature=None,
        top_p=None,
        max_new_tokens=8,
        num_generations=1,
        input_batch_size=2,
        client_replicas=1,
        timeout=10,
        retries=0,
        hf_output_dataset=None,
        private=False,
    )
    cli_generate.run_cli(args)
    assert isinstance(calls["cfg"], cli_generate.DistilabelGenerationConfig)
    assert calls["cfg"].model == "demo"