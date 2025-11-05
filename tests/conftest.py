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

import sys
import types
from pathlib import Path
import pytest


def _ensure_module(name: str):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


def pytest_configure(config):
    # Ensure the src/ directory is importable (src-layout project)
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    # Stub minimal trl API used by the codebase
    trl = _ensure_module("trl")
    if not hasattr(trl, "ScriptArguments"):
        class ScriptArguments:  # minimal base
            pass
        class GRPOConfig:
            pass
        class SFTConfig:
            pass
        class ModelConfig:
            pass
        def get_kbit_device_map():
            return None
        def get_quantization_config(*_a, **_k):
            return None
        trl.ScriptArguments = ScriptArguments
        trl.GRPOConfig = GRPOConfig
        trl.SFTConfig = SFTConfig
        trl.ModelConfig = ModelConfig
        trl.get_kbit_device_map = get_kbit_device_map
        trl.get_quantization_config = get_quantization_config

    # Stub minimal transformers API used by the codebase
    tf = _ensure_module("transformers")
    if not hasattr(tf, "set_seed"):
        def set_seed(_):
            return None
        tf.set_seed = set_seed
    # trainer_utils.get_last_checkpoint
    tu = _ensure_module("transformers.trainer_utils")
    if not hasattr(tu, "get_last_checkpoint"):
        def get_last_checkpoint(_):
            return None
        tu.get_last_checkpoint = get_last_checkpoint
    # Attach submodule back to parent
    setattr(tf, "trainer_utils", tu)

    # Tokenizer/model/config stubs
    if not hasattr(tf, "AutoTokenizer"):
        class _Tok:
            def __init__(self):
                self.chat_template = None
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "".join(m.get("content", "") for m in messages) + ("<GEN>" if add_generation_prompt else "")
        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *a, **k):
                return _Tok()
        tf.AutoTokenizer = AutoTokenizer
    if not hasattr(tf, "PreTrainedTokenizer"):
        class PreTrainedTokenizer:  # only for typing
            pass
        tf.PreTrainedTokenizer = PreTrainedTokenizer
    if not hasattr(tf, "AutoModelForCausalLM"):
        class _Model:
            def __init__(self):
                self.config = types.SimpleNamespace()
        class AutoModelForCausalLM:
            last_call = None
            @classmethod
            def from_pretrained(cls, model_name_or_path, **kwargs):
                cls.last_call = (model_name_or_path, kwargs)
                return _Model()
        tf.AutoModelForCausalLM = AutoModelForCausalLM
    if not hasattr(tf, "AutoConfig"):
        class AutoConfig:
            @classmethod
            def from_pretrained(cls, *a, **k):
                obj = types.SimpleNamespace()
                obj.num_attention_heads = 64
                return obj
        tf.AutoConfig = AutoConfig

    # transformers.utils.import_utils for is_package_available
    tf_utils = _ensure_module("transformers.utils")
    tf_import_utils = _ensure_module("transformers.utils.import_utils")
    if not hasattr(tf_import_utils, "_is_package_available"):
        def _is_package_available(_pkg):
            return False
        tf_import_utils._is_package_available = _is_package_available
    setattr(tf_utils, "import_utils", tf_import_utils)
    setattr(tf, "utils", tf_utils)

    # Minimal torch with torch.distributed and torch.utils.data.Dataset
    torch = _ensure_module("torch")
    if not hasattr(torch, "float16"):
        # Provide a few dtype attrs used by getattr(torch, name)
        torch.float16 = object()
        torch.bfloat16 = object()
        torch.float32 = object()
    td = _ensure_module("torch.distributed")
    setattr(torch, "distributed", td)
    tutils = _ensure_module("torch.utils")
    tdata = _ensure_module("torch.utils.data")
    if not hasattr(tdata, "Dataset"):
        class Dataset:
            def __len__(self):
                return 0
            def __getitem__(self, idx):
                raise IndexError
        tdata.Dataset = Dataset
    setattr(tutils, "data", tdata)
    setattr(torch, "utils", tutils)

    # Minimal huggingface_hub API used by utils.hub
    hfh = _ensure_module("huggingface_hub")
    if not hasattr(hfh, "create_repo"):
        def create_repo(repo_id, private=True, exist_ok=True):
            return f"https://hub/{repo_id}"
        def list_repo_commits(repo_id):
            return [types.SimpleNamespace(commit_id="init")]
        def create_branch(repo_id, branch, revision, exist_ok=True):
            return None
        def upload_folder(**kwargs):
            return types.SimpleNamespace(done=True)
        def list_repo_refs(repo_id):
            return types.SimpleNamespace(branches=[types.SimpleNamespace(name="main")])
        def list_repo_files(repo_id, revision=None):
            return []
        def repo_exists(repo_id):
            return True
        def get_safetensors_metadata(repo_id):
            class _Meta:
                parameter_count = {"model": 1000}
            return _Meta()
        hfh.create_repo = create_repo
        hfh.list_repo_commits = list_repo_commits
        hfh.create_branch = create_branch
        hfh.upload_folder = upload_folder
        hfh.list_repo_refs = list_repo_refs
        hfh.list_repo_files = list_repo_files
        hfh.repo_exists = repo_exists
        hfh.get_safetensors_metadata = get_safetensors_metadata

    # Minimal distilabel API to import src/generate.py
    disti = _ensure_module("distilabel")
    if not hasattr(disti, "pipeline"):
        # Pipeline context manager with .ray()
        class _Pipe:
            def __init__(self):
                self._ray = False
            def ray(self):
                self._ray = True
                return self
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
        pipe_mod = types.ModuleType("distilabel.pipeline")
        pipe_mod.Pipeline = _Pipe
        disti.pipeline = pipe_mod

        # Steps and resources
        steps_mod = types.ModuleType("distilabel.steps")
        class StepResources:
            def __init__(self, replicas=1):
                self.replicas = replicas
        steps_mod.StepResources = StepResources

        tasks_mod = types.ModuleType("distilabel.steps.tasks")
        class TextGeneration:
            def __init__(self, *a, **k):
                pass
        tasks_mod.TextGeneration = TextGeneration
        disti.steps = steps_mod
        disti.steps.tasks = tasks_mod

        # LLMs
        llms_mod = types.ModuleType("distilabel.llms")
        class OpenAILLM:
            def __init__(self, *a, **k):
                pass
        llms_mod.OpenAILLM = OpenAILLM
        disti.llms = llms_mod


def pytest_collection_modifyitems(config, items):
    """Ensure every test has at least one standard mark.

    If a test has none of the recognized marks (unit/integration/slow), we add
    the 'unit' mark by default so that all tests can be filtered via -m.
    """
    for item in items:
        existing = {m.name for m in item.iter_markers()}
        if not existing.intersection({"unit", "integration", "slow"}):
            item.add_marker(pytest.mark.unit)
