"""
Smoke tests that run the Hydra entrypoints for training/generation/inference
with a fake ``GRPO_RECIPE`` so the fallback path is exercised end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List
from types import SimpleNamespace

import pytest
from string import Template
import maxent_grpo.cli.hydra_cli as hydra_cli

_USERCUSTOMIZE_TEMPLATE = Template("""\
import json
import os
import sys
from pathlib import Path
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

ROOT = Path("$PROJECT_ROOT")
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import importlib
importlib.import_module("ops.sitecustomize")

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
import maxent_grpo.cli.hydra_cli as hydra_cli
import types as _types
try:
    from maxent_grpo.generation.errors import GenerationServiceError as _GenErr
    import maxent_grpo.generation.vllm_requests as _vreq_mod
    setattr(_vreq_mod, "VLLMServiceError", _GenErr)
except Exception:
    pass

_HYDRA_CFG = None
def _set_hydra_cfg(command, **sections):
    global _HYDRA_CFG
    cfg = hydra_cli.HydraRootConfig()
    cfg.command = command
    for name, section in sections.items():
        setattr(cfg, name, section)
    _HYDRA_CFG = cfg
_hydra_mod = _types.ModuleType("hydra")
def _hydra_main(*_args, **_kwargs):
    def decorator(fn):
        def wrapper(*fn_args, **fn_kwargs):
            return fn(_HYDRA_CFG)
        return wrapper
    return decorator
_hydra_mod.main = _hydra_main
hydra_cli.hydra = _hydra_mod

class _OmegaStub:
    @staticmethod
    def structured(cfg):
        return cfg

    @staticmethod
    def to_object(cfg):
        return cfg

    @staticmethod
    def to_yaml(cfg):
        return str(cfg)

    @staticmethod
    def create(payload):
        return payload

hydra_cli.OmegaConf = _OmegaStub

@contextmanager
def _open_dict(cfg):
    yield cfg

hydra_cli.open_dict = _open_dict
hydra_cli.DictConfig = dict

def _load_recipe(recipe_path, model_config_cls):
    payload = json.loads(Path(recipe_path).read_text())
    script = GRPOScriptArguments(**payload.get("script", {}))
    training = GRPOConfig(**payload.get("training", {}))
    model = model_config_cls(**payload.get("model", {}))
    return script, training, model

hydra_cli.load_grpo_recipe = _load_recipe

stub_cli = ModuleType("maxent_grpo.training.cli")
def _parse_stub(*_a, **_k):
    raise RuntimeError("force hydra")
stub_cli.parse_grpo_args = _parse_stub
sys.modules["maxent_grpo.training.cli"] = stub_cli

trl_stub = ModuleType("trl")
class _ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
trl_stub.ModelConfig = _ModelConfig
sys.modules["trl"] = trl_stub

_original_build_grpo_configs = hydra_cli._build_grpo_configs
def _shim_build_grpo_configs(cmd):
    recipe_path = getattr(cmd, "recipe", None) or os.environ.get("GRPO_RECIPE")
    if recipe_path:
        return hydra_cli.load_grpo_recipe(recipe_path, model_config_cls=trl_stub.ModelConfig)
    return _original_build_grpo_configs(cmd)
hydra_cli._build_grpo_configs = _shim_build_grpo_configs

marker = Path(os.environ["MAXENT_CLI_SMOKE_MARKER"])
kind = os.environ.get("MAXENT_CLI_SMOKE_KIND")
if kind == "baseline":
    import maxent_grpo.pipelines.training.baseline as _pipe
    def _run_baseline(*_args, **_kwargs):
        marker.write_text("baseline")
    _pipe.run_baseline_training = _run_baseline
    sys.argv.append(f"baseline.recipe={os.environ['GRPO_RECIPE']}")
elif kind == "maxent":
    import maxent_grpo.pipelines.training.maxent as _pipe
    def _run_maxent(*_args, **_kwargs):
        marker.write_text("maxent")
    _pipe.run_maxent_training = _run_maxent
    sys.argv.append(f"maxent.recipe={os.environ['GRPO_RECIPE']}")
elif kind == "baseline_meta":
    import maxent_grpo.pipelines.training.baseline as _base
    import maxent_grpo.pipelines.training.maxent as _pipe
    def _fail_baseline(*_args, **_kwargs):
        raise RuntimeError("baseline path should not run when controller_meta_enabled=true")
    _base.run_baseline_training = _fail_baseline
    def _run_meta_grpo(*_args, **_kwargs):
        marker.write_text("maxent")
    _pipe.run_maxent_training = _run_meta_grpo
    sys.argv.append(f"baseline.recipe={os.environ['GRPO_RECIPE']}")
elif kind == "maxent_dryrun":
    from types import SimpleNamespace
    import sys
    from maxent_grpo.training.runtime.torch_stub import _build_torch_stub

    torch = _build_torch_stub()
    sys.modules["torch"] = torch
    import numpy as _np

    def _ensure_tensor_fn(name, fill_value):
        if hasattr(torch, name):
            return

        def _fn(shape, dtype=None):
            arr = _np.full(shape, fill_value, dtype=dtype if dtype is not None else _np.float32)
            return torch.tensor(arr)

        setattr(torch, name, _fn)

    _ensure_tensor_fn("zeros", 0.0)
    _ensure_tensor_fn("ones", 1.0)

    if not hasattr(torch, "log"):
        def _log_tensor(tensor):
            data = getattr(tensor, "arr", tensor)
            arr = _np.asarray(data)
            return torch.tensor(_np.log(arr))

        torch.log = _log_tensor
    from maxent_grpo.pipelines.training import loop_common
    from maxent_grpo.training import loop as loop_mod
    from maxent_grpo.training import pipeline as pipeline_mod
    from maxent_grpo.training.weighting import loss as loss_mod
    from maxent_grpo.training.weighting.loss import SequenceScores
    import maxent_grpo.pipelines.training.maxent as _pipe

    class _TestAccelerator:
        def __init__(self):
            self.is_main_process = True
            self.num_processes = 1
            self.process_index = 0
            self.device = "cpu"
            self.state = SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=0), distributed_type=None)

        def accumulate(self, *_args, **_kwargs):
            from contextlib import nullcontext
            return nullcontext()

        def no_sync(self, *_args, **_kwargs):
            from contextlib import nullcontext
            return nullcontext()

        def backward(self, *_a, **_k):
            return None

        def optimizer_step(self, optimizer):
            optimizer.step()

        def gather_object(self, obj):
            return [obj]

        def broadcast_object_list(self, obj_list, src=0):
            return obj_list

    class _Loader:
        def __init__(self, dataset, batch_size=1, **_kwargs):
            self._dataset = list(dataset)

        def __iter__(self):
            for row in self._dataset:
                yield row

    loop_common.require_accelerator = lambda *_a, **_k: _TestAccelerator
    loop_common.require_dataloader = lambda *_a, **_k: _Loader
    loop_common.get_model = lambda *_a, **_k: SimpleNamespace(parameters=lambda: [], config=SimpleNamespace())

    class _Tok:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, texts, **_kwargs):
            batch = len(texts)
            return {
                "input_ids": torch.zeros((batch, 1), dtype=torch.int64),
                "attention_mask": torch.ones((batch, 1), dtype=torch.int64),
            }

    loop_common.get_tokenizer = lambda *_a, **_k: _Tok()
    loop_common.load_datasets = lambda *_a, **_k: ([{"prompt": ["p"], "answer": ["a"]}], [])
    loop_common.load_reward_functions = lambda *_a, **_k: ([], [])
    loop_common.load_eval_reward_functions = lambda *_a, **_k: ([], [])

    def _fake_prepare_training_batch(ctx, generator, batch):
        reward_comp = SimpleNamespace(
            total_utils=[1.0],
            advantage_samples=[0.2],
            per_reward_values={"reward": [0.1]},
            q_grouped=[[1.0]],
            advantage=SimpleNamespace(grouped=[[0.0]]),
            pairs=SimpleNamespace(prompts=["prompt"], completions=["completion"]),
        )
        weight_stats = SimpleNamespace(
            weights_grouped=[[1.0]],
            flat_weights=[1.0],
            weight_entropy=0.0,
            weight_entropy_min=0.0,
            weight_entropy_max=0.0,
            advantage_entropy=[0.0],
        )
        ref_stats = SimpleNamespace(
            ref_logp_sum=[0.0],
            ref_logp_sum_raw=[0.0],
            ref_tok_counts=[1],
            ref_logp_mean=0.0,
            avg_completion_tokens=1.0,
        )
        length_stats = SimpleNamespace(
            min_length=1.0,
            mean_length=1.0,
            max_length=1.0,
            clipped_ratio=0.0,
            min_terminated=1.0,
            mean_terminated=1.0,
            max_terminated=1.0,
        )
        batch_stats = pipeline_mod._BatchStats(  # type: ignore[attr-defined]
            score_batch=SimpleNamespace(),
            ref_stats=ref_stats,
            weight_stats=weight_stats,
            length_stats=length_stats,
            num_completion_tokens=1.0,
            prompt_token_count=1.0,
        )
        scores = SequenceScores(
            cur_logp_sum=torch.tensor([0.0]),
            behavior_logp_sum=torch.tensor([0.0]),
            log_ratio_train=torch.tensor([0.0]),
            denom_tok_tensor=torch.tensor([1.0]),
        )
        return pipeline_mod.PreparedBatch(
            grouped_completions=[["completion"]],
            reward_comp=reward_comp,
            batch_stats=batch_stats,
            total_input_tokens=2.0,
            scores=scores,
            seed_metrics={"seed": 1.0},
        )

    pipeline_mod.prepare_training_batch = _fake_prepare_training_batch
    loss_out = SimpleNamespace(
        loss=SimpleNamespace(requires_grad=False),
        total_loss_scalar=0.1,
        policy_loss_scalar=0.05,
        kl_loss_scalar=0.01,
        weighted_kl_loss_scalar=0.01,
        clip_loss_scalar=0.0,
        scalars=SimpleNamespace(kl_loss=0.01),
    )
    diagnostics = SimpleNamespace(
        clip_ratio=0.0,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
        kl_value=0.01,
        kl_per_token_by_len_bucket={},
        kl_token_count_by_len_bucket={},
    )
    loss_mod.build_loss_inputs = lambda *_a, **_k: (SimpleNamespace(), SimpleNamespace())
    loss_mod.evaluate_losses = lambda *_a, **_k: (loss_out, diagnostics)
    loop_mod.build_loss_inputs = lambda *_a, **_k: (SimpleNamespace(), SimpleNamespace())
    loop_mod.evaluate_losses = lambda *_a, **_k: (loss_out, diagnostics)

    class _GeneratorStub:
        def __init__(self, ctx):
            self.ctx = ctx

        def generate(self, *_a, **_k):
            return ([["completion"]], None)

    loop_mod.CompletionGenerator = lambda ctx: _GeneratorStub(ctx)
    loop_mod._maybe_patch_zero_no_sync = lambda *_a, **_k: False

    last_step = {"value": 0}
    _orig_opt_step = loop_mod._optimizer_step

    def _recording_opt_step(ctx, state, current_lr):
        result = _orig_opt_step(ctx, state, current_lr)
        last_step["value"] = state.global_step
        return result

    loop_mod._optimizer_step = _recording_opt_step
    _real_run_maxent = _pipe.run_maxent_training

    def _run_maxent_dryrun(*args, **kwargs):
        _real_run_maxent(*args, **kwargs)
        marker.write_text(str(last_step["value"]))

    _pipe.run_maxent_training = _run_maxent_dryrun
    sys.argv.append(f"maxent.recipe={os.environ['GRPO_RECIPE']}")
elif kind == "infoseed":
    import maxent_grpo.pipelines.training.infoseed as _pipe
    def _run_infoseed(*_args, **_kwargs):
        marker.write_text("infoseed")
    _pipe.run_infoseed_training = _run_infoseed
    sys.argv.append(f"infoseed.recipe={os.environ['GRPO_RECIPE']}")
elif kind == "generate":
    import maxent_grpo.pipelines.generation.distilabel as _pipe
    def _run_generation_job(*_args, **_kwargs):
        marker.write_text("generate")
    _pipe.run_generation_job = _run_generation_job
    hydra_cli.run_generation_job = _run_generation_job
    _set_hydra_cfg(
        "generate",
        generate=hydra_cli.GenerateCommand(
            args={
                "hf_dataset": "dummy/dataset",
                "model": "dummy-model",
            }
        ),
    )
elif kind == "inference":
    import maxent_grpo.pipelines.inference.inference as _pipe
    def _run_inference(specs, **_kwargs):
        marker.write_text("inference")
        return [
            SimpleNamespace(model=specs[0].model_name_or_path, metrics={"pass@1": 1.0})
        ]
    _pipe.run_math_inference = _run_inference
    hydra_cli.run_math_inference = _run_inference
    _set_hydra_cfg(
        "inference",
        inference=hydra_cli.InferenceCommand(
            models=[{"model_name_or_path": "stub"}],
            dataset="math_500",
        ),
    )
else:
    raise RuntimeError(f"Unknown CLI smoke kind: {kind}")
""")


@pytest.mark.parametrize(
    ("cli_args", "kind"),
    [
        (["-m", "maxent_grpo.grpo"], "baseline"),
        (["-m", "maxent_grpo.grpo"], "baseline_meta"),
        (["-m", "maxent_grpo.maxent_grpo"], "maxent"),
        (["-m", "maxent_grpo.maxent_grpo"], "maxent_dryrun"),
        (["-m", "maxent_grpo.infoseed"], "infoseed"),
        (
            ["-c", "from maxent_grpo.cli.hydra_cli import generate_entry; generate_entry()"],
            "generate",
        ),
        (
            ["-c", "from maxent_grpo.cli.hydra_cli import inference_entry; inference_entry()"],
            "inference",
        ),
    ],
)
def test_cli_hydra_smoke(tmp_path, cli_args: List[str], kind: str):
    project_root = Path(__file__).resolve().parents[2]
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    recipe_path = tmp_path / "recipe.json"
    training_payload = {"output_dir": str(tmp_path / "out"), "num_train_epochs": 0}
    if kind == "infoseed":
        training_payload.update({"info_seed_enabled": True, "info_seed_num_seeds": 1})
    if kind in {"maxent", "baseline_meta"}:
        training_payload.update(
            {
                "controller_meta_enabled": True,
                "controller_meta_lr": 0.05,
                "controller_meta_method": "analytic",
            }
        )
    if kind == "maxent_dryrun":
        training_payload.update(
            {
                "num_train_epochs": 1,
                "train_grpo_objective": False,
                "num_generations": 2,
                "gradient_accumulation_steps": 1,
                "per_device_train_batch_size": 1,
                "max_steps": 1,
                "max_prompt_length": 8,
                "max_completion_length": 16,
                "save_strategy": "steps",
                "save_steps": 1,
            }
        )
    recipe_payload = {
        "script": {"dataset_name": "dummy"},
        "training": training_payload,
        "model": {},
    }
    recipe_path.write_text(json.dumps(recipe_payload), encoding="utf-8")
    marker_path = tmp_path / f"{kind}.marker"
    usercustomize_path = shim_dir / "usercustomize.py"
    usercustomize_path.write_text(
        _USERCUSTOMIZE_TEMPLATE.substitute(PROJECT_ROOT=str(project_root)),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTEST_CURRENT_TEST", None)
    existing = env.get("PYTHONPATH", "")
    src_path = project_root / "src"
    shim_plus_src = [str(shim_dir), str(src_path)]
    env["PYTHONPATH"] = (
        os.pathsep.join(shim_plus_src + [existing])
        if existing
        else os.pathsep.join(shim_plus_src)
    )
    env["GRPO_RECIPE"] = str(recipe_path)
    env["MAXENT_CLI_SMOKE_MARKER"] = str(marker_path)
    env["MAXENT_CLI_SMOKE_KIND"] = kind
    cmd = [sys.executable, *cli_args]
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    marker_text = marker_path.read_text(encoding="utf-8").strip()
    if kind == "maxent_dryrun":
        assert float(marker_text) >= 1.0
    else:
        expected_marker = "maxent" if kind in {"maxent", "baseline_meta"} else kind
        assert marker_text == expected_marker


def test_build_grpo_configs_prefers_recipe(monkeypatch, tmp_path):
    recipe_path = tmp_path / "recipe.json"
    payload = {
        "script": {"dataset_name": "ds"},
        "training": {
            "output_dir": "/tmp/out",
            "logging_steps": 1,
            "save_steps": 1,
            "train_grpo_objective": False,
            "maxent_tau": 0.3,
        },
        "model": {"model_name_or_path": "stub"},
    }
    recipe_path.write_text(json.dumps(payload), encoding="utf-8")
    called = {}

    def _fake_load(recipe, model_config_cls):
        called["recipe"] = recipe
        called["cls"] = model_config_cls
        return ("script", "training", "model")

    monkeypatch.setattr(hydra_cli, "load_grpo_recipe", _fake_load)
    cmd = hydra_cli.MaxentCommand(recipe=str(recipe_path))
    result = hydra_cli._build_grpo_configs(cmd)
    assert result == ("script", "training", "model")
    assert called["recipe"] == str(recipe_path)
    assert "cls" in called


def test_build_grpo_configs_applies_recipe_overrides(monkeypatch):
    script_ns = SimpleNamespace(dataset_name="base_ds", extra=1)
    training_ns = SimpleNamespace(maxent_tau=0.1, gradient_checkpointing_kwargs={"use_reentrant": True})
    model_ns = SimpleNamespace(model_name_or_path="base")

    def _fake_load(recipe, model_config_cls):
        return (script_ns, training_ns, model_ns)

    monkeypatch.setattr(hydra_cli, "load_grpo_recipe", _fake_load)
    cmd = hydra_cli.MaxentCommand(
        recipe="stub",
        script={"dataset_name": "override_ds"},
        training={
            "maxent_tau": 0.5,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        },
        model={"model_name_or_path": "override"},
    )
    script_args, training_args, model_args = hydra_cli._build_grpo_configs(cmd)
    assert script_args.dataset_name == "override_ds"
    assert script_args.extra == 1
    assert training_args.maxent_tau == pytest.approx(0.5)
    assert training_args.gradient_checkpointing_kwargs["use_reentrant"] is False
    assert model_args.model_name_or_path == "override"


def test_build_grpo_configs_merges_inline_sections(monkeypatch):
    class _ModelCfg(SimpleNamespace):
        pass

    monkeypatch.setattr(
        hydra_cli, "_resolve_model_config_cls", lambda: _ModelCfg
    )
    class _ScriptArgs(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class _TrainingCfg(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    monkeypatch.setattr(hydra_cli, "GRPOScriptArguments", _ScriptArgs)
    monkeypatch.setattr(hydra_cli, "GRPOConfig", _TrainingCfg)
    cmd = hydra_cli.MaxentCommand(
        recipe=None,
        script={"dataset_name": "math_ds", "reward_funcs": ["acc"]},
        training={
            "output_dir": "/tmp/out",
            "logging_steps": 1,
            "save_steps": 1,
            "train_grpo_objective": False,
            "maxent_tau": 0.25,
            "reward_weights": [1.0],
        },
        model={"model_name_or_path": "stub-model"},
    )
    script_args, training_args, model_args = hydra_cli._build_grpo_configs(cmd)
    assert script_args.dataset_name == "math_ds"
    assert training_args.maxent_tau == pytest.approx(0.25)
    assert training_args.reward_weights == [1.0]
    assert isinstance(model_args, _ModelCfg)
