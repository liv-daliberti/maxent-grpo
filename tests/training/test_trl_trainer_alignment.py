from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from maxent_grpo.rewards.basic import pure_accuracy_reward_math
from maxent_grpo.training import trl_trainer as trainer_mod
from maxent_grpo.training.trl_trainer import build_custom_grpo_trainer


class _FakeAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.process_index = 0
        self.is_main_process = True

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def wait_for_everyone(self) -> None:
        return


class _FakeVLLMClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def close_communicator(self) -> None:
        self.close_calls += 1


class _ParentTrainerStub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args
        cfg = kwargs.get("args")
        if cfg is None:
            cfg = SimpleNamespace(train_grpo_objective=True, use_vllm=False)
        self.parent_received_use_vllm = bool(getattr(cfg, "use_vllm", False))
        self.parent_vllm_mode = str(getattr(cfg, "vllm_mode", "server") or "server")
        self.parent_vllm_init_calls = 0
        self.parent_generate_calls = 0
        self.args = cfg
        self.model = SimpleNamespace(training=True)
        self.processing_class = SimpleNamespace(
            eos_token_id=2,
            batch_decode=lambda ids, skip_special_tokens=True: ["decoded"] * len(ids),
        )
        self.accelerator = _FakeAccelerator()
        self.reward_funcs = [
            lambda prompts, completions, completion_ids, **kwargs: [
                0.0 for _ in prompts
            ]
        ]
        self.reward_func_names = ["reward_0"]
        self.reward_weights = torch.tensor([1.0], dtype=torch.float32)
        self.num_generations = int(getattr(cfg, "num_generations", 1) or 1)
        self.num_iterations = 1
        self.mask_truncated_completions = False
        self.scale_rewards = False
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._textual_logs = {
            "prompt": [],
            "completion": [],
            "rewards": defaultdict(list),
            "advantages": [],
        }
        self.state = SimpleNamespace(
            global_step=0,
            num_input_tokens_seen=0.0,
            epoch=0.0,
            max_steps=0,
        )
        self.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])
        self.lr_scheduler = None
        if self.parent_received_use_vllm and self.parent_vllm_mode == "server":
            self.parent_vllm_init_calls += 1
            self.vllm_client = _FakeVLLMClient()
        else:
            self.vllm_client = None
        self.vllm_mode = str(getattr(cfg, "vllm_mode", "server"))
        self.callback_handler = SimpleNamespace(callbacks=[])
        self._train_batch_size = 1
        self.train_dataset = []
        self.data_collator = lambda x: x
        self._step = 0
        self._buffered_inputs = None

    def add_callback(self, cb: Any) -> None:
        self.callback_handler.callbacks.append(cb)

    def compute_loss(
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
    ) -> torch.Tensor:
        del model, inputs, return_outputs
        return torch.tensor(0.0)

    def _prepare_inputs(self, inputs: Any) -> Any:
        return inputs

    def _generate_and_score_completions(
        self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        self.parent_generate_calls += 1
        total = len(inputs)
        prompt_ids = torch.zeros((total, 1), dtype=torch.long)
        prompt_mask = torch.ones((total, 1), dtype=torch.long)
        completion_ids = torch.ones((total, 2), dtype=torch.long)
        completion_mask = torch.ones_like(completion_ids)
        advantages = torch.zeros((total,), dtype=torch.float32)
        old_per_token_logps = (
            torch.arange(total, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, completion_ids.size(1))
            * 0.01
        )
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }

    def _get_per_token_logps(
        self,
        model: Any,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int,
    ) -> torch.Tensor:
        del model, attention_mask, batch_size
        return torch.zeros(
            (prompt_completion_ids.size(0), logits_to_keep), dtype=torch.float32
        )

    def log(self, logs: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args, kwargs
        return logs


class _AttrWrappedModule(torch.nn.Module):
    def __init__(self, attr_name: str, module: torch.nn.Module) -> None:
        super().__init__()
        setattr(self, attr_name, module)


_WrappedTrainer = build_custom_grpo_trainer(_ParentTrainerStub)


def _make_args(
    train_grpo_objective: Any,
    *,
    use_vllm: bool = False,
    maxent_alpha: float = 0.1,
) -> SimpleNamespace:
    return SimpleNamespace(
        train_grpo_objective=train_grpo_objective,
        maxent_alpha=maxent_alpha,
        use_vllm=use_vllm,
        vllm_mode="server",
        num_generations=2,
        steps_per_generation=1,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_drop_last=False,
        dataloader_prefetch_factor=None,
        process_index=0,
        kl_target=0.0,
        kl_horizon=0,
        kl_ctl_step_size=0.0,
        maxent_tau=0.0,
        beta=0.0,
    )


def _make_trainer(
    train_grpo_objective: bool,
    *,
    use_vllm: bool = False,
    maxent_alpha: float = 0.1,
) -> Any:
    return _WrappedTrainer(
        args=_make_args(
            train_grpo_objective,
            use_vllm=use_vllm,
            maxent_alpha=maxent_alpha,
        ),
    )


@pytest.mark.parametrize("train_grpo_objective", [True, False])
def test_parent_vllm_init_kept_for_native_trl_pathways(
    train_grpo_objective: bool,
) -> None:
    trainer = _make_trainer(
        train_grpo_objective,
        use_vllm=True,
        maxent_alpha=0.1,
    )

    assert trainer.args.use_vllm is True
    # Both objectives stay on parent TRL rollout/compute pathways.
    assert trainer.parent_received_use_vllm is True
    assert trainer.parent_vllm_init_calls == 1
    assert trainer.vllm_client is not None


def test_objective_flag_string_values_route_correctly() -> None:
    grpo_like = _make_trainer(
        train_grpo_objective="true",
        use_vllm=True,
        maxent_alpha=0.1,
    )
    maxent_like = _make_trainer(
        train_grpo_objective="false",
        use_vllm=True,
        maxent_alpha=0.1,
    )

    assert grpo_like.maxent_enabled is False
    assert grpo_like.parent_received_use_vllm is True
    assert grpo_like.parent_vllm_init_calls == 1

    assert maxent_like.maxent_enabled is True
    assert maxent_like.parent_received_use_vllm is True
    assert maxent_like.parent_vllm_init_calls == 1


def test_maxent_alpha_zero_routes_to_native_grpo_path() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        use_vllm=True,
        maxent_alpha=0.0,
    )

    assert trainer.maxent_enabled is True
    assert trainer.maxent_alpha == pytest.approx(0.0)
    assert trainer.parent_received_use_vllm is True
    assert trainer.parent_vllm_init_calls == 1

    loss = trainer.compute_loss(
        model=None,
        inputs=[{"prompt": "hello", "answer": "world"}],
        return_outputs=False,
    )
    assert float(loss.item()) == pytest.approx(0.0)

    outputs = trainer._generate_and_score_completions(
        [
            {"prompt": "hello", "answer": "world"},
            {"prompt": "hi", "answer": "earth"},
        ]
    )
    assert isinstance(outputs, dict)
    assert trainer.parent_generate_calls == 1
    diversity_keys = [
        key
        for key in trainer._metrics["train"].keys()
        if key.startswith("completions/diversity/")
    ]
    assert diversity_keys


def test_compute_loss_routes_by_objective() -> None:
    grpo_trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.1)
    maxent_trainer = _make_trainer(train_grpo_objective=False, maxent_alpha=0.1)
    for trainer in (grpo_trainer, maxent_trainer):
        loss = trainer.compute_loss(
            model=None,
            inputs=[{"prompt": "hello", "answer": "world"}],
            return_outputs=False,
        )
        # Both objectives now route through native parent compute_loss.
        assert float(loss.item()) == pytest.approx(0.0)


def test_generate_and_score_routes_by_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer_mod,
        "_apply_eos_completion_mask",
        lambda completion_ids, eos_token_id: torch.ones_like(
            completion_ids, dtype=torch.long
        ),
    )

    inputs = [{"prompt": "p0", "answer": "a0"}, {"prompt": "p1", "answer": "a1"}]
    results: Dict[str, Any] = {}
    for label, is_grpo in (("grpo", True), ("maxent", False)):
        trainer = _make_trainer(
            train_grpo_objective=is_grpo,
            maxent_alpha=0.1,
        )
        trainer.model.training = True
        outputs = trainer._generate_and_score_completions(inputs)
        assert isinstance(outputs, dict)
        results[label] = trainer

    grpo_trainer = results["grpo"]
    maxent_trainer = results["maxent"]

    assert grpo_trainer.parent_generate_calls == 1
    assert maxent_trainer.parent_generate_calls == 1
    grpo_diversity_keys = [
        key
        for key in grpo_trainer._metrics["train"].keys()
        if key.startswith("completions/diversity/")
    ]
    maxent_diversity_keys = [
        key
        for key in maxent_trainer._metrics["train"].keys()
        if key.startswith("completions/diversity/")
    ]
    assert grpo_diversity_keys
    assert maxent_diversity_keys


def test_maxent_alpha_applies_advantage_bonus_in_native_path() -> None:
    trainer = _make_trainer(train_grpo_objective=False, maxent_alpha=0.2)
    trainer.model.training = True
    trainer.num_generations = 3

    trainer.processing_class.batch_decode = lambda ids, skip_special_tokens=True: [
        ("alpha beta" if idx % 3 in (0, 1) else "gamma delta")
        for idx in range(len(ids))
    ]
    inputs = [
        {"prompt": "p0", "answer": "a0"},
        {"prompt": "p1", "answer": "a1"},
        {"prompt": "p2", "answer": "a2"},
    ]
    outputs = trainer._generate_and_score_completions(inputs)
    assert isinstance(outputs.get("advantages"), torch.Tensor)
    adv = outputs["advantages"]
    assert float(adv.abs().sum().item()) > 0.0
    assert trainer.parent_generate_calls == 1
    assert trainer._metrics["train"]["maxent/alpha"][-1] == pytest.approx(0.2)


def test_eval_logs_pass_at_8_metric() -> None:
    trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.0)
    trainer.model.training = False
    trainer.num_generations = 8
    trainer.reward_funcs = [
        lambda prompts, completions, completion_ids, **kwargs: [
            1.0 if idx == 0 else 0.0 for idx, _ in enumerate(prompts)
        ]
    ]
    trainer.reward_weights = torch.tensor([1.0], dtype=torch.float32)

    outputs = trainer._generate_and_score_completions(
        [{"prompt": "p0", "answer": "a0"} for _ in range(8)]
        + [{"prompt": "p1", "answer": "a1"} for _ in range(8)]
    )
    assert isinstance(outputs, dict)
    assert trainer.parent_generate_calls == 1
    assert trainer._metrics["eval"]["pass_at_8"][-1] == pytest.approx(0.5)
    assert trainer._metrics["eval"]["pass_at_1"][-1] == pytest.approx(0.5)
    assert trainer._metrics["eval"]["mean_at_8"][-1] == pytest.approx(1.0 / 16.0)


def test_eval_pass_at_8_ignores_small_shaping_rewards() -> None:
    trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.0)
    trainer.model.training = False
    trainer.num_generations = 8

    # Wrong-but-formatted samples can carry small positive shaping rewards.
    # Pass@k should only count solved samples (reward ~= 1.0), not shaping.
    def _reward_fn(prompts, completions, completion_ids, **kwargs):
        del completions, completion_ids, kwargs
        values = [0.05 for _ in prompts]
        values[8] = 1.0
        return values

    trainer.reward_funcs = [_reward_fn]
    trainer.reward_weights = torch.tensor([1.0], dtype=torch.float32)

    outputs = trainer._generate_and_score_completions(
        [{"prompt": "p0", "answer": "a0"} for _ in range(8)]
        + [{"prompt": "p1", "answer": "a1"} for _ in range(8)]
    )
    assert isinstance(outputs, dict)
    assert trainer.parent_generate_calls == 1
    assert trainer._metrics["eval"]["pass_at_8"][-1] == pytest.approx(0.5)
    assert trainer._metrics["eval"]["pass_at_1"][-1] == pytest.approx(0.0)
    assert trainer._metrics["eval"]["mean_at_8"][-1] == pytest.approx(1.0 / 16.0)


def test_eval_pass_at_8_uses_math_answer_correctness() -> None:
    trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.0)
    trainer.model.training = False
    trainer.num_generations = 8
    trainer.reward_funcs = [pure_accuracy_reward_math]
    trainer.reward_weights = torch.tensor([1.0], dtype=torch.float32)

    completions = (
        ["<answer>42</answer>"]
        + ["<answer>0</answer>" for _ in range(7)]
        + ["<answer>0</answer>" for _ in range(8)]
    )
    trainer.processing_class.batch_decode = (
        lambda ids, skip_special_tokens=True: completions[: len(ids)]
    )
    outputs = trainer._generate_and_score_completions(
        [{"prompt": "p0", "answer": "42"} for _ in range(8)]
        + [{"prompt": "p1", "answer": "42"} for _ in range(8)]
    )
    assert isinstance(outputs, dict)
    assert trainer.parent_generate_calls == 1
    assert trainer._metrics["eval"]["pass_at_8"][-1] == pytest.approx(0.5)
    assert trainer._metrics["eval"]["pass_at_1"][-1] == pytest.approx(0.5)
    assert trainer._metrics["eval"]["mean_at_8"][-1] == pytest.approx(1.0 / 16.0)


def test_reference_model_ema_updates_with_warmup_and_interval() -> None:
    trainer = _make_trainer(train_grpo_objective=False, maxent_alpha=1.0)
    trainer.model = torch.nn.Linear(2, 1, bias=False)
    trainer.ref_model = torch.nn.Linear(2, 1, bias=False)
    trainer.model.train()
    trainer.ref_model.eval()

    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
        trainer.ref_model.weight.zero_()

    trainer.args.maxent_reference_ema_enabled = True
    trainer.args.maxent_share_reference_model = False
    trainer.args.maxent_reference_ema_beta = 0.5
    trainer.args.maxent_reference_ema_warmup_steps = 5
    trainer.args.maxent_reference_ema_update_interval = 2

    trainer.state.global_step = 4
    trainer.compute_loss(model=None, inputs=[{"prompt": "p", "answer": "a"}])
    assert trainer.ref_model.weight.detach().cpu().flatten().tolist() == pytest.approx(
        [0.0, 0.0]
    )

    trainer.state.global_step = 5
    trainer.compute_loss(model=None, inputs=[{"prompt": "p", "answer": "a"}])
    assert trainer.ref_model.weight.detach().cpu().flatten().tolist() == pytest.approx(
        [0.5, 0.5]
    )

    trainer.state.global_step = 6
    trainer.compute_loss(model=None, inputs=[{"prompt": "p", "answer": "a"}])
    assert trainer.ref_model.weight.detach().cpu().flatten().tolist() == pytest.approx(
        [0.5, 0.5]
    )

    # Same step should not apply EMA twice.
    trainer.compute_loss(model=None, inputs=[{"prompt": "p", "answer": "a"}])
    assert trainer.ref_model.weight.detach().cpu().flatten().tolist() == pytest.approx(
        [0.5, 0.5]
    )

    trainer.state.global_step = 7
    trainer.compute_loss(model=None, inputs=[{"prompt": "p", "answer": "a"}])
    assert trainer.ref_model.weight.detach().cpu().flatten().tolist() == pytest.approx(
        [0.75, 0.75]
    )
    assert trainer._metrics["train"]["maxent/ref_ema_applied"][-1] == pytest.approx(1.0)


def test_reference_model_ema_matches_prefixed_parameter_names() -> None:
    trainer = _make_trainer(train_grpo_objective=False, maxent_alpha=1.0)
    policy = _AttrWrappedModule("module", torch.nn.Linear(2, 1, bias=False))
    reference = _AttrWrappedModule("model", torch.nn.Linear(2, 1, bias=False))
    trainer.model = policy
    trainer.ref_model = reference
    trainer.model.train()
    trainer.ref_model.eval()

    with torch.no_grad():
        policy.module.weight.fill_(1.0)
        reference.model.weight.zero_()

    trainer.args.maxent_reference_ema_enabled = True
    trainer.args.maxent_share_reference_model = False
    trainer.args.maxent_reference_ema_beta = 0.5
    trainer.args.maxent_reference_ema_warmup_steps = 0
    trainer.args.maxent_reference_ema_update_interval = 1

    trainer.state.global_step = 1
    trainer.compute_loss(model=None, inputs=[{"prompt": "p", "answer": "a"}])

    assert reference.model.weight.detach().cpu().flatten().tolist() == pytest.approx(
        [0.5, 0.5]
    )
    assert trainer._metrics["train"]["maxent/ref_ema_updated_frac"][
        -1
    ] == pytest.approx(1.0)
    assert trainer._metrics["train"]["maxent/ref_ema_alias_hit_frac"][
        -1
    ] == pytest.approx(1.0)
