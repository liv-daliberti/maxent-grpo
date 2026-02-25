from __future__ import annotations

from collections import defaultdict
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch

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
            lambda prompts, completions, completion_ids, **kwargs: [0.0 for _ in prompts]
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
            torch.arange(total, dtype=torch.float32).unsqueeze(1).repeat(1, completion_ids.size(1))
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


def _prepared_rollout_stub() -> Any:
    completion_ids = torch.tensor([[11, 2, 0], [12, 0, 0]], dtype=torch.long)
    score_batch = SimpleNamespace(
        prompt_entries=[object(), object()],
        max_prompt_len=1,
        pad_token_id=0,
        completion_ids=completion_ids,
    )
    reward_comp = SimpleNamespace(
        per_reward_values={"reward_0": [0.5, 0.7]},
        pairs=SimpleNamespace(prompts=["p0", "p1"], completions=["c0", "c1"]),
        advantage=SimpleNamespace(samples=[0.25, -0.25]),
    )
    return SimpleNamespace(
        batch_stats=SimpleNamespace(score_batch=score_batch),
        reward_comp=reward_comp,
        seed_metrics={},
        diversity_metrics={"distinct_1": 0.33},
        total_input_tokens=2.0,
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
    assert trainer._shared_rollout_vllm is False


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
    assert grpo_like._shared_rollout_vllm is False

    assert maxent_like.maxent_enabled is True
    assert maxent_like.parent_received_use_vllm is True
    assert maxent_like.parent_vllm_init_calls == 1
    assert maxent_like._shared_rollout_vllm is False


def test_maxent_alpha_zero_routes_to_native_grpo_path() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        use_vllm=True,
        maxent_alpha=0.0,
    )

    assert trainer.maxent_enabled is True
    assert trainer.maxent_alpha == pytest.approx(0.0)
    assert trainer._maxent_custom_path is False
    assert trainer.parent_received_use_vllm is True
    assert trainer.parent_vllm_init_calls == 1
    assert trainer._shared_rollout_vllm is False

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


def test_compute_loss_routes_by_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grpo_trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.1)
    maxent_trainer = _make_trainer(train_grpo_objective=False, maxent_alpha=0.1)

    monkeypatch.setattr(
        trainer_mod,
        "maybe_apply_chat_template",
        lambda example, _tok: {"prompt": example["prompt"]},
    )
    for trainer in (grpo_trainer, maxent_trainer):
        loss = trainer.compute_loss(
            model=None,
            inputs=[{"prompt": "hello", "answer": "world"}],
            return_outputs=False,
        )
        # Both objectives now route through native parent compute_loss.
        assert float(loss.item()) == pytest.approx(0.0)


def test_grpo_objective_runner_uses_native_parent_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.1)

    def _should_not_run(self: Any, **_kwargs: Any) -> torch.Tensor:
        raise AssertionError("Shared MaxEnt loss core should never run for GRPO.")

    trainer._compute_shared_objective_loss = MethodType(_should_not_run, trainer)
    loss = trainer._compute_grpo_objective_loss(
        model=None,
        inputs=[{"prompt": "hello", "answer": "world"}],
        return_outputs=False,
        mode="train",
        ctx=None,
        prepared=None,
    )
    assert float(loss.item()) == pytest.approx(0.0)


def test_generate_and_score_routes_by_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer_mod,
        "maybe_apply_chat_template",
        lambda example, _tok: {"prompt": example["prompt"]},
    )
    monkeypatch.setattr(
        trainer_mod,
        "_prepare_prompt_slice",
        lambda prompt_entries, max_prompt_len, pad_token_id, completion_dtype, prompt_dtype: (
            torch.zeros((len(prompt_entries), 1), dtype=torch.long),
            torch.ones((len(prompt_entries), 1), dtype=torch.long),
            None,
        ),
    )
    monkeypatch.setattr(
        trainer_mod,
        "_apply_eos_completion_mask",
        lambda completion_ids, eos_token_id: torch.ones_like(completion_ids, dtype=torch.long),
    )

    inputs = [{"prompt": "p0", "answer": "a0"}, {"prompt": "p1", "answer": "a1"}]
    results: Dict[str, Tuple[Any, List[Dict[str, Any]]]] = {}
    for label, is_grpo in (("grpo", True), ("maxent", False)):
        trainer = _make_trainer(
            train_grpo_objective=is_grpo,
            maxent_alpha=0.1,
        )
        calls: List[Dict[str, Any]] = []

        def _spy_prepare_rollout(
            self: Any,
            mode: str,
            ctx: Any,
            generator: Any,
            batch: Dict[str, List[str]],
            *,
            use_generation_cache: bool,
        ) -> Any:
            calls.append(
                {
                    "mode": mode,
                    "use_generation_cache": use_generation_cache,
                    "batch_prompts": list(batch["prompt"]),
                }
            )
            return _prepared_rollout_stub()

        trainer._prepare_rollout_batch = MethodType(_spy_prepare_rollout, trainer)
        trainer.model.training = True
        outputs = trainer._generate_and_score_completions(inputs)
        assert isinstance(outputs, dict)
        results[label] = (trainer, calls)

    grpo_trainer, grpo_calls = results["grpo"]
    maxent_trainer, maxent_calls = results["maxent"]

    assert grpo_trainer.parent_generate_calls == 1
    assert len(grpo_calls) == 0
    assert maxent_trainer.parent_generate_calls == 1
    assert len(maxent_calls) == 0
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


def test_maxent_alpha_applies_advantage_bonus_in_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(train_grpo_objective=False, maxent_alpha=0.2)
    trainer.model.training = True
    trainer.num_generations = 3

    monkeypatch.setattr(
        trainer_mod,
        "maybe_apply_chat_template",
        lambda example, _tok: {"prompt": example["prompt"]},
    )
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
