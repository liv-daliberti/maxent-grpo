from __future__ import annotations

from collections import defaultdict
import math
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from maxent_grpo.rewards.basic import pure_accuracy_reward_math
from maxent_grpo.training import trl_trainer as trainer_mod
from maxent_grpo.training.trl_trainer import build_custom_grpo_trainer

pytestmark = pytest.mark.skipif(
    getattr(torch, "__MAXENT_STUB__", False),
    reason="requires real torch",
)


class _FakeAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.process_index = 0
        self.is_main_process = True

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def unwrap_model(self, model: Any) -> Any:
        return model

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
        self.parent_compute_loss_calls = 0
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
        self.temperature = 1.0
        self.epsilon_low = 0.2
        self.epsilon_high = 0.2
        self.loss_type = "bnpo"
        self.max_completion_length = 2
        self.use_liger_loss = False
        self.beta = float(getattr(cfg, "beta", 0.0) or 0.0)
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
        self.parent_compute_loss_calls += 1
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


class _UniformPolicyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 3) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.training = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> Any:
        del attention_mask, logits_to_keep
        batch, seq_len = input_ids.shape
        logits = torch.zeros((batch, seq_len, self.vocab_size), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: List[float]) -> None:
        super().__init__()
        self._logits = torch.tensor(logits, dtype=torch.float32)
        self.training = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> Any:
        del attention_mask, logits_to_keep
        batch, seq_len = input_ids.shape
        logits = self._logits.view(1, 1, -1).expand(batch, seq_len, -1).clone()
        return SimpleNamespace(logits=logits)


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
    maxent_objective_variant: str = "entropy",
) -> SimpleNamespace:
    maxent_tau = 1.0 if (not train_grpo_objective and maxent_objective_variant == "listwise") else 0.0
    return SimpleNamespace(
        train_grpo_objective=train_grpo_objective,
        maxent_alpha=maxent_alpha,
        maxent_objective_variant=maxent_objective_variant,
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
        controller_meta_enabled=False,
        controller_meta_method="analytic",
        controller_meta_lr=0.0,
        controller_meta_tau_lr=0.0,
        controller_meta_beta_lr=0.0,
        controller_meta_beta_grad_clip=0.0,
        controller_meta_update_interval=1,
        grpo_beta_controller_enabled=False,
        maxent_tau=maxent_tau,
        maxent_q_temperature=1.0,
        maxent_q_epsilon=1e-6,
        maxent_length_normalize_ref=True,
        maxent_logprob_chunk_size=0,
        maxent_reference_logprobs_source="auto",
        maxent_trl_reference_scoring=False,
        maxent_policy_entropy_mode="exact",
        maxent_use_clip_objective=False,
        maxent_clip_objective_coef=1.0,
        maxent_clip_adv_baseline=None,
        maxent_clip_range=None,
        beta=0.0,
        delta=None,
        maxent_reference_ema_enabled=True,
        maxent_share_reference_model=False,
        maxent_reference_ema_beta=0.995,
        maxent_reference_ema_warmup_steps=100,
        maxent_reference_ema_update_interval=10,
    )


def _make_trainer(
    train_grpo_objective: bool,
    *,
    use_vllm: bool = False,
    maxent_alpha: float = 0.1,
    maxent_objective_variant: str = "entropy",
) -> Any:
    return _WrappedTrainer(
        args=_make_args(
            train_grpo_objective,
            use_vllm=use_vllm,
            maxent_alpha=maxent_alpha,
            maxent_objective_variant=maxent_objective_variant,
        ),
    )


def _refresh_weighting(trainer: Any) -> None:
    trainer._maxent_weighting = trainer_mod.build_weighting_settings(trainer.args)
    trainer._maxent_controller_objective = trainer_mod.build_controller_objective(
        trainer.args,
        trainer._maxent_weighting,
    )
    trainer._sync_weighting_scalars()


def _install_real_logprob_scorer(trainer: Any) -> None:
    def _score(
        model: Any,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int,
    ) -> torch.Tensor:
        logps, _ = trainer._get_per_token_logps_and_entropy(
            model,
            prompt_completion_ids,
            attention_mask,
            logits_to_keep,
            entropy_mode="exact",
            batch_size=batch_size,
        )
        return logps

    trainer._get_per_token_logps = _score


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
    assert trainer.objective_routing.uses_entropy_regularized_loss is True
    assert trainer.objective_routing.uses_native_grpo_loss is False
    assert trainer.objective_routing.route_mode == "maxent_entropy"
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


def test_three_way_objective_routing_hits_expected_loss_implementation() -> None:
    grpo_trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.0)
    entropy_trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.2,
        maxent_objective_variant="entropy",
    )
    listwise_trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=3.0,
        maxent_objective_variant="listwise",
    )

    assert grpo_trainer.objective_routing.route_mode == "grpo"
    assert entropy_trainer.objective_routing.route_mode == "maxent_entropy"
    assert listwise_trainer.objective_routing.route_mode == "maxent_listwise"

    entropy_trainer.model = _UniformPolicyModel(vocab_size=3)
    listwise_trainer.model = _FixedLogitModel([math.log(0.75), math.log(0.25)])
    _install_real_logprob_scorer(listwise_trainer)

    grpo_trainer.compute_loss(
        model=None,
        inputs=[{"prompt": "hello", "answer": "world"}],
        return_outputs=False,
    )
    entropy_inputs = {
        "prompt_ids": torch.zeros((1, 1), dtype=torch.long),
        "prompt_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "completion_mask": torch.ones((1, 2), dtype=torch.long),
        "advantages": torch.zeros((1,), dtype=torch.float32),
        "old_per_token_logps": None,
    }
    entropy_trainer.compute_loss(
        model=entropy_trainer.model,
        inputs=entropy_inputs,
        return_outputs=False,
    )
    listwise_inputs = {
        "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "completion_mask": torch.ones((2, 2), dtype=torch.long),
        "advantages": torch.zeros((2,), dtype=torch.float32),
        "old_per_token_logps": None,
        "maxent_listwise_q": torch.tensor([[0.8, 0.2]], dtype=torch.float32),
    }
    listwise_trainer.compute_loss(
        model=listwise_trainer.model,
        inputs=listwise_inputs,
        return_outputs=False,
    )

    assert grpo_trainer.parent_compute_loss_calls == 1
    assert entropy_trainer.parent_compute_loss_calls == 0
    assert listwise_trainer.parent_compute_loss_calls == 0


def test_grpo_flag_overrides_maxent_variant_and_alpha() -> None:
    trainer = _make_trainer(
        train_grpo_objective=True,
        maxent_alpha=10.0,
        maxent_objective_variant="listwise",
    )

    loss = trainer.compute_loss(
        model=None,
        inputs=[{"prompt": "hello", "answer": "world"}],
        return_outputs=False,
    )

    assert float(loss.item()) == pytest.approx(0.0)
    assert trainer.parent_compute_loss_calls == 1


def test_listwise_trainer_requires_positive_tau() -> None:
    args = _make_args(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    args.maxent_tau = 0.0
    with pytest.raises(ValueError, match="maxent_tau > 0"):
        _WrappedTrainer(args=args)


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


def test_maxent_alpha_applies_true_entropy_loss_in_native_path() -> None:
    trainer = _make_trainer(train_grpo_objective=False, maxent_alpha=0.2)
    trainer.model = _UniformPolicyModel(vocab_size=3)
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
    assert float(adv.abs().sum().item()) == pytest.approx(0.0)
    loss = trainer.compute_loss(model=trainer.model, inputs=outputs, return_outputs=False)
    expected_entropy = math.log(3.0)
    assert float(loss.item()) == pytest.approx(-0.2 * expected_entropy, rel=1e-5)
    assert trainer.parent_generate_calls == 1
    assert trainer._metrics["train"]["maxent/alpha"][-1] == pytest.approx(0.2)
    assert trainer._metrics["train"]["maxent/policy_entropy_mean"][-1] == pytest.approx(
        expected_entropy, rel=1e-5
    )
    assert trainer._metrics["train"]["maxent/objective_variant_entropy"][-1] == pytest.approx(
        1.0
    )
    assert trainer._metrics["train"]["maxent/objective_variant_listwise"][-1] == pytest.approx(
        0.0
    )


def test_entropy_maxent_forces_exact_entropy_when_sample_mode_is_requested() -> None:
    trainer = _make_trainer(train_grpo_objective=False, maxent_alpha=0.2)
    trainer.model = _FixedLogitModel([4.0, -4.0, -4.0])
    trainer.args.maxent_policy_entropy_mode = "sample"

    inputs = {
        "prompt_ids": torch.zeros((1, 1), dtype=torch.long),
        "prompt_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "completion_mask": torch.ones((1, 2), dtype=torch.long),
        "advantages": torch.zeros((1,), dtype=torch.float32),
        "old_per_token_logps": None,
    }

    loss = trainer.compute_loss(model=trainer.model, inputs=inputs, return_outputs=False)
    logits = torch.tensor([4.0, -4.0, -4.0], dtype=torch.float32)
    probs = torch.softmax(logits, dim=0)
    exact_entropy = float((-(probs * probs.log())).sum().item())
    sample_cross_entropy = float(
        -0.5
        * (
            torch.log(probs[0]).item()
            + torch.log(probs[1]).item()
        )
    )

    assert float(loss.item()) == pytest.approx(-0.2 * exact_entropy, rel=1e-5)
    assert float(loss.item()) != pytest.approx(-0.2 * sample_cross_entropy, rel=1e-3)
    assert trainer._metrics["train"]["maxent/policy_entropy_mean"][-1] == pytest.approx(
        exact_entropy, rel=1e-5
    )


def test_entropy_maxent_ignores_listwise_knobs() -> None:
    base_inputs = {
        "prompt_ids": torch.zeros((1, 1), dtype=torch.long),
        "prompt_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "completion_mask": torch.ones((1, 2), dtype=torch.long),
        "advantages": torch.zeros((1,), dtype=torch.float32),
        "old_per_token_logps": None,
    }

    trainer_a = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.2,
        maxent_objective_variant="entropy",
    )
    trainer_b = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.2,
        maxent_objective_variant="entropy",
    )
    trainer_a.model = _UniformPolicyModel(vocab_size=3)
    trainer_b.model = _UniformPolicyModel(vocab_size=3)
    trainer_a.args.maxent_tau = 0.1
    trainer_a.args.maxent_q_temperature = 0.3
    trainer_b.args.maxent_tau = 5.0
    trainer_b.args.maxent_q_temperature = 3.0

    loss_a = trainer_a.compute_loss(
        model=trainer_a.model, inputs=base_inputs, return_outputs=False
    )
    loss_b = trainer_b.compute_loss(
        model=trainer_b.model, inputs=base_inputs, return_outputs=False
    )

    assert float(loss_a.item()) == pytest.approx(float(loss_b.item()), rel=1e-6)


def test_entropy_maxent_prefers_higher_entropy_policy() -> None:
    trainer_uniform = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.2,
        maxent_objective_variant="entropy",
    )
    trainer_peaked = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.2,
        maxent_objective_variant="entropy",
    )
    trainer_uniform.model = _UniformPolicyModel(vocab_size=3)
    trainer_peaked.model = _FixedLogitModel([4.0, -4.0, -4.0])

    inputs = {
        "prompt_ids": torch.zeros((1, 1), dtype=torch.long),
        "prompt_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "completion_mask": torch.ones((1, 2), dtype=torch.long),
        "advantages": torch.zeros((1,), dtype=torch.float32),
        "old_per_token_logps": None,
    }

    uniform_loss = trainer_uniform.compute_loss(
        model=trainer_uniform.model,
        inputs=inputs,
        return_outputs=False,
    )
    peaked_loss = trainer_peaked.compute_loss(
        model=trainer_peaked.model,
        inputs=inputs,
        return_outputs=False,
    )

    assert float(uniform_loss.item()) < float(peaked_loss.item())


def test_entropy_maxent_keeps_reference_model_in_the_kl_term() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.2,
        maxent_objective_variant="entropy",
    )
    trainer.model = _UniformPolicyModel(vocab_size=3)
    trainer.ref_model = _FixedLogitModel([4.0, -4.0, -4.0])
    trainer.beta = 0.3
    trainer.args.beta = 0.3
    trainer.args.maxent_reference_logprobs_source = "model"
    trainer.args.maxent_trl_reference_scoring = True
    _install_real_logprob_scorer(trainer)

    inputs = {
        "prompt_ids": torch.zeros((1, 1), dtype=torch.long),
        "prompt_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "completion_mask": torch.ones((1, 2), dtype=torch.long),
        "advantages": torch.zeros((1,), dtype=torch.float32),
        "old_per_token_logps": None,
    }

    loss = trainer.compute_loss(
        model=trainer.model,
        inputs=inputs,
        return_outputs=False,
    )

    logits_policy = torch.zeros(3, dtype=torch.float32)
    logits_ref = torch.tensor([4.0, -4.0, -4.0], dtype=torch.float32)
    logp_policy = torch.log_softmax(logits_policy, dim=0)
    logp_ref = torch.log_softmax(logits_ref, dim=0)
    completion_ids = inputs["completion_ids"][0]
    per_token_logp = logp_policy[completion_ids]
    per_token_ref = logp_ref[completion_ids]
    per_token_kl = torch.exp(per_token_ref - per_token_logp) - (
        per_token_ref - per_token_logp
    ) - 1.0
    expected_kl = float(per_token_kl.mean().item())
    expected_entropy = float((-(torch.softmax(logits_policy, dim=0) * logp_policy)).sum())
    expected_loss = 0.3 * expected_kl - 0.2 * expected_entropy

    assert float(loss.item()) == pytest.approx(expected_loss, rel=1e-5)
    assert trainer._metrics["train"]["kl"][-1] == pytest.approx(expected_kl, rel=1e-5)
    assert trainer._metrics["train"]["maxent/policy_entropy_mean"][-1] == pytest.approx(
        expected_entropy, rel=1e-5
    )


def test_entropy_controller_meta_updates_beta_from_kl() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.2,
        maxent_objective_variant="entropy",
    )
    trainer.model = _UniformPolicyModel(vocab_size=3)
    trainer.ref_model = _FixedLogitModel([4.0, -4.0, -4.0])
    trainer.args.beta = 0.2
    trainer.beta = 0.2
    trainer.args.kl_target = 0.1
    trainer.args.controller_meta_enabled = True
    trainer.args.controller_meta_method = "analytic"
    trainer.args.controller_meta_beta_lr = 0.5
    trainer.args.maxent_reference_logprobs_source = "model"
    trainer.args.maxent_trl_reference_scoring = True
    trainer.state.global_step = 1
    _refresh_weighting(trainer)
    _install_real_logprob_scorer(trainer)

    inputs = {
        "prompt_ids": torch.zeros((1, 1), dtype=torch.long),
        "prompt_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "completion_mask": torch.ones((1, 2), dtype=torch.long),
        "advantages": torch.zeros((1,), dtype=torch.float32),
        "old_per_token_logps": None,
    }

    before = trainer.beta
    trainer.compute_loss(model=trainer.model, inputs=inputs, return_outputs=False)

    assert trainer.beta > before
    assert trainer.tau == pytest.approx(0.0)


def test_grpo_controller_meta_updates_beta_from_logged_kl() -> None:
    trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.0)
    trainer.args.beta = 0.2
    trainer.beta = 0.2
    trainer.args.kl_target = 0.1
    trainer.args.controller_meta_enabled = True
    trainer.args.controller_meta_method = "first_order"
    trainer.args.controller_meta_lr = 0.5
    trainer.state.global_step = 1
    _refresh_weighting(trainer)

    def _fake_native_loss(**kwargs: Any) -> torch.Tensor:
        del kwargs
        trainer._metrics["train"]["kl"].append(0.3)
        return torch.tensor(0.0)

    trainer._compute_grpo_native_loss = _fake_native_loss

    before = trainer.beta
    trainer.compute_loss(
        model=None,
        inputs=[{"prompt": "hello", "answer": "world"}],
        return_outputs=False,
    )

    assert trainer.beta > before
    assert trainer.tau == pytest.approx(0.0)


def test_listwise_generate_and_score_prepares_grouped_q_targets() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer.num_generations = 2
    trainer.reward_funcs = [
        lambda prompts, completions, completion_ids, **kwargs: [
            float(idx) for idx, _ in enumerate(prompts)
        ]
    ]
    trainer.reward_weights = torch.tensor([1.0], dtype=torch.float32)

    inputs = [
        {"prompt": "p0", "answer": "a0"},
        {"prompt": "p0", "answer": "a0"},
        {"prompt": "p1", "answer": "a1"},
        {"prompt": "p1", "answer": "a1"},
    ]
    outputs = trainer._generate_and_score_completions(inputs)

    assert "maxent_listwise_q" in outputs
    q_grouped = outputs["maxent_listwise_q"]
    assert isinstance(q_grouped, torch.Tensor)
    expected = torch.stack(
        [
            torch.softmax(torch.tensor([0.0, 1.0]), dim=0),
            torch.softmax(torch.tensor([2.0, 3.0]), dim=0),
        ]
    ).to(torch.float32)
    assert torch.allclose(q_grouped, expected, atol=1e-6, rtol=1e-6)


def test_listwise_rollout_targets_use_gathered_rewards() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer.num_generations = 2
    trainer.accelerator.process_index = 1
    trainer._recompute_local_rewards_for_outputs = lambda inputs, outputs: torch.tensor(
        [0.0, 0.0], dtype=torch.float32
    )

    original_gather = trainer_mod.gather
    trainer_mod.gather = lambda value: torch.tensor(
        [1.0, 3.0, 0.0, 2.0],
        dtype=value.dtype,
        device=value.device,
    )
    try:
        outputs: Dict[str, Any] = {}
        trainer._prepare_listwise_rollout_targets(
            [{"prompt": "p1", "answer": "a1"}, {"prompt": "p1", "answer": "a1"}],
            outputs,
        )
    finally:
        trainer_mod.gather = original_gather

    expected = torch.softmax(torch.tensor([[0.0, 2.0]], dtype=torch.float32), dim=1)
    assert "maxent_listwise_q" in outputs
    assert torch.allclose(outputs["maxent_listwise_q"], expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        outputs["maxent_listwise_rewards"],
        torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def test_listwise_q_temperature_changes_rollout_targets_once() -> None:
    sharp = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    smooth = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    for trainer in (sharp, smooth):
        trainer.num_generations = 2
        trainer.reward_funcs = [
            lambda prompts, completions, completion_ids, **kwargs: [
                float(idx) for idx, _ in enumerate(prompts)
            ]
        ]
        trainer.reward_weights = torch.tensor([1.0], dtype=torch.float32)
    sharp.args.maxent_q_temperature = 0.25
    smooth.args.maxent_q_temperature = 4.0

    inputs = [
        {"prompt": "p0", "answer": "a0"},
        {"prompt": "p0", "answer": "a0"},
        {"prompt": "p1", "answer": "a1"},
        {"prompt": "p1", "answer": "a1"},
    ]
    sharp_outputs = sharp._generate_and_score_completions(inputs)
    smooth_outputs = smooth._generate_and_score_completions(inputs)

    sharp_q = sharp_outputs["maxent_listwise_q"]
    smooth_q = smooth_outputs["maxent_listwise_q"]

    assert float(sharp_q[0, 1].item()) > float(smooth_q[0, 1].item())
    sharp_entropy = float((-(sharp_q[0] * sharp_q[0].log())).sum().item())
    smooth_entropy = float((-(smooth_q[0] * smooth_q[0].log())).sum().item())
    assert smooth_entropy > sharp_entropy


def test_listwise_prepare_inputs_preserves_whole_prompt_groups() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer.num_generations = 2
    trainer.args.num_generations = 2
    trainer.args.steps_per_generation = 2
    trainer.args.per_device_train_batch_size = 2
    trainer.model.training = True
    trainer.reward_funcs = [
        lambda prompts, completions, completion_ids, **kwargs: [
            float(idx) for idx, _ in enumerate(prompts)
        ]
    ]
    trainer.reward_weights = torch.tensor([1.0], dtype=torch.float32)

    generation_batch = [
        {"prompt": "p0", "answer": "a0"},
        {"prompt": "p0", "answer": "a0"},
        {"prompt": "p1", "answer": "a1"},
        {"prompt": "p1", "answer": "a1"},
    ]

    first = trainer._prepare_inputs(generation_batch)
    second = trainer._prepare_inputs(generation_batch)

    for chunk in (first, second):
        assert chunk["prompt_ids"].shape[0] == 2
        assert chunk["completion_ids"].shape[0] == 2
        assert chunk["maxent_listwise_q"].shape == (1, 2)
        assert torch.allclose(
            chunk["maxent_listwise_q"].sum(dim=1),
            torch.ones((1,), dtype=torch.float32),
        )


def test_listwise_prepare_inputs_rejects_incomplete_prompt_groups() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer.num_generations = 2
    trainer.args.num_generations = 2
    trainer.model.training = True
    trainer.reward_funcs = [
        lambda prompts, completions, completion_ids, **kwargs: [
            float(idx) for idx, _ in enumerate(prompts)
        ]
    ]
    trainer.reward_weights = torch.tensor([1.0], dtype=torch.float32)

    generation_batch = [
        {"prompt": "p0", "answer": "a0"},
        {"prompt": "p0", "answer": "a0"},
        {"prompt": "p1", "answer": "a1"},
    ]

    with pytest.raises(ValueError, match="whole prompt groups"):
        trainer._prepare_inputs(generation_batch)


def test_listwise_maxent_loss_matches_q_weighted_sequence_cross_entropy() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer.model = _FixedLogitModel([math.log(0.75), math.log(0.25)])
    trainer.args.maxent_tau = 1.0
    trainer.args.beta = 0.0
    trainer.args.maxent_q_temperature = 1.0
    trainer.args.maxent_reference_logprobs_source = "none"
    trainer.args.maxent_trl_reference_scoring = False
    _refresh_weighting(trainer)
    _install_real_logprob_scorer(trainer)

    prepared = {
        "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "completion_mask": torch.ones((2, 2), dtype=torch.long),
        "advantages": torch.zeros((2,), dtype=torch.float32),
        "old_per_token_logps": None,
        "maxent_listwise_q": torch.tensor([[0.8, 0.2]], dtype=torch.float32),
    }

    loss = trainer.compute_loss(model=trainer.model, inputs=prepared, return_outputs=False)
    expected_probs = torch.tensor([0.9, 0.1], dtype=torch.float32)
    expected_loss = -(
        0.8 * math.log(float(expected_probs[0]))
        + 0.2 * math.log(float(expected_probs[1]))
    )

    assert float(loss.item()) == pytest.approx(expected_loss, rel=1e-5)
    assert trainer._metrics["train"]["maxent/objective_variant_listwise"][-1] == pytest.approx(
        1.0
    )
    assert trainer._metrics["train"]["weight_entropy"][-1] == pytest.approx(
        -(0.8 * math.log(0.8) + 0.2 * math.log(0.2)),
        rel=1e-5,
    )


def test_listwise_maxent_requires_rollout_q_targets() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer.model = _FixedLogitModel([math.log(0.75), math.log(0.25)])
    trainer.args.maxent_reference_logprobs_source = "none"
    trainer.args.maxent_trl_reference_scoring = False
    _refresh_weighting(trainer)
    _install_real_logprob_scorer(trainer)

    prepared = {
        "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "completion_mask": torch.ones((2, 2), dtype=torch.long),
        "advantages": torch.zeros((2,), dtype=torch.float32),
        "old_per_token_logps": None,
    }

    with pytest.raises(ValueError, match="Listwise MaxEnt requires rollout q targets"):
        trainer.compute_loss(model=trainer.model, inputs=prepared, return_outputs=False)


def test_listwise_maxent_rejects_misaligned_q_shape() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer.model = _FixedLogitModel([math.log(0.75), math.log(0.25)])
    trainer.args.maxent_reference_logprobs_source = "none"
    trainer.args.maxent_trl_reference_scoring = False
    _refresh_weighting(trainer)
    _install_real_logprob_scorer(trainer)

    prepared = {
        "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "completion_mask": torch.ones((2, 2), dtype=torch.long),
        "advantages": torch.zeros((2,), dtype=torch.float32),
        "old_per_token_logps": None,
        "maxent_listwise_q": torch.tensor(
            [[0.8, 0.2], [0.4, 0.6]],
            dtype=torch.float32,
        ),
    }

    with pytest.raises(ValueError, match="shape"):
        trainer.compute_loss(model=trainer.model, inputs=prepared, return_outputs=False)


def test_listwise_maxent_ignores_alpha_scale() -> None:
    trainer_a = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer_b = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=10.0,
        maxent_objective_variant="listwise",
    )
    for trainer in (trainer_a, trainer_b):
        trainer.model = _FixedLogitModel([math.log(0.75), math.log(0.25)])
        trainer.args.maxent_tau = 1.0
        trainer.args.beta = 0.0
        trainer.args.maxent_reference_logprobs_source = "none"
        trainer.args.maxent_trl_reference_scoring = False
        _refresh_weighting(trainer)
        _install_real_logprob_scorer(trainer)

    inputs = {
        "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "completion_mask": torch.ones((2, 2), dtype=torch.long),
        "advantages": torch.zeros((2,), dtype=torch.float32),
        "old_per_token_logps": None,
        "maxent_listwise_q": torch.tensor([[0.8, 0.2]], dtype=torch.float32),
    }

    loss_a = trainer_a.compute_loss(
        model=trainer_a.model, inputs=inputs, return_outputs=False
    )
    loss_b = trainer_b.compute_loss(
        model=trainer_b.model, inputs=inputs, return_outputs=False
    )

    assert float(loss_a.item()) == pytest.approx(float(loss_b.item()), rel=1e-6)


def test_listwise_fixed_q_targets_ignore_q_temperature_knob() -> None:
    trainer_a = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer_b = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    for trainer in (trainer_a, trainer_b):
        trainer.model = _FixedLogitModel([math.log(0.75), math.log(0.25)])
        trainer.args.maxent_tau = 1.0
        trainer.args.beta = 0.0
        trainer.args.maxent_reference_logprobs_source = "none"
        trainer.args.maxent_trl_reference_scoring = False
        _install_real_logprob_scorer(trainer)
    trainer_a.args.maxent_q_temperature = 0.25
    trainer_b.args.maxent_q_temperature = 4.0
    _refresh_weighting(trainer_a)
    _refresh_weighting(trainer_b)

    inputs = {
        "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "completion_mask": torch.ones((2, 2), dtype=torch.long),
        "advantages": torch.zeros((2,), dtype=torch.float32),
        "old_per_token_logps": None,
        "maxent_listwise_q": torch.tensor([[0.8, 0.2]], dtype=torch.float32),
    }

    loss_a = trainer_a.compute_loss(
        model=trainer_a.model, inputs=inputs, return_outputs=False
    )
    loss_b = trainer_b.compute_loss(
        model=trainer_b.model, inputs=inputs, return_outputs=False
    )

    assert float(loss_a.item()) == pytest.approx(float(loss_b.item()), rel=1e-6)


def test_listwise_maxent_prefers_policy_closer_to_target_distribution() -> None:
    trainer_match = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer_mismatch = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer_match.model = _FixedLogitModel([math.log(0.75), math.log(0.25)])
    trainer_mismatch.model = _FixedLogitModel([math.log(0.25), math.log(0.75)])
    for trainer in (trainer_match, trainer_mismatch):
        trainer.args.maxent_tau = 1.0
        trainer.args.beta = 0.0
        trainer.args.maxent_reference_logprobs_source = "none"
        trainer.args.maxent_trl_reference_scoring = False
        _refresh_weighting(trainer)
        _install_real_logprob_scorer(trainer)

    inputs = {
        "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "completion_mask": torch.ones((2, 2), dtype=torch.long),
        "advantages": torch.zeros((2,), dtype=torch.float32),
        "old_per_token_logps": None,
        "maxent_listwise_q": torch.tensor([[0.8, 0.2]], dtype=torch.float32),
    }

    matching_loss = trainer_match.compute_loss(
        model=trainer_match.model,
        inputs=inputs,
        return_outputs=False,
    )
    mismatching_loss = trainer_mismatch.compute_loss(
        model=trainer_mismatch.model,
        inputs=inputs,
        return_outputs=False,
    )

    assert float(matching_loss.item()) < float(mismatching_loss.item())


def test_listwise_maxent_uses_reference_logprobs_in_target_weights() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer.model = _UniformPolicyModel(vocab_size=2)
    trainer.ref_model = _FixedLogitModel([math.log(0.75), math.log(0.25)])
    trainer.args.maxent_tau = 1.0
    trainer.args.beta = 1.0
    trainer.args.maxent_q_temperature = 1.0
    trainer.args.maxent_length_normalize_ref = False
    trainer.args.maxent_reference_logprobs_source = "model"
    trainer.args.maxent_trl_reference_scoring = False
    _refresh_weighting(trainer)
    _install_real_logprob_scorer(trainer)

    prepared = {
        "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "completion_mask": torch.ones((2, 2), dtype=torch.long),
        "advantages": torch.zeros((2,), dtype=torch.float32),
        "old_per_token_logps": None,
        "maxent_listwise_q": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    }

    loss = trainer.compute_loss(model=trainer.model, inputs=prepared, return_outputs=False)
    expected_weights = torch.tensor([0.9, 0.1], dtype=torch.float32)

    assert float(loss.item()) == pytest.approx(math.log(2.0), rel=1e-5)
    assert trainer._metrics["train"]["weight_entropy"][-1] == pytest.approx(
        float(-(expected_weights * expected_weights.log()).sum().item()),
        rel=1e-5,
    )
    assert trainer._metrics["train"]["kl"][-1] > 0.0


def test_listwise_reference_length_normalization_removes_length_bias() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer.model = _UniformPolicyModel(vocab_size=2)
    trainer.ref_model = _FixedLogitModel([math.log(0.75), math.log(0.25)])
    trainer.args.maxent_tau = 1.0
    trainer.args.beta = 1.0
    trainer.args.maxent_length_normalize_ref = True
    trainer.args.maxent_reference_logprobs_source = "model"
    trainer.args.maxent_trl_reference_scoring = False
    _refresh_weighting(trainer)
    _install_real_logprob_scorer(trainer)

    prepared = {
        "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 0], [0, 0]], dtype=torch.long),
        "completion_mask": torch.tensor([[1, 0], [1, 1]], dtype=torch.long),
        "advantages": torch.zeros((2,), dtype=torch.float32),
        "old_per_token_logps": None,
        "maxent_listwise_q": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    }

    trainer.compute_loss(model=trainer.model, inputs=prepared, return_outputs=False)

    assert trainer._metrics["train"]["weight_entropy"][-1] == pytest.approx(
        math.log(2.0),
        rel=1e-5,
    )


def test_listwise_tau_update_moves_toward_target_weight_entropy() -> None:
    trainer = _make_trainer(
        train_grpo_objective=False,
        maxent_alpha=0.0,
        maxent_objective_variant="listwise",
    )
    trainer.model = _FixedLogitModel([math.log(0.75), math.log(0.25)])
    trainer.args.maxent_tau = 0.5
    trainer.args.beta = 0.0
    trainer.args.maxent_q_temperature = 1.0
    trainer.args.maxent_reference_logprobs_source = "none"
    trainer.args.maxent_trl_reference_scoring = False
    trainer.args.maxent_target_weight_entropy = 0.6
    trainer.args.maxent_tau_lr = 0.5
    trainer.args.maxent_tau_warmup_steps = 0
    trainer.state.global_step = 1
    _refresh_weighting(trainer)
    _install_real_logprob_scorer(trainer)

    prepared = {
        "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
        "prompt_mask": torch.ones((2, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "completion_mask": torch.ones((2, 2), dtype=torch.long),
        "advantages": torch.zeros((2,), dtype=torch.float32),
        "old_per_token_logps": None,
        "maxent_listwise_q": torch.tensor([[0.8, 0.2]], dtype=torch.float32),
    }

    before = trainer.tau
    trainer.compute_loss(model=trainer.model, inputs=prepared, return_outputs=False)

    assert trainer.tau > before


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
    assert trainer._metrics["eval"]["pass_at_1"][-1] == pytest.approx(0.5)
    assert trainer._metrics["eval"]["mean_at_8"][-1] == pytest.approx(1.0 / 16.0)


def test_eval_pass_at_8_uses_prompt_major_grouping() -> None:
    trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.0)
    trainer.model.training = False
    trainer.num_generations = 8

    def _reward_fn(prompts, completions, completion_ids, **kwargs):
        del prompts, completions, completion_ids, kwargs
        values = [0.0 for _ in range(16)]
        values[0] = 1.0
        values[1] = 1.0
        return values

    trainer.reward_funcs = [_reward_fn]
    trainer.reward_weights = torch.tensor([1.0], dtype=torch.float32)

    outputs = trainer._generate_and_score_completions(
        [{"prompt": "p0", "answer": "a0"} for _ in range(8)]
        + [{"prompt": "p1", "answer": "a1"} for _ in range(8)]
    )
    assert isinstance(outputs, dict)
    assert trainer._metrics["eval"]["pass_at_8"][-1] == pytest.approx(0.5)
    assert trainer._metrics["eval"]["pass_at_1"][-1] == pytest.approx(0.5)
    assert trainer._metrics["eval"]["mean_at_8"][-1] == pytest.approx(2.0 / 16.0)


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


def test_grpo_path_does_not_apply_reference_ema_side_effects() -> None:
    trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.0)
    trainer.model = torch.nn.Linear(2, 1, bias=False)
    trainer.ref_model = torch.nn.Linear(2, 1, bias=False)
    trainer.model.train()
    trainer.ref_model.eval()

    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
        trainer.ref_model.weight.zero_()

    trainer.args.maxent_reference_ema_enabled = True
    trainer.state.global_step = 200
    trainer.compute_loss(model=None, inputs=[{"prompt": "p", "answer": "a"}])

    assert trainer.ref_model.weight.detach().cpu().flatten().tolist() == pytest.approx(
        [0.0, 0.0]
    )
    assert "maxent/ref_ema_applied" not in trainer._metrics["train"]


def test_grpo_beta_controller_is_disabled_by_default() -> None:
    trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.0)
    trainer.beta = 1.0
    trainer.args.kl_target = 0.1
    trainer.args.kl_horizon = 10
    trainer.args.kl_ctl_step_size = 0.5
    trainer._append_metric_value("train", "kl", 0.3)

    trainer._maybe_update_grpo_beta("train")

    assert trainer.beta == pytest.approx(1.0)


def test_grpo_beta_controller_requires_explicit_opt_in() -> None:
    trainer = _make_trainer(train_grpo_objective=True, maxent_alpha=0.0)
    trainer.beta = 1.0
    trainer.args.grpo_beta_controller_enabled = True
    trainer.args.kl_target = 0.1
    trainer.args.kl_horizon = 10
    trainer.args.kl_ctl_step_size = 0.5
    trainer._append_metric_value("train", "kl", 0.3)

    trainer._maybe_update_grpo_beta("train")

    assert trainer.beta == pytest.approx(1.05)


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
