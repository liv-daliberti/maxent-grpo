import types

import torch

from maxent_grpo.training.eval import run_validation_step
from maxent_grpo.training.types import (
    EvaluationSettings,
    RewardSpec,
    ValidationContext,
)
from maxent_grpo.training.types.logging import LoggingHandles
from maxent_grpo.training.weighting.loss import (
    GroupLossData,
    RatioContext,
    SequenceScores,
    evaluate_losses,
)
from maxent_grpo.training.weighting.types import (
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightStats,
    WeightingSettings,
)


class _MetricWriter:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step):
        self.logged.append((step, metrics))

    def flush(self):
        return


class _DummyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True):
        if isinstance(texts, str):
            texts = [texts]
        max_len = max(len(t) for t in texts) if texts else 0
        ids = []
        mask = []
        for t in texts:
            arr = [ord(c) % 255 for c in t]
            pad_len = max_len - len(arr)
            ids.append(arr + [self.pad_token_id] * pad_len)
            mask.append([1] * len(arr) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class _DummyModel(torch.nn.Module):
    def __init__(self, hidden_size=8, num_seeds=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.seed_head = torch.nn.Linear(hidden_size, num_seeds)
        self.device = torch.device("cpu")
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_hidden_states=False,
    ):
        batch, seq = input_ids.shape
        hidden = torch.randn(batch, seq, self.hidden_size)
        logits = torch.randn(batch, seq, 10)
        return types.SimpleNamespace(logits=logits, hidden_states=[hidden])


def test_seed_eval_metrics_logged():
    prompts = [{"prompt": "q", "answer": "a"}]
    tokenizer = _DummyTokenizer()
    model = _DummyModel()

    def _generator(prompts_list, num_samples, per_prompt_counts=None):
        grouped = []
        for p in prompts_list:
            count = per_prompt_counts[0] if per_prompt_counts else num_samples
            grouped.append([f"{p}-c{i}" for i in range(count)])
        return grouped, None

    reward_funcs = [lambda completions, answers: [1.0 for _ in completions]]
    reward_spec = RewardSpec(reward_funcs=reward_funcs, reward_weights=[1.0])
    writer = _MetricWriter()
    logging_handles = LoggingHandles(
        metric_writer=writer,
        save_checkpoint=lambda _: None,
        save_strategy="no",
        save_steps=0,
        wandb_run=None,
    )
    eval_settings = EvaluationSettings(
        enabled=True,
        rows=prompts,
        batch_size=1,
        every_n_steps=1,
        seed_eval={
            "enabled": True,
            "num_seeds": 2,
            "samples_per_seed": 1,
            "template": "\n[seed={seed}]",
            "pooling": "mean",
        },
    )
    ctx = ValidationContext(
        evaluation=eval_settings,
        accelerator=types.SimpleNamespace(
            num_processes=1, process_index=0, is_main_process=True
        ),
        model=model,
        tokenizer=tokenizer,
        reward=reward_spec,
        generator=_generator,
        logging=logging_handles,
    )
    run_validation_step(1, ctx)
    logged = dict(writer.logged[-1][1])
    assert "eval_seed/pass_at_1" in logged
    assert "eval_seed/pred_acc" in logged


def test_seed_loss_gradients_flow():
    torch.manual_seed(0)
    group_sizes = [2]
    weight_stats = WeightStats(
        weights_grouped=[[1.0, 1.0]],
        flat_weights=[1.0, 1.0],
        weight_entropy=0.0,
        weight_entropy_min=0.0,
        weight_entropy_max=0.0,
        advantage_entropy=[0.0],
    )
    scores = SequenceScores(
        cur_logp_sum=torch.tensor([0.0, 0.0], requires_grad=True),
        behavior_logp_sum=torch.tensor([0.0, 0.0]),
        log_ratio_train=torch.tensor([0.0, 0.0]),
        denom_tok_tensor=torch.tensor([1.0, 1.0]),
        pooled_hidden=torch.randn(2, 4, requires_grad=True),
    )
    group_data, ratio_ctx = (
        GroupLossData(
            group_sizes=group_sizes,
            weight_tensor=torch.tensor(weight_stats.flat_weights),
            logp_sums=scores.cur_logp_sum,
            token_counts=scores.denom_tok_tensor,
        ),
        RatioContext(
            log_ratio_train=scores.log_ratio_train,
            denom_tok_tensor=scores.denom_tok_tensor,
            clip_cfg=types.SimpleNamespace(
                clip_range=0.2, use_clip_objective=False, clip_objective_coef=1.0
            ),
            weighting_cfg=WeightingSettings(
                tau=1.0,
                beta=0.0,
                normalization=WeightNormalizationSettings(
                    denom=1.0, len_norm_ref=False
                ),
                q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
                tau_schedule=TauSchedule(
                    target_entropy=None,
                    learning_rate=0.0,
                    minimum_value=0.0,
                    maximum_value=0.0,
                    warmup_steps=0,
                ),
                kl_controller=KlControllerSettings(
                    target=0.0, horizon=1, step_size=0.0
                ),
                train_grpo_objective=True,
            ),
            ref_stats=types.SimpleNamespace(
                ref_logp_sum=scores.behavior_logp_sum,
                ref_tok_counts=scores.denom_tok_tensor,
                ref_logp_sum_raw=scores.behavior_logp_sum,
            ),
            cur_logp_sum=scores.cur_logp_sum,
            behavior_logp_sum=scores.behavior_logp_sum,
        ),
    )
    seed_inputs = types.SimpleNamespace(
        seed_ids=torch.tensor([0, 1]),
        pooled_hidden=scores.pooled_hidden,
        is_seed_aug=None,
        logits=torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True),
    )
    loss_outputs, _ = evaluate_losses(
        group_data,
        ratio_ctx,
        seed_inputs=seed_inputs,
        info_seed_lambda=1.0,
        info_seed_temperature=0.1,
        info_seed_loss_type="ce",
        info_seed_alpha_entropy=0.5,
    )
    loss_outputs.loss.backward()
    assert seed_inputs.logits.grad is not None
    assert scores.cur_logp_sum.grad is not None
