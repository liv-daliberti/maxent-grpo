import types

import torch

from maxent_grpo.rewards.basic import pure_accuracy_reward_math
from maxent_grpo.training.eval import run_validation_step
from maxent_grpo.training.types import (
    EvaluationSettings,
    RewardSpec,
    ValidationContext,
)
from maxent_grpo.training.types.logging import LoggingHandles


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


def test_seed_eval_pass_metric_ignores_positive_shaping_only_rewards():
    prompts = [{"prompt": "q", "answer": "a"}]
    tokenizer = _DummyTokenizer()
    model = _DummyModel()

    def _generator(prompts_list, num_samples, per_prompt_counts=None):
        del num_samples
        grouped = []
        for _prompt in prompts_list:
            count = per_prompt_counts[0] if per_prompt_counts else 1
            grouped.append(["<answer>wrong</answer>" for _ in range(count)])
        return grouped, None

    reward_funcs = [lambda completions, answers: [0.05 for _ in completions]]
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
            "num_seeds": 1,
            "samples_per_seed": 2,
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
    assert logged["eval_seed/pass_at_1"] == 0.0


def test_seed_eval_pass_metric_uses_math_answer_correctness():
    prompts = [{"prompt": "q", "answer": "7"}]
    tokenizer = _DummyTokenizer()
    model = _DummyModel()

    def _generator(prompts_list, num_samples, per_prompt_counts=None):
        del prompts_list, num_samples
        grouped = []
        count = per_prompt_counts[0] if per_prompt_counts else 1
        grouped.append(
            ["<answer>7</answer>"]
            + ["<answer>0</answer>" for _ in range(max(count - 1, 0))]
        )
        return grouped, None

    reward_spec = RewardSpec(
        reward_funcs=[pure_accuracy_reward_math],
        reward_weights=[1.0],
    )
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
            "num_seeds": 1,
            "samples_per_seed": 2,
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
    assert logged["eval_seed/pass_at_1"] == 1.0
