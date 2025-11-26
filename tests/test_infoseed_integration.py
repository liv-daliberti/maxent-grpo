import types

import torch

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.pipelines.training.infoseed import run_infoseed_training


class _TinyModel(torch.nn.Module):
    def __init__(self, hidden_size=4, vocab=10, num_seeds=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab)
        self.seed_head = torch.nn.Linear(hidden_size, num_seeds)
        self.config = types.SimpleNamespace(pad_token_id=0)
        self.device = torch.device("cpu")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_hidden_states=False,
    ):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        else:
            loss = None
        outputs = types.SimpleNamespace(logits=logits)
        if output_hidden_states:
            outputs.hidden_states = [hidden]
        if loss is not None:
            outputs.loss = loss
        return outputs


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=True,
        truncation=True,
        add_special_tokens=False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        max_len = max(len(t) for t in texts) if texts else 0
        ids = []
        mask = []
        for t in texts:
            arr = [ord(c) % 9 + 1 for c in t]
            pad_len = max_len - len(arr)
            ids.append(arr + [self.pad_token_id] * pad_len)
            mask.append([1] * len(arr) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def test_infoseed_runner_end_to_end(monkeypatch):
    # Minimal script/training/model configs
    script_args = GRPOScriptArguments(
        dataset_name="dummy",
        dataset_mixture=None,
        dataset_config=None,
    )
    training_args = GRPOConfig(
        info_seed_enabled=True,
        info_seed_num_seeds=2,
        info_seed_lambda=0.1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_generations=2,
        num_train_epochs=1,
        max_steps=1,
        clip_range=0.0,
        gradient_accumulation_steps=1,
        max_prompt_length=8,
        max_completion_length=4,
        gen_temperature=1.0,
        gen_top_p=1.0,
        use_vllm=False,
        learning_rate=1e-3,
        train_grpo_objective=True,
        reward_weights=[1.0],
        reward_funcs=["pure_accuracy_math"],
    )
    model_args = types.SimpleNamespace(model_name_or_path="dummy")

    tiny_ds = [{"prompt": "a", "answer": "b"}]
    script_args.dataset = tiny_ds

    # Patch model/tokenizer/builders used by the runner
    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.infoseed.get_model",
        lambda *a, **k: _TinyModel(),
    )
    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.infoseed.get_tokenizer",
        lambda *a, **k: _TinyTokenizer(),
    )
    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.infoseed.require_accelerator",
        lambda *_a, **_k: types.SimpleNamespace(
            device=torch.device("cpu"),
            is_main_process=True,
            num_processes=1,
            process_index=0,
            clip_grad_norm_=lambda params, max_norm: 0.0,
            backward=lambda loss: loss.backward(),
            accumulate=lambda model: types.SimpleNamespace(
                __enter__=lambda s: None, __exit__=lambda *a: False
            ),
        ),
    )
    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.infoseed.require_dataloader",
        lambda *_a, **_k: lambda ds, batch_size: ds,
    )
    # Run should complete without exceptions and attach seed metrics.
    run_infoseed_training(script_args, training_args, model_args)
