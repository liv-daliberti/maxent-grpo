"""Minimal, clean GRPO training entrypoint.

This script intentionally avoids experimental features (replay buffers,
mixing schedules, task routers, custom trainers). It wires up a standard TRL
GRPOTrainer with:

- dataset loading via `open_r1.utils.get_dataset`
- simple chat-templated prompts from `dataset_prompt_column`
- rewards selected by `open_r1.rewards.get_reward_funcs`
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer


def _to_prompt(example: Dict, tokenizer, prompt_column: str, system_prompt: str | None) -> Dict:
    # Build minimal chat: optional system + user(prompt)
    user = str(example.get(prompt_column, example.get("prompt", "")))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: flat join
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

    out = {
        "prompt": prompt,
        "answer": str(example.get("answer", example.get("solution", ""))),
    }
    return out


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    if not getattr(training_args, "return_reward", False):
        training_args.return_reward = True

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logging.getLogger(__name__).setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Data / model
    raw_ds = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    # Ensure PAD token exists (left padding recommended for causal LMs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    # Map dataset → prompt text + gold answer
    pc = getattr(script_args, "dataset_prompt_column", "problem")
    sc = getattr(script_args, "dataset_solution_column", "answer")

    def _map_fn(ex):
        out = _to_prompt(ex, tokenizer, pc, training_args.system_prompt)
        # Ensure answer is present from the configured column
        out["answer"] = str(ex.get(sc, out.get("answer", "")))
        return out

    dataset = raw_ds.map(_map_fn)
    for split in list(dataset.keys()):
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # Resolve splits
    train_split = getattr(script_args, "dataset_train_split", "train")
    test_split = getattr(script_args, "dataset_test_split", None)
    if test_split is None:
        # prefer 'validation' then 'test' if present
        test_split = "validation" if "validation" in dataset.keys() else ("test" if "test" in dataset.keys() else None)

    train_ds = dataset[train_split]
    eval_ds = None
    if training_args.do_eval and test_split is not None and test_split in dataset:
        full_eval = dataset[test_split]
        n_total = len(full_eval)
        # Simple sampler: take 10% up to 1000 items (at least 1)
        n_keep = min(1000, max(1, int(0.1 * n_total)))
        eval_ds = full_eval.shuffle(seed=training_args.seed).select(range(n_keep))

    # Rewards
    reward_funcs = get_reward_funcs(script_args, ref_model=None, tokenizer=tokenizer)

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    # Train
    last_ckpt = (
        training_args.resume_from_checkpoint
        or (get_last_checkpoint(training_args.output_dir) if getattr(training_args, "output_dir", None) and os.path.isdir(training_args.output_dir) else None)
    )
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Save
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(dataset_name=script_args.dataset_name, tags=["open-r1"])
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Eval
    if training_args.do_eval and eval_ds is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Hub
    if getattr(training_args, "push_to_hub", False):
        trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
