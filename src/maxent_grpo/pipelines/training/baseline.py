"""
Minimal GRPO training entrypoint built on TRL.

This script wires up a standard ``trl.GRPOTrainer`` with:

* Dataset loading via ``core.data.get_dataset``.
* Simple chat‑templated prompts built from a dataset column.
* A small registry of reward functions from ``maxent_grpo.rewards.basic``.

It aims to be a clean baseline without experimental features (e.g., replay
buffers, schedulers, or custom trainers). Use together with
``maxent_grpo.config.ScriptArguments``/``maxent_grpo.config.GRPOConfig`` and TRL's ``TrlParser``.

Key functions

* ``_to_prompt``: Convert a dataset row to a chat prompt + gold answer.
* ``main``: Load data/model, construct ``GRPOTrainer``, train/eval, and handle
  Hub push and model card creation.

License
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

# The module is import‑light: heavy libs are imported lazily inside functions.

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import sys
from typing import (
    Dict,
    Optional,
    Any,
    List,
    Union,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
)
from types import ModuleType, SimpleNamespace
from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.rewards.basic import get_reward_funcs
from maxent_grpo.core.data import get_dataset, load_dataset_split
from maxent_grpo.core.model import get_model, get_tokenizer
from maxent_grpo.patches.trl import ensure_vllm_group_port
from maxent_grpo.training.runtime import log_run_header
from maxent_grpo.training.runtime.prompts import PROMPT_CHAR_LIMIT, _to_prompt

if TYPE_CHECKING:
    from trl import ModelConfig
    from transformers import PreTrainedTokenizer
else:
    PreTrainedTokenizer = Any

try:  # Expose a transformers handle for tests that monkeypatch logging.
    import transformers as transformers
except ImportError:  # pragma: no cover - optional dependency
    transformers = ModuleType("transformers")
    transformers.__spec__ = None
    transformers.__path__ = []
    trainer_utils = ModuleType("transformers.trainer_utils")
    trainer_utils.get_last_checkpoint = lambda *_args, **_kwargs: None
tf_logging = SimpleNamespace(
    set_verbosity=lambda *args, **kwargs: None,
    enable_default_handler=lambda *args, **kwargs: None,
    enable_explicit_format=lambda *args, **kwargs: None,
)

# Ensure trainer_utils is available even when transformers is installed.
if "trainer_utils" not in locals():
    try:
        import transformers.trainer_utils as trainer_utils  # type: ignore
    except (ImportError, AttributeError):
        trainer_utils = ModuleType("transformers.trainer_utils")
        trainer_utils.get_last_checkpoint = lambda *_args, **_kwargs: None


@contextmanager
def _force_vllm_dtype(training_args: GRPOConfig):
    """Ensure colocated vLLM uses the requested dtype instead of model defaults."""

    dtype_override = None
    if getattr(training_args, "fp16", False):
        dtype_override = "float16"
    elif getattr(training_args, "bf16", False):
        dtype_override = "bfloat16"

    if not (dtype_override and getattr(training_args, "use_vllm", False)):
        yield
        return

    try:
        import trl.trainer.grpo_trainer as grpo_mod
        from vllm import LLM as _LLM
    except (ImportError, AttributeError, RuntimeError):
        # If vLLM/TRL isn't available, fall through without patching.
        yield
        return

    orig_llm = getattr(grpo_mod, "LLM", None)

    def _patched_llm(*args, **kwargs):
        kwargs.setdefault("dtype", dtype_override)
        return _LLM(*args, **kwargs)

    if orig_llm is not None:
        grpo_mod.LLM = _patched_llm
    try:
        yield
    finally:
        if orig_llm is not None:
            grpo_mod.LLM = orig_llm
    utils_module = ModuleType("transformers.utils")
    utils_module.logging = tf_logging
    transformers.trainer_utils = trainer_utils
    transformers.utils = utils_module
    sys.modules.setdefault("transformers", transformers)
    sys.modules.setdefault("transformers.trainer_utils", trainer_utils)
    sys.modules.setdefault("transformers.utils", utils_module)


if not hasattr(transformers, "set_seed"):
    transformers.set_seed = lambda *_args, **_kwargs: None
if not hasattr(transformers, "utils"):
    transformers.utils = SimpleNamespace(
        logging=SimpleNamespace(
            set_verbosity=lambda *args, **kwargs: None,
            enable_default_handler=lambda *args, **kwargs: None,
            enable_explicit_format=lambda *args, **kwargs: None,
        )
    )

LOG = logging.getLogger(__name__)

GRPOTrainerOverride: Optional[type] = None
get_peft_config_override: Optional[Any] = (
    None  # Callable but kept lax to avoid importing typing.Callable
)

__all__ = [
    "GRPOTrainerOverride",
    "get_peft_config_override",
    "run_baseline_training",
    "_to_prompt",
    "PROMPT_CHAR_LIMIT",
]


@runtime_checkable
class ChatTemplate(Protocol):
    """Protocol for objects with chat templating capabilities."""

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> Union[str, List[int]]:
        """Render a chat conversation according to an internal template.

        :param conversation: Ordered list of chat messages.
        :type conversation: list[dict[str, str]]
        :param tokenize: Whether to return token IDs instead of text.
        :type tokenize: bool
        :param add_generation_prompt: Append assistant prefix at the end.
        :type add_generation_prompt: bool
        :returns: The templated conversation as text or token IDs.
        :rtype: str | list[int]
        """


def run_baseline_training(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: "ModelConfig",
) -> None:
    """Entrypoint that loads data/model, builds trainer, and runs GRPO.

    The function also performs a small eval subsample for speed if
    ``training_args.do_eval`` is enabled and an eval split exists.

    :param script_args: Script configuration including dataset and rewards.
    :type script_args: GRPOScriptArguments
    :param training_args: GRPO trainer arguments from TRL.
    :type training_args: GRPOConfig
    :param model_args: Model configuration for TRL/transformers.
    :type model_args: ``trl.ModelConfig``
    :returns: ``None``. Side effects include training, evaluation, and checkpointing.
    :rtype: None
    """
    # Ensure logs directory exists for any file redirections by launchers
    os.makedirs(os.environ.get("LOG_DIR", "logs"), exist_ok=True)

    # Import selected pieces lazily to keep module import light-weight
    import transformers as transformers_mod
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOTrainer as _GRPOTrainer, get_peft_config as _get_peft_config

    trainer_cls = GRPOTrainerOverride or _GRPOTrainer
    peft_factory = get_peft_config_override or _get_peft_config

    # Ensure TRL's VLLM client honours distributed port overrides before
    # the trainer instantiates it.
    ensure_vllm_group_port()

    set_seed_fn = getattr(transformers_mod, "set_seed", None)
    if callable(set_seed_fn):
        set_seed_fn(training_args.seed)
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
    log_run_header(training_args)
    # Optional: datasets logging if available
    try:  # pragma: no cover - environment dependent
        import datasets as _hf_datasets

        _hf_datasets.utils.logging.set_verbosity(log_level)
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    tf_logging_module = getattr(
        getattr(transformers_mod, "utils", None), "logging", None
    )
    if tf_logging_module is not None:
        set_verbosity = getattr(tf_logging_module, "set_verbosity", None)
        if callable(set_verbosity):
            set_verbosity(log_level)
        enable_default_handler = getattr(
            tf_logging_module, "enable_default_handler", None
        )
        if callable(enable_default_handler):
            enable_default_handler()
        enable_explicit_format = getattr(
            tf_logging_module, "enable_explicit_format", None
        )
        if callable(enable_explicit_format):
            enable_explicit_format()

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
    except AttributeError:
        pass

    # Map dataset → prompt text + gold answer
    pc = getattr(script_args, "dataset_prompt_column", "problem")
    sc = getattr(script_args, "dataset_solution_column", "answer")

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        """Map a training split example to prompt/answer text.

        :param ex: Dataset row containing prompt/answer fields.
        :type ex: dict[str, Any]
        :returns: Mapping with ``prompt``/``answer`` keys for training.
        :rtype: dict[str, str]
        """
        out = _to_prompt(ex, tokenizer, pc, training_args.system_prompt)
        # Ensure answer is present from the configured column
        out["answer"] = str(ex.get(sc, out.get("answer", "")))
        return out

    if hasattr(raw_ds, "map"):
        dataset = raw_ds.map(_map_fn)
    else:

        class _Split:
            def __init__(self, rows):
                self._rows = rows

            @property
            def column_names(self):
                return []

            def remove_columns(self, *_cols):
                return self

            def shuffle(self, seed=None):
                _ = seed
                return self

            def select(self, _indices):
                return self

            def __len__(self):
                return len(self._rows)

        class _DictDataset(dict):
            def map(self, fn):
                return _DictDataset(
                    {k: _Split([fn(ex) for ex in v]) for k, v in self.items()}
                )

        dataset = _DictDataset(raw_ds).map(_map_fn)
    for split in list(dataset.keys()):
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # Resolve splits
    train_split = getattr(script_args, "dataset_train_split", "train")
    test_split = getattr(script_args, "dataset_test_split", None)
    if test_split is None:
        # prefer 'validation' then 'test' if present
        test_split = (
            "validation"
            if "validation" in dataset.keys()
            else ("test" if "test" in dataset.keys() else None)
        )

    train_ds = dataset[train_split]
    eval_ds = None
    eval_dataset_name = getattr(script_args, "eval_dataset_name", None)
    eval_prompt_col = getattr(script_args, "eval_dataset_prompt_column", None) or pc
    eval_solution_col = getattr(script_args, "eval_dataset_solution_column", None) or sc

    if training_args.do_eval:
        if eval_dataset_name:
            eval_split = getattr(script_args, "eval_dataset_split", "validation")
            eval_ds_raw = load_dataset_split(
                eval_dataset_name,
                getattr(script_args, "eval_dataset_config", None),
                eval_split,
            )

            def _map_eval_fn(ex: Dict[str, Any]) -> Dict[str, str]:
                """Convert evaluation dataset rows into prompt/answer pairs.

                :param ex: Evaluation dataset row.
                :type ex: dict[str, Any]
                :returns: Prompt-answer mapping used for validation.
                :rtype: dict[str, str]
                """
                out = _to_prompt(
                    ex, tokenizer, eval_prompt_col, training_args.system_prompt
                )
                out["answer"] = str(ex.get(eval_solution_col, out.get("answer", "")))
                return out

            eval_ds = eval_ds_raw.map(_map_eval_fn)
            if "messages" in eval_ds.column_names:
                eval_ds = eval_ds.remove_columns("messages")
        elif test_split is not None and test_split in dataset:
            full_eval = dataset[test_split]
            n_total = len(full_eval)
            # Simple sampler: take 10% up to 1000 items (at least 1)
            n_keep = min(1000, max(1, int(0.1 * n_total)))
            eval_ds = full_eval.shuffle(seed=training_args.seed).select(range(n_keep))

    # Rewards
    reward_funcs = get_reward_funcs(script_args, None, tokenizer)

    # Trainer
    with _force_vllm_dtype(training_args):
        trainer = trainer_cls(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_factory(model_args),
            processing_class=tokenizer,
        )

    # Train
    logger = logging.getLogger(__name__)
    resume_request = getattr(training_args, "resume_from_checkpoint", None)
    last_ckpt: Optional[str] = None
    if isinstance(resume_request, str) and resume_request:
        if os.path.isdir(resume_request):
            last_ckpt = resume_request
        else:
            logger.warning(
                "resume_from_checkpoint=%s was provided but the path does not exist; "
                "starting from scratch.",
                resume_request,
            )
    elif resume_request:
        output_dir = getattr(training_args, "output_dir", None)
        if output_dir and os.path.isdir(output_dir):
            last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt is None:
            logger.warning(
                "resume_from_checkpoint was requested but no checkpoint was found under %s; "
                "starting from scratch.",
                output_dir or "<unspecified>",
            )
    else:
        output_dir = getattr(training_args, "output_dir", None)
        if output_dir and os.path.isdir(output_dir):
            last_ckpt = get_last_checkpoint(output_dir)

    if last_ckpt is not None:
        training_args.resume_from_checkpoint = last_ckpt
    else:
        training_args.resume_from_checkpoint = False
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    if hasattr(trainer, "log_metrics"):
        trainer.log_metrics("train", train_result.metrics)
    if hasattr(trainer, "save_metrics"):
        trainer.save_metrics("train", train_result.metrics)
    if hasattr(trainer, "save_state"):
        trainer.save_state()

    # Save
    try:
        trainer.save_model(training_args.output_dir)
    except TypeError:
        trainer.save_model()
    if getattr(trainer, "accelerator", None) is not None and getattr(
        trainer.accelerator, "is_main_process", False
    ):
        if hasattr(trainer, "create_model_card"):
            trainer.create_model_card(
                dataset_name=script_args.dataset_name, tags=["open-r1"]
            )
        if hasattr(trainer, "model") and hasattr(trainer.model, "config"):
            trainer.model.config.use_cache = True
            if hasattr(trainer.model.config, "save_pretrained"):
                trainer.model.config.save_pretrained(training_args.output_dir)

    # Eval
    if training_args.do_eval and eval_ds is not None:
        if hasattr(trainer, "evaluate"):
            metrics = trainer.evaluate()
            if hasattr(trainer, "log_metrics"):
                trainer.log_metrics("eval", metrics)
            if hasattr(trainer, "save_metrics"):
                trainer.save_metrics("eval", metrics)

    # Hub
    if getattr(training_args, "push_to_hub", False):
        trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])
