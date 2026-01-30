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
    cast,
    runtime_checkable,
    TYPE_CHECKING,
)
from types import ModuleType, SimpleNamespace
from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.training.rewards import load_reward_functions
from maxent_grpo.training.data import resolve_dataloader_kwargs
from maxent_grpo.rewards.basic import get_reward_funcs as _compat_get_reward_funcs
from maxent_grpo.core.data import get_dataset, load_dataset_split
from maxent_grpo.core.model import get_model, get_tokenizer
from maxent_grpo.training.runtime import log_run_header
from maxent_grpo.training.runtime.prompts import (
    PROMPT_CHAR_LIMIT,
    _prompt_char_limit_from_tokens,
    _to_prompt,
)
from maxent_grpo.telemetry.trl_logging import ensure_weighting_logging
from maxent_grpo.utils.deps_guard import ensure_real_dependencies

if TYPE_CHECKING:
    from trl import ModelConfig  # type: ignore[reportMissingTypeStubs]

try:  # Expose a transformers handle for tests that monkeypatch logging.
    import transformers as transformers
except ImportError:  # pragma: no cover - optional dependency
    transformers = ModuleType("transformers")
    transformers.__spec__ = None
    transformers.__path__ = []
    trainer_utils = ModuleType("transformers.trainer_utils")
    setattr(trainer_utils, "get_last_checkpoint", lambda *_args, **_kwargs: None)
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
        setattr(trainer_utils, "get_last_checkpoint", lambda *_args, **_kwargs: None)


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
        import trl.trainer.grpo_trainer as grpo_mod  # type: ignore[reportMissingTypeStubs]
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
    setattr(utils_module, "logging", tf_logging)
    setattr(transformers, "trainer_utils", trainer_utils)
    setattr(transformers, "utils", utils_module)
    sys.modules.setdefault("transformers", transformers)
    sys.modules.setdefault("transformers.trainer_utils", trainer_utils)
    sys.modules.setdefault("transformers.utils", utils_module)


if getattr(transformers, "set_seed", None) is None:
    setattr(transformers, "set_seed", lambda *_args, **_kwargs: None)
if getattr(transformers, "utils", None) is None:
    setattr(
        transformers,
        "utils",
        SimpleNamespace(
            logging=SimpleNamespace(
                set_verbosity=lambda *args, **kwargs: None,
                enable_default_handler=lambda *args, **kwargs: None,
                enable_explicit_format=lambda *args, **kwargs: None,
            )
        ),
    )

LOG = logging.getLogger(__name__)

GRPOTrainerOverride: Optional[type] = None
get_peft_config_override: Optional[Any] = (
    None  # Callable but kept lax to avoid importing typing.Callable
)

__all__ = [
    "GRPOTrainerOverride",
    "get_peft_config_override",
    "get_reward_funcs",
    "run_baseline_training",
    "_to_prompt",
    "PROMPT_CHAR_LIMIT",
]

# Backward compatibility hook for tests/legacy callers that monkeypatch reward resolution.
get_reward_funcs = _compat_get_reward_funcs


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
        raise NotImplementedError


def _collect_dataset_columns(dataset: Any) -> Dict[str, List[str]]:
    """Return per-split column names when discoverable."""

    col_map: Dict[str, List[str]] = {}
    cols = getattr(dataset, "column_names", None)
    if isinstance(cols, dict):
        for split, names in cols.items():
            if isinstance(names, (list, tuple)) and names:
                col_map[str(split)] = list(names)
        return col_map
    if isinstance(cols, (list, tuple)) and cols:
        return {"all": list(cols)}
    if isinstance(dataset, dict):
        for split, split_ds in dataset.items():
            split_cols = getattr(split_ds, "column_names", None)
            if isinstance(split_cols, (list, tuple)) and split_cols:
                col_map[str(split)] = list(split_cols)
                continue
            if isinstance(split_ds, list) and split_ds:
                first = split_ds[0]
                if isinstance(first, dict):
                    col_map[str(split)] = list(first.keys())
    return col_map


def _validate_dataset_columns(
    dataset: Any,
    *,
    prompt_column: str,
    solution_column: str,
    label: str,
) -> None:
    """Fail fast if required dataset columns are missing."""

    col_map = _collect_dataset_columns(dataset)
    if not col_map:
        LOG.debug(
            "Unable to infer columns for %s; skipping early validation.", label
        )
        return
    message_only = {"messages", "message"}
    if all(cols and set(cols).issubset(message_only) for cols in col_map.values()):
        LOG.debug(
            "Detected message-only dataset columns for %s; skipping early validation.",
            label,
        )
        return
    missing_by_split: Dict[str, List[str]] = {}
    for split, cols in col_map.items():
        if "messages" in cols and prompt_column not in cols and solution_column not in cols:
            continue
        missing = [
            name
            for name in (prompt_column, solution_column)
            if name not in cols
        ]
        if missing:
            missing_by_split[split] = missing
    if missing_by_split:
        if all(
            set(missing) == {solution_column} for missing in missing_by_split.values()
        ):
            LOG.info(
                "%s is missing '%s'; continuing with empty answers.",
                label,
                solution_column,
            )
            return
        missing_desc = "; ".join(
            f"{split} missing {', '.join(cols)}"
            for split, cols in missing_by_split.items()
        )
        available_desc = "; ".join(
            f"{split}={sorted(cols)}" for split, cols in col_map.items()
        )
        raise ValueError(
            f"{label} is missing required columns: {missing_desc}. "
            f"Available columns: {available_desc}"
        )


def _resolve_prompt_column(dataset: Any, prompt_column: str) -> str:
    """Return an inferred prompt column when the default is missing."""
    if prompt_column != "problem":
        return prompt_column
    col_map = _collect_dataset_columns(dataset)
    if not col_map:
        return prompt_column
    if all("problem" in cols for cols in col_map.values()):
        return prompt_column
    if all("prompt" in cols for cols in col_map.values()):
        LOG.info("Prompt column '%s' missing; falling back to 'prompt'.", prompt_column)
        return "prompt"
    return prompt_column


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
    os.makedirs(os.environ.get("LOG_DIR", "var/artifacts/logs"), exist_ok=True)

    ensure_real_dependencies(context="baseline GRPO training")

    # Import selected pieces lazily to keep module import light-weight
    import transformers as transformers_mod
    from transformers.trainer_utils import get_last_checkpoint
    from trl import (  # type: ignore[reportMissingTypeStubs]
        GRPOTrainer as _GRPOTrainer,
        get_peft_config as _get_peft_config,
    )

    override = getattr(sys.modules[__name__], "GRPOTrainerOverride", None)
    trainer_cls = override or _GRPOTrainer
    # Avoid leaking overrides across calls/tests.
    setattr(sys.modules[__name__], "GRPOTrainerOverride", None)
    trainer_cls = ensure_weighting_logging(trainer_cls)
    peft_factory = get_peft_config_override or _get_peft_config

    # Ensure TRL's VLLM client honours distributed port overrides before
    # the trainer instantiates it.

    set_seed_fn = getattr(transformers_mod, "set_seed", None)
    if callable(set_seed_fn):
        set_seed_fn(training_args.seed)
    if not getattr(training_args, "return_reward", False):
        setattr(training_args, "return_reward", True)
    # Keep stop sequences aligned across train/eval and vLLM/HF generation.
    vllm_stops = getattr(training_args, "vllm_stop_sequences", None)
    if getattr(training_args, "gen_stop_sequences", None) in (None, []):
        setattr(training_args, "gen_stop_sequences", vllm_stops)
    if getattr(training_args, "eval_stop_sequences", None) in (None, []):
        setattr(training_args, "eval_stop_sequences", vllm_stops)

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logging.getLogger(__name__).setLevel(log_level)
    log_run_header(training_args)
    dl_kwargs = resolve_dataloader_kwargs(training_args)
    if dl_kwargs:
        # Normalize dataloader settings onto training_args for TRL/Trainer usage.
        try:
            training_args.dataloader_num_workers = int(
                dl_kwargs.get("num_workers", getattr(training_args, "dataloader_num_workers", 0))
            )
        except (AttributeError, TypeError, ValueError) as exc:
            LOG.debug("Failed to set dataloader_num_workers: %s", exc)
        if "pin_memory" in dl_kwargs:
            try:
                training_args.dataloader_pin_memory = bool(dl_kwargs["pin_memory"])
            except (AttributeError, TypeError, ValueError) as exc:
                LOG.debug("Failed to set dataloader_pin_memory: %s", exc)
        if getattr(training_args, "dataloader_num_workers", 0) > 0:
            if "prefetch_factor" in dl_kwargs:
                try:
                    training_args.dataloader_prefetch_factor = int(
                        dl_kwargs["prefetch_factor"]
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    LOG.debug("Failed to set dataloader_prefetch_factor: %s", exc)
            if "persistent_workers" in dl_kwargs:
                try:
                    training_args.dataloader_persistent_workers = bool(
                        dl_kwargs["persistent_workers"]
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    LOG.debug("Failed to set dataloader_persistent_workers: %s", exc)
        else:
            # Avoid invalid prefetch/persistent settings when workers are disabled.
            try:
                training_args.dataloader_prefetch_factor = None
            except (AttributeError, TypeError, ValueError) as exc:
                LOG.debug("Failed to clear dataloader_prefetch_factor: %s", exc)
            try:
                training_args.dataloader_persistent_workers = None
            except (AttributeError, TypeError, ValueError) as exc:
                LOG.debug("Failed to clear dataloader_persistent_workers: %s", exc)
        LOG.info(
            "Baseline dataloader settings | num_workers=%s | pin_memory=%s | prefetch_factor=%s | persistent_workers=%s",
            getattr(training_args, "dataloader_num_workers", None),
            getattr(training_args, "dataloader_pin_memory", None),
            getattr(training_args, "dataloader_prefetch_factor", None),
            getattr(training_args, "dataloader_persistent_workers", None),
        )
    # Optional: datasets logging if available
    try:  # pragma: no cover - environment dependent
        import datasets as _hf_datasets  # type: ignore[reportMissingTypeStubs]

        _hf_datasets.utils.logging.set_verbosity(log_level)
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        LOG.debug("Skipping datasets logging setup: %s", exc)
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
    pc = getattr(script_args, "dataset_prompt_column", "problem")
    pc = _resolve_prompt_column(raw_ds, pc)
    sc = getattr(script_args, "dataset_solution_column", "answer")
    dataset_label = getattr(script_args, "dataset_name", None) or getattr(
        script_args, "dataset_mixture", None
    )
    _validate_dataset_columns(
        raw_ds,
        prompt_column=pc,
        solution_column=sc,
        label=f"training dataset {dataset_label or ''}".strip(),
    )
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)
    ensure_real_dependencies(
        context="baseline GRPO training",
        require_torch=False,
        require_transformers=False,
        require_trl=False,
        require_datasets=False,
        model=model,
        tokenizer=tokenizer,
    )

    # Ensure PAD token exists (left padding recommended for causal LMs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            eos_token = tokenizer.eos_token
            if isinstance(eos_token, list):
                eos_token = eos_token[0] if eos_token else None
            if isinstance(eos_token, str):
                tokenizer.pad_token = eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            resize_fn = getattr(model, "resize_token_embeddings", None)
            if callable(resize_fn):
                resize_fn(len(tokenizer))
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "pad_token_id", None) is None:
        setattr(config, "pad_token_id", tokenizer.pad_token_id)
    try:
        tokenizer.padding_side = "left"
    except AttributeError as exc:
        LOG.debug("Unable to set tokenizer.padding_side: %s", exc)

    # Map dataset → prompt text + gold answer
    char_limit = _prompt_char_limit_from_tokens(
        getattr(training_args, "max_prompt_length", 0)
    )

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        """Map a training split example to prompt/answer text.

        :param ex: Dataset row containing prompt/answer fields.
        :type ex: dict[str, Any]
        :returns: Mapping with ``prompt``/``answer`` keys for training.
        :rtype: dict[str, str]
        """
        prompt_col = pc
        if prompt_col not in ex and prompt_col == "problem" and "prompt" in ex:
            prompt_col = "prompt"
        out = _to_prompt(
            ex,
            cast(Any, tokenizer),
            prompt_col,
            training_args.system_prompt,
            char_limit=char_limit,
        )
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
            eval_prompt_col = _resolve_prompt_column(eval_ds_raw, eval_prompt_col)
            _validate_dataset_columns(
                eval_ds_raw,
                prompt_column=eval_prompt_col,
                solution_column=eval_solution_col,
                label=f"eval dataset {eval_dataset_name}:{eval_split}",
            )

            def _map_eval_fn(ex: Dict[str, Any]) -> Dict[str, str]:
                """Convert evaluation dataset rows into prompt/answer pairs.

                :param ex: Evaluation dataset row.
                :type ex: dict[str, Any]
                :returns: Prompt-answer mapping used for validation.
                :rtype: dict[str, str]
                """
                prompt_col = eval_prompt_col
                if prompt_col not in ex and prompt_col == "problem" and "prompt" in ex:
                    prompt_col = "prompt"
                out = _to_prompt(
                    ex,
                    cast(Any, tokenizer),
                    prompt_col,
                    training_args.system_prompt,
                    char_limit=char_limit,
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
    reward_funcs, reward_weights = load_reward_functions(
        script_args, tokenizer, training_args
    )
    # Keep TRL args aligned with the resolved reward spec so GRPOTrainer's
    # validation (length match) succeeds even when recipes store rewards on
    # script_args only.
    try:
        setattr(training_args, "reward_weights", reward_weights)
    except (AttributeError, TypeError) as exc:
        LOG.debug("Failed to attach reward_weights to training_args: %s", exc)

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
        # Expose trainer kwargs for tests that introspect trainer construction.
        setattr(
            trainer,
            "_init_kwargs",
            dict(
                model=model,
                reward_funcs=reward_funcs,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                peft_config=peft_factory(model_args),
                processing_class=tokenizer,
            ),
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
    elif resume_request is None:
        # Backward compatible behavior: if the output directory already contains a
        # checkpoint and the user did not explicitly opt out of resuming, prefer
        # picking up from the latest checkpoint.
        output_dir = getattr(training_args, "output_dir", None)
        if output_dir and os.path.isdir(output_dir):
            last_ckpt = get_last_checkpoint(output_dir)
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
        last_ckpt = None

    if last_ckpt is not None:
        training_args.resume_from_checkpoint = last_ckpt
    else:
        training_args.resume_from_checkpoint = None
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
