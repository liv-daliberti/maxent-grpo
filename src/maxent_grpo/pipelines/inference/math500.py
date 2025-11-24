"""
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

Inference utilities for running the math_500 benchmark.

The functions in this module are intentionally lightweight so that inference
jobs can be launched outside of the training loop (e.g., on developer
workstations).  The entry point is :func:`run_math500_inference`, which takes a
list of :class:`InferenceModelSpec` entries describing checkpoints trained with
either ``maxent_grpo.grpo`` or ``maxent_grpo.maxent_grpo``.  For each model the runner:

1. Loads the ``HuggingFaceH4/MATH-500`` dataset (configurable via
   :class:`Math500EvalConfig`).
2. Builds chat-formatted prompts that mirror the truncation and system prompt
   behaviour used during training.
3. Generates answers with ``transformers.AutoModelForCausalLM``,
   accumulating exact-match accuracy via :func:`maxent_grpo.rewards.pure_accuracy_reward_math`.

Consumers can provide a custom ``runner_factory`` to stub out the generation
stack in tests or to plug in alternative inference backends (vLLM, hosted
endpoints, etc.).  The default implementation keeps everything local to the
current process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

try:  # pragma: no cover - optional dependency for import-time availability
    import torch
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    torch = None

if TYPE_CHECKING:  # pragma: no cover - import torch only for type checking
    from torch import Tensor

    TorchDevice = torch.device
    TorchDType = Optional[Union[str, torch.dtype]]
else:
    Tensor = Any
    TorchDevice = Any
    TorchDType = Optional[Union[str, Any]]
try:  # pragma: no cover - optional dependency for import-time availability
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PreTrainedModel = Any
    PreTrainedTokenizerBase = Any

from maxent_grpo.rewards.basic import pure_accuracy_reward_math
from maxent_grpo.pipelines.base import log_pipeline_banner

try:  # pragma: no cover - optional dependency for import-time availability
    from datasets import Dataset, load_dataset
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    Dataset = Any
    load_dataset = None


LOG = logging.getLogger(__name__)


class PromptRunner(Protocol):
    """Protocol describing the minimal interface for inference backends."""

    def generate(self, problems: Sequence[str]) -> List[str]:
        """Generate completions for the provided raw problem statements."""

    def close(self) -> None:
        """Release any resources attached to the runner (optional hook)."""


RunnerFactory = Callable[["InferenceModelSpec"], PromptRunner]


@dataclass
class Math500EvalConfig:
    """Configuration describing how to fetch the math_500 dataset."""

    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_config: Optional[str] = None
    split: str = "test"
    prompt_column: str = "problem"
    solution_column: str = "answer"
    limit: Optional[int] = None


@dataclass
class InferenceModelSpec:
    """Describe a checkpoint that should be evaluated on math_500."""

    model_name_or_path: str
    label: Optional[str] = None
    style: str = "grpo"
    revision: Optional[str] = None
    system_prompt: Optional[str] = None
    chat_template: Optional[str] = None
    max_new_tokens: int = 768
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.9
    batch_size: int = 1
    device: Optional[str] = None
    torch_dtype: TorchDType = "auto"
    trust_remote_code: bool = False
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    def resolve_label(self) -> str:
        """Return a display label for logging/results."""

        if self.label:
            return self.label
        tail = self.model_name_or_path.split("/")[-1]
        if self.style:
            return f"{tail} ({self.style})"
        return tail


@dataclass
class Math500InferenceResult:
    """Container returned by :func:`run_math500_inference` per evaluated model."""

    model_id: str
    style: str
    total: int
    correct: int
    accuracy: float
    label: str
    generations: Optional[List[str]] = None


def load_math500_dataset(cfg: Math500EvalConfig) -> Dataset:
    """Return the configured math_500 dataset split from ``datasets``.

    :param cfg: Dataset configuration.
    :type cfg: Math500EvalConfig
    :raises ImportError: If the datasets library is unavailable.
    """

    if load_dataset is None:  # pragma: no cover - exercised when datasets missing
        raise ImportError("datasets library is required to load math_500")
    LOG.info(
        "Loading math_500 dataset %s (config=%s, split=%s)",
        cfg.dataset_name,
        cfg.dataset_config,
        cfg.split,
    )
    return load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)


def _prepare_examples(
    dataset: Iterable[Dict[str, Any]],
    cfg: Math500EvalConfig,
    limit: Optional[int],
) -> List[Tuple[str, str]]:
    """Materialize the requested dataset slice as (problem, answer) pairs."""

    examples: List[Tuple[str, str]] = []
    effective_limit = limit if limit is not None else cfg.limit
    for idx, row in enumerate(dataset):
        try:
            prompt_val = row[cfg.prompt_column]
        except KeyError as exc:
            available = sorted(row.keys()) if hasattr(row, "keys") else []
            raise ValueError(
                f"math_500 row {idx} missing required column "
                f"'{cfg.prompt_column}' (available: {available})"
            ) from exc
        try:
            answer_val = row[cfg.solution_column]
        except KeyError as exc:
            available = sorted(row.keys()) if hasattr(row, "keys") else []
            raise ValueError(
                f"math_500 row {idx} missing required column "
                f"'{cfg.solution_column}' (available: {available})"
            ) from exc
        examples.append((str(prompt_val), str(answer_val)))
        if effective_limit is not None and (idx + 1) >= effective_limit:
            break
    if not examples:
        raise ValueError(
            "math_500 dataset contained no usable rows; check dataset split and columns."
        )
    LOG.info("Prepared %d math_500 examples for inference", len(examples))
    return examples


def _resolve_default_device(device_override: Optional[str]) -> TorchDevice:
    """Return a torch.device based on ``device_override`` and CUDA availability."""

    if torch is None:
        raise ImportError("torch is required for the default prompt runner")
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_dtype(value: TorchDType) -> TorchDType:
    """Convert string dtype hints to ``torch.dtype`` when possible."""

    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"auto", ""}:
            return "auto"
        if hasattr(torch, norm):
            return getattr(torch, norm)
    return value


class TransformersPromptRunner(PromptRunner):
    """Default inference backend powered by ``transformers``."""

    def __init__(self, spec: InferenceModelSpec):
        if torch is None:
            raise ImportError("torch is required for TransformersPromptRunner")
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers is required for TransformersPromptRunner")
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_name_or_path,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            **spec.tokenizer_kwargs,
        )
        if spec.chat_template is not None:
            self.tokenizer.chat_template = spec.chat_template
        self.device = _resolve_default_device(spec.device)
        torch_dtype = _normalize_dtype(spec.torch_dtype)
        model_kwargs = {
            "revision": spec.revision,
            "trust_remote_code": spec.trust_remote_code,
            "torch_dtype": torch_dtype if torch_dtype != "auto" else None,
        }
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            spec.model_name_or_path,
            **model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            eos_id = self.tokenizer.eos_token_id
            if isinstance(eos_id, list):
                eos_id = eos_id[0]
            if eos_id is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                eos_id = self.tokenizer.pad_token_id
                # Update embedding table if tokenizer size changed.
                self.model.resize_token_embeddings(len(self.tokenizer))
            self.tokenizer.pad_token_id = eos_id
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, problem: str) -> str:
        """Format a raw math problem using the tokenizer chat template."""

        messages = [{"role": "user", "content": problem}]
        if self.spec.system_prompt:
            messages.insert(0, {"role": "system", "content": self.spec.system_prompt})
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, TypeError, ValueError):
            rendered = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
            rendered += "\nASSISTANT:"
        return rendered

    def generate(self, problems: Sequence[str]) -> List[str]:
        """Generate a completion per problem using ``model.generate``."""

        prompts = [self._build_prompt(problem) for problem in problems]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in encoded.items()
        }
        gen_kwargs = {
            "max_new_tokens": self.spec.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": bool(self.spec.temperature and self.spec.temperature > 0),
        }
        if self.spec.temperature is not None:
            gen_kwargs["temperature"] = self.spec.temperature
        if self.spec.top_p is not None:
            gen_kwargs["top_p"] = self.spec.top_p
        gen_kwargs.update(self.spec.generation_kwargs)
        with torch.no_grad():
            generated = self.model.generate(**encoded, **gen_kwargs)
        token_counts = encoded["attention_mask"].sum(dim=1)
        outputs: List[str] = []
        for row, prompt_len in zip(generated, token_counts):
            completion_ids = row[int(prompt_len) :]
            text = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            outputs.append(text.strip())
        return outputs

    def close(self) -> None:  # pragma: no cover - trivial resource cleanup
        """Nothing to clean up beyond allowing GC to reclaim the model."""
        return None


def run_math500_inference(
    model_specs: Sequence[InferenceModelSpec],
    eval_cfg: Optional[Math500EvalConfig] = None,
    dataset: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    limit: Optional[int] = None,
    collect_generations: bool = False,
    runner_factory: Optional[RunnerFactory] = None,
) -> List[Math500InferenceResult]:
    """Evaluate each model spec on math_500 and return aggregate metrics.

    :param model_specs: Checkpoints and metadata to evaluate.
    :type model_specs: Sequence[InferenceModelSpec]
    :param eval_cfg: Dataset configuration; defaults to :class:`Math500EvalConfig`.
    :type eval_cfg: Math500EvalConfig | None
    :param dataset: Optional iterable override containing rows with the prompt
        and answer columns. When ``None``, :func:`load_math500_dataset` is used.
    :type dataset: Iterable[dict] | datasets.Dataset | None
    :param limit: Optional cap on the number of examples to evaluate. Overrides
        :attr:`Math500EvalConfig.limit` when provided.
    :type limit: int | None
    :param collect_generations: When ``True``, store raw model generations in
        the returned :class:`Math500InferenceResult`.
    :type collect_generations: bool
    :param runner_factory: Optional hook returning a :class:`PromptRunner` for
        each :class:`InferenceModelSpec`. Defaults to
        :class:`TransformersPromptRunner`.
    :type runner_factory: Callable[[InferenceModelSpec], PromptRunner] | None
    :returns: List of per-model inference results.
    :rtype: list[Math500InferenceResult]
    """

    if not model_specs:
        raise ValueError("run_math500_inference requires at least one model spec")
    cfg = eval_cfg or Math500EvalConfig()
    log_pipeline_banner(
        "inference.math500",
        {"models": [spec.resolve_label() for spec in model_specs], "limit": limit},
    )
    raw_dataset = dataset if dataset is not None else load_math500_dataset(cfg)
    examples = _prepare_examples(raw_dataset, cfg, limit)
    problems = [pair[0] for pair in examples]
    answers = [pair[1] for pair in examples]
    results: List[Math500InferenceResult] = []
    for spec in model_specs:
        LOG.info(
            "Evaluating %s on math_500 (%d problems)",
            spec.resolve_label(),
            len(examples),
        )
        runner = (
            runner_factory(spec) if runner_factory else TransformersPromptRunner(spec)
        )
        generations: List[str] = []
        total = len(examples)
        correct = 0
        try:
            batch_size = max(1, spec.batch_size)
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch_probs = problems[start:end]
                batch_answers = answers[start:end]
                batch_generations = runner.generate(batch_probs)
                if len(batch_generations) != len(batch_answers):
                    raise RuntimeError(
                        f"Runner for {spec.model_name_or_path} returned {len(batch_generations)} "
                        f"generations for {len(batch_answers)} prompts"
                    )
                batch_scores = pure_accuracy_reward_math(
                    batch_generations, batch_answers
                )
                correct += int(sum(batch_scores))
                if collect_generations:
                    generations.extend(batch_generations)
        finally:
            runner.close()
        accuracy = correct / total
        results.append(
            Math500InferenceResult(
                model_id=spec.model_name_or_path,
                style=spec.style,
                total=total,
                correct=correct,
                accuracy=accuracy,
                label=spec.resolve_label(),
                generations=generations if collect_generations else None,
            )
        )
    return results


__all__ = [
    "InferenceModelSpec",
    "Math500EvalConfig",
    "Math500InferenceResult",
    "run_math500_inference",
    "TransformersPromptRunner",
    "load_math500_dataset",
]