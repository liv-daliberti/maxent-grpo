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

Inference utilities for running math benchmarks (math_500, AIME, AMC, Minerva).

The functions in this module are intentionally lightweight so that inference
jobs can be launched outside of the training loop (e.g., on developer
workstations).  The entry point is :func:`run_math_inference`, which takes a
list of :class:`InferenceModelSpec` entries describing checkpoints trained with
either ``maxent_grpo.grpo`` or ``maxent_grpo.maxent_grpo``.  For each model the runner:

1. Loads the configured dataset preset (``math_500`` by default).
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
import random
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

    def generate(self, problems: Sequence[str]) -> List[Any]:
        """Generate completions for the provided raw problem statements.

        May return a flat list of strings (one per prompt) or a nested list when
        multiple generations per prompt are produced.
        """

    def close(self) -> None:
        """Release any resources attached to the runner (optional hook)."""


RunnerFactory = Callable[["InferenceModelSpec"], PromptRunner]


@dataclass
class MathEvalConfig:
    """Configuration describing how to fetch math benchmarks.

    :param dataset_name: Hugging Face dataset identifier.
    :param dataset_config: Optional dataset config name.
    :param split: Dataset split to evaluate.
    :param prompt_column: Column containing the problem text.
    :param solution_column: Column containing the reference answer.
    :param prompt_column_aliases: Optional fallbacks for the prompt column.
    :param solution_column_aliases: Optional fallbacks for the solution column.
    :param limit: Optional cap on number of rows to evaluate.
    """

    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_config: Optional[str] = None
    split: str = "test"
    prompt_column: str = "problem"
    solution_column: str = "answer"
    prompt_column_aliases: Tuple[str, ...] = field(default_factory=tuple)
    solution_column_aliases: Tuple[str, ...] = field(default_factory=tuple)
    limit: Optional[int] = None


@dataclass
class InferenceModelSpec:
    """Describe a checkpoint that should be evaluated on math_500.

    :param model_name_or_path: HF model id or local path to load.
    :param label: Optional short label used in results/logs.
    :param style: Training style identifier (e.g., ``grpo`` or ``maxent``).
    :param revision: Optional model revision to load.
    :param system_prompt: Optional system message to prepend.
    :param chat_template: Optional chat template to override tokenizer defaults.
    :param max_new_tokens: Maximum tokens to generate per completion.
    :param temperature: Sampling temperature (``0`` or ``None`` for greedy).
    :param top_p: Nucleus sampling parameter.
    :param batch_size: Batch size for prompt generation.
    :param device: Optional device override (e.g., ``cuda:0``).
    :param torch_dtype: Optional dtype override or ``\"auto\"``.
    :param trust_remote_code: Whether to allow custom model code.
    :param num_generations: Number of completions sampled per prompt.
    :param generation_kwargs: Extra kwargs forwarded to ``model.generate``.
    :param tokenizer_kwargs: Extra kwargs forwarded to the tokenizer.
    """

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
    num_generations: int = 1
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    def resolve_label(self) -> str:
        """Return a display label for logging/results.

        :returns: Prefer the explicit ``label`` else a synthesized tail.
        """

        if self.label:
            return self.label
        tail = self.model_name_or_path.split("/")[-1]
        if self.style:
            return f"{tail} ({self.style})"
        return tail


@dataclass
class MathInferenceResult:
    """Container returned by :func:`run_math_inference` per evaluated model.

    :param model_id: HF id or path evaluated.
    :param style: Style string from :class:`InferenceModelSpec`.
    :param total: Number of evaluated examples.
    :param correct: Count of correct predictions.
    :param accuracy: Accuracy ratio ``correct / total``.
    :param label: Human-friendly display label.
    :param generations: Optional raw generations when requested.
    :param pass_at_1: Pass@1 averaged across seeds.
    :param pass_at_k: Pass@k averaged across seeds (k=num_generations).
    :param avg_pass_at_1: Alias for averaged Pass@1 (useful for reporting Avg@k).
    :param seeds: Seeds used during evaluation.
    :param num_generations: Number of generations per prompt.
    """

    model_id: str
    style: str
    total: int
    correct: int
    accuracy: float
    label: str
    generations: Optional[List[str]] = None
    pass_at_1: float = 0.0
    pass_at_k: float = 0.0
    avg_pass_at_1: float = 0.0
    seeds: List[int] = field(default_factory=list)
    num_generations: int = 1


# Canonical inference dataset presets shared across math benchmarks.
INFERENCE_DATASETS: Dict[str, MathEvalConfig] = {
    "math_500": MathEvalConfig(
        dataset_name="HuggingFaceH4/MATH-500",
        dataset_config=None,
        split="test",
        prompt_column="problem",
        solution_column="answer",
        solution_column_aliases=("solution",),
    ),
    "aime24": MathEvalConfig(
        dataset_name="HuggingFaceH4/aime_2024",
        dataset_config="default",
        split="train",
        prompt_column="problem",
        solution_column="answer",
        solution_column_aliases=("solution",),
    ),
    "aime25": MathEvalConfig(
        dataset_name="yentinglin/aime_2025",
        dataset_config="default",
        split="train",
        prompt_column="problem",
        solution_column="answer",
        solution_column_aliases=("solution",),
    ),
    "amc": MathEvalConfig(
        dataset_name="AI-MO/aimo-validation-amc",
        dataset_config="default",
        split="train",
        prompt_column="problem",
        solution_column="answer",
    ),
    "minerva": MathEvalConfig(
        dataset_name="svc-huggingface/minerva-math",
        dataset_config=None,
        split="train",
        prompt_column="problem",
        prompt_column_aliases=("question", "prompt"),
        solution_column="answer",
        solution_column_aliases=("solution", "final_answer"),
    ),
}

_INFERENCE_DATASET_ALIASES: Dict[str, str] = {
    "math": "math_500",
    "aime_24": "aime24",
    "aime_2024": "aime24",
    "aime_25": "aime25",
    "aime_2025": "aime25",
}


def list_inference_datasets() -> List[str]:
    """Return the canonical names of built-in inference presets."""

    return sorted(INFERENCE_DATASETS.keys())


def resolve_inference_dataset(
    name: str, overrides: Optional[Dict[str, Any]] | None = None
) -> MathEvalConfig:
    """Return a :class:`MathEvalConfig` for a registered dataset preset.

    :param name: Preset name (case-insensitive, hyphens/underscores normalized).
    :param overrides: Optional field overrides applied to the preset.
    :raises ValueError: If ``name`` is empty or unknown.
    """

    if not name:
        raise ValueError("inference dataset name must be provided")
    normalized = name.lower().replace("-", "_")
    normalized = _INFERENCE_DATASET_ALIASES.get(normalized, normalized)
    try:
        base_cfg = INFERENCE_DATASETS[normalized]
    except KeyError as exc:
        supported = ", ".join(list_inference_datasets())
        raise ValueError(
            f"Unknown inference dataset '{name}'. Supported presets: {supported}"
        ) from exc
    payload = {**vars(base_cfg)}
    if overrides:
        payload.update(overrides)
    return MathEvalConfig(**payload)


def load_math_dataset(cfg: MathEvalConfig) -> Dataset:
    """Return the configured math dataset split from ``datasets``.

    :param cfg: Dataset configuration.
    :type cfg: MathEvalConfig
    :raises ImportError: If the datasets library is unavailable.
    :returns: Loaded :class:`datasets.Dataset` with the requested split/config.
    """

    if load_dataset is None:  # pragma: no cover - exercised when datasets missing
        raise ImportError("datasets library is required to load math_500")
    LOG.info(
        "Loading inference dataset %s (config=%s, split=%s)",
        cfg.dataset_name,
        cfg.dataset_config,
        cfg.split,
    )
    return load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)


def _format_dataset_label(cfg: MathEvalConfig) -> str:
    """Return a friendly dataset label for logs/errors."""

    if cfg.dataset_name:
        label = cfg.dataset_name
        if cfg.dataset_config:
            label = f"{label}:{cfg.dataset_config}"
        return label
    return "dataset"


def _seed_everything(seed: int) -> None:
    """Seed common RNGs for reproducible sampling."""

    random.seed(seed)
    try:  # pragma: no cover - optional dependency
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except (ImportError, AttributeError, ValueError):
        pass
    if torch is not None:  # pragma: no cover - optional dependency
        try:
            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and hasattr(torch.cuda, "manual_seed_all"):
                torch.cuda.manual_seed_all(seed)
        except (TypeError, ValueError, RuntimeError, AttributeError):
            pass


def _resolve_field(
    row: Dict[str, Any],
    primary: str,
    aliases: Tuple[str, ...],
    dataset_label: str,
    row_idx: int,
    field_desc: str,
) -> Any:
    """Fetch a column value from ``row`` using primary/alias keys."""

    candidates = (primary, *aliases)
    for key in candidates:
        if key in row:
            return row[key]
    available = sorted(row.keys()) if hasattr(row, "keys") else []
    alias_note = f" (aliases tried: {list(aliases)})" if aliases else ""
    raise ValueError(
        f"{dataset_label} row {row_idx} missing required {field_desc} column "
        f"'{primary}'{alias_note} (available: {available})"
    )


def _normalize_generations(
    outputs: Sequence[Any], expected: int, batch_size: int
) -> List[List[str]]:
    """Convert runner outputs into a per-example list of generations.

    :param outputs: Raw outputs from a :class:`PromptRunner`.
    :param expected: Number of generations requested per prompt.
    :param batch_size: Number of prompts in the batch.
    :returns: List of length ``batch_size`` where each element is a list of strings.
    :raises RuntimeError: If counts do not align with requested samples.
    """

    if not outputs:
        raise RuntimeError("Runner returned no generations")
    per_example: List[List[str]] = []
    if all(isinstance(o, (list, tuple)) for o in outputs):
        # Already grouped per example.
        per_example = [list(map(str, group)) for group in outputs]  # type: ignore[arg-type]
    else:
        flat = [str(o) for o in outputs]
        if len(flat) % batch_size != 0:
            raise RuntimeError(
                f"Runner returned {len(flat)} generations for batch size {batch_size}"
            )
        group_size = len(flat) // batch_size
        for i in range(batch_size):
            per_example.append(flat[i * group_size : (i + 1) * group_size])
    for idx, group in enumerate(per_example):
        if len(group) != expected:
            raise RuntimeError(
                f"Runner returned {len(group)} generations for example {idx}; "
                f"expected {expected}"
            )
    return per_example


def _prepare_examples(
    dataset: Iterable[Dict[str, Any]],
    cfg: MathEvalConfig,
    limit: Optional[int],
) -> List[Tuple[str, str]]:
    """Materialize the requested dataset slice as (problem, answer) pairs.

    :param dataset: Iterable of rows containing prompt/solution columns.
    :param cfg: Dataset config specifying column names.
    :param limit: Optional limit overriding :attr:`MathEvalConfig.limit`.
    :returns: List of ``(problem, answer)`` tuples.
    :raises ValueError: If required columns are missing or no rows are usable.
    """

    examples: List[Tuple[str, str]] = []
    dataset_label = _format_dataset_label(cfg)
    effective_limit = limit if limit is not None else cfg.limit
    for idx, row in enumerate(dataset):
        prompt_val = _resolve_field(
            row,
            cfg.prompt_column,
            cfg.prompt_column_aliases,
            dataset_label,
            idx,
            "prompt",
        )
        answer_val = _resolve_field(
            row,
            cfg.solution_column,
            cfg.solution_column_aliases,
            dataset_label,
            idx,
            "solution",
        )
        examples.append((str(prompt_val), str(answer_val)))
        if effective_limit is not None and (idx + 1) >= effective_limit:
            break
    if not examples:
        raise ValueError(
            f"{dataset_label} dataset contained no usable rows; "
            "check dataset split and columns."
        )
    LOG.info("Prepared %d %s examples for inference", len(examples), dataset_label)
    return examples


def _resolve_default_device(device_override: Optional[str]) -> TorchDevice:
    """Return a torch.device based on ``device_override`` and CUDA availability.

    :param device_override: Optional explicit device string (e.g., ``cuda:0``).
    :returns: Resolved :class:`torch.device`.
    :raises ImportError: If torch is not installed.
    """

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
    """Convert string dtype hints to ``torch.dtype`` when possible.

    :param value: User-provided dtype string, torch dtype, or ``None``.
    :returns: Normalized dtype or original value when unrecognized.
    """

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

    def __init__(self, spec: InferenceModelSpec, num_generations: Optional[int] = None):
        """Initialize tokenizer/model for prompt generation.

        :param spec: Model specification describing checkpoint and generation params.
        :raises ImportError: When torch or transformers are unavailable.
        """
        if torch is None:
            raise ImportError("torch is required for TransformersPromptRunner")
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers is required for TransformersPromptRunner")
        self.spec = spec
        self.spec = spec
        self.num_generations = max(1, num_generations or spec.num_generations or 1)
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
        """Format a raw math problem using the tokenizer chat template.

        :param problem: Raw problem statement from the dataset.
        :returns: Chat-formatted prompt string ready for generation.
        """

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

    def generate(self, problems: Sequence[str]) -> List[Any]:
        """Generate one or more completions per problem using ``model.generate``.

        :param problems: Sequence of math problems to solve.
        :returns: List of decoded completions; nested when ``num_generations > 1``.
        """

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
            "num_return_sequences": self.num_generations,
            "do_sample": self.num_generations > 1
            or bool(self.spec.temperature and self.spec.temperature > 0),
        }
        if self.spec.temperature is not None:
            gen_kwargs["temperature"] = self.spec.temperature
        if self.spec.top_p is not None:
            gen_kwargs["top_p"] = self.spec.top_p
        gen_kwargs.update(self.spec.generation_kwargs)
        with torch.no_grad():
            generated = self.model.generate(**encoded, **gen_kwargs)
        token_counts = encoded["attention_mask"].sum(dim=1)
        if self.num_generations > 1:
            token_counts = token_counts.repeat_interleave(self.num_generations, dim=0)
        outputs: List[str] = []
        for row, prompt_len in zip(generated, token_counts):
            completion_ids = row[int(prompt_len) :]
            text = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            outputs.append(text.strip())
        if self.num_generations == 1:
            return outputs
        grouped: List[List[str]] = []
        for i in range(0, len(outputs), self.num_generations):
            grouped.append(outputs[i : i + self.num_generations])
        return grouped

    def close(self) -> None:  # pragma: no cover - trivial resource cleanup
        """Nothing to clean up beyond allowing GC to reclaim the model.

        :returns: ``None`` after cleanup.
        """
        return None


def run_math_inference(
    model_specs: Sequence[InferenceModelSpec],
    eval_cfg: Optional[MathEvalConfig] = None,
    dataset: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    limit: Optional[int] = None,
    collect_generations: bool = False,
    runner_factory: Optional[RunnerFactory] = None,
    num_generations: int = 1,
    seeds: Optional[Sequence[int]] = None,
    temperature: Optional[float] = None,
) -> List[MathInferenceResult]:
    """Evaluate model checkpoints on math benchmarks (Pass@1/Pass@k averaged over seeds).

    :param model_specs: Checkpoints and metadata to evaluate.
    :type model_specs: Sequence[InferenceModelSpec]
    :param eval_cfg: Dataset configuration; defaults to :class:`MathEvalConfig`.
    :type eval_cfg: MathEvalConfig | None
    :param dataset: Optional iterable override containing rows with the prompt
        and answer columns. When ``None``, :func:`load_math_dataset` is used.
    :type dataset: Iterable[dict] | datasets.Dataset | None
    :param limit: Optional cap on the number of examples to evaluate. Overrides
        :attr:`MathEvalConfig.limit` when provided.
    :type limit: int | None
    :param collect_generations: When ``True``, store raw model generations in
        the returned :class:`MathInferenceResult`.
    :type collect_generations: bool
    :param runner_factory: Optional hook returning a :class:`PromptRunner` for
        each :class:`InferenceModelSpec`. Defaults to
        :class:`TransformersPromptRunner`.
    :type runner_factory: Callable[[InferenceModelSpec], PromptRunner] | None
    :param num_generations: Number of sampled completions per prompt (Pass@k where k=num_generations).
    :type num_generations: int
    :param seeds: RNG seeds to evaluate over (averaged). Defaults to ``[0]``.
    :type seeds: Sequence[int] | None
    :param temperature: Optional temperature override applied to every model spec.
    :type temperature: float | None
    :returns: List of per-model inference results.
    :rtype: list[MathInferenceResult]
    :raises ValueError: If no model specs are provided or runner outputs mismatch.
    """

    if not model_specs:
        raise ValueError("run_math_inference requires at least one model spec")
    if num_generations < 1:
        raise ValueError("num_generations must be >= 1")
    seeds_list: List[int] = list(seeds) if seeds else [0]
    cfg = eval_cfg or MathEvalConfig()
    log_pipeline_banner(
        "inference.math",
        {
            "models": [spec.resolve_label() for spec in model_specs],
            "limit": limit,
            "num_generations": num_generations,
            "seeds": seeds_list,
        },
    )
    raw_dataset = dataset if dataset is not None else load_math_dataset(cfg)
    examples = _prepare_examples(raw_dataset, cfg, limit)
    problems = [pair[0] for pair in examples]
    answers = [pair[1] for pair in examples]
    results: List[MathInferenceResult] = []
    for spec in model_specs:
        LOG.info(
            "Evaluating %s on %s (%d problems) with k=%d over seeds=%s",
            spec.resolve_label(),
            _format_dataset_label(cfg),
            len(examples),
            num_generations,
            seeds_list,
        )
        seed_pass1: List[float] = []
        seed_passk: List[float] = []
        stored_generations: List[List[str]] = []
        for seed in seeds_list:
            _seed_everything(seed)
            spec_payload = {**spec.__dict__, "num_generations": num_generations}
            if temperature is not None:
                spec_payload["temperature"] = temperature
            spec_seed = InferenceModelSpec(**spec_payload)
            runner = (
                runner_factory(spec_seed)
                if runner_factory
                else TransformersPromptRunner(
                    spec_seed, num_generations=num_generations
                )
            )
            total = len(examples)
            correct_first = 0
            correct_k = 0
            per_seed_generations: List[List[str]] = []
            try:
                batch_size = max(1, spec_seed.batch_size)
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    batch_probs = problems[start:end]
                    batch_answers = answers[start:end]
                    raw_gens = runner.generate(batch_probs)
                    grouped = _normalize_generations(
                        raw_gens, expected=num_generations, batch_size=len(batch_probs)
                    )
                    if collect_generations:
                        per_seed_generations.extend(grouped)
                    for gens, answer in zip(grouped, batch_answers):
                        scores = [
                            pure_accuracy_reward_math([gen], [answer])[0]
                            for gen in gens
                        ]
                        if scores and scores[0] >= 1.0:
                            correct_first += 1
                        if any(score >= 1.0 for score in scores):
                            correct_k += 1
            finally:
                runner.close()
            if total == 0:
                raise RuntimeError("Runner evaluated zero examples.")
            seed_pass1.append(correct_first / total)
            seed_passk.append(correct_k / total)
            if collect_generations and not stored_generations:
                stored_generations = per_seed_generations
        avg_pass1 = sum(seed_pass1) / len(seed_pass1)
        avg_passk = sum(seed_passk) / len(seed_passk)
        accuracy = avg_pass1
        results.append(
            MathInferenceResult(
                model_id=spec.model_name_or_path,
                style=spec.style,
                total=len(examples),
                correct=int(round(avg_pass1 * len(examples))),
                accuracy=accuracy,
                label=spec.resolve_label(),
                generations=stored_generations if collect_generations else None,
                pass_at_1=avg_pass1,
                pass_at_k=avg_passk,
                avg_pass_at_1=avg_pass1,
                seeds=seeds_list,
                num_generations=num_generations,
            )
        )
    return results


run_math_eval_inference = run_math_inference


__all__ = [
    "InferenceModelSpec",
    "MathEvalConfig",
    "MathInferenceResult",
    "INFERENCE_DATASETS",
    "list_inference_datasets",
    "resolve_inference_dataset",
    "run_math_inference",
    "run_math_eval_inference",
    "TransformersPromptRunner",
    "load_math_dataset",
]
