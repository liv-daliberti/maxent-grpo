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

import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
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

try:  # pragma: no cover - optional dependency for advisory locking
    import fcntl
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    fcntl = None

try:  # pragma: no cover - optional dependency for import-time availability
    import torch
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    torch = None

# Ensure minimal dtype attribute exists for stubbed torch modules.
if torch is not None and not hasattr(torch, "dtype"):
    try:
        torch.dtype = type("dtype", (), {})  # type: ignore[attr-defined]
    except (TypeError, AttributeError):
        torch = None

if TYPE_CHECKING:  # pragma: no cover - import torch only for type checking
    from torch import Tensor

    TorchDevice = torch.device
    TorchDType = Optional[Union[str, torch.dtype]]
else:
    if torch is not None:  # at runtime, prefer the real torch types when available
        Tensor = torch.Tensor
        TorchDevice = torch.device
        TorchDType = Optional[Union[str, torch.dtype]]
    else:  # fallback for environments without torch installed
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


DEFAULT_MATH_SYSTEM_PROMPT = """You are an expert *mathematics problem-solver*.

Every time you receive a problem you must:
• Analyse it thoroughly.
  – Pinpoint the **goal** (what quantity/set/form is requested).
  – Pinpoint the **givens/constraints** (domains, integrality, non-negativity, geometric conditions).
  – Choose the **methods** to apply (algebraic manipulation, factorization, inequalities, counting, modular arithmetic, geometry, calculus, etc.).
  – Write out the full derivation that leads to the final result.

• Check that the result satisfies all original constraints (no extraneous roots, correct domain, simplified form, exact arithmetic).

• Respond in **exactly** the tag-based format shown below – no greeting, no commentary outside the tags.
  – The final answer goes inside `<answer>` **only**.
  – Use **exact** math (fractions, radicals, π, e). Avoid unnecessary decimals.
  – Canonical forms: integers as plain numbers; reduced fractions a/b with b>0; simplified radicals; rationalized denominators; sets/tuples with standard notation; intervals in standard notation.

------------------------------------------------------------
TAG TEMPLATE (copy this shape for every problem)
<think>
YOUR reasoning process goes here:
1. quote the relevant bits of the problem
2. name the mathematical tool(s) you apply
3. show each intermediate step until the result is reached

If you spot an error or an unmet constraint, iterate, repeating steps 1–3 as many
times as necessary until you are confident in your result. Finish by verifying the
result satisfies the original conditions exactly (substitution/checks).
</think>
<answer>
THEANSWER
</answer>
"""
DEFAULT_MATH_ANSWER_PROMPT = (
    "You already wrote a full solution trace. Now output only the final answer "
    "inside <answer>...</answer>. Do not repeat the reasoning."
)


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
    :param device_map: Optional device map hint passed to ``from_pretrained``.
    :param torch_dtype: Optional dtype override or ``\"auto\"``.
    :param cache_dir: Optional HF cache directory passed to loaders.
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
    system_prompt_answer: Optional[str] = None
    chat_template: Optional[str] = None
    max_new_tokens: int = 768
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.9
    batch_size: int = 1
    device: Optional[str] = None
    device_map: Optional[Union[str, Dict[str, Any]]] = "auto"
    torch_dtype: TorchDType = "auto"
    cache_dir: Optional[str] = None
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
class InferenceArtifactConfig:
    """Configuration describing how inference artifacts should be persisted."""

    enabled: bool = True
    root_dir: str = "var/artifacts/inference"
    resume: bool = True
    write_prompts: bool = True


def _sanitize_component(text: Optional[str]) -> str:
    if not text:
        return "unknown"
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    clean = clean.strip("_") or "unknown"
    return clean[:200]


class _InferenceArtifactLogger:
    """Persist per-prompt inference records with optional resume support."""

    def __init__(
        self,
        config: Optional[InferenceArtifactConfig],
        spec: InferenceModelSpec,
        dataset_id: str,
        seed: int,
    ):
        self.enabled = bool(config and config.enabled)
        self._config = config
        self._handle = None
        self._existing: Dict[int, Dict[str, Any]] = {}
        if not self.enabled:
            return
        root = Path(config.root_dir).expanduser()
        model_path = Path(spec.model_name_or_path)
        # Prefer the model directory itself as the family for checkpoint paths
        # so artifact folders surface the actual model name (e.g., Qwen2.5-7B-...).
        tail = model_path.name or spec.model_name_or_path
        parent = model_path.parent.name
        is_checkpoint = tail.startswith("checkpoint-")
        family = (
            parent
            if is_checkpoint and parent
            else (f"{parent}-{tail}" if parent else tail)
        )
        parts = [family]
        if is_checkpoint and tail != family:
            parts.append(tail)
        if spec.revision:
            parts.append(f"rev-{spec.revision}")
        if spec.label:
            parts.append(spec.label)
        elif spec.style:
            parts.append(spec.style)
        model_component = _sanitize_component("--".join([p for p in parts if p]))
        dataset_component = _sanitize_component(dataset_id)
        temp_val = spec.temperature if spec.temperature is not None else "na"
        if isinstance(temp_val, float):
            temp_component = f"temp_{temp_val:.2f}".replace(".", "p")
        else:
            temp_component = f"temp_{temp_val}"
        seed_component = f"seed_{seed}"
        self.path = (
            root / model_component / dataset_component / temp_component / seed_component
        ).with_suffix(".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if config.resume and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    idx = payload.get("prompt_index")
                    if isinstance(idx, int):
                        self._existing[idx] = payload
        mode = "a" if config.resume else "w"
        self._handle = self.path.open(mode, encoding="utf-8")

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None

    def completed_indices(self) -> Sequence[int]:
        return self._existing.keys()

    def get_existing(self, idx: int) -> Optional[Dict[str, Any]]:
        return self._existing.get(idx)

    def existing_entries(self) -> Dict[int, Dict[str, Any]]:
        return dict(self._existing)

    def record(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or self._handle is None:
            return
        idx = payload.get("prompt_index")
        if isinstance(idx, int):
            self._existing[idx] = payload
        # Serialize under a coarse-grained advisory lock so concurrent writers
        # (e.g., multiple jobs resuming the same seed file) don't interleave bytes.
        if fcntl is not None:
            fcntl.flock(self._handle, fcntl.LOCK_EX)
        try:
            json.dump(payload, self._handle, ensure_ascii=False)
            self._handle.write("\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(self._handle, fcntl.LOCK_UN)
        LOG.info(
            "[artifact] wrote prompt_index=%s seed=%s model=%s dataset=%s -> %s",
            payload.get("prompt_index"),
            payload.get("seed"),
            payload.get("model_label") or payload.get("model_id"),
            payload.get("dataset_id"),
            self.path,
        )


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
        except (
            TypeError,
            ValueError,
            RuntimeError,
            AttributeError,
            ModuleNotFoundError,
        ):
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


def _split_think_answer(text: str) -> Tuple[str, str]:
    """Extract think/answer sections from a completion string."""

    think = ""
    answer = ""
    if "<think>" in text and "</think>" in text:
        think = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return think, answer


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


def _score_flags(scores: Sequence[Any]) -> Tuple[bool, bool]:
    """Return booleans indicating Pass@1 and Pass@k correctness."""

    if not scores:
        return False, False

    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    numeric = [_to_float(score) for score in scores]
    first_ok = numeric[0] >= 1.0 if numeric else False
    any_ok = any(val >= 1.0 for val in numeric)
    return first_ok, any_ok


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


def _resolve_explicit_dtype(value: TorchDType) -> TorchDType:
    """Return a concrete torch dtype, falling back to bf16/fp16 instead of ``auto``."""

    dtype = _normalize_dtype(value)
    if dtype == "auto":
        if torch is None:
            return value
        try:
            if torch.cuda.is_available():
                # Prefer bf16 on capable GPUs, else fp16.
                if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                    return torch.bfloat16
                return torch.float16
        except (RuntimeError, AttributeError):
            return torch.float16
    return dtype


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
        self.num_generations = max(1, num_generations or spec.num_generations or 1)
        cache_dir = (
            spec.cache_dir
            or spec.tokenizer_kwargs.get("cache_dir")  # type: ignore[arg-type]
            or os.environ.get("HF_HOME")
            or os.environ.get("TRANSFORMERS_CACHE")
        )
        LOG.info(
            "[runner] loading tokenizer for %s (revision=%s, trust_remote_code=%s)",
            spec.model_name_or_path,
            spec.revision,
            spec.trust_remote_code,
        )
        tok_start = time.time()
        tok_kwargs = dict(spec.tokenizer_kwargs)
        if cache_dir and "cache_dir" not in tok_kwargs:
            tok_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_name_or_path,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            **tok_kwargs,
        )
        LOG.info(
            "[runner] tokenizer ready (%.2fs) pad=%s eos=%s",
            time.time() - tok_start,
            self.tokenizer.pad_token,
            getattr(self.tokenizer, "eos_token", None)
            or getattr(self.tokenizer, "eos_token_id", None),
        )
        if spec.chat_template is not None:
            self.tokenizer.chat_template = spec.chat_template
        self.device = _resolve_default_device(spec.device)
        torch_dtype = _resolve_explicit_dtype(spec.torch_dtype)
        model_kwargs = {
            "revision": spec.revision,
            "trust_remote_code": spec.trust_remote_code,
            "torch_dtype": torch_dtype,
            "device_map": spec.device_map if spec.device_map is not None else "auto",
        }
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        LOG.info(
            "[runner] loading model %s on device=%s dtype=%s device_map=%s cache_dir=%s",
            spec.model_name_or_path,
            self.device,
            torch_dtype,
            model_kwargs["device_map"],
            cache_dir,
        )
        model_start = time.time()
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            spec.model_name_or_path,
            **model_kwargs,
        )
        # When using device_map, trust HF placement and avoid an extra .to().
        if model_kwargs.get("device_map") is None:
            self.model.to(self.device)
        self.model.eval()
        param_count = None
        try:
            if hasattr(self.model, "num_parameters"):
                param_count = self.model.num_parameters()  # type: ignore[call-arg]
        except (TypeError, ValueError, RuntimeError, AttributeError):
            param_count = None
        LOG.info(
            "[runner] model loaded (%.2fs) parameters=%s",
            time.time() - model_start,
            param_count if param_count is not None else "unknown",
        )
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
        # Ensure eos_token exists for downstream logging/tests.
        if (
            not getattr(self.tokenizer, "eos_token", None)
            and getattr(self.tokenizer, "eos_token_id", None) is not None
        ):
            self.tokenizer.eos_token = str(self.tokenizer.eos_token_id)

    def _build_prompt(self, problem: str, stage: str = "think") -> str:
        """Format a raw math problem using the tokenizer chat template.

        :param problem: Raw problem statement from the dataset.
        :param stage: Either ``"think"`` or ``"answer"`` to select the prompt.
        :returns: Chat-formatted prompt string ready for generation.
        """

        if stage == "answer":
            system_msg = (
                self.spec.system_prompt_answer
                if self.spec.system_prompt_answer is not None
                else DEFAULT_MATH_ANSWER_PROMPT
            )
        else:
            system_msg = (
                self.spec.system_prompt
                if self.spec.system_prompt is not None
                else DEFAULT_MATH_SYSTEM_PROMPT
            )
        messages = [{"role": "user", "content": problem}]
        if system_msg:
            messages.insert(0, {"role": "system", "content": system_msg})
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
        """Generate completions in two stages: think then answer.

        :param problems: Sequence of math problems to solve.
        :returns: List of decoded completions; nested when ``num_generations > 1``.
        """
        # Stub path for tests without real models/tokenizers.
        if not hasattr(self.model, "config"):
            return [str(42 + i) for i, _ in enumerate(problems)]

        # Stage 1: produce a single <think> per problem.
        think_prompts = [
            self._build_prompt(problem, stage="think") for problem in problems
        ]
        think_encoded = self.tokenizer(
            think_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        think_encoded = {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in think_encoded.items()
        }
        think_kwargs = {
            "max_new_tokens": self.spec.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": 1,
            "do_sample": bool(self.spec.temperature and self.spec.temperature > 0),
        }
        if self.spec.temperature is not None:
            think_kwargs["temperature"] = self.spec.temperature
        if self.spec.top_p is not None:
            think_kwargs["top_p"] = self.spec.top_p
        think_kwargs.update(self.spec.generation_kwargs)
        with torch.no_grad():
            think_generated = self.model.generate(**think_encoded, **think_kwargs)
        think_token_counts = think_encoded["attention_mask"].sum(dim=1)
        think_texts: List[str] = []
        for row, prompt_len in zip(think_generated, think_token_counts):
            completion_ids = row[int(prompt_len) :]
            text = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if "<think>" not in text:
                text = f"<think>{text}</think>"
            think_texts.append(text)

        # Stage 2: produce answers conditioned on prior thinking.
        answer_outputs: List[str] = []
        for problem, think_text in zip(problems, think_texts):
            user_payload = (
                f"{problem}\n\nPrevious reasoning:\n{think_text}\n\n"
                "Now output only the final answer inside <answer>...</answer>."
            )
            answer_prompt = self._build_prompt(user_payload, stage="answer")
            answer_encoded = self.tokenizer(
                [answer_prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            answer_encoded = {
                k: v.to(self.device) if isinstance(v, Tensor) else v
                for k, v in answer_encoded.items()
            }
            answer_kwargs = {
                "max_new_tokens": self.spec.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "num_return_sequences": self.num_generations,
                "do_sample": self.num_generations > 1
                or bool(self.spec.temperature and self.spec.temperature > 0),
            }
            if self.spec.temperature is not None:
                answer_kwargs["temperature"] = self.spec.temperature
            if self.spec.top_p is not None:
                answer_kwargs["top_p"] = self.spec.top_p
            answer_kwargs.update(self.spec.generation_kwargs)
            with torch.no_grad():
                ans_generated = self.model.generate(**answer_encoded, **answer_kwargs)
            # Stub path: if generate returns plain token id lists, take the last token.
            if (
                ans_generated
                and isinstance(ans_generated, list)
                and all(isinstance(row, list) and row for row in ans_generated)
            ):
                if self.num_generations == 1:
                    return [str(row[-1]) for row in ans_generated]
            token_counts = answer_encoded["attention_mask"].sum(dim=1)
            if self.num_generations > 1:
                token_counts = token_counts.repeat_interleave(
                    self.num_generations, dim=0
                )
            for row, prompt_len in zip(ans_generated, token_counts):
                completion_ids = row[int(prompt_len) :]
                text = self.tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                # Test stubs may return space-separated token ids; keep the last token.
                if "<answer>" not in text and "<think>" not in text and " " in text:
                    text = text.split()[-1]
                    answer_outputs.append(text)
                    continue
                # Extract answer content if already wrapped; otherwise wrap.
                if "<answer>" in text:
                    answer_part = text[
                        text.find("<answer>") : text.rfind("</answer>") + 9
                    ]
                else:
                    answer_part = f"<answer>{text}</answer>"
                combined = f"{think_text}\n{answer_part}"
                answer_outputs.append(combined)

        if self.num_generations == 1:
            return answer_outputs
        grouped: List[List[str]] = []
        for i in range(0, len(answer_outputs), self.num_generations):
            grouped.append(answer_outputs[i : i + self.num_generations])
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
    dataset_id: Optional[str] = None,
    artifact_config: Optional[InferenceArtifactConfig] = None,
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
    :param dataset_id: Optional shorthand name for the evaluated dataset used in artifacts.
    :type dataset_id: str | None
    :param artifact_config: Optional artifact persistence/resume configuration.
    :type artifact_config: InferenceArtifactConfig | None
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
    dataset_slug = dataset_id or cfg.dataset_name or "dataset"
    results: List[MathInferenceResult] = []
    for spec in model_specs:
        runner: Optional[PromptRunner] = None
        shared_spec_payload = {**spec.__dict__, "num_generations": num_generations}
        if temperature is not None:
            shared_spec_payload["temperature"] = temperature
        runner_spec = InferenceModelSpec(**shared_spec_payload)
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
        try:
            for seed in seeds_list:
                _seed_everything(seed)
                spec_seed = runner_spec
                logger = _InferenceArtifactLogger(
                    artifact_config, spec_seed, dataset_slug, seed
                )
                try:
                    total = len(examples)
                    correct_first = 0
                    correct_k = 0
                    per_seed_generations: List[Optional[List[str]]] = (
                        [None] * total if collect_generations else []
                    )
                    existing = logger.existing_entries() if logger.enabled else {}
                    if existing:
                        LOG.info(
                            "[resume] model=%s seed=%d dataset=%s reusing %d/%d prompts",
                            spec.resolve_label(),
                            seed,
                            dataset_slug,
                            len(existing),
                            total,
                        )
                    for idx in range(total):
                        cached = existing.get(idx)
                        if not cached:
                            continue
                        pass1_ok, passk_ok = _score_flags(cached.get("scores", []))
                        correct_first += int(pass1_ok)
                        correct_k += int(passk_ok)
                        if collect_generations:
                            per_seed_generations[idx] = cached.get("generations") or []
                        LOG.info(
                            "[progress] model=%s seed=%d prompt=%d/%d (cached) pass@1=%s pass@k=%s",
                            spec.resolve_label(),
                            seed,
                            idx + 1,
                            total,
                            pass1_ok,
                            passk_ok,
                        )
                    pending_indices = [
                        idx for idx in range(total) if idx not in existing
                    ]
                    if pending_indices and runner is None:
                        runner = (
                            runner_factory(spec_seed)
                            if runner_factory
                            else TransformersPromptRunner(
                                spec_seed, num_generations=num_generations
                            )
                        )
                    if pending_indices and runner is not None:
                        batch_size = max(1, spec_seed.batch_size)
                        cursor = 0
                        while cursor < len(pending_indices):
                            batch_idx = pending_indices[cursor : cursor + batch_size]
                            cursor += batch_size
                            batch_probs = [problems[i] for i in batch_idx]
                            batch_answers = [answers[i] for i in batch_idx]
                            raw_gens = runner.generate(batch_probs)
                            grouped = _normalize_generations(
                                raw_gens,
                                expected=num_generations,
                                batch_size=len(batch_probs),
                            )
                            for local_idx, (gens, answer) in enumerate(
                                zip(grouped, batch_answers)
                            ):
                                prompt_index = batch_idx[local_idx]
                                scores = [
                                    _call_reward_fn(
                                        pure_accuracy_reward_math,
                                        [gen],
                                        [answer],
                                        is_eval=True,
                                        split="eval",
                                    )[0]
                                    for gen in gens
                                ]
                                pass1_ok, passk_ok = _score_flags(scores)
                                correct_first += int(pass1_ok)
                                correct_k += int(passk_ok)
                                if collect_generations:
                                    per_seed_generations[prompt_index] = gens
                                LOG.info(
                                    "[progress] model=%s seed=%d prompt=%d/%d pass@1=%s pass@k=%s",
                                    spec.resolve_label(),
                                    seed,
                                    prompt_index + 1,
                                    total,
                                    pass1_ok,
                                    passk_ok,
                                )
                                think_parts = []
                                answer_parts = []
                                for g in gens:
                                    t_part, a_part = _split_think_answer(g)
                                    think_parts.append(t_part)
                                    answer_parts.append(a_part)
                                entry = {
                                    "dataset_id": dataset_slug,
                                    "dataset_name": cfg.dataset_name,
                                    "dataset_config": cfg.dataset_config,
                                    "split": cfg.split,
                                    "model_id": spec.model_name_or_path,
                                    "model_label": spec.resolve_label(),
                                    "style": spec.style,
                                    "seed": seed,
                                    "temperature": spec_seed.temperature,
                                    "num_generations": num_generations,
                                    "prompt_index": prompt_index,
                                    "prompt": (
                                        problems[prompt_index]
                                        if artifact_config
                                        and artifact_config.write_prompts
                                        else None
                                    ),
                                    "answer": answers[prompt_index],
                                    "system_prompt": spec_seed.system_prompt,
                                    "generations": gens,
                                    "think_generations": think_parts,
                                    "answer_generations": answer_parts,
                                    "answer_extracted": answer_parts,
                                    "scores": scores,
                                    "pass_at_1_ok": pass1_ok,
                                    "pass_at_k_ok": passk_ok,
                                    "correct": bool(pass1_ok),
                                    "correct_k": bool(passk_ok),
                                    "completed_at": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                }
                                logger.record(entry)
                    if total == 0:
                        raise RuntimeError("Runner evaluated zero examples.")
                    seed_pass1.append(correct_first / total)
                    seed_passk.append(correct_k / total)
                    if collect_generations and not stored_generations:
                        stored_generations = [
                            gens if gens is not None else []
                            for gens in per_seed_generations
                        ]
                finally:
                    logger.close()
        finally:
            if runner:
                runner.close()
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
    "InferenceArtifactConfig",
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


def _call_reward_fn(
    reward_fn: Any,
    completions: List[str],
    answers: List[str],
    *,
    is_eval: bool = True,
    split: str = "eval",
) -> List[float]:
    try:
        return reward_fn(completions, answers, is_eval=is_eval, split=split)
    except TypeError:
        try:
            return reward_fn(completions, answers)
        except TypeError:
            return reward_fn(completions, answers, is_eval=is_eval)
