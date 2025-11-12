"""
MaxEnt‑GRPO: sequence‑level maximum‑entropy variant of GRPO.

This training entrypoint implements a lightweight loop that realizes the
per‑context maximum‑entropy update at the sequence level. For each prompt we:
1) Generate ``K`` completions (via vLLM ``/generate`` when enabled, or local
   ``model.generate``).
2) Convert per‑sequence utilities (rewards) into a listwise distribution ``q``
   using a temperature and epsilon floor for full support.
3) Compute sequence log‑probs under a frozen reference model.
4) Form per‑sequence weights ``w_i ∝ q_i^{1/(τ+β)} · π_ref(i)^{β/(τ+β)}`` and
   normalize within each prompt group.
5) Apply a weighted MLE update (no explicit KL term required).

Key pieces
- ``_to_prompt``: Local copy of the prompt builder to avoid circular imports.
- ``MaxEntOptions``: Environment‑driven knobs (τ, q temperature/epsilon, length
  normalization) for convenience.
- ``_group_softmax``, ``_prepare_labels_for_ce``, ``_sequence_logprobs``,
  ``_batch_tokenize_pairs``: Helpers for weighting and scoring sequences.
- ``main``: End‑to‑end training loop using ``utils.*`` helpers and
  ``utils.vllm_patch.safe_generate`` when vLLM is enabled.

This is intentionally minimal for readability and prototyping. For production
features (controllers, schedulers, DDP), prefer the TRL trainer path in
``src/grpo.py`` and extend it.

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

# See the module docstring above for a detailed overview.

from __future__ import annotations

import logging
import math
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

try:  # Optional dependency for reading accelerate config files
    import yaml  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional
    yaml = None

from configs import GRPOConfig, GRPOScriptArguments
from rewards import get_reward_funcs
from utils.data import get_dataset, load_dataset_split
from utils.model_utils import get_model, get_tokenizer
from utils.trl_patches import ensure_vllm_group_port
from utils.wandb_logging import init_wandb_training
from utils.vllm_patch import safe_generate

try:  # Optional dependency when running under DeepSpeed ZeRO
    from deepspeed import zero as ds_zero  # type: ignore
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - environment dependent
    ds_zero = None
    ZeroParamStatus = None  # type: ignore

# Apply TRL compatibility patch eagerly so any downstream usage of VLLMClient
# inherits the environment-driven group_port override.
ensure_vllm_group_port()

LOG = logging.getLogger(__name__)
PROMPT_CHAR_LIMIT = int(os.environ.get("MAX_PROMPT_CHARS", "2048"))
_TRUNC_STATE = {"warned": False}

# -------------------------------
# Helpers
# -------------------------------

@contextmanager
def _maybe_zero_gather_embedding(model: Optional[PreTrainedModel]):
    """Gather ZeRO-sharded embedding weights before a forward pass."""
    if ds_zero is None or ZeroParamStatus is None or model is None:
        yield
        return
    base_model = getattr(model, "module", model)
    embedding = getattr(base_model, "get_input_embeddings", lambda: None)()
    weight = getattr(embedding, "weight", None) if embedding is not None else None
    if weight is None:
        yield
        return
    if getattr(weight, "ndim", 2) == 2:
        yield
        return
    status = getattr(weight, "ds_status", None)
    if status is not None and status != ZeroParamStatus.NOT_AVAILABLE:
        yield
        return
    with ds_zero.GatheredParameters([weight], modifier_rank=None):  # type: ignore[union-attr]
        yield


def _zero_param_list(model: Optional[nn.Module]) -> List[nn.Parameter]:
    """Return a parameter list for ZeRO-gather contexts, unwrapping DeepSpeed engines."""
    if model is None:
        return []
    base_model = getattr(model, "module", model)
    if not hasattr(base_model, "parameters"):
        return []
    try:
        return list(base_model.parameters())
    except TypeError:
        return []


def _report_to_contains(report_to: Union[str, Sequence[str], None], target: str) -> bool:
    """Case-insensitive membership check for TrainingArguments.report_to."""
    if report_to is None:
        return False
    if isinstance(report_to, str):
        entries = [report_to]
    else:
        entries = list(report_to)
    target = target.lower()
    return any(str(item).lower() == target for item in entries)


def _maybe_init_wandb_run(
    accelerator: Accelerator,
    training_args: GRPOConfig,
    wandb_config: Dict[str, Any],
) -> Optional[Any]:
    """Initialize a W&B run when report_to includes wandb."""
    if not _report_to_contains(getattr(training_args, "report_to", None), "wandb"):
        return None
    init_wandb_training(training_args)
    if not accelerator.is_main_process:
        os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))
        return None
    try:
        import wandb  # type: ignore
    except (ImportError, ModuleNotFoundError):
        LOG.warning("report_to includes wandb but the wandb package is not installed; skipping logging.")
        return None

    run_name = getattr(training_args, "run_name", None)
    wandb_kwargs: Dict[str, Any] = {
        "config": wandb_config,
        "reinit": True,
        "dir": os.environ.get("WANDB_DIR") or os.getcwd(),
    }
    if run_name:
        wandb_kwargs["name"] = run_name
    project = os.environ.get("WANDB_PROJECT")
    if project:
        wandb_kwargs["project"] = project
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        wandb_kwargs["entity"] = entity
    group = os.environ.get("WANDB_RUN_GROUP")
    if group:
        wandb_kwargs["group"] = group
    return wandb.init(**wandb_kwargs)


def _log_wandb(run: Optional[Any], metrics: Dict[str, Any], step: int) -> None:
    """Safely log metrics to a W&B run."""
    if run is None or not metrics:
        return
    try:
        run.log(metrics, step=step)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOG.warning("Failed to log metrics to W&B: %s", exc)


def _maybe_create_deepspeed_plugin() -> Optional[DeepSpeedPlugin]:
    """Construct a DeepSpeedPlugin from Accelerate env/config when available."""
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() != "true":
        return None

    ds_cfg: Dict[str, Any] = {}
    cfg_path = os.environ.get("ACCELERATE_CONFIG_FILE")
    if cfg_path and yaml is not None and os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            ds_cfg = raw.get("deepspeed_config") or {}
        except Exception:
            ds_cfg = {}

    # Defaults match recipes/accelerate_configs/zero3.yaml
    zero_stage = int(ds_cfg.get("zero_stage", 3))
    offload_param = ds_cfg.get("offload_param_device")
    offload_optim = ds_cfg.get("offload_optimizer_device")
    zero3_init_flag = ds_cfg.get("zero3_init_flag")
    zero3_save = ds_cfg.get("zero3_save_16bit_model")

    kwargs = {
        "zero_stage": zero_stage,
        "offload_param_device": offload_param,
        "offload_optimizer_device": offload_optim,
        "zero3_init_flag": zero3_init_flag,
        "zero3_save_16bit_model": zero3_save,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if not kwargs:
        return None
    return DeepSpeedPlugin(**kwargs)


@runtime_checkable
class ChatTokenizer(Protocol):
    """Protocol for tokenizers with chat template capabilities."""
    def apply_chat_template(
        self, 
        conversation: List[Dict[str, str]], 
        tokenize: bool = True,
        add_generation_prompt: bool = True
    ) -> Union[str, List[int]]: ...

def _truncate_prompt(prompt: str) -> str:
    """Clamp prompt strings to a safe length for vLLM/http payloads."""
    if PROMPT_CHAR_LIMIT <= 0 or len(prompt) <= PROMPT_CHAR_LIMIT:
        return prompt
    if not _TRUNC_STATE["warned"]:
        LOG.warning(
            "Prompt length exceeded %d characters; truncating. "
            "Override via MAX_PROMPT_CHARS if needed.",
            PROMPT_CHAR_LIMIT,
        )
        _TRUNC_STATE["warned"] = True
    return prompt[:PROMPT_CHAR_LIMIT]


def _to_prompt(
    example: Dict[str, Any],
    tokenizer: Union[PreTrainedTokenizer, ChatTokenizer],
    prompt_column: str,
    system_prompt: Optional[str]
) -> Dict[str, str]:
    """Light copy of src/grpo.py:_to_prompt (kept local to avoid circular import).

    Builds a minimal chat conversation with an optional system message and a
    single user turn extracted from ``prompt_column`` (fallback to ``prompt``).
    The tokenizer's chat template is applied if available to produce the final
    prompt string.
    """
    user = str(example.get(prompt_column, example.get("prompt", "")))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        prompt = (
            "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            + "\nASSISTANT:"
        )
    prompt = _truncate_prompt(prompt)
    return {
        "prompt": prompt,
        "answer": str(example.get("answer", example.get("solution", ""))),
    }


@dataclass
class MaxEntOptions:
    """Lightweight knobs specific to MaxEnt sequence‑level updates.

    These are parsed from environment variables or defaulted to reasonable
    values so the script can be used with existing GRPO recipes as‑is.
    """

    tau: float = field(default_factory=lambda: float(os.environ.get("MAXENT_TAU", 0.2)))
    q_temperature: float = field(
        default_factory=lambda: float(os.environ.get("MAXENT_Q_TEMPERATURE", 1.0))
    )
    q_epsilon: float = field(default_factory=lambda: float(os.environ.get("MAXENT_Q_EPS", 1e-6)))
    length_normalize_ref: bool = field(
        default_factory=lambda: os.environ.get("MAXENT_LENGTH_NORM_REF", "1") not in {"0", "false", "False"}
    )


def _group_softmax(
    xs: List[float],
    temperature: float = 1.0,
    eps: float = 1e-6
) -> List[float]:
    """Numerically stable softmax with optional temperature and epsilon floor."""
    if len(xs) == 0:
        return []
    x = torch.tensor(xs, dtype=torch.float32)
    x = x / max(temperature, 1e-8)
    x = x - x.max()
    probs = torch.softmax(x, dim=0)
    probs = probs * (1.0 - eps * len(xs)) + eps
    probs = probs / probs.sum()
    return probs.tolist()


def _prepare_labels_for_ce(
    input_ids: Tensor,
    prompt_lengths: List[int]
) -> Tensor:
    """Create labels tensor with prompt tokens masked as -100 for CE.

    :param input_ids: LongTensor [B, T]
    :param prompt_lengths: list of ints (len B) giving prompt token counts
    :returns: LongTensor labels [B, T] with prompt labels masked to -100
    """
    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lengths):
        labels[i, :plen] = -100
    return labels


def _sequence_logprobs(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute per‑sequence (completion) log‑prob sums and token counts.

    Uses teacher forcing on the full prompt+completion with labels masked to
    only score completion tokens. Returns the sum of log‑probs per sequence and
    the number of scored tokens per sequence.
    """
    params_ctx = nullcontext()
    if ds_zero is not None and model is not None:
        params = _zero_param_list(model)
        if params:
            params_ctx = ds_zero.GatheredParameters(params, modifier_rank=None)
    with params_ctx:
        with _maybe_zero_gather_embedding(model):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    # Mask for valid (non -100) labels
    mask = (shift_labels != -100)
    # Gather per‑token log‑probs at the gold token ids
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    # Per‑sequence sums and counts
    seq_logp = token_logp.sum(dim=1)
    seq_counts = mask.sum(dim=1)
    return seq_logp, seq_counts


def _batch_tokenize_pairs(
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    completions: List[str]
) -> Tuple[Tensor, Tensor, List[int]]:
    """Tokenize prompt+completion pairs and return tensors + prompt lengths."""
    pairs = [p + c for p, c in zip(prompts, completions)]
    # Compute prompt lengths with tokenize=True for chat template fallback
    enc_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    prompt_lengths = enc_prompts["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1).tolist()
    enc = tokenizer(
        pairs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    return input_ids, attn, prompt_lengths


def main(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: Any  # from transformers.ModelArguments
) -> None:
    ds_plugin = _maybe_create_deepspeed_plugin()
    accelerator = Accelerator(deepspeed_plugin=ds_plugin)
    # Ensure logs directory exists for any file redirections by launchers
    os.makedirs(os.environ.get("LOG_DIR", "logs"), exist_ok=True)

    # Seed and logging
    from transformers import set_seed
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level() if hasattr(training_args, "get_process_log_level") else logging.INFO
    logging.getLogger(__name__).setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Options
    # Read MaxEnt knobs from YAML (training_args), with env fallback for convenience
    beta = float(getattr(training_args, "init_kl_coeff", 0.0))
    tau = float(getattr(training_args, "maxent_tau", os.environ.get("MAXENT_TAU", 0.2)))
    q_temp = float(getattr(training_args, "maxent_q_temperature", os.environ.get("MAXENT_Q_TEMPERATURE", 1.0)))
    q_eps = float(getattr(training_args, "maxent_q_epsilon", os.environ.get("MAXENT_Q_EPS", 1e-6)))
    len_norm_ref = bool(getattr(training_args, "maxent_length_normalize_ref", os.environ.get("MAXENT_LENGTH_NORM_REF", "1") not in {"0", "false", "False"}))
    denom = tau + beta if (tau + beta) > 0 else 1.0

    # Data / model / tokenizer
    raw_ds = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)
    model.train()

    # Ensure PAD token exists
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
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
        # Some tokenizers expose padding_side as a property-less attribute
        # or do not support setting it; ignore in that case.
        pass

    # Reference model (frozen)
    ref_model = get_model(model_args, training_args)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Map dataset → prompt text + gold answer
    pc = getattr(script_args, "dataset_prompt_column", "problem")
    sc = getattr(script_args, "dataset_solution_column", "answer")

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, str]:
        out = _to_prompt(ex, tokenizer, pc, training_args.system_prompt)
        out["answer"] = str(ex.get(sc, out.get("answer", "")))
        return out

    dataset = raw_ds.map(_map_fn)
    for split in list(dataset.keys()):
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    train_split = getattr(script_args, "dataset_train_split", "train")
    train_ds = dataset[train_split]

    # Training hyperparameters (keep minimal)
    bsz = int(getattr(training_args, "per_device_train_batch_size", 4))
    num_epochs = int(getattr(training_args, "num_train_epochs", 1))
    _ = int(getattr(training_args, "gradient_accumulation_steps", 1))  # unused in minimal loop
    lr = float(getattr(training_args, "learning_rate", 5e-6))
    max_prompt_len = int(getattr(training_args, "max_prompt_length", 512))
    max_completion_len = int(getattr(training_args, "max_completion_length", 256))
    num_generations = int(getattr(training_args, "num_generations", 4))
    use_vllm = bool(getattr(training_args, "use_vllm", False))
    gen_temperature = float(getattr(training_args, "gen_temperature", 0.8))
    gen_top_p = float(getattr(training_args, "gen_top_p", 0.9))
    vllm_url = str(getattr(training_args, "vllm_url", os.environ.get("VLLM_URL", "http://localhost:8000/generate")))

    device = accelerator.device

    # Optimizer (no schedulers/weight decay to keep simple)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-6)
    max_grad_norm = float(getattr(training_args, "max_grad_norm", 1.0))

    # Reward function(s)
    reward_funcs = get_reward_funcs(script_args, None, tokenizer)
    reward_weights = getattr(training_args, "reward_weights", None)
    if reward_weights is None or len(reward_weights) != len(reward_funcs):
        reward_weights = [1.0 for _ in range(len(reward_funcs))]
    if not reward_funcs:
        raise RuntimeError("No reward functions resolved for MaxEnt‑GRPO training.")

    # Dataloader over raw prompts/answers (we use Python lists due to generation step)
    def _collate(batch):
        prompts = [row["prompt"] for row in batch]
        answers = [row.get("answer", "") for row in batch]
        return {"prompt": prompts, "answer": answers}

    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True, collate_fn=_collate)

    # Wrap with Accelerate/DeepSpeed (ZeRO-aware)
    model, optim, train_loader = accelerator.prepare(model, optim, train_loader)
    ref_model = accelerator.prepare_model(ref_model, evaluation_mode=True)

    wandb_base_config: Dict[str, Any] = {
        "script": "maxent-grpo",
        "model_name_or_path": getattr(model_args, "model_name_or_path", None),
        "dataset_name": getattr(script_args, "dataset_name", None),
        "per_device_train_batch_size": bsz,
        "learning_rate": lr,
        "num_generations": num_generations,
        "max_prompt_length": max_prompt_len,
        "max_completion_length": max_completion_len,
        "tau": tau,
        "beta": beta,
        "q_temperature": q_temp,
        "q_epsilon": q_eps,
        "length_norm_ref": len_norm_ref,
        "use_vllm": use_vllm,
        "seed": getattr(training_args, "seed", None),
    }
    wandb_run = _maybe_init_wandb_run(accelerator, training_args, wandb_base_config)

    def _log_metrics(metrics: Dict[str, Any], step: int) -> None:
        _log_wandb(wandb_run, metrics, step)

    # Optional dedicated eval dataset (e.g., MATH-500)
    eval_dataset_name = getattr(script_args, "eval_dataset_name", None)
    eval_prompt_col = getattr(script_args, "eval_dataset_prompt_column", None) or pc
    eval_solution_col = getattr(script_args, "eval_dataset_solution_column", None) or sc
    eval_split = getattr(script_args, "eval_dataset_split", "validation")
    eval_rows: List[Dict[str, str]] = []
    eval_enabled = bool(getattr(training_args, "do_eval", False))
    eval_every = int(getattr(training_args, "eval_steps", 0))
    eval_every = eval_every if eval_every > 0 else None
    eval_bsz = int(getattr(training_args, "per_device_eval_batch_size", bsz))

    if eval_enabled and eval_dataset_name:
        eval_raw = load_dataset_split(
            eval_dataset_name,
            getattr(script_args, "eval_dataset_config", None),
            eval_split,
        )

        def _map_eval_fn(ex: Dict[str, Any]) -> Dict[str, str]:
            out = _to_prompt(ex, tokenizer, eval_prompt_col, training_args.system_prompt)
            out["answer"] = str(ex.get(eval_solution_col, out.get("answer", "")))
            return out

        eval_processed = eval_raw.map(_map_eval_fn)
        if "messages" in eval_processed.column_names:
            eval_processed = eval_processed.remove_columns("messages")
        eval_rows = eval_processed.to_list()
        if not eval_rows:
            eval_enabled = False
    else:
        eval_enabled = False

    def _generate_completions(prompts: List[str], n: int) -> List[List[str]]:
        """Shared helper to produce n completions per prompt."""
        if not prompts:
            return []
        if use_vllm:
            return safe_generate(
                prompts=prompts,
                url=vllm_url,
                max_tokens=max_completion_len,
                temperature=gen_temperature,
                top_p=gen_top_p,
                n=n,
                stream=False,
                tokenizer=tokenizer,
            )
        grouped: List[List[str]] = []
        gen_model = accelerator.unwrap_model(model)
        for p in prompts:
            enc = tokenizer(
                p,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_len,
            ).to(device)
            outs: List[str] = []
            with torch.no_grad():
                gen_out = gen_model.generate(
                    **enc,
                    do_sample=True,
                    temperature=gen_temperature,
                    top_p=gen_top_p,
                    max_new_tokens=max_completion_len,
                    num_return_sequences=n,
                )
            for i in range(gen_out.shape[0]):
                text = tokenizer.decode(gen_out[i], skip_special_tokens=True)
                outs.append(text[len(p):])
            grouped.append(outs)
        return grouped

    def _run_validation(step: int) -> None:
        """Generate single completions on the eval set and log mean reward."""
        if not eval_enabled or not eval_rows:
            return
        if not accelerator.is_main_process:
            return
        prev_mode = model.training
        model.eval()
        eval_scores: List[float] = []
        for idx in range(0, len(eval_rows), eval_bsz):
            batch = eval_rows[idx : idx + eval_bsz]
            prompts = [row["prompt"] for row in batch]
            answers = [row.get("answer", "") for row in batch]
            if not prompts:
                continue
            grouped = _generate_completions(prompts, 1)
            completions = [grp[0] if grp else "" for grp in grouped]
            total_utils = [0.0] * len(completions)
            for w, rf in zip(reward_weights, reward_funcs):
                rs = rf(completions, answers)
                if w != 1.0:
                    rs = [float(w) * float(r) for r in rs]
                total_utils = [u + float(r) for u, r in zip(total_utils, rs)]
            eval_scores.extend(total_utils)
        mean_reward = float(sum(eval_scores) / max(len(eval_scores), 1))
        logging.getLogger(__name__).info(
            "eval step %d | mean_reward=%.4f | samples=%d",
            step,
            mean_reward,
            len(eval_scores),
        )
        if accelerator.is_main_process:
            _log_metrics({"eval/mean_reward": mean_reward}, step)
        if prev_mode:
            model.train()

    global_step = 0
    try:
        for epoch in range(num_epochs):
            for batch in train_loader:
                prompts: List[str] = batch["prompt"]
                answers: List[str] = batch["answer"]

                # 1) Generate K candidates per prompt
                grouped_comps = _generate_completions(prompts, num_generations)

                # 2) Rewards → listwise q per prompt
                # Flatten for reward functions, then regroup
                flat_comps: List[str] = [c for group in grouped_comps for c in group]
                flat_answers: List[str] = []
                for a in answers:
                    flat_answers.extend([a] * num_generations)

                # Combine multiple reward functions additively (weights default to 1)
                total_utils = [0.0] * len(flat_comps)
                for w, rf in zip(reward_weights, reward_funcs):
                    rs = rf(flat_comps, flat_answers)
                    if w != 1.0:
                        rs = [float(w) * float(r) for r in rs]
                    total_utils = [u + float(r) for u, r in zip(total_utils, rs)]
                utils_tensor = torch.tensor(
                    total_utils,
                    dtype=torch.float32,
                    device=device if device.type != "cpu" else torch.device("cpu"),
                )
                train_reward_mean = float(utils_tensor.mean().item())
                train_reward_std = float(utils_tensor.std(unbiased=False).item()) if utils_tensor.numel() > 1 else 0.0

                # Group utilities per prompt and turn into q via softmax
                utils_grouped: List[List[float]] = []
                for i in range(0, len(total_utils), num_generations):
                    utils_grouped.append(total_utils[i : i + num_generations])
                q_grouped: List[List[float]] = [
                    _group_softmax(us, temperature=q_temp, eps=q_eps)
                    for us in utils_grouped
                ]

                # 3) Sequence log‑probs under reference (per completion)
                # Build tokenized prompt+completion tensors batch‑wise for efficiency
                # To keep memory bounded, process in mini‑batches of up to bsz*num_generations
                prompt_batch: List[str] = []
                comp_batch: List[str] = []
                for p, comps in zip(prompts, grouped_comps):
                    for c in comps:
                        prompt_batch.append(p)
                        comp_batch.append(c)

                # Tokenize
                enc_pairs = tokenizer(
                    [p + c for p, c in zip(prompt_batch, comp_batch)],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_len + max_completion_len + 8,
                )
                # Compute per‑row prompt lengths by re‑encoding each pair's prompt part
                prompt_lengths: List[int] = []
                for p in prompt_batch:
                    prompt_lengths.append(
                        tokenizer(p, return_tensors="pt", truncation=True, max_length=max_prompt_len)["input_ids"].shape[-1]
                    )

                input_ids = enc_pairs["input_ids"].to(device)
                attention_mask = enc_pairs["attention_mask"].to(device)
                labels = _prepare_labels_for_ce(input_ids.clone(), prompt_lengths).to(device)

                with torch.no_grad():
                    ref_logp_sum, ref_tok_counts = _sequence_logprobs(
                        ref_model, input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    )
                # Optional length normalization to reduce bias towards shorter completions
                if len_norm_ref:
                    ref_logp_sum = ref_logp_sum / ref_tok_counts.clamp(min=1).to(ref_logp_sum.dtype)
                ref_logp_mean = float(ref_logp_sum.mean().detach().cpu())
                avg_completion_tokens = float(ref_tok_counts.float().mean().detach().cpu())

                # Regroup ref log‑probs per prompt
                ref_logp_grouped: List[List[float]] = []
                idx = 0
                for _ in prompts:
                    vals = ref_logp_sum[idx : idx + num_generations].tolist()
                    ref_logp_grouped.append(vals)
                    idx += num_generations

                # 4) Build MaxEnt weights per prompt in log‑space and normalize
                weights_grouped: List[List[float]] = []
                for q_i, logp_i in zip(q_grouped, ref_logp_grouped):
                    # log w_i = (1/(tau+beta)) * log q_i + (beta/(tau+beta)) * log pi_ref_i
                    logw = []
                    for qi, lpi in zip(q_i, logp_i):
                        lq = math.log(max(qi, 1e-12))
                        logw.append((lq / denom) + (beta / denom) * lpi)
                    # normalize
                    w = torch.softmax(torch.tensor(logw, dtype=torch.float32), dim=0).tolist()
                    weights_grouped.append(w)
                weight_entropy = 0.0
                if weights_grouped:
                    for w in weights_grouped:
                        w_t = torch.tensor(w, dtype=torch.float32)
                        weight_entropy += float((-w_t.clamp(min=1e-12).log() * w_t).sum().item())
                    weight_entropy = weight_entropy / len(weights_grouped)

                # 5) Weighted MLE update on current model
                model.train()
                optim.zero_grad(set_to_none=True)

                # Compute model log‑probs for the same sequences
                cur_logp_sum, _ = _sequence_logprobs(
                    model, input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                # Negative log‑likelihood sums per sequence
                nll_sums = -cur_logp_sum  # [B*K]

                # Aggregate by group with weights
                loss = torch.tensor(0.0, device=device)
                idx = 0
                for w in weights_grouped:
                    group_nll = nll_sums[idx : idx + num_generations]
                    w_t = torch.tensor(w, device=device, dtype=group_nll.dtype)
                    # Weighted average NLL per group
                    loss = loss + (w_t * group_nll).sum()
                    idx += num_generations

                # Normalize by number of groups to keep loss scale stable
                loss = loss / max(1, len(prompts))
                loss_scalar = float(loss.detach().float().cpu())
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optim.step()

                global_step += 1
                if global_step % 10 == 0 and accelerator.is_main_process:
                    logging.getLogger(__name__).info(
                        "step %d | epoch %d | loss=%.4f | tau=%.3f beta=%.3f",
                        global_step,
                        epoch,
                        loss_scalar,
                        tau,
                        beta,
                    )
                if wandb_run is not None and accelerator.is_main_process:
                    current_lr = float(optim.param_groups[0].get("lr", lr))
                    _log_metrics(
                        {
                            "train/loss": loss_scalar,
                            "train/mean_reward": train_reward_mean,
                            "train/reward_std": train_reward_std,
                            "train/lr": current_lr,
                            "train/beta": beta,
                            "train/tau": tau,
                            "train/ref_logp_mean": ref_logp_mean,
                            "train/weight_entropy": weight_entropy,
                            "train/avg_completion_tokens": avg_completion_tokens,
                        },
                        global_step,
                    )
                if eval_enabled and eval_every and (global_step % eval_every == 0):
                    _run_validation(global_step)
    finally:
        if wandb_run is not None and accelerator.is_main_process:
            try:
                wandb_run.finish()
            except Exception as exc:  # pragma: no cover - defensive logging
                LOG.warning("Failed to close W&B run cleanly: %s", exc)

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        out_dir = getattr(training_args, "output_dir", "./maxent-grpo-out")
        os.makedirs(out_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(out_dir)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    # Keep the same CLI as src/grpo.py for compatibility with recipes
    try:
        from trl import ModelConfig, TrlParser
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - CLI import guard for docs/CI
        print(
            "This script requires TRL installed to parse configs (pip install trl).",
            file=sys.stderr,
        )
        raise

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    cli_script_args, cli_training_args, cli_model_args = parser.parse_args_and_config()
    main(cli_script_args, cli_training_args, cli_model_args)
