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
"""

# MaxEnt‑GRPO training entrypoint (sequence‑level maximum‑entropy variant).
#
# This script implements a lightweight, self‑contained training loop that
# realizes the per‑context MaxEnt update described in the idea document:
#   max_pi  E_pi[log q] + tau * H(pi) - beta * KL(pi || pi_ref)
# whose unique maximizer is
#   pi* ∝ q^{1/(tau+beta)} * pi_ref^{beta/(tau+beta)}
#
# We operationalize this at candidate (sequence) level by:
# 1) Generating K completions per prompt (via vLLM server if enabled, else
#    using the local model's .generate()).
# 2) Computing listwise q from per‑candidate utilities (rewards), with a small
#    epsilon floor for full support.
# 3) Computing sequence log‑probabilities under a frozen reference model.
# 4) Building per‑candidate weights w_i ∝ q_i^{1/(tau+beta)} * pi_ref_i^{beta/(tau+beta)}
#    in log‑space and normalizing within each prompt group.
# 5) Performing a weighted MLE update on the current model with those weights
#    (no extra KL term is required since the pi_ref factor is in w_i).
#
# Notes
# - This is intentionally minimal to keep the code readable and dependency‑
#   light. It does not attempt to replicate all bells/whistles of TRL's
#   GRPOTrainer (e.g., KL controllers, schedulers, DDP, etc.). It is a
#   reference implementation meant for single‑process training or prototyping.
# - The same dataset/model utilities and reward functions from this repo are
#   reused for consistency with src/grpo.py.

from __future__ import annotations

import math
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers
from transformers import AutoModelForCausalLM

from configs import GRPOConfig, ScriptArguments
from rewards import get_reward_funcs
from utils.data import get_dataset
from utils.model_utils import get_model, get_tokenizer
from utils.vllm_patch import safe_generate


# -------------------------------
# Helpers
# -------------------------------

def _to_prompt(example: Dict, tokenizer, prompt_column: str, system_prompt: str | None) -> Dict:
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


def _group_softmax(xs: List[float], temperature: float = 1.0, eps: float = 1e-6) -> List[float]:
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


def _prepare_labels_for_ce(input_ids: torch.Tensor, prompt_lengths: List[int]) -> torch.Tensor:
    """Create labels tensor with prompt tokens masked as -100 for CE.

    :param input_ids: LongTensor [B, T]
    :param prompt_lengths: list of ints (len B) giving prompt token counts
    :returns: LongTensor labels [B, T] with prompt labels masked to -100
    """
    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lengths):
        labels[i, :plen] = -100
    return labels


@torch.no_grad()
def _sequence_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per‑sequence (completion) log‑prob sums and token counts.

    Uses teacher forcing on the full prompt+completion with labels masked to
    only score completion tokens. Returns the sum of log‑probs per sequence and
    the number of scored tokens per sequence.
    """
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


def _batch_tokenize_pairs(tokenizer, prompts: List[str], completions: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
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


def main(script_args: ScriptArguments, training_args: GRPOConfig, model_args):
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

    def _map_fn(ex):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ref_model.to(device)

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

    global_step = 0
    for epoch in range(num_epochs):
        for batch in train_loader:
            prompts: List[str] = batch["prompt"]
            answers: List[str] = batch["answer"]

            # 1) Generate K candidates per prompt
            if use_vllm:
                grouped_comps = safe_generate(
                    prompts=prompts,
                    url=vllm_url,
                    max_tokens=max_completion_len,
                    temperature=gen_temperature,
                    top_p=gen_top_p,
                    n=num_generations,
                    stream=False,
                    tokenizer=tokenizer,
                )
            else:
                # Local generation fallback (simple loop to avoid large VRAM use)
                grouped_comps = []
                for p in prompts:
                    # Encode prompt
                    enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
                    outs: List[str] = []
                    with torch.no_grad():
                        gen_out = model.generate(
                            **enc,
                            do_sample=True,
                            temperature=gen_temperature,
                            top_p=gen_top_p,
                            max_new_tokens=max_completion_len,
                            num_return_sequences=num_generations,
                        )
                    for i in range(gen_out.shape[0]):
                        text = tokenizer.decode(gen_out[i], skip_special_tokens=True)
                        # Strip the prompt prefix to approximate the completion
                        outs.append(text[len(p):])
                    grouped_comps.append(outs)

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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optim.step()

            global_step += 1
            if global_step % 10 == 0:
                logging.getLogger(__name__).info(
                    "step %d | epoch %d | loss=%.4f | tau=%.3f beta=%.3f",
                    global_step,
                    epoch,
                    float(loss.detach().cpu()),
                    tau,
                    beta,
                )

    # Save final model
    out_dir = getattr(training_args, "output_dir", "./maxent-grpo-out")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
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

    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    cli_script_args, cli_training_args, cli_model_args = parser.parse_args_and_config()
    main(cli_script_args, cli_training_args, cli_model_args)
