#!/usr/bin/env python3
"""Single-GPU online MaxRL/Re:Max pilot using the production paper loss.

The same HF/PEFT policy generates, scores and learns. There is no vLLM weight
bridge and no reward-model surrogate. Domain adapters expose load_tasks(config)
with task_id, prompt (complete rendered chat), family, split and verify(text).
One invocation runs one arm; use the same config and seed for a matched pair.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from oat.utils.ops import masked_sum
from oat_drgrpo.args import ZeroMathArgs
from oat_drgrpo.learner.base import ZeroMathLearnerBaseMixin
from oat_drgrpo.learner.grpo import ZeroMathGrpoMixin
from oat_drgrpo.maxrl import binary_maxrl_advantages
from oat_drgrpo.online_canonical_bank import VerifiedCanonicalReplayGroup


SCHEMA = "real-domain-online-maxrl-remax-pilot-20260921-v2-matched-scoring-width"
DEFAULTS = {
    "seed": 77123, "updates": 32, "group_size": 16,
    "max_new_tokens": 1024, "max_context_tokens": 8192,
    "generation_batch_size": 4, "train_microbatch_size": 1,
    "replay_capacity": 16, "replay_alpha": 0.1,
    "learning_rate": 1e-5, "lora_rank": 16, "lora_alpha": 32,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "max_grad_norm": 1.0, "cliprange": 0.2,
    "eval_samples": 64, "eval_seed": 97123,
    "checkpoint_every": 16, "execution_workers": 8,
    "max_gpu_hours": 3.0, "initial_evaluation": True,
    "attention_implementation": "sdpa",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_rows(handle: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def resolve_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    config = {**DEFAULTS, **dict(raw)}
    for name in ("model", "model_revision", "adapter_module", "adapter_config", "train_ids", "eval_ids"):
        if name not in config:
            raise ValueError(f"missing configuration field {name}")
    if not config["train_ids"] or not config["eval_ids"]:
        raise ValueError("nonempty explicit train and evaluation IDs required")
    train, evaluation = list(config["train_ids"]), list(config["eval_ids"])
    if len(set(train)) != len(train) or len(set(evaluation)) != len(evaluation) or set(train) & set(evaluation):
        raise ValueError("training/evaluation IDs must be unique and disjoint")
    for name in ("updates", "group_size", "max_new_tokens", "max_context_tokens", "generation_batch_size", "train_microbatch_size", "replay_capacity", "lora_rank", "lora_alpha", "eval_samples", "checkpoint_every", "execution_workers"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["train_microbatch_size"] != 1:
        raise ValueError("matched behavior/live scoring currently requires microbatch size one")
    if config["group_size"] < 2 or config["group_size"] % config["train_microbatch_size"]:
        raise ValueError("microbatch must divide the complete MaxRL group")
    if config["replay_capacity"] != 16:
        raise ValueError("this pilot fixes a sixteen-mode replay capacity")
    if config["generation_batch_size"] > config["group_size"]:
        raise ValueError("generation batch must not exceed training group")
    for name in ("learning_rate", "replay_alpha", "max_grad_norm", "max_gpu_hours", "cliprange"):
        if not math.isfinite(float(config[name])) or float(config[name]) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config["replay_alpha"] != 0.1:
        raise ValueError("pilot uses the production fixed replay alpha 0.1")
    if config["max_new_tokens"] >= config["max_context_tokens"]:
        raise ValueError("response cap must leave room for a prompt")
    if config["attention_implementation"] not in ("sdpa", "eager"):
        raise ValueError("unsupported attention implementation")
    if len(str(config["model_revision"])) != 40 or any(c not in "0123456789abcdef" for c in str(config["model_revision"])) or Path(config["model"]).name != config["model_revision"]:
        raise ValueError("model must be a local revision-named snapshot")
    return config


def validate_verdict(verdict: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(verdict)
    if not isinstance(result.get("accepted"), bool) or not isinstance(result.get("hard_violations"), list):
        raise ValueError("adapter must return boolean acceptance and hard violations")
    if result["hard_violations"]:
        raise RuntimeError(f"verifier hard violation: {result['hard_violations']}")
    key = result.get("canonical_key")
    if result["accepted"] != (isinstance(key, str) and bool(key)):
        raise ValueError("binary reward and verified canonical key disagree")
    if not result["accepted"] and key is not None:
        raise ValueError("rejected responses cannot carry a mode key")
    json.dumps(result, allow_nan=False)
    return result


def trim_response(tokens: Sequence[int], eos_ids: set[int]) -> tuple[int, ...]:
    """Keep original generated tokens, including the first EOS, excluding pad."""
    result = []
    for token in tokens:
        result.append(int(token))
        if int(token) in eos_ids:
            break
    if not result:
        raise ValueError("empty generated response")
    return tuple(result)


def tensor_batch(prompt: Sequence[int], responses: Sequence[Sequence[int]], pad: int, device: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not prompt or not responses or any(not response for response in responses):
        raise ValueError("nonempty prompt and responses required")
    width = len(prompt) + max(map(len, responses))
    ids = torch.full((len(responses), width), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    masks = torch.zeros((len(responses), width - 1), dtype=torch.float32, device=device)
    for i, response in enumerate(responses):
        sequence = tuple(prompt) + tuple(response)
        ids[i, :len(sequence)] = torch.tensor(sequence, device=device)
        attention[i, :len(sequence)] = 1
        masks[i, len(prompt) - 1:len(sequence) - 1] = 1
    return ids, attention, masks


@dataclass
class ReplayBank:
    """First discovered policy exemplar per task/key; deterministic global cursor."""
    capacity: int = 16
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    cursor: int = 0

    def observe(self, task_id: str, prompt: Sequence[int], samples: Sequence[Mapping[str, Any]]) -> int:
        added = 0
        for sample in samples:
            verdict = validate_verdict(sample["verdict"])
            if not verdict["accepted"]:
                continue
            entry = self.entries.setdefault(task_id, {"prompt": list(prompt), "modes": {}})
            if entry["prompt"] != list(prompt):
                raise ValueError("prompt changed for a banked task")
            modes, key = entry["modes"], verdict["canonical_key"]
            if key in modes:
                modes[key]["fresh_count"] += 1
            elif len(modes) < self.capacity:
                tokens = list(sample["token_ids"])
                if not tokens:
                    raise ValueError("verified exemplar lacks original policy tokens")
                modes[key] = {"token_ids": tokens, "fresh_count": 1, "request_id": sample["request_id"], "text_sha256": hashlib.sha256(sample["text"].encode()).hexdigest()}
                added += 1
        return added

    def next_group(self) -> tuple[str | None, list[VerifiedCanonicalReplayGroup]]:
        tasks = sorted(self.entries)
        if not tasks:
            return None, []
        task_id = tasks[self.cursor % len(tasks)]
        self.cursor += 1
        entry = self.entries[task_id]
        keys = sorted(entry["modes"])
        return task_id, [VerifiedCanonicalReplayGroup(
            prompt_token_ids=tuple(entry["prompt"]), outcome_keys=tuple(keys),
            response_token_ids=tuple(tuple(entry["modes"][key]["token_ids"]) for key in keys),
            fresh_observation_counts=tuple(entry["modes"][key]["fresh_count"] for key in keys),
        )]

    def state(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "entries": self.entries, "cursor": self.cursor}


class TorchAccumulationStrategy:
    """Exactly the accumulation division expected by the production learner."""
    def __init__(self, accumulation: int, max_norm: float):
        self.grad_acc_step = accumulation
        self.max_norm = max_norm
        self.micro_calls = 0
        self.updates = 0
        self.last_grad_norm = 0.0

    def backward(self, loss: torch.Tensor, model: Any, optimizer: Any) -> None:
        (loss / self.grad_acc_step).backward()

    def get_gradient_norm(self, model: Any) -> float:
        gradients = [parameter.grad.detach().float().norm() for parameter in model.parameters() if parameter.grad is not None]
        return float(torch.stack(gradients).norm()) if gradients else 0.0

    def optimizer_step(self, optimizer: Any, model: Any, scheduler: Any) -> None:
        self.micro_calls += 1
        if self.micro_calls % self.grad_acc_step:
            return
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], self.max_norm, error_if_nonfinite=True)
        self.last_grad_norm = float(norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        self.updates += 1

    def is_rank_0(self) -> bool:
        return False


class ProductionLearner(ZeroMathGrpoMixin, ZeroMathLearnerBaseMixin):
    def _should_skip_baseline_grad_norm_logging(self) -> bool:
        return False


def build_learner(model: Any, tokenizer: Any, config: Mapping[str, Any], arm: str, optimizer: Any = None) -> ProductionLearner:
    learner = ProductionLearner()
    args = ZeroMathArgs()
    args.seed = config["seed"]
    args.train_batch_size = args.num_samples = config["group_size"]
    args.train_batch_size_per_device = config["train_microbatch_size"]
    args.num_ppo_epochs = 1
    args.critic_type = "drgrpo"
    args.maxrl_task_objective = True
    args.beta = args.maxent_alpha = args.policy_entropy_coef = 0.0
    args.temperature = 1.0
    args.cliprange = config["cliprange"]
    args.reinforce_update = False
    args.generate_max_length = config["max_new_tokens"]
    args.online_canonical_replay = True
    args.online_canonical_replay_objective = "verified_likelihood_per_rollout"
    args.online_canonical_replay_alpha = config["replay_alpha"]
    args.online_canonical_replay_compute_only = arm == "maxrl"
    args.online_canonical_replay_key_weighting = "uniform"
    args.online_canonical_replay_bank_normalized = False
    # Zero-advantage truncation is a production optimization; preserve complete
    # pilot rows so measured memory/work reflects the actual response budget.
    args.baseline_zero_adv_response_tokens = config["max_new_tokens"]
    learner.args, learner.model, learner.tokenizer = args, model, tokenizer
    learner.strategy = TorchAccumulationStrategy(config["group_size"] // config["train_microbatch_size"], config["max_grad_norm"])
    learner.optimizer = optimizer if optimizer is not None else torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    learner.scheduler = None
    learner.masked_aggregator = functools.partial(masked_sum, constant_normalizer=config["max_new_tokens"])
    learner._canonical_action_token_ids = None
    learner._canonical_action_space = None
    learner._invalid_scoring_token_ids_warned_contexts = set()
    learner._invalid_logit_columns_warned_contexts = set()
    learner._baseline_grad_norm_logging_disabled_warned = False
    return learner


def policy_update(learner: ProductionLearner, prompt: Sequence[int], samples: Sequence[Mapping[str, Any]], groups: Sequence[VerifiedCanonicalReplayGroup]) -> dict[str, float]:
    model, args = learner.model, learner.args
    device = next(model.parameters()).device
    if args.train_batch_size_per_device != 1:
        raise ValueError("matched behavior/live scoring currently requires microbatch size one")
    if len(samples) != args.num_samples:
        raise ValueError("a complete fresh on-policy group is required")
    ids, attention, masks = tensor_batch(prompt, [sample["token_ids"] for sample in samples], learner.tokenizer.pad_token_id, device)
    rewards = torch.tensor([float(validate_verdict(s["verdict"])["accepted"]) for s in samples], device=device)
    advantages = binary_maxrl_advantages(rewards[None])[0]
    old = torch.zeros_like(masks)
    upper = learner._resolve_scoring_vocab_upper_bound(model)
    model.eval()
    # These are behavior log probabilities of the SAME unchanged model which
    # generated the raw token IDs, with exactly T=1/top-p=1/unrestricted top-k.
    with torch.no_grad():
        for start in range(0, len(samples), args.train_batch_size_per_device):
            stop = start + args.train_batch_size_per_device
            # The production learner trims each live microbatch to its final
            # attention position. Match that rectangle before behavior scoring:
            # BF16/SDPA can change logps with padded width even at fixed weights.
            # Microbatch one makes this width invariant to learner row shuffling.
            occupied = attention[start:stop].sum(0)
            empty_positions = torch.where(occupied == 0)[0]
            width = int(empty_positions[0]) if len(empty_positions) else int(attention.shape[1])
            mb_ids = ids[start:stop, :width]
            mb_attention = attention[start:stop, :width]
            mb_masks = masks[start:stop, :width - 1]
            logits = model(mb_ids, attention_mask=mb_attention)["logits"]
            logits = learner._mask_invalid_scoring_logit_columns(logits, valid_vocab_size=upper, context="pilot_behavior_logps")
            old[start:stop, :width - 1], _ = learner._policy_logps_and_optional_entropy(logits, mb_ids, mb_masks, need_entropy=False)
            del logits
    if not bool(torch.isfinite(old[masks.bool()]).all()):
        raise RuntimeError("nonfinite behavior log probabilities")
    model.train()
    before = learner.strategy.updates
    infos = learner._baseline_update_with_precomputed_advantages(
        input_ids=ids, att_mask=attention, prompt_id_lens=[len(prompt)] * len(samples),
        loss_masks=torch.ones(len(samples), device=device), response_masks=masks,
        logps=old, ref_logps=None, advantages=advantages[:, None], final_rewards=rewards[:, None],
        policy_vocab_upper_bound=upper, canonical_replay_groups=list(groups),
    )
    if learner.strategy.updates != before + 1:
        raise RuntimeError("production learner did not make exactly one optimizer update")
    values = {key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value) for key, value in infos.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("nonfinite production diagnostics")
    if groups:
        applied = values["canonical_replay_applied_score_gradient_l2"]
        if (applied == 0) != bool(args.online_canonical_replay_compute_only):
            raise RuntimeError("replay treatment derivative differs from registered arm")
    values.update({"actual_gradient_norm": learner.strategy.last_grad_norm, "fresh_successes": float(rewards.sum()), "mixed_group": float(0 < rewards.sum() < len(samples)), "fresh_tokens": float(masks.sum()), "optimizer_updates": float(learner.strategy.updates)})
    return values


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pass_at_k(total: int, correct: int, k: int) -> float:
    return 1.0 if total - correct < k else 1.0 - math.comb(total - correct, k) / math.comb(total, k)


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from collections import Counter
    counts = Counter(s["verdict"]["canonical_key"] for s in samples if s["verdict"]["accepted"])
    correct, total = sum(counts.values()), len(samples)
    return {
        "samples": total, "accepted": correct, "mode_counts": dict(counts), "distinct_modes": len(counts),
        **{f"pass_at_{k}": pass_at_k(total, correct, k) for k in (1, 8, 32) if k <= total},
        **{f"distinct_valid_at_{k}": sum(pass_at_k(total, n, k) for n in counts.values()) for k in (8, 32) if k <= total},
        "pcmd": 1 - sum(n * (n - 1) for n in counts.values()) / (correct * (correct - 1)) if correct >= 30 else None,
        "pcmd_eligible": correct >= 30,
    }


def generate_samples(model: Any, tokenizer: Any, task: Any, prompt_tokens: Sequence[int], count: int, config: Mapping[str, Any], phase: str, step: int, candidate_file: Any, *, seed_base: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    from transformers import GenerationConfig
    model.eval()
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
    if not eos_ids or None in eos_ids:
        raise ValueError("a finite EOS token set is required")
    upper = min(len(tokenizer), int(model.config.vocab_size))
    generation = GenerationConfig(
        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        max_new_tokens=config["max_new_tokens"], eos_token_id=sorted(eos_ids),
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        num_beams=1, repetition_penalty=1.0, use_cache=True,
        suppress_tokens=list(range(upper, int(model.config.vocab_size))) or None,
    )
    raw = []
    generation_seconds = 0.0
    for offset in range(0, count, config["generation_batch_size"]):
        size = min(config["generation_batch_size"], count - offset)
        batch_seed = seed_base + offset
        seed_all(batch_seed)
        ids = torch.tensor([list(prompt_tokens)] * size, device=device)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.no_grad():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation)
        torch.cuda.synchronize()
        generation_seconds += time.monotonic() - started
        for i, sequence in enumerate(output.tolist()):
            if sequence[:len(prompt_tokens)] != list(prompt_tokens):
                raise RuntimeError("generation changed its prompt token prefix")
            tokens = trim_response(sequence[len(prompt_tokens):], eos_ids)
            text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            raw.append({"request_id": f"{phase}:{step}:{task.task_id}:{offset+i}", "task_id": task.task_id, "phase": phase, "step": step, "sample_index": offset+i, "batch_seed": batch_seed, "batch_offset": i, "token_ids": list(tokens), "text": text, "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(), "prompt_token_ids": list(prompt_tokens), "finish_reason": "eos" if tokens[-1] in eos_ids else "length"})
        # Persist exact output before executing generated code or classifying it.
        append_rows(candidate_file, raw[-size:])
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=config["execution_workers"]) as executor:
        verdicts = list(executor.map(task.verify, [row["text"] for row in raw]))
    verification_seconds = time.monotonic() - started
    samples = [{**row, "verdict": validate_verdict(verdict)} for row, verdict in zip(raw, verdicts)]
    return samples, {"generation_seconds": generation_seconds, "verification_seconds": verification_seconds, "generated_tokens": sum(len(row["token_ids"]) for row in samples)}


def read_checkpoint(path: Path, config_hash: str, arm: str) -> tuple[dict[str, Any], ReplayBank]:
    sealed = json.loads((path / "complete.json").read_text())
    if sealed.get("schema") != SCHEMA:
        raise ValueError("checkpoint trainer schema differs from corrected scoring contract")
    if sealed["config_sha256"] != config_hash or sealed.get("arm") != arm or sealed["bank_sha256"] != digest(path / "bank.json") or sealed["training_state_sha256"] != digest(path / "training.pt"):
        raise ValueError("checkpoint/config/arm identity drift")
    for relative, expected in sealed["adapter_files"].items():
        if digest(path / "adapter" / relative) != expected:
            raise ValueError("checkpoint adapter identity drift")
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if state["config_sha256"] != config_hash or state["completed_updates"] != sealed["completed_updates"]:
        raise ValueError("checkpoint training state differs from completion seal")
    return state, ReplayBank(**json.loads((path / "bank.json").read_text()))


def save_checkpoint(path: Path, model: Any, learner: ProductionLearner, bank: ReplayBank, completed: int, config_hash: str, arm: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.mkdir()
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    torch.save({"optimizer": learner.optimizer.state_dict(), "completed_updates": completed, "strategy_updates": learner.strategy.updates, "strategy_micro_calls": learner.strategy.micro_calls, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "config_sha256": config_hash}, temporary / "training.pt")
    write_json(temporary / "bank.json", bank.state())
    adapter_files = {p.relative_to(temporary / "adapter").as_posix(): digest(p) for p in sorted((temporary / "adapter").rglob("*")) if p.is_file()}
    write_json(temporary / "complete.json", {"schema": SCHEMA, "arm": arm, "completed_updates": completed, "config_sha256": config_hash, "bank_sha256": digest(temporary / "bank.json"), "training_state_sha256": digest(temporary / "training.pt"), "adapter_files": adapter_files})
    os.replace(temporary, path)


def run(config_path: Path, output: Path, arm: str, resume: Path | None = None, preflight: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    config = resolve_config(json.loads(config_path.read_text()))
    config_hash = identity(config)
    if output.exists():
        raise FileExistsError("new output directory required; resume into a new directory")
    output.mkdir(parents=True)
    module = importlib.import_module(config["adapter_module"])
    tasks = module.load_tasks(config["adapter_config"])
    task_map = {task.task_id: task for task in tasks}
    if len(task_map) != len(tasks):
        raise ValueError("duplicate adapter task ID")
    for task_id in config["train_ids"] + config["eval_ids"]:
        if task_id not in task_map or not task_map[task_id].prompt:
            raise ValueError(f"missing task or complete prompt: {task_id}")
    for task_id in config["train_ids"]:
        if task_map[task_id].split.lower() in {"test", "evaluation", "heldout", "held_out"}:
            raise ValueError("cannot train on an evaluation task")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = {task_id: tokenizer.encode(task_map[task_id].prompt, add_special_tokens=False) for task_id in config["train_ids"] + config["eval_ids"]}
    if any(not tokens or len(tokens) + config["max_new_tokens"] > config["max_context_tokens"] for tokens in prompts.values()):
        raise ValueError("task prompt exceeds frozen context rectangle; truncation is forbidden")
    record = {
        "schema": SCHEMA, "arm": arm, "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": config_hash, "input_config_sha256": digest(config_path),
        "adapter_module": str(Path(module.__file__).resolve()), "adapter_module_sha256": digest(Path(module.__file__)),
        "dataset_identity": module.dataset_identity(config["adapter_config"]) if hasattr(module, "dataset_identity") else None,
        "tasks": [{"task_id": task_id, "family": task_map[task_id].family, "split": task_map[task_id].split, "prompt_sha256": hashlib.sha256(task_map[task_id].prompt.encode()).hexdigest(), "prompt_tokens": len(prompts[task_id])} for task_id in prompts],
        "objective": {"fresh": "binary_maxrl_advantages plus production clipped token importance ratios", "fresh_normalizer": "group_size * max_new_tokens", "replay": "verified_likelihood_per_rollout; uniform one policy exemplar per prompt-mode", "replay_raw_alpha": config["replay_alpha"], "replay_effective_coefficient": config["replay_alpha"] * (config["group_size"] - 1) / config["group_size"]**2, "behavior_scoring": "attention-trimmed rectangle identical to each shuffled live microbatch; microbatch1 required", "compute_only": arm == "maxrl", "initial_bank": "empty", "scheduler": "one global sorted-task round-robin bank per update", "extra_reward_or_entropy_terms": False},
        "resume": str(resume.resolve()) if resume else None,
        "production_source_sha256": {name: digest(Path(importlib.import_module(name).__file__)) for name in ("oat_drgrpo.learner.grpo", "oat_drgrpo.learner.base", "oat_drgrpo.canonical_replay", "oat_drgrpo.maxrl", "oat_drgrpo.online_canonical_bank", "oat_drgrpo.scoring", "oat_drgrpo.tensor_utils", "oat_drgrpo.args")},
        "runner_sha256": digest(Path(__file__)),
        "model_config_sha256": digest(Path(config["model"]) / "config.json"),
    }
    write_json(output / "identity.json", record)
    if preflight:
        write_json(output / "result.json", {"status": "preflight_pass", **record})
        return record
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one allocated CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required")
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    resume_state = read_checkpoint(resume, config_hash, arm) if resume else None
    seed_all(config["seed"])
    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(config["model"], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=config["attention_implementation"], device_map={"": 0})
    if resume:
        model = PeftModel.from_pretrained(model, resume / "adapter", is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"], target_modules=config["lora_target_modules"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for child in model.modules():
        if isinstance(child, torch.nn.Dropout):
            child.p = 0.0
    learner = build_learner(model, tokenizer, config, arm)
    learner.optimizer.zero_grad(set_to_none=True)
    bank, completed = ReplayBank(config["replay_capacity"]), 0
    if resume:
        state, bank = resume_state
        learner.optimizer.load_state_dict(state["optimizer"])
        completed = int(state["completed_updates"])
        learner.strategy.updates = state["strategy_updates"]
        learner.strategy.micro_calls = state["strategy_micro_calls"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    load_seconds = time.monotonic() - load_started
    trainable = {name: hashlib.sha256(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for name, parameter in model.named_parameters() if parameter.requires_grad}
    record.update({"torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(), "initial_trainable_parameters_sha256": identity(trainable), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "model_load_seconds": load_seconds})
    write_json(output / "identity.json", record)
    torch.cuda.reset_peak_memory_stats()
    evaluation_results = {}

    def within_budget() -> None:
        if time.monotonic() - started >= config["max_gpu_hours"] * 3600:
            raise TimeoutError("registered allocated-GPU-hour bound reached")

    with (output / "candidates.jsonl").open("x") as candidates, (output / "verified.jsonl").open("x") as verified, (output / "metrics.jsonl").open("x") as metrics:
        def evaluate(phase: str) -> None:
            rows = []
            # Evaluation uses a separate fixed RNG namespace. It never observes
            # or mutates the training bank, even for teacher-forced replay.
            for task_index, task_id in enumerate(config["eval_ids"]):
                within_budget()
                samples, timing = generate_samples(model, tokenizer, task_map[task_id], prompts[task_id], config["eval_samples"], config, phase, completed, candidates, seed_base=config["eval_seed"] + task_index * 10000)
                append_rows(verified, samples)
                rows.append({"task_id": task_id, "family": task_map[task_id].family, **summarize_samples(samples), "timing": timing})
            evaluation_results[phase] = rows
            write_json(output / f"{phase}.json", {"completed_updates": completed, "tasks": rows})

        try:
            if not resume and config["initial_evaluation"]:
                evaluate("initial")
            for step in range(completed, config["updates"]):
                within_budget()
                task_id = config["train_ids"][step % len(config["train_ids"])]
                task = task_map[task_id]
                update_started = time.monotonic()
                samples, timing = generate_samples(model, tokenizer, task, prompts[task_id], config["group_size"], config, "train", step, candidates, seed_base=config["seed"] + 1000000 + step * 10000)
                append_rows(verified, samples)
                added = bank.observe(task_id, prompts[task_id], samples)
                replay_task, groups = bank.next_group()
                torch.cuda.synchronize()
                learn_started = time.monotonic()
                diagnostics = policy_update(learner, prompts[task_id], samples, groups)
                torch.cuda.synchronize()
                timing["learning_seconds"] = time.monotonic() - learn_started
                timing["total_update_seconds"] = time.monotonic() - update_started
                completed = step + 1
                row = {"completed_updates": completed, "task_id": task_id, "replay_task_id": replay_task, "new_modes": added, "bank_tasks": len(bank.entries), "bank_modes": sum(len(e["modes"]) for e in bank.entries.values()), "replay_modes": sum(len(group.outcome_keys) for group in groups), **timing, **diagnostics}
                append_rows(metrics, [row])
                write_json(output / "progress.json", {"status": "training", **row})
                if completed % config["checkpoint_every"] == 0 or completed == config["updates"]:
                    save_checkpoint(output / f"checkpoint-{completed}", model, learner, bank, completed, config_hash, arm)
                print(f"[real-domain-pilot] arm={arm} step={completed}/{config['updates']} successes={row['fresh_successes']} bank={row['bank_modes']} seconds={timing['total_update_seconds']:.2f}", flush=True)
            evaluate("final")
            result = {"schema": SCHEMA, "status": "complete", "arm": arm, "completed_updates": completed, "config_sha256": config_hash, "allocated_gpu_hours_during_runner": (time.monotonic()-started)/3600, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "bank_tasks": len(bank.entries), "bank_modes": sum(len(e["modes"]) for e in bank.entries.values()), "evaluation": evaluation_results}
            write_json(output / "result.json", result)
            return result
        except Exception as error:
            write_json(output / "result.json", {"schema": SCHEMA, "status": "failed", "completed_updates": completed, "arm": arm, "config_sha256": config_hash, "allocated_gpu_hours_during_runner": (time.monotonic()-started)/3600, "error": f"{type(error).__name__}: {error}"})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("maxrl", "remax"), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.arm, args.resume, args.preflight)


if __name__ == "__main__":
    main()
