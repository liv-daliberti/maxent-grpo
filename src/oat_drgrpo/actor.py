"""Actor and oracle wiring for zero-math rollouts."""

from __future__ import annotations

import functools
import itertools
import logging
import time
from multiprocessing import TimeoutError
from multiprocessing.pool import ThreadPool
from typing import Any

import numpy as np
import torch
import tree
from oat.algorithms.ppo import PPOActor
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric, TrajectoryData
from transformers import AutoTokenizer

from .args import ZeroMathArgs
from .math_grader import (
    answer_tag_reward_fn,
    boxed_reward_fn,
    extract_normalized_final_answer_for_clustering,
)


class MATHOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for math answer grading."""

    def __init__(self, template, verifier_version) -> None:
        super().__init__()
        if template == "r1":
            math_reward_fn = answer_tag_reward_fn
        else:
            math_reward_fn = boxed_reward_fn
        self.math_reward_fn = functools.partial(
            math_reward_fn,
            fast=verifier_version == "fast",
        )
        # Avoid forking from the actor process after vLLM/CUDA initialization:
        # native state corruption here can surface later inside tokenizer/NCCL.
        self.mp_pool = ThreadPool(2)

    def get_reward(
        self,
        inputs: list[str],
        responses: list[str],
        references: list[str],
        batch_size: int = 4,
    ) -> tuple[torch.Tensor, Metric]:
        del inputs, batch_size

        rewards = []
        infos = []
        for resp, ref in zip(responses, references):
            res = self.mp_pool.apply_async(self.math_reward_fn, (resp, ref))
            try:
                info, reward = res.get(timeout=1)
                rewards.append(reward)
                infos.append(info)
            except TimeoutError:
                rewards.append(0.0)
                infos.append({"formatted": False})

        return torch.tensor(rewards), infos

    def compare(
        self,
        inputs: list[str],
        candidates_A: list[str],
        candidates_B: list[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> tuple[list[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""

        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info


class ZeroMathActor(PPOActor):
    def __init__(self, ipc_server, vllm_args, args: ZeroMathArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self._prompt_token_id_cache: dict[str, list[int]] = {}
        # OAT 0.0.9 configures actor sampling in __init__, while newer OAT
        # versions populate these fields in init(actor_id, save_path).
        if hasattr(self, "sampling_params"):
            self._configure_math_actor()

    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)
        self._configure_math_actor()

    def _configure_math_actor(self) -> None:
        if not hasattr(self, "_prompt_token_id_cache"):
            self._prompt_token_id_cache = {}
        if not hasattr(self, "prompt_tokenizer"):
            try:
                self.prompt_tokenizer = AutoTokenizer.from_pretrained(
                    self.args.pretrain,
                    trust_remote_code=True,
                    use_fast=False,
                )
                logging.info(
                    "actor prompt tokenizer type=%s is_fast=%s",
                    type(self.prompt_tokenizer).__name__,
                    getattr(self.prompt_tokenizer, "is_fast", None),
                )
            except Exception:
                logging.exception(
                    "Failed to load slow prompt tokenizer; falling back to actor tokenizer"
                )
                self.prompt_tokenizer = self.tokenizer

        self.oracle = MATHOracle(
            template=self.args.prompt_template,
            verifier_version=self.args.verifier_version,
        )

        if self.args.prompt_template in ["qwen_boxed", "qwen_math", "no"]:
            self.sampling_params.stop = None
            self.sampling_params.stop_token_ids = None
            self.eval_sampling_params.stop = None
            self.eval_sampling_params.stop_token_ids = None
        elif self.args.prompt_template == "r1":
            self.sampling_params.stop = ["</answer>"]
            self.sampling_params.include_stop_str_in_output = True
            self.eval_sampling_params.stop = ["</answer>"]
            self.eval_sampling_params.include_stop_str_in_output = True

    def _encode_formatted_prompts(
        self, formatted_prompts: list[str]
    ) -> list[list[int]]:
        """Encode prompts once in the actor before handing them to vLLM."""

        prompt_tokenizer = getattr(self, "prompt_tokenizer", self.tokenizer)
        bos_token = getattr(prompt_tokenizer, "bos_token", None) or getattr(
            self.tokenizer, "bos_token", None
        )
        tokenized_prompts = []
        for prompt in formatted_prompts:
            if bos_token:
                prompt = prompt.removeprefix(bos_token)
            cached_ids = self._prompt_token_id_cache.get(prompt)
            if cached_ids is None:
                cached_ids = list(prompt_tokenizer.encode(prompt))
                self._prompt_token_id_cache[prompt] = cached_ids
            tokenized_prompts.append(list(cached_ids))
        return tokenized_prompts

    def generate_for_mode_coverage(
        self,
        formatted_prompts: list[str],
        refs: list[str],
        n: int,
        temperature: float,
    ) -> dict[str, list]:
        """Sample n completions per prompt and return per-prompt rewards and answer keys.

        Used by the learner to compute sampled mode-coverage metrics during training
        without saving a checkpoint. Scoring uses the same oracle as training rollouts.
        """
        import vllm

        params = vllm.SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=self.eval_sampling_params.max_tokens,
            stop=self.eval_sampling_params.stop,
            stop_token_ids=self.eval_sampling_params.stop_token_ids,
            include_stop_str_in_output=getattr(
                self.eval_sampling_params, "include_stop_str_in_output", False
            ),
        )

        outputs = self.generate(formatted_prompts, params)

        # Flatten: score all (prompt, response) pairs in one oracle call.
        all_responses: list[str] = []
        all_refs_flat: list[str] = []
        counts: list[int] = []
        for output, ref in zip(outputs, refs):
            for sample in output.outputs:
                all_responses.append(sample.text.strip())
                all_refs_flat.append(ref)
            counts.append(len(output.outputs))

        rewards_tensor, _ = self.oracle.get_reward(
            [""] * len(all_responses),
            all_responses,
            all_refs_flat,
        )
        rewards_flat = rewards_tensor.tolist()

        answer_keys_flat = [
            extract_normalized_final_answer_for_clustering(
                resp,
                template=self.args.prompt_template,
                gt_answer=ref,
            )
            for resp, ref in zip(all_responses, all_refs_flat)
        ]

        per_prompt_rewards: list[list[float]] = []
        per_prompt_keys: list[list] = []
        idx = 0
        for count in counts:
            per_prompt_rewards.append(rewards_flat[idx : idx + count])
            per_prompt_keys.append(answer_keys_flat[idx : idx + count])
            idx += count

        return {"rewards": per_prompt_rewards, "answer_keys": per_prompt_keys}

    def step(
        self,
        prompts: list[str],
        formatted_prompts: list[str],
        references: list[str] | None = None,
    ) -> list[TrajectoryData]:
        """Generate trajectories and score them with the math oracle."""

        assert not self.eval_mode
        info = {}
        logging.info("actor start")

        st = time.time()
        formatted_prompt_token_ids = self._encode_formatted_prompts(formatted_prompts)
        outputs = self.generate(formatted_prompt_token_ids, self.sampling_params)

        candidates = []
        prompt_token_ids = []
        no_eos = []
        response_ids = []
        response_logprobs = []
        resp_lens = []
        for i in range(len(outputs)):
            prompt_token_ids.append(outputs[i].prompt_token_ids)
            candidates.append([])
            response_logprobs.append([])
            response_ids.append([])
            for k in range(self.sampling_params.n):
                candidates[i].append(outputs[i].outputs[k].text)
                no_eos.append(outputs[i].outputs[k].finish_reason == "length")
                token_ids = outputs[i].outputs[k].token_ids
                logps = outputs[i].outputs[k].logprobs
                logps = [item[token_ids[i]].logprob for i, item in enumerate(logps)]
                response_logprobs[i].append(logps)
                response_ids[i].append(token_ids)
                resp_lens.append(len(token_ids))

        info["actor/generate_time"] = time.time() - st

        st = time.time()
        rewards, oracle_infos = self.oracle.get_reward(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in prompts
                )
            ),
            tree.flatten(candidates),
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in references
                )
            ),
        )

        info["actor/verify_time"] = time.time() - st
        logging.info("actor reward %s", rewards.mean())
        info["actor/rewards"] = rewards.mean().item()
        info["actor/num_data"] = rewards.numel()
        info["actor/formatted"] = np.mean([i["formatted"] for i in oracle_infos])
        info["actor/response_tok_len"] = np.mean(resp_lens)
        info["actor/sampling_max_tokens"] = self.sampling_params.max_tokens
        info["actor/sampling_temperature"] = self.sampling_params.temperature

        rewards = rewards.reshape(len(prompts), -1)
        no_eos = np.array(no_eos).reshape(len(prompts), -1)
        info["actor/no_eos_count"] = no_eos.sum()

        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            reference = references[i] if references is not None else None
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                reward = rewards[i][j].item()
                if no_eos[i][j]:
                    reward = 0
                dense_rewards = [0] * len(response_ids[i][j])
                dense_rewards[-1] = reward
                trajectory = TrajectoryData(
                    prompt=prompt,
                    prompt_ids=prompt_token_ids[i],
                    response=candidates_per_prompt[j],
                    response_ids=response_ids[i][j],
                    response_logprobs=response_logprobs[i][j],
                    rewards=dense_rewards,
                    loss_mask=not no_eos[i][j] if self.args.ignore_no_eos else True,
                    info=info,
                )
                setattr(trajectory, "reference", reference)
                trajectory_data.append(trajectory)
        logging.info("actor finished data_len=%s", len(trajectory_data))
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle
