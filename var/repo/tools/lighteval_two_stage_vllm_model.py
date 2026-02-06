"""Two-stage (think -> answer) vLLM-backed LightEval model.

This is intended to be used with `lighteval custom` so we can perform two
sequential generations per prompt:
1) generate a reasoning trace wrapped in <think>...</think>
2) generate the final answer conditioned on that trace, wrapped in <answer>...</answer>
"""

from __future__ import annotations

from typing import Optional

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.models.model_output import GenerativeResponse, LoglikelihoodResponse
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)


def _wrap_block(text: str, tag: str) -> str:
    raw = (text or "").strip()
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if open_tag in raw:
        if close_tag not in raw:
            return f"{raw}\n{close_tag}"
        return raw
    return f"{open_tag}\n{raw}\n{close_tag}"


def _extract_block(text: str, tag: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = raw.find(open_tag)
    if start != -1:
        start += len(open_tag)
        rest = raw[start:]
        end = rest.find(close_tag)
        if end != -1:
            return rest[:end].strip()
        return rest.strip()
    return raw


def _split_model_args(raw: str) -> tuple[str, dict[str, str]]:
    """Split a `key=value,...` string into `vllm_args` and `two_stage_*` extras."""

    raw = (raw or "").strip()
    if not raw:
        raise ValueError("model_name must be non-empty")
    if "=" not in raw:
        return raw, {}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    keep: list[str] = []
    extras: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            keep.append(part)
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key.startswith("two_stage_"):
            extras[key] = val
            continue
        keep.append(part)
    return ",".join(keep), extras


class TwoStageVLLMModel(LightevalModel):
    """Custom LightEval model that uses vLLM and does think->answer sequential decoding."""

    def __init__(self, config: CustomModelConfig):
        from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig

        self._config = config

        raw_model_args, extras = _split_model_args(config.model_name)
        if "=" not in raw_model_args:
            # Accept a plain HF id or local path.
            raw_model_args = f"model_name={raw_model_args}"

        self.think_max_new_tokens = int(
            extras.get(
                "two_stage_think_max_new_tokens",
                "1024",
            )
        )
        self.think_temperature = float(extras.get("two_stage_think_temperature", "0"))
        self.answer_max_new_tokens = int(extras.get("two_stage_answer_max_new_tokens", "512"))

        vllm_cfg = VLLMModelConfig.from_args(raw_model_args)
        self._inner = VLLMModel(config=vllm_cfg)
        # LightEval's pipeline expects `model.model_info` to exist.
        self.model_info = getattr(self._inner, "model_info", None)

    @property
    def tokenizer(self):
        return self._inner.tokenizer

    @property
    def add_special_tokens(self) -> bool:
        return self._inner.add_special_tokens

    @property
    def max_length(self) -> int:
        return self._inner.max_length

    @property
    def disable_tqdm(self) -> bool:
        return getattr(self._inner, "disable_tqdm", False)

    def cleanup(self):
        return self._inner.cleanup()

    def _generate_texts(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        stop_tokens: list[str],
        num_samples: int,
        temperature: Optional[float],
        returns_logits: bool,
    ) -> list[GenerativeResponse]:
        from vllm import SamplingParams

        tok = self.tokenizer(prompts, add_special_tokens=self.add_special_tokens)
        inputs: list[list[int]] = tok["input_ids"]

        # Keep prompts within the configured max length.
        # (Match lighteval's vllm backend behavior: left truncate when needed.)
        if max_new_tokens is not None:
            context_keep = max(1, self.max_length - int(max_new_tokens))
            inputs = [row[-context_keep:] for row in inputs]
        else:
            inputs = [row[-self.max_length :] for row in inputs]

        sampling_params = SamplingParams(**self._inner._config.generation_parameters.to_vllm_dict())
        sampling_params.n = num_samples
        sampling_params.max_tokens = max_new_tokens
        sampling_params.stop = stop_tokens
        sampling_params.logprobs = 1 if returns_logits else 0
        if temperature is not None:
            sampling_params.temperature = float(temperature)

        outputs = self._inner.model.generate(
            prompt_token_ids=inputs,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        results: list[GenerativeResponse] = []
        for out in outputs:
            output_token_ids = [o.token_ids for o in out.outputs]
            texts = [o.text for o in out.outputs]
            logprobs = [o.logprobs for o in out.outputs] or []
            token_logprobs = None
            if returns_logits and logprobs and output_token_ids:
                token_logprobs = [
                    lp[token_id].logprob
                    for token_id, lp in zip(output_token_ids[0], logprobs[0])
                ]
            results.append(
                GenerativeResponse(
                    result=texts,
                    logits=token_logprobs,
                    input_tokens=getattr(out, "prompt_token_ids", []) or [],
                    generated_tokens=list(output_token_ids),
                )
            )
        return results

    def greedy_until(self, requests: list[GreedyUntilRequest]) -> list[GenerativeResponse]:
        if not requests:
            return []

        first = requests[0]
        if any((r.num_samples or 1) != (first.num_samples or 1) for r in requests):
            raise ValueError("TwoStageVLLMModel expects all requests to share the same num_samples.")
        if any((r.generation_size or 0) != (first.generation_size or 0) for r in requests):
            raise ValueError("TwoStageVLLMModel expects all requests to share the same generation_size.")
        if any((r.do_sample or False) != (first.do_sample or False) for r in requests):
            raise ValueError("TwoStageVLLMModel expects all requests to share the same do_sample.")

        # NOTE: LightEval has already built the full `request.context` (system prompt,
        # few-shots, etc). We append stage instructions here.
        base_prompts = [req.context for req in requests]

        num_samples = int(first.num_samples or 1)

        think_prompts = [
            prompt
            + "\n\nFirst, write your full reasoning inside <think>...</think>."
            for prompt in base_prompts
        ]
        think_max = min(
            max(16, int(self.think_max_new_tokens)),
            max(16, int(first.generation_size or self.think_max_new_tokens)),
        )
        think_responses = self._generate_texts(
            think_prompts,
            max_new_tokens=think_max,
            stop_tokens=["</think>"],
            num_samples=num_samples,
            temperature=self.think_temperature,
            returns_logits=False,
        )

        think_blocks_by_request: list[list[str]] = []
        for resp in think_responses:
            blocks: list[str] = []
            for raw in (resp.result or [])[:num_samples]:
                blocks.append(_wrap_block(_extract_block(raw, "think"), "think"))
            while len(blocks) < num_samples:
                blocks.append("<think>\n</think>")
            think_blocks_by_request.append(blocks)

        answer_prompts_flat: list[str] = []
        think_flat: list[str] = []
        for prompt, blocks in zip(base_prompts, think_blocks_by_request):
            for think in blocks:
                think_flat.append(think)
                answer_prompts_flat.append(
                    prompt
                    + "\n\nPrevious reasoning:\n"
                    + think
                    + "\n\nIgnore any earlier formatting/tag instructions. Now output only the final answer."
                )

        # Do NOT require or rely on </answer> stop tokens; cap answer length explicitly.
        stop_tokens: list[str] = []
        answer_max = min(int(first.generation_size or self.answer_max_new_tokens), int(self.answer_max_new_tokens))
        temperature = None
        if not first.do_sample:
            temperature = 0.0

        answer_responses = self._generate_texts(
            answer_prompts_flat,
            max_new_tokens=answer_max,
            stop_tokens=stop_tokens,
            num_samples=1,
            temperature=temperature,
            returns_logits=False,
        )

        stitched: list[GenerativeResponse] = []
        flat_i = 0
        for req_i, _req in enumerate(requests):
            combined: list[str] = []
            generated_tokens: list[list[int]] = []
            first_resp_for_req: GenerativeResponse | None = None

            for sample_i in range(num_samples):
                resp = answer_responses[flat_i]
                flat_i += 1
                if first_resp_for_req is None:
                    first_resp_for_req = resp

                ans_raw = resp.result[0] if resp.result else ""
                ans_block = _wrap_block(_extract_block(ans_raw, "answer"), "answer")
                think = think_blocks_by_request[req_i][sample_i]
                combined.append(f"{think}\n{ans_block}")

                if resp.generated_tokens and resp.generated_tokens[0]:
                    generated_tokens.append(resp.generated_tokens[0])
                else:
                    generated_tokens.append([])

            if not first_resp_for_req:
                first_resp_for_req = GenerativeResponse(
                    result=[],
                    logits=None,
                    input_tokens=[],
                    generated_tokens=[],
                )
            stitched.append(
                GenerativeResponse(
                    result=combined,
                    logits=None,
                    input_tokens=first_resp_for_req.input_tokens,
                    generated_tokens=generated_tokens,
                    truncated_tokens_count=first_resp_for_req.truncated_tokens_count,
                    padded_tokens_count=first_resp_for_req.padded_tokens_count,
                )
            )
        return stitched

    def loglikelihood(self, requests: list[LoglikelihoodRequest]) -> list[LoglikelihoodResponse]:
        raise NotImplementedError("TwoStageVLLMModel only supports greedy_until for now.")

    def loglikelihood_rolling(self, requests: list[LoglikelihoodRollingRequest]):
        raise NotImplementedError("TwoStageVLLMModel only supports greedy_until for now.")

    def loglikelihood_single_token(self, requests: list[LoglikelihoodSingleTokenRequest]):
        raise NotImplementedError("TwoStageVLLMModel only supports greedy_until for now.")
