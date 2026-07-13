"""Semantic helper mixin for Dr.X/listwise learning."""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..listwise import (
    coerce_non_negative_float,
    compute_semantic_tiebreak_from_anchor_logits,
    reshape_prompt_major_tensor,
    select_semantic_tiebreak_anchor_logits,
)
from ..math_grader import (
    extract_normalized_final_answer_for_clustering,
    extract_reasoning_signature_from_trace,
    extract_reasoning_trace_for_clustering,
)
from ..semantic_features import truncate_text_to_max_tokens
from ..semantic_remix import compute_anchor_relative_weights


class ZeroMathDrxSemanticMixin:
    """Semantic reward-shaping and mass helpers for Dr.X/listwise paths."""

    def _sequence_logps_grouped(
        self,
        per_token_logps: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
        *,
        length_normalize: bool,
        context: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_logps = (per_token_logps * response_masks).sum(dim=1)
        grouped_seq_logps = reshape_prompt_major_tensor(seq_logps, group_size)
        token_counts = reshape_prompt_major_tensor(
            response_masks.sum(dim=1).to(torch.float32),
            group_size,
        )
        if grouped_seq_logps is None or token_counts is None:
            raise ValueError(f"Could not reshape {context} into whole prompt groups.")
        if length_normalize:
            grouped_seq_logps = grouped_seq_logps / token_counts.clamp(min=1.0)
        return grouped_seq_logps, token_counts

    def _response_texts_grouped(
        self,
        input_ids: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
    ) -> list[list[str]]:
        label_ids = input_ids[:, 1:]
        rows: list[str] = []
        for row_ids, row_mask in zip(label_ids, response_masks):
            token_ids = row_ids[row_mask.to(torch.bool)].detach().cpu().tolist()
            rows.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        return [rows[i : i + group_size] for i in range(0, len(rows), group_size)]

    def _semantic_cluster_inputs(
        self,
        input_ids: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
        references_grouped: list[list[Any | None]] | None = None,
    ) -> tuple[
        list[list[str | None]],
        list[list[str | None]],
        list[list[str | None]],
    ]:
        response_texts_grouped = self._response_texts_grouped(
            input_ids,
            response_masks,
            group_size,
        )
        semantic_cluster_max_tokens = max(
            int(getattr(self.args, "maxent_semantic_cluster_max_tokens", 0)),
            0,
        )
        final_answer_keys_grouped: list[list[str | None]] = []
        reasoning_trace_texts_grouped: list[list[str | None]] = []
        reasoning_signature_keys_grouped: list[list[str | None]] = []
        for prompt_index, prompt_rows in enumerate(response_texts_grouped):
            prompt_refs = (
                [None] * len(prompt_rows)
                if references_grouped is None
                else list(references_grouped[prompt_index])
            )
            final_answer_keys_grouped.append(
                [
                    extract_normalized_final_answer_for_clustering(
                        row_text,
                        template=str(self.args.prompt_template),
                        gt_answer=prompt_refs[row_index],
                    )
                    for row_index, row_text in enumerate(prompt_rows)
                ]
            )
            prompt_reasoning_traces: list[str | None] = []
            for row_text in prompt_rows:
                reasoning_trace = extract_reasoning_trace_for_clustering(
                    row_text,
                    template=str(self.args.prompt_template),
                )
                if semantic_cluster_max_tokens > 0:
                    reasoning_trace, _ = truncate_text_to_max_tokens(
                        reasoning_trace,
                        tokenizer=self.tokenizer,
                        max_tokens=semantic_cluster_max_tokens,
                    )
                prompt_reasoning_traces.append(reasoning_trace)
            reasoning_trace_texts_grouped.append(prompt_reasoning_traces)
            reasoning_signature_keys_grouped.append(
                [
                    extract_reasoning_signature_from_trace(reasoning_trace)
                    for reasoning_trace in prompt_reasoning_traces
                ]
            )
        return (
            final_answer_keys_grouped,
            reasoning_trace_texts_grouped,
            reasoning_signature_keys_grouped,
        )

    def _semantic_trace_embeddings_grouped(
        self,
        *,
        reasoning_trace_texts_grouped: list[list[str | None]],
        valid_row_mask_grouped: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _empty_trace_embeddings() -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            return (
                torch.zeros(
                    (num_prompts, group_size, 1),
                    device=valid_row_mask_grouped.device,
                    dtype=torch.float32,
                ),
                torch.zeros_like(valid_row_mask_grouped, dtype=torch.bool),
                torch.tensor(
                    0.0,
                    device=valid_row_mask_grouped.device,
                    dtype=torch.float32,
                ),
            )

        if valid_row_mask_grouped.dim() != 2:
            raise ValueError("valid_row_mask_grouped must have shape [prompts, group].")
        num_prompts, group_size = valid_row_mask_grouped.shape
        if len(reasoning_trace_texts_grouped) != num_prompts:
            raise ValueError(
                "reasoning_trace_texts_grouped must match the prompt-major minibatch shape."
            )
        if float(self.args.maxent_semantic_embedding_similarity_threshold) > 1.0:
            return _empty_trace_embeddings()
        trace_model_wrapper = (
            self.ref_model if self.ref_model is not None else self.model
        )
        if trace_model_wrapper is None:
            return _empty_trace_embeddings()
        trace_model = self._unwrap_hidden_state_model(trace_model_wrapper)

        flat_texts: list[str] = []
        flat_positions: list[tuple[int, int]] = []
        valid_mask = valid_row_mask_grouped.to(torch.bool)
        for p in range(num_prompts):
            prompt_rows = reasoning_trace_texts_grouped[p]
            if len(prompt_rows) != group_size:
                raise ValueError(
                    "Each prompt must provide one reasoning trace per candidate."
                )
            for i in range(group_size):
                if not bool(valid_mask[p, i].item()):
                    continue
                trace_text = prompt_rows[i]
                if trace_text is None:
                    continue
                trace_text = str(trace_text).strip()
                if not trace_text:
                    continue
                flat_texts.append(trace_text)
                flat_positions.append((p, i))

        if not flat_texts:
            return _empty_trace_embeddings()

        batch_size = max(int(self.args.train_batch_size_per_device), 1)
        max_tokens = max(int(self.args.maxent_semantic_embedding_max_tokens), 1)
        model_vocab_upper_bound = self._resolve_scoring_vocab_upper_bound(trace_model)
        pooled_batches: list[torch.Tensor] = []
        batch_positions_all: list[list[tuple[int, int]]] = []
        truncated_trace_count = 0
        attempted_trace_count = 0

        with torch.no_grad():
            for start in range(0, len(flat_texts), batch_size):
                end = min(start + batch_size, len(flat_texts))
                batch_texts = flat_texts[start:end]
                batch_positions = flat_positions[start:end]
                batch_tokenized_no_trunc = self.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    truncation=False,
                )
                truncated_trace_count += sum(
                    1
                    for token_ids in batch_tokenized_no_trunc["input_ids"]
                    if len(token_ids) > max_tokens
                )
                attempted_trace_count += len(batch_texts)
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                trace_input_ids = encoded["input_ids"].to(valid_row_mask_grouped.device)
                trace_att_mask = encoded["attention_mask"].to(
                    valid_row_mask_grouped.device
                )
                trace_input_ids = self._sanitize_scoring_token_ids(
                    trace_input_ids,
                    upper_bound=model_vocab_upper_bound,
                    context="semantic_trace_input",
                )
                position_ids = trace_att_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(trace_att_mask == 0, 1)
                model_outputs = None
                model_call_kwargs = {
                    "input_ids": trace_input_ids,
                    "attention_mask": trace_att_mask,
                    "position_ids": position_ids,
                    "return_dict": True,
                }
                last_type_error: TypeError | None = None
                for optional_kwargs in (
                    {"output_hidden_states": True, "use_cache": False},
                    {"output_hidden_states": True},
                    {},
                ):
                    try:
                        model_outputs = trace_model(
                            **model_call_kwargs,
                            **optional_kwargs,
                        )
                        break
                    except TypeError as exc:
                        last_type_error = exc
                if model_outputs is None:
                    if last_type_error is not None:
                        raise last_type_error
                    raise RuntimeError(
                        "Trace embedding model did not return outputs for semantic clustering."
                    )
                hidden_states = getattr(model_outputs, "hidden_states", None)
                last_hidden = None
                if hidden_states is not None and len(hidden_states) > 0:
                    last_hidden = hidden_states[-1]
                else:
                    last_hidden = getattr(model_outputs, "last_hidden_state", None)
                if last_hidden is None and isinstance(model_outputs, (tuple, list)):
                    if len(model_outputs) > 0 and isinstance(
                        model_outputs[0],
                        torch.Tensor,
                    ):
                        last_hidden = model_outputs[0]
                if last_hidden is None:
                    get_embeddings = getattr(trace_model, "get_input_embeddings", None)
                    if not callable(get_embeddings):
                        raise RuntimeError(
                            "Trace embedding model does not expose hidden states or input embeddings."
                        )
                    embedding_module = get_embeddings()
                    last_hidden = embedding_module(trace_input_ids)
                last_hidden = last_hidden.to(torch.float32)
                trace_att_mask_f = trace_att_mask.to(torch.float32)
                pooled = (last_hidden * trace_att_mask_f.unsqueeze(-1)).sum(
                    dim=1
                ) / trace_att_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
                pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)
                pooled_batches.append(pooled)
                batch_positions_all.append(batch_positions)

        embedding_dim = int(pooled_batches[0].shape[-1])
        reasoning_trace_embeddings_grouped = torch.zeros(
            (num_prompts, group_size, embedding_dim),
            device=valid_row_mask_grouped.device,
            dtype=torch.float32,
        )
        reasoning_trace_valid_row_mask_grouped = torch.zeros_like(
            valid_row_mask_grouped,
            dtype=torch.bool,
        )
        for pooled, batch_positions in zip(pooled_batches, batch_positions_all):
            for offset, (p, i) in enumerate(batch_positions):
                reasoning_trace_embeddings_grouped[p, i] = pooled[offset]
                reasoning_trace_valid_row_mask_grouped[p, i] = True
        semantic_trace_truncated_frac = torch.tensor(
            float(truncated_trace_count) / float(max(attempted_trace_count, 1)),
            device=valid_row_mask_grouped.device,
            dtype=torch.float32,
        )
        return (
            reasoning_trace_embeddings_grouped,
            reasoning_trace_valid_row_mask_grouped,
            semantic_trace_truncated_frac,
        )

    def _seed_answer_keys_grouped(
        self,
        input_ids: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
        references_grouped: list[list[Any | None]] | None = None,
    ) -> list[list[str | None]]:
        """Return SEED-style answer-only cluster keys."""

        response_texts_grouped = self._response_texts_grouped(
            input_ids,
            response_masks,
            group_size,
        )
        final_answer_keys_grouped: list[list[str | None]] = []
        for prompt_index, prompt_rows in enumerate(response_texts_grouped):
            prompt_refs = (
                [None] * len(prompt_rows)
                if references_grouped is None
                else list(references_grouped[prompt_index])
            )
            prompt_answers: list[str | None] = []
            for row_index, row_text in enumerate(prompt_rows):
                answer = extract_normalized_final_answer_for_clustering(
                    row_text,
                    template=str(self.args.prompt_template),
                    gt_answer=prompt_refs[row_index],
                )
                prompt_answers.append(answer)
            final_answer_keys_grouped.append(prompt_answers)
        return final_answer_keys_grouped

    def _reference_seq_logps_grouped(
        self,
        *,
        input_ids: torch.Tensor,
        att_mask: torch.Tensor,
        response_masks: torch.Tensor,
        group_size: int,
        behavior_seq_logps_grouped: torch.Tensor,
        force_reference_scores: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        needs_reference_scores = (
            bool(force_reference_scores)
            or float(self.args.beta) > 0.0
            or coerce_non_negative_float(
                getattr(self.args, "maxent_candidate_kl_coef", 0.0),
                default=0.0,
            )
            > 0.0
        )
        if not needs_reference_scores:
            return torch.zeros_like(behavior_seq_logps_grouped), None
        if self.args.maxent_reference_logprobs_source == "behavior":
            return behavior_seq_logps_grouped.detach(), None
        if self.ref_model is None:
            logging.warning(
                "Listwise MaxEnt requested model reference log-probs but no ref_model "
                "is available; falling back to rollout behavior log-probs."
            )
            return behavior_seq_logps_grouped.detach(), None

        ref_logps = self._compute_batched_logps(
            self.ref_model,
            input_ids,
            att_mask,
            response_masks,
        )
        ref_seq_logps_grouped, _ = self._sequence_logps_grouped(
            ref_logps,
            response_masks,
            group_size,
            length_normalize=bool(self.args.maxent_length_normalize_ref),
            context="reference log-probs",
        )
        return ref_seq_logps_grouped, ref_logps

    def _semantic_neutral_tiebreak_rewards_grouped(
        self,
        *,
        behavior_seq_logps_grouped: torch.Tensor,
        reference_seq_logps_grouped: torch.Tensor,
        candidate_correctness_grouped: torch.Tensor,
        cluster_ids_grouped: torch.Tensor,
        valid_row_mask_grouped: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None, str | None]:
        """Return small semantic tie-break rewards for exact DrX neutral groups."""

        tie_alpha = coerce_non_negative_float(
            getattr(self.args, "maxent_reward_shaping_alpha", 0.0),
            default=0.0,
        )
        if tie_alpha <= 0.0:
            return (
                torch.zeros_like(candidate_correctness_grouped),
                None,
                None,
            )

        anchor_logits_grouped, anchor_source = select_semantic_tiebreak_anchor_logits(
            behavior_seq_logps_grouped=behavior_seq_logps_grouped.detach(),
            reference_seq_logps_grouped=reference_seq_logps_grouped.detach(),
            anchor=str(self.args.maxent_tiebreak_anchor),
            beta=float(self.args.beta),
            reference_available=True,
        )
        bonus_grouped, bonus_diag = compute_semantic_tiebreak_from_anchor_logits(
            anchor_logits_grouped=anchor_logits_grouped.to(
                device=candidate_correctness_grouped.device,
                dtype=candidate_correctness_grouped.dtype,
            ),
            candidate_correctness_grouped=candidate_correctness_grouped.detach(),
            cluster_ids_grouped=cluster_ids_grouped.to(
                device=candidate_correctness_grouped.device
            ),
            bonus_alpha=tie_alpha,
            bonus_clip_max=float(self.args.maxent_tiebreak_clip_max),
            valid_row_mask_grouped=valid_row_mask_grouped.detach(),
        )
        reward_scale_denom = max(abs(float(self.args.reward_scale)), 1e-8)
        return (
            reward_scale_denom * bonus_grouped.to(candidate_correctness_grouped.dtype),
            bonus_diag,
            anchor_source,
        )

    def _anchor_relative_semantic_mass_weights_grouped(
        self,
        *,
        behavior_seq_logps_grouped: torch.Tensor,
        reference_seq_logps_grouped: torch.Tensor,
        valid_row_mask_grouped: torch.Tensor,
        tau: float,
        candidate_kl_coef: float,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return anchor-relative utilities and semantic mass weights."""

        anchor_seq_logps_grouped, anchor_source = (
            select_semantic_tiebreak_anchor_logits(
                behavior_seq_logps_grouped=behavior_seq_logps_grouped.detach(),
                reference_seq_logps_grouped=reference_seq_logps_grouped.detach(),
                anchor=str(self.args.maxent_tiebreak_anchor),
                beta=float(candidate_kl_coef),
                reference_available=True,
            )
        )
        anchor_utility_grouped, anchor_mass_weights_grouped = (
            compute_anchor_relative_weights(
                anchor_seq_logps_grouped=anchor_seq_logps_grouped.to(
                    device=valid_row_mask_grouped.device,
                    dtype=torch.float32,
                ),
                tau=tau,
                candidate_kl_coef=candidate_kl_coef,
                valid_row_mask_grouped=valid_row_mask_grouped.detach(),
            )
        )
        return anchor_utility_grouped, anchor_mass_weights_grouped, anchor_source
