# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Model logprob computation and sequence-score assembly helpers."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
import inspect
import os
import time
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np

from .scoring_batching import iter_batch_slices
from .scoring_common import (
    LOG,
    TorchDevice,
    _PadTokenGuard,
    _SCORING_EXCEPTIONS,
    _TorchModuleLike,
    _autocast_context,
    _coerce_optional_int,
    _describe_embedding_module,
    _dist_all,
    _dist_any,
    _dist_collective_ready,
    _get_config_value,
    _get_embedding_vocab_size,
    _prefetch_iterator,
    _progress_log_enabled,
    _refresh_torch,
    _score_slice_log_enabled,
    _to_numpy_array,
    _weight_is_stub_tensor,
    _weight_is_two_dimensional,
)
from .types import (
    BatchingSettings,
    PreTrainedModel,
    ReferenceLogprobs,
    RuntimeHandles,
    ScoreBatch,
    SequenceScores,
    Tensor,
)
from .zero_utils import _maybe_zero_gather_params

torch = _refresh_torch()


def _summon_fsdp_full_param_context(model: PreTrainedModel) -> ContextManager[object]:
    """Return a context manager that gathers FSDP parameters when available."""
    summon_fn = getattr(model, "summon_full_params", None)
    summon_callable = cast(Optional[Callable[..., ContextManager[object]]], summon_fn)
    if not callable(summon_callable):
        return nullcontext()
    try:
        return summon_callable()
    except TypeError:
        try:
            return summon_callable(recurse=True)
        except TypeError:
            return nullcontext()


def _chunked_sequence_logprobs(
    model: PreTrainedModel,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    chunk_size: int,
    gather_full_params: bool = False,  # retained for parity
    zero_gather_all_ranks: bool = False,
    return_hidden: bool = False,
    pooling: str = "mean",
    return_entropy: bool = False,
    entropy_mode: str = "exact",
    return_token_logp: bool = False,
) -> Optional[
    Tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]
]:
    """Compute summed log-probabilities per sequence with optional chunking/pooled states/entropy."""

    torch_mod = _refresh_torch()
    slice_log = _score_slice_log_enabled()
    entropy_mode_norm = str(entropy_mode or "exact").strip().lower()
    if entropy_mode_norm in {"", "none"}:
        entropy_mode_norm = "exact"
    if entropy_mode_norm in {"exact", "full", "distribution"}:
        entropy_mode_norm = "exact"
    elif entropy_mode_norm in {
        "sample",
        "estimate",
        "estimated",
        "approx",
        "approximate",
        "token",
        "token_logp",
        "nll",
        "logp",
    }:
        entropy_mode_norm = "sample"
    else:
        if return_entropy:
            warned = getattr(_chunked_sequence_logprobs, "_entropy_mode_warned", False)
            if not warned:
                LOG.warning(
                    "Unknown entropy_mode=%s; falling back to 'exact'.",
                    entropy_mode,
                )
                setattr(_chunked_sequence_logprobs, "_entropy_mode_warned", True)
        entropy_mode_norm = "exact"
    use_exact_entropy = return_entropy and entropy_mode_norm == "exact"
    use_sample_entropy = return_entropy and entropy_mode_norm == "sample"
    _ = (chunk_size,)  # parity with distributed APIs; currently unused

    def _compress_completion_token_logp(
        token_logp: Tensor, token_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Return completion-only token logprobs and mask (pad to max completion length)."""
        try:
            torch_tensor = getattr(torch_mod, "Tensor", None)
            if torch_tensor is None:
                return token_logp, token_mask
            if not isinstance(token_logp, torch_tensor) or not isinstance(
                token_mask, torch_tensor
            ):
                return token_logp, token_mask
        except (TypeError, AttributeError):
            return token_logp, token_mask
        if token_logp.ndim < 2 or token_mask.ndim < 2:
            return token_logp, token_mask
        if token_logp.shape != token_mask.shape:
            return token_logp, token_mask
        mask_bool = token_mask != 0
        try:
            mask_int = mask_bool.to(dtype=getattr(torch_mod, "long", None))
        except _SCORING_EXCEPTIONS:
            try:
                mask_int = mask_bool.long()
            except _SCORING_EXCEPTIONS:
                return token_logp, token_mask
        counts = mask_int.sum(dim=1)
        if getattr(counts, "numel", lambda: 0)() == 0:
            return token_logp[:, :0], token_mask[:, :0]
        try:
            max_len = int(counts.max().item())
        except _SCORING_EXCEPTIONS:
            return token_logp, token_mask
        if max_len <= 0:
            try:
                empty = torch_mod.zeros(
                    (token_logp.shape[0], 0),
                    dtype=getattr(token_logp, "dtype", None),
                    device=getattr(token_logp, "device", None),
                )
            except _SCORING_EXCEPTIONS:
                empty = torch_mod.zeros((token_logp.shape[0], 0))
            return empty, empty
        try:
            comp_logp = torch_mod.zeros(
                (token_logp.shape[0], max_len),
                dtype=getattr(token_logp, "dtype", None),
                device=getattr(token_logp, "device", None),
            )
            comp_mask = torch_mod.zeros(
                (token_logp.shape[0], max_len),
                dtype=getattr(token_logp, "dtype", None),
                device=getattr(token_logp, "device", None),
            )
        except _SCORING_EXCEPTIONS:
            comp_logp = torch_mod.zeros((token_logp.shape[0], max_len))
            comp_mask = torch_mod.zeros((token_logp.shape[0], max_len))
        pos = mask_int.cumsum(dim=1) - 1
        try:
            rows, cols = mask_bool.nonzero(as_tuple=True)
        except (TypeError, AttributeError):
            try:
                nz = mask_bool.nonzero()
                rows, cols = nz[:, 0], nz[:, 1]
            except _SCORING_EXCEPTIONS:
                return token_logp, token_mask
        if getattr(rows, "numel", lambda: 0)() > 0:
            comp_cols = pos[rows, cols]
            try:
                comp_logp[rows, comp_cols] = token_logp[rows, cols]
                comp_mask[rows, comp_cols] = 1
            except _SCORING_EXCEPTIONS:
                return token_logp, token_mask
        return comp_logp, comp_mask

    # Fallback for lightweight stubs
    if not hasattr(model, "forward"):
        label_arr = np.asarray(getattr(labels, "arr", labels))
        valid_counts = (label_arr != -100).sum(axis=1)
        logp = torch_mod.tensor(
            np.zeros(len(valid_counts), dtype=float),
            dtype=getattr(torch_mod, "float32", None),
        )
        tok_tensor = torch_mod.tensor(
            valid_counts, dtype=getattr(torch_mod, "float32", None)
        )
        entropy_sum = (
            torch_mod.tensor(
                np.zeros(len(valid_counts), dtype=float),
                dtype=getattr(torch_mod, "float32", None),
            )
            if return_entropy
            else None
        )
        if return_token_logp:
            token_logp = torch_mod.zeros(
                (len(valid_counts), 0),
                dtype=getattr(torch_mod, "float32", None),
            )
            return logp, tok_tensor, None, entropy_sum, token_logp, token_logp
        return logp, tok_tensor, None, entropy_sum
    shape = getattr(input_ids, "shape", None)
    if shape and len(shape) > 1 and shape[1] == 0:
        batch_size = shape[0]
        zero = torch_mod.tensor(
            np.zeros(batch_size, dtype=float), dtype=getattr(torch_mod, "float32", None)
        )
        tok_tensor = torch_mod.tensor(
            np.zeros(batch_size, dtype=float), dtype=getattr(torch_mod, "float32", None)
        )
        entropy_sum = (
            torch_mod.tensor(
                np.zeros(batch_size, dtype=float),
                dtype=getattr(torch_mod, "float32", None),
            )
            if return_entropy
            else None
        )
        if return_token_logp:
            token_logp = torch_mod.zeros(
                (batch_size, 0),
                dtype=getattr(torch_mod, "float32", None),
            )
            return zero, tok_tensor, None, entropy_sum, token_logp, token_logp
        return zero, tok_tensor, None, entropy_sum
    # DeepSpeed ZeRO-3 shards parameters to 1-D partitions; gather embeddings
    # so the reference forward sees 2-D weights without full-parameter all-gather.
    gather_ctx = nullcontext()
    zero_gather_strategy = "none"
    dist = _dist_collective_ready(torch_mod)
    if gather_full_params:
        try:  # pragma: no cover - exercised in distributed runs
            import deepspeed

            params: list[Any] = []
            param_iter = getattr(model, "parameters", None)
            if callable(param_iter):
                try:
                    params = list(cast(Iterable[Any], param_iter()))
                except _SCORING_EXCEPTIONS:
                    params = []
            # Always invoke GatheredParameters when available, even with an empty
            # list, so stubbed environments can verify the context is entered.
            try:
                gather_ctx = deepspeed.zero.GatheredParameters(
                    params or [], modifier_rank=None
                )
            except TypeError:
                gather_ctx = deepspeed.zero.GatheredParameters(params or [])
            zero_gather_strategy = "manual_full"
        except ImportError:
            gather_ctx = nullcontext()
    else:
        try:  # pragma: no cover - exercised in distributed runs
            import deepspeed

            zero_mod = getattr(deepspeed, "zero", None)
            is_enabled_fn = (
                getattr(zero_mod, "is_enabled", None) if zero_mod is not None else None
            )
            if callable(is_enabled_fn) and is_enabled_fn():
                to_gather: list[Any] = []
                if os.environ.get("MAXENT_DISABLE_SCORING_ZERO_GATHER", "").strip():
                    gather_ctx = nullcontext()
                inp_emb = (
                    model.get_input_embeddings()
                    if hasattr(model, "get_input_embeddings")
                    else None
                )
                input_weight = (
                    getattr(inp_emb, "weight", None)
                    if inp_emb is not None and hasattr(inp_emb, "weight")
                    else None
                )
                out_emb = (
                    model.get_output_embeddings()
                    if hasattr(model, "get_output_embeddings")
                    else None
                )
                if out_emb is None and hasattr(model, "lm_head"):
                    out_emb = model.lm_head
                output_weight = (
                    getattr(out_emb, "weight", None)
                    if out_emb is not None and hasattr(out_emb, "weight")
                    else None
                )

                input_present_local = input_weight is not None
                output_present_local = output_weight is not None
                input_present_all = (
                    _dist_all(dist, input_present_local)
                    if dist is not None
                    else input_present_local
                )
                output_present_all = (
                    _dist_all(dist, output_present_local)
                    if dist is not None
                    else output_present_local
                )

                needs_input_local = bool(
                    input_weight is not None
                    and not _weight_is_two_dimensional(input_weight)
                )
                needs_output_local = bool(
                    output_weight is not None
                    and not _weight_is_two_dimensional(output_weight)
                )
                needs_input_any = (
                    _dist_any(dist, needs_input_local)
                    if dist is not None
                    else needs_input_local
                )
                needs_output_any = (
                    _dist_any(dist, needs_output_local)
                    if dist is not None
                    else needs_output_local
                )

                if needs_input_any and input_present_all and input_weight is not None:
                    to_gather.append(input_weight)
                if (
                    needs_output_any
                    and output_present_all
                    and output_weight is not None
                ):
                    to_gather.append(output_weight)

                if (needs_input_any and not input_present_all) or (
                    needs_output_any and not output_present_all
                ):
                    LOG.warning(
                        "DeepSpeed ZeRO gather decision mismatch across ranks; skipping embed gather to avoid deadlock | "
                        "input_present_local=%s input_present_all=%s needs_input_local=%s needs_input_any=%s | "
                        "output_present_local=%s output_present_all=%s needs_output_local=%s needs_output_any=%s",
                        input_present_local,
                        input_present_all,
                        needs_input_local,
                        needs_input_any,
                        output_present_local,
                        output_present_all,
                        needs_output_local,
                        needs_output_any,
                    )
                    to_gather = []

                if to_gather:
                    # De-duplicate parameters to avoid repeated GatheredParameters calls
                    # on shared/tied embedding weights.
                    seen: set[int] = set()
                    unique: list[Any] = []
                    for param in to_gather:
                        param_id = id(param)
                        if param_id in seen:
                            continue
                        seen.add(param_id)
                        unique.append(param)
                    LOG.debug(
                        "DeepSpeed ZeRO gather for scoring | tensors=%d | shapes=%s",
                        len(unique),
                        [getattr(param, "shape", None) for param in unique],
                    )
                    try:
                        gather_ctx = deepspeed.zero.GatheredParameters(
                            unique, modifier_rank=None
                        )
                    except TypeError:
                        gather_ctx = deepspeed.zero.GatheredParameters(unique)
                    zero_gather_strategy = "manual_embed"
        except ImportError:
            gather_ctx = nullcontext()
        except _SCORING_EXCEPTIONS:
            gather_ctx = nullcontext()

    use_helper_zero_gather = zero_gather_strategy == "none"
    helper_gather_all_ranks = bool(zero_gather_all_ranks or gather_full_params)
    fsdp_ctx = _summon_fsdp_full_param_context(model)
    stack = ExitStack()
    if slice_log:
        LOG.info(
            "chunked_sequence_logprobs enter gather_ctx | gather_full_params=%s strategy=%s helper_zero_gather=%s",
            gather_full_params,
            zero_gather_strategy,
            use_helper_zero_gather,
        )
    stack.enter_context(gather_ctx)
    if slice_log:
        LOG.info("chunked_sequence_logprobs entered gather_ctx")
    if use_helper_zero_gather:
        # Enforce a single ZeRO gather strategy per scoring pass. If we already
        # entered an explicit DeepSpeed gather context above, skip helper gathers
        # to avoid nested GatheredParameters over overlapping tensors.
        if slice_log:
            LOG.info("chunked_sequence_logprobs enter zero_gather_params")
        stack.enter_context(
            _maybe_zero_gather_params(
                model, enabled=True, gather_all_ranks=helper_gather_all_ranks
            )
        )
        if slice_log:
            LOG.info("chunked_sequence_logprobs entered zero_gather_params")
    else:
        LOG.debug(
            "Using manual ZeRO gather strategy=%s; skipping helper gather contexts.",
            zero_gather_strategy,
        )
    if slice_log:
        LOG.info("chunked_sequence_logprobs enter fsdp_ctx")
    stack.enter_context(fsdp_ctx)
    if slice_log:
        LOG.info("chunked_sequence_logprobs entered fsdp_ctx")
    config = getattr(model, "config", None)
    padding_idx = _coerce_optional_int(_get_config_value(config, "pad_token_id", None))
    embedding_vocab_size = _get_embedding_vocab_size(model, config)
    vocab_size = _coerce_optional_int(_get_config_value(config, "vocab_size", None))
    pad_targets: list[tuple[Any, str]] = []
    if config is not None:
        pad_targets.append((config, "pad_token_id"))
    seen_modules: set[int] = set()
    embed_token_module = getattr(model, "embed_tokens", None)
    if embed_token_module is not None:
        seen_modules.add(id(embed_token_module))
        if hasattr(embed_token_module, "padding_idx"):
            pad_targets.append((embed_token_module, "padding_idx"))
    try:
        input_embed_module = model.get_input_embeddings()
    except _SCORING_EXCEPTIONS:
        input_embed_module = None
    if (
        input_embed_module is not None
        and id(input_embed_module) not in seen_modules
        and hasattr(input_embed_module, "padding_idx")
    ):
        pad_targets.append((input_embed_module, "padding_idx"))
    final_padding_idx = padding_idx
    if padding_idx is not None:
        limit: Optional[int] = None
        if embedding_vocab_size is not None:
            limit = embedding_vocab_size - 1
        if vocab_size is not None:
            vocab_limit = vocab_size - 1
            if limit is None or vocab_limit < limit:
                limit = vocab_limit
        if limit is not None:
            limit = max(limit, 0)
            if padding_idx > limit:
                final_padding_idx = limit
    pad_ctx = nullcontext()
    if (
        padding_idx is not None
        and final_padding_idx is not None
        and final_padding_idx != padding_idx
        and pad_targets
    ):
        LOG.debug(
            "Clamping padding idx for scoring | original=%s final=%s",
            padding_idx,
            final_padding_idx,
        )
        pad_ctx = _PadTokenGuard(pad_targets, final_padding_idx)
        padding_idx = final_padding_idx
    stack.enter_context(pad_ctx)
    with stack:
        if slice_log:
            LOG.info(
                "chunked_sequence_logprobs start | input_ids_shape=%s attention_mask_shape=%s labels_shape=%s",
                getattr(input_ids, "shape", None),
                getattr(attention_mask, "shape", None),
                getattr(labels, "shape", None),
            )
        LOG.debug(
            "chunked_sequence_logprobs start | gather_full_params=%s return_hidden=%s pooling=%s | "
            "input_ids_shape=%s dtype=%s device=%s | attention_mask_shape=%s | labels_shape=%s dtype=%s device=%s",
            gather_full_params,
            return_hidden,
            pooling,
            getattr(input_ids, "shape", None),
            getattr(input_ids, "dtype", None),
            getattr(input_ids, "device", None),
            getattr(attention_mask, "shape", None),
            getattr(labels, "shape", None),
            getattr(labels, "dtype", None),
            getattr(labels, "device", None),
        )
        # High-level shape logging for reference scoring.
        LOG.debug(
            "reference scoring inputs | input_ids_shape=%s attention_mask_shape=%s labels_shape=%s",
            getattr(input_ids, "shape", None),
            getattr(attention_mask, "shape", None),
            getattr(labels, "shape", None),
        )
        LOG.debug(
            "reference scoring pad metadata | model.config.pad_token_id=%s embedding_vocab_size=%s",
            padding_idx,
            embedding_vocab_size,
        )

        embed_descs: list[str] = []
        embed_tokens = getattr(model, "embed_tokens", None)
        embed_descs.append(_describe_embedding_module(embed_tokens, "embed_tokens"))
        try:
            input_embeddings = model.get_input_embeddings()
        except _SCORING_EXCEPTIONS:
            input_embeddings = None
        if input_embeddings is not None and input_embeddings is not embed_tokens:
            embed_descs.append(
                _describe_embedding_module(input_embeddings, "input_embeddings")
            )
        # Conservative guard: if the reference model's embedding weights are
        # not 2-D under the gathered parameter contexts, skip reference
        # scoring to avoid noisy runtime errors from torch.embedding.
        for module in (embed_tokens, input_embeddings):
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is not None and not _weight_is_two_dimensional(weight):
                if _weight_is_stub_tensor(weight):
                    LOG.debug(
                        "Non-2D stub embedding weight; continuing reference scoring | %s",
                        " | ".join(embed_descs),
                    )
                    continue
                LOG.warning(
                    "Skipping reference scoring due to non-2D embedding weight | %s",
                    " | ".join(embed_descs),
                )
                return None
        LOG.debug(
            "reference scoring embeddings | %s | pad_token_id=%s vocab=%s",
            " | ".join(embed_descs),
            padding_idx,
            embedding_vocab_size,
        )
        batch = getattr(input_ids, "shape", None)
        batch = batch[0] if batch else 0
        chunk_limit = int(chunk_size) if chunk_size is not None else 0
        if (
            chunk_limit <= 0
            and batch > 1
            and not os.environ.get("MAXENT_DISABLE_LOGPROB_AUTOBATCH", "").strip()
        ):
            try:
                vocab_size_guess = int(
                    getattr(getattr(model, "config", None), "vocab_size", 0) or 0
                )
            except (TypeError, ValueError):
                vocab_size_guess = 0
            seq_len = getattr(input_ids, "shape", None)
            seq_len = int(seq_len[1]) if seq_len and len(seq_len) > 1 else 0
            device_str = str(getattr(input_ids, "device", "")).lower()
            if vocab_size_guess > 0 and seq_len > 0 and "cuda" in device_str:
                try:
                    model_dtype = getattr(model, "dtype", None)
                    dtype_str = str(
                        getattr(model_dtype, "name", model_dtype) or ""
                    ).lower()
                except _SCORING_EXCEPTIONS:
                    dtype_str = ""
                bytes_per_elem = 2
                if "float32" in dtype_str or "fp32" in dtype_str:
                    bytes_per_elem = 4
                target_mb_raw = os.environ.get("MAXENT_LOGPROB_TARGET_LOGITS_MB", "256")
                try:
                    target_bytes = int(float(target_mb_raw) * 1024 * 1024)
                except (TypeError, ValueError):
                    target_bytes = 256 * 1024 * 1024
                bytes_per_seq = max(1, seq_len * vocab_size_guess * bytes_per_elem)
                auto_limit = max(1, min(batch, target_bytes // bytes_per_seq))
                if auto_limit < batch:
                    warned = getattr(
                        _chunked_sequence_logprobs, "_autobatch_warned", False
                    )
                    if not warned:
                        LOG.warning(
                            "Auto-tuning reference scoring batch chunk size to avoid large logits tensors | "
                            "requested_chunk_size=%s auto_chunk_size=%s batch=%s seq_len=%s vocab=%s target_logits_mb=%s "
                            "(set MAXENT_DISABLE_LOGPROB_AUTOBATCH=1 to disable)",
                            chunk_size,
                            auto_limit,
                            batch,
                            seq_len,
                            vocab_size_guess,
                            target_mb_raw,
                        )
                        setattr(_chunked_sequence_logprobs, "_autobatch_warned", True)
                    chunk_limit = auto_limit
        if chunk_limit <= 0 or chunk_limit >= batch:
            chunk_indices = [(0, batch)]
        else:
            chunk_indices = [
                (start, min(start + chunk_limit, batch))
                for start in range(0, batch, chunk_limit)
            ]
        LOG.debug(
            "reference scoring chunk plan | total_batch=%s chunk_size=%s chunks=%s",
            batch,
            chunk_limit,
            len(chunk_indices),
        )
    logp_chunks: list[Tensor] = []
    tok_chunks: list[Tensor] = []
    pooled_chunks: list[Tensor] = [] if return_hidden else []
    entropy_chunks: list[Tensor] = [] if return_entropy else []
    token_logp_chunks: list[Tensor] = [] if return_token_logp else []
    token_mask_chunks: list[Tensor] = [] if return_token_logp else []
    for idx, (start, end) in enumerate(chunk_indices):
        LOG.debug(
            "reference scoring chunk begin | chunk=%s | slice=[%s:%s] | rows=%s",
            idx,
            start,
            end,
            end - start,
        )
        if slice_log:
            LOG.info(
                "chunked_sequence_logprobs forward start | chunk=%s slice=[%s:%s]",
                idx,
                start,
                end,
            )
            forward_start = time.monotonic()
        ids_chunk = input_ids[start:end]
        mask_chunk = attention_mask[start:end] if attention_mask is not None else None
        label_chunk = labels[start:end]
        wants_labels = False
        for callable_name in ("forward", "__call__"):
            candidate = getattr(model, callable_name, None)
            if not callable(candidate):
                continue
            try:
                sig = inspect.signature(candidate)
                param = sig.parameters.get("labels")
                if param is not None and param.default is inspect.Signature.empty:
                    wants_labels = True
                    break
            except (TypeError, ValueError):
                continue
        call_kwargs: dict[str, Any] = {
            "input_ids": ids_chunk,
            "attention_mask": mask_chunk,
            "output_hidden_states": return_hidden,
        }
        if wants_labels:
            call_kwargs["labels"] = label_chunk
        call_target = model if callable(model) else getattr(model, "forward", None)
        if not callable(call_target):
            raise TypeError("Model is not callable and lacks a forward method")
        try:
            outputs = call_target(**call_kwargs)
        except TypeError:
            call_kwargs.pop("output_hidden_states", None)
            try:
                outputs = call_target(**call_kwargs)
            except TypeError:
                if not wants_labels:
                    call_kwargs.pop("labels", None)
                outputs = call_target(**call_kwargs)
        if slice_log:
            LOG.info(
                "chunked_sequence_logprobs forward done | chunk=%s seconds=%.2f",
                idx,
                time.monotonic() - forward_start,
            )
        outputs_any = cast(Any, outputs)
        logits = getattr(outputs_any, "logits", None)
        if logits is None:
            raise AttributeError("Model outputs missing logits")
        LOG.debug(
            "reference scoring logits metadata | chunk=%s | shape=%s dtype=%s device=%s",
            idx,
            getattr(logits, "shape", None),
            getattr(logits, "dtype", None),
            getattr(logits, "device", None),
        )
        # Causal LM logits at position t predict token t+1. Align labels by
        # shifting so we score next-token log-probs for all non-masked targets.
        if logits.size(1) <= 1:
            batch_rows = logits.size(0)
            try:
                seq_logp_chunk = torch_mod.zeros(
                    (batch_rows,),
                    dtype=getattr(torch_mod, "float32", None),
                    device=getattr(logits, "device", None),
                )
            except _SCORING_EXCEPTIONS:
                seq_logp_chunk = torch_mod.tensor(
                    np.zeros(batch_rows, dtype=float),
                    dtype=getattr(torch_mod, "float32", None),
                    device=getattr(logits, "device", None),
                )
            tok_tensor_chunk = torch_mod.ones(
                (batch_rows,),
                dtype=getattr(torch_mod, "long", None),
                device=getattr(logits, "device", None),
            )
            logp_chunks.append(seq_logp_chunk)
            tok_chunks.append(tok_tensor_chunk)
            if return_token_logp:
                try:
                    token_logp_chunk = torch_mod.zeros(
                        (batch_rows, 0),
                        dtype=getattr(torch_mod, "float32", None),
                        device=getattr(seq_logp_chunk, "device", None),
                    )
                except _SCORING_EXCEPTIONS:
                    token_logp_chunk = torch_mod.tensor(
                        np.zeros((batch_rows, 0), dtype=float),
                        dtype=getattr(torch_mod, "float32", None),
                        device=getattr(seq_logp_chunk, "device", None),
                    )
                token_mask_chunk = token_logp_chunk
                token_logp_chunks.append(token_logp_chunk)
                token_mask_chunks.append(token_mask_chunk)
            if return_entropy:
                try:
                    entropy_chunk = torch_mod.zeros(
                        (batch_rows,),
                        dtype=getattr(torch_mod, "float32", None),
                        device=getattr(seq_logp_chunk, "device", None),
                    )
                except _SCORING_EXCEPTIONS:
                    entropy_chunk = torch_mod.tensor(
                        np.zeros(batch_rows, dtype=float),
                        dtype=getattr(torch_mod, "float32", None),
                        device=getattr(seq_logp_chunk, "device", None),
                    )
                entropy_chunks.append(entropy_chunk)
            preview_vals = None
            preview_source = seq_logp_chunk
            detach_fn = getattr(seq_logp_chunk, "detach", None)
            if callable(detach_fn):
                try:
                    preview_source = detach_fn()
                except _SCORING_EXCEPTIONS:
                    preview_source = seq_logp_chunk
            preview_source_any = cast(Any, preview_source)
            if hasattr(preview_source_any, "cpu"):
                try:
                    preview_vals = preview_source_any.cpu().reshape(-1)[:3].tolist()
                except _SCORING_EXCEPTIONS:
                    preview_vals = None
            LOG.debug(
                "reference scoring chunk stats | chunk=%s | ids_shape=%s | mask_shape=%s | "
                "seq_logp_shape=%s dtype=%s device=%s | tok_shape=%s dtype=%s device=%s | "
                "tok_sum=%s | valid_token_mask_sum=%s | seq_logp_preview=%s",
                idx,
                getattr(ids_chunk, "shape", None),
                getattr(mask_chunk, "shape", None),
                getattr(seq_logp_chunk, "shape", None),
                getattr(seq_logp_chunk, "dtype", None),
                getattr(seq_logp_chunk, "device", None),
                getattr(tok_tensor_chunk, "shape", None),
                getattr(tok_tensor_chunk, "dtype", None),
                getattr(tok_tensor_chunk, "device", None),
                float(batch_rows),
                0,
                preview_vals,
            )
            continue

        shifted_logits = logits[:, :-1, :]
        shifted_labels = label_chunk[:, 1:]
        label_mask = shifted_labels != -100
        safe_labels = shifted_labels.masked_fill(~label_mask, 0)
        # Memory-lean path for MaxEnt/reference scoring when we only need
        # sequence-level log-prob sums (no per-token outputs/entropy).
        if not return_token_logp and not return_entropy:
            try:
                nonzero_kwargs = {"as_tuple": True}
                valid_rows, valid_cols = label_mask.nonzero(**nonzero_kwargs)
                valid_count = int(getattr(valid_rows, "numel", lambda: 0)())
                batch_rows = shifted_logits.size(0)
                seq_logp_chunk = torch_mod.zeros(
                    (batch_rows,),
                    dtype=getattr(shifted_logits, "dtype", None),
                    device=getattr(shifted_logits, "device", None),
                )
                if valid_count > 0:
                    raw_chunk = os.getenv("MAXENT_LOGPROB_TOKEN_CHUNK", "256")
                    try:
                        token_chunk = max(1, int(raw_chunk))
                    except (TypeError, ValueError):
                        token_chunk = 256
                    target_ids = safe_labels[valid_rows, valid_cols]
                    logsumexp_fn = getattr(torch_mod, "logsumexp", None)
                    for start_pos in range(0, valid_count, token_chunk):
                        end_pos = min(start_pos + token_chunk, valid_count)
                        row_chunk = valid_rows[start_pos:end_pos]
                        col_chunk = valid_cols[start_pos:end_pos]
                        tgt_chunk = target_ids[start_pos:end_pos]
                        selected_logits = shifted_logits[row_chunk, col_chunk, :]
                        target_logits = selected_logits.gather(
                            dim=-1, index=tgt_chunk.unsqueeze(-1)
                        ).squeeze(-1)
                        if callable(logsumexp_fn):
                            log_denom = logsumexp_fn(selected_logits, dim=-1)
                            to_fn = getattr(target_logits, "to", None)
                            if callable(to_fn):
                                target_logits = to_fn(
                                    dtype=getattr(log_denom, "dtype", None)
                                )
                            token_logp_chunk = cast(Any, target_logits) - log_denom
                        else:
                            chunk_log_probs = torch_mod.nn.functional.log_softmax(
                                selected_logits, dim=-1
                            )
                            token_logp_chunk = chunk_log_probs.gather(
                                dim=-1, index=tgt_chunk.unsqueeze(-1)
                            ).squeeze(-1)
                        seq_logp_chunk = cast(
                            Tensor,
                            seq_logp_chunk.index_add(
                                0,
                                row_chunk,
                                cast(
                                    Any,
                                    token_logp_chunk.to(
                                        dtype=getattr(seq_logp_chunk, "dtype", None)
                                    ),
                                ),
                            ),
                        )
                tok_tensor_chunk = label_mask.sum(dim=1).clamp(min=1)
                logp_chunks.append(seq_logp_chunk)
                tok_chunks.append(tok_tensor_chunk)
                continue
            except _SCORING_EXCEPTIONS as exc:
                LOG.debug(
                    "Memory-lean sequence logprob path failed; falling back to dense path: %s",
                    exc,
                )
        gather_labels = safe_labels.unsqueeze(-1)
        log_probs = None
        token_logp = None
        if use_exact_entropy:
            try:
                log_probs = torch_mod.nn.functional.log_softmax(shifted_logits, dim=-1)
                token_logp = log_probs.gather(dim=-1, index=gather_labels).squeeze(-1)
            except _SCORING_EXCEPTIONS:
                log_probs = None
                token_logp = None
        if token_logp is None:
            try:
                logsumexp_fn = getattr(torch_mod, "logsumexp", None)
                if not callable(logsumexp_fn):
                    raise AttributeError("torch.logsumexp unavailable")
                log_denom = logsumexp_fn(shifted_logits, dim=-1)
                target_logits = shifted_logits.gather(
                    dim=-1, index=gather_labels
                ).squeeze(-1)
                to_fn = getattr(target_logits, "to", None)
                if callable(to_fn):
                    target_logits = to_fn(dtype=getattr(log_denom, "dtype", None))
                token_logp = cast(Any, target_logits) - log_denom
            except _SCORING_EXCEPTIONS:
                log_probs = torch_mod.nn.functional.log_softmax(shifted_logits, dim=-1)
                token_logp = log_probs.gather(dim=-1, index=gather_labels).squeeze(-1)
        # If mask tensors are real torch tensors but token_logp is a stub tensor,
        # coerce token_logp into the active torch module to avoid type mismatches.
        torch_tensor = getattr(torch_mod, "Tensor", None)
        if (
            torch_tensor is not None
            and isinstance(label_mask, torch_tensor)
            and not isinstance(token_logp, torch_tensor)
        ):
            try:
                token_logp = torch_mod.tensor(_to_numpy_array(token_logp))
            except _SCORING_EXCEPTIONS as exc:
                LOG.debug("Failed to coerce token_logp into torch tensor: %s", exc)
        mask_float = label_mask
        type_as_fn = getattr(mask_float, "type_as", None)
        if callable(type_as_fn):
            try:
                is_tensor_fn = getattr(torch_mod, "is_tensor", None)
                if callable(is_tensor_fn) and not is_tensor_fn(token_logp):
                    raise TypeError("type_as requires a torch tensor")
                mask_mod = getattr(type(mask_float), "__module__", "")
                token_mod = getattr(type(token_logp), "__module__", "")
                if mask_mod.startswith("torch") != token_mod.startswith("torch"):
                    raise TypeError("type_as requires matching torch tensors")
                if not isinstance(token_logp, type(mask_float)):
                    raise TypeError("type_as requires same tensor types")
                mask_float = type_as_fn(token_logp)
            except _SCORING_EXCEPTIONS:
                type_as_fn = None
        if not callable(type_as_fn):
            to_fn = getattr(mask_float, "to", None)
            if callable(to_fn):
                try:
                    mask_float = to_fn(dtype=getattr(token_logp, "dtype", None))
                except _SCORING_EXCEPTIONS:
                    float_fn = getattr(mask_float, "float", None)
                    if callable(float_fn):
                        mask_float = float_fn()
        seq_logp_chunk = cast(
            Tensor,
            (cast(Any, token_logp) * cast(Any, mask_float)).sum(dim=1),
        )
        tok_tensor_chunk = label_mask.sum(dim=1).clamp(min=1)
        logp_chunks.append(seq_logp_chunk)
        tok_chunks.append(tok_tensor_chunk)
        if return_token_logp:
            try:
                comp_logp, comp_mask = _compress_completion_token_logp(
                    cast(Tensor, token_logp), cast(Tensor, label_mask)
                )
            except _SCORING_EXCEPTIONS:
                comp_logp, comp_mask = token_logp, label_mask
            try:
                token_logp_chunks.append(cast(Tensor, comp_logp))
            except _SCORING_EXCEPTIONS:
                token_logp_chunks.append(torch_mod.tensor(_to_numpy_array(comp_logp)))
            token_mask_chunks.append(cast(Tensor, comp_mask))
        if return_entropy:
            entropy_chunk = None
            if use_sample_entropy:
                try:
                    entropy_chunk = (-seq_logp_chunk).to(
                        dtype=getattr(torch_mod, "float32", None)
                        or getattr(seq_logp_chunk, "dtype", None)
                    )
                except _SCORING_EXCEPTIONS:
                    entropy_chunk = None
            if entropy_chunk is None:
                try:
                    if log_probs is None:
                        log_probs = torch_mod.nn.functional.log_softmax(
                            shifted_logits, dim=-1
                        )
                    ent = -(log_probs.exp() * log_probs).sum(dim=-1)
                    entropy_chunk = (ent * cast(Any, mask_float)).sum(dim=1)
                except _SCORING_EXCEPTIONS as exc:
                    LOG.debug("Failed to compute policy entropy: %s", exc)
            if entropy_chunk is None:
                try:
                    entropy_chunk = torch_mod.zeros(
                        (seq_logp_chunk.shape[0],),
                        dtype=getattr(torch_mod, "float32", None),
                        device=getattr(seq_logp_chunk, "device", None),
                    )
                except _SCORING_EXCEPTIONS:
                    entropy_chunk = torch_mod.tensor(
                        np.zeros(seq_logp_chunk.shape[0], dtype=float),
                        dtype=getattr(torch_mod, "float32", None),
                        device=getattr(seq_logp_chunk, "device", None),
                    )
            entropy_chunks.append(entropy_chunk)
        try:
            tok_sum = tok_tensor_chunk.detach().cpu().sum().item()
        except _SCORING_EXCEPTIONS:
            tok_sum = None
        try:
            logp_preview = seq_logp_chunk.detach().cpu().reshape(-1)[:3].tolist()
        except _SCORING_EXCEPTIONS:
            logp_preview = None
        try:
            valid_tokens = int(label_mask.sum().detach().cpu().item())
        except _SCORING_EXCEPTIONS:
            valid_tokens = None
        LOG.debug(
            "reference scoring chunk stats | chunk=%s | ids_shape=%s | mask_shape=%s | "
            "seq_logp_shape=%s dtype=%s device=%s | tok_shape=%s dtype=%s device=%s | "
            "tok_sum=%s | valid_token_mask_sum=%s | seq_logp_preview=%s",
            idx,
            getattr(ids_chunk, "shape", None),
            getattr(mask_chunk, "shape", None),
            getattr(seq_logp_chunk, "shape", None),
            getattr(seq_logp_chunk, "dtype", None),
            getattr(seq_logp_chunk, "device", None),
            getattr(tok_tensor_chunk, "shape", None),
            getattr(tok_tensor_chunk, "dtype", None),
            getattr(tok_tensor_chunk, "device", None),
            tok_sum,
            valid_tokens,
            logp_preview,
        )
        hidden_states = getattr(outputs_any, "hidden_states", None)
        if return_hidden and hidden_states is not None:
            hidden = hidden_states[-1]
            mask = mask_chunk
            if pooling == "last":
                pooled = hidden[:, -1, :]
            else:
                if mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    mask = mask.unsqueeze(-1)
                    type_as_fn = getattr(mask, "type_as", None)
                    if callable(type_as_fn):
                        try:
                            is_tensor_fn = getattr(torch_mod, "is_tensor", None)
                            if callable(is_tensor_fn) and not is_tensor_fn(hidden):
                                raise TypeError("type_as requires a torch tensor")
                            mask_mod = getattr(type(mask), "__module__", "")
                            hidden_mod = getattr(type(hidden), "__module__", "")
                            if mask_mod.startswith("torch") != hidden_mod.startswith(
                                "torch"
                            ):
                                raise TypeError(
                                    "type_as requires matching torch tensors"
                                )
                            if not isinstance(hidden, type(mask)):
                                raise TypeError("type_as requires same tensor types")
                            mask = type_as_fn(hidden)
                        except _SCORING_EXCEPTIONS:
                            type_as_fn = None
                    if not callable(type_as_fn):
                        to_fn = getattr(mask, "to", None)
                        if callable(to_fn):
                            try:
                                mask = to_fn(dtype=hidden.dtype)
                            except _SCORING_EXCEPTIONS:
                                float_fn = getattr(mask, "float", None)
                                if callable(float_fn):
                                    mask = float_fn()
                        else:
                            float_fn = getattr(mask, "float", None)
                            if callable(float_fn):
                                mask = float_fn()
                    mask_any = cast(Any, mask)
                    pooled = (hidden * mask_any).sum(dim=1) / mask_any.sum(dim=1).clamp(
                        min=1.0
                    )
            pooled_chunks.append(pooled)
    seq_logp = (
        logp_chunks[0] if len(logp_chunks) == 1 else torch_mod.cat(logp_chunks, dim=0)
    )
    tok_tensor = (
        tok_chunks[0] if len(tok_chunks) == 1 else torch_mod.cat(tok_chunks, dim=0)
    )
    pooled_hidden: Optional[Tensor] = None
    if pooled_chunks:
        pooled_hidden = (
            pooled_chunks[0]
            if len(pooled_chunks) == 1
            else torch_mod.cat(pooled_chunks, dim=0)
        )
    entropy_sum: Optional[Tensor] = None
    if return_entropy:
        if entropy_chunks:
            entropy_sum = (
                entropy_chunks[0]
                if len(entropy_chunks) == 1
                else torch_mod.cat(entropy_chunks, dim=0)
            )
        else:
            entropy_sum = torch_mod.zeros_like(tok_tensor)
    try:
        logp_sample = seq_logp.detach().cpu().reshape(-1)[:4].tolist()
    except _SCORING_EXCEPTIONS:
        logp_sample = None
    LOG.debug(
        "chunked_sequence_logprobs finish | seq_logp_shape=%s tok_shape=%s pooled_shape=%s | "
        "seq_logp_dtype=%s tok_dtype=%s pooled_dtype=%s | seq_logp_device=%s tok_device=%s pooled_device=%s | "
        "seq_logp_numel=%s tok_numel=%s | seq_logp_preview=%s",
        getattr(seq_logp, "shape", None),
        getattr(tok_tensor, "shape", None),
        getattr(pooled_hidden, "shape", None) if pooled_hidden is not None else None,
        getattr(seq_logp, "dtype", None),
        getattr(tok_tensor, "dtype", None),
        getattr(pooled_hidden, "dtype", None) if pooled_hidden is not None else None,
        getattr(seq_logp, "device", None),
        getattr(tok_tensor, "device", None),
        getattr(pooled_hidden, "device", None) if pooled_hidden is not None else None,
        getattr(seq_logp, "numel", lambda: None)(),
        getattr(tok_tensor, "numel", lambda: None)(),
        logp_sample,
    )
    if return_token_logp:
        token_logp: Optional[Tensor] = None
        token_mask: Optional[Tensor] = None
        if token_logp_chunks:
            token_logp = (
                token_logp_chunks[0]
                if len(token_logp_chunks) == 1
                else torch_mod.cat(token_logp_chunks, dim=0)
            )
            token_mask = (
                token_mask_chunks[0]
                if len(token_mask_chunks) == 1
                else torch_mod.cat(token_mask_chunks, dim=0)
            )
        return seq_logp, tok_tensor, pooled_hidden, entropy_sum, token_logp, token_mask
    return seq_logp, tok_tensor, pooled_hidden, entropy_sum


def selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor:
    """Memory-efficient log_softmax + gather (TRL-style)."""
    torch_mod = _refresh_torch()
    float32 = getattr(torch_mod, "float32", None)
    float64 = getattr(torch_mod, "float64", None)
    dtype = getattr(logits, "dtype", None)
    if dtype in {float32, float64}:
        try:
            selected_logits = torch_mod.gather(
                logits, dim=-1, index=index.unsqueeze(-1)
            ).squeeze(-1)
            logsumexp_values = torch_mod.stack(
                [torch_mod.logsumexp(lg, dim=-1) for lg in logits]
            )
            return cast(Tensor, selected_logits - logsumexp_values)
        except _SCORING_EXCEPTIONS:
            pass
    per_token_logps: List[Tensor] = []
    log_softmax_fn = getattr(getattr(torch_mod, "nn", None), "functional", None)
    log_softmax = (
        getattr(log_softmax_fn, "log_softmax", None) if log_softmax_fn else None
    )
    for row_logits, row_labels in zip(logits, index):
        if callable(log_softmax):
            row_logps = log_softmax(row_logits, dim=-1)
        else:
            logsumexp_fn = getattr(torch_mod, "logsumexp", None)
            if callable(logsumexp_fn):
                row_logps = row_logits - logsumexp_fn(row_logits, dim=-1, keepdim=True)
            else:  # pragma: no cover - best effort fallback
                row_logps = row_logits
        row_per_token_logps = row_logps.gather(
            dim=-1, index=row_labels.unsqueeze(-1)
        ).squeeze(-1)
        per_token_logps.append(cast(Tensor, row_per_token_logps))
    return cast(Tensor, torch_mod.stack(per_token_logps))


def _trl_get_per_token_logps(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    logits_to_keep: int,
    *,
    temperature: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> Tensor:
    """TRL-style per-token log-probabilities for completion tokens."""
    torch_mod = _refresh_torch()
    if logits_to_keep <= 0:
        return cast(
            Tensor,
            torch_mod.zeros(
                (int(getattr(input_ids, "shape", [0])[0] or 0), 0),
                dtype=getattr(torch_mod, "float32", None),
                device=getattr(input_ids, "device", None),
            ),
        )
    temp = float(temperature if temperature is not None else 1.0)
    step = int(batch_size or 0)
    if step <= 0:
        step = int(getattr(input_ids, "shape", [0])[0] or 1)
    all_logps: List[Tensor] = []
    for i in range(0, int(getattr(input_ids, "shape", [0])[0] or 0), step):
        input_ids_batch = input_ids[i : i + step]
        attention_mask_batch = attention_mask[i : i + step]
        logits = None
        try:
            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            )
            logits = getattr(outputs, "logits", outputs)
        except TypeError:
            outputs = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch
            )
            logits = getattr(outputs, "logits", outputs)
        if logits is None:
            raise ValueError("Model forward returned no logits for TRL scoring.")
        logits = logits[:, :-1, :]
        input_ids_batch = input_ids_batch[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        if temp != 1.0:
            logits = logits / temp
        logps = selective_log_softmax(logits, input_ids_batch)
        all_logps.append(cast(Tensor, logps))
    return cast(Tensor, torch_mod.cat(all_logps, dim=0))


def score_model_outputs(
    model: PreTrainedModel,
    score_batch: ScoreBatch,
    batching_cfg: BatchingSettings,
    runtime: RuntimeHandles,
    *,
    return_hidden: bool = False,
    pooling: str = "mean",
    return_entropy: bool = False,
    entropy_mode: str = "exact",
    return_token_logp: bool = False,
) -> Optional[
    Tuple[
        Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]
    ]
]:
    """Compute current model log-probs for the batch and optional pooled states.

    :param model: Current policy model used for scoring.
    :param score_batch: Prepared scoring batch.
    :param batching_cfg: Batching config controlling logprob chunking.
    :param runtime: Runtime handles providing device and accelerator state.
    :param return_hidden: When ``True``, also return pooled hidden states.
    :param pooling: Pooling strategy applied to hidden states.
    :returns: Tuple of ``(cur_logp_sum, pooled_hidden[, policy_entropy_sum][, token_logp, token_mask])``
        or ``None`` if empty.
    :rtype: tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor | None] | tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None] | None
    """
    cur_logp_slices: List[Tensor] = []
    pooled_slices: List[Tensor] = []
    entropy_slices: List[Tensor] = []
    token_logp_slices: List[Tensor] = []
    token_mask_slices: List[Tensor] = []
    slice_log = _score_slice_log_enabled()
    progress_log = _progress_log_enabled()
    eos_token_id = getattr(getattr(runtime, "tokenizer", None), "eos_token_id", None)
    slice_iter = iter_batch_slices(
        score_batch,
        runtime.device,
        eos_token_id=eos_token_id,
        apply_eos_mask=True,
    )
    slice_iter = _prefetch_iterator(
        slice_iter, getattr(batching_cfg, "slice_prefetch", 0)
    )
    detach_token_logp = not torch.is_grad_enabled()
    if progress_log:
        LOG.info(
            "score_model_outputs start | total_sequences=%s slice_size=%s device=%s logprob_chunk_size=%s",
            score_batch.total_sequences,
            score_batch.slice_size,
            getattr(runtime.device, "type", runtime.device),
            getattr(batching_cfg, "logprob_chunk_size", None),
        )
    LOG.debug(
        "current scoring batch metadata | total_sequences=%s slice_size=%s device=%s",
        score_batch.total_sequences,
        score_batch.slice_size,
        getattr(runtime.device, "type", runtime.device),
    )
    with _autocast_context(runtime.accelerator, runtime.device):
        slice_idx = 0
        for slice_inputs, slice_mask, slice_labels in slice_iter:
            if slice_log:
                LOG.info(
                    "score_model_outputs slice start | idx=%d input_ids_shape=%s attention_mask_shape=%s labels_shape=%s",
                    slice_idx,
                    getattr(slice_inputs, "shape", None),
                    getattr(slice_mask, "shape", None),
                    getattr(slice_labels, "shape", None),
                )
                slice_start = time.monotonic()
            LOG.debug(
                "current scoring slice inputs | input_ids_shape=%s attention_mask_shape=%s labels_shape=%s",
                getattr(slice_inputs, "shape", None),
                getattr(slice_mask, "shape", None),
                getattr(slice_labels, "shape", None),
            )
            result = _chunked_sequence_logprobs(
                model,
                input_ids=slice_inputs,
                attention_mask=slice_mask,
                labels=slice_labels,
                chunk_size=batching_cfg.logprob_chunk_size,
                return_hidden=return_hidden,
                pooling=pooling,
                return_entropy=return_entropy,
                entropy_mode=entropy_mode,
                return_token_logp=return_token_logp,
            )
            if slice_log:
                if result is None:
                    LOG.info(
                        "score_model_outputs slice done | idx=%d seconds=%.2f result=None",
                        slice_idx,
                        time.monotonic() - slice_start,
                    )
                else:
                    seq_result = cast(Sequence[Any], result)
                    logp_slice = seq_result[0] if len(seq_result) >= 1 else None
                    tok_counts = seq_result[1] if len(seq_result) >= 2 else None
                    pooled = seq_result[2] if len(seq_result) >= 3 else None
                    entropy_sum = seq_result[3] if len(seq_result) >= 4 else None
                    LOG.info(
                        "score_model_outputs slice done | idx=%d seconds=%.2f logp_shape=%s tok_shape=%s pooled=%s entropy=%s",
                        slice_idx,
                        time.monotonic() - slice_start,
                        getattr(logp_slice, "shape", None),
                        getattr(tok_counts, "shape", None),
                        getattr(pooled, "shape", None),
                        getattr(entropy_sum, "shape", None),
                    )
            if result is None:
                return None
            cur_logp_slice, _tok_counts, pooled, entropy_sum = result[:4]
            token_logp = None
            token_mask = None
            if return_token_logp and len(result) >= 6:
                token_logp = result[4]
                token_mask = result[5]
            cur_logp_slices.append(cur_logp_slice)
            if pooled is not None:
                pooled_slices.append(pooled.detach())
            if entropy_sum is not None:
                entropy_slices.append(entropy_sum.detach())
            if return_token_logp:
                if token_logp is not None:
                    token_logp_slices.append(
                        token_logp.detach() if detach_token_logp else token_logp
                    )
                if token_mask is not None:
                    token_mask_slices.append(token_mask.detach())
            slice_idx += 1
    if not cur_logp_slices:
        return None
    pooled_hidden = torch.cat(pooled_slices, dim=0) if pooled_slices else None
    token_logp = None
    token_mask = None
    if return_token_logp and token_logp_slices:
        max_len = max(getattr(t, "shape", [0, 0])[1] for t in token_logp_slices)
        if max_len < 0:
            max_len = 0
        padded_logps: List[Tensor] = []
        padded_masks: List[Tensor] = []
        for logp_slice, mask_slice in zip(token_logp_slices, token_mask_slices):
            cur_len = getattr(logp_slice, "shape", [0, 0])[1]
            if cur_len == max_len:
                padded_logps.append(logp_slice)
                padded_masks.append(mask_slice)
                continue
            pad_len = max_len - cur_len
            if pad_len <= 0:
                padded_logps.append(logp_slice[:, :max_len])
                padded_masks.append(mask_slice[:, :max_len])
                continue
            pad_device = getattr(logp_slice, "device", None)
            pad_dtype = getattr(logp_slice, "dtype", None)
            try:
                pad_logp = torch.zeros(
                    (logp_slice.shape[0], pad_len),
                    device=pad_device,
                    dtype=pad_dtype,
                )
            except TypeError:
                pad_logp = torch.zeros((logp_slice.shape[0], pad_len))
            try:
                pad_mask = torch.zeros(
                    (mask_slice.shape[0], pad_len),
                    device=getattr(mask_slice, "device", None),
                    dtype=getattr(mask_slice, "dtype", None),
                )
            except TypeError:
                pad_mask = torch.zeros((mask_slice.shape[0], pad_len))
            padded_logps.append(torch.cat([logp_slice, pad_logp], dim=1))
            padded_masks.append(torch.cat([mask_slice, pad_mask], dim=1))
        token_logp = torch.cat(padded_logps, dim=0) if padded_logps else None
        token_mask = torch.cat(padded_masks, dim=0) if padded_masks else None

    if not return_entropy:
        output = (
            torch.cat(cur_logp_slices, dim=0),
            pooled_hidden,
            token_logp,
            token_mask,
        )
        if progress_log:
            LOG.info(
                "score_model_outputs done | slices=%d logp_shape=%s pooled=%s token_logp=%s",
                len(cur_logp_slices),
                getattr(output[0], "shape", None),
                getattr(pooled_hidden, "shape", None),
                getattr(token_logp, "shape", None),
            )
        if return_token_logp:
            return output
        return output[:2]
    entropy_sum = torch.cat(entropy_slices, dim=0) if entropy_slices else None
    output = (
        torch.cat(cur_logp_slices, dim=0),
        pooled_hidden,
        entropy_sum,
        token_logp,
        token_mask,
    )
    if progress_log:
        LOG.info(
            "score_model_outputs done | slices=%d logp_shape=%s pooled=%s entropy=%s token_logp=%s",
            len(cur_logp_slices),
            getattr(output[0], "shape", None),
            getattr(pooled_hidden, "shape", None),
            getattr(entropy_sum, "shape", None),
            getattr(token_logp, "shape", None),
        )
    if return_token_logp:
        return output
    return output[:3]


def _as_torch_tensor(
    torch_mod: _TorchModuleLike,
    value: object,
    *,
    device: Optional[TorchDevice],
    dtype: Optional[object],
) -> Tensor:
    """Best-effort conversion of ``value`` into a torch tensor on ``device``."""

    ctor = getattr(torch_mod, "as_tensor", getattr(torch_mod, "tensor", None))
    if ctor is None:
        raise RuntimeError("Torch tensor constructor unavailable")
    if isinstance(value, torch_mod.Tensor):
        tensor = value
    else:
        payload = getattr(value, "arr", None)
        if payload is None:
            payload = getattr(value, "data", value)
        try:
            tensor = ctor(payload)
        except _SCORING_EXCEPTIONS:
            tensor = ctor([])
    if dtype is not None:
        to_fn = getattr(tensor, "to", None)
        if callable(to_fn):
            try:
                tensor = to_fn(dtype=dtype)
            except _SCORING_EXCEPTIONS:
                clone_fn = getattr(tensor, "clone", None)
                if callable(clone_fn):
                    tensor = clone_fn()
                    to_fn = getattr(tensor, "to", None)
                    if callable(to_fn):
                        tensor = to_fn(dtype=dtype)
    if device is not None and getattr(tensor, "device", None) != device:
        to_fn = getattr(tensor, "to", None)
        if callable(to_fn):
            tensor = to_fn(device=device)
    return cast(Tensor, tensor)


def _match_tensor_length(
    torch_mod: _TorchModuleLike,
    tensor: Tensor,
    target_len: int,
    *,
    device: Optional[TorchDevice],
    dtype: Optional[object],
    fill_value: float = 0.0,
) -> Tensor:
    """Return ``tensor`` reshaped/padded to ``target_len`` elements."""

    def _full(shape: Tuple[int, ...], value: float) -> Tensor:
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch_mod.full(shape, value, **kwargs)

    def _zeros(shape: Tuple[int, ...]) -> Tensor:
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        try:
            return torch_mod.zeros(shape, **kwargs)
        except TypeError:
            return torch_mod.zeros(shape)

    tensor = tensor.view(-1)
    cur_len = int(getattr(tensor, "numel", lambda: 0)())
    if target_len <= 0:
        return _zeros((0,))
    if cur_len == target_len:
        return tensor
    if cur_len == 0:
        return _full((target_len,), fill_value)
    if cur_len == 1:
        scalar_val = float(getattr(tensor[0], "item", lambda: tensor[0])())
        return _full((target_len,), scalar_val)
    min_len = min(cur_len, target_len)
    tensor = tensor[:min_len]
    if min_len == target_len:
        return tensor
    pad_val = float(getattr(tensor[-1], "item", lambda: tensor[-1])())
    pad = _full((target_len - min_len,), pad_val)
    return torch_mod.cat([tensor, pad], dim=0)


def build_sequence_scores(
    cur_logp_sum: Tensor,
    ref_stats: ReferenceLogprobs,
    pooled_hidden: Optional[Tensor] = None,
    *,
    behavior_logp_sum: Optional[Tensor] = None,
    policy_entropy_sum: Optional[Tensor] = None,
    token_logp: Optional[Tensor] = None,
    token_mask: Optional[Tensor] = None,
    old_token_logp: Optional[Tensor] = None,
) -> SequenceScores:
    """Return ``SequenceScores`` built from current and reference log-probs.

    :param cur_logp_sum: Current policy log-prob sums per sequence.
    :param ref_stats: Reference log-prob stats used for KL and weighting.
    :param pooled_hidden: Optional pooled hidden states for auxiliary losses.
    :param behavior_logp_sum: Optional behavior-policy log-probs for off-policy scoring.
    :returns: ``SequenceScores`` dataclass with normalized log-probs and KL terms.
    :rtype: SequenceScores
    """

    torch_mod = _refresh_torch()
    base_dtype = getattr(torch_mod, "float32", None)
    cur_tensor = _as_torch_tensor(
        torch_mod,
        cur_logp_sum,
        device=getattr(cur_logp_sum, "device", None),
        dtype=base_dtype,
    ).view(-1)
    device = getattr(cur_tensor, "device", None)
    cur_len = int(getattr(cur_tensor, "numel", lambda: 0)())
    ref_source = getattr(ref_stats, "ref_logp_sum_raw", None)
    if ref_source is None:
        ref_source = getattr(ref_stats, "ref_logp_sum", None)
    ref_tensor = _as_torch_tensor(
        torch_mod,
        ref_source if ref_source is not None else [],
        device=device,
        dtype=base_dtype,
    )
    ref_tensor = _match_tensor_length(
        torch_mod,
        ref_tensor,
        cur_len,
        device=device,
        dtype=base_dtype,
        fill_value=0.0,
    )
    denom_source = getattr(ref_stats, "ref_tok_counts", None)
    denom_tensor = _as_torch_tensor(
        torch_mod,
        denom_source if denom_source is not None else [],
        device=device,
        dtype=base_dtype,
    )
    denom_tensor = _match_tensor_length(
        torch_mod,
        denom_tensor,
        cur_len,
        device=device,
        dtype=base_dtype,
        fill_value=1.0,
    ).clamp(min=1.0)
    if behavior_logp_sum is None:
        behavior_tensor = cur_tensor.detach()
    else:
        behavior_tensor = _as_torch_tensor(
            torch_mod,
            behavior_logp_sum,
            device=device,
            dtype=base_dtype,
        )
        behavior_tensor = _match_tensor_length(
            torch_mod,
            behavior_tensor,
            cur_len,
            device=device,
            dtype=base_dtype,
            fill_value=0.0,
        )
    policy_entropy_tensor: Optional[Tensor] = None
    if policy_entropy_sum is not None:
        policy_entropy_tensor = _as_torch_tensor(
            torch_mod,
            policy_entropy_sum,
            device=device,
            dtype=base_dtype,
        )
        policy_entropy_tensor = _match_tensor_length(
            torch_mod,
            policy_entropy_tensor,
            cur_len,
            device=device,
            dtype=base_dtype,
            fill_value=0.0,
        )
    token_logp_tensor: Optional[Tensor] = None
    token_mask_tensor: Optional[Tensor] = None
    old_token_logp_tensor: Optional[Tensor] = None
    if token_logp is not None:
        token_logp_tensor = _as_torch_tensor(
            torch_mod,
            token_logp,
            device=device,
            dtype=base_dtype,
        )
    if token_mask is not None:
        token_mask_tensor = _as_torch_tensor(
            torch_mod,
            token_mask,
            device=device,
            dtype=base_dtype,
        )
    if old_token_logp is not None:
        old_token_logp_tensor = _as_torch_tensor(
            torch_mod,
            old_token_logp,
            device=device,
            dtype=base_dtype,
        )
    if token_logp_tensor is not None and old_token_logp_tensor is None:
        try:
            old_token_logp_tensor = token_logp_tensor.detach()
        except _SCORING_EXCEPTIONS:
            old_token_logp_tensor = _as_torch_tensor(
                torch_mod,
                token_logp_tensor,
                device=device,
                dtype=base_dtype,
            )
    log_ratio_train = cur_tensor - ref_tensor
    if getattr(log_ratio_train, "numel", lambda: 0)() == 0 and cur_len > 0:
        log_ratio_train = torch_mod.zeros((cur_len,), device=device, dtype=base_dtype)
    return SequenceScores(
        cur_logp_sum=cur_tensor,
        behavior_logp_sum=behavior_tensor,
        log_ratio_train=log_ratio_train,
        denom_tok_tensor=denom_tensor,
        pooled_hidden=pooled_hidden,
        policy_entropy_sum=policy_entropy_tensor,
        token_logp=token_logp_tensor,
        token_mask=token_mask_tensor,
        old_token_logp=old_token_logp_tensor,
    )
