# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reference-logprob and vLLM metadata scoring helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
from contextlib import nullcontext
import numbers
import time
from typing import Any, List, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import ArrayLike

from .scoring_batching import iter_batch_slices, iter_batch_slices_trl
from .scoring_logprob import (
    _as_torch_tensor,
    _chunked_sequence_logprobs,
    _trl_get_per_token_logps,
)
from .scoring_common import (
    LOG,
    TorchDevice,
    _SCORING_EXCEPTIONS,
    _dist_all,
    _dist_collective_ready,
    _model_has_non2d_embeddings,
    _prefetch_iterator,
    _progress_log_enabled,
    _refresh_torch,
    _score_slice_log_enabled,
    _to_numpy_array,
)
from .types import (
    BatchingSettings,
    ReferenceLogprobs,
    RuntimeHandles,
    ScoreBatch,
    Tensor,
)

torch = _refresh_torch()


def reference_from_model_trl(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    batching_cfg: BatchingSettings,
    *,
    temperature: Optional[float] = None,
) -> Optional[Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]]:
    """Run the frozen reference model using TRL-style log-prob computations."""
    torch_mod = _refresh_torch()
    ref_model = runtime.get_ref_model()
    ref_logp_chunks: List[Tensor] = []
    ref_tok_chunks: List[Tensor] = []
    ref_token_logp_chunks: List[Tensor] = []
    ref_token_mask_chunks: List[Tensor] = []
    slice_log = _score_slice_log_enabled()
    progress_log = _progress_log_enabled()
    eos_token_id = getattr(getattr(runtime, "tokenizer", None), "eos_token_id", None)
    if progress_log:
        LOG.info(
            "reference_from_model_trl start | device=%s slice_prefetch=%s logprob_chunk_size=%s",
            getattr(runtime, "device", None),
            getattr(batching_cfg, "slice_prefetch", None),
            getattr(batching_cfg, "logprob_chunk_size", None),
        )

    def _log_once(reason: str) -> None:
        if getattr(reference_from_model_trl, "_diag_logged", False):
            return
        LOG.warning("reference_from_model_trl returning None: %s", reason)
        setattr(reference_from_model_trl, "_diag_logged", True)

    slices_seen = 0
    slice_iter = iter_batch_slices_trl(score_batch, runtime, eos_token_id)
    slice_iter = _prefetch_iterator(
        slice_iter, getattr(batching_cfg, "slice_prefetch", 0)
    )
    for slice_inputs, slice_mask, completion_mask, logits_to_keep in slice_iter:
        slices_seen += 1
        if slice_log:
            LOG.info(
                "reference_from_model_trl slice start | idx=%d inputs_shape=%s mask_shape=%s comp_mask_shape=%s",
                slices_seen - 1,
                getattr(slice_inputs, "shape", None),
                getattr(slice_mask, "shape", None),
                getattr(completion_mask, "shape", None),
            )
            slice_start = time.monotonic()
        no_grad_ctx = getattr(torch_mod, "no_grad", None) or nullcontext
        with no_grad_ctx():
            try:
                per_token_logps = _trl_get_per_token_logps(
                    ref_model,
                    slice_inputs,
                    slice_mask,
                    logits_to_keep,
                    temperature=temperature,
                    batch_size=getattr(batching_cfg, "logprob_chunk_size", None),
                )
            except _SCORING_EXCEPTIONS as exc:
                _log_once(
                    f"_trl_get_per_token_logps raised {type(exc).__name__}: {exc}"
                )
                return None
        if per_token_logps is None:
            _log_once("per_token_logps is None")
            return None
        if completion_mask is None:
            _log_once("completion_mask missing for TRL reference scoring")
            return None
        mask_float = completion_mask
        to_fn = getattr(mask_float, "to", None)
        if callable(to_fn):
            try:
                mask_float = to_fn(dtype=getattr(per_token_logps, "dtype", None))
            except _SCORING_EXCEPTIONS:
                pass
        try:
            seq_logp = (per_token_logps * mask_float).sum(dim=1)
            tok_counts = completion_mask.sum(dim=1).clamp(min=1)
        except _SCORING_EXCEPTIONS as exc:
            _log_once(f"Failed to reduce per-token logps: {exc}")
            return None
        if slice_log:
            LOG.info(
                "reference_from_model_trl slice done | idx=%d seconds=%.2f logp_shape=%s tok_shape=%s",
                slices_seen - 1,
                time.monotonic() - slice_start,
                getattr(seq_logp, "shape", None),
                getattr(tok_counts, "shape", None),
            )
        if (
            getattr(seq_logp, "numel", lambda: 0)() == 0
            or getattr(tok_counts, "numel", lambda: 0)() == 0
        ):
            _log_once(
                f"empty slice tensors | logp_numel={getattr(seq_logp, 'numel', lambda: 0)()} "
                f"tok_numel={getattr(tok_counts, 'numel', lambda: 0)()}"
            )
            return None
        ref_logp_chunks.append(seq_logp.detach().cpu())
        ref_tok_chunks.append(
            tok_counts.to(dtype=getattr(torch_mod, "float32", None)).detach().cpu()
        )
        ref_token_logp_chunks.append(per_token_logps.detach())
        ref_token_mask_chunks.append(completion_mask.detach())
    if slices_seen == 0:
        _log_once("iter_batch_slices_trl yielded zero slices")
        return None
    if not ref_logp_chunks:
        _log_once("no reference slices produced any chunks")
        return None
    ref_logp = torch_mod.cat(ref_logp_chunks, dim=0).to(runtime.device)
    ref_tok = torch_mod.cat(ref_tok_chunks, dim=0).to(runtime.device)
    ref_token_logp = None
    ref_token_mask = None
    if ref_token_logp_chunks:
        max_len = max(getattr(t, "shape", [0, 0])[1] for t in ref_token_logp_chunks)
        if max_len < 0:
            max_len = 0
        padded_logps: List[Tensor] = []
        padded_masks: List[Tensor] = []
        for logp_slice, mask_slice in zip(ref_token_logp_chunks, ref_token_mask_chunks):
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
                pad_logp = torch_mod.zeros(
                    (logp_slice.shape[0], pad_len),
                    device=pad_device,
                    dtype=pad_dtype,
                )
            except TypeError:
                pad_logp = torch_mod.zeros((logp_slice.shape[0], pad_len))
            try:
                pad_mask = torch_mod.zeros(
                    (mask_slice.shape[0], pad_len),
                    device=getattr(mask_slice, "device", None),
                    dtype=getattr(mask_slice, "dtype", None),
                )
            except TypeError:
                pad_mask = torch_mod.zeros((mask_slice.shape[0], pad_len))
            padded_logps.append(torch_mod.cat([logp_slice, pad_logp], dim=1))
            padded_masks.append(torch_mod.cat([mask_slice, pad_mask], dim=1))
        ref_token_logp = torch_mod.cat(padded_logps, dim=0) if padded_logps else None
        ref_token_mask = torch_mod.cat(padded_masks, dim=0) if padded_masks else None
    if progress_log:
        LOG.info(
            "reference_from_model_trl done | slices=%d ref_logp_shape=%s ref_tok_shape=%s",
            slices_seen,
            getattr(ref_logp, "shape", None),
            getattr(ref_tok, "shape", None),
        )
    return ref_logp, ref_tok, ref_token_logp, ref_token_mask


def reference_from_model(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    batching_cfg: BatchingSettings,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Run the frozen reference model to compute log-probs.

    :param score_batch: Prepared scoring batch with prompts/completions.
    :param runtime: Runtime handles exposing device and reference model.
    :param batching_cfg: Batching config controlling logprob chunking.
    :returns: Tuple of ``(ref_logp_sum, ref_token_counts)`` or ``None`` on failure.
    :rtype: tuple[Tensor, Tensor] | None
    """
    torch_mod = _refresh_torch()
    ref_model = runtime.get_ref_model()
    ref_logp_chunks: List[Tensor] = []
    ref_tok_chunks: List[Tensor] = []
    slice_log = _score_slice_log_enabled()
    progress_log = _progress_log_enabled()
    if progress_log:
        LOG.info(
            "reference_from_model start | device=%s slice_prefetch=%s logprob_chunk_size=%s",
            getattr(runtime, "device", None),
            getattr(batching_cfg, "slice_prefetch", None),
            getattr(batching_cfg, "logprob_chunk_size", None),
        )

    def _log_once(reason: str) -> None:
        if getattr(reference_from_model, "_diag_logged", False):
            return
        LOG.warning("reference_from_model returning None: %s", reason)
        setattr(reference_from_model, "_diag_logged", True)

    slices_seen = 0
    gather_full_params = False
    try:
        if _model_has_non2d_embeddings(ref_model):
            gather_full_params = True
            if not getattr(reference_from_model, "_forced_full_params_logged", False):
                LOG.warning(
                    "Reference scoring detected non-2D embeddings; forcing gather_full_params=True."
                )
                setattr(reference_from_model, "_forced_full_params_logged", True)
    except _SCORING_EXCEPTIONS:
        pass
    warned_full_params = False
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
    for slice_inputs, slice_mask, slice_labels in slice_iter:
        slices_seen += 1
        if slice_log:
            LOG.info(
                "reference_from_model slice start | idx=%d inputs_shape=%s mask_shape=%s labels_shape=%s",
                slices_seen - 1,
                getattr(slice_inputs, "shape", None),
                getattr(slice_mask, "shape", None),
                getattr(slice_labels, "shape", None),
            )
            slice_start = time.monotonic()
        no_grad_ctx = getattr(torch_mod, "no_grad", None) or nullcontext
        with no_grad_ctx():
            try:
                result = _chunked_sequence_logprobs(
                    ref_model,
                    input_ids=slice_inputs,
                    attention_mask=slice_mask,
                    labels=slice_labels,
                    chunk_size=batching_cfg.logprob_chunk_size,
                    gather_full_params=gather_full_params,
                    zero_gather_all_ranks=True,
                )
            except (
                _SCORING_EXCEPTIONS
            ) as exc:  # pragma: no cover - defensive diagnostics
                msg = str(exc) or ""
                if not gather_full_params and ("weight" in msg and "2-D" in msg):
                    gather_full_params = True
                    if not warned_full_params:
                        LOG.warning(
                            "Reference scoring saw non-2D embedding weight; retrying with gather_full_params=True."
                        )
                        warned_full_params = True
                    try:
                        result = _chunked_sequence_logprobs(
                            ref_model,
                            input_ids=slice_inputs,
                            attention_mask=slice_mask,
                            labels=slice_labels,
                            chunk_size=batching_cfg.logprob_chunk_size,
                            gather_full_params=gather_full_params,
                            zero_gather_all_ranks=True,
                        )
                    except _SCORING_EXCEPTIONS as exc_retry:
                        _log_once(
                            f"_chunked_sequence_logprobs raised {type(exc_retry).__name__}: {exc_retry}"
                        )
                        return None
                else:
                    _log_once(
                        f"_chunked_sequence_logprobs raised {type(exc).__name__}: {exc}"
                    )
                    return None
        # _chunked_sequence_logprobs normally returns (logp, tok_counts, pooled_hidden).
        if not isinstance(result, (tuple, list)):
            _log_once(
                f"chunked_sequence_logprobs returned invalid result | type={type(result)} len=n/a "
                f"inputs_shape={getattr(slice_inputs, 'shape', None)}"
            )
            return None
        seq_result = cast(Sequence[Any], result)
        if len(seq_result) < 2:
            _log_once(
                f"chunked_sequence_logprobs returned invalid result | type={type(result)} len={len(seq_result)} "
                f"inputs_shape={getattr(slice_inputs, 'shape', None)}"
            )
            return None
        ref_logp_slice, ref_tok_slice = seq_result[0], seq_result[1]
        if slice_log:
            LOG.info(
                "reference_from_model slice done | idx=%d seconds=%.2f logp_shape=%s tok_shape=%s",
                slices_seen - 1,
                time.monotonic() - slice_start,
                getattr(ref_logp_slice, "shape", None),
                getattr(ref_tok_slice, "shape", None),
            )
        if ref_logp_slice.numel() == 0 or ref_tok_slice.numel() == 0:
            _log_once(
                f"empty slice tensors | logp_numel={ref_logp_slice.numel()} tok_numel={ref_tok_slice.numel()} "
                f"inputs_shape={getattr(slice_inputs, 'shape', None)}"
            )
            return None
        LOG.debug(
            "Reference slice gathered | slice_idx=%d | logp_shape=%s | tok_shape=%s | inputs_shape=%s",
            slices_seen - 1,
            getattr(ref_logp_slice, "shape", None),
            getattr(ref_tok_slice, "shape", None),
            getattr(slice_inputs, "shape", None),
        )
        ref_logp_chunks.append(ref_logp_slice.detach().cpu())
        ref_tok_chunks.append(ref_tok_slice.detach().cpu())
    if slices_seen == 0:
        _log_once("iter_batch_slices yielded zero slices")
        return None
    if not ref_logp_chunks:
        _log_once("no reference slices produced any chunks")
        return None
    ref_logp = torch_mod.cat(ref_logp_chunks, dim=0).to(runtime.device)
    ref_tok = torch_mod.cat(ref_tok_chunks, dim=0).to(runtime.device)
    if progress_log:
        LOG.info(
            "reference_from_model done | slices=%d ref_logp_shape=%s ref_tok_shape=%s",
            slices_seen,
            getattr(ref_logp, "shape", None),
            getattr(ref_tok, "shape", None),
        )
    LOG.debug(
        "Reference gather succeeded | slices=%d | ref_logp_shape=%s | ref_tok_shape=%s",
        slices_seen,
        getattr(ref_logp, "shape", None),
        getattr(ref_tok, "shape", None),
    )
    return ref_logp, ref_tok


def gather_reference_logprobs(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    batching_cfg: BatchingSettings,
    *,
    trl_reference_scoring: bool = False,
    temperature: Optional[float] = None,
) -> Optional[ReferenceLogprobs]:
    """Compute log-probabilities by running the frozen reference model.

    This function handles distributed preflight checks to avoid ZeRO hangs and
    aggregates reference statistics into a ``ReferenceLogprobs`` object.

    :param score_batch: Prepared scoring batch with prompts/completions.
    :param runtime: Runtime handles exposing device, accelerator, and models.
    :param batching_cfg: Batching config controlling logprob chunking.
    :param trl_reference_scoring: When True, use TRL/open-r1 reference scoring logic.
    :param temperature: Optional temperature for TRL-style logprob scaling.
    :returns: ``ReferenceLogprobs`` or ``None`` when reference scoring fails.
    :rtype: ReferenceLogprobs | None
    """
    torch_mod = _refresh_torch()
    progress_log = _progress_log_enabled()
    slice_log = _score_slice_log_enabled()
    if progress_log:
        LOG.info(
            "gather_reference_logprobs start | device=%s slice_size=%s chunk_size=%s slice_prefetch=%s",
            getattr(runtime, "device", None),
            getattr(score_batch, "slice_size", None),
            getattr(batching_cfg, "logprob_chunk_size", None),
            getattr(batching_cfg, "slice_prefetch", None),
        )
        gather_start = time.monotonic()

    def _dim0(obj: object) -> int:
        if obj is None:
            return 0
        shape = getattr(obj, "shape", None)
        if shape is not None:
            try:
                return int(shape[0])
            except _SCORING_EXCEPTIONS as exc:
                LOG.debug(
                    "Failed to read shape[0] from tensor; falling back to len: %s", exc
                )
        if isinstance(obj, Sized):
            try:
                return int(len(obj))
            except _SCORING_EXCEPTIONS:
                return 0
        return 0

    def _first_slice_rows(sb: object) -> int:
        total = int(getattr(sb, "total_sequences", 0) or 0)
        slice_size = int(getattr(sb, "slice_size", 0) or 0)
        prompt_len = len(getattr(sb, "prompt_entries", []) or [])
        comp0 = _dim0(getattr(sb, "completion_ids", None))
        mask0 = _dim0(getattr(sb, "completion_attention_mask", None))
        # Bound by the first slice end index; if any of these are zero on a rank,
        # that rank will not enter the reference forward and ZeRO collectives can hang.
        return max(0, min(total, slice_size, prompt_len, comp0, mask0))

    def _safe_numel(obj: object) -> int | str | None:
        numel_fn = getattr(obj, "numel", None)
        if callable(numel_fn):
            try:
                value = numel_fn()
            except _SCORING_EXCEPTIONS:
                return "error"
            if value is None:
                return None
            if isinstance(value, numbers.Real):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return "error"
            if isinstance(value, str):
                return value
            return str(value)
        return None

    accelerator = getattr(runtime, "accelerator", None)
    num_processes = int(getattr(accelerator, "num_processes", 1) or 1)
    gather_obj = getattr(accelerator, "gather_object", None)
    preflight_rows = _first_slice_rows(score_batch)
    if num_processes > 1:
        # Preflight: if any rank has an empty first slice, skip reference scoring
        # everywhere to avoid deadlocking inside ZeRO parameter all-gathers.
        all_rows: Optional[List[Any]] = None
        if callable(gather_obj):
            try:
                gathered_rows = gather_obj(int(preflight_rows))
                if isinstance(gathered_rows, list):
                    all_rows = cast(List[Any], gathered_rows)
                elif gathered_rows is not None:
                    all_rows = [gathered_rows]
            except _SCORING_EXCEPTIONS:
                all_rows = None
        if all_rows is None:
            dist = _dist_collective_ready(torch_mod)
            if dist is not None:
                gathered: List[Any] = [
                    None for _ in range(max(int(dist.get_world_size()), 1))
                ]
                try:
                    dist.all_gather_object(gathered, int(preflight_rows))
                    all_rows = gathered
                except _SCORING_EXCEPTIONS:
                    all_rows = None
        if isinstance(all_rows, list) and all_rows:
            try:
                min_rows = min(int(v) for v in all_rows)
            except _SCORING_EXCEPTIONS:
                min_rows = int(preflight_rows)
            if min_rows <= 0:
                LOG.warning(
                    "Skipping reference scoring preflight: at least one rank has empty first slice "
                    "| local_rows=%s all_rows=%s total_sequences=%s slice_size=%s prompt_entries=%s comp_ids0=%s comp_mask0=%s",
                    preflight_rows,
                    all_rows,
                    getattr(score_batch, "total_sequences", None),
                    getattr(score_batch, "slice_size", None),
                    len(getattr(score_batch, "prompt_entries", []) or []),
                    _dim0(getattr(score_batch, "completion_ids", None)),
                    _dim0(getattr(score_batch, "completion_attention_mask", None)),
                )
                return None
        if slice_log and isinstance(all_rows, list):
            LOG.info(
                "gather_reference_logprobs preflight rows | local_rows=%s all_rows=%s",
                preflight_rows,
                all_rows,
            )

    if progress_log:
        LOG.info(
            "gather_reference_logprobs reference_from_model start | trl_reference_scoring=%s",
            trl_reference_scoring,
        )
    if trl_reference_scoring:
        tensors = reference_from_model_trl(
            score_batch, runtime, batching_cfg, temperature=temperature
        )
    else:
        tensors = reference_from_model(score_batch, runtime, batching_cfg)
    if progress_log:
        LOG.info(
            "gather_reference_logprobs reference_from_model done | ok=%s",
            bool(tensors is not None),
        )
    local_ok = tensors is not None
    global_ok = local_ok
    if num_processes > 1:
        if callable(gather_obj):
            try:
                gathered_obj: Any = gather_obj(bool(local_ok))
                if isinstance(gathered_obj, list):
                    global_ok = all(bool(x) for x in gathered_obj)
                else:
                    global_ok = bool(gathered_obj)
            except _SCORING_EXCEPTIONS:
                global_ok = local_ok
        else:
            dist = _dist_collective_ready(torch_mod)
            global_ok = _dist_all(dist, local_ok) if dist is not None else local_ok
    if tensors is None or not global_ok:
        LOG.debug(
            "gather_reference_logprobs returning None due to reference_from_model failure on at least one rank | "
            "slice_size=%s | chunk_size=%s | device=%s | local_ok=%s",
            getattr(score_batch, "slice_size", None),
            getattr(batching_cfg, "logprob_chunk_size", None),
            getattr(runtime, "device", None),
            local_ok,
        )
        return None
    ref_logp = tensors[0]
    ref_tok = tensors[1]
    ref_token_logp = None
    ref_token_mask = None
    if isinstance(tensors, (tuple, list)) and len(tensors) >= 4:
        ref_token_logp = tensors[2]
        ref_token_mask = tensors[3]
    LOG.debug(
        "reference_from_model tensors | ref_logp_shape=%s | ref_tok_shape=%s | ref_logp_numel=%s | ref_tok_numel=%s | ref_logp_dtype=%s | ref_tok_dtype=%s | ref_logp_device=%s | ref_tok_device=%s | ref_token_logp_shape=%s | ref_token_mask_shape=%s",
        getattr(ref_logp, "shape", None),
        getattr(ref_tok, "shape", None),
        _safe_numel(ref_logp),
        _safe_numel(ref_tok),
        getattr(ref_logp, "dtype", None),
        getattr(ref_tok, "dtype", None),
        getattr(ref_logp, "device", None),
        getattr(ref_tok, "device", None),
        getattr(ref_token_logp, "shape", None),
        getattr(ref_token_mask, "shape", None),
    )
    try:
        stats = finalize_reference_stats(
            ref_logp,
            ref_tok,
            ref_token_logp=ref_token_logp,
            ref_token_mask=ref_token_mask,
        )
    except _SCORING_EXCEPTIONS as exc:  # pragma: no cover - defensive diagnostics
        LOG.error(
            "finalize_reference_stats raised %s: %s | ref_logp_shape=%s | ref_tok_shape=%s | ref_logp_dtype=%s | ref_tok_dtype=%s | ref_logp_device=%s | ref_tok_device=%s",
            type(exc).__name__,
            exc,
            getattr(tensors[0], "shape", None),
            getattr(tensors[1], "shape", None),
            getattr(tensors[0], "dtype", None),
            getattr(tensors[1], "dtype", None),
            getattr(tensors[0], "device", None),
            getattr(tensors[1], "device", None),
        )
        return None
    LOG.debug(
        "gather_reference_logprobs built stats | ref_logp_sum_shape=%s | ref_tok_counts_shape=%s | ref_logp_sum_numel=%s | ref_tok_counts_numel=%s",
        getattr(getattr(stats, "ref_logp_sum", None), "shape", None),
        getattr(getattr(stats, "ref_tok_counts", None), "shape", None),
        _safe_numel(getattr(stats, "ref_logp_sum", None)),
        _safe_numel(getattr(stats, "ref_tok_counts", None)),
    )
    if progress_log:
        LOG.info(
            "gather_reference_logprobs done | seconds=%.2f",
            time.monotonic() - gather_start,
        )
    return stats


def finalize_reference_stats(
    ref_logp_sum: Tensor,
    ref_tok_counts: Tensor,
    *,
    ref_token_logp: Optional[Tensor] = None,
    ref_token_mask: Optional[Tensor] = None,
) -> ReferenceLogprobs:
    """Build a ``ReferenceLogprobs`` object and derived scalars.

    :param ref_logp_sum: Per-sequence sum of reference log-probabilities.
    :param ref_tok_counts: Per-sequence token counts.
    :param ref_token_logp: Optional per-token reference log-probs (completion-only).
    :param ref_token_mask: Optional per-token completion mask aligned to ``ref_token_logp``.
    :returns: Normalized reference stats and summary scalars.
    :rtype: ReferenceLogprobs
    :raises ValueError: If log-prob tensors cannot be safely normalized.
    """
    torch_mod = _refresh_torch()
    try:
        logp_arr_raw = _to_numpy_array(ref_logp_sum)
        tok_arr_raw = _to_numpy_array(ref_tok_counts)
        logp_numel = None
        tok_numel = None
        numel_fn = getattr(ref_logp_sum, "numel", None)
        if callable(numel_fn):
            try:
                logp_numel = numel_fn()
            except _SCORING_EXCEPTIONS:
                logp_numel = None
        numel_fn_tok = getattr(ref_tok_counts, "numel", None)
        if callable(numel_fn_tok):
            try:
                tok_numel = numel_fn_tok()
            except _SCORING_EXCEPTIONS:
                tok_numel = None

        # If numpy conversion dropped elements (e.g., CUDA+bfloat16 -> empty array),
        # retry with a CPU float32 view so we do not lose reference stats.
        logp_numel_val = (
            float(logp_numel) if isinstance(logp_numel, numbers.Real) else 0.0
        )
        tok_numel_val = float(tok_numel) if isinstance(tok_numel, numbers.Real) else 0.0
        if getattr(logp_arr_raw, "size", 0) == 0 and (
            logp_numel_val > 0.0 or tok_numel_val > 0.0
        ):
            try:
                tensor_cpu = ref_logp_sum.detach().to("cpu")
                if hasattr(tensor_cpu, "float"):
                    tensor_cpu = tensor_cpu.float()
                logp_arr_raw = np.asarray(tensor_cpu.reshape(-1).cpu().numpy())
                LOG.debug(
                    "finalize_reference_stats fallback tensor->numpy succeeded | target_numel=%s | result_size=%s",
                    logp_numel,
                    getattr(logp_arr_raw, "size", None),
                )
            except _SCORING_EXCEPTIONS as exc:
                LOG.debug(
                    "finalize_reference_stats fallback tensor->numpy failed | target_numel=%s | exc=%s",
                    logp_numel,
                    exc,
                )
        # If still empty but token counts exist, treat as a hard failure. Returning
        # zeros here can yield pathological KL (e.g., delta ~= -cur_logp) and
        # destabilize training; let callers retry/skip the batch instead.
        if getattr(logp_arr_raw, "size", 0) == 0 and tok_numel_val > 0.0:
            raise ValueError(
                "Reference logp tensor conversion produced an empty array while token counts exist; "
                "cannot finalize reference stats safely."
            )

        def _safe_size(x: object) -> Optional[int]:
            try:
                return int(np.size(cast(ArrayLike, x)))
            except _SCORING_EXCEPTIONS:
                return None

        LOG.debug(
            "finalize_reference_stats raw inputs | ref_logp_len=%s | ref_tok_len=%s | ref_logp_numel=%s | ref_tok_numel=%s | ref_logp_dtype=%s | ref_tok_dtype=%s | ref_logp_device=%s | ref_tok_device=%s | ref_logp_sample=%s",
            _safe_size(logp_arr_raw),
            _safe_size(tok_arr_raw),
            logp_numel,
            tok_numel,
            getattr(ref_logp_sum, "dtype", None),
            getattr(ref_tok_counts, "dtype", None),
            getattr(ref_logp_sum, "device", None),
            getattr(ref_tok_counts, "device", None),
            logp_arr_raw[:4] if hasattr(logp_arr_raw, "__getitem__") else None,
        )
        logp_arr = np.asarray(logp_arr_raw, dtype=float)
        tok_arr = np.asarray(tok_arr_raw, dtype=float)
        if tok_arr.size == 0:
            tok_arr = np.asarray([0.0])
        invalid_mask = ~np.isfinite(tok_arr) | (tok_arr < 0)
        if invalid_mask.any():
            finite_replaced = np.nan_to_num(tok_arr, nan=0.0, posinf=0.0, neginf=0.0)
            tok_arr = np.maximum(finite_replaced, 0.0)
            LOG.debug(
                "Clamped %d invalid reference token counts to non-negative finite values.",
                int(invalid_mask.sum()),
            )
        if tok_arr.size == 0:
            tok_arr = np.asarray([0.0])
        safe_tok = np.maximum(tok_arr, 1.0)
        logp_norm = np.divide(
            logp_arr,
            safe_tok,
            out=np.zeros_like(logp_arr, dtype=float),
            where=safe_tok > 0,
        )
        ref_logp_sum_tensor = torch_mod.tensor(
            logp_norm, dtype=getattr(torch_mod, "float32", None)
        )
        ref_tok_counts_tensor = torch_mod.tensor(
            tok_arr, dtype=getattr(torch_mod, "float32", None)
        )
        ref_logp_raw_tensor = torch_mod.tensor(
            logp_arr, dtype=getattr(torch_mod, "float32", None)
        )
        ref_logp_mean = (
            float(np.asarray(logp_arr, dtype=float).mean()) if logp_arr.size else 0.0
        )
        avg_completion_tokens = float(np.asarray(tok_arr, dtype=float).mean())
        return ReferenceLogprobs(
            ref_logp_sum=ref_logp_sum_tensor,
            ref_tok_counts=ref_tok_counts_tensor,
            ref_logp_sum_raw=ref_logp_raw_tensor,
            ref_logp_mean=ref_logp_mean,
            avg_completion_tokens=avg_completion_tokens,
            ref_token_logp=ref_token_logp,
            ref_token_mask=ref_token_mask,
        )
    except _SCORING_EXCEPTIONS as exc:  # pragma: no cover - defensive diagnostics
        LOG.exception(
            "finalize_reference_stats failed | exc=%s | ref_logp_sum_shape=%s | ref_tok_counts_shape=%s | ref_logp_sum_dtype=%s | ref_tok_counts_dtype=%s | ref_logp_sum_device=%s | ref_tok_counts_device=%s",
            type(exc).__name__,
            getattr(ref_logp_sum, "shape", None),
            getattr(ref_tok_counts, "shape", None),
            getattr(ref_logp_sum, "dtype", None),
            getattr(ref_tok_counts, "dtype", None),
            getattr(ref_logp_sum, "device", None),
            getattr(ref_tok_counts, "device", None),
        )
        raise


def reference_stats_from_policy_logprobs(
    cur_logp_sum: Tensor,
    tok_counts: Tensor,
) -> ReferenceLogprobs:
    """Build ``ReferenceLogprobs`` assuming reference == current policy (KL ~= 0).

    :param cur_logp_sum: Current policy log-prob sums per sequence.
    :param tok_counts: Token counts per sequence.
    :returns: Reference stats derived directly from the current policy.
    :rtype: ReferenceLogprobs
    """
    torch_mod = _refresh_torch()
    base_dtype = getattr(torch_mod, "float32", None)
    device = getattr(cur_logp_sum, "device", None)
    cur_tensor = _as_torch_tensor(
        torch_mod, cur_logp_sum, device=device, dtype=base_dtype
    ).view(-1)
    tok_tensor = _as_torch_tensor(
        torch_mod, tok_counts, device=device, dtype=base_dtype
    ).view(-1)
    cur_tensor = cast(Tensor, cur_tensor)
    tok_tensor = cast(Tensor, tok_tensor)
    try:
        tok_tensor = tok_tensor.clamp(min=1.0)
    except _SCORING_EXCEPTIONS as exc:
        LOG.debug("Failed to clamp token counts; continuing: %s", exc)
    tok_tensor = cast(Tensor, tok_tensor)
    detach_fn = getattr(cur_tensor, "detach", None)
    if callable(detach_fn):
        cur_tensor = detach_fn()
        cur_tensor = cast(Tensor, cur_tensor)
    detach_fn = getattr(tok_tensor, "detach", None)
    if callable(detach_fn):
        tok_tensor = detach_fn()
        tok_tensor = cast(Tensor, tok_tensor)
    ref_logp_sum_raw = cast(Tensor, cur_tensor)
    ref_logp_sum = cast(Any, ref_logp_sum_raw) / tok_tensor
    try:
        ref_logp_mean = float(
            cast(Any, ref_logp_sum_raw).detach().float().cpu().mean().item()
        )
    except _SCORING_EXCEPTIONS:
        try:
            ref_logp_mean = float(cast(Any, ref_logp_sum_raw).mean())
        except _SCORING_EXCEPTIONS:
            ref_logp_mean = 0.0
    try:
        avg_completion_tokens = float(
            cast(Any, tok_tensor).detach().float().cpu().mean().item()
        )
    except _SCORING_EXCEPTIONS:
        try:
            avg_completion_tokens = float(cast(Any, tok_tensor).mean())
        except _SCORING_EXCEPTIONS:
            avg_completion_tokens = 0.0
    return ReferenceLogprobs(
        ref_logp_sum=ref_logp_sum,
        ref_tok_counts=tok_tensor,
        ref_logp_sum_raw=ref_logp_sum_raw,
        ref_logp_mean=ref_logp_mean,
        avg_completion_tokens=avg_completion_tokens,
    )


def _meta_field(entry: object, *names: str) -> object | None:
    """Return the first matching field from a metadata entry."""
    if entry is None:
        return None
    sentinel = object()
    for name in names:
        if isinstance(entry, Mapping):
            if name in entry:
                return entry.get(name)
        else:
            value = getattr(entry, name, sentinel)
            if value is not sentinel:
                return value
    return None


def _coerce_int_optional(value: object) -> Optional[int]:
    """Return ``int(value)`` when possible, otherwise ``None``."""
    if isinstance(value, numbers.Integral):
        return int(value)
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _coerce_logprob_value(value: object) -> Optional[float]:
    """Best-effort conversion of a token logprob payload into a float."""
    if value is None:
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, Mapping):
        if "logprob" in value:
            return _coerce_logprob_value(value.get("logprob"))
        if len(value) == 1:
            return _coerce_logprob_value(next(iter(value.values())))
        return None
    attr_val = getattr(value, "logprob", None)
    if attr_val is not None:
        return _coerce_logprob_value(attr_val)
    return None


def _sum_token_logprobs(token_logprobs: object) -> Optional[float]:
    """Return the sum of per-token logprobs when the payload is parseable."""
    if token_logprobs is None or isinstance(token_logprobs, Mapping):
        return None
    if not isinstance(token_logprobs, Iterable):
        return None
    try:
        iterator = iter(token_logprobs)
    except TypeError:
        return None
    total = 0.0
    saw_any = False
    for item in iterator:
        val = _coerce_logprob_value(item)
        if val is None:
            return None
        total += float(val)
        saw_any = True
    return total if saw_any else None


def reference_from_vllm_meta(
    flat_meta: Sequence[Optional[object]],
    total_sequences: int,
    device: TorchDevice,
) -> Optional[ReferenceLogprobs]:
    """Convert flattened vLLM log-prob metadata into ``ReferenceLogprobs``.

    :param flat_meta: Flat list of vLLM metadata entries (one per completion).
    :param total_sequences: Expected number of sequences in the batch.
    :param device: Device for the resulting tensors.
    :returns: ``ReferenceLogprobs`` or ``None`` when metadata is incomplete.
    :rtype: ReferenceLogprobs | None
    """
    if not flat_meta:
        return None
    if len(flat_meta) != total_sequences:
        return None
    logp_vals: List[float] = []
    tok_counts: List[int] = []
    for entry in flat_meta:
        if entry is None:
            return None
        logprob_sum = _meta_field(entry, "logprob_sum", "cumulative_logprob")
        token_count = _meta_field(entry, "token_count", "num_tokens")
        token_logprobs = _meta_field(entry, "token_logprobs", "logprobs")
        if logprob_sum is None and token_logprobs is not None:
            logprob_sum = _sum_token_logprobs(token_logprobs)
        if token_count is None and token_logprobs is not None:
            if isinstance(token_logprobs, Sized):
                try:
                    token_count = len(token_logprobs)
                except (TypeError, ValueError, AttributeError):
                    token_count = None
            else:
                token_count = None
        logprob_sum_val = _coerce_logprob_value(logprob_sum)
        if logprob_sum_val is None or token_count is None:
            return None
        token_count_int = _coerce_int_optional(token_count)
        if token_count_int is None:
            return None
        logp_vals.append(logprob_sum_val)
        tok_counts.append(max(1, token_count_int))
    torch_mod = _refresh_torch()
    ref_logp_sum = torch_mod.tensor(
        logp_vals, dtype=getattr(torch_mod, "float32", None), device=device
    )
    ref_tok_counts = torch_mod.tensor(
        tok_counts, dtype=getattr(torch_mod, "float32", None), device=device
    )
    return finalize_reference_stats(ref_logp_sum, ref_tok_counts)


def vllm_meta_has_logprobs(
    flat_meta: Optional[Sequence[Optional[object]]],
    total_sequences: Optional[int] = None,
) -> bool:
    """Return True when vLLM metadata includes per-completion logprob info.

    :param flat_meta: Flat list of vLLM metadata entries.
    :param total_sequences: Optional expected length used for sanity checks.
    :returns: ``True`` when logprob metadata appears complete.
    :rtype: bool
    """

    if not flat_meta:
        return False
    if total_sequences is not None and total_sequences >= 0:
        try:
            if len(flat_meta) != int(total_sequences):
                return False
        except (TypeError, ValueError):
            return False
    for entry in flat_meta:
        if entry is None:
            return False
        logprob_sum = getattr(entry, "logprob_sum", None)
        token_count = getattr(entry, "token_count", None)
        if isinstance(entry, dict):
            if logprob_sum is None:
                logprob_sum = entry.get("logprob_sum") or entry.get(
                    "cumulative_logprob"
                )
            if token_count is None:
                token_count = entry.get("token_count")
        if logprob_sum is None or token_count is None:
            return False
    return True
