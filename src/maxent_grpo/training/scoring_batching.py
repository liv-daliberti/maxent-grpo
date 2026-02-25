# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Batch construction and slice materialization helpers for scoring."""

from __future__ import annotations

from . import scoring_common as _common

for _name in dir(_common):
    if _name.startswith("__"):
        continue
    globals().setdefault(_name, getattr(_common, _name))
del _name

torch = _refresh_torch()

@dataclass
class CompletionTensors:
    """Completion token IDs and masks."""

    ids: Tensor
    mask: Tensor


@dataclass
class _SliceState:
    """Cached tensors and metadata required for batch slicing."""

    total_sequences: int
    slice_size: int
    completion_ids: Tensor
    completion_mask: Tensor
    prompt_entries: List[PromptCacheEntry]
    pad_token_id: int
    max_prompt_len: int
    score_tail_tokens: Optional[int] = None

    @classmethod
    def from_score_batch(cls, score_batch: ScoreBatch) -> "_SliceState":
        """Build a state snapshot derived from a score batch."""
        total_sequences = score_batch.total_sequences
        if total_sequences == 0:
            slice_size = 0
        else:
            slice_size = (
                score_batch.slice_size
                if score_batch.slice_size > 0
                else total_sequences
            )
        return cls(
            total_sequences=total_sequences,
            slice_size=slice_size,
            completion_ids=score_batch.completion_ids,
            completion_mask=score_batch.completion_attention_mask,
            prompt_entries=score_batch.prompt_entries,
            pad_token_id=score_batch.pad_token_id,
            max_prompt_len=max(1, score_batch.max_prompt_len),
            score_tail_tokens=getattr(score_batch, "score_tail_tokens", None),
        )


@dataclass
class _PromptCacheConfig:
    prompt_length_cache_get: Optional[Callable[[str], PromptCacheEntry]]
    prompt_cache_size: int = 0


def _collect_prompt_entries(
    prompt_batch: List[str],
    batching_cfg: _PromptCacheConfig,
) -> Optional[List[PromptCacheEntry]]:
    """Resolve cached prompt tokenization for a batch of strings.

    :param prompt_batch: Raw prompt strings to fetch from the cache.
    :type prompt_batch: list[str]
    :param batching_cfg: Prompt cache configuration containing the cache getter.
    :type batching_cfg: _PromptCacheConfig
    :returns: Cached prompt entries or ``None`` when the batch is empty.
    :rtype: list[PromptCacheEntry] | None
    """
    cache_size = getattr(batching_cfg, "prompt_cache_size", 0) or 0
    prompt_fn = getattr(batching_cfg, "prompt_length_cache_get", None)
    if cache_size > 0 and callable(prompt_fn):
        cached = getattr(batching_cfg, "_cached_prompt_lookup", None)
        underlying = getattr(batching_cfg, "_cached_prompt_source", None)
        if cached is None or underlying is not prompt_fn:
            cached = lru_cache(maxsize=cache_size)(prompt_fn)
            setattr(batching_cfg, "_cached_prompt_lookup", cached)
            setattr(batching_cfg, "_cached_prompt_source", prompt_fn)
        prompt_fn = cached
    if not callable(prompt_fn):
        return None
    prompt_entries = cast(
        List[PromptCacheEntry], [prompt_fn(prompt) for prompt in prompt_batch]
    )
    if not prompt_entries:
        return None
    return prompt_entries


def _tokenize_completions(
    completion_batch: List[str],
    tokenizer: PreTrainedTokenizer,
    generation_cfg: GenerationSettings,
) -> CompletionTensors:
    """Tokenize completions into padded tensors.

    :param completion_batch: Completion strings aligned with prompts.
    :type completion_batch: list[str]
    :param tokenizer: Tokenizer used to encode the completions.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param generation_cfg: Generation settings (controls max length).
    :type generation_cfg: GenerationSettings
    :returns: Completion token IDs and attention masks.
    :rtype: CompletionTensors
    """
    _refresh_torch()
    old_padding_side = getattr(tokenizer, "padding_side", None)
    try:
        try:
            if old_padding_side is not None:
                tokenizer.padding_side = "right"
            completion_enc = tokenizer(
                completion_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=generation_cfg.max_completion_len,
                add_special_tokens=False,
            )
        except TypeError:
            completion_enc = tokenizer(completion_batch)
    finally:
        if old_padding_side is not None:
            try:
                tokenizer.padding_side = old_padding_side
            except Exception:
                pass
    torch_mod = cast(_TorchModuleLike, sys.modules.get("torch", torch))
    ids = _maybe_long_tensor(completion_enc["input_ids"], torch_mod)
    mask = _maybe_long_tensor(completion_enc["attention_mask"], torch_mod)
    return CompletionTensors(
        ids=ids,
        mask=mask,
    )


def _completion_tensors_from_token_ids(
    token_ids: List[List[int]],
    *,
    pad_token_id: int,
    max_length: int,
) -> CompletionTensors:
    """Build completion tensors from pre-tokenized token-id sequences."""
    torch_mod = _refresh_torch()
    limit = int(max_length or 0)
    clipped: List[List[int]] = []
    for seq in token_ids:
        seq_list = list(seq)
        if limit > 0:
            seq_list = seq_list[:limit]
        clipped.append(seq_list)
    max_len = max((len(seq) for seq in clipped), default=0)
    batch = len(clipped)
    ids_arr = np.full((batch, max_len), int(pad_token_id), dtype=np.int64)
    mask_arr = np.zeros((batch, max_len), dtype=np.int64)
    for row, seq in enumerate(clipped):
        if not seq:
            continue
        ids_arr[row, : len(seq)] = np.asarray(seq, dtype=np.int64)
        mask_arr[row, : len(seq)] = 1
    ids = torch_mod.tensor(ids_arr, dtype=getattr(torch_mod, "long", None))
    mask = torch_mod.tensor(mask_arr, dtype=getattr(torch_mod, "long", None))
    return CompletionTensors(ids=ids, mask=mask)


def _prepare_prompt_slice(
    prompt_slice: List[PromptCacheEntry],
    max_prompt_len: int,
    pad_token_id: int,
    ids_dtype: TorchDType,
    mask_dtype: TorchDType,
) -> Tuple[Tensor, Tensor, List[int]]:
    """Materialize prompt tensors for one scoring slice.

    :param prompt_slice: Cached prompt entries for the current slice.
    :type prompt_slice: list[PromptCacheEntry]
    :param max_prompt_len: Maximum prompt length to materialize.
    :type max_prompt_len: int
    :param pad_token_id: Token ID used to pad prompts.
    :type pad_token_id: int
    :param ids_dtype: Dtype for the generated ID tensor.
    :type ids_dtype: torch.dtype
    :param mask_dtype: Dtype for the generated attention mask tensor.
    :type mask_dtype: torch.dtype
    :returns: Tuple of (prompt_ids, prompt_mask, prompt_lengths).
    :rtype: tuple[Tensor, Tensor, list[int]]
    """
    torch_mod = _refresh_torch()
    ids_dtype = getattr(ids_dtype, "np_dtype", ids_dtype)
    mask_dtype = getattr(mask_dtype, "np_dtype", mask_dtype)

    def _coerce_np_dtype(dtype: object) -> np.dtype | type[np.generic]:
        # Catch torch-style dtype strings/objects early.
        if isinstance(dtype, str) and dtype.startswith("torch"):
            return np.int64
        dtype_str = str(dtype)
        if dtype_str.startswith("torch."):
            return np.int64
        resolved = _resolve_dtype(dtype)
        if resolved is None:
            name_attr = getattr(dtype, "name", None)
            if isinstance(name_attr, str):
                try:
                    resolved = np.dtype(name_attr)
                except (TypeError, ValueError):
                    resolved = None
        if resolved is None:
            return np.int64
        try:
            return np.dtype(resolved)
        except (TypeError, ValueError):
            return np.int64

    ids_np_dtype = _coerce_np_dtype(ids_dtype) or np.int64
    mask_np_dtype = _coerce_np_dtype(mask_dtype) or np.int64
    prompt_lengths = [min(entry.length, max_prompt_len) for entry in prompt_slice]
    max_prompt_tokens = max(prompt_lengths) if prompt_lengths else 0
    batch_size = len(prompt_slice)
    if max_prompt_tokens > 0:
        prompt_ids_arr = np.full(
            (batch_size, max_prompt_tokens),
            pad_token_id,
            dtype=ids_np_dtype,
        )
        prompt_mask_arr = np.zeros(
            (batch_size, max_prompt_tokens),
            dtype=mask_np_dtype,
        )
        for row, (entry, length) in enumerate(zip(prompt_slice, prompt_lengths)):
            if length == 0:
                continue
            start = max_prompt_tokens - length
            prompt_ids_arr[row, start:start + length] = entry.input_ids[:length]
            prompt_mask_arr[row, start:start + length] = entry.attention_mask[:length]

        def _safe_dtype(dtype: object) -> object | None:
            return (
                None
                if (
                    isinstance(dtype, str)
                    or str(dtype).startswith("torch.")
                    or isinstance(dtype, np.dtype)
                )
                else dtype
            )

        tensor_ids_dtype = _safe_dtype(ids_dtype)
        tensor_mask_dtype = _safe_dtype(mask_dtype)
        prompt_ids = torch_mod.tensor(prompt_ids_arr, dtype=tensor_ids_dtype)
        prompt_mask = torch_mod.tensor(prompt_mask_arr, dtype=tensor_mask_dtype)
    else:
        # Avoid torch.empty here because minimal stubs may omit it.
        def _safe_dtype(dtype: object) -> object | None:
            return (
                None
                if (
                    isinstance(dtype, str)
                    or str(dtype).startswith("torch.")
                    or isinstance(dtype, np.dtype)
                )
                else dtype
            )

        tensor_ids_dtype = _safe_dtype(ids_dtype)
        tensor_mask_dtype = _safe_dtype(mask_dtype)
        prompt_ids = torch_mod.zeros((batch_size, 0), dtype=tensor_ids_dtype)
        prompt_mask = torch_mod.zeros((batch_size, 0), dtype=tensor_mask_dtype)
    return prompt_ids, prompt_mask, prompt_lengths


def _slice_tail_window(
    start_idx: int,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Slice tail tokens without closing over loop variables."""
    if start_idx <= 0:
        return input_ids, attention_mask, labels
    return (
        input_ids[:, start_idx:],
        attention_mask[:, start_idx:],
        labels[:, start_idx:],
    )


def iter_batch_slices(
    score_batch: ScoreBatch,
    device: TorchDevice,  # kept for API symmetry with callers
    *,
    eos_token_id: Optional[int] = None,
    apply_eos_mask: bool = False,
) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
    """Yield scoring slices for a batch, assembling prompt tensors on demand.

    :param score_batch: Prepared prompt/completion tensors and metadata.
    :type score_batch: ScoreBatch
    :param device: Device where tensors should be materialized.
    :type device: torch.device
    :param eos_token_id: Optional EOS token id for TRL-style completion masking.
    :type eos_token_id: int | None
    :param apply_eos_mask: When ``True``, apply EOS-aware completion masks.
    :type apply_eos_mask: bool
    :yields: Tuples of ``(input_ids, attention_mask, labels)`` per slice.
    :rtype: Iterator[tuple[Tensor, Tensor, Tensor]]
    """
    torch_mod = _refresh_torch()
    state = _SliceState.from_score_batch(score_batch)
    if state.total_sequences == 0 or state.slice_size <= 0:
        return
    as_tensor = getattr(torch_mod, "as_tensor", getattr(torch_mod, "tensor", None))
    if as_tensor is None:
        raise AttributeError("torch.as_tensor (or tensor) is required for scoring.")
    as_tensor_fn = cast(Callable[..., Tensor], as_tensor)

    def _ensure_tensor(obj: object, *, target_device: object | None = None) -> Tensor:
        """Best-effort conversion that tolerates numpy arrays/stubs."""
        is_tensor_fn = getattr(torch_mod, "is_tensor", None)
        try:
            if callable(is_tensor_fn) and is_tensor_fn(obj):
                return cast(Tensor, obj)
        except _SCORING_EXCEPTIONS as exc:  # pragma: no cover - defensive
            LOG.debug("torch.is_tensor check failed; continuing: %s", exc)
        tensor_type = getattr(torch_mod, "Tensor", None)
        if tensor_type is not None and isinstance(obj, tensor_type):
            return cast(Tensor, obj)
        tensor_ctor = getattr(torch_mod, "tensor", None)
        if callable(tensor_ctor):
            data = getattr(obj, "arr", None)
            if data is None:
                data = obj
            try:
                return cast(
                    Tensor,
                    tensor_ctor(
                        np.asarray(data),
                        device=target_device,
                        dtype=getattr(obj, "dtype", None),
                    ),
                )
            except TypeError:
                return cast(Tensor, tensor_ctor(np.asarray(data)))
        return cast(Tensor, obj)

    def as_tensor_typed(*args: object, **kwargs: object) -> Tensor:
        return cast(Tensor, as_tensor_fn(*args, **kwargs))

    for start in range(0, state.total_sequences, state.slice_size):
        end = min(start + state.slice_size, state.total_sequences)
        prompt_slice = state.prompt_entries[start:end]
        comp_ids_slice = state.completion_ids[start:end]
        comp_mask_slice = state.completion_mask[start:end]
        if device is not None:
            try:
                comp_ids_slice = comp_ids_slice.to(device)
                comp_mask_slice = comp_mask_slice.to(device)
            except (AttributeError, TypeError, ValueError) as exc:
                # Some lightweight torch stubs treat the ``device`` argument as
                # a dtype. When that happens we leave tensors on their current
                # device and rely on ``as_tensor`` below to normalize types.
                LOG.debug("Failed to move completion tensors to device: %s", exc)
        batch_size = len(prompt_slice)
        if batch_size == 0:
            continue
        prompt_ids, prompt_mask, prompt_lengths = _prepare_prompt_slice(
            prompt_slice,
            state.max_prompt_len,
            state.pad_token_id,
            comp_ids_slice.dtype,
            comp_mask_slice.dtype,
        )
        if device is not None:
            try:
                prompt_ids = prompt_ids.to(device)
                prompt_mask = prompt_mask.to(device)
            except (AttributeError, TypeError, ValueError) as exc:
                LOG.debug("Failed to move prompt tensors to device: %s", exc)
        # Ensure tensors before concatenation (protect against stubs/numpy).
        prompt_ids = as_tensor_typed(prompt_ids, device=device)
        prompt_mask = as_tensor_typed(prompt_mask, device=device)
        comp_ids_slice = as_tensor_typed(comp_ids_slice, device=device)
        comp_mask_slice = as_tensor_typed(comp_mask_slice, device=device)
        if apply_eos_mask:
            completion_mask = _apply_eos_completion_mask(
                comp_ids_slice, eos_token_id, completion_mask=comp_mask_slice
            )
            try:
                completion_mask = completion_mask * comp_mask_slice
            except _SCORING_EXCEPTIONS:
                comp_arr = np.asarray(getattr(comp_mask_slice, "arr", comp_mask_slice))
                eos_arr = np.asarray(getattr(completion_mask, "arr", completion_mask))
                completion_mask = as_tensor_typed(
                    comp_arr * eos_arr, device=getattr(comp_mask_slice, "device", None)
                )
            comp_mask_slice = completion_mask
        # Drop completion columns that are padding for every sequence so tail-only
        # scoring keeps real tokens instead of global pad regions.
        comp_tokens_present = None
        active_comp_columns: List[int] = []
        try:
            comp_tokens_present = (comp_mask_slice != 0).any(dim=0)
        except _SCORING_EXCEPTIONS as exc:
            LOG.debug("Failed to compute completion token presence mask: %s", exc)
        if comp_tokens_present is None:
            comp_tokens_arr = np.asarray(
                getattr(comp_mask_slice, "arr", comp_mask_slice)
            ) != 0
            col_activity = comp_tokens_arr.any(axis=0)
            active_idx_np = np.nonzero(col_activity)[0]
            active_comp_columns = [int(idx) for idx in active_idx_np.tolist()]
            if active_comp_columns:
                last_valid_idx = active_comp_columns[-1] + 1
            else:
                last_valid_idx = 0
        else:
            try:
                nonzero_cols = torch_mod.nonzero(comp_tokens_present, as_tuple=False).view(
                    -1
                )
                active_comp_columns = [int(idx.item()) for idx in nonzero_cols]
                last_valid_idx = active_comp_columns[-1] + 1 if active_comp_columns else 0
            except _SCORING_EXCEPTIONS:
                active_comp_columns = []
                last_valid_idx = 0
        if last_valid_idx <= 0:
            comp_ids_slice = comp_ids_slice[:, :0]
            comp_mask_slice = comp_mask_slice[:, :0]
        elif last_valid_idx < getattr(comp_ids_slice, "shape", [0, 0])[1]:
            comp_ids_slice = comp_ids_slice[:, :last_valid_idx]
            comp_mask_slice = comp_mask_slice[:, :last_valid_idx]
        full_input_ids = torch_mod.cat([prompt_ids, comp_ids_slice], dim=1)
        full_attention_mask = torch_mod.cat([prompt_mask, comp_mask_slice], dim=1)
        labels_tensor = full_input_ids.clone()
        for idx, plen in enumerate(prompt_lengths):
            labels_tensor[idx, :plen] = -100
        prompt_width = getattr(prompt_ids, "shape", [0, 0])[1] if prompt_ids is not None else 0
        comp_width = getattr(comp_ids_slice, "shape", [0, 0])[1] if comp_ids_slice is not None else 0
        if comp_width > 0:
            comp_slice = slice(prompt_width, prompt_width + comp_width)
            comp_labels = labels_tensor[:, comp_slice]
            updated_labels = comp_labels
            pad_mask = None
            pad_mask_arr = None
            has_padding = False
            try:
                pad_mask = comp_mask_slice == 0
                has_padding = bool(pad_mask.any())
            except _SCORING_EXCEPTIONS:
                pad_mask = None
            if pad_mask is None:
                try:
                    pad_mask_arr = np.asarray(
                        getattr(comp_mask_slice, "arr", comp_mask_slice)
                    ) == 0
                    has_padding = bool(pad_mask_arr.any())
                except _SCORING_EXCEPTIONS:
                    pad_mask_arr = None
                    has_padding = False
            if has_padding:
                try:
                    if pad_mask is None:
                        raise AttributeError
                    updated_labels = comp_labels.masked_fill(pad_mask, -100)
                except _SCORING_EXCEPTIONS:
                    if pad_mask_arr is None:
                        pad_mask_arr = np.asarray(
                            getattr(comp_mask_slice, "arr", comp_mask_slice)
                        ) == 0
                    comp_arr = np.asarray(getattr(comp_labels, "arr", comp_labels))
                    comp_arr[pad_mask_arr] = -100
                    updated_labels = as_tensor_typed(
                        comp_arr, device=getattr(labels_tensor, "device", None)
                    )
                labels_tensor[:, comp_slice] = updated_labels
        full_labels = labels_tensor
        input_ids = full_input_ids
        attention_mask = full_attention_mask
        tail_tokens = getattr(state, "score_tail_tokens", None)
        if tail_tokens is not None:
            try:
                tail_tokens = int(tail_tokens)
            except (TypeError, ValueError):
                tail_tokens = None
        if tail_tokens is not None and tail_tokens > 0:
            max_len = int(getattr(full_input_ids, "shape", [0, 0])[1] or 0)
            tail_tokens = min(tail_tokens, max_len) if max_len > 0 else 0
            if tail_tokens > 0 and tail_tokens < max_len:
                slice_start = max(0, max_len - tail_tokens)
                first_comp_global = None
                last_comp_global = None
                if active_comp_columns:
                    first_comp_global = prompt_width + active_comp_columns[0]
                    last_comp_global = prompt_width + active_comp_columns[-1] + 1
                input_ids, attention_mask, labels_tensor = _slice_tail_window(
                    slice_start,
                    full_input_ids,
                    full_attention_mask,
                    full_labels,
                )
                if (
                    first_comp_global is not None
                    and last_comp_global is not None
                    and slice_start >= last_comp_global
                ):
                    safe_start = max(first_comp_global, last_comp_global - tail_tokens)
                    input_ids, attention_mask, labels_tensor = _slice_tail_window(
                        safe_start,
                        full_input_ids,
                        full_attention_mask,
                        full_labels,
                    )
        # Materialize tensors for the ref model; keep device parity with completions.
        target_device = device or getattr(comp_ids_slice, "device", None)
        target_dtype = getattr(torch_mod, "long", None)
        input_ids_out = as_tensor_typed(
            input_ids, device=target_device, dtype=target_dtype
        )
        attention_mask_out = as_tensor_typed(
            attention_mask, device=target_device, dtype=target_dtype
        )
        labels_out = as_tensor_typed(
            labels_tensor, device=target_device, dtype=target_dtype
        )
        input_ids_out = _ensure_tensor(input_ids_out, target_device=target_device)
        attention_mask_out = _ensure_tensor(attention_mask_out, target_device=target_device)
        labels_out = _ensure_tensor(labels_out, target_device=target_device)
        # Some lightweight stubs do not preserve the requested dtype object on
        # the Tensor wrapper (they store only the underlying numpy dtype). For
        # tests that compare against ``torch.long`` we make a best-effort pass
        # to align the exposed ``dtype`` attribute with the module constant.
        long_dtype = getattr(torch_mod, "long", None)
        if long_dtype is not None:
            proxy = _LongDTypeProxy(long_dtype)
            for _tensor in (input_ids_out, labels_out):
                try:  # pragma: no cover - exercised via stubbed environments
                    setattr(_tensor, "dtype", proxy)
                except _SCORING_EXCEPTIONS as exc:
                    LOG.debug("Unable to patch tensor dtype proxy: %s", exc)
        yield (input_ids_out, attention_mask_out, labels_out)


def build_score_batch(
    reward_comp: RewardComputation,
    tokenizer: PreTrainedTokenizer,
    generation_cfg: GenerationSettings,
    batching_cfg: BatchingSettings,
) -> Optional[ScoreBatch]:
    """Tokenize prompt+completion pairs and prepare masks/labels.

    :param reward_comp: Reward computation payload containing prompts and completions.
    :param tokenizer: Tokenizer used to encode completions and determine padding.
    :param generation_cfg: Generation settings (max lengths, etc.).
    :param batching_cfg: Batching settings controlling scoring slice sizes.
    :returns: Prepared ``ScoreBatch`` or ``None`` when no sequences are available.
    :rtype: ScoreBatch | None
    """
    prompt_batch = getattr(reward_comp.pairs, "prompts", reward_comp.pairs.completions)
    completion_batch = reward_comp.pairs.completions
    total_sequences = len(prompt_batch)
    if total_sequences == 0:
        return None
    def _coerce_int(value: object, default: int = 0) -> int:
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default
        if isinstance(value, (str, bytes, bytearray)):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default
    prompt_length_cache_fn: Callable[[str], PromptCacheEntry]
    prompt_length_cache = getattr(batching_cfg, "prompt_length_cache_get", None)
    if not callable(prompt_length_cache) and callable(batching_cfg):
        prompt_length_cache = batching_cfg
    if callable(prompt_length_cache):
        prompt_length_cache_fn = cast(Callable[[str], PromptCacheEntry], prompt_length_cache)
    else:

        def _default_prompt_length_cache(_p: str) -> PromptCacheEntry:
            return PromptCacheEntry(input_ids=[], attention_mask=[])

        prompt_length_cache_fn = _default_prompt_length_cache
    cache_cfg = _PromptCacheConfig(
        prompt_length_cache_get=prompt_length_cache_fn,
        prompt_cache_size=int(getattr(batching_cfg, "prompt_cache_size", 0) or 0),
    )
    prompt_entries = _collect_prompt_entries(prompt_batch, cache_cfg)
    if prompt_entries is None:
        return None
    completion_tensors: Optional[CompletionTensors] = None
    completion_meta = getattr(reward_comp, "completion_metadata", None)
    if (
        completion_meta
        and isinstance(completion_meta, list)
        and len(completion_meta) == len(completion_batch)
    ):
        token_ids: List[List[int]] = []
        ok = True
        for entry in completion_meta:
            if not isinstance(entry, dict):
                ok = False
                break
            raw_ids = entry.get("token_ids")
            if raw_ids is None:
                ok = False
                break
            if hasattr(raw_ids, "tolist"):
                try:
                    raw_ids = raw_ids.tolist()
                except _SCORING_EXCEPTIONS as exc:
                    LOG.debug("Failed to convert completion metadata token_ids: %s", exc)
            if isinstance(raw_ids, list) and raw_ids and isinstance(raw_ids[0], list):
                raw_ids = raw_ids[0]
            if not isinstance(raw_ids, list):
                ok = False
                break
            try:
                token_ids.append([_coerce_int(val) for val in raw_ids])
            except (TypeError, ValueError):
                ok = False
                break
        if ok:
            pad_token_raw = tokenizer.pad_token_id
            if pad_token_raw is None:
                pad_token_raw = tokenizer.eos_token_id or 0
            pad_token_id = _coerce_int(pad_token_raw, 0)
            vocab_size = getattr(tokenizer, "vocab_size", None)
            if isinstance(vocab_size, numbers.Integral):
                vocab_size_int = int(vocab_size)
                if pad_token_id >= vocab_size_int:
                    fallback = tokenizer.eos_token_id
                    if fallback is None:
                        fallback = vocab_size_int - 1
                    pad_token_id = _coerce_int(fallback, 0)
            completion_tensors = _completion_tensors_from_token_ids(
                token_ids,
                pad_token_id=pad_token_id,
                max_length=_coerce_int(getattr(generation_cfg, "max_completion_len", 0), 0),
            )
            LOG.debug(
                "Using pre-tokenized completion token_ids from completion_metadata | sequences=%d",
                len(token_ids),
            )
    if completion_tensors is None:
        completion_tensors = _tokenize_completions(
            completion_batch,
            tokenizer,
            generation_cfg,
        )
    slice_size = (
        batching_cfg.score_slice if batching_cfg.score_slice > 0 else total_sequences
    )
    slice_size = max(1, slice_size)
    pad_token_raw = tokenizer.pad_token_id
    if pad_token_raw is None:
        pad_token_raw = tokenizer.eos_token_id or 0
    pad_token_id = _coerce_int(pad_token_raw, 0)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, numbers.Integral):
        vocab_size_int = int(vocab_size)
        if pad_token_id >= vocab_size_int:
            fallback = tokenizer.eos_token_id
            if fallback is None:
                fallback = vocab_size_int - 1
            pad_token_id = _coerce_int(fallback, 0)
    return ScoreBatch(
        prompt_entries=prompt_entries,
        completion_ids=completion_tensors.ids,
        completion_attention_mask=completion_tensors.mask,
        pad_token_id=pad_token_id,
        max_prompt_len=generation_cfg.max_prompt_len,
        slice_size=slice_size,
        total_sequences=total_sequences,
        score_tail_tokens=getattr(batching_cfg, "score_tail_tokens", None),
    )


def _apply_eos_completion_mask(
    completion_ids: Tensor,
    eos_token_id: Optional[int],
    completion_mask: Optional[Tensor] = None,
) -> Tensor:
    """Mask completion tokens after the first EOS token (TRL-style)."""
    torch_mod = _refresh_torch()
    if eos_token_id is None:
        if completion_mask is not None:
            return completion_mask
        return cast(
            Tensor,
            torch_mod.ones_like(
                completion_ids, dtype=getattr(torch_mod, "long", None)
            ),
        )
    try:
        is_eos = completion_ids == eos_token_id
        batch = int(is_eos.size(0))
        seq_len = int(is_eos.size(1))
        eos_idx = torch_mod.full(
            (batch,),
            seq_len,
            dtype=getattr(torch_mod, "long", None),
            device=getattr(completion_ids, "device", None),
        )
        any_eos = is_eos.any(dim=1)
        if bool(any_eos.any()):
            eos_pos = is_eos.int().argmax(dim=1)
            eos_idx = eos_idx.clone()
            eos_idx[any_eos] = eos_pos[any_eos]
        seq_idx = torch_mod.arange(
            seq_len, device=getattr(completion_ids, "device", None)
        ).unsqueeze(0)
        seq_idx = seq_idx.expand(batch, -1)
        mask = seq_idx <= eos_idx.unsqueeze(1)
        to_fn = getattr(mask, "to", None)
        if callable(to_fn):
            mask = to_fn(dtype=getattr(torch_mod, "long", None))
        return cast(Tensor, mask)
    except _SCORING_EXCEPTIONS:
        comp_arr = _to_numpy_array(completion_ids)
        mask_arr = np.ones_like(comp_arr, dtype=np.int64)
        for row_idx, row in enumerate(comp_arr):
            eos_positions = np.where(row == eos_token_id)[0]
            if eos_positions.size:
                first = int(eos_positions[0])
                if first + 1 < mask_arr.shape[1]:
                    mask_arr[row_idx, first + 1 :] = 0
        return cast(
            Tensor,
            torch_mod.tensor(
                mask_arr,
                dtype=getattr(torch_mod, "long", None),
                device=getattr(completion_ids, "device", None),
            ),
        )


def iter_batch_slices_trl(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    eos_token_id: Optional[int],
) -> Iterator[Tuple[Tensor, Tensor, Tensor, int]]:
    """Yield prompt+completion slices for TRL-style logprob computation."""
    torch_mod = _refresh_torch()
    state = _SliceState.from_score_batch(score_batch)
    if state.total_sequences == 0 or state.slice_size <= 0:
        return
    device = getattr(runtime, "device", None)
    as_tensor = getattr(torch_mod, "as_tensor", getattr(torch_mod, "tensor", None))
    if as_tensor is None:
        raise AttributeError("torch.as_tensor (or tensor) is required for scoring.")
    as_tensor_fn = cast(Callable[..., Tensor], as_tensor)

    def _ensure_tensor(obj: object, *, target_device: object | None = None) -> Tensor:
        is_tensor_fn = getattr(torch_mod, "is_tensor", None)
        try:
            if callable(is_tensor_fn) and is_tensor_fn(obj):
                return cast(Tensor, obj)
        except _SCORING_EXCEPTIONS:
            pass
        tensor_type = getattr(torch_mod, "Tensor", None)
        if tensor_type is not None and isinstance(obj, tensor_type):
            return cast(Tensor, obj)
        tensor_ctor = getattr(torch_mod, "tensor", None)
        if callable(tensor_ctor):
            data = getattr(obj, "arr", None)
            if data is None:
                data = obj
            try:
                return cast(
                    Tensor,
                    tensor_ctor(
                        np.asarray(data),
                        device=target_device,
                        dtype=getattr(obj, "dtype", None),
                    ),
                )
            except TypeError:
                return cast(Tensor, tensor_ctor(np.asarray(data)))
        return cast(Tensor, obj)

    def as_tensor_typed(*args: object, **kwargs: object) -> Tensor:
        return cast(Tensor, as_tensor_fn(*args, **kwargs))

    for start in range(0, state.total_sequences, state.slice_size):
        end = min(start + state.slice_size, state.total_sequences)
        prompt_slice = state.prompt_entries[start:end]
        comp_ids_slice = state.completion_ids[start:end]
        comp_mask_slice = state.completion_mask[start:end]
        if device is not None:
            try:
                comp_ids_slice = comp_ids_slice.to(device)
                comp_mask_slice = comp_mask_slice.to(device)
            except (AttributeError, TypeError, ValueError):
                pass
        batch_size = len(prompt_slice)
        if batch_size == 0:
            continue
        prompt_ids, prompt_mask, _prompt_lengths = _prepare_prompt_slice(
            prompt_slice,
            state.max_prompt_len,
            state.pad_token_id,
            comp_ids_slice.dtype,
            comp_mask_slice.dtype,
        )
        if device is not None:
            try:
                prompt_ids = prompt_ids.to(device)
                prompt_mask = prompt_mask.to(device)
            except (AttributeError, TypeError, ValueError):
                pass
        prompt_ids = as_tensor_typed(prompt_ids, device=device)
        prompt_mask = as_tensor_typed(prompt_mask, device=device)
        comp_ids_slice = as_tensor_typed(comp_ids_slice, device=device)
        comp_mask_slice = as_tensor_typed(comp_mask_slice, device=device)
        full_input_ids = torch_mod.cat([prompt_ids, comp_ids_slice], dim=1)
        completion_mask = _apply_eos_completion_mask(
            comp_ids_slice, eos_token_id, completion_mask=None
        )
        completion_mask = _ensure_tensor(
            completion_mask, target_device=getattr(comp_ids_slice, "device", None)
        )
        try:
            completion_mask = completion_mask * comp_mask_slice
        except _SCORING_EXCEPTIONS:
            comp_arr = np.asarray(getattr(comp_mask_slice, "arr", comp_mask_slice))
            eos_arr = np.asarray(getattr(completion_mask, "arr", completion_mask))
            completion_mask = as_tensor_typed(
                comp_arr * eos_arr, device=getattr(comp_mask_slice, "device", None)
            )
        full_attention_mask = torch_mod.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = int(getattr(comp_ids_slice, "shape", [0, 0])[1] or 0)
        yield full_input_ids, full_attention_mask, completion_mask, logits_to_keep


def token_counts_from_score_batch(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    batching_cfg: BatchingSettings,
) -> Tensor:
    """Compute per-sequence token counts from the score batch labels mask.

    :param score_batch: Prepared scoring batch.
    :param runtime: Runtime handles exposing device/accelerator.
    :param batching_cfg: Batching config controlling slice sizes.
    :returns: 1D tensor of token counts per sequence.
    :rtype: Tensor
    """
    torch_mod = _refresh_torch()
    tok_chunks: List[Tensor] = []
    eos_token_id = getattr(getattr(runtime, "tokenizer", None), "eos_token_id", None)
    slice_iter = iter_batch_slices(
        score_batch,
        runtime.device,
        eos_token_id=eos_token_id,
        apply_eos_mask=True,
    )
    slice_iter = _prefetch_iterator(slice_iter, getattr(batching_cfg, "slice_prefetch", 0))
    for _slice_inputs, _slice_mask, slice_labels in slice_iter:
        label_mask = slice_labels != -100
        tok = label_mask.sum(dim=1).clamp(min=1)
        to_fn = getattr(tok, "to", None)
        if callable(to_fn):
            tok = to_fn(dtype=getattr(torch_mod, "float32", None))
        tok = cast(Tensor, tok)
        tok_chunks.append(tok)
    if not tok_chunks:
        try:
            return cast(
                Tensor,
                torch_mod.zeros(
                (0,),
                dtype=getattr(torch_mod, "float32", None),
                device=runtime.device,
                ),
            )
        except _SCORING_EXCEPTIONS:
            return cast(
                Tensor,
                torch_mod.tensor([], dtype=getattr(torch_mod, "float32", None)),
            )
    try:
        out = torch_mod.cat(tok_chunks, dim=0)
    except _SCORING_EXCEPTIONS:
        out = tok_chunks[0]
        for chunk in tok_chunks[1:]:
            out = torch_mod.cat([out, chunk], dim=0)
    out_tensor: Tensor = cast(Tensor, out)
    to_fn = getattr(out_tensor, "to", None)
    if callable(to_fn):
        try:
            out_tensor = cast(Tensor, to_fn(device=runtime.device))
        except _SCORING_EXCEPTIONS as exc:
            LOG.debug("Failed to move token counts to runtime device: %s", exc)
    out_tensor = cast(Tensor, out_tensor)
    return out_tensor


def summarize_completion_lengths(
    ref_stats: ReferenceLogprobs,
    max_completion_len: int,
) -> Tuple[Tensor, LengthStats, float]:
    """Summarize completion lengths for metrics.

    :param ref_stats: Reference log-prob stats containing token counts.
    :param max_completion_len: Maximum completion length used for clipping stats.
    :returns: Tuple of ``(completion_lengths, length_stats, total_tokens)``.
    :rtype: tuple[Tensor, LengthStats, float]
    """
    torch_mod = _refresh_torch()
    lengths_arr = _to_numpy_array(ref_stats.ref_tok_counts).astype(float)
    num_completion_tokens = float(lengths_arr.sum()) if lengths_arr.size else 0.0
    clipped_mask = lengths_arr >= float(max_completion_len)
    if lengths_arr.size > 0:
        min_length = float(lengths_arr.min())
        mean_length = float(lengths_arr.mean())
        max_length = float(lengths_arr.max())
        clipped_ratio = float(clipped_mask.mean())
    else:
        min_length = mean_length = max_length = clipped_ratio = 0.0
    terminated = lengths_arr[~clipped_mask] if lengths_arr.size else np.asarray([])
    if terminated.size > 0:
        min_terminated = float(terminated.min())
        mean_terminated = float(terminated.mean())
        max_terminated = float(terminated.max())
    else:
        min_terminated = mean_terminated = max_terminated = 0.0
    completion_lengths = cast(
        Tensor,
        torch_mod.tensor(lengths_arr, dtype=getattr(torch_mod, "float32", None)),
    )
    return (
        completion_lengths,
        LengthStats(
            min_length=min_length,
            mean_length=mean_length,
            max_length=max_length,
            clipped_ratio=clipped_ratio,
            min_terminated=min_terminated,
            mean_terminated=mean_terminated,
            max_terminated=max_terminated,
        ),
        num_completion_tokens,
    )

