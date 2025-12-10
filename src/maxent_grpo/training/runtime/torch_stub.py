"""Lightweight torch stub helpers used when torch is unavailable in tests."""

from __future__ import annotations

from contextlib import nullcontext, contextmanager
from numbers import Integral
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Tuple

import numpy as np

# Expected conversion/indexing errors when emulating torch with numpy.
_NUMPY_EXCEPTIONS = (TypeError, ValueError, OverflowError, IndexError)


def _build_torch_stub() -> Any:
    """Return a lightweight, numpy-backed torch stub for test environments."""

    class _DType:
        def __init__(self, name: str, np_dtype: np.dtype):
            self.name = name
            self.np_dtype = np.dtype(np_dtype)

        def __repr__(self) -> str:  # pragma: no cover - representational only
            return f"torch.{self.name}"

    def _resolve_dtype(dtype: object | None):
        if dtype is None:
            return None
        np_dtype = getattr(dtype, "np_dtype", None)
        if np_dtype is not None:
            return np_dtype
        try:
            return np.dtype(dtype)
        except _NUMPY_EXCEPTIONS:
            return None

    def _normalize_shape(shape: object | None) -> Tuple[int, ...]:
        if shape is None:
            return (0,)
        if isinstance(shape, _Tensor):
            return tuple(shape.arr.shape)
        if isinstance(shape, (list, tuple)):
            if len(shape) == 0:
                return (0,)
            return tuple(int(s) for s in shape)
        try:
            return (int(shape),)
        except _NUMPY_EXCEPTIONS:
            try:
                return tuple(shape)  # type: ignore[arg-type]
            except _NUMPY_EXCEPTIONS:
                return (0,)

    def _is_bool_dtype(dtype: object | None) -> bool:
        np_dtype = getattr(dtype, "np_dtype", None)
        if np_dtype is None and dtype is not None:
            try:
                np_dtype = np.dtype(dtype)
            except _NUMPY_EXCEPTIONS:
                np_dtype = None
        if np_dtype is None:
            return False
        try:
            return np.issubdtype(np_dtype, np.bool_)
        except _NUMPY_EXCEPTIONS:
            return False

    class _Device:
        def __init__(self, device: str = "cpu"):
            self.type = str(device)

        def __repr__(self):
            return f"torch.device('{self.type}')"

    bool_dtype = _DType("bool", np.bool_)
    int64_dtype = _DType("int64", np.int64)
    float32_dtype = _DType("float32", np.float32)
    float64_dtype = _DType("float64", np.float64)

    class _Tensor:
        __array_priority__ = 100.0

        def __init__(self, data=None, dtype=None, requires_grad: bool = False):
            np_dtype = _resolve_dtype(dtype)
            if isinstance(data, _Tensor):
                data = data.arr
            try:
                arr = np.array([] if data is None else data, dtype=np_dtype)
            except _NUMPY_EXCEPTIONS:
                arr = np.array([] if data is None else data, dtype=np_dtype, copy=False)
            self.arr = arr
            self.dtype = dtype if dtype is not None else arr.dtype
            self.requires_grad = bool(requires_grad)
            self.grad = None

        # Container/utility helpers
        @property
        def shape(self):
            return self.arr.shape

        @property
        def ndim(self):
            return self.arr.ndim

        @property
        def device(self):
            return _Device("cpu")

        def __iter__(self):
            return iter(self.arr)

        def __len__(self):
            if self.arr.ndim == 0:
                return 0 if self.arr.size == 0 else 1
            return int(self.arr.shape[0])

        def any(self):
            try:
                return bool(np.any(self.arr))
            except _NUMPY_EXCEPTIONS:
                return False

        def numel(self):
            if self.arr.ndim <= 1:
                return int(self.arr.size)
            return int(self.arr.shape[0])

        def item(self):
            try:
                return self.arr.item()
            except _NUMPY_EXCEPTIONS:
                return self.arr.tolist() if self.arr.size else []

        def to(self, *args, **kwargs):
            dtype = kwargs.get("dtype", None)
            if len(args) >= 2 and dtype is None:
                dtype = args[1]
            np_dtype = _resolve_dtype(dtype)
            if np_dtype is None:
                return self
            return _Tensor(self.arr.astype(np_dtype), dtype=dtype)

        def cpu(self):
            return self

        def detach(self):
            return self

        def clone(self):
            return _Tensor(self.arr.copy(), dtype=self.dtype)

        def float(self):
            return self

        def long(self):
            return self

        def reshape(self, *shape):
            return _Tensor(self.arr.reshape(*shape), dtype=self.dtype)

        view = reshape

        def unsqueeze(self, dim: int):
            return _Tensor(np.expand_dims(self.arr, axis=dim), dtype=self.dtype)

        def squeeze(self, dim: int | None = None):
            if dim is None:
                return _Tensor(np.squeeze(self.arr), dtype=self.dtype)
            return _Tensor(np.squeeze(self.arr, axis=dim), dtype=self.dtype)

        def size(self, dim: int | None = None):
            if dim is None:
                return self.arr.shape
            if self.arr.ndim <= dim:
                return 0
            return int(self.arr.shape[dim])

        def masked_fill(self, mask, value):
            mask_arr = getattr(mask, "arr", None)
            if mask_arr is None:
                mask_arr = np.array(mask, dtype=bool)
            filled = np.where(mask_arr.astype(bool), value, self.arr)
            return _Tensor(filled, dtype=self.dtype)

        def masked_fill_(self, mask, value):
            """In-place variant mirroring ``torch.Tensor.masked_fill_``."""
            mask_arr = getattr(mask, "arr", None)
            if mask_arr is None:
                mask_arr = np.array(mask, dtype=bool)
            try:
                np.putmask(self.arr, mask_arr.astype(bool), value)
            except _NUMPY_EXCEPTIONS:
                pass
            return self

        def bool(self):
            return _Tensor(self.arr.astype(bool), dtype=bool_dtype)

        # Reductions
        def sum(self, dim: int | None = None):
            """Sum elements along an axis, promoting bool tensors to integer counts."""
            try:
                arr = self.arr
                result_dtype = self.dtype
                is_bool_tensor = _is_bool_dtype(result_dtype) or _is_bool_dtype(
                    getattr(arr, "dtype", None)
                )
                if is_bool_tensor:
                    arr = arr.astype(np.int64)
                    result_dtype = int64_dtype
                if dim is None:
                    result = np.sum(arr)
                else:
                    result = np.sum(arr, axis=dim)
                return _Tensor(result, dtype=result_dtype)
            except _NUMPY_EXCEPTIONS:
                return _Tensor([0.0], dtype=self.dtype)

        def mean(self, dim: int | None = None):
            if not np.issubdtype(self.arr.dtype, np.number):
                return _Tensor(np.array([0.0], dtype=float), dtype=float)
            try:
                if self.arr.size == 0:
                    return _Tensor([0.0], dtype=self.dtype)
                if dim is None:
                    return _Tensor(np.mean(self.arr), dtype=self.dtype)
                return _Tensor(np.mean(self.arr, axis=dim), dtype=self.dtype)
            except _NUMPY_EXCEPTIONS:
                return _Tensor([0.0], dtype=self.dtype)

        def std(self, unbiased: bool = False):
            if self.arr.size == 0:
                return _Tensor([0.0], dtype=self.dtype)
            if not np.issubdtype(self.arr.dtype, np.number):
                return _Tensor(np.array([0.0], dtype=float), dtype=float)
            try:
                ddof = 1 if unbiased else 0
                val = np.std(self.arr, ddof=ddof)
                if not np.isfinite(val):
                    val = 0.0
                return _Tensor(np.array(val, dtype=float), dtype=self.dtype)
            except _NUMPY_EXCEPTIONS:
                return _Tensor([0.0], dtype=self.dtype)

        def min(self, dim: int | None = None):
            try:
                if dim is None:
                    return _Tensor(
                        np.min(self.arr) if self.arr.size else np.array(0),
                        dtype=self.dtype,
                    )
                return _Tensor(np.min(self.arr, axis=dim), dtype=self.dtype)
            except _NUMPY_EXCEPTIONS:
                return _Tensor([0], dtype=self.dtype)

        def max(self, dim: int | None = None):
            try:
                if dim is None:
                    return _Tensor(
                        np.max(self.arr) if self.arr.size else np.array(0),
                        dtype=self.dtype,
                    )
                return _Tensor(np.max(self.arr, axis=dim), dtype=self.dtype)
            except _NUMPY_EXCEPTIONS:
                return _Tensor([0], dtype=self.dtype)

        def clamp(self, min_val=None, max_val=None, **kwargs):
            # Accept torch-style keyword aliases clamp(min=..., max=...)
            if "min" in kwargs and min_val is None:
                min_val = kwargs.get("min")
            if "max" in kwargs and max_val is None:
                max_val = kwargs.get("max")
            try:
                return _Tensor(
                    np.clip(self.arr, a_min=min_val, a_max=max_val), dtype=self.dtype
                )
            except _NUMPY_EXCEPTIONS:
                return _Tensor(self.arr, dtype=self.dtype)

        def fill_diagonal_(self, value):
            try:
                np.fill_diagonal(self.arr, value)
            except _NUMPY_EXCEPTIONS:
                pass
            return self

        def exp(self):
            try:
                return _Tensor(np.exp(self.arr), dtype=self.dtype)
            except _NUMPY_EXCEPTIONS:
                return _Tensor(self.arr, dtype=self.dtype)

        # Indexing
        def __getitem__(self, key):
            if isinstance(key, _Tensor):
                key = key.arr
            elif isinstance(key, tuple):
                key = tuple(k.arr if isinstance(k, _Tensor) else k for k in key)
            try:
                return _Tensor(self.arr[key], dtype=self.dtype)
            except _NUMPY_EXCEPTIONS:
                return _Tensor(self.arr, dtype=self.dtype)

        def __setitem__(self, key, value):
            val = value.arr if isinstance(value, _Tensor) else value
            # Convert tensor-based indices to numpy-friendly equivalents.
            if isinstance(key, tuple):
                key = tuple(k.arr if isinstance(k, _Tensor) else k for k in key)
            if isinstance(key, tuple) and len(key) == 2:
                row_key = key[0]
                allow_full_slice = (
                    isinstance(row_key, slice)
                    and row_key.start is None
                    and row_key.stop is None
                    and row_key.step is None
                )
                if not isinstance(row_key, Integral) and not allow_full_slice:
                    raise TypeError("Row indices must be integers for tensor writes")
            try:
                self.arr[key] = val
                return
            except _NUMPY_EXCEPTIONS:
                pass
            # Legacy fallback for environments where numpy indexing still fails.
            if isinstance(key, tuple) and len(key) == 2:
                row, col = key
                allow_full_slice = (
                    isinstance(row, slice)
                    and row.start is None
                    and row.stop is None
                    and row.step is None
                )
                if not isinstance(row, Integral) and not allow_full_slice:
                    raise TypeError("Row indices must be integers for tensor writes")
                if isinstance(row, slice):
                    row_indices = list(range(*row.indices(self.arr.shape[0])))
                else:
                    row_indices = [int(row)]
                data = self.arr.tolist()
                for r in row_indices:
                    while len(data) <= r:
                        data.append([])
                    row_data = list(data[r])
                    if isinstance(col, slice):
                        start, stop, step = col.indices(
                            max(len(row_data), col.stop or 0)
                        )
                        if stop > len(row_data):
                            row_data.extend([0] * (stop - len(row_data)))
                        payload = val
                        if isinstance(payload, np.ndarray):
                            payload = payload.tolist()
                        if isinstance(payload, Iterable) and not isinstance(
                            payload, (str, bytes)
                        ):
                            row_data[start:stop:step] = list(payload)
                        else:
                            row_data[start:stop:step] = [payload] * len(
                                range(start, stop, step)
                            )
                    else:
                        while len(row_data) <= int(col):
                            row_data.append(0)
                        row_data[int(col)] = val
                    data[r] = row_data
                try:
                    self.arr = np.array(data, dtype=_resolve_dtype(self.dtype))
                except _NUMPY_EXCEPTIONS:
                    self.arr = np.array(data, dtype=object)
                return
            self.arr[key] = val

        # Array adaptors
        def __array__(self, dtype=None):
            if dtype is None:
                return np.array(self.arr)
            return np.array(self.arr, dtype=dtype)

        def tolist(self):
            res = self.arr.tolist()
            return res if isinstance(res, list) else [res]

        def __float__(self):
            try:
                return float(self.arr)
            except _NUMPY_EXCEPTIONS:
                return float(self.arr.reshape(-1)[0]) if self.arr.size else 0.0

        # Binary ops
        def _binary(self, other, op):
            other_arr = other.arr if isinstance(other, _Tensor) else other
            try:
                return _Tensor(op(self.arr, other_arr), dtype=self.dtype)
            except _NUMPY_EXCEPTIONS:
                try:
                    return _Tensor(op(self.arr, np.array(other_arr)), dtype=self.dtype)
                except _NUMPY_EXCEPTIONS:
                    return _Tensor(self.arr, dtype=self.dtype)

        def __add__(self, other):
            return self._binary(other, np.add)

        def __sub__(self, other):
            return self._binary(other, np.subtract)

        def __rsub__(self, other):
            return _Tensor(other)._binary(self, np.subtract)

        def __truediv__(self, other):
            other_arr = other.arr if isinstance(other, _Tensor) else other
            return _Tensor(np.divide(self.arr, other_arr), dtype=float)

        def __rtruediv__(self, other):
            return _Tensor(np.divide(other, self.arr), dtype=float)

        def __mul__(self, other):
            return self._binary(other, np.multiply)

        __rmul__ = __mul__

        def __matmul__(self, other):
            return self._binary(other, np.matmul)

        def __neg__(self):
            return _Tensor(-self.arr, dtype=self.dtype)

        def __invert__(self):
            try:
                return _Tensor(np.logical_not(self.arr.astype(bool)), dtype=bool_dtype)
            except _NUMPY_EXCEPTIONS:
                return _Tensor([], dtype=bool_dtype)

        # Comparisons / logic
        def _binary_bool(self, other, op):
            other_arr = other.arr if isinstance(other, _Tensor) else other
            try:
                return _Tensor(op(self.arr, other_arr), dtype=bool_dtype)
            except _NUMPY_EXCEPTIONS:
                try:
                    return _Tensor(op(self.arr, np.array(other_arr)), dtype=bool_dtype)
                except _NUMPY_EXCEPTIONS:
                    return _Tensor([], dtype=bool_dtype)

        def __eq__(self, other):
            return self._binary_bool(other, np.equal)

        def __ne__(self, other):
            return self._binary_bool(other, np.not_equal)

        def __ge__(self, other):
            return self._binary_bool(other, np.greater_equal)

        def __lt__(self, other):
            return self._binary_bool(other, np.less)

        def __gt__(self, other):
            return self._binary_bool(other, np.greater)

        def ge(self, other):
            return self.__ge__(other)

        def ne(self, other):
            return self.__ne__(other)

        def __and__(self, other):
            return self._binary_bool(other, np.logical_and)

        def __rand__(self, other):
            return self._binary_bool(other, np.logical_and)

        def __or__(self, other):
            return self._binary_bool(other, np.logical_or)

        def __ror__(self, other):
            return self._binary_bool(other, np.logical_or)

        def __xor__(self, other):
            return self._binary_bool(other, np.logical_xor)

        def __rxor__(self, other):
            return self._binary_bool(other, np.logical_xor)

    # Tensor factories
    def _tensor(data=None, dtype=None, device=None, requires_grad: bool = False):
        del device
        if isinstance(data, _Tensor):
            return data
        return _Tensor(data, dtype=dtype, requires_grad=requires_grad)

    def _zeros(shape, dtype=None, device=None):
        del device
        norm = _normalize_shape(shape)
        return _Tensor(np.zeros(norm, dtype=_resolve_dtype(dtype)), dtype=dtype)

    def _ones(shape, dtype=None, device=None):
        del device
        norm = _normalize_shape(shape)
        return _Tensor(np.ones(norm, dtype=_resolve_dtype(dtype)), dtype=dtype)

    def _full(shape, fill_value, dtype=None, device=None):
        del device
        norm = _normalize_shape(shape)
        return _Tensor(
            np.full(norm, fill_value, dtype=_resolve_dtype(dtype)), dtype=dtype
        )

    def _empty(shape, dtype=None, device=None):
        del device
        norm = _normalize_shape(shape)
        return _Tensor(np.empty(norm, dtype=_resolve_dtype(dtype)), dtype=dtype)

    def _zeros_like(arr, dtype=None):
        if arr is None:
            return _Tensor([])
        data = arr.arr if isinstance(arr, _Tensor) else arr
        return _Tensor(np.zeros_like(data, dtype=_resolve_dtype(dtype)), dtype=dtype)

    def _ones_like(arr, dtype=None):
        if arr is None:
            return _Tensor([])
        data = arr.arr if isinstance(arr, _Tensor) else arr
        return _Tensor(np.ones_like(data, dtype=_resolve_dtype(dtype)), dtype=dtype)

    def _full_like(arr, fill_value):
        data = arr.arr if isinstance(arr, _Tensor) else arr
        return _Tensor(
            np.full_like(data, fill_value), dtype=getattr(arr, "dtype", None)
        )

    def _arange(end, dtype=None):
        return _Tensor(np.arange(end, dtype=_resolve_dtype(dtype)), dtype=dtype)

    def _cat(seq, dim: int = 0):
        arrays = []
        for item in seq:
            if item is None:
                continue
            if isinstance(item, _Tensor):
                arrays.append(item.arr)
            else:
                arrays.append(np.array(item))
        if not arrays:
            return _Tensor([], dtype=None)
        try:
            return _Tensor(np.concatenate(arrays, axis=dim))
        except _NUMPY_EXCEPTIONS:
            flat = []
            for arr in arrays:
                flat.extend(arr.tolist() if hasattr(arr, "tolist") else list(arr))
            return _Tensor(flat)

    def _stack(tensors, dim: int = 0):
        arrays = [t.arr if isinstance(t, _Tensor) else np.array(t) for t in tensors]
        return _Tensor(np.stack(arrays, axis=dim)) if arrays else _Tensor([])

    def _all(x):
        data = x.arr if isinstance(x, _Tensor) else x
        return bool(np.all(data))

    def _where(cond, x, y):
        cond_arr = cond.arr if isinstance(cond, _Tensor) else cond
        x_arr = x.arr if isinstance(x, _Tensor) else x
        y_arr = y.arr if isinstance(y, _Tensor) else y
        return _Tensor(np.where(cond_arr, x_arr, y_arr))

    def _minimum(a, b):
        a_arr = a.arr if isinstance(a, _Tensor) else a
        b_arr = b.arr if isinstance(b, _Tensor) else b
        return _Tensor(np.minimum(a_arr, b_arr))

    def _maximum(a, b):
        a_arr = a.arr if isinstance(a, _Tensor) else a
        b_arr = b.arr if isinstance(b, _Tensor) else b
        return _Tensor(np.maximum(a_arr, b_arr))

    def _log_fn(x):
        data = x.arr if isinstance(x, _Tensor) else x
        try:
            return _Tensor(np.log(data), dtype=getattr(x, "dtype", None))
        except _NUMPY_EXCEPTIONS:
            return _Tensor(data, dtype=getattr(x, "dtype", None))

    def _softmax(tensor, dim: int = 0):
        vals = tensor.arr if isinstance(tensor, _Tensor) else np.array(tensor)
        max_val = np.max(vals, axis=dim, keepdims=True) if vals.size else 0.0
        exps = np.exp(vals - max_val)
        denom = np.sum(exps, axis=dim, keepdims=True)
        return _Tensor(
            exps / np.maximum(denom, 1e-12), dtype=getattr(tensor, "dtype", None)
        )

    def _log_softmax(tensor, dim: int = -1):
        soft = _softmax(tensor, dim=dim)
        safe = np.clip(soft.arr, 1e-12, None)
        return _Tensor(np.log(safe), dtype=getattr(tensor, "dtype", None))

    def _pdist(tensor, p: int = 2):
        arr = tensor.arr if isinstance(tensor, _Tensor) else np.array(tensor)
        if arr.ndim != 2:
            return _Tensor([], dtype=getattr(tensor, "dtype", None))
        diff = arr[:, None, :] - arr[None, :, :]
        dist = np.power(np.abs(diff), p).sum(axis=-1) ** (1.0 / p)
        idx = np.triu_indices(dist.shape[0], k=1)
        return _Tensor(dist[idx], dtype=getattr(tensor, "dtype", None))

    def _unique(x):
        arr = x.arr if isinstance(x, _Tensor) else np.array(x)
        return _Tensor(np.unique(arr), dtype=getattr(x, "dtype", None))

    def _randn(*shape, requires_grad: bool = False):
        return _Tensor(
            np.random.standard_normal(size=shape),
            dtype=np.float32,
            requires_grad=requires_grad,
        )

    @contextmanager
    def _no_grad():
        yield

    def _autocast(**_kwargs):
        return nullcontext()

    torch_mod = SimpleNamespace()
    torch_mod.Tensor = _Tensor
    torch_mod.tensor = _tensor
    torch_mod.zeros = _zeros
    torch_mod.ones = _ones
    torch_mod.full = _full
    torch_mod.empty = _empty
    torch_mod.randn = _randn
    torch_mod.arange = _arange
    torch_mod.cat = _cat
    torch_mod.stack = _stack
    torch_mod.matmul = lambda a, b: _Tensor(np.matmul(_tensor(a).arr, _tensor(b).arr))
    torch_mod.size = lambda arr, dim=None: (
        0
        if arr is None
        else (
            len(arr)
            if dim is None
            else (len(arr[dim]) if hasattr(arr, "__getitem__") else 0)
        )
    )
    torch_mod.zeros_like = _zeros_like
    torch_mod.ones_like = _ones_like
    torch_mod.full_like = _full_like
    torch_mod.all = _all
    torch_mod.where = _where
    torch_mod.minimum = _minimum
    torch_mod.maximum = _maximum
    torch_mod.unique = _unique
    torch_mod.softmax = _softmax
    torch_mod.log = _log_fn
    torch_mod.clamp = lambda inp, min_val=None, max_val=None: _Tensor(
        np.clip(_tensor(inp).arr, a_min=min_val, a_max=max_val)
    )
    # Persistence helpers (minimal stubs so tests can monkeypatch torch.save).
    def _save(obj, path):
        try:  # pragma: no cover - side-effect only
            with open(path, "wb") as handle:
                # Write a small marker; callers in tests overwrite via monkeypatch.
                handle.write(b"stub")
        except Exception:
            # Silently ignore I/O failures in stub environments.
            return None

    torch_mod.save = _save
    torch_mod.no_grad = _no_grad
    torch_mod.autocast = _autocast
    torch_mod.float32 = float32_dtype
    torch_mod.float64 = float64_dtype
    torch_mod.int64 = int64_dtype
    torch_mod.long = torch_mod.int64
    torch_mod.bool = bool_dtype
    torch_mod.dtype = type("dtype", (), {})  # pragma: no cover - placeholder
    torch_mod.device = lambda name="cpu": _Device(name)
    torch_mod.is_grad_enabled = lambda: True
    torch_mod.manual_seed = lambda seed=None: np.random.seed(int(seed or 0))
    torch_mod.autograd = SimpleNamespace(no_grad=_no_grad)

    class _SymBool:
        def __init__(self, value=False):
            self.value = bool(value)

        def __bool__(self):
            return self.value

    torch_mod.SymBool = _SymBool

    # cuda namespace stub
    torch_mod.cuda = SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *_a, **_k: None,
        _is_in_bad_fork=lambda: False,
    )
    torch_mod.xpu = SimpleNamespace(
        is_available=lambda: False,
        manual_seed=lambda *_a, **_k: None,
        manual_seed_all=lambda *_a, **_k: None,
        _is_in_bad_fork=lambda: False,
    )

    # nn + optim namespaces
    torch_mod.nn = SimpleNamespace(
        Module=type("Module", (), {}),
        Parameter=type("Parameter", (), {}),
        functional=SimpleNamespace(
            log_softmax=lambda *a, **k: _log_softmax(
                a[0], dim=k.get("dim", -1) if a else -1
            )
            if a
            else _Tensor([]),
            pdist=_pdist,
        ),
        Linear=lambda *_a, **_k: (lambda x: _Tensor([])),
        Embedding=lambda *_a, **_k: SimpleNamespace(
            weight=_Tensor([]), __call__=lambda ids: _Tensor([])
        ),
    )
    torch_mod.backends = SimpleNamespace()
    torch_mod.mps = SimpleNamespace(
        is_available=lambda: False,
        is_built=False,
        _is_in_bad_fork=lambda: False,
        manual_seed=lambda *_a, **_k: None,
    )
    torch_mod.backends.mps = torch_mod.mps
    torch_mod.optim = SimpleNamespace(
        Optimizer=type("Optimizer", (), {}),
        AdamW=lambda params=None, lr=1e-3: SimpleNamespace(
            param_groups=[{"lr": lr}],
            step=lambda: None,
            zero_grad=lambda set_to_none=True: None,
            parameters=lambda: params,
        ),
    )

    # _dynamo shim
    dynamo_mod = sys.modules.get("torch._dynamo")
    if dynamo_mod is None:
        dynamo_mod = ModuleType("torch._dynamo")
        dynamo_mod.disable = lambda fn=None, recursive=False: fn
        dynamo_mod.graph_break = lambda: None
        dynamo_mod.__spec__ = SimpleNamespace()
        sys.modules["torch._dynamo"] = dynamo_mod
    setattr(torch_mod, "_dynamo", dynamo_mod)

    # utils.data stubs
    utils_mod = getattr(torch_mod, "utils", None)
    if utils_mod is None:
        utils_mod = SimpleNamespace()
        torch_mod.utils = utils_mod
    data_mod = getattr(utils_mod, "data", None)
    if data_mod is None:
        data_mod = SimpleNamespace()
        utils_mod.data = data_mod
    if getattr(utils_mod, "__spec__", None) is None:
        utils_mod.__spec__ = SimpleNamespace()
    if getattr(data_mod, "__spec__", None) is None:
        data_mod.__spec__ = SimpleNamespace()

    class DataLoader:
        def __iter__(self):
            return iter([])

    data_mod.DataLoader = DataLoader

    # Register stub modules for import resolution
    torch_mod.__spec__ = SimpleNamespace()
    torch_mod.__path__ = []
    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch.cuda", torch_mod.cuda)
    sys.modules.setdefault("torch.xpu", torch_mod.xpu)
    sys.modules.setdefault("torch.nn", torch_mod.nn)
    sys.modules.setdefault("torch.nn.functional", torch_mod.nn.functional)
    sys.modules.setdefault("torch.optim", torch_mod.optim)
    sys.modules.setdefault("torch.utils", torch_mod.utils)
    sys.modules.setdefault("torch.utils.data", data_mod)
    sys.modules.setdefault("torch.mps", torch_mod.mps)
    # Register nested namespace stubs to placate imports triggered during backward passes.
    nested_mod = ModuleType("torch.nested")
    nested_mod.__spec__ = SimpleNamespace()
    nested_internal = ModuleType("torch.nested._internal")
    nested_internal.__spec__ = SimpleNamespace()
    nested_tensor_mod = ModuleType("torch.nested._internal.nested_tensor")
    nested_tensor_mod.__spec__ = SimpleNamespace()
    nested_tensor_mod.NestedTensor = type("NestedTensor", (), {})
    sys.modules.setdefault("torch.nested", nested_mod)
    sys.modules.setdefault("torch.nested._internal", nested_internal)
    sys.modules.setdefault("torch.nested._internal.nested_tensor", nested_tensor_mod)

    return torch_mod


__all__ = ["_build_torch_stub"]
