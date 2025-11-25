"""Lightweight torch stub helpers used when torch is unavailable in tests."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any


def _build_torch_stub() -> Any:
    """Return a lightweight torch stub for test environments."""

    class _Tensor:
        def __init__(self, data=None, dtype=None):
            self.data = list(data) if data is not None else []
            self.dtype = dtype

        def __iter__(self):
            return iter(self.data) if self.data is not None else iter([])

        def __len__(self):
            return len(self.data) if self.data is not None else 0

        @property
        def shape(self):
            if self.data and hasattr(self.data[0], "__len__"):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)

        def __getitem__(self, key):
            if isinstance(key, tuple) and len(key) == 2:
                rows, cols = key
                selected = []
                row_indices = (
                    range(len(self.data)) if isinstance(rows, slice) else [rows]
                )
                for r in row_indices:
                    row_val = self.data[r]
                    if isinstance(row_val, list):
                        selected.append(row_val[cols])
                return _Tensor(selected, self.dtype)
            return _Tensor(self.data[key], self.dtype)

        def __setitem__(self, key, value):
            val = value.data if isinstance(value, _Tensor) else value
            if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], slice):
                row, sl = key
                if isinstance(row, slice):
                    raise TypeError("Row slice assignment is not supported in the stub")
                if isinstance(row, int):
                    while len(self.data) <= row:
                        self.data.append([])
                    row_data = list(self.data[row])
                    start, stop, step = sl.indices(len(row_data))
                    if isinstance(val, list):
                        row_data[start:stop:step] = val
                    else:
                        row_data[start:stop:step] = [val] * len(
                            range(start, stop, step)
                        )
                    self.data[row] = row_data
                else:
                    self.data[key] = val
            else:
                self.data[key] = val

        def tolist(self):
            return list(self.data)

        def _identity(self):
            return self

        long = _identity
        float = _identity

        def numel(self):
            return len(self.data)

        def sum(self, dim=None):
            if dim is None:
                return _Tensor([sum(self.data)], self.dtype)
            if self.data and isinstance(self.data[0], list):
                return _Tensor([sum(row) for row in self.data], self.dtype)
            return _Tensor([sum(self.data)], self.dtype)

        def std(self, unbiased=False):
            import math

            count = len(self.data)
            if count == 0:
                return _Tensor([0.0], self.dtype)
            try:
                mean_val = sum(self.data) / count
                var = sum((x - mean_val) ** 2 for x in self.data)
                denom = count - 1 if unbiased and count > 1 else count
                return _Tensor([math.sqrt(var / denom)], self.dtype)
            except (TypeError, ValueError, ZeroDivisionError):
                return _Tensor([0.0], self.dtype)

        def mean(self):
            count = len(self.data)
            if count == 0:
                return _Tensor([0.0], self.dtype)
            try:
                return _Tensor([sum(self.data) / count], self.dtype)
            except (TypeError, ValueError):
                return _Tensor([0.0], self.dtype)

        def item(self):
            try:
                return self.data[0]
            except (IndexError, TypeError):
                return self.data

        def _binary(self, other, op):
            other_data = other.data if isinstance(other, _Tensor) else other
            if (
                isinstance(self.data, list)
                and self.data
                and isinstance(self.data[0], list)
            ):
                res = [
                    [
                        op(a, b if isinstance(other_data, list) else other_data)
                        for a, b in zip(
                            row,
                            (
                                other_data[row_idx]
                                if isinstance(other_data, list)
                                else [other_data] * len(row)
                            ),
                        )
                    ]
                    for row_idx, row in enumerate(self.data)
                ]
            else:
                if isinstance(other_data, list):
                    res = [op(a, b) for a, b in zip(self.data, other_data)]
                else:
                    res = [op(a, other_data) for a in self.data]
            return _Tensor(res, self.dtype)

        def __add__(self, other):
            return self._binary(other, lambda a, b: a + b)

        def __sub__(self, other):
            return self._binary(other, lambda a, b: a - b)

        def __rsub__(self, other):
            return _Tensor([other], self.dtype)._binary(self, lambda a, b: a - b)

        def __truediv__(self, other):
            return self._binary(other, lambda a, b: a / b)

        cpu = _identity

        def __eq__(self, other):
            return self._binary(other, lambda a, b: a == b)

        def __ge__(self, other):
            return self._binary(other, lambda a, b: a >= b)

    def _tensor(data=None, dtype=None, device=None):
        del device
        return _Tensor(data, dtype)

    def _zeros(shape, *_args, **_kwargs):
        n = int(shape[0]) if shape else 0
        if len(shape) > 1:
            m = int(shape[1])
            return _Tensor([[0] * m for _ in range(n)])
        return _Tensor([0] * n)

    def _ones_like(arr, *_args, **_kwargs):
        try:
            return _Tensor([1 for _ in arr])
        except (TypeError, ValueError):
            return _Tensor([])

    def _full(shape, fill_value, *_args, **_kwargs):
        n = int(shape[0]) if shape else 0
        if len(shape) > 1:
            m = int(shape[1])
            return _Tensor([[fill_value] * m for _ in range(n)])
        return _Tensor([fill_value] * n)

    def _cat(seq, dim=0):
        del dim
        out: list[Any] = []
        for item in seq:
            if isinstance(item, _Tensor):
                out.extend(item.data)
            elif isinstance(item, list):
                out.extend(item)
        return _Tensor(out)

    def _size(arr, dim=None):
        try:
            return len(arr) if dim is None else len(arr[dim])
        except (TypeError, AttributeError):
            return 0

    def _to(self, *_args, **_kwargs):
        return self

    def _autocast(**_kwargs):
        return nullcontext()

    class _SymBool:
        def __init__(self, value=True):
            self.value = bool(value)
            self.node = None

        def __bool__(self):
            return self.value

    stub = SimpleNamespace(
        Tensor=_Tensor,
        tensor=_tensor,
        full=_full,
        ones_like=_ones_like,
        zeros=_zeros,
        cat=_cat,
        size=_size,
        autocast=_autocast,
        SymBool=_SymBool,
    )

    def _device(*args, **kwargs):
        del kwargs
        return SimpleNamespace(type=str(args[0]) if args else "cpu")

    stub.device = _device
    stub.nn = SimpleNamespace(
        functional=SimpleNamespace(log_softmax=lambda *a, **k: None)
    )

    def _no_grad():
        return nullcontext()

    stub.autograd = SimpleNamespace(no_grad=_no_grad)
    stub.no_grad = _no_grad

    def _all(x):
        data = x.data if isinstance(x, _Tensor) else x

        def _flatten(val):
            for item in val:
                if isinstance(item, list):
                    yield from _flatten(item)
                else:
                    yield item

        return all(_flatten(data))

    stub.all = _all
    stub.ones = _ones_like
    stub.zeros_like = _ones_like
    stub.long = int
    stub.float32 = float
    stub.int64 = int
    _Tensor.to = _to
    _Tensor.detach = lambda self: self
    _Tensor.clone = lambda self: _Tensor(list(self.data), self.dtype)
    _Tensor.size = lambda self, dim=None: _size(self.data, dim)

    def _tensor_clamp(self, min_val=None, max_val=None, **_kwargs):
        # Accept both keyword styles while avoiding built-in shadowing.
        if "max" in _kwargs and max_val is None:
            max_val = _kwargs.pop("max")
        if "min" in _kwargs and min_val is None:
            min_val = _kwargs.pop("min")
        result = []
        for v in self.data:
            val = v
            if min_val is not None and val < min_val:
                val = min_val
            if max_val is not None and val > max_val:
                val = max_val
            result.append(val)
        return _Tensor(result, self.dtype)

    _Tensor.clamp = _tensor_clamp
    return stub


__all__ = ["_build_torch_stub"]
