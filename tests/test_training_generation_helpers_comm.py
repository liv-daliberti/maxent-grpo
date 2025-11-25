"""
Communication helper coverage for training.generation.helpers.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace


import maxent_grpo.training.generation.helpers as helpers


def test_optional_import_returns_none_on_import_error(monkeypatch):
    monkeypatch.setattr(
        helpers.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    assert helpers._optional_import("does.not.exist") is None


def test_zero3_gather_factory_defaults_to_nullcontext(monkeypatch):
    accel = SimpleNamespace(state=None)
    factory = helpers._zero3_gather_factory(accel)
    assert isinstance(factory([]), nullcontext)


def test_gather_object_list_prefers_accelerator(monkeypatch):
    accel = SimpleNamespace(gather_object=lambda value: ["accel", value])
    out = helpers._gather_object_list(accel, ["x"])
    assert out == ["accel", ["x"]]


def test_gather_broadcast_scatter_fall_back_to_dist(monkeypatch):
    class _Dist:
        def __init__(self):
            self.broadcast_called = False
            self.scatter_payload = None

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        def all_gather_object(self, out, value):
            out[0] = value
            out[1] = ["peer"]

        def broadcast_object_list(self, payload, src=0):
            self.broadcast_called = True
            return payload

        def scatter_object_list(self, output, input_list, src=0):
            self.scatter_payload = input_list
            output[0] = input_list[0]

    dist = _Dist()
    torch_stub = SimpleNamespace(distributed=dist)
    monkeypatch.setattr(helpers, "torch", torch_stub)
    monkeypatch.setattr(helpers, "dist", dist)

    # gather_object_list path (dist)
    accel = SimpleNamespace(gather_object=None)
    gathered = helpers._gather_object_list(accel, ["me"])
    assert gathered == [["me"], ["peer"]]

    # broadcast_object_list path (dist)
    accel_b = SimpleNamespace(num_processes=2, process_index=1, broadcast_object=None)
    helpers._broadcast_object_list(accel_b, ["p1", "p2"], src=0)
    assert dist.broadcast_called is True

    # scatter_object path (dist)
    accel_s = SimpleNamespace(num_processes=2, process_index=0, scatter_object=None)
    scattered = helpers._scatter_object(accel_s, [["a"], ["b"]], src=0)
    assert scattered == ["a"]
    assert dist.scatter_payload == [["a"], ["b"]]


def test_broadcast_and_scatter_fallback_to_accelerator(monkeypatch):
    accel = SimpleNamespace(
        num_processes=2,
        process_index=1,
        broadcast_object=lambda payload, src=0: payload,
        scatter_object=lambda payload, src=0: payload[1] if payload else None,
    )
    assert helpers._broadcast_object_list(accel, ["a", "b"], src=0) is None
    assert helpers._scatter_object(accel, [["a"], ["b"]], src=0) is None


def test_broadcast_and_scatter_single_process_fallback():
    accel = SimpleNamespace(
        num_processes=1, process_index=0, broadcast_object=None, scatter_object=None
    )
    assert helpers._broadcast_object_list(accel, ["x"], src=0) is None
    assert helpers._scatter_object(accel, ["x", "y"], src=0) == "x"
