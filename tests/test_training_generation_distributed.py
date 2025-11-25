"""Coverage for training.generation.distributed collectives."""

from __future__ import annotations

from types import SimpleNamespace


import maxent_grpo.training.generation.distributed as distributed


def test_gather_object_list_prefers_accelerator(monkeypatch):
    accel = SimpleNamespace(gather_object=lambda value: ["accel", value])
    monkeypatch.setattr(distributed, "dist", None)

    assert distributed._gather_object_list(accel, ["payload"]) == ["accel", ["payload"]]


def test_collectives_use_dist_when_initialized(monkeypatch):
    class _Dist:
        def __init__(self):
            self.broadcast_called = False
            self.broadcast_payload = None
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

        def all_gather_object(self, output, value):
            output[0] = value
            output[1] = ["remote"]

        def broadcast_object_list(self, payload, src=0):
            self.broadcast_called = True
            self.broadcast_payload = (list(payload), src)

        def scatter_object_list(self, output, input_list, src=0):
            self.scatter_payload = input_list
            if input_list is not None:
                output[0] = input_list[src]

    dist = _Dist()
    monkeypatch.setattr(distributed, "dist", dist)

    accel = SimpleNamespace(gather_object=None)
    gathered = distributed._gather_object_list(accel, ["local"])
    assert gathered == [["local"], ["remote"]]

    accel_b = SimpleNamespace(
        num_processes=2,
        process_index=1,
        broadcast_object_list=None,
    )
    distributed._broadcast_object_list(accel_b, ["a", "b"], src=0)
    assert dist.broadcast_called is True
    assert dist.broadcast_payload == (["a", "b"], 0)

    accel_s = SimpleNamespace(num_processes=2, process_index=0, scatter_object=None)
    scattered = distributed._scatter_object(accel_s, [["a"], ["b"]], src=0)
    assert scattered == ["a"]
    assert dist.scatter_payload == [["a"], ["b"]]


def test_broadcast_prefers_accelerator(monkeypatch):
    called = {}
    accel = SimpleNamespace(
        broadcast_object_list=lambda payload, src=0: called.setdefault(
            "payload", (list(payload), src)
        )
    )
    monkeypatch.setattr(distributed, "dist", None)

    distributed._broadcast_object_list(accel, ["hello"], src=1)
    assert called["payload"] == (["hello"], 1)


def test_scatter_prefers_accelerator(monkeypatch):
    class _Dist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        def scatter_object_list(self, *_a, **_k):  # pragma: no cover - guard rail
            raise AssertionError("scatter should rely on accelerator")

    accel = SimpleNamespace(
        num_processes=2,
        process_index=0,
        scatter_object=lambda payload, src=0: ("accel", payload, src),
    )
    monkeypatch.setattr(distributed, "dist", _Dist())

    result = distributed._scatter_object(accel, [["x"], ["y"]], src=0)
    assert result == ("accel", [["x"], ["y"]], 0)


def test_scatter_returns_none_when_no_input(monkeypatch):
    monkeypatch.setattr(distributed, "dist", None)
    accel = SimpleNamespace(num_processes=2, process_index=1, scatter_object=None)

    assert distributed._scatter_object(accel, None, src=0) is None


def test_scatter_single_process_and_index_fallback(monkeypatch):
    monkeypatch.setattr(distributed, "dist", None)

    accel_single = SimpleNamespace(num_processes=1, process_index=0)
    assert distributed._scatter_object(accel_single, None) is None
    assert distributed._scatter_object(accel_single, ["only"]) == "only"

    accel_multi = SimpleNamespace(num_processes=2, process_index=1, scatter_object=None)
    assert distributed._scatter_object(accel_multi, ["r0", "r1"], src=0) == "r1"
