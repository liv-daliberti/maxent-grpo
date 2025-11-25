"""Unit tests for distributed generation helpers covering fallback branches."""

from types import SimpleNamespace


def test_gather_object_prefers_accelerate(monkeypatch):
    from maxent_grpo.training.generation import distributed

    calls = []

    def _gather(val):
        calls.append(val)
        return [["a"], ["b"]]

    accelerator = SimpleNamespace(gather_object=_gather)
    monkeypatch.setattr(distributed, "dist", None)
    result = distributed._gather_object_list(accelerator, ["x"])
    assert result == [["a"], ["b"]]
    assert calls == [["x"]]


def test_gather_object_uses_dist_when_available(monkeypatch):
    from maxent_grpo.training.generation import distributed

    class _Dist:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        def all_gather_object(self, output_list, input_obj):
            self.calls += 1
            for idx in range(len(output_list)):
                output_list[idx] = [input_obj, idx]

    dist_stub = _Dist()
    monkeypatch.setattr(distributed, "dist", dist_stub)
    accelerator = SimpleNamespace(gather_object=None)
    result = distributed._gather_object_list(accelerator, ["p"])
    assert result == [[["p"], 0], [["p"], 1]]
    assert dist_stub.calls == 1


def test_gather_object_fallback_single(monkeypatch):
    from maxent_grpo.training.generation import distributed

    accelerator = SimpleNamespace(gather_object=None)
    monkeypatch.setattr(distributed, "dist", None)
    val = ["solo"]
    assert distributed._gather_object_list(accelerator, val) == [val]


def test_broadcast_prefers_accelerate(monkeypatch):
    from maxent_grpo.training.generation import distributed

    calls = []

    def _broadcast(payload, src=0):
        calls.append((list(payload), src))

    accelerator = SimpleNamespace(broadcast_object_list=_broadcast)
    payload = ["a"]
    monkeypatch.setattr(distributed, "dist", None)
    distributed._broadcast_object_list(accelerator, payload, src=1)
    assert calls == [(["a"], 1)]


def test_broadcast_uses_dist(monkeypatch):
    from maxent_grpo.training.generation import distributed

    class _Dist:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        def broadcast_object_list(self, payload, src=0):
            self.calls += 1
            payload.append(src)

    dist_stub = _Dist()
    monkeypatch.setattr(distributed, "dist", dist_stub)
    accelerator = SimpleNamespace(broadcast_object_list=None)
    payload = ["b"]
    distributed._broadcast_object_list(accelerator, payload, src=2)
    assert dist_stub.calls == 1
    assert payload[-1] == 2


def test_scatter_single_process(monkeypatch):
    from maxent_grpo.training.generation import distributed

    accelerator = SimpleNamespace(num_processes=1, process_index=0)
    monkeypatch.setattr(distributed, "dist", None)
    assert distributed._scatter_object(accelerator, ["x", "y"]) == "x"
    assert distributed._scatter_object(accelerator, None) is None


def test_scatter_prefers_accelerate(monkeypatch):
    from maxent_grpo.training.generation import distributed

    calls = []

    def _scatter(payload, src=0):
        calls.append(payload)
        return "from-accelerate"

    accelerator = SimpleNamespace(
        num_processes=2, process_index=1, scatter_object=_scatter
    )
    monkeypatch.setattr(distributed, "dist", None)
    result = distributed._scatter_object(accelerator, ["a", "b"], src=0)
    # Non-src rank should pass None into scatter_object
    assert calls == [None]
    assert result == "from-accelerate"


def test_scatter_uses_dist(monkeypatch):
    from maxent_grpo.training.generation import distributed

    class _Dist:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        def scatter_object_list(self, output, payload, src=0):
            self.calls += 1
            output[0] = "from-dist" if payload is None else payload[0]

    dist_stub = _Dist()
    monkeypatch.setattr(distributed, "dist", dist_stub)
    accelerator = SimpleNamespace(num_processes=2, process_index=1, scatter_object=None)
    result = distributed._scatter_object(accelerator, ["c", "d"], src=0)
    assert dist_stub.calls == 1
    assert result == "from-dist"


def test_scatter_fallback_local_selection(monkeypatch):
    from maxent_grpo.training.generation import distributed

    accelerator = SimpleNamespace(num_processes=3, process_index=2, scatter_object=None)
    monkeypatch.setattr(distributed, "dist", None)
    assert distributed._scatter_object(accelerator, ["p0", "p1", "p2"], src=0) == "p2"
