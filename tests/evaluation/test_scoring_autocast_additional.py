"""Additional coverage for scoring autocast helpers."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace

from maxent_grpo.training import scoring


def test_autocast_context_returns_null_without_autocast(monkeypatch):
    accel = SimpleNamespace(autocast=None)
    device = SimpleNamespace(type=None)
    torch_stub = SimpleNamespace(autocast=None)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    ctx = scoring._autocast_context(accel, device)
    # When torch.autocast is None, we should fall back to nullcontext
    assert isinstance(ctx, nullcontext)


def test_autocast_context_swallows_type_error(monkeypatch):
    """Exercise the autocast exception path when torch.autocast raises."""

    def _raising_autocast(**_kwargs):
        raise TypeError("no autocast available")

    monkeypatch.setattr(scoring, "torch", SimpleNamespace(autocast=_raising_autocast))
    ctx = scoring._autocast_context(
        SimpleNamespace(autocast=None), SimpleNamespace(type="cuda")
    )
    assert isinstance(ctx, nullcontext)


def test_autocast_context_fallback_without_test_scoring_torch(monkeypatch):
    """Ensure nullcontext is returned when no autocast hooks exist anywhere."""

    accel = SimpleNamespace(autocast="not-callable")
    device = SimpleNamespace(type=None)
    torch_stub = SimpleNamespace(autocast=None)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(scoring, "torch", torch_stub)
    monkeypatch.delitem(sys.modules, "tests.evaluation.test_scoring", raising=False)
    ctx = scoring._autocast_context(accel, device)
    assert isinstance(ctx, nullcontext)
