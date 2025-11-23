"""Tests covering reference embedding hydration for the training setup."""

from __future__ import annotations

from types import SimpleNamespace

from .helpers.run_setup_stubs import (
    FakeEmbedding,
    FakeLM,
    FakeParameter,
    load_run_setup as _load_run_setup,
)


def test_reference_embedding_hydration_rebuilds_flat_weights(monkeypatch):
    """Hydration copies trainable embeddings and re-ties the LM head."""

    run_setup = _load_run_setup(monkeypatch)

    vocab = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    train_model = FakeLM(vocab)
    ref_model = FakeLM([[0.0]])  # overwritten immediately
    ref_model.embed_tokens.weight = FakeParameter([], requires_grad=False)
    ref_model.lm_head.weight = ref_model.embed_tokens.weight

    updated = run_setup._hydrate_reference_embeddings(train_model, ref_model)
    assert updated is True
    assert ref_model.embed_tokens.weight.requires_grad is False
    assert ref_model.lm_head.weight is ref_model.embed_tokens.weight
    assert ref_model.embed_tokens.weight.as_lists() == vocab


def test_reference_embedding_hydration_fails_on_missing_embeddings(monkeypatch):
    run_setup = _load_run_setup(monkeypatch)
    train_model = SimpleNamespace(get_input_embeddings=lambda: None)
    ref_model = SimpleNamespace(get_input_embeddings=lambda: None)
    assert run_setup._hydrate_reference_embeddings(train_model, ref_model) is False


def test_hydration_skips_invalid_tensors(monkeypatch):
    run_setup = _load_run_setup(monkeypatch)

    class _BrokenLM:
        def __init__(self):
            self.embed_tokens = FakeEmbedding([])
            self.lm_head = SimpleNamespace(weight=self.embed_tokens.weight)

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    train_model = _BrokenLM()
    ref_model = FakeLM([[0.0]])
    assert run_setup._hydrate_reference_embeddings(train_model, ref_model) is False
