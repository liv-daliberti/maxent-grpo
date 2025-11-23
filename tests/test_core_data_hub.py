"""Unit tests for core.data and core.hub helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_load_dataset_split_raises_when_missing_datasets(monkeypatch):
    import core.data as data

    monkeypatch.setattr(data, "datasets", None)
    with pytest.raises(ImportError):
        data.load_dataset_split("demo", split="train")


def test_get_dataset_mixture_subsampling(monkeypatch):
    import core.data as data
    from maxent_grpo.config.dataset import DatasetConfig, DatasetMixtureConfig, ScriptArguments

    class _FakeDS:
        def __init__(self, rows):
            self._rows = rows
            self.selected = None
            self.shuffled = False

        def __len__(self):
            return len(self._rows)

        def select_columns(self, cols):
            self.selected = cols
            return self

        def shuffle(self, seed=None):
            self.shuffled = True
            return self

        def select(self, indices):
            return _FakeDS([self._rows[i] for i in indices])

    def _load_dataset(name, config=None, split=None):
        return _FakeDS(list(range(10)))

    monkeypatch.setattr(
        data, "datasets", SimpleNamespace(load_dataset=_load_dataset, concatenate_datasets=lambda seq: seq[0])
    )
    mixture = {
        "datasets": [{"id": "a", "config": None, "split": "train", "columns": ["x"], "weight": 0.5}],
        "seed": 0,
        "test_split_size": None,
    }
    args = ScriptArguments(dataset_name=None, dataset_mixture=mixture)
    ds_dict = data.get_dataset(args)
    assert isinstance(ds_dict["train"], _FakeDS)
    assert ds_dict["train"].selected == ["x"]
    # weight=0.5 on len=10 should produce 5 examples
    assert len(ds_dict["train"]) == 5


def test_get_param_count_from_repo_id_uses_safetensors(monkeypatch):
    import core.hub as hub

    monkeypatch.setattr(hub, "get_safetensors_metadata", lambda repo: SimpleNamespace(parameter_count={"w": 123}))
    assert hub.get_param_count_from_repo_id("org/unknown") == 123
