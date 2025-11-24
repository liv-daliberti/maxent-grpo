"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from types import SimpleNamespace
import builtins
import importlib.util
from pathlib import Path
import sys
import pytest

import maxent_grpo.core.data as D


class FakeDS:
    def __init__(self, n):
        self._n = int(n)
        self.column_names = []

    def __len__(self):
        return self._n

    def select_columns(self, cols):
        self.column_names = list(cols)
        return self

    def shuffle(self, seed=0):
        return self

    def select(self, rng):
        # range(N) -> object with __len__
        n = len(list(rng))
        return FakeDS(n)

    def train_test_split(self, test_size=0.1, seed=0):
        test_n = int(round(self._n * float(test_size)))
        train_n = self._n - test_n
        return {"train": FakeDS(train_n), "test": FakeDS(test_n)}


def test_get_dataset_single_name(monkeypatch):
    def fake_load_dataset(name, config):
        assert name == "ds/name" and config is None
        return {"train": FakeDS(3)}

    monkeypatch.setattr(D, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    args = SimpleNamespace(
        dataset_name="ds/name", dataset_config=None, dataset_mixture=None
    )
    out = D.get_dataset(args)
    assert "train" in out


def test_get_dataset_mixture_weights_and_split(monkeypatch):
    # Two datasets, with weights 0.5 each, total should sum and split
    loads = {}

    def fake_load_dataset(id, config=None, split="train"):
        loads.setdefault(id, 0)
        loads[id] += 1
        return FakeDS(100)

    def fake_concat(list_ds):
        return FakeDS(sum(len(ds) for ds in list_ds))

    monkeypatch.setattr(D, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setattr(D, "concatenate_datasets", fake_concat)

    mixture = {
        "datasets": [
            {"id": "a", "config": None, "weight": 0.5, "columns": ["x", "y"]},
            {"id": "b", "config": None, "weight": 0.5, "columns": ["x", "y"]},
        ],
        "seed": 123,
        "test_split_size": 0.2,
    }
    args = D.ScriptArguments(dataset_name=None, dataset_mixture=mixture)

    out = D.get_dataset(args)
    # Expect keys train/test and total ~100 since 50+50 then split 20%
    assert set(out.keys()) == {"train", "test"}


def _load_data_module_without_datasets(monkeypatch):
    """Reload core.data while forcing datasets import to fail."""
    module_name = "core.data_no_datasets"
    path = Path(D.__file__)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ModuleNotFoundError("datasets missing")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_get_dataset_errors_without_config(monkeypatch):
    """ValueError when neither dataset_name nor mixture is provided."""

    def fake_load_dataset(*_args, **_kwargs):
        return {"train": FakeDS(1)}

    monkeypatch.setattr(D, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    args = SimpleNamespace(dataset_name=None, dataset_config=None, dataset_mixture=None)
    with pytest.raises(ValueError):
        D.get_dataset(args)


def test_get_dataset_empty_mixture(monkeypatch):
    """ValueError when mixture list is empty."""

    def fake_load_dataset(*_args, **_kwargs):
        return {"train": FakeDS(1)}

    monkeypatch.setattr(D, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    mixture = SimpleNamespace(datasets=[], seed=0, test_split_size=None)
    args = SimpleNamespace(
        dataset_name=None, dataset_config=None, dataset_mixture=mixture
    )
    with pytest.raises(ValueError):
        D.get_dataset(args)


def test_get_dataset_import_guard(monkeypatch):
    monkeypatch.setattr(D, "datasets", None)
    args = SimpleNamespace(dataset_name="ds/name", dataset_config=None, dataset_mixture=None)
    with pytest.raises(ImportError):
        D.get_dataset(args)


def test_load_dataset_split_guards(monkeypatch):
    """load_dataset_split raises for missing deps or split."""
    monkeypatch.setattr(D, "datasets", None)
    with pytest.raises(ImportError):
        D.load_dataset_split("repo/name", split="train")

    def fake_load_dataset(*_args, **_kwargs):
        return {"train": FakeDS(1)}

    monkeypatch.setattr(D, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    with pytest.raises(ValueError):
        D.load_dataset_split("repo/name", split="")


def test_fallback_dataset_methods_raise(monkeypatch):
    """Placeholder Dataset raises NotImplemented in fallback mode."""
    mod = _load_data_module_without_datasets(monkeypatch)
    ds = mod.Dataset()
    with pytest.raises(NotImplementedError):
        len(ds)
    with pytest.raises(NotImplementedError):
        ds.shuffle()
    with pytest.raises(NotImplementedError):
        ds.select([])
    with pytest.raises(NotImplementedError):
        ds.select_columns([])


"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
