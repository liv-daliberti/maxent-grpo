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

import core.data as D


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
