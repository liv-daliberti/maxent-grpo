# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.utils.data import Dataset
import os, random, copy, torch


class ReplayMixDataset(Dataset):
    def __init__(self, base_ds, tok):
        self.base_ds  = base_ds
        self.tok      = tok          # just for the debug assert

    def __len__(self):                       # unchanged
        return len(self.base_ds)

    def __getitem__(self, idx):              # no replay logic here
        item = copy.deepcopy(self.base_ds[idx])

        # → quick sanity–check only if already tokenised
        if "input_ids" in item:
            last_user = next(m for m in reversed(item["prompt"])
                             if m["role"] == "user")
            assert last_user["content"] in self.tok.decode(item["input_ids"]), \
                   "⛔ clue missing from encoded prompt!"

        item["is_replay"] = 0                # mark as fresh
        return item

def replay_collate(batch, *, replay_buffer, replay_prob):
    """
    Identity collate — no sampling here.
    We only want to clear the old `is_replay` flag,
    but we *must not* drop 'accuracy'.
    """
    for ex in batch:
        ex["is_replay"] = 0
        # do NOT pop accuracy here!
        # ex.pop("accuracy", None)    ← remove this line
    return batch
