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

import os
from types import SimpleNamespace

from telemetry.wandb import init_wandb_training


def test_init_wandb_training_sets_env(monkeypatch):
    for k in ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_RUN_GROUP"]:
        monkeypatch.delenv(k, raising=False)
    args = SimpleNamespace(
        wandb_entity="ent", wandb_project="proj", wandb_run_group="grp"
    )
    init_wandb_training(args)
    assert os.environ["WANDB_ENTITY"] == "ent"
    assert os.environ["WANDB_PROJECT"] == "proj"
    assert os.environ["WANDB_RUN_GROUP"] == "grp"


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
