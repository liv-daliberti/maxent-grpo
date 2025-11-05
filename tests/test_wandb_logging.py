import os
from types import SimpleNamespace

from src.utils.wandb_logging import init_wandb_training


def test_init_wandb_training_sets_env(monkeypatch):
    for k in ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_RUN_GROUP"]:
        monkeypatch.delenv(k, raising=False)
    args = SimpleNamespace(wandb_entity="ent", wandb_project="proj", wandb_run_group="grp")
    init_wandb_training(args)
    assert os.environ["WANDB_ENTITY"] == "ent"
    assert os.environ["WANDB_PROJECT"] == "proj"
    assert os.environ["WANDB_RUN_GROUP"] == "grp"

