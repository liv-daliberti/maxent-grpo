"""Unit tests for recipe loading helpers."""

from __future__ import annotations

from maxent_grpo.config.recipes import _dataclass_field_names, _split_recipe_payload, load_grpo_recipe
from maxent_grpo.config.grpo import GRPOScriptArguments, GRPOConfig


def test_dataclass_field_names():
    fields = _dataclass_field_names(GRPOScriptArguments)
    assert "reward_funcs" in fields
    assert "cosine_max_len" in fields


def test_split_recipe_payload_routes_fields():
    class _ModelCfg:
        __dataclass_fields__ = {"model_name_or_path": None, "trust_remote_code": None}

    payload = {
        "reward_funcs": ["foo"],
        "beta": 0.2,
        "model_name_or_path": "m",
        "trust_remote_code": True,
        "extra": "x",
    }
    script, training, model = _split_recipe_payload(payload, _ModelCfg)
    assert script == {"reward_funcs": ["foo"]}
    assert training.get("beta") == 0.2
    assert model == {"model_name_or_path": "m", "trust_remote_code": True}


def test_load_grpo_recipe_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "recipe.yaml"
    path.write_text("reward_funcs: ['r1']\nbeta: 0.3\ndataset_name: ds\nmodel_name_or_path: repo/model\n", encoding="utf-8")

    class _ModelCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    args, cfg, model_cfg = load_grpo_recipe(str(path), model_config_cls=_ModelCfg)
    assert isinstance(args, GRPOScriptArguments)
    assert isinstance(cfg, GRPOConfig)
    assert isinstance(model_cfg, _ModelCfg)
    assert args.reward_funcs == ["r1"]
    assert cfg.beta == 0.3
    assert model_cfg.kwargs["model_name_or_path"] == "repo/model"
