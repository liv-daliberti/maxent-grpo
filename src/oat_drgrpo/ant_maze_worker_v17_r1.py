"""Cached-loader execution amendment for the Ant v17 worker."""

from __future__ import annotations

from pathlib import Path

from . import ant_maze_worker_v17 as base


_MODEL = None


def _executor_identity():
    payload = base._executor_identity()
    payload["worker_source_sha256"] = base._sha256(Path(__file__))
    return payload


def controller_receipt_sha256():
    base.controller_identity()
    return base._canonical_sha256(_executor_identity())


def _model():
    global _MODEL
    if _MODEL is None:
        from stable_baselines3 import PPO

        base.controller_identity()
        _MODEL = PPO.load(base.MODEL_PATH, device="cpu")
    return _MODEL


base._executor_identity = _executor_identity
base.controller_receipt_sha256 = controller_receipt_sha256
base._model = _model
base.v12.controller_receipt_sha256 = controller_receipt_sha256
base.v12._model = _model

controller_identity = base.controller_identity
execute_ant_v17_raw = base.execute_ant_v17_raw
execute_ant_v17 = base.execute_ant_v17
