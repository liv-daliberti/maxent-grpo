from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("gymnasium")
import train_ant_waypoint_controller_v9 as v9


def _fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def test_v9_training_and_development_maps_are_disjoint_and_heading_complete():
    train = {_fingerprint(value) for value in v9.TRAIN_MAPS}
    development = {_fingerprint(value) for value in v9.DEVELOPMENT_MAPS}
    assert len(train) == 4
    assert len(development) == 4
    assert train.isdisjoint(development)
    assert all(len(value) == 8 for value in v9.TRAIN_EDGES)
    assert all(len(value) == 8 for value in v9.DEVELOPMENT_EDGES)
    assert all(edges for maps in v9.TRAIN_EDGES for edges in maps)
    assert all(edges for maps in v9.DEVELOPMENT_EDGES for edges in maps)


def test_v9_local_waypoint_environment_matches_controller_observation_shape():
    env = v9.AntMazeLocalWaypointEnv(rank=0, episode_steps=400)
    try:
        observation, info = env.reset(seed=73_009)
        assert observation.shape == env.observation_space.shape
        assert observation.dtype == np.float32
        assert info["waypoint_heading_index"] in range(8)
        assert info["waypoint_training_map_index"] == 0
        next_observation, reward, terminated, truncated, step_info = env.step(
            env.action_space.sample()
        )
        assert next_observation.shape == observation.shape
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert np.isfinite(step_info["waypoint_distance"])
    finally:
        env.close()
