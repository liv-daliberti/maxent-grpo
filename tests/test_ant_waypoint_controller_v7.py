from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ops/train_ant_waypoint_controller_v7.py"


def test_ant_waypoint_v7_training_source_is_maze_blind():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'gym.make(\n            "Ant-v5"' in text
    assert "AntMaze" not in text
    assert "maze_map=" not in text
    assert "route_gates" not in text
    assert "MaxEnt outcome" in text


def test_ant_waypoint_v7_contract_has_fixed_distance_and_success_radius():
    text = SOURCE.read_text(encoding="utf-8")
    assert "WAYPOINT_DISTANCE = 4.0" in text
    assert "TRAINING_DISTANCES = np.asarray((4.0,)" in text
    assert "heading_index = (self.rank + self.episode_index) % len(HEADINGS)" in text
    assert "SUCCESS_THRESHOLD = 0.45" in text
    assert "INITIAL_MODEL_SHA256" in text
    assert "model.policy.load_state_dict(initial.policy.state_dict())" in text
    assert "success_rate_at_least_0p90" in text
    assert "minimum_heading_success_rate_at_least_0p75" in text
    assert "DEFAULT_LEARNING_RATE = 1e-5" in text
    assert "DEFAULT_TIMESTEPS = 5_000_000" in text
    assert 'ant_waypoint_v6.zip' in text
    assert "EVALUATION_SEED_OFFSET = 2_000_000" in text
    assert "seed=args.seed + EVALUATION_SEED_OFFSET" in text
