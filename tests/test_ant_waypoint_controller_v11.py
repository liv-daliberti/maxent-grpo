from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))


def test_v11_protocol_freezes_v10_failure_and_fresh_evaluation():
    text = (ROOT / "paper/preregistration/ant_waypoint_controller_v11_20260730.md").read_text()
    assert "BEFORE V11 TRAINING OR EVALUATION" in text
    assert "8/12" in text and "10/12" in text
    assert "four new 12×12 maps" in text
    assert "6073011" in text
    assert "unchanged gate" in text


def test_v11_schedule_and_fresh_maps_match_contract():
    pytest.importorskip("gymnasium")
    import train_ant_waypoint_controller_v11 as v11

    assert Counter(v11.TRAINING_HEADING_SCHEDULE) == Counter({3: 6, 1: 4, 0: 1, 2: 1, 4: 1, 5: 1, 6: 1, 7: 1})
    assert len(v11.DEVELOPMENT_MAPS) == 4
    assert all(len(maze) == 12 and all(len(row) == 12 for row in maze) for maze in v11.DEVELOPMENT_MAPS)
    assert len({repr(maze) for maze in v11.DEVELOPMENT_MAPS}) == 4
    for by_heading in v11.DEVELOPMENT_EDGES:
        for edges in by_heading:
            indices = [v11._stratified_edge_index(edges, replicate) for replicate in range(3)]
            assert indices == [0, (len(edges) - 1) // 2, len(edges) - 1]
            assert len(set(indices)) == 3


def test_v11_launcher_is_held_and_forbids_v10_route_or_lm_execution():
    text = (ROOT / "ops/exp_scaling/launch_ant_waypoint_controller_v11.sh").read_text()
    assert "Ant v11 requires no v10 route or language-model execution" in text
    assert "sbatch --parsable --hold" in text
    assert '"language_model_sampled":False' in text
    assert '"route_map_executed":False' in text
    assert '"evaluation_map_size":12' in text
    assert '"prelaunch_canceled_job_id":30198258' in text
    assert 'scontrol update "JobId=$job_id" Partition=all' in text
