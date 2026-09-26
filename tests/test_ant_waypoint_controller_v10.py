from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))


def test_v10_protocol_freezes_failed_v9_boundary_and_new_evaluation():
    protocol = (
        ROOT / "paper/preregistration/ant_waypoint_controller_v10_20260729.md"
    ).read_text()
    assert "BEFORE V10 TRAINING OR EVALUATION" in protocol
    assert "4/12" in protocol
    assert "2,000,000" in protocol
    assert "2e-6" in protocol
    assert "four new 10x10 maps" in protocol
    assert "5073010" in protocol
    assert "floor((n-1)/2)" in protocol


def test_v10_schedule_and_fresh_maps_match_the_frozen_contract():
    pytest.importorskip("gymnasium")
    import train_ant_waypoint_controller_v10 as v10

    counts = Counter(v10.TRAINING_HEADING_SCHEDULE)
    assert counts == Counter({1: 7, 0: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1})
    assert len(v10.DEVELOPMENT_MAPS) == 4
    assert all(len(maze) == 10 and all(len(row) == 10 for row in maze) for maze in v10.DEVELOPMENT_MAPS)
    assert len({repr(maze) for maze in v10.DEVELOPMENT_MAPS}) == 4
    for by_heading in v10.DEVELOPMENT_EDGES:
        for edges in by_heading:
            indices = [v10._stratified_edge_index(edges, replicate) for replicate in range(3)]
            assert indices == [0, (len(edges) - 1) // 2, len(edges) - 1]
            assert len(set(indices)) == 3


def test_v10_launcher_is_held_and_forbids_v9_route_or_lm_execution():
    launcher = (
        ROOT / "ops/exp_scaling/launch_ant_waypoint_controller_v10.sh"
    ).read_text()
    assert 'receipt.get("status") != "fail"' in launcher
    assert "Ant v10 requires no v9 route or language-model execution" in launcher
    assert "sbatch --parsable --hold" in launcher
    assert '"language_model_sampled":False' in launcher
    assert '"route_map_executed":False' in launcher
