from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fullscale_data_contract_is_exact_and_balanced():
    maker = _load(
        "ops/make_point_maze_fullscale_v7_data.py",
        "point_maze_fullscale_v7_data",
    )
    assert maker.SPLIT_ROUNDS == {"train": 12, "dev": 2, "eval": 4}
    assert len(maker.FAMILIES) == 8
    assert 12 * 8 * 4 == 384
    assert 2 * 8 * 4 == 64
    assert 4 * 8 * 4 == 128


def test_fullscale_protocol_freezes_real_384_128_online_study():
    text = (
        ROOT
        / "paper/preregistration/"
        "point_maze_fullscale_v7_05b_12pass_20260730.md"
    ).read_text()
    # The protocol is hard-wrapped prose, so required phrases may span lines.
    text = " ".join(text.split())
    for required in (
        "BEFORE THE BALANCED-V6 TERMINAL OUTCOME",
        "384 distinct executable PointMaze prompts",
        "128 untouched prompts",
        "4,608 optimizer updates per cell",
        "4,224 trajectories per cell and coordinate",
        "76641, 76642, 76643, 76644, and 76645",
        "No efficacy threshold",
    ):
        assert required in text
