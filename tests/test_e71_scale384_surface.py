"""Guard the E71 supersession rules that the manuscript surface depends on."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agg = _load("e71agg", "ops/exp_scaling/aggregate_e70_paper_checkpoints.py")
audit = _load("e71audit", "ops/exp_scaling/audit_e71_scale384_05b.py")


PROMPTS_PER_PASS = 6144.0


def _curve(max_pass, drop=()):
    """Scaling-curve rows at the real quarter-pass cadence, 0..max_pass.

    `load_curve` infers one pass from four evaluation strides, so fixtures must
    carry the same quarter-pass resolution the campaigns actually emit.
    `drop` removes (seed, pass) coordinates to simulate uneven seed progress.
    """
    rows = []
    quarters = int(round(max_pass * 4))
    for arm in agg.ARMS:
        for seed in agg.SEEDS:
            for quarter in range(quarters + 1):
                training_pass = quarter / 4
                if (seed, training_pass) in drop:
                    continue
                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "split": "multi_answer",
                    "step": int(training_pass * 384),
                    "prompt_consumed": training_pass * PROMPTS_PER_PASS,
                    "greedy": 0.5,
                    "mean8": 0.5,
                    "pass8": 0.5,
                    "distinct8": 1.0,
                })
    return rows


def test_incomplete_successor_does_not_pull_a_finished_row_backwards(tmp_path):
    """A freshly launched E71 curve at pass 0 must not supersede E70 at pass 12."""
    (tmp_path / "gce71_x_scaling_curve.json").write_text(json.dumps(_curve(3)))
    (tmp_path / "gce70_x_scaling_curve.json").write_text(json.dumps(_curve(12)))
    name, _, depth, _ = agg.resolve_curve(
        ["gce71_x_scaling_curve.json", "gce70_x_scaling_curve.json"], tmp_path
    )
    assert name.startswith("gce70")
    assert depth == 12


def test_caught_up_successor_wins_the_tie(tmp_path):
    """Once E71 reaches pass 12 it takes over from the equally deep E70 curve."""
    (tmp_path / "gce71_x_scaling_curve.json").write_text(json.dumps(_curve(12)))
    (tmp_path / "gce70_x_scaling_curve.json").write_text(json.dumps(_curve(12)))
    name, _, depth, _ = agg.resolve_curve(
        ["gce71_x_scaling_curve.json", "gce70_x_scaling_curve.json"], tmp_path
    )
    assert name.startswith("gce71")
    assert depth == 12


def test_partial_seed_coverage_is_refused(tmp_path):
    """A checkpoint missing any seed of either arm is not a reportable endpoint."""
    rows = _curve(12, drop={(47, 12.0), (47, 11.75), (47, 10.0)})
    (tmp_path / "only_scaling_curve.json").write_text(json.dumps(rows))
    _, _, depth, _ = agg.resolve_curve(["only_scaling_curve.json"], tmp_path)
    assert depth == 9


def test_audit_tree_hash_matches_the_launcher_shell_digest(tmp_path):
    """The audit's data-drift check must reproduce the launcher's hash exactly."""
    (tmp_path / "a.txt").write_text("alpha")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.txt").write_text("beta")
    (tmp_path / "a.suffix").write_text("gamma")  # '.' sorts before '/'

    shell = subprocess.run(
        "find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum "
        "| cut -d' ' -f1",
        shell=True, cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert audit._tree_hash(tmp_path) == shell


def test_e71_audit_expects_the_registered_384_design():
    assert audit.TRAIN_ROWS == 384
    assert audit.EVAL_ROWS == 128
    assert audit.EXPECTED_STEPS == {
        "graph_coloring": 4608,
        "pantry_plan": 4608,
    }
    assert set(audit.DATA_ROOTS) == {"graph_coloring", "pantry_plan"}


@pytest.mark.skipif(
    not (ROOT / "var/artifacts/e71_scale384_05b_identity.json").is_file(),
    reason="E71 identity not yet written",
)
def test_live_identity_records_the_registered_design():
    identity = json.loads(
        (ROOT / "var/artifacts/e71_scale384_05b_identity.json").read_text()
    )
    assert identity["schema"] == "e71_scale384_05b_v1"
    assert identity["train_rows"] == 384
    assert identity["eval_rows"] == 128
    assert identity["optimizer_updates_per_run"] == 4608
    assert sum(len(v) for v in identity["jobs"].values()) == 20
