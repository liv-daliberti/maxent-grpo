from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.exp_scaling import e16_canonical_plan as plan
from ops.exp_scaling import verify_e16_execution_surface as surface
from ops.exp_scaling import verify_e16_held_cohort as gate


ROOT = Path(__file__).resolve().parents[1]


def _manifest(path: Path, task: str, stage: str, start: int) -> list[str]:
    campaign = plan.campaign_plan(stage)
    prefix = campaign["tasks"][task]["prefix"]
    job_ids = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=("arm", "seed", "job_id", "run_stamp"),
        )
        writer.writeheader()
        index = start
        for seed in campaign["seeds"]:
            for arm in plan.ARMS:
                job_id = str(index)
                index += 1
                job_ids.append(job_id)
                writer.writerow(
                    {
                        "arm": arm,
                        "seed": seed,
                        "job_id": job_id,
                        "run_stamp": f"{prefix}_{arm}_s{seed}",
                    }
                )
    return job_ids


def test_complete_smoke_cohort_must_be_user_held(tmp_path, monkeypatch):
    graph = tmp_path / "graph.tsv"
    countdown = tmp_path / "countdown.tsv"
    job_ids = _manifest(graph, "graph_coloring", "smoke", 100)
    job_ids += _manifest(countdown, "countdown", "smoke", 200)

    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="JobId=1 JobState=PENDING Reason=JobHeldUser\n"
        ),
    )
    result = gate.verify_cohort(
        stage="smoke", graph_manifest=graph, countdown_manifest=countdown
    )
    assert result["cell_count"] == 6
    assert set(result["job_ids"]) == set(job_ids)


def test_partial_or_released_cohort_is_rejected(tmp_path, monkeypatch):
    graph = tmp_path / "graph.tsv"
    countdown = tmp_path / "countdown.tsv"
    _manifest(graph, "graph_coloring", "smoke", 100)
    _manifest(countdown, "countdown", "smoke", 200)
    rows = countdown.read_text().splitlines()
    countdown.write_text("\n".join(rows[:-1]) + "\n")
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="JobId=1 JobState=PENDING Reason=JobHeldUser\n"
        ),
    )
    with pytest.raises(ValueError, match="incomplete"):
        gate.verify_cohort(
            stage="smoke", graph_manifest=graph, countdown_manifest=countdown
        )

    _manifest(countdown, "countdown", "smoke", 200)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="JobId=1 JobState=PENDING Reason=Priority\n"
        ),
    )
    with pytest.raises(ValueError, match="not pending on a user hold"):
        gate.verify_cohort(
            stage="smoke", graph_manifest=graph, countdown_manifest=countdown
        )


def test_frozen_execution_surface_contains_recursive_local_dependencies():
    required = {
        "ops/make_modebench_data.py",
        "ops/exp_scaling/audit_e14_checkpoint.py",
        "ops/exp_scaling/check_e14_preflight.py",
        "ops/exp_scaling/verify_e14_dataset.py",
        "ops/exp_scaling/verify_e14_runtime.py",
    }
    assert required <= set(surface.EXECUTION_FILES)


def test_launcher_rolls_back_only_invocation_owned_artifacts():
    launcher = (
        ROOT / "ops/exp_scaling/launch_e16_canonical_maxent_replication.sh"
    ).read_text(encoding="utf-8")
    assert 'mkdir "$snapshot_parent"' in launcher
    assert '"$graph_manifest" "$countdown_manifest"' in launcher
    assert 'rm -rf -- "$snapshot_parent"' in launcher
    assert "scancel failed; retaining frozen artifacts" in launcher
    assert launcher.index("trap rollback_unreleased_cohort EXIT") < launcher.index(
        'mkdir "$snapshot_parent"'
    )
