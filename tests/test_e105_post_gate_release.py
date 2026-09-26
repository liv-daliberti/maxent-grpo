from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "var/artifacts/e105_post_gate_release_job.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_post_gate_release_is_dependency_locked_cpu_only_and_fail_closed():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["schema"] == "e105_post_gate_release_job_v1"
    assert payload["job_id"] == 30648755
    assert payload["dependency"] == "afterany:30647379"
    assert payload["dependency_job_id"] == 30647379
    assert payload["dependency_locked"] is True
    assert payload["fail_closed_on_gate"] is True
    assert payload["terminal_cells_at_submission"] == 14
    assert payload["scientific_configuration_changed"] is False
    assert payload["outcome_metrics_inspected"] is False
    assert payload["pointmaze"] == "excluded"
    assert payload["downstream_order"] == [
        "refresh_gate",
        "render_mechanism",
        "paired_placement",
        "submit_e109",
        "submit_e105",
        "campaign_stats",
    ]

    gate_path = Path(payload["gate_at_submission"])
    assert gate_path.is_file()
    assert len(payload["gate_at_submission_sha256"]) == 64

    # The dependency refreshes this named gate in place. Its recorded digest
    # is a historical submission receipt, unlike the frozen inputs below.
    for path_key, digest_key in (
        ("e110_ledger", "e110_ledger_sha256"),
        ("protocol", "protocol_sha256"),
        ("slurm_script", "slurm_script_sha256"),
        ("launcher", "launcher_sha256"),
    ):
        path = Path(payload[path_key])
        assert path.is_file()
        assert payload[digest_key] == _digest(path)

    record = payload["scheduler_record"]
    assert "JobState=PENDING" in record
    assert "Reason=Dependency" in record
    assert "Dependency=afterany:30647379(unfulfilled)" in record
    assert "RunTime=00:00:00" in record
    assert "ReqTRES=cpu=1,mem=8G,node=1" in record
    assert "gres/gpu" not in record
    assert "Requeue=0" in record

    script = Path(payload["slurm_script"]).read_text(encoding="utf-8")
    commands = (
        "audit_e106_python_lambda_normalization.py",
        "plot_e106_group_centered_mechanism_diagnostic.py",
        "apply_e105_qwen3_paired_a6000_placement_amendment.py --apply",
        "launch_e109_repaired_python_replay_comparators.py --submit",
        "launch_e105_group_centered_semantic_repair_full_three_scale.py --submit",
        "campaign_stats.py --markdown",
    )
    positions = [script.index(command) for command in commands]
    assert positions == sorted(positions)
    assert "set -euo pipefail" in script
