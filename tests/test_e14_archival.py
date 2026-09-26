from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from exp_scaling.e14_archival import (
    ArchiveReceiptError,
    EXPECTED_CHECKPOINT_STEPS,
    REMOVED_CHECKPOINT_STEPS,
    archive_receipt_path,
    replay_archive_authorization_if_needed,
    verify_archive_receipt,
    write_archive_receipt,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weights_hash(path: Path) -> str:
    weight = path / "model.safetensors"
    digest = hashlib.sha256()
    digest.update(weight.name.encode("utf-8"))
    digest.update(bytes.fromhex(_hash(weight)))
    return digest.hexdigest()


def _validated_c0(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    debug = run / "debug_1"
    saved = debug / "saved_models"
    for tag in EXPECTED_CHECKPOINT_STEPS:
        checkpoint = saved / tag
        checkpoint.mkdir(parents=True)
        content = b"endpoint" if tag in {"step_00128", "step_00129"} else tag.encode()
        (checkpoint / "model.safetensors").write_bytes(content)
        (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")

    metrics = debug / "train_metrics.jsonl"
    metrics.write_text("{}\n", encoding="utf-8")
    identity_file = tmp_path / "identity.tsv"
    identity_file.write_text("identity\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    evidence_paths = {
        "identity": identity_file,
        "metrics": metrics,
        "endpoint_audit": audit,
        "checkpoint_config": saved / "step_00128" / "config.json",
    }
    artifact = tmp_path / "c0_approval.json"
    payload = {
        "approved": True,
        "gate": "e14_canonical_c0",
        "protocol": "E14",
        "arm": "C0",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": "12345",
        "run_dir": str(run.resolve()),
        "slurm": {"state": "COMPLETED", "exit_code": "0:0"},
        "identity": {
            "stamp": "e14_c0_test",
            "source_hash": "a" * 64,
        },
        "checks": {"exact_step_00128_audit": True},
        "metrics_summary": {
            "step_128_weights_manifest_sha256": _weights_hash(
                saved / "step_00128"
            ),
            "step_129_byte_identical_alias": True,
        },
        "evidence": {
            label: {"path": str(path.resolve()), "sha256": _hash(path)}
            for label, path in evidence_paths.items()
        },
    }
    artifact.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact, saved


def _remove_intermediates(saved: Path) -> None:
    for tag in REMOVED_CHECKPOINT_STEPS:
        shutil.rmtree(saved / tag)


def test_pre_removal_receipt_authorizes_only_exact_archived_layout(tmp_path):
    artifact, saved = _validated_c0(tmp_path)
    receipt = write_archive_receipt(artifact)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["recording_phase"] == "pre_removal"

    # Full-schedule replay remains strict and needs no archival exception.
    assert replay_archive_authorization_if_needed(artifact) is None
    _remove_intermediates(saved)

    authorization = replay_archive_authorization_if_needed(artifact)
    assert authorization is not None
    assert authorization["removed_steps"] == REMOVED_CHECKPOINT_STEPS


def test_post_removal_recovery_is_explicit_and_records_chronology(tmp_path):
    artifact, saved = _validated_c0(tmp_path)
    _remove_intermediates(saved)

    with pytest.raises(ArchiveReceiptError, match="explicit post-removal"):
        write_archive_receipt(artifact)
    receipt = write_archive_receipt(
        artifact, allow_post_removal_recovery=True
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["recording_phase"] == "post_removal_recovery"
    assert payload["schedule"]["observed_steps_when_recorded"] == [
        "step_00128",
        "step_00129",
    ]
    verify_archive_receipt(receipt, validation_artifact=artifact)


@pytest.mark.parametrize("mutation", ("artifact", "checkpoint", "receipt"))
def test_archive_receipt_fails_closed_on_any_binding_drift(tmp_path, mutation):
    artifact, saved = _validated_c0(tmp_path)
    receipt = write_archive_receipt(artifact)
    _remove_intermediates(saved)

    if mutation == "artifact":
        artifact.write_text(
            artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        message = "not bound"
    elif mutation == "checkpoint":
        (saved / "step_00128" / "config.json").write_text(
            '{"changed": true}\n', encoding="utf-8"
        )
        message = "evidence 'checkpoint_config' changed"
    else:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        payload["schedule"]["removed_steps"] = ["step_00032"]
        receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        message = "exactly steps 32/64/96"

    with pytest.raises(ArchiveReceiptError, match=message):
        verify_archive_receipt(receipt, validation_artifact=artifact)


def test_partial_layout_without_exact_receipt_cannot_replay(tmp_path):
    artifact, saved = _validated_c0(tmp_path)
    shutil.rmtree(saved / "step_00032")

    with pytest.raises(ArchiveReceiptError, match="partial/unknown"):
        replay_archive_authorization_if_needed(artifact)
    assert not archive_receipt_path(artifact).exists()
