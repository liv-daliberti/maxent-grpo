"""Batch-eight runtime keeps v2 transitions and requires new smoke evidence."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import continue_modebench_scale_runtime_v3 as controller
import test_continue_modebench_scale_runtime_v2 as regression


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    # Reuse the isolated scientific fixture, with this version's controller and
    # amendment/profile. No real campaign paths, jobs, or model inference run.
    monkeypatch.setattr(regression, 'controller', controller)
    return regression.campaign.__wrapped__(tmp_path, monkeypatch)


def test_batch8_profile_and_preserved_controller_provenance():
    assert controller.runtime_profile()['batch_size'] == 8
    assert controller.runtime_profile()['concurrency'] == 1
    assert controller.DEFAULT_ARTIFACTS.name == 'modebench_scale_runtime_v3'
    assert controller.AMENDMENT_SCHEMA == 'modebench_scale_runtime_amendment_v3'
    assert controller.file_sha(controller.ORIGINAL_CONTROLLER) == controller.ORIGINAL_CONTROLLER_SHA256
    assert controller.file_sha(controller.PREVIOUS_CONTROLLER) == controller.PREVIOUS_CONTROLLER_SHA256


def test_v3_arm_seals_new_amendment_and_preserved_v2_source(campaign):
    regression.sweep(campaign)
    seal = controller.read(campaign.artifacts / 'controller_identity.json')
    assert seal['schema'] == 'modebench_scale_controller_runtime_v3'
    assert seal['runtime_profile']['batch_size'] == 8
    assert seal['previous_controller_sha256'] == controller.PREVIOUS_CONTROLLER_SHA256
    assert seal['files_sha256'][str(controller.PREVIOUS_CONTROLLER)] == controller.PREVIOUS_CONTROLLER_SHA256
    assert seal['files_sha256'][str(campaign.artifacts / 'runtime_amendment.json')] == controller.file_sha(
        campaign.artifacts / 'runtime_amendment.json')


def test_batch2_smoke_receipts_cannot_authorize_batch8_runtime(campaign):
    amendment_path = campaign.artifacts / 'runtime_amendment.json'
    amendment = controller.read(amendment_path)
    for reference in amendment['runtime_smoke_receipts'].values():
        path = Path(reference['path'])
        receipt = controller.read(path)
        receipt['runtime_profile']['batch_size'] = 2
        regression.write(path, receipt)
        reference['sha256'] = controller.file_sha(path)
    regression.write(amendment_path, amendment)
    with pytest.raises(ValueError, match='runtime smoke receipt did not pass'):
        regression.sweep(campaign)
    assert not any(campaign.calls.values())
    assert not (campaign.artifacts / 'controller_identity.json').exists()


def test_amendment_requires_preserved_v2_source_pin(campaign):
    path = campaign.artifacts / 'runtime_amendment.json'
    amendment = controller.read(path)
    amendment['files_sha256'].pop(str(controller.PREVIOUS_CONTROLLER))
    regression.write(path, amendment)
    with pytest.raises(ValueError, match='omits required source pins'):
        regression.sweep(campaign)


def test_batch8_controller_completes_fake_campaign_once(campaign):
    regression.test_complete_campaign_fits_freezes_submits_audits_and_releases_once(campaign)


def test_batch8_controller_preserves_heldout_failure_gate(campaign):
    regression.test_failed_heldout_keeps_failure_and_never_publishes_admission(campaign)


def test_batch8_controller_serializes_all_arrays(campaign):
    regression.test_confirmation_arrays_depend_on_development_and_prior_arrays_for_global_cap(campaign)


def test_batch8_controller_retains_expanded_jobid_accounting(campaign, monkeypatch):
    regression.test_failure_queries_expand_array_job_ids_instead_of_raw_allocation_ids(campaign, monkeypatch)


def test_batch8_controller_watch_survives_transient_accounting_error(campaign, monkeypatch, capsys):
    regression.test_watch_survives_accounting_timeout_and_reports_recovery(campaign, monkeypatch, capsys)
