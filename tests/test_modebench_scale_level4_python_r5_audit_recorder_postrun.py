"""Post-run companion to the sealed v2 recorder tests.

`test_the_real_refused_directory_on_disk_is_reconcilable_today` in
tests/test_modebench_scale_level4_python_r5_audit_recorder_v2.py was a pre-flight
binding: it asserted the real execution_audit/ still held nothing but the empty
action.lock the refused first attempt left, so the retry could reconcile it. The
retry then ran and wrote a real audit there, which falsifies that test by design.

That file is sealed 0444 and its digest is pinned by the review that authorized the
run, so it is not edited. This file records the property that holds now, and it is
the stronger one: a completed audit on disk must be REFUSED by the reconciler, so no
future retry can ever replay the claim.
"""
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
RECORDER=ROOT/'artifacts/run_modebench_scale_level4_python_r5_audit_soak_v2_20260913.py'
AUDIT=ROOT/'var/artifacts/modebench_scale_level4_python_r5_20260913/development/execution_audit'


def recorder():
    spec=importlib.util.spec_from_file_location('_v2_recorder_postrun',RECORDER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_the_retry_ran_and_the_real_directory_now_holds_a_complete_audit():
    assert sorted(p.name for p in AUDIT.iterdir())==['action.lock','claim.json','registration.json','tiers']
    assert not (AUDIT/'failure.json').exists()
    claim=json.loads((AUDIT/'claim.json').read_text())
    assert claim['status']=='original_grader_audit_claimed'
    assert claim['host']=='wash.cs.princeton.edu' and int(claim['uid'])==363432
    assert sorted(p.name for p in (AUDIT/'tiers').iterdir())==['0.json','1.json','2.json','3.json']


def test_the_completed_audit_is_now_refused_so_no_retry_can_replay_the_claim():
    with pytest.raises(ValueError,match='never replay any claim|only an empty lock'):
        recorder().reconcile_audit_directory(AUDIT)


def test_the_certificate_the_run_produced_exists_beside_it():
    proof=AUDIT.parent/'execution_reconciliation.json'
    value=json.loads(proof.read_text())
    assert value['array_job_id']==31267007 and value['scheduler_success'] is False
    assert value['exit_cause']=='post_completion_teardown'
    assert sum(r['new_grader_invocations'] for r in
        (json.loads((AUDIT/'tiers'/f'{t}.json').read_text()) for t in range(4)))==24704


def test_the_sealed_preflight_test_is_untouched_and_its_precondition_is_simply_spent():
    """The sealed file must stay byte-identical; only its precondition has expired."""
    sealed=ROOT/'tests/test_modebench_scale_level4_python_r5_audit_recorder_v2.py'
    assert oct(sealed.stat().st_mode)[-3:]=='444' and oct(RECORDER.stat().st_mode)[-3:]=='444'
    assert 'test_the_real_refused_directory_on_disk_is_reconcilable_today' in sealed.read_text()
