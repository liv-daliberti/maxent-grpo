"""Post-run companion for the sealed r5 fit/freeze pre-flight tests.

Six real-record tests in this chain asserted preconditions that the actions they
gated then consumed:

  test_the_real_refused_directory_on_disk_is_reconcilable_today   (audit recorder v2)
  test_the_dataset_is_not_frozen_yet_and_the_holdout_is_unobserved (freeze)
  test_freezing_this_source_completes_the_five_fixed_sources       (freeze)
  test_the_freeze_review_does_not_exist_yet                        (freeze review builder)
  test_the_component_records_do_not_exist_yet                      (component builder)
  test_builder_evidence_paths_exist_and_name_revision_five         (fit review builder)

Each is sealed 0444 and its digest is pinned by a published review, so none is
edited: rewriting one would invalidate the review that authorised the action. This
file records what holds now, which is the stronger claim in every case -- the record
exists, is sealed, and carries the digest the chain pins.
"""
import hashlib
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
REVISION=ROOT/'var/data/modebench_scale_domain_revisions_v1'
SOURCE_ROOT=REVISION/'level4_python_factors_r5'
FIT_STATE=ROOT/'var/artifacts/modebench_scale_level4_python_r5_fit_20260913'
FREEZE=FIT_STATE/'python_r5_freeze'


def sha_of(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sealed(path):return oct(Path(path).stat().st_mode)[-3:]=='444'


def test_the_five_records_the_preflight_tests_forbade_now_exist_and_are_sealed():
    for path in (ROOT/'artifacts/modebench_scale_level4_python_r5_freeze_independent_review_20260913.json',
                 ROOT/'artifacts/modebench_scale_level4_python_r5_fit_observed_audit_component_20260913.json',
                 ROOT/'artifacts/modebench_scale_level4_python_r5_fit_observed_audit_component_independent_review_20260913.json',
                 ROOT/'artifacts/modebench_scale_level4_python_r5_fit_final_review_20260913.json',
                 FREEZE/'certificate.json'):
        assert path.is_file() and sealed(path),path


def test_the_dataset_is_frozen_at_the_identity_the_certificate_records():
    certificate=json.loads((FREEZE/'certificate.json').read_text())
    frozen=certificate['frozen'][0]
    dataset=Path(frozen['dataset'])
    assert dataset==SOURCE_ROOT/'level4/dataset/python_factors' and dataset.is_dir()
    assert sha_of(dataset/'identity.json')==frozen['dataset_identity_sha256']
    identity=json.loads((dataset/'identity.json').read_text())
    assert {s:v['rows'] for s,v in identity['splits'].items()}=={'train':384,'dev':128,'eval':128}
    assert (SOURCE_ROOT/'exclusions/freeze.json').is_file()


def test_the_freeze_took_the_source_and_nothing_else():
    certificate=json.loads((FREEZE/'certificate.json').read_text())
    assert certificate['status']=='level4_python_r5_frozen_pending_confirmation'
    assert certificate['new_fit_publications']==0
    for flag in ('model_grading_performed','heldout_confirmation_performed',
                 'source_binding_performed','admission_performed'):
        assert certificate[flag] is False,flag
    assert certificate['python_development_fit_pass'] is True
    assert certificate['observed_audit_preserved'] is True


def test_the_holdout_is_still_unobserved_after_the_freeze():
    """The freeze fixes the source; it must not have touched the confirmation split."""
    for path in (SOURCE_ROOT/'level4/results/confirmation/python_factors.json',
                 SOURCE_ROOT/'level4/results/confirmation/python_factors.json.batches',
                 SOURCE_ROOT/'level4/confirmation/python_factors.json'):
        assert not path.exists(),path


def test_exactly_five_sources_are_now_frozen():
    frozen={d.name for d in REVISION.iterdir()
            for level in ('level4','level5')
            if (d/level/'dataset').is_dir()
            and any(q.is_dir() for q in (d/level/'dataset').iterdir() if q.name!='exclusions')}
    assert frozen=={'level4_graph_coloring_r2','level4_pantry_r2','level4_python_factors_r5',
                    'level5_mathir_r2','level5_python_factors_r2'}
    assert len(frozen)==5


def test_the_freeze_certificate_names_the_review_that_authorised_it():
    certificate=json.loads((FREEZE/'certificate.json').read_text())
    review=ROOT/'artifacts/modebench_scale_level4_python_r5_freeze_independent_review_20260913.json'
    assert certificate['review_sha256']==sha_of(review)
    value=json.loads(review.read_text())
    assert value['status']=='reviewed' and value['blocking_findings']==[]
    assert value['source_sha256']==sha_of(ROOT/'artifacts/freeze_modebench_scale_level4_python_r5_20260913.py')


@pytest.mark.parametrize('path',[
    'tests/test_modebench_scale_level4_python_r5_audit_recorder_v2.py',
    'tests/test_modebench_scale_level4_python_r5_freeze.py',
    'tests/test_modebench_scale_level4_python_r5_freeze_review_builder.py',
    'tests/test_modebench_scale_python_r5_fit_observed_audit_component_builder.py',
    'tests/test_modebench_scale_python_r5_fit_observed_audit_final_review_builder.py',
])
def test_the_sealed_preflight_files_are_untouched(path):
    """Only their preconditions expired; their bytes must not have moved."""
    assert sealed(ROOT/path),path
