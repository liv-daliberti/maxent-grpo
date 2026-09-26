from pathlib import Path
import json
import shutil
import pytest
import build_constructive_code_hardened_20260921 as h


def test_source_identity_survives_snapshot_relocation():
    h.compare_relocated_sources({'/original/repo/src/pkg/a.py':'x','/original/repo/ops/b.py':'y'}, {'/frozen/src/pkg/a.py':'x','ops/b.py':'y'})


def test_same_basename_cannot_substitute_another_module():
    with pytest.raises(ValueError, match='source bytes differ'):
        h.compare_relocated_sources({'src/one/a.py':'x'}, {'src/two/a.py':'x'})


def test_colliding_relative_module_identity_is_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        h.compare_relocated_sources({'/old/src/one/a.py':'x','/new/src/one/a.py':'x'}, {'src/one/a.py':'x'})


def test_changed_verifier_source_cannot_reuse_admission():
    with pytest.raises(ValueError, match='source bytes differ'):
        h.compare_relocated_sources({'ops/a.py':'old'}, {'ops/a.py':'changed'})


def test_pinned_testlib_checked_before_loading(monkeypatch):
    monkeypatch.setattr(h.base,'digest',lambda path:'unexpected')
    with pytest.raises(ValueError,match='testlib source drift'):
        h.require_pinned_testlib()


def test_source_sealed_admitted_panel_and_old_slate_rejection():
    root=h.base.ROOT/'var/data/constructive_code_hardened_initial_20260921'
    if not root.exists():
        pytest.skip('source audit artifact unavailable')
    q=h.validate_quality({'slate_root':str(root)})
    assert q['admitted']==21
    assert {r['task_id'] for r in q['tasks'] if r['status']!='pass'}=={'1073_A','1399_D'}
    with pytest.raises(ValueError,match='strict hardened source-quality gate'):
        h.validate_quality({'slate_root':str(root),'problem_ids':['1073_A']})


def test_weakened_negative_gate_is_rejected_even_with_fresh_file_hash(tmp_path):
    root=h.base.ROOT/'var/data/constructive_code_hardened_initial_20260921'
    if not root.exists():
        pytest.skip('source audit artifact unavailable')
    for name in ('manifest.json','hardening_quality.json','fixed_source_probe_fixture.json'):
        shutil.copy2(root/name,tmp_path/name)
    q=json.loads((tmp_path/'hardening_quality.json').read_text())
    q['tasks'][0]['negative_rejected']=11
    h.base.write_json(tmp_path/'hardening_quality.json',q)
    m=json.loads((tmp_path/'manifest.json').read_text());m['hardening_quality_sha256']=h.base.digest(tmp_path/'hardening_quality.json');h.base.write_json(tmp_path/'manifest.json',m)
    with pytest.raises(ValueError,match='strict hardened source-quality gate'):
        h.validate_quality({'slate_root':str(tmp_path),'problem_ids':[q['tasks'][0]['task_id']]})
