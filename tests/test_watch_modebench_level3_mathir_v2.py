"""Focused MathIR-only watcher isolation checks; no fitting or scheduler actions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import pytest

SPEC=importlib.util.spec_from_file_location('mathir_only_watcher',Path(__file__).resolve().parents[1]/'ops/exp_scaling/watch_modebench_level3_mathir_v2.py')
watcher=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(watcher)


def exclusive_json(path,value):
    with Path(path).open('x') as handle:json.dump(value,handle)


def test_fit_receives_exactly_mathir_baseline_and_four_candidates(tmp_path):
    receipts=[tmp_path/name.name for name in watcher.RECEIPTS]
    for path in receipts:path.write_text('{}')
    recipe_path,report_path=tmp_path/'mathir.json',tmp_path/'mathir_report.json'
    result={'decision':'registered-decision','weights':[0,0,1,0],'development':{}}
    calls=[]
    def fit(baseline,candidates,domain,output):
        calls.append((baseline,candidates,domain,output))
        exclusive_json(output,result)
        return result
    with patch.object(watcher,'RECEIPTS',receipts),patch.object(watcher,'RECIPE',recipe_path),\
         patch.object(watcher,'REPORT',report_path),patch.object(watcher,'verify') as verify,\
         patch.dict('sys.modules',{'fit_modebench_level3_independent':SimpleNamespace(fit_recipe=fit),
                                  'evaluate_modebench_level3':SimpleNamespace(atomic_new=exclusive_json)}):
        report=watcher.fit_once('source-sha')
    assert calls==[(receipts[0],receipts[1:],'mathir',recipe_path)]
    assert [path.name for path in receipts]==['calibration_05b_mathir.json',*[f'calibration_3b_mathir_d{i}.json' for i in range(4)]]
    assert report['domain']=='mathir' and report['confirmation_outcomes_used'] is False
    assert verify.call_count==2 and report_path.exists()


@pytest.mark.parametrize('existing',['recipe','report'])
def test_existing_mathir_recipe_or_report_prevents_any_refit(tmp_path,existing):
    recipe,report=tmp_path/'recipe',tmp_path/'report'
    (recipe if existing=='recipe' else report).write_text('preserve existing bytes')
    fit=Mock()
    with patch.object(watcher,'RECIPE',recipe),patch.object(watcher,'REPORT',report),patch.object(watcher,'verify'),\
         patch.dict('sys.modules',{'fit_modebench_level3_independent':SimpleNamespace(fit_recipe=fit)}):
        with pytest.raises(FileExistsError,match='never fit a second recipe'):watcher.fit_once('source-sha')
    fit.assert_not_called()
    assert (recipe if existing=='recipe' else report).read_text()=='preserve existing bytes'


def test_changed_frozen_input_blocks_before_fitter_import_or_publication(tmp_path):
    recipe,report=tmp_path/'recipe',tmp_path/'report';fit=Mock()
    with patch.object(watcher,'RECIPE',recipe),patch.object(watcher,'REPORT',report),\
         patch.object(watcher,'verify',side_effect=ValueError('Frozen scientific/execution input changed')),\
         patch.dict('sys.modules',{'fit_modebench_level3_independent':SimpleNamespace(fit_recipe=fit)}):
        with pytest.raises(ValueError,match='Frozen scientific'):watcher.fit_once('source-sha')
    fit.assert_not_called();assert not recipe.exists() and not report.exists()
