"""A completion audit cannot convert partial or altered evidence into a pass."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import audit_modebench_fresh_completion as m


def scope_plan(tmp_path):
    tasks=[]
    for scale,seeds in m.SEEDS.items():
        for domain in m.DOMAINS:
            for method in ('initial',*m.METHODS):
                for seed in seeds:
                    tasks.append({'task_id':f'{scale}-{domain}-{method}-{seed}','model_scale':scale,
                                  'domain':domain,'method':method,'level':1,
                                  'checkpoint_stage':'initial' if method=='initial' else 'terminal',
                                  'training_seed':None if method=='initial' else seed,
                                  'eval_replica_id':seed if method=='initial' else None,
                                  'files':[{'name':'model.safetensors','sha256':scale}]})
    return {'tasks':tasks,'code_sha256':{str(i):'hash' for i in range(112)},
            'prompts_per_task':128,'draws_per_prompt':64,'output_root':str(tmp_path/'results')}


def test_scope_keeps_all_four_methods_and_initial_mc_semantics(tmp_path):
    plan=scope_plan(tmp_path);m.validate_scope(plan)
    bad=copy.deepcopy(plan);bad['tasks'][-1]=copy.deepcopy(bad['tasks'][-2])
    with pytest.raises(ValueError,match='Duplicate|inventory'):
        m.validate_scope(bad)
    bad=copy.deepcopy(plan);bad['tasks'][0]['training_seed']=43
    with pytest.raises(ValueError,match='Monte Carlo'):
        m.validate_scope(bad)


def test_scope_rejects_changed_seeds_and_different_initial_domain_weights(tmp_path):
    plan=scope_plan(tmp_path)
    for task in plan['tasks']:
        if task['model_scale']=='qwen05b':
            key='eval_replica_id' if task['checkpoint_stage']=='initial' else 'training_seed'
            task[key]+=100
    with pytest.raises(ValueError,match='inventory'):
        m.validate_scope(plan)
    plan=scope_plan(tmp_path)
    for task in plan['tasks']:
        if task['model_scale']=='qwen05b' and task['domain']=='pantry_plan' and task['checkpoint_stage']=='initial':
            task['files']=[{'name':'model.safetensors','sha256':'substituted'}]
    with pytest.raises(ValueError,match='share the same initial'):
        m.validate_scope(plan)


def test_missing_deliverables_fail_before_response_or_model_reads(tmp_path):
    plan=scope_plan(tmp_path);pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan))
    with patch.object(m,'PLAN_SHA256',m.digest(pp)),patch.object(m.panel,'build_report') as authenticate,patch.object(m,'validate_restoration_metadata') as weights:
        with pytest.raises(m.Incomplete) as error:m.audit_campaign(tmp_path)
    authenticate.assert_not_called();weights.assert_not_called()
    missing=error.value.missing
    assert str(tmp_path/'fresh_vs_retrospective/report.json') in missing
    assert str(tmp_path/'fresh_figures/manifest.json') in missing
    assert str(tmp_path/'INTERPRETATION.md') in missing
    assert len([p for p in missing if '/results/' in p])==150


def test_one_missing_task_stays_incomplete_even_with_deliverable_stubs(tmp_path):
    plan=scope_plan(tmp_path)
    with pytest.raises(m.Incomplete) as initial:m.preflight(tmp_path,plan)
    for name in initial.value.missing:
        path=Path(name);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    missing=Path(plan['output_root'])/plan['tasks'][149]['task_id']/'result.json';missing.unlink()
    with pytest.raises(m.Incomplete) as error:m.preflight(tmp_path,plan)
    assert error.value.missing==[str(missing)]


def test_manifest_rejects_changed_missing_and_unlisted_outputs(tmp_path):
    report=tmp_path/'report.json';report.write_text('{}')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'files':{'report.json':m.digest(report)}}))
    m.read_publication(tmp_path,['report.json'])
    report.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='hash differs'):m.read_publication(tmp_path,['report.json'])
    report.write_text('{}');(tmp_path/'extra.csv').write_text('unregistered')
    with pytest.raises(ValueError,match='Unlisted'):m.read_publication(tmp_path,['report.json'])
    (tmp_path/'extra.csv').unlink()
    with pytest.raises(ValueError,match='Required files'):m.read_publication(tmp_path,['report.json','comparisons.csv'])


def test_manifest_cannot_escape_publication_directory(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'files':{'../outside':'hash'}}))
    with pytest.raises(ValueError,match='Unsafe'):m.read_publication(tmp_path,[])


def old_contrasts():
    result=[]
    for grade in ('strict','normalized_secondary'):
        for wording in ('original','neutral'):
            for level in (2,3):
                for domain,seeds in m.analysis.EXPECTED_SEEDS.items():
                    for left,right in m.analysis.CONTRASTS:
                        result.append({'grading':grade,'wording':wording,'level':level,'domain':domain,
                                       'contrast':right+'_minus_'+left,'left_method':left,'right_method':right,
                                       'summary':{'independent_initial_checkpoints':1},
                                       'seeds':[{'training_seed':s,'expected_prompts':16,
                                                 'prompts':[{'prompt_id':str(i)} for i in range(16)]} for s in seeds]})
    return result


def test_old_audit_requires_all_initial_replay_and_sensitivity_comparisons():
    contrasts=old_contrasts();m.validate_contrasts(contrasts,fresh=False)
    for predicate in (lambda c:c['left_method']=='initial',lambda c:c['grading']=='normalized_secondary'):
        incomplete=[c for c in contrasts if not predicate(c)]
        with pytest.raises(ValueError,match='Contrast inventory'):
            m.validate_contrasts(incomplete,fresh=False)
    contrasts[-1]['seeds'].pop()
    with pytest.raises(ValueError,match='comparison seeds'):
        m.validate_contrasts(contrasts,fresh=False)


def test_cli_incomplete_never_writes_success_artifact(tmp_path,capsys):
    output=tmp_path/'audit.json'
    with patch.object(sys,'argv',['audit','--output',str(output)]),patch.object(m,'audit_campaign',side_effect=m.Incomplete(['missing'])):
        assert m.main()==2
    assert not output.exists()
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='incomplete' and result['missing']==['missing']


def test_numeric_reconstruction_tolerance_does_not_relax_inventory_or_hashes():
    assert m.reconstructed_equal({'effect':0.1},{'effect':0.10000000000000002})
    assert not m.reconstructed_equal({'effect':0.1},{'effect':0.1000001})
    assert not m.reconstructed_equal({'count':5},{'count':5.0})
    assert not m.reconstructed_equal({'seed':43},{'seed':44})
    assert not m.reconstructed_equal({'sha256':'a'},{'sha256':'b'})
    assert not m.reconstructed_equal([1,2],[2,1])
