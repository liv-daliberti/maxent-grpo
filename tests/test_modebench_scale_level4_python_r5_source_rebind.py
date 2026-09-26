"""Guards for the Level4 source supersession. No sealed record is ever written here."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/rebind_modebench_scale_level4_python_r5_source_20260914.py'


@pytest.fixture
def a():
    spec=importlib.util.spec_from_file_location('scratch_r5_rebind',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True));return path


@pytest.fixture
def scene(a,tmp_path,monkeypatch):
    """A complete synthetic Level4 selection, with every precondition satisfied."""
    parent=tmp_path/'parent';release=tmp_path/'release';artifacts=tmp_path/'artifacts'
    old=tmp_path/'rev/level4_python_factors_r2';new=tmp_path/'rev/level4_python_factors_r5'
    monkeypatch.setattr(a,'OLD_ROOT',old);monkeypatch.setattr(a,'NEW_ROOT',new)
    monkeypatch.setattr(a,'PRESERVED',tmp_path/'preserved')
    domains=('countdown','graph_coloring','python_factors','mathir','pantry')
    failed={'graph_coloring','python_factors','pantry'}
    roots={'countdown':parent,'mathir':parent,'graph_coloring':tmp_path/'rev/g2',
           'pantry':tmp_path/'rev/p2','python_factors':old}
    sources={d:{'source_root':str(roots[d].resolve()),
                'source_kind':'domain_revision_v1' if d in failed else 'campaign_v1'} for d in domains}
    manifest=write(release/'level4/source_manifest.json',{'level':'level4','sources':sources})
    write(manifest.with_name('source_manifest.sha256.json'),{'sha256':a.sha(manifest)})
    write(old/'level4/recipes/python_factors.json',{'development_fit_pass':False})
    write(new/'level4/recipes/python_factors.json',{'development_fit_pass':True})
    (new/'level4/dataset/python_factors').mkdir(parents=True)
    recipes={d:{'development_fit_pass':d not in failed} for d in domains}
    composite=SimpleNamespace(DEFAULT_PARENT=parent,DEFAULT_RELEASE=release,DEFAULT_ARTIFACTS=artifacts,
        DOMAINS=domains,original=SimpleNamespace(authenticate=lambda p:{'x':1}),
        parent_recipe=lambda p,l,d,proto,advance:recipes[d])
    monkeypatch.setattr(a,'composite',lambda:composite)
    return SimpleNamespace(a=a,parent=parent,release=release,artifacts=artifacts,old=old,new=new,
        manifest=manifest,roots=roots,tmp=tmp_path)


def test_the_clean_scene_passes_and_moves_exactly_one_source(scene):
    value=scene.a.check()
    assert value['status']=='rebind_preconditions_met_no_change_written'
    assert set(value['changing'])=={'python_factors'}
    assert value['heldout_observed'] is False
    assert set(value['unchanged'])=={'graph_coloring','pantry'}
    assert sorted(value['failed_domains'])==['graph_coloring','pantry','python_factors']


def test_check_writes_nothing(scene):
    before={p:p.stat().st_mtime_ns for p in scene.tmp.rglob('*') if p.is_file()}
    scene.a.check()
    after={p:p.stat().st_mtime_ns for p in scene.tmp.rglob('*') if p.is_file()}
    assert before==after and not (scene.tmp/'preserved').exists()


@pytest.mark.parametrize('where',['receipt','batches','audit','new_root'])
def test_any_observed_heldout_blocks_the_switch(scene,where):
    """This is the guard's whole purpose: never switch a source after seeing an outcome."""
    root={'receipt':scene.roots['graph_coloring'],'batches':scene.roots['pantry'],
          'audit':scene.roots['graph_coloring'],'new_root':scene.new}[where]
    name={'receipt':'level4/results/confirmation/graph_coloring.json',
          'batches':'level4/results/confirmation/pantry.json.batches',
          'audit':'level4/confirmation/graph_coloring.json',
          'new_root':'level4/results/confirmation/python_factors.json'}[where]
    target=root/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.mkdir() if where=='batches' else target.write_text('{}')
    with pytest.raises(ValueError,match='cherry-picking'):scene.a.check()


def test_an_unfrozen_replacement_is_refused(scene):
    import shutil;shutil.rmtree(scene.new/'level4/dataset/python_factors')
    with pytest.raises(ValueError,match='must be frozen'):scene.a.check()


def test_a_replacement_whose_fit_failed_is_refused(scene):
    write(scene.new/'level4/recipes/python_factors.json',{'development_fit_pass':False})
    with pytest.raises(ValueError,match='must be the passing fit'):scene.a.check()


def test_refuses_when_the_manifest_no_longer_names_revision_two(scene):
    value=json.loads(scene.manifest.read_text())
    value['sources']['python_factors']['source_root']=str(scene.new.resolve())
    write(scene.manifest,value)
    write(scene.manifest.with_name('source_manifest.sha256.json'),{'sha256':scene.a.sha(scene.manifest)})
    with pytest.raises(ValueError,match='still name revision 2'):scene.a.check()


def test_a_tampered_digest_sidecar_is_refused(scene):
    write(scene.manifest.with_name('source_manifest.sha256.json'),{'sha256':'0'*64})
    with pytest.raises(ValueError,match='digest sidecar disagrees'):scene.a.check()


def test_the_mapping_comes_from_the_manifest_not_the_bindings(scene):
    """The bindings file is stale -- it omits pantry -- so it must not be the source."""
    write(scene.artifacts/'revision_bindings.json',
        {'schema':'modebench_scale_composite_revision_bindings_v1',
         'revisions':{'level4':{'graph_coloring':'/wrong','python_factors':'/wrong'}}})
    value=scene.a.check()
    assert value['unchanged']['graph_coloring']==str(scene.roots['graph_coloring'].resolve())
    assert value['unchanged']['pantry']==str(scene.roots['pantry'].resolve())


def test_publish_is_refused_outright(scene):
    """The design was refused after review: it would break four sealed records at once."""
    with pytest.raises(ValueError,match='refused: this supersession'):scene.a.publish()


def test_publish_refuses_before_touching_anything(scene):
    before={p:p.stat().st_mtime_ns for p in scene.tmp.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):scene.a.publish()
    assert {p:p.stat().st_mtime_ns for p in scene.tmp.rglob('*') if p.is_file()}==before
    assert not (scene.tmp/'preserved').exists()


def test_the_refusal_record_exists_and_names_the_four_sealed_records(a):
    record=json.loads(Path(a.REFUSAL).read_text())
    assert record['status']=='refused_never_published'
    assert record['refused']['publish_ever_run'] is False
    pinned={Path(e['record']).name for e in record['blocking_pin_lattice']}
    assert pinned=={'source_set_identity.json','controller_activation.json',
                    'controller_identity.json','plan.json'}


# --- Real-record bindings -----------------------------------------------------

def test_the_real_level4_selection_still_names_revision_two(a):
    release=ROOT/'var/data/modebench_scale_release_v1/level4'
    sources=json.loads((release/'source_manifest.json').read_text())['sources']
    assert Path(sources['python_factors']['source_root']).name=='level4_python_factors_r2'
    assert json.loads((release/'source_manifest.sha256.json').read_text())['sha256']==a.sha(release/'source_manifest.json')


def test_the_real_replacement_passed_and_froze_and_the_old_one_did_not(a):
    rev=ROOT/'var/data/modebench_scale_domain_revisions_v1'
    old=json.loads((rev/'level4_python_factors_r2/level4/recipes/python_factors.json').read_text())
    new=json.loads((rev/'level4_python_factors_r5/level4/recipes/python_factors.json').read_text())
    assert old['development_fit_pass'] is False and new['development_fit_pass'] is True
    assert (rev/'level4_python_factors_r5/level4/dataset/python_factors').is_dir()
    assert not (rev/'level4_python_factors_r2/level4/dataset/python_factors').is_dir()


def test_no_real_level4_heldout_has_been_observed(a):
    """If this ever fails, the supersession is no longer legitimate and must stop."""
    rev=ROOT/'var/data/modebench_scale_domain_revisions_v1'
    roots=[ROOT/'var/data/modebench_scale_v1',rev/'level4_graph_coloring_r2',
           rev/'level4_pantry_r2',rev/'level4_python_factors_r2',rev/'level4_python_factors_r5']
    for root in roots:
        for domain in ('countdown','graph_coloring','python_factors','mathir','pantry'):
            receipt=root/'level4/results/confirmation'/(domain+'.json')
            assert not receipt.exists(),receipt
            assert not Path(str(receipt)+'.batches').exists()
            assert not (root/'level4/confirmation'/(domain+'.json')).exists()


def test_the_preservation_directory_does_not_exist_yet(a):
    assert not Path(a.PRESERVED).exists()
