import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('fresh_restore',Path(__file__).resolve().parents[1]/'ops/restore_modebench_fresh_concentration.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture_record(tmp_path, archived=False):
    run=tmp_path/'original';run.mkdir();model=run/'saved';model.mkdir()
    files=[]
    for name,raw in [('config.json',b'{}'),('model.safetensors',b'original trained weights')]:
        p=model/name;p.write_bytes(raw)
        files.append({'name':name,'bytes':len(raw),'sha256':m.digest(p)})
    completion=run/'TRAINING_COMPLETE.json';completion.write_text(json.dumps({'terminal_export':str(model)}))
    r={'cell_id':'level1/qwen05b/graph_coloring/drgrpo/43','model_path':str(model),'files':files,
       'completion_receipt':{'path':str(completion),'sha256':m.digest(completion)}}
    if archived:
        archive=tmp_path/'archive';archive.mkdir();entries=[]
        for x in files:
            local=model/x['name'];saved=archive/x['name'];saved.write_bytes(local.read_bytes())
            entries.append({'relative_path':x['name'],'size':x['bytes'],'sha256':x['sha256'],'path_in_repo':'model/'+x['name']})
        manifest=archive/'manifest.json';manifest.write_text(json.dumps({'files':entries}))
        receipt=run/'MODEL_ARCHIVE.json';receipt.write_text(json.dumps({'commit_sha':'a'*40,'repo_id':'owner/models','manifest_sha256':m.digest(manifest)}))
        r['archive_manifest']={'path':str(manifest),'sha256':m.digest(manifest)}
        r['restore']={'receipt':{'path':str(receipt),'sha256':m.digest(receipt)},'revision':'a'*40,'repo_id':'owner/models'}
        (model/'model.safetensors').unlink()
    return r


def test_verified_copy_is_idempotent_and_leaves_original_bytes(tmp_path):
    r=fixture_record(tmp_path);original={p.name:p.read_bytes() for p in Path(r['model_path']).iterdir()}
    one=m.restore_record(r,cache=tmp_path/'cache');two=m.restore_record(r,cache=tmp_path/'cache')
    assert one==two
    assert {p.name:p.read_bytes() for p in Path(r['model_path']).iterdir()}==original
    assert one['original_run_modified'] is False


def test_archived_weights_use_pinned_public_revision_and_verified_download(tmp_path):
    r=fixture_record(tmp_path,archived=True);calls=[]
    def download(**kw):
        calls.append(kw);return str(tmp_path/'archive'/Path(kw['filename']).name)
    result=m.restore_record(r,cache=tmp_path/'cache',downloader=download)
    assert calls[0]['revision']=='a'*40 and calls[0]['token'] is False
    assert not (Path(r['model_path'])/'model.safetensors').exists()
    assert Path(result['model_path'],'model.safetensors').read_bytes()==b'original trained weights'


def test_wrong_download_cannot_publish_verified_receipt(tmp_path):
    r=fixture_record(tmp_path,archived=True)
    bad=tmp_path/'bad';bad.write_bytes(b'wrong weights')
    with pytest.raises(ValueError,match='restoration source differs'):
        m.restore_record(r,cache=tmp_path/'cache',downloader=lambda **_:str(bad))
    assert not list((tmp_path/'cache'/'receipts').glob('*.json'))


def test_changed_existing_target_is_never_overwritten(tmp_path):
    r=fixture_record(tmp_path);result=m.restore_record(r,cache=tmp_path/'cache')
    target=Path(result['model_path'])/'model.safetensors';target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='restored target differs'):
        m.restore_record(r,cache=tmp_path/'cache')
    assert target.read_bytes()==b'changed'


@pytest.mark.parametrize('name',['../weights','/weights',''])
def test_model_paths_cannot_escape_target(name):
    with pytest.raises(ValueError,match='invalid model-relative path'):m.valid_relative(name)


def test_hub_snapshot_symlink_is_resolved_and_dangling_campaign_link_repaired(tmp_path):
    r=fixture_record(tmp_path,archived=True)
    cache=tmp_path/'cache'
    blob=cache/'hub'/'blobs'/'weights';blob.parent.mkdir(parents=True)
    blob.write_bytes(b'original trained weights')
    snapshot=cache/'hub'/'snapshots'/'revision'/'model.safetensors'
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to('../../blobs/weights')
    destination=m.target_path(r,cache)/'model.safetensors'
    destination.parent.mkdir(parents=True)
    destination.symlink_to('../../blobs/weights')
    assert destination.is_symlink() and not destination.exists()
    result=m.restore_record(r,cache=cache,downloader=lambda **_:str(snapshot))
    assert not destination.is_symlink()
    assert destination.read_bytes()==blob.read_bytes()
    assert destination.stat().st_ino==blob.stat().st_ino
    assert result['files'][1]['sha256']==m.digest(blob)
