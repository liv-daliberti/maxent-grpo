"""Offline checks of public archive mappings and stable paper links."""
import copy
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('archive_public_docs_under_test', ROOT / 'ops/archive_public_docs.py')
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)

@pytest.fixture
def source():
    plan = json.loads((ROOT / 'var/artifacts/hf_model_archive_20260911/all/plan.json').read_text())
    rows = [{'experiment': docs.STUDIES[m['source_experiment']]['name'], 'model': docs.MODELS[m['model_key']],
             'domain': m['domain'], 'method': m['source_arm'], 'seed': m['seed'], 'step': m['terminal_step'],
             'repo_prefix': m['repo_prefix'], 'commit_sha': 'a' * 40, 'local_weights_retired': True}
            for m in plan['models']]
    return plan, {'repo_id': plan['repo_id'], 'expected_model_count': len(rows), 'verified_model_count': len(rows), 'models': rows}

def assert_page_links(pages):
    for name, text in pages.items():
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith('https://'): continue
            path, _, fragment = target.partition('#')
            if path == 'catalog.json': continue
            if path:
                parts = list(PurePosixPath(name).parent.parts)
                for part in PurePosixPath(path).parts:
                    if part == '..': parts.pop()
                    elif part != '.': parts.append(part)
                resolved = '/'.join(parts)
            else: resolved = name
            assert resolved in pages, (name, target)
            if fragment: assert docs.anchor(fragment) in pages[resolved], (name, target)

def test_all_selected_models_map_once_and_paper_anchors_exist(source):
    pages = docs.render_readmes(*source)
    assert len(pages) == 7
    assert '418 / 418 selected exports verified' in pages['README.md']
    assert 'does not imply that every paper artifact' in pages['README.md']
    for study in docs.STUDIES.values():
        page = pages[docs.landing(study['name'])]
        for name in ('models', 'comparators', 'study-design', 'archive-coverage'):
            assert docs.anchor(name) in page
        for domain in docs.DOMAINS: assert docs.anchor('domain-' + domain) in page
        assert 'optimizer step 3072' in page and 'export-step label is 3073' in page
    assert_page_links(pages)

def test_zero_verified_models_have_no_unavailable_folder_links(source):
    plan, catalog = source
    catalog.update(models=[], verified_model_count=0)
    pages = docs.render_readmes(plan, catalog)
    assert len(pages) == 7
    assert '0 / 418 selected exports verified' in pages['README.md']
    for text in pages.values(): assert '/tree/' not in text
    assert_page_links(pages)

def test_partial_publication_links_only_actual_verified_model(source):
    plan, catalog = source
    catalog.update(models=catalog['models'][:1], verified_model_count=1)
    pages = docs.render_readmes(plan, catalog)
    public = '\n'.join(pages.values())
    assert catalog['models'][0]['repo_prefix'] in public
    assert plan['models'][1]['repo_prefix'] not in public
    assert '/tree/' + 'a' * 40 + '/' in public
    assert_page_links(pages)

def test_uniform_comparators_keep_source_experiments_without_duplicates(source):
    plan, catalog = source
    page = docs.render_readmes(plan, catalog)['experiments/E120-R1/README.md']
    assert '45 uniform comparators' in page
    for exp in ('E78', 'E79', 'E80-R1'): assert f'../{exp}/README.md#domain-' in page
    assert len([m for m in plan['models'] if m['campaign'] == 'e120_uniform_comparators']) == 45
    assert len(catalog['models']) == 418

@pytest.mark.parametrize('change', ['commit', 'duplicate', 'unadmitted', 'method', 'count', 'repo'])
def test_invalid_catalog_fails_closed(source, change):
    plan, catalog = source
    if change == 'commit': catalog['models'][0]['commit_sha'] = 'main'
    elif change == 'duplicate': catalog['models'][1] = copy.deepcopy(catalog['models'][0])
    elif change == 'unadmitted': catalog['models'][0]['repo_prefix'] = 'experiments/NEW/model'
    elif change == 'method': catalog['models'][0]['method'] = 'other'
    elif change == 'count': catalog['verified_model_count'] -= 1
    elif change == 'repo': catalog['repo_id'] = 'other/repo'
    with pytest.raises(ValueError): docs.render_readmes(plan, catalog)

def test_private_plan_fields_never_escape(source):
    text = '\n'.join(docs.render_readmes(*source).values())
    for private in ('/n/fs/', 'debug_job', 'job_id', 'run_stamp', 'token-file', 'terminal_export'):
        assert private not in text

def test_specific_licenses_and_required_attribution(source):
    text = docs.render_readmes(*source)['README.md']
    assert 'Qwen2.5-3B-Instruct | Qwen Research License' in text
    assert 'Qwen2.5-0.5B-Instruct | Apache License 2.0' in text
    assert 'Falcon3-1B-Instruct | TII Falcon-LLM License 2.0' in text
    assert 'Built with Qwen' in text
    assert 'built using artificial intelligence technology from the Technology Innovation Institute' in text

def test_navigation_links_require_explicit_available_sections(source):
    root = docs.render_readmes(*source)['README.md']
    assert 'Data (in preparation)' in root and '[Data](' not in root
    updated = docs.render_readmes(*source, available_sections={'Data': 'data/README.md'})['README.md']
    assert '[Data](data/README.md)' in updated
    with pytest.raises(ValueError): docs.render_readmes(*source, available_sections={'Data': '../private'})

def test_future_studies_require_explicit_new_definitions(source):
    plan, catalog = source
    model = copy.deepcopy(plan['models'][0]);model.update(source_experiment='e999',source_arm='new_method')
    model['repo_prefix'] = model['repo_prefix'].replace('/E118/', '/E999/').replace('/maxrl/', '/new_method/')
    plan['models'].append(model);plan['expected_model_count']+=1;catalog['expected_model_count']+=1
    with pytest.raises(KeyError): docs.render_readmes(plan, catalog)
    pages = docs.render_readmes(plan,catalog,additional_studies={'e999':{'name':'E999','title':'Future cohort','registered':1,'question':'Future question','design':'Future design','paper':'Future scope'}},additional_methods={'new_method':('Future method','Future description')})
    assert 'experiments/E999/README.md' in pages
    assert '/tree/' not in pages['experiments/E999/README.md']
    with pytest.raises(ValueError): docs.render_readmes(*source,additional_studies={'e118':{}})
