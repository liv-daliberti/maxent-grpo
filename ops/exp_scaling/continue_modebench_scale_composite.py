#!/usr/bin/env python3
"""Advance development-only source selection into an audited composite release.

Activation is explicit and sealed. The predecessor must be stopped before this
controller becomes the sole submission authority. Failed development or heldout
gates never trigger automatic replacement selection or another submission.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
import continue_modebench_scale_runtime_v3 as previous
import fit_modebench_scale as original_fit
import materialize_modebench_scale as original
import modebench_scale_domain_revision as revision
import launch_modebench_scale_domains as launch
from fit_modebench_level3 import atomic_new, file_sha, sha
from materialize_modebench_harder_v2 import SPLITS, load_rows

read, require = original.read, original.require
DOMAINS, LEVELS = original.DOMAINS, original.LEVELS
DEFAULT_PARENT = ROOT / 'var/data/modebench_scale_v1'
DEFAULT_RELEASE = ROOT / 'var/data/modebench_scale_release_v1'
DEFAULT_ARTIFACTS = ROOT / 'var/artifacts/modebench_scale_composite_v1'
PARENT_ARTIFACTS = ROOT / 'var/artifacts/modebench_scale_runtime_v3'
PARENT_ARRAY_JOB_ID = 31243495
PREVIOUS_CONTROLLER_PID = 2511076
MANIFEST_SCHEMA = 'modebench_scale_composite_sources_v1'
ACTIVATION_SCHEMA = 'modebench_scale_composite_activation_v1'
POLICY = 'carry_if_development_fit_pass_else_registered_revision'
TERMINAL = {'admitted', 'needs_new_development_revision', 'heldout_confirmation_failed',
            'execution_failed', 'submission_needs_reconciliation', 'generation_needs_reconciliation'}


def verify_dataset(root, level, domain):
    from modebench_scale_source_disjointness import verify_dataset as verify
    return verify(root, level, domain)


def pin_files(paths):
    return {str(Path(path).resolve()): file_sha(Path(path)) for path in paths}


def check_pins(pins):
    require(isinstance(pins, dict) and pins, 'nonempty immutable input pins required')
    for path, digest in pins.items():
        require(file_sha(Path(path)) == digest, 'recorded input changed: ' + path)


def domain_paths(root, level, domain):
    root = Path(root).resolve()
    return {'protocol': root / 'protocol.json', 'recipe': root / level / 'recipes' / (domain + '.json'),
            'pools': root / level / 'pools' / domain, 'dataset': root / level / 'dataset' / domain,
            'receipt': root / level / 'results/confirmation' / (domain + '.json'),
            'audit': root / level / 'confirmation' / (domain + '.json')}


def development_receipts(root, level, domain):
    return [Path(root) / level / 'results/development' / domain / f'difficulty_{tier}.json' for tier in range(4)]


def parent_recipe(parent, level, domain, protocol, *, advance):
    if not all(path.is_file() for path in development_receipts(parent, level, domain)):
        return None
    return previous.recipe(parent, level, domain, protocol, advance=advance)


def bind_level(parent, release_root, level, revisionsdict):
    """Freeze one deterministic source map; explicit mappings are required for failures."""
    parent, release_root = Path(parent).resolve(), Path(release_root).resolve()
    require(level in LEVELS and isinstance(revisionsdict, dict), 'valid level and revision mapping required')
    protocol = original.authenticate(parent / 'protocol.json')
    require(all(domain_paths(parent, item, domain)['recipe'].is_file() for item in LEVELS for domain in DOMAINS),
            'all ten original development fits must exist before source binding')
    recipes = {domain: parent_recipe(parent, level, domain, protocol, advance=False) for domain in DOMAINS}
    require(all(value is not None for value in recipes.values()), 'complete original development evidence required')
    failed = {domain for domain, value in recipes.items() if not value['development_fit_pass']}
    require(set(revisionsdict) == failed, 'revision mapping must cover exactly the failed original domains')
    destination = release_root / level / 'source_manifest.json'
    sources, pins = {}, pin_files([parent / 'protocol.json'])
    for domain in DOMAINS:
        recipe = recipes[domain]
        require(recipe == original_fit.fit_domain(parent, level, domain, publish=False), 'original recipe is not reproducible')
        parent_path = domain_paths(parent, level, domain)['recipe']
        pins.update(pin_files([parent_path])); pins.update(recipe['input_sha256'])
        root = Path(revisionsdict[domain]).resolve() if domain in failed else parent
        kind = 'domain_revision_v1' if domain in failed else 'campaign_v1'
        p = revision.authenticate(root) if domain in failed else protocol
        if domain in failed:
            require(p['level'] == level and p['domain'] == domain and Path(p['parent_root']).resolve() == parent
                    and p['parent_protocol_sha256'] == file_sha(parent / 'protocol.json')
                    and p['parent_recipe_sha256'] == file_sha(parent_path)
                    and p['models'] == protocol['models'] and p['targets'][domain] == protocol['targets'][domain]
                    and p['histograms'][domain] == protocol['histograms'][domain]
                    and p['tolerances'] == protocol['tolerances'], 'revision differs from selected parent domain')
            pins.update(p['files_sha256']); pins.update(pin_files([root / 'protocol.sha256.json']))
        paths = domain_paths(root, level, domain)
        if not destination.exists():
            for receipt in (domain_paths(parent, level, domain)['receipt'], paths['receipt']):
                require(not receipt.exists() and not Path(str(receipt) + '.batches').exists(),
                        'source choice requires unobserved heldout outcomes')
        pins.update(pin_files([paths['protocol']]))
        sources[domain] = {'source_kind': kind, 'source_root': str(root),
                           'protocol_sha256': file_sha(paths['protocol']),
                           'parent_recipe_sha256': file_sha(parent_path)}
    value = {'schema': MANIFEST_SCHEMA, 'level': level, 'model_label': LEVELS[level],
             'parent_root': str(parent), 'parent_protocol_sha256': file_sha(parent / 'protocol.json'),
             'selection_policy': POLICY, 'targets': protocol['targets'], 'sources': sources, 'files_sha256': pins}
    if destination.exists():
        require(read(destination) == value, 'immutable source selection changed')
    else:
        atomic_new(destination, value)
    immutable_record(destination.with_name('source_manifest.sha256.json'), {'sha256': file_sha(destination)})
    return value


def source_manifest(parent, release_root, level):
    path = release_root / level / 'source_manifest.json'
    require(read(path.with_name('source_manifest.sha256.json')).get('sha256') == file_sha(path), 'source manifest changed')
    value = read(path); check_pins(value['files_sha256'])
    protocol = original.authenticate(parent / 'protocol.json')
    require(value.get('schema') == MANIFEST_SCHEMA and value.get('level') == level
            and value.get('model_label') == LEVELS[level] and value.get('parent_root') == str(parent)
            and value.get('parent_protocol_sha256') == file_sha(parent / 'protocol.json')
            and value.get('targets') == protocol['targets'] and value.get('selection_policy') == POLICY
            and set(value['sources']) == set(DOMAINS), 'wrong source manifest')
    for domain, source in value['sources'].items():
        recipe = parent_recipe(parent, level, domain, protocol, advance=False)
        require(recipe is not None, 'source manifest lost its original recipe')
        paths = domain_paths(parent, level, domain); root = Path(source['source_root']).resolve()
        expected_kind = 'campaign_v1' if recipe['development_fit_pass'] else 'domain_revision_v1'
        require(source['source_kind'] == expected_kind and source['parent_recipe_sha256'] == file_sha(paths['recipe'])
                and source['protocol_sha256'] == file_sha(root / 'protocol.json')
                and value['files_sha256'].get(str(root / 'protocol.json')) == source['protocol_sha256'],
                'source routing or protocol changed')
        if expected_kind == 'campaign_v1':
            require(root == parent, 'passing original domain must be carried')
        else:
            p = revision.authenticate(root)
            require(p['level'] == level and p['domain'] == domain and Path(p['parent_root']).resolve() == parent
                    and p['parent_protocol_sha256'] == file_sha(parent / 'protocol.json')
                    and p['parent_recipe_sha256'] == file_sha(paths['recipe'])
                    and p['models'] == protocol['models'] and p['targets'][domain] == protocol['targets'][domain]
                    and p['histograms'][domain] == protocol['histograms'][domain]
                    and p['tolerances'] == protocol['tolerances'], 'revision source policy changed')
    return value


def old_controller_running():
    """Reject the recorded predecessor and any replacement process running its script."""
    expected = str(Path(previous.__file__).resolve())
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = (directory / 'cmdline').read_bytes().split(b'\0')
        except (OSError, ProcessLookupError):
            continue
        if expected.encode() in args and b'--advance' in args:
            return True
    return False


def revision_bindings(artifacts):
    path = artifacts / 'revision_bindings.json'
    if not path.exists(): return {}
    value = read(path)
    require(value.get('schema') == 'modebench_scale_composite_revision_bindings_v1'
            and isinstance(value.get('revisions'), dict) and set(value['revisions']) <= set(LEVELS),
            'wrong explicit revision bindings')
    for level, domains in value['revisions'].items():
        require(isinstance(domains, dict) and set(domains) <= set(DOMAINS), 'unknown revision binding domain')
        for root in domains.values():
            require(isinstance(root, str) and Path(root).is_absolute() and str(Path(root).resolve()) == root,
                    'canonical explicit revision root required')
    return value['revisions']


def activation_sources(parent, artifacts=DEFAULT_ARTIFACTS):
    pointers = []
    if (artifacts / 'revision_bindings.json').exists():
        pointers.append(artifacts / 'revision_bindings.json')
        for domains in revision_bindings(artifacts).values():
            for root in domains.values():
                pointers.extend([Path(root) / 'protocol.json', Path(root) / 'protocol.sha256.json'])
    return [*pointers,Path(__file__).resolve(), Path(previous.__file__).resolve(), Path(original.__file__).resolve(),
            Path(original_fit.__file__).resolve(), Path(revision.__file__).resolve(), Path(launch.__file__).resolve(),
            ROOT / 'ops/exp_scaling/modebench_scale_source_disjointness.py', parent / 'protocol.json',
            PARENT_ARTIFACTS / 'runtime_amendment.json', PARENT_ARTIFACTS / 'development/plan.json',
            PARENT_ARTIFACTS / 'development/submission_intent.json', PARENT_ARTIFACTS / 'development/submission_result.json',
            PARENT_ARTIFACTS / 'controller_arm_result.json']


def validate_activation(parent, release_root, artifacts):
    path = artifacts / 'controller_activation.json'
    require(path.is_file(), 'sealed composite activation required before advance')
    value = read(path)
    require(value.get('schema') == ACTIVATION_SCHEMA and value.get('parent_root') == str(parent)
            and value.get('release_root') == str(release_root) and value.get('artifacts_root') == str(artifacts)
            and value.get('parent_array_job_id') == PARENT_ARRAY_JOB_ID
            and value.get('previous_controller_pid') == PREVIOUS_CONTROLLER_PID
            and value.get('previous_controller_stopped') is True
            and value.get('host') == os.uname().nodename and value.get('source_selection_policy') == POLICY
            and value.get('runtime_profile') == launch.RUNTIME_PROFILE,
            'composite activation scope or runtime changed')
    check_pins(value.get('files_sha256'))
    require(all(str(source) in value['files_sha256'] for source in activation_sources(parent, artifacts)),
            'activation omits required source/pointer pins')
    require(not old_controller_running(), 'previous controller is still running; sole submission authority required')
    previous.runtime_amendment(parent, PARENT_ARTIFACTS)
    return path


def original_plan(parent):
    path = PARENT_ARTIFACTS / 'development/plan.json'
    plan = previous.launch.verify(path)
    require(plan['phase'] == 'dev' and Path(plan['data_root']).resolve() == parent
            and len(plan['cells']) == 10 and {cell['id'] for cell in plan['cells']} == {
                level + '_' + domain for level in LEVELS for domain in DOMAINS}, 'wrong original development plan')
    result = read(path.parent / 'submission_result.json'); intent = read(path.parent / 'submission_intent.json')
    require(result.get('status') == 'submitted' and result.get('array_job_id') == PARENT_ARRAY_JOB_ID
            and intent.get('plan_sha256') == file_sha(path), 'original submission provenance changed')
    return path, plan


def observe_array(plan_path):
    plan = read(plan_path); result_path = plan_path.parent / 'submission_result.json'
    if not result_path.exists():
        return {'status': 'submission_needs_reconciliation', 'path': str(result_path)}
    job = read(result_path)['array_job_id']
    tasks = {f'{job}_{index}': read(cell['tasks']) for index, cell in enumerate(plan['cells'])}
    command = ['sacct', '-n', '-X', '-P', '--array', '-j', str(job), '--format=JobID,State,End']
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as error:
        return {'status': 'scheduler_query_unavailable', 'array_job_id': job, 'error': type(error).__name__, 'detail': str(error)[:2000]}
    if result.returncode != 0:
        return {'status': 'scheduler_query_unavailable', 'array_job_id': job, 'error': 'sacct_nonzero_exit',
                'returncode': result.returncode, 'detail': result.stderr[:2000]}
    failed = []
    for line in result.stdout.splitlines():
        fields = line.split('|'); name = fields[0].strip()
        if len(fields) < 2 or name not in tasks or all(Path(task['output']).is_file() for task in tasks[name]):
            continue
        state = fields[1].strip().split(' ')[0].rstrip('+')
        stale = False
        if state == 'COMPLETED' and len(fields) > 2 and fields[2].strip() not in ('', 'Unknown'):
            ended = datetime.fromisoformat(fields[2].strip())
            stale = (datetime.now(ended.tzinfo) - ended).total_seconds() >= 120
        if state in {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'} or stale:
            failed.append({'job': name, 'state': state, **({'error': 'missing_receipts_after_visibility_grace'} if stale else {})})
    return {'status': 'execution_failed', 'jobs': failed} if failed else None


def prior_submissions(artifacts):
    """Ledger of scientific arrays; synthetic capacity checks are auxiliary."""
    directories = {PARENT_ARTIFACTS / 'development'}
    recognized = {previous.launch.SCHEMA, launch.SCHEMA}
    for root in (PARENT_ARTIFACTS, artifacts):
        candidates = {path.parent for name in ('submission_intent.json', 'submission_result.json')
                      for path in root.rglob(name)}
        for directory in candidates:
            plan_path = directory / 'plan.json'
            if not plan_path.exists():
                return None
            if read(plan_path).get('schema') in recognized:
                directories.add(directory)
    jobs = set()
    for directory in directories:
        result, intent = directory / 'submission_result.json', directory / 'submission_intent.json'
        if not result.exists() or not intent.exists():
            return None
        value = read(result)
        require(value.get('status') == 'submitted' and type(value.get('array_job_id')) is int and value['array_job_id'] > 0
                and read(intent).get('plan_sha256') == file_sha(directory / 'plan.json'), 'invalid prior scientific submission')
        jobs.add(value['array_job_id'])
    return sorted(jobs)


def immutable_record(path, value):
    if path.exists():
        require(read(path) == value, 'immutable stage record changed: ' + str(path))
    else:
        atomic_new(path, value)


def submitted_stage(parent, artifacts, name, cells, pins, *, advance, required_pins=None):
    required_pins = pins if required_pins is None else required_pins
    require(all(pins.get(source) == digest for source, digest in required_pins.items()),
            'required stage inputs missing from supplied scientific pins')
    directory = artifacts / name; path = directory / 'plan.json'
    submitted, intent = directory / 'submission_result.json', directory / 'submission_intent.json'
    if intent.exists() and not submitted.exists():
        return path, {'status': 'submission_needs_reconciliation', 'intent': str(intent)}
    deps = prior_submissions(artifacts) if not submitted.exists() else []
    if deps is None:
        return path, {'status': 'submission_needs_reconciliation', 'reason': 'unresolved earlier array submission'}
    if not path.exists():
        if not advance:
            return path, {'status': 'ready_' + name + '_submission'}
        protocol = original.authenticate(parent / 'protocol.json')
        launch.prepare(directory, cells, {label: value['path'] for label, value in protocol['models'].items()},
                       dependency_ids=deps, pins=pins)
    plan = launch.verify(path)
    keys = ('id', 'level', 'domain', 'phase', 'source_kind', 'source_root')
    require([{**{key: cell[key] for key in keys}, 'tasks': read(cell['tasks'])} for cell in plan['cells']] ==
            [{**{key: cell[key] for key in keys}, 'tasks': cell['tasks']} for cell in cells], 'wrong stage domain plan')
    require(all(plan['scientific_inputs_sha256'].get(source) == value for source, value in required_pins.items()),
            'stage omits or changes required scientific inputs')
    require(all(plan['scientific_inputs_sha256'][source] == value for source, value in pins.items()
                if source in plan['scientific_inputs_sha256']), 'stage scientific inputs changed')
    if not submitted.exists():
        if not advance:
            return path, {'status': 'ready_' + name + '_submission'}
        require(set(deps) <= set(plan['dependency_ids']), 'stage dependencies omit earlier arrays; reconcile plan')
        launch.submit(path)
    require(intent.exists() and read(intent).get('plan_sha256') == file_sha(path)
            and read(submitted).get('status') == 'submitted' and type(read(submitted).get('array_job_id')) is int,
            'stage submission provenance changed')
    return path, None


def saved_revision_recipe(root, *, advance):
    paths = revision.paths(root)
    if not paths['recipe'].exists():
        return revision.fit_domain(root) if advance else None
    value = read(paths['recipe'])
    require(value == revision.fit_domain(root, publish=False), 'saved revision recipe changed')
    return value


def carried_inputs(parent, level, domain):
    paths = domain_paths(parent, level, domain)
    source = paths['dataset'] / 'eval.jsonl'
    pins = previous.launch.confirmation_gate(parent, [{'level': level, 'domain': domain, 'rows_jsonl': str(source)}])
    pins.update(original.authenticate(parent / 'protocol.json')['files_sha256'])
    task = {'level': level, 'domain': domain, 'split': 'eval', 'interface': launch.evaluator().INTERFACE,
            'rows_jsonl': str(source), 'seeds': original.labels(level, 'eval'), 'batch_size': 8,
            'row_offset': 0, 'row_limit': 0, 'output': str(paths['receipt'])}
    launch.evaluator().validate_task(task, True)
    return {'id': level + '_' + domain + '_carry_eval', 'level': level, 'domain': domain, 'phase': 'eval',
            'source_kind': 'campaign_v1', 'source_root': str(parent), 'tasks': [task]}, pins


def frozen_identity(root, level, domain, kind):
    if kind == 'campaign_v1':
        return previous.dataset(root, level, domain)
    paths = domain_paths(root, level, domain); value = read(paths['dataset'] / 'identity.json')
    require(value.get('schema') == revision.DATASET_SCHEMA and value.get('status') == 'frozen_pending_heldout_confirmation'
            and value.get('level') == level and value.get('domain') == domain
            and value.get('protocol_sha256') == file_sha(paths['protocol'])
            and value.get('recipe_sha256') == file_sha(paths['recipe']), 'revision dataset identity changed')
    for split, (count, subset) in SPLITS.items():
        rows = revision.rows_from_jsonl(paths['dataset'] / (split + '.jsonl'))
        require(len(rows) == count == value['splits'][split]['rows'] and sha(rows) == value['splits'][split]['rows_sha256']
                and load_rows(paths['dataset'] / split, subset) == rows, 'revision frozen rows changed')
    return value


def validate_revision_pool_exclusions(root, level, domain, carried):
    """Require revision development to exclude every retained same-domain split."""
    root = Path(root).resolve(); paths = domain_paths(root, level, domain)
    certificate = read(paths['pools'] / 'identity.json')
    snapshot_path = root / 'exclusions/development.json'
    require(certificate.get('schema') == 'modebench_scale_revision_pools_v1'
            and certificate.get('level') == level and certificate.get('domain') == domain
            and certificate.get('protocol_sha256') == file_sha(paths['protocol'])
            and certificate.get('exclusions') == str(snapshot_path)
            and certificate.get('exclusions_sha256') == file_sha(snapshot_path),
            'revision pool exclusion certificate changed')
    snapshot = read(snapshot_path)
    require(snapshot.get('schema') == 'modebench_scale_revision_exclusions_v1'
            and snapshot.get('stage') == 'development'
            and snapshot.get('protocol_sha256') == file_sha(paths['protocol']),
            'wrong revision development exclusion snapshot')
    check_pins(snapshot['files_sha256'])
    blocked = {original.tuples(value) for value in snapshot['identities']}
    prompts = set(snapshot['prompt_sha256'])
    for retained_level, retained_domain, source in carried:
        if retained_domain != domain: continue
        dataset = domain_paths(source['source_root'], retained_level, domain)['dataset']
        for split, (_, subset) in SPLITS.items():
            files = pin_files(path for path in (dataset / split).rglob('*') if path.is_file())
            require(files and all(snapshot['files_sha256'].get(path) == digest for path, digest in files.items()),
                    'revision development exclusions predate carried dataset files')
            rows = load_rows(dataset / split, subset)
            require(original.identity_set(domain, rows) <= blocked
                    and {sha(row['problem']) for row in rows} <= prompts,
                    'revision development exclusions omit carried dataset rows')
    return {**snapshot['files_sha256'], **pin_files([snapshot_path, paths['pools'] / 'identity.json'])}


def audit_source(parent, level, domain, source):
    root = Path(source['source_root']); paths = domain_paths(root, level, domain)
    if source['source_kind'] == 'campaign_v1':
        return previous.confirmation(parent, level, domain, original.authenticate(parent / 'protocol.json'))
    value = revision.confirm_domain(root, publish=not paths['audit'].exists())
    if paths['audit'].exists():
        saved = read(paths['audit'])
        require({key: item for key, item in saved.items() if key != 'cross_source_disjointness'} ==
                {key: item for key, item in value.items() if key != 'cross_source_disjointness'},
                'saved revision confirmation changed')
        check_pins(saved['cross_source_disjointness']['files_sha256'])
        return saved
    return value


def publish_level(parent, release_root, level, manifest, audits, *, publish):
    pins = {**manifest['files_sha256'], **pin_files([release_root / level / 'source_manifest.json'])}
    domains, cross_pins = {}, {}
    destination = release_root / level / 'admission.json'
    saved_admission = read(destination) if destination.exists() else None
    if saved_admission is not None: check_pins(saved_admission['files_sha256'])
    for domain, source in manifest['sources'].items():
        require(audits[domain]['difficulty_matched'] is True
                and audits[domain]['original_grader_replayed_attempts'] == 128 * 32, 'all five regraded heldout domains must pass')
        root = Path(source['source_root']); paths = domain_paths(root, level, domain)
        identity = frozen_identity(root, level, domain, source['source_kind'])
        cross = verify_dataset(root, level, domain); cross_pins.update(cross['files_sha256'])
        pins.update(pin_files([paths['protocol'], paths['recipe'], paths['audit'], paths['receipt']]))
        pins.update(pin_files(path for path in paths['dataset'].rglob('*') if path.is_file()))
        domains[domain] = {**source, 'splits': {split: {'path': str(paths['dataset'] / split), **identity['splits'][split]}
                                               for split in SPLITS}}
    if saved_admission is None:
        pins.update(cross_pins)
    else:
        require(all(saved_admission['files_sha256'].get(path) == digest for path, digest in pins.items()),
                'admitted scientific source files changed')
        pins = saved_admission['files_sha256']
    value = {'schema': 'modebench_scale_composite_level_admission_v1', 'level': level, 'model_label': LEVELS[level],
             'difficulty_matched': True, 'targets': manifest['targets'], 'domains': domains, 'files_sha256': pins,
             'test_split': 'eval', 'treatment_training_started': False}
    if publish:
        base = release_root / level; immutable_record(base / 'admission.json', value)
        (base / 'dataset').mkdir(exist_ok=True)
        for domain, source in manifest['sources'].items():
            link = base / 'dataset' / domain; target = domain_paths(source['source_root'], level, domain)['dataset']
            if link.exists() or link.is_symlink():
                require(link.is_symlink() and link.resolve() == target, 'release dataset pointer changed')
            else:
                link.symlink_to(target, target_is_directory=True)
        text = (f'# ModeBench {level} ({LEVELS[level]})\n\nAll five domains passed their registered heldout gates. '
                'Each domain has 384 train, 128 development and 128 test rows. Test is named `eval`; no model training was run.\n\n'
                '| Domain | Train | Test |\n|---|---|---|\n' + ''.join(
                    f'| {domain} | [train](dataset/{domain}/train) | [test](dataset/{domain}/eval) |\n' for domain in DOMAINS))
        path = base / 'README.md'
        if path.exists():
            require(path.read_text() == text, 'release README changed')
        else:
            with path.open('x') as handle: handle.write(text)
    return value


def release_pointers_ready(release_root, level, manifest):
    base = release_root / level
    if not (base / 'README.md').is_file(): return False
    for domain, source in manifest['sources'].items():
        link = base / 'dataset' / domain
        if not link.exists() and not link.is_symlink(): return False
        require(link.is_symlink() and link.resolve() == domain_paths(source['source_root'], level, domain)['dataset'],
                'release dataset pointer changed')
    return True


@contextmanager
def controller_lock(artifacts):
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / 'controller.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sweep(parent=DEFAULT_PARENT, release_root=DEFAULT_RELEASE, artifacts=DEFAULT_ARTIFACTS, *, advance=False):
    parent, release_root, artifacts = Path(parent).resolve(), Path(release_root).resolve(), Path(artifacts).resolve()
    original.authenticate(parent / 'protocol.json'); original_plan(parent)
    if not advance:
        return _sweep(parent, release_root, artifacts, advance=False)
    with controller_lock(artifacts):
        activation = validate_activation(parent, release_root, artifacts)
        immutable_record(artifacts / 'controller_identity.json', {'schema': 'modebench_scale_composite_controller_v1',
            'parent_root': str(parent), 'release_root': str(release_root),
            'files_sha256': pin_files([activation, *activation_sources(parent, artifacts)])})
        return _sweep(parent, release_root, artifacts, advance=True)


def _sweep(parent, release_root, artifacts, *, advance):
    protocol = original.authenticate(parent / 'protocol.json')
    recipes = {(level, domain): parent_recipe(parent, level, domain, protocol, advance=advance)
               for level in LEVELS for domain in DOMAINS}
    missing = [level + '_' + domain for (level, domain), value in recipes.items() if value is None]
    if missing:
        observed = observe_array(PARENT_ARTIFACTS / 'development/plan.json')
        return observed or {'status': 'waiting_original_development' if any(not all(path.is_file() for path in
            development_receipts(parent, level, domain)) for level, domain in recipes) else 'ready_original_fit', 'missing': missing}
    manifests = {}; missing_sources = {}; proposals = revision_bindings(artifacts)
    for level in LEVELS:
        path = release_root / level / 'source_manifest.json'
        if not path.exists():
            failed = [domain for domain in DOMAINS if not recipes[level, domain]['development_fit_pass']]
            mapping = {domain: proposals.get(level, {})[domain] for domain in failed if domain in proposals.get(level, {})}
            if len(mapping) != len(failed) or not advance:
                missing_sources[level] = [domain for domain in failed if domain not in mapping] if advance else failed
                continue
            bind_level(parent, release_root, level, mapping)
        manifests[level] = source_manifest(parent, release_root, level)
    if missing_sources:
        return {'status': 'needs_source_manifest', 'failed_domains': missing_sources}
    source_pointers = {'schema': 'modebench_scale_composite_source_set_v1', 'files_sha256': pin_files(
        release_root / level / name for level in LEVELS for name in ('source_manifest.json', 'source_manifest.sha256.json'))}
    if advance:
        immutable_record(artifacts / 'source_set_identity.json', source_pointers)
    elif (artifacts / 'source_set_identity.json').exists():
        require(read(artifacts / 'source_set_identity.json') == source_pointers, 'source set changed')
    chosen = [(level, domain, source) for level, manifest in manifests.items() for domain, source in manifest['sources'].items()]
    carried = [entry for entry in chosen if entry[2]['source_kind'] == 'campaign_v1']
    revised = [entry for entry in chosen if entry[2]['source_kind'] == 'domain_revision_v1']
    carried_pins = {}
    for level, domain, source in carried:
        paths = domain_paths(parent, level, domain)
        if not paths['dataset'].exists():
            if not advance: return {'status': 'ready_to_freeze_carried'}
            original.freeze_dataset(parent, level, domain)
        frozen_identity(parent, level, domain, source['source_kind'])
        carried_pins.update(pin_files(path for path in paths['dataset'].rglob('*') if path.is_file()))
    if advance:
        immutable_record(artifacts / 'carried_freeze_identity.json', {'schema': 'modebench_scale_composite_carried_freeze_v1',
                                                                  'files_sha256': carried_pins})
    dev_cells, dev_pins = [], dict(source_pointers['files_sha256'])
    for level, domain, source in revised:
        root = Path(source['source_root']); q = revision.paths(root)
        if not q['pools'].exists():
            if not advance: return {'status': 'ready_to_materialize_revision_pools'}
            if (root / 'exclusions/development.json').exists():
                return {'status': 'generation_needs_reconciliation', 'source_root': str(root)}
            revision.materialize_pools(root)
        dev_pins.update(validate_revision_pool_exclusions(root, level, domain, carried))
        cell, pins = revision.launch_inputs(root, 'dev'); dev_cells.append(cell); dev_pins.update(pins)
    if revised:
        path, status = submitted_stage(parent, artifacts, 'revision_development', dev_cells, dev_pins, advance=advance)
        if status: return status
        if any(not all(path.is_file() for path in development_receipts(source['source_root'], level, domain))
               for level, domain, source in revised):
            return observe_array(path) or {'status': 'waiting_revision_development'}
        revision_recipes = {level + '_' + domain: saved_revision_recipe(Path(source['source_root']), advance=advance)
                            for level, domain, source in revised}
        if any(value is None for value in revision_recipes.values()): return {'status': 'ready_revision_fit'}
        failed = [key for key, value in revision_recipes.items() if not value['development_fit_pass']]
        if failed: return {'status': 'needs_new_development_revision', 'failed_domains': failed}
    for level, domain, source in revised:
        root = Path(source['source_root']); paths = domain_paths(root, level, domain)
        if not paths['dataset'].exists():
            if not advance: return {'status': 'ready_to_freeze_revisions'}
            if (root / 'exclusions/freeze.json').exists():
                return {'status': 'generation_needs_reconciliation', 'source_root': str(root)}
            revision.freeze_dataset(root)
    cells, pins, required_pins = [], {}, dict(source_pointers['files_sha256'])
    for level, domain, source in chosen:
        root = Path(source['source_root']); paths = domain_paths(root, level, domain)
        frozen_identity(root, level, domain, source['source_kind'])
        cross = verify_dataset(root, level, domain); pins.update(cross['files_sha256'])
        cell, extra = carried_inputs(parent, level, domain) if source['source_kind'] == 'campaign_v1' else revision.launch_inputs(root, 'eval')
        cells.append(cell); pins.update(extra)
        # A later fresh audit may discover additional disjoint external files.
        # Every selected scientific source remains mandatory on a resumed plan.
        required_pins.update({key: value for key, value in extra.items() if key not in cross['files_sha256']})
        required_pins.update(manifests[level]['files_sha256'])
        required_pins.update(pin_files([paths['protocol'], paths['recipe']]))
        required_pins.update(read(paths['recipe'])['input_sha256'])
        required_pins.update(pin_files(item for item in paths['dataset'].rglob('*') if item.is_file()))
    pins.update(required_pins)
    path, status = submitted_stage(parent, artifacts, 'confirmation', cells, pins, advance=advance,
                                   required_pins=required_pins)
    if status: return status
    if any(not domain_paths(source['source_root'], level, domain)['receipt'].is_file() for level, domain, source in chosen):
        return observe_array(path) or {'status': 'waiting_confirmation'}
    if not advance and any(not domain_paths(source['source_root'], level, domain)['audit'].is_file() for level, domain, source in chosen):
        return {'status': 'ready_to_audit'}
    audits = {level: {} for level in LEVELS}
    for level, domain, source in chosen:
        audits[level][domain] = audit_source(parent, level, domain, source)
    results = {}
    for level in LEVELS:
        failed = [domain for domain, value in audits[level].items() if not value['difficulty_matched']]
        if failed:
            results[level] = {'status': 'heldout_confirmation_failed', 'failed_domains': failed}
            continue
        admission = publish_level(parent, release_root, level, manifests[level], audits[level], publish=advance)
        destination = release_root / level / 'admission.json'
        if destination.exists(): require(read(destination) == admission, 'existing admission changed')
        ready = advance or (destination.exists() and release_pointers_ready(release_root, level, manifests[level]))
        results[level] = {'status': 'admitted' if ready else 'ready_to_release', 'admission': str(destination)}
    if all(value['status'] == 'admitted' for value in results.values()):
        value = {'schema': 'modebench_scale_composite_campaign_admission_v1', 'difficulty_matched': True,
                 'levels': results, 'files_sha256': pin_files(release_root / level / 'admission.json' for level in LEVELS)}
        if advance: immutable_record(release_root / 'admission.json', value)
        elif (release_root / 'admission.json').exists(): require(read(release_root / 'admission.json') == value, 'campaign admission changed')
        else: return {'status': 'ready_to_release', 'levels': results}
        return {'status': 'admitted', 'levels': results}
    return {'status': 'heldout_confirmation_failed' if any(value['status'] == 'heldout_confirmation_failed' for value in results.values())
            else 'ready_to_release', 'levels': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--release-root', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--artifacts-root', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--advance', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args(argv)
    require(1 <= args.interval <= 60, 'watch interval must be between one and sixty seconds')
    while True:
        result = sweep(args.parent, args.release_root, args.artifacts_root, advance=args.advance)
        print(json.dumps(result, sort_keys=True), flush=True)
        if not args.watch or result['status'] in TERMINAL: return result
        time.sleep(args.interval)


if __name__ == '__main__': main()
