#!/usr/bin/env python3
"""Authenticate completed adaptive v3 development and fresh candidate confirmation.

Producers replay every new sampled answer with the unchanged original grader.
Historical Level1 confirmation remains a fixed empirical benchmark. Historical
Level3 outcomes never enter fresh mixture fitting. Publication is exclusive;
this module performs no model generation, scheduler mutation, or recipe writes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    sys.path.insert(0, str(ROOT / directory))
import modebench_level3_v3_common as common
from evaluate_modebench_level3_independent import (load_rows, prompt_messages, sha,
    summarize, validate_seed_receipt, validate_task)
from audit_modebench_level3_python_v6_independent_match import validate_receipt
from audit_modebench_level3_python_v6_development import verify_completed_cache
from audit_modebench_level3_recovery_execution import (scheduler_records, completion_status,
    verify_engine_log, grade_completed_receipt)

DEV_SCHEMA = 'modebench_level3_v3_completed_development_audit_v1'
CONF_SCHEMA = 'modebench_level3_v3_fixed_reference_confirmation_v1'
CANONICAL = common.CAMPAIGN / 'development/completed_development_audit.json'
CONFIRMATION_REPORT = common.CAMPAIGN / 'confirmation/confirmation_report.json'
TEST_SOURCE = ROOT / 'tests/test_modebench_level3_v3_audit.py'
METRICS = ('pass1', 'pass8')
STAGES = ('development', 'confirmation')

require = common.require
read = common.read
digest = common.digest


def now():
    return datetime.now(timezone.utc).isoformat()


def add_pin(pins, path, expected=None):
    name = str(Path(path).resolve())
    actual = digest(name)
    require(expected is None or actual == expected, 'bound evidence changed: ' + name)
    require(name not in pins or pins[name] == actual, 'conflicting bound evidence: ' + name)
    pins[name] = actual


def parse_jobid(stdout):
    match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?', stdout.strip())
    require(match is not None, 'malformed scheduler submission identity')
    return match.group(1)


def single_rtx6000_tres(value):
    """An optional generic aggregate describes the same one typed device."""
    found = []
    for token in value.split(','):
        if not re.match(r'^(?:gres/)?gpu(?:[:=]|$)', token):
            continue
        match = re.fullmatch(r'(?:gres/)?gpu(?::([^:=]+))?(?:=|:)([0-9]+)', token)
        if match is None or match.group(2) != '1':
            return False
        found.append(match.group(1))
    return found.count('rtx_6000') == 1 and found.count(None) <= 1 and set(found) <= {'rtx_6000', None}


def allocated_resources(value):
    fields = {}
    for token in value.split(','):
        pair = token.split('=', 1)
        require(len(pair) == 2 and pair[0] and pair[1] and pair[0] not in fields,
                'duplicate or malformed allocated TRES token')
        fields[pair[0]] = pair[1]
    require(single_rtx6000_tres(value), 'actual allocated GPU types/count differ')
    return fields


class OriginalGraderTimeout(BaseException):
    pass


def replay_original(receipt, rows, seconds=900):
    """A timed-out replay aborts the proof, including BaseException exits."""
    require(type(seconds) in (int, float) and 0 < seconds <= 900, 'bounded original-grader replay required')
    require(signal.getitimer(signal.ITIMER_REAL)[0] == 0, 'original-grader replay requires exclusive timer ownership')
    previous = signal.getsignal(signal.SIGALRM)
    def deadline(signum, frame):
        raise OriginalGraderTimeout('original-grader replay timed out; no completion proof may be published')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return grade_completed_receipt(receipt, rows)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def verify_account(account, jobid):
    require(account['JobIDRaw'] == jobid and account['State'] == 'COMPLETED'
            and account['ExitCode'] == '0:0', 'actual job did not complete successfully')
    require(account['Partition'] == 'all' and account['AllocCPUS'] == '6'
            and account['ReqMem'] in ('48G', '48Gn') and account['Timelimit'] == '01:00:00'
            and account['NodeList'] not in ('', 'Unknown', 'None assigned')
            and account['Start'] not in ('', 'Unknown') and account['End'] not in ('', 'Unknown')
            and single_rtx6000_tres(account['ReqTRES']) and single_rtx6000_tres(account['AllocTRES']),
            'completed job allocation differs from fixed one-card resources')


def verify_runtime(job, jobid, seal_sha256, claim, runtime, account, log_text, launcher, model_path):
    verify_account(account, jobid)
    require(claim['identity'] == {'job_id': jobid, 'cell': job['name'], 'seal_sha256': seal_sha256,
            'task_sha256': digest(job['tasks']), 'output': job['output']}, 'worker claim ownership differs')
    identity, scheduler = runtime['identity'], runtime['scheduler']
    require(claim['runtime'] == runtime and claim['gpu_names'] == identity.get('gpu_names'),
            'worker claim differs from separate saved runtime or GPU evidence')
    gpu_names = identity.get('gpu_names')
    require(gpu_names in (['Quadro RTX 6000'], ['Quadro RTX6000']), 'actual single RTX6000 device differs')
    require(identity == {'job_id': jobid, 'cell': job['name'], 'seal_sha256': seal_sha256,
            'gpu_names': gpu_names, 'partition': 'all', 'preemption_mode': 'OFF', 'cpus': 6,
            'effective_qos': scheduler['effective_qos']} and scheduler['effective_qos'] in ('normal', 'none'),
            'worker actual GPU/queue/runtime identity differs')
    require(scheduler['job_id'] == jobid and scheduler['partition'] == 'all'
            and scheduler['preemption_mode'] == 'OFF' and scheduler['cpus'] == 6,
            'worker saved scheduler metadata differs')
    fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/]*)=(\S+)', scheduler['scheduler_job']))
    allocated = allocated_resources(fields.get('AllocTRES', ''))
    require(fields.get('JobId') == jobid and fields.get('JobState') == 'RUNNING'
            and fields.get('JobName') == launcher.JOB_PREFIX + job['name']
            and fields.get('Command') == str(launcher.WORKER)
            and fields.get('UserId', '').endswith(f'({os.getuid()})')
            and fields.get('Partition') == 'all' and fields.get('NumCPUs') == '6'
            and fields.get('MinMemoryNode') == '48G' and allocated.get('cpu') == '6'
            and fields.get('QOS') == scheduler['effective_qos'] and fields.get('TimeLimit') == '01:00:00'
            and fields.get('NodeList') == account['NodeList']
            and single_rtx6000_tres(fields.get('AllocTRES', ''))
            and allocated.get('mem') in ('48G', '49152M'), 'worker saved scheduler allocation differs')
    require(re.search(r'(?:^|\s)PartitionName=all(?:\s|$)', scheduler['scheduler_partition']) is not None
            and re.search(r'(?:^|\s)PreemptMode=OFF(?:\s|$)', scheduler['scheduler_partition']) is not None,
            'worker partition preemption mode differs')
    events = []
    for line in log_text.splitlines():
        if line.startswith('{'):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('event') == 'worker_authenticated':
                events.append(event)
    require(events and all(event.get('job_id') == jobid and event.get('cell') == job['name']
            and event.get('seal_sha256') == seal_sha256 and event.get('gpu_names') == gpu_names
            for event in events), 'actual worker authentication log differs')
    verify_engine_log(log_text, model_path)


def stage_launcher(stage, seal_sha256, registration_sha256):
    require(stage in STAGES, 'unknown audit stage')
    path = ROOT / f'ops/exp_scaling/modebench_level3_v3_{stage}.py'
    seal_path = common.CAMPAIGN / stage / 'seal.json'
    require(re.fullmatch('[0-9a-f]{64}', seal_sha256 or '') is not None
            and digest(seal_path) == seal_sha256, 'explicit canonical execution seal pin required')
    saved = read(seal_path)
    require(saved['files_sha256'].get(str(path)) == digest(path), 'stage launcher is not sealed')
    launcher = importlib.import_module('modebench_level3_v3_' + stage)
    seal = launcher.authenticate_saved_seal(seal_sha256)
    registration = common.validate_registration(common.REGISTRATION, registration_sha256)
    require(seal['registration_path'] == str(common.REGISTRATION)
            and seal['registration_sha256'] == registration_sha256
            and all(seal['files_sha256'].get(path) == expected
                    for path, expected in registration['files_sha256'].items()),
            'execution seal omits prospective registration or inherited bytes')
    return launcher, seal, registration


def verify_plan(stage, plan):
    require(stage in STAGES, 'unknown plan stage')
    expected = []
    for domain, revision in zip(common.REVISED, ('graph_v8', 'python_v7')):
        for tier in range(4) if stage == 'development' else (None,):
            name = f'{revision}_d{tier}' if tier is not None else '3b_' + domain
            output = common.RESULTS / (f'calibration_3b_{revision}_d{tier}.json' if tier is not None
                                      else f'confirmation/confirmation_3b_{domain}.json')
            rows = common.contract()['calibration_rows_per_tier'][domain] if tier is not None else 128
            expected.append((name, domain, tier, str(output), rows, 4 * ((rows + 7) // 8)))
    jobs = plan['jobs']
    require(len(jobs) == len(expected), 'exact new candidate job count required')
    for job, (name, domain, tier, output, rows, batches) in zip(jobs, expected):
        require(job['name'] == name and job['domain'] == domain and job.get('tier') == tier
                and job['model_label'] == '3b' and job['output'] == output
                and job['rows'] == rows and job['batch_count'] == batches,
                'canonical candidate job/cell/row/batch mapping differs')
    return jobs


def submission_chain(launcher, jobs, seal_sha256, registration_sha256, pins):
    claim = read(launcher.CLAIM)
    require(claim['registration_path'] == str(common.REGISTRATION)
            and claim['registration_sha256'] == registration_sha256
            and claim['seal'] == str(launcher.SEAL) and claim['seal_sha256'] == seal_sha256
            and claim['plan'] == str(launcher.PLAN) and claim['plan_sha256'] == digest(launcher.PLAN)
            and claim['jobs'] == len(jobs), 'actual stage execution claim differs')
    add_pin(pins, launcher.CLAIM)
    identities = []
    for index, job in enumerate(jobs):
        intent_path = launcher.HERE / f'submission_{index:02d}_intent.json'
        result_path = launcher.HERE / f'submission_{index:02d}_result.json'
        intent, result = read(intent_path), read(result_path)
        command = launcher.command_for(index, job, seal_sha256)
        require(intent['command'] == result['command'] == command and intent['cell'] == result['cell'] == job['name']
                and intent['seal_sha256'] == result['seal_sha256'] == seal_sha256
                and intent['registration_sha256'] == result['registration_sha256'] == registration_sha256
                and intent['task_sha256'] == result['task_sha256'] == digest(job['tasks'])
                and result['intent_sha256'] == digest(intent_path)
                and result['returncode'] == 0 and isinstance(result['stderr'], str),
                'actual immutable submission command/task/result differs')
        jobid = parse_jobid(result['stdout'])
        require(result.get('job_id', jobid) == jobid, 'recorded scheduler identity differs')
        identities.append(jobid)
        add_pin(pins, intent_path); add_pin(pins, result_path)
    require(len(identities) == len(set(identities)) == len(jobs), 'unique actual scheduler jobs required')
    return identities


def authenticate_receipt(stage, task, job, seal, source_record, tokenizer):
    validate_task(task, True)
    require(stage in STAGES and task['split'] == ('dev' if stage == 'development' else 'eval'),
            'candidate receipt crosses the DEV/CONF stage boundary')
    labels = common.DEV_LABELS if stage == 'development' else common.CONF_LABELS
    require(task['domain'] == job['domain'] and task['output'] == job['output']
            and task['level'] == 'level3' and task['seeds'] == list(labels)
            and task['row_offset'] == task['row_limit'] == 0 and task['batch_size'] == 8,
            'full fixed new candidate rows/draws/task required')
    rows, source = load_rows(task)
    receipt = read(job['output'])
    require(len(rows) == job['rows'], 'candidate source row count differs')
    validate_receipt(receipt, domain=job['domain'], role='candidate', development=stage == 'development')
    reference = read(common.benchmark_metadata()[job['domain']]['receipt_path'])
    require(receipt['identity']['model'] == seal['models']['3b'] and receipt['identity']['source'] == source
            and receipt['identity']['batch_size'] == 8
            and receipt['identity']['interface'] == reference['identity']['interface']
            and receipt['identity']['code_sha256'] == reference['identity']['code_sha256'],
            'candidate source/model/batch/interface/original evaluator-grader identity differs')
    schedule = validate_seed_receipt(receipt, rows=rows)
    require(source == source_record['identity']
            and schedule['seed_schedule_sha256'] == source_record['seed_schedule_sha256']
            and source_record['distinct_request_blocks'] == len(rows) * 4
            and source_record['distinct_child_seeds'] == len(rows) * 32,
            'candidate source or independent RNG schedule differs from seal')
    prompts = [tokenizer.apply_chat_template(prompt_messages(job['domain'], row['problem'],
                receipt['identity']['interface']['prompt_profile']), tokenize=False, add_generation_prompt=True) for row in rows]
    require(receipt['identity']['rendered_prompts_sha256'] == sha(prompts), 'native rendered prompts differ')
    require(receipt['answer_mode_histogram'] == dict(Counter(str(row['answer_mode_count']) for row in rows))
            and sha(receipt['metrics']) == sha(summarize(receipt['prompt_results'])),
            'receipt support histogram or full metrics summary differs')
    return receipt, rows, source, schedule


def audit_cells(stage, launcher, seal, plan, seal_sha256, registration_sha256,
                pins, trees, *, regrade, saved_accounts=None):
    """Replay complete actual jobs; validators reuse immutable accounting snapshots."""
    jobs = verify_plan(stage, plan)
    jobids = submission_chain(launcher, jobs, seal_sha256, registration_sha256, pins)
    accounts = scheduler_records(jobids) if saved_accounts is None else saved_accounts
    require(set(accounts) == set(jobids), 'exact accounting coverage required')
    status, failed, waiting = completion_status(accounts)
    require(status == 'complete', f'candidate jobs incomplete: failed={failed}, waiting={waiting}')
    sources = {source['name']: source for source in seal['new_sources']}
    require(len(sources) == len(seal['new_sources']) == len(jobs), 'one sealed source per new candidate job required')
    prior = common.old_rng_inventory()
    prior_blocks = set(prior['blocks'])
    if stage == 'confirmation':
        development = read(CANONICAL)
        prior_blocks.update(development['new_request_blocks'])
        require(len(prior_blocks) == 28736, 'all historical and new DEV draws must be excluded')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(seal['models']['3b']['path'], local_files_only=True)
    cells, new_blocks = [], set()
    for index, (job, jobid) in enumerate(zip(jobs, jobids)):
        task_list = read(job['tasks'])
        require(isinstance(task_list, list) and len(task_list) == 1, 'exactly one candidate task per worker required')
        task = task_list[0]
        receipt, rows, source, schedule = authenticate_receipt(stage, task, job, seal, sources[job['name']], tokenizer)
        add_pin(pins, job['output']); add_pin(pins, job['tasks'])
        blocks = {base for row in receipt['identity']['seed_schedule']['request_seeds'] for base in row}
        require(len(blocks) == len(rows) * 4 and not blocks & (prior_blocks | new_blocks)
                and all(type(base) is int and base % 8 == 0 for base in blocks),
                'new candidate sampling blocks overlap or are not n8-aligned')
        new_blocks |= blocks
        cache = verify_completed_cache(task, receipt, rows, pins)
        require(cache['batches'] == job['batch_count'] and cache['attempts'] == len(rows) * 32,
                'complete batch/attempt coverage including final short batch required')
        trees[cache['directory']] = cache['directory_files']
        worker_path = launcher.HERE / f'worker_{index:02d}_execution_claim.json'
        runtime_path = launcher.HERE / f'worker_{index:02d}_runtime.json'
        log = ROOT / f'var/logs/modebench_level3/{jobid}.out'; error = log.with_suffix('.err')
        runtime = read(runtime_path)
        verify_runtime(job, jobid, seal_sha256, read(worker_path), runtime, accounts[jobid],
                       log.read_text() + '\n' + error.read_text(), launcher, seal['models']['3b']['path'])
        for path in (worker_path, runtime_path, log, error):
            add_pin(pins, path)
        count = replay_original(receipt, rows) if regrade else 0
        require(not regrade or count == len(rows) * 32, 'full original-grader replay required')
        cells.append({'name': job['name'], 'domain': job['domain'], 'tier': job.get('tier'),
            'receipt_path': job['output'], 'receipt_sha256': digest(job['output']),
            'identity_sha256': receipt['identity_sha256'], 'source': source,
            'seed_schedule_sha256': schedule['seed_schedule_sha256'], 'cache': cache,
            'metrics': receipt['metrics'], 'attempts_regraded': count,
            'job_id': jobid, 'accounting': accounts[jobid],
            'worker_claim_path': str(worker_path), 'runtime_path': str(runtime_path),
            'runtime_identity': runtime['identity']})
        if regrade:
            print(json.dumps({'status': 'receipt_regraded', 'stage': stage, 'cell': job['name'], 'attempts': count}), flush=True)
    expected_blocks, expected_attempts, expected_batches = ((4928, 39424, 624) if stage == 'development' else (1024, 8192, 128))
    require(len(new_blocks) == expected_blocks and sum(cell['cache']['attempts'] for cell in cells) == expected_attempts
            and sum(cell['cache']['batches'] for cell in cells) == expected_batches,
            'full new stage request/attempt/batch coverage differs')
    return cells, new_blocks


def validate_recipe_gates(recipe):
    import fit_modebench_level3_fixed_reference as fitter
    require(recipe['schema'] == fitter.SCHEMA and type(recipe['development_fit_pass']) is bool,
            'explicit fixed-reference recipe schema/decision required')
    development = recipe['development']
    gates = development['gates']
    require(set(gates) == {'expected_dev', 'expected_eval', 'selected'}
            and all(set(values) == set(METRICS) and all(type(value) is bool for value in values.values())
                    for values in gates.values()) and recipe['development_fit_pass'] is all(
                        value for values in gates.values() for value in values.values()),
            'all six explicit consistent development gates required')
    require(development['selected_development_sets_scored'] == 1
            and development['selected_evaluation_sets_scored'] == 0
            and development['weight_combinations_considered'] == 1771
            and development['tolerances'] == common.TOLERANCES,
            'frozen one-selection/grid/tolerance policy required')
    for stage, field in (('expected_dev', 'expected_dev_differences'),
                         ('expected_eval', 'expected_eval_differences'), ('selected', 'differences')):
        require(gates[stage] == {metric: abs(development[field][metric]) <= common.TOLERANCES[metric]
                                for metric in METRICS}, 'recipe gate does not follow exact unchanged tolerance')
    return gates


def verify_recipes(registration_sha256, pins):
    import fit_modebench_level3_fixed_reference as fitter
    result = {}
    for domain in common.REVISED:
        path = common.CAMPAIGN / 'recipes' / (domain + '.json')
        intent_path = path.with_suffix('.fit_intent.json')
        intent, recipe = read(intent_path), read(path)
        require(intent == {'schema': 'modebench_level3_v3_fixed_reference_fit_intent_v1', 'domain': domain,
                'registration_sha256': registration_sha256, 'source_sha256': digest(fitter.__file__),
                'output': str(path)}, 'immutable sole-owner fit intent differs')
        repeated = fitter.fit_recipe(common.REGISTRATION, registration_sha256, domain)
        require(recipe == json.loads(json.dumps(repeated, allow_nan=False)), 'recipe does not reproduce exactly in memory')
        gates = validate_recipe_gates(recipe)
        add_pin(pins, path); add_pin(pins, intent_path)
        result[domain] = {'path': str(path), 'sha256': digest(path), 'fit_intent_path': str(intent_path),
                         'fit_intent_sha256': digest(intent_path), 'development_fit_pass': recipe['development_fit_pass'],
                         'development_gates': gates, 'weights': recipe['weights'],
                         'fixed_reference': recipe['provenance']['reference_target']}
    return result


def audit_completed_development(*, seal_sha256, registration_sha256, publish=False, regrade=True, saved_accounts=None):
    require(not publish or (regrade and saved_accounts is None), 'publication requires original grading and live accounting')
    if publish:
        require(not CANONICAL.exists(), 'immutable completed DEV proof already exists')
    launcher, seal, registration = stage_launcher('development', seal_sha256, registration_sha256)
    pins = dict(seal['files_sha256']); trees = dict(seal['directory_files'])
    for path in (launcher.SEAL, common.REGISTRATION, Path(__file__), TEST_SOURCE):
        add_pin(pins, path)
    cells, blocks = audit_cells('development', launcher, seal, read(launcher.PLAN), seal_sha256,
                               registration_sha256, pins, trees, regrade=regrade, saved_accounts=saved_accounts)
    recipes = verify_recipes(registration_sha256, pins)
    passed = all(recipe['development_fit_pass'] for recipe in recipes.values())
    common.verify_pins(pins, trees)
    payload = {'schema': DEV_SCHEMA, 'status': 'passed_development' if passed else 'needs_calibration_revision',
        'registration_path': str(common.REGISTRATION), 'registration_sha256': registration_sha256,
        'scientific_seal_path': str(launcher.SEAL), 'scientific_seal_sha256': seal_sha256,
        'source_path': str(Path(__file__).resolve()), 'source_sha256': digest(__file__),
        'tests_path': str(TEST_SOURCE), 'tests_sha256': digest(TEST_SOURCE),
        'recipes': recipes, 'all_revised_development_gates_pass': passed,
        'canonical_json_exact_recipe_reproduction': True, 'jobs': 8, 'rows': 1232,
        'attempts': 39424, 'batches': 624, 'new_request_blocks': sorted(blocks),
        'scientific_sources': 55, 'scientific_request_blocks': 28736,
        'receipt_records': cells, 'all_attempts_regraded_with_original_grader': regrade,
        'every_attempt_including_failures_and_full_canonical_keys_compared': regrade,
        'historical_level1_confirmation_used_as_fixed_reference': True,
        'historical_level3_confirmation_used_for_mixture_fitting': False,
        'fresh_candidate_confirmation_outcomes_used': False,
        'new_recipe_choices_published': False, 'scheduler_mutations_performed': False,
        'files_sha256': pins, 'directory_files': trees}
    if publish:
        common.atomic_new(CANONICAL, {**payload, 'created_at': now()})
    return payload


def validate_completed_development_audit(path=CANONICAL):
    path = Path(path).resolve()
    require(path == CANONICAL.resolve(), 'canonical completed v3 DEV audit required')
    before = digest(path); saved = read(path)
    require(saved['schema'] == DEV_SCHEMA and saved['status'] == 'passed_development'
            and saved['all_revised_development_gates_pass'] is True
            and saved['canonical_json_exact_recipe_reproduction'] is True
            and saved['jobs'] == 8 and saved['rows'] == 1232 and saved['attempts'] == 39424
            and saved['batches'] == 624 and len(saved['receipt_records']) == 8
            and [cell['attempts_regraded'] for cell in saved['receipt_records']] == [4544] * 4 + [5312] * 4
            and saved['scientific_sources'] == 55
            and saved['scientific_request_blocks'] == 28736
            and len(saved['new_request_blocks']) == len(set(saved['new_request_blocks'])) == 4928
            and saved['all_attempts_regraded_with_original_grader'] is True
            and saved['every_attempt_including_failures_and_full_canonical_keys_compared'] is True,
            'passed complete original-grader new DEV proof required before finalization')
    require(saved['source_path'] == str(Path(__file__).resolve()) and saved['source_sha256'] == digest(__file__)
            and saved['tests_path'] == str(TEST_SOURCE) and saved['tests_sha256'] == digest(TEST_SOURCE),
            'completed development proof implementation changed')
    common.verify_pins(saved['files_sha256'], saved['directory_files'])
    accounts = {cell['job_id']: cell['accounting'] for cell in saved['receipt_records']}
    current = audit_completed_development(seal_sha256=saved['scientific_seal_sha256'],
        registration_sha256=saved['registration_sha256'], regrade=False, saved_accounts=accounts)
    expected = {key: value for key, value in saved.items() if key != 'created_at'}
    expected['all_attempts_regraded_with_original_grader'] = False
    expected['every_attempt_including_failures_and_full_canonical_keys_compared'] = False
    expected['receipt_records'] = [{**cell, 'attempts_regraded': 0} for cell in expected['receipt_records']]
    require(sha(current) == sha(expected) and digest(path) == before, 'saved DEV proof differs from bound execution semantics')
    pins = dict(saved['files_sha256']); add_pin(pins, path, before)
    return {'path': str(path), 'sha256': before, 'files_sha256': pins, 'directory_files': saved['directory_files'],
            **{key: saved[key] for key in ('status', 'jobs', 'rows', 'attempts', 'recipes',
                'all_revised_development_gates_pass', 'scientific_seal_path', 'scientific_seal_sha256',
                'registration_path', 'registration_sha256')}}


def fixed_reference_report_domains(old_report, targets, fresh_cells):
    """Aggregate retained and fresh evidence without pretending one fresh trial."""
    require(set(old_report['domains']) == set(common.DOMAINS) and set(targets) == set(common.DOMAINS)
            and {cell['domain'] for cell in fresh_cells} == set(common.REVISED) and len(fresh_cells) == 2,
            'all five fixed targets, three retained outcomes and exactly two fresh candidate cells required')
    domains = {}
    for domain in common.RETAINED:
        old = old_report['domains'][domain]
        require(old['observed_approximate_match'] is True and old['within_tolerance'] == {'pass1': True, 'pass8': True},
                'only previously passed domains may be retained')
        baseline = {metric: old['means']['baseline'][metric] for metric in METRICS}
        candidate = {metric: old['means']['candidate'][metric] for metric in METRICS}
        require(baseline == targets[domain]['metrics'], 'retained domain fixed target changed')
        differences = {metric: candidate[metric] - baseline[metric] for metric in METRICS}
        domains[domain] = {'evidence_role': 'retained_historical_level3_confirmation',
            'fresh_candidate_confirmation': False, 'candidate_confirmation_round': 1,
            'rows': 128, 'means': {'baseline': baseline, 'candidate': candidate},
            'differences': differences, 'within_tolerance': {metric: abs(differences[metric]) <= common.TOLERANCES[metric]
                                                          for metric in METRICS},
            'fixed_reference': targets[domain]['provenance'], 'candidate_receipt': old['receipts']['candidate'],
            'source_datasets': old['source_datasets'],
            'retained_original_report_domain': old, 'retained_report_path': str(common.OLD_REPORT),
            'retained_report_sha256': common.OLD_REPORT_SHA}
    for cell in fresh_cells:
        domain = cell['domain']
        baseline = dict(targets[domain]['metrics'])
        candidate = {metric: cell['metrics'][metric] for metric in METRICS}
        require(all(type(value) in (int, float) and 0 <= value <= 1
                    for value in (*baseline.values(), *candidate.values())), 'finite actual pass rates required')
        differences = {metric: candidate[metric] - baseline[metric] for metric in METRICS}
        domains[domain] = {'evidence_role': 'fresh_level3_against_fixed_measured_level1_reference',
            'fresh_candidate_confirmation': True, 'candidate_confirmation_round': 2, 'rows': 128,
            'means': {'baseline': baseline, 'candidate': candidate}, 'differences': differences,
            'within_tolerance': {metric: abs(differences[metric]) <= common.TOLERANCES[metric] for metric in METRICS},
            'fixed_reference': targets[domain]['provenance'],
            'candidate_receipt': {'path': cell['receipt_path'], 'sha256': cell['receipt_sha256']},
            'candidate_source': cell['source'], 'fresh_job_id': cell['job_id'],
            'draw_labels': list(common.CONF_LABELS)}
    for record in domains.values():
        record['observed_approximate_match'] = all(record['within_tolerance'].values())
    return {domain: domains[domain] for domain in common.DOMAINS}


def audit_completed_confirmation(*, seal_sha256, registration_sha256, publish=False, regrade=True, saved_accounts=None):
    require(not publish or (regrade and saved_accounts is None), 'confirmation publication requires live accounting and full grading')
    if publish:
        require(not CONFIRMATION_REPORT.exists(), 'immutable final report already exists')
    launcher, seal, _ = stage_launcher('confirmation', seal_sha256, registration_sha256)
    require(seal['files_sha256'].get(str(Path(__file__).resolve())) == digest(__file__)
            and seal['files_sha256'].get(str(TEST_SOURCE)) == digest(TEST_SOURCE),
            'final confirmation must prospectively seal auditor implementation and tests')
    development = validate_completed_development_audit()
    require(development['registration_sha256'] == registration_sha256
            and all(seal['files_sha256'].get(path) == expected for path, expected in development['files_sha256'].items()),
            'confirmation seal omits completed new development proof or source bytes')
    import modebench_level3_v3_finalize as finalizer
    dataset = finalizer.authenticate_dataset(registration_sha256=registration_sha256)
    require(all(seal['files_sha256'].get(path) == expected for path, expected in dataset['files_sha256'].items()),
            'confirmation seal omits final dataset/provenance bytes')
    pins = dict(seal['files_sha256']); trees = dict(seal['directory_files'])
    for path in (launcher.SEAL, common.REGISTRATION, Path(__file__), TEST_SOURCE):
        add_pin(pins, path)
    cells, blocks = audit_cells('confirmation', launcher, seal, read(launcher.PLAN), seal_sha256,
                               registration_sha256, pins, trees, regrade=regrade, saved_accounts=saved_accounts)
    inherited = common.authenticate_inherited(); targets = common.fixed_references()
    domains = fixed_reference_report_domains(inherited['report'], targets, cells)
    passed = all(record['observed_approximate_match'] for record in domains.values())
    common.verify_pins(pins, trees)
    payload = {'schema': CONF_SCHEMA, 'phase': 'confirmation',
        'status': 'matched_fixed_reference' if passed else 'outside_fixed_reference_tolerance',
        'all_five_domains_complete': True, 'confirmation_match_verified': passed,
        'domains': domains, 'errors': {}, 'missing_domains': [], 'tolerances': dict(common.TOLERANCES),
        'registration_path': str(common.REGISTRATION), 'registration_sha256': registration_sha256,
        'scientific_seal_path': str(launcher.SEAL), 'scientific_seal_sha256': seal_sha256,
        'completed_development_audit': {'path': development['path'], 'sha256': development['sha256']},
        'dataset': dataset['identity_metadata'], 'source_path': str(Path(__file__).resolve()),
        'source_sha256': digest(__file__), 'tests_path': str(TEST_SOURCE), 'tests_sha256': digest(TEST_SOURCE),
        'new_confirmation_jobs': 2, 'new_confirmation_rows': 256, 'attempts': 8192, 'batches': 128,
        'new_request_blocks': sorted(blocks), 'scientific_sources': 57, 'scientific_request_blocks': 29760,
        'receipt_records': cells, 'all_attempts_regraded_with_original_grader': regrade,
        'every_attempt_including_failures_and_full_canonical_keys_compared': regrade,
        'information_boundary': {'reference_semantics': 'fixed_measured_level1_benchmark',
            'adaptive_confirmation_round': 2, 'retained_domains': list(common.RETAINED),
            'fresh_candidate_domains': list(common.REVISED),
            'historical_level1_confirmation_used_as_fixed_reference': True,
            'historical_level3_confirmation_used_for_mixture_fitting': False,
            'new_candidate_fitting_uses_development_outcomes_only': True,
            'fresh_heldout_claim_applies_to_revised_level3_only': True,
            'all_five_fresh_same_round': False, 'statistical_equivalence_claimed': False,
            'fixed_reference_uncertainty_is_not_reestimated_as_new_level1_sampling': True,
            'treatment_training_started': False},
        'files_sha256': pins, 'directory_files': trees}
    if publish:
        common.atomic_new(CONFIRMATION_REPORT, {**payload, 'created_at': now()})
    return payload


def validate_confirmation_report(path=CONFIRMATION_REPORT):
    path = Path(path).resolve()
    require(path == CONFIRMATION_REPORT.resolve(), 'canonical adaptive v3 confirmation report required')
    before = digest(path); saved = read(path)
    require(saved['schema'] == CONF_SCHEMA and saved['phase'] == 'confirmation'
            and saved['status'] in ('matched_fixed_reference', 'outside_fixed_reference_tolerance')
            and saved['all_five_domains_complete'] is True and saved['errors'] == {} and saved['missing_domains'] == []
            and saved['new_confirmation_jobs'] == 2 and saved['new_confirmation_rows'] == 256
            and saved['attempts'] == 8192 and saved['batches'] == 128
            and saved['scientific_sources'] == 57 and saved['scientific_request_blocks'] == 29760
            and len(saved['receipt_records']) == 2 and all(cell['attempts_regraded'] == 4096 for cell in saved['receipt_records'])
            and saved['all_attempts_regraded_with_original_grader'] is True
            and saved['every_attempt_including_failures_and_full_canonical_keys_compared'] is True,
            'complete fresh candidate original-grader confirmation required')
    require(saved['source_path'] == str(Path(__file__).resolve()) and saved['source_sha256'] == digest(__file__)
            and saved['tests_path'] == str(TEST_SOURCE) and saved['tests_sha256'] == digest(TEST_SOURCE),
            'final report implementation changed')
    common.verify_pins(saved['files_sha256'], saved['directory_files'])
    accounts = {cell['job_id']: cell['accounting'] for cell in saved['receipt_records']}
    current = audit_completed_confirmation(seal_sha256=saved['scientific_seal_sha256'],
        registration_sha256=saved['registration_sha256'], regrade=False, saved_accounts=accounts)
    expected = {key: value for key, value in saved.items() if key != 'created_at'}
    expected['all_attempts_regraded_with_original_grader'] = False
    expected['every_attempt_including_failures_and_full_canonical_keys_compared'] = False
    expected['receipt_records'] = [{**cell, 'attempts_regraded': 0} for cell in expected['receipt_records']]
    require(sha(current) == sha(expected) and digest(path) == before, 'saved final report differs from actual evidence')
    pins = dict(saved['files_sha256']); add_pin(pins, path, before)
    return {'path': str(path), 'sha256': before, 'status': saved['status'],
            'confirmation_match_verified': saved['confirmation_match_verified'],
            'files_sha256': pins, 'directory_files': saved['directory_files'], 'report': saved}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--seal-sha256', required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(argv)
    require(digest(__file__) == args.source_sha256, 'explicit reviewed auditor source pin required')
    require(not args.watch or args.publish, 'watch requires explicit new proof publication')
    launcher, _, _ = stage_launcher(args.stage, args.seal_sha256, args.registration_sha256)
    if args.watch:
        jobs = verify_plan(args.stage, read(launcher.PLAN))
        required = [Path(job['output']) for job in jobs]
        required += ([common.CAMPAIGN / 'recipes' / (domain + '.json') for domain in common.REVISED]
                     if args.stage == 'development' else [CANONICAL])
        while not all(path.is_file() for path in required):
            print(json.dumps({'status': 'waiting_for_complete_' + args.stage, 'observed_at': now()}), flush=True)
            time.sleep(30)
            require(digest(__file__) == args.source_sha256, 'auditor source changed while waiting')
        while True:
            jobids = [parse_jobid(read(launcher.HERE / f'submission_{index:02d}_result.json')['stdout']) for index in range(len(jobs))]
            status, failed, waiting = completion_status(scheduler_records(jobids))
            require(not failed, 'candidate scheduler execution failed: ' + repr(failed))
            if status == 'complete':
                break
            print(json.dumps({'status': 'waiting_for_scheduler_completion', 'jobs': waiting}), flush=True)
            time.sleep(15)
    function = audit_completed_development if args.stage == 'development' else audit_completed_confirmation
    with (launcher.HERE / '.completed_audit.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(digest(__file__) == args.source_sha256, 'reviewed auditor source changed before replay')
        result = function(seal_sha256=args.seal_sha256, registration_sha256=args.registration_sha256, publish=args.publish)
    print(json.dumps({'status': result['status'], 'attempts': result['attempts'],
                      'output': str(CANONICAL if args.stage == 'development' else CONFIRMATION_REPORT) if args.publish else None}), flush=True)


if __name__ == '__main__':
    main()
