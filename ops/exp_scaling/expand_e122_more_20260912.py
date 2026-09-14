#!/usr/bin/env python3
"""Admit another finite E122 batch against current qualified capacity.

The prior release helper and all scientific/controller sources remain frozen.
This additive wrapper registers the replacement inference array and counts
capacity on every healthy node already eligible for the selected held jobs.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ops/exp_scaling'), str(ROOT / 'ops'), str(ROOT / 'src')]
BASE_SOURCE = ROOT / 'ops/exp_scaling/expand_e122_after_cli_20260912.py'
BASE_SHA = 'ceba171da3f82c973c031af5be06395bf14f1b2129c92a101a1ee297e9a1f929'
if hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError('Reviewed parent expansion source changed')
import expand_e122_after_cli_20260912 as base
import prioritize_e118_capacity_20260905 as identity

SOURCE = Path(__file__).resolve()
ART = ROOT / 'var/artifacts/e122_more_capacity_20260912'
INFERENCE = ROOT / 'var/artifacts/modebench_scale_revised_cs_recovery_20260912'
WORKER = INFERENCE / 'worker.slurm'
INFERENCE_PLAN = INFERENCE / 'plan.json'
INFERENCE_ID = '31259131'
INFERENCE_PLAN_SHA = 'c393df92378e5141852ba5654a60312f2aae4c1f03b49af28b92f44679d4b035'
INFERENCE_WORKER_SHA = 'c3c647abb85e0cd2c184c17150441397f6b8b85a82afc3e28a0e34b58d5d4204'
TEST = ROOT / 'tests/test_expand_e122_more_20260912.py'
base.ART = ART
base.PLAN = ART / 'plan.json'
base.SOURCE = SOURCE
c, m = base.c, base.m
read, sha, require, run = base.read, base.sha, base.require, base.run
ORIGINAL_SETUP = base.setup
ORIGINAL_NEW = base.new
REGISTRATION = None


def inference_registration():
    """Bind both future writers to exact immutable submission receipts."""
    require(sha(INFERENCE_PLAN) == INFERENCE_PLAN_SHA, 'Inference plan changed')
    require(sha(WORKER) == INFERENCE_WORKER_SHA, 'Inference worker changed')
    plan = read(INFERENCE_PLAN)
    intent_path = INFERENCE / 'submission_intent.json'
    result_path = INFERENCE / 'submission_result.json'
    identity_path = INFERENCE / 'submission_identity.json'
    intent, result, submitted = map(read, [intent_path, result_path, identity_path])
    command = plan['submit_command']
    require(len(plan['cells']) == 2 and '--array=0-1%1' in command,
            'Inference array bounds changed')
    require(command == intent['command'] == result['command'], 'Inference commands differ')
    require(command[-1] == str(WORKER), 'Inference worker path differs')
    require(result['returncode'] == 0 and result['stdout'].strip() == INFERENCE_ID,
            'Inference submission receipt differs')
    require(str(submitted['array_job_id']) == INFERENCE_ID and submitted['status'] == 'submitted',
            'Inference submission identity differs')
    require(intent['plan_sha256'] == result['plan_sha256'] == submitted['plan_sha256'] == sha(INFERENCE_PLAN),
            'Inference plan receipt hash differs')
    require(result['intent_sha256'] == submitted['intent_sha256'] == sha(intent_path)
            and submitted['result_sha256'] == sha(result_path), 'Inference receipt hash differs')
    raw = run(['scontrol', 'show', 'job', '-dd', '-o', INFERENCE_ID + '_0'])
    require(c.field(raw, 'ArrayJobId') == INFERENCE_ID and c.field(raw, 'ArrayTaskId') == '0'
            and c.field(raw, 'ArrayTaskThrottle') == '1', 'Inference scheduler array differs')
    require(c.field(raw, 'UserId').endswith(f'({os.getuid()})')
            and c.field(raw, 'WorkDir') == str(ROOT)
            and c.field(raw, 'Command') == str(WORKER), 'Inference scheduler owner/worker differs')
    require(identity.submit_tokens(raw) == command, 'Inference scheduler submitted command differs')
    paths = [WORKER, INFERENCE_PLAN, intent_path, result_path, identity_path]
    return {'inference_job_id': INFERENCE_ID, 'array_tasks': 2, 'reserve_gib_per_task': 32,
            'scheduler_record': raw, 'pins': {str(p): sha(p) for p in paths}}


def capacity_rows(records):
    rows = []
    for node in records:
        if node['name'] not in base.capacity.POOL:
            continue
        healthy = not set(node['state']) & {'DOWN', 'DRAIN', 'FAIL', 'MAINT', 'RESERVED', 'PLANNED'}
        allowed = healthy and 'lowprio' in node['partitions'] and any(
            kind in node['gres'] for kind in ['gpu:a6000:', 'gpu:a100:'])
        total = sum(int(n) for n in re.findall(r'gpu:[^:,(]+:(\d+)', node['gres']))
        used = sum(int(n) for n in re.findall(r'gpu:[^:,(]+:(\d+)', node['gres_used']))
        memory = max(0, node['real_memory'] - node['alloc_memory'])
        cpus = max(0, node['cpus'] - node['alloc_cpus'])
        slots = min(max(0, total - used), memory // 65536, cpus // 8) if allowed else 0
        rows.append({'node': node['name'], 'state': node['state'], 'free_64g_slots': slots,
                     'free_memory_mib': memory, 'free_gpu': max(0, total - used), 'record': node})
    require(any(row['free_64g_slots'] > 0 for row in rows), 'No schedulable qualified capacity')
    return rows


def node_capacity():
    return capacity_rows(json.loads(run(['scontrol', 'show', 'nodes', '--json']))['nodes'])


def setup(cap, candidates):
    global REGISTRATION
    REGISTRATION = inference_registration()
    m.budget_helper.ARRAYS[INFERENCE_ID] = (2, WORKER, INFERENCE_PLAN)
    campaign = ORIGINAL_SETUP(cap, candidates)
    original_budget = m.budget_helper.storage_budget
    def budget(*args, **kwargs):
        for path, expected in REGISTRATION['pins'].items():
            require(sha(Path(path)) == expected, 'Inference registration source changed: ' + path)
        return original_budget(*args, **kwargs)
    m.budget_helper.storage_budget = budget
    return campaign


def require_existing_routes(plan):
    capacity_nodes = {row['node'] for row in plan['capacity'] if row['free_64g_slots'] > 0}
    for jid, raw in plan['held_records'].items():
        nodes = set(run(['scontrol', 'show', 'hostnames', c.field(raw, 'ReqNodeList')]).split())
        require(capacity_nodes <= nodes <= base.capacity.POOL,
                'Selected job lacks the measured capacity route: ' + jid)


def write_plan(path, value):
    if path == base.PLAN:
        require_existing_routes(value)
        value.update(schema='e122_more_finite_capacity_v1',
                     authorization='User requested more E122 concurrency and recovery of reported failures on September12.',
                     inference_registration=REGISTRATION,
                     route_policy='Preserve existing job eligibility; measured free nodes are already eligible.',
                     capacity_policy='Count any healthy previously qualified pool node; preserve GPU/CPU/64GiB limits.')
        value['source_pins'].update({str(BASE_SOURCE): BASE_SHA, str(TEST): sha(TEST),
                                     **REGISTRATION['pins']})
    return ORIGINAL_NEW(path, value)


def preserve_route(jid, cell):
    raw = run(['scontrol', 'show', 'job', '-dd', '-o', jid])
    require(c.field(raw, 'JobState') in {'PENDING', 'RUNNING', 'COMPLETING', 'COMPLETED'},
            'Released job needs reconciliation: ' + jid)
    require(identity.exports(identity.submit_tokens(raw)) == cell['environment'],
            'Released job scientific environment differs')
    plan = read(base.PLAN)
    before = plan['held_records'][jid]
    require(identity.submit_tokens(raw) == identity.submit_tokens(before), 'Submitted command changed')
    for key in base.capacity.PRESERVE:
        require(c.field(raw, key) == c.field(before, key), 'Resource/science changed: ' + key)
    require(c.field(raw, 'ReqNodeList') == c.field(before, 'ReqNodeList'), 'Existing route changed')
    ORIGINAL_NEW(ART / 'routes' / jid / 'verified.json',
                 {'after': raw, 'science_and_resources_unchanged': True, 'route_unchanged': True})


base.capacity.node_capacity = node_capacity
base.setup = setup
base.new = write_plan
base.route = preserve_route

if __name__ == '__main__':
    base.main()
