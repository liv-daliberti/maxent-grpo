#!/usr/bin/env python3
"""Freeze the authorized 32 x 15 x 8 x 7 reasoning-off comparison; no API calls."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_chat_frontier_modebench import atomic, file_sha, now, sha, verify_snapshot, read_jsonl, write_jsonl
from audit_hosted_modebench_completion import load_inventory

BASE = ROOT / 'artifacts/hosted_reasoning_off_32_20260911_v2'
CONDITION = 'hosted-reasoning-off-first32-v1'
SOURCES = {
    'gpt56sol': 'gpt-5.6-sol', 'claude_opus5': 'claude-opus-5',
    'gpt54': 'gpt-5.4', 'grok43': 'grok-4.3', 'kimi_k3': 'FW-Kimi-K3',
    'claude_opus48': 'claude-opus-4-8', 'deepseek_v4_pro': 'DeepSeek-V4-Pro',
}
RUNNERS = {
    'frontier-modebench-responses-v1': 'ops/evaluate_frontier_modebench.py',
    'frontier-modebench-anthropic-messages-v1': 'ops/evaluate_claude_modebench.py',
    'frontier-modebench-native-chat-responses-v1': 'ops/evaluate_chat_frontier_modebench.py',
}
DOCS = {
    'gpt56sol': {'url': 'https://developers.openai.com/api/docs/models/gpt-5.6-sol', 'control': 'reasoning.effort=none', 'support': 'Model reference explicitly lists none.'},
    'gpt54': {'url': 'https://developers.openai.com/api/docs/models/gpt-5.4', 'control': 'reasoning.effort=none', 'support': 'Model reference explicitly lists none.'},
    'claude_opus5': {'url': 'https://platform.claude.com/docs/en/models/opus-5/whats-new-opus-5', 'control': 'thinking.type=disabled; retain output_config.effort=medium', 'support': 'Explicit disabled supported at high effort and below, including medium. Omitting thinking would enable it.'},
    'claude_opus48': {'url': 'https://platform.claude.com/docs/en/models/opus-5/whats-new-opus-5', 'control': 'thinking.type=disabled; retain output_config.effort=medium', 'support': 'Documentation contrasts Opus 4.8, where disabling thinking is independent of effort.'},
    'grok43': {'url': 'https://docs.x.ai/developers/models/grok-4.3', 'control': 'reasoning_effort=none', 'support': 'Model reference explicitly describes none as no reasoning.'},
    'kimi_k3': {'url': 'https://docs.fireworks.ai/api-reference/post-chatcompletions', 'control': 'reasoning_effort=none', 'support': 'Kimi K3 explicitly supports none to disable thinking; medium maps to high.'},
    'deepseek_v4_pro': {'url': 'https://docs.fireworks.ai/api-reference/post-chatcompletions', 'control': 'reasoning_effort=none', 'support': 'DeepSeek V4 supports none under the standardized hosted adapter contract. This Azure adapter rejected the native DeepSeek thinking field with explicit HTTP400 before generation; the standardized off setting requires a retained canary.'},
}


def identity(row):
    return row['level'], row['domain'], row['row_index']


def off_payload(payload, slug):
    result = copy.deepcopy(payload)
    if slug in ('gpt56sol', 'gpt54'):
        result['reasoning']['effort'] = 'none'
    elif slug in ('grok43', 'kimi_k3', 'deepseek_v4_pro'):
        result['reasoning_effort'] = 'none'
    else:
        result['thinking'] = {'type': 'disabled'}
    return result


def prepare_one(slug, base):
    reference = ROOT / f'artifacts/frontier_modebench_{slug}_20260911'
    output = base / slug
    old = json.loads((reference / 'manifest.json').read_text())
    verify_snapshot(reference, old)
    if old['model'] != SOURCES[slug] or old['request_count'] != 15360:
        raise ValueError('Source model or inventory differs: ' + slug)
    if (output / 'manifest.json').exists():
        existing = json.loads((output / 'manifest.json').read_text())
        if existing.get('experiment_condition') != CONDITION or existing['reference_manifest_sha256'] != file_sha(reference / 'manifest.json'):
            raise ValueError('Existing frozen condition differs')
        verify_snapshot(output, existing)
        return existing
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite nonempty unfrozen condition: ' + str(output))
    rows = [r for r in read_jsonl(reference / 'rows.jsonl') if 0 <= r['row_index'] < 32]
    counts = Counter((r['level'], r['domain']) for r in rows)
    if len(rows) != 480 or len(counts) != 15 or set(counts.values()) != {32}:
        raise ValueError('Expected exactly first32 rows in all 15 cells')
    selected = {identity(r): r for r in rows}
    originals = [r for r in read_jsonl(reference / 'requests.jsonl') if identity(r) in selected]
    requests = []
    for item in originals:
        if item['request_sha256'] != sha(item['request']) or item['row_sha256'] != sha(selected[identity(item)]):
            raise ValueError('Original request identity differs')
        adapted = copy.deepcopy(item)
        adapted['reference_request_sha256'] = item['request_sha256']
        adapted['request'] = off_payload(item['request'], slug)
        adapted['request_sha256'] = sha(adapted['request'])
        adapted['experiment_condition'] = CONDITION
        requests.append(adapted)
    if len(requests) != 3840 or len({r['sample_id'] for r in requests}) != 3840:
        raise ValueError('Expected 3840 unique sample slots')
    for key in selected:
        if sorted(r['sample_index'] for r in requests if identity(r) == key) != list(range(8)):
            raise ValueError('Incomplete eight-draw prompt')
    primary = reference / 'audited_primary_samples.jsonl'
    primary_rows = [r for r in read_jsonl(primary) if identity(r) in selected]
    original_lookup = {r['sample_id']: r for r in originals}
    if len(primary_rows) != 3840 or len({r['sample_id'] for r in primary_rows}) != 3840:
        raise ValueError('Paired medium baseline is incomplete')
    for row in primary_rows:
        original = original_lookup[row['sample_id']]
        if row['request_sha256'] != original['request_sha256'] or row['row_sha256'] != original['row_sha256']:
            raise ValueError('Paired baseline provenance mismatch')
    output.mkdir(parents=True)
    write_jsonl(output / 'rows.jsonl', rows)
    write_jsonl(output / 'requests.jsonl', requests)
    write_jsonl(output / 'paired_medium_samples.jsonl', primary_rows)
    shutil.copyfile(reference / 'datasets.json', output / 'datasets.json')
    assets = ['rows.jsonl', 'requests.jsonl', 'datasets.json', 'paired_medium_samples.jsonl']
    manifest = copy.deepcopy(old)
    manifest.pop('operational_amendments', None)
    manifest.update(prepared_at_utc=now(), experiment_condition=CONDITION,
                    prompt_count=480, request_count=3840, sample_count=8,
                    reasoning_effort='none' if slug in ('gpt56sol', 'gpt54', 'grok43', 'kimi_k3', 'deepseek_v4_pro') else 'thinking disabled',
                    original_prompt_cohort=False, original_prompt_bytes=True,
                    reference_run=str(reference), reference_manifest_sha256=file_sha(reference / 'manifest.json'),
                    reference_artifact_sha256=old['artifact_sha256'],
                    reference_primary_samples_sha256=file_sha(primary),
                    original_reasoning_configuration={k: old.get(k) for k in ('reasoning_effort', 'requested_thinking', 'requested_output_config', 'requested_settings')},
                    native_runner=RUNNERS[old['schema']])
    if old['schema'] == 'frontier-modebench-native-chat-responses-v1':
        groups = [{'group_id': r['group_id'], 'request': r['request'], 'request_sha256': r['request_sha256'],
                   'sample_ids': [r['sample_id']], 'sample_count': 1, 'protocol': old['protocol']} for r in requests]
        if old['samples_per_http_request'] != 1:
            raise ValueError('Only independent stateless calls are supported')
        write_jsonl(output / 'http_requests.jsonl', groups)
        profile = copy.deepcopy(old['model_profile'])
        profile['request_parameters'] = {k: v for k, v in requests[0]['request'].items() if k not in ('model', 'messages', 'input')}
        atomic(output / 'model_profile.json', profile)
        manifest.update(http_request_count=3840, model_profile=profile, model_profile_sha256=sha(profile), requested_settings=profile['request_parameters'])
        assets += ['http_requests.jsonl', 'model_profile.json']
    elif slug in ('claude_opus5', 'claude_opus48'):
        manifest.update(requested_thinking={'type': 'disabled'})
    else:
        manifest.update(requested_settings={k: v for k, v in requests[0]['request'].items() if k not in ('model', 'input')})
    atomic(output / 'reasoning_condition.json', {
        'schema': CONDITION, 'selection_rule': 'row_index 0 through 31 inclusive in each original frozen domain-level cell',
        'selection_uses_outputs': False, 'samples_per_prompt': 8, 'prompts_per_cell': 32,
        'retained_order': 'Original preregistered request order', 'all_prompt_bytes_unchanged': True,
        'sampling_controls': 'Retain original omitted/default temperature and top_p; only provider reasoning switch changes.',
        'off_control': DOCS[slug], 'documentation_checked_utc_date': '2026-09-11',
        'support_limit': 'Public native documentation does not certify internal Azure aliases; first retained canary and every received response must pass control validation before admission.',
        'on_baseline': 'Existing audited medium cohort for these exact rows and sample slots; no new medium calls. Generation slots are not paired RNG seeds.',
        'opus5_python': 'Original system/user prompt in both conditions; revised Python wording is a separate experiment.',
        'max_output_tokens': 8192, 'planned_terminal_generations': 3840,
        'preflight': 'First registered sample is the retained canary and counts toward 3840. No outcome-based prompt replacement.',
        'interpretation': 'Compare the final-output distribution at each deployment control; disabled explicit deliberation does not imply absence of internal computation.',
    })
    assets.append('reasoning_condition.json')
    code_hashes = {}
    for name in old['code_sha256']:
        target = output / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / 'code' / name, target)
        code_hashes[name] = file_sha(target)
    for name in ('prepare_hosted_reasoning_off.py', 'run_hosted_reasoning_off.py', 'evaluate_hosted_reasoning_off.py'):
        target = output / 'code/ops' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'ops' / name, target)
        code_hashes['ops/' + name] = file_sha(target)
    manifest['code_sha256'] = code_hashes
    manifest['artifact_sha256'] = {name: file_sha(output / name) for name in assets}
    manifest['notes'] = [
        'Separately frozen reasoning-off condition. Original paid evidence remains unchanged.',
        'No temperature or prompt adjustment; all eight draws retained including refusal/invalid/truncated outcomes.',
        'First sample canary counts toward budget; provider control is checked on every response.',
        'No private chain-of-thought is needed or requested; only native control metadata and presence/absence of exposed reasoning are audited.',
        'One attempt per call on initial collection; uncertain outcomes are never automatically resampled.',
        'Per-level pass@8 and distinct@8 average the five domains equally, including zero-correct prompts.',
    ]
    atomic(output / 'manifest.json', manifest)
    load_inventory(output, expected_samples=3840)
    return manifest


def prepare(base):
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    manifests = {slug: prepare_one(slug, base) for slug in SOURCES}
    row_hashes = {m['artifact_sha256']['rows.jsonl'] for m in manifests.values()}
    if len(row_hashes) != 1:
        raise ValueError('Models do not share the same selected task rows')
    registry = {'schema': CONDITION, 'prepared_at_utc': now(), 'planned_terminal_generations': 26880,
        'new_on_generations': 0, 'canaries_included': True, 'max_output_tokens_per_generation': 8192,
        'maximum_native_output_tokens_before_retries': 26880 * 8192,
        'source_selection': 'Original row indices 0..31 in every domain/level, independent of outputs',
        'runs': [{'slug': slug, 'model': SOURCES[slug], 'run_directory': str(base / slug),
                  'manifest_sha256': file_sha(base / slug / 'manifest.json')} for slug in SOURCES],
        'authorization': 'User authorized matched 32 prompts per domain/level across seven models, 26,880 new reasoning-off generations; on responses reused.'}
    path = base / 'experiment.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['runs'] != registry['runs']:
            raise ValueError('Existing root experiment registry differs')
        return existing
    atomic(path, registry)
    return registry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base), indent=2))
