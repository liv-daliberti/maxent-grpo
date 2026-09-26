#!/usr/bin/env python3
"""Audit provider-declared refusals, filtering, and empty hosted-model answers.

Classifications describe native API metadata. They do not infer why a policy
triggered, detect ordinary-text refusals semantically, or alter prompts/grades.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from audit_hosted_modebench_completion import atomic, file_sha, load_inventory, read_jsonl, sha


def filtered_categories(value, path=()):
    found = set()
    if isinstance(value, dict):
        if value.get('filtered') is True:
            found.add(path[-1] if path else 'unspecified')
        for key, child in value.items():
            found.update(filtered_categories(child, path + (str(key),)))
    elif isinstance(value, list):
        for child in value:
            found.update(filtered_categories(child, path))
    return found


def classify_native(body, protocol, choice_index=0):
    """Extract explicit provider signals without examining answer semantics."""
    native_refusal_fields = []
    if protocol == 'anthropic_messages':
        blocks = body.get('content') or []
        stop_reason = body.get('stop_reason')
        stop_details = body.get('stop_details')
        text = ''.join(part.get('text', '') for part in blocks if part.get('type') == 'text')
        reasoning_exposed = any(part.get('type') in ('thinking', 'redacted_thinking') for part in blocks)
        filter_metadata = {key: body[key] for key in ('content_filter_results', 'prompt_filter_results') if key in body}
    elif protocol == 'responses':
        output = body.get('output') or []
        blocks = [part for item in output if item.get('type') == 'message' for part in item.get('content', [])]
        stop_reason = body.get('status')
        stop_details = body.get('incomplete_details')
        text = ''.join(part.get('text', '') for part in blocks if part.get('type') == 'output_text')
        reasoning_exposed = any(item.get('type') == 'reasoning' for item in output)
        filter_metadata = {key: body[key] for key in ('content_filter_results', 'prompt_filter_results') if key in body}
    elif protocol == 'chat_completions':
        choices = [choice for choice in body.get('choices', []) if choice.get('index') == choice_index]
        if len(choices) != 1:
            raise ValueError('Missing or duplicate native choice')
        choice = choices[0]
        message = choice.get('message') or {}
        content = message.get('content')
        blocks = content if isinstance(content, list) else ([{'type': 'text', 'text': content}] if isinstance(content, str) and content else [])
        text = ''.join(part.get('text', '') for part in blocks if part.get('type') in ('text', 'output_text'))
        stop_reason = choice.get('finish_reason')
        stop_details = choice.get('stop_details', body.get('stop_details'))
        reasoning_exposed = bool(message.get('reasoning_content') or message.get('reasoning'))
        if message.get('refusal'):
            native_refusal_fields.append('choices.message.refusal')
        filter_metadata = {key: body[key] for key in ('content_filter_results', 'prompt_filter_results') if key in body}
        filter_metadata.update({'choice.' + key: choice[key] for key in ('content_filter_results',) if key in choice})
    else:
        raise ValueError('Unsupported native response protocol')
    if stop_reason == 'refusal':
        native_refusal_fields.append('stop_reason')
    if isinstance(stop_details, dict) and stop_details.get('type') == 'refusal':
        native_refusal_fields.append('stop_details.type')
    if any(part.get('type') == 'refusal' for part in blocks):
        native_refusal_fields.append('content.refusal_block')
    if body.get('refusal'):
        native_refusal_fields.append('response.refusal')
    categories = set()
    if isinstance(stop_details, dict) and isinstance(stop_details.get('category'), str):
        categories.add(stop_details['category'])
    filter_categories = sorted(filtered_categories(filter_metadata))
    content_filtered = (stop_reason == 'content_filter' or
                        (isinstance(stop_details, dict) and stop_details.get('reason') == 'content_filter') or
                        bool(filter_categories))
    block_types = [part.get('type', 'unspecified') for part in blocks]
    empty_answer = not text.strip()
    # For Responses/Chat, reasoning may be a sibling field instead of an answer block.
    thinking_only = empty_answer and reasoning_exposed and all(
        kind in ('thinking', 'redacted_thinking') for kind in block_types)
    return {'native_stop_reason': stop_reason, 'native_stop_details': stop_details,
            'refusal': bool(native_refusal_fields), 'refusal_signal_fields': native_refusal_fields,
            'content_filtered': bool(content_filtered), 'category_labels': sorted(categories),
            'filtered_category_labels': filter_categories, 'native_filter_metadata': filter_metadata,
            'answer_text_empty': empty_answer, 'content_empty': len(blocks) == 0,
            'thinking_only': thinking_only, 'content_block_types': block_types,
            'reasoning_record_present': reasoning_exposed, 'answer_text_sha256': hashlib.sha256(text.encode()).hexdigest()}


def aggregate(records):
    return {'responses': len(records), 'refusals': sum(r['refusal'] for r in records),
            'content_filtered': sum(r['content_filtered'] for r in records),
            'provider_restriction_signals': sum(r['refusal'] or r['content_filtered'] for r in records),
            'empty_answer_text': sum(r['answer_text_empty'] for r in records),
            'empty_content': sum(r['content_empty'] for r in records),
            'thinking_only': sum(r['thinking_only'] for r in records),
            'refusals_with_empty_answer_text': sum(r['refusal'] and r['answer_text_empty'] for r in records),
            'refusals_with_empty_content': sum(r['refusal'] and r['content_empty'] for r in records),
            'refusals_thinking_only': sum(r['refusal'] and r['thinking_only'] for r in records),
            'nonrefusal_empty_answer_text': sum(not r['refusal'] and r['answer_text_empty'] for r in records),
            'category_counts': dict(Counter(label for r in records for label in r['category_labels'])),
            'refusal_category_counts': dict(Counter(label for r in records if r['refusal'] for label in r['category_labels'])),
            'filtered_category_counts': dict(Counter(label for r in records for label in r['filtered_category_labels'])),
            'stop_reason_counts': dict(Counter(str(r['native_stop_reason']) for r in records))}


def audit(directory, output_dir=None, expected_samples=15360, io_workers=16):
    directory = Path(directory).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else directory
    inventory = load_inventory(directory, expected_samples)
    completion = json.loads((directory / 'completion_audit.json').read_text())
    if completion.get('status') != 'pass' or completion.get('saved_samples') != expected_samples:
        raise ValueError('Provider-outcome audit requires a passed complete native receipt audit')
    evidence = json.loads((directory / 'evidence_file_sha256.json').read_text())
    if sha(evidence) != completion['evidence_inventory_sha256']:
        raise ValueError('Completion evidence inventory has changed')
    if file_sha(directory / 'samples.jsonl') != evidence['samples.jsonl']:
        raise ValueError('Primary sample export has changed since completion audit')
    samples = read_jsonl(directory / 'samples.jsonl')
    if len(samples) != expected_samples or {s['sample_id'] for s in samples} != set(inventory['expected']):
        raise ValueError('Incomplete or duplicate source sample inventory')
    raw_names = sorted({sample['raw_receipt'] for sample in samples})
    def load_native(relative):
        data = (directory / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != evidence.get(relative):
            raise ValueError('Raw native receipt changed after completion audit: ' + relative)
        return relative, (json.loads(data), digest)
    if not 1 <= io_workers <= 32:
        raise ValueError('I/O workers must be 1..32')
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        raws = dict(pool.map(load_native, raw_names))
    protocol = ('anthropic_messages' if inventory['manifest']['schema'] == 'frontier-modebench-anthropic-messages-v1'
                else inventory['manifest'].get('protocol', 'responses'))
    outcomes = []
    for sample in samples:
        raw, raw_digest = raws[sample['raw_receipt']]
        canonical_digest = sha(raw)
        recorded_digest = sample.get('raw_receipt_sha256')
        if recorded_digest is not None and canonical_digest != recorded_digest:
            raise ValueError('Sample canonical receipt digest mismatch')
        classified = classify_native(raw['response'], protocol, sample.get('choice_index', 0))
        if classified['answer_text_sha256'] != hashlib.sha256(sample['text'].encode()).hexdigest():
            raise ValueError('Native outcome answer extraction differs from authenticated grader text')
        outcomes.append({key: sample[key] for key in ('sample_id', 'level', 'domain', 'row_index', 'sample_index',
                                                      'row_sha256', 'request_sha256', 'raw_receipt')}
                        | {'response_id': raw['response']['id'], 'raw_receipt_file_sha256': raw_digest,
                           'raw_receipt_sha256': canonical_digest,
                           'raw_receipt_digest_present_in_source_sample': recorded_digest is not None,
                           'native_protocol': protocol, 'choice_index': sample.get('choice_index', 0)} | classified)
    cells = defaultdict(list)
    for outcome in outcomes:
        cells[f"level{outcome['level']}/{outcome['domain']}"].append(outcome)
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar = output_dir / 'provider_outcome_samples.jsonl'
    temporary = sidecar.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in outcomes))
    temporary.replace(sidecar)
    sources = {name: {'path': str(directory / name), 'sha256': file_sha(directory / name)}
               for name in ('manifest.json', 'samples.jsonl', 'completion_audit.json', 'evidence_file_sha256.json')}
    result = {'schema': 'hosted-modebench-provider-outcomes-v1', 'status': 'complete',
              'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'model': inventory['manifest']['model'],
              'source_run': str(directory), 'native_protocol': protocol, 'expected_responses': expected_samples,
              'responses': len(outcomes), 'raw_receipts_read': len(raws), 'sources': sources,
              'sample_outcomes': {'path': str(sidecar), 'sha256': file_sha(sidecar), 'records': len(outcomes)},
              'totals': aggregate(outcomes), 'cells': {cell: aggregate(rows) for cell, rows in sorted(cells.items())},
              'non_python_refusals': sum(row['refusal'] and row['domain'] != 'python_factors' for row in outcomes),
              'definitions': {
                  'refusals': 'Explicit native stop_reason=refusal, stop_details.type=refusal, refusal content block, or nonempty native refusal field.',
                  'content_filtered': 'Explicit native content_filter finish/incomplete reason or filtered=true in returned provider filter metadata.',
                  'empty_answer_text': 'Authenticated answer-only text is empty or whitespace; this alone does not imply refusal.',
                  'thinking_only': 'No final-answer text, a native reasoning output record/field present, and no non-thinking content blocks; full reasoning text need not be exposed.',
                  'empty_content': 'No blocks in the native content surface: Anthropic content, Responses message content, or Chat message content; these surfaces differ across protocols.',
                  'category_counts': 'Provider-reported stop_details.category labels; no independent judgment of harmfulness.',
              },
              'interpretation': [
                  'These are deployed-interface outcomes reported by the provider; the audit does not infer why a policy triggered.',
                  'Refusals with no answer are not evidence of syntactic failure or concentration among correct solution modes.',
                  'If a cell has no correct answer pairs, its correct-mode collision statistic is undefined.',
                  'Ordinary answer text is not semantically classified as a refusal by this audit.',
                  'No API calls, prompt changes, policy changes, or grading changes were made.',
              ],
              'audit_source_path': str(Path(__file__).resolve()), 'audit_source_sha256': file_sha(Path(__file__)),
              'audit_helper_source_sha256': file_sha(Path(__file__).with_name('audit_hosted_modebench_completion.py')),
              'independent_io_workers': io_workers}
    atomic(output_dir / 'provider_outcomes.json', result)
    lines = [f"# Native provider outcomes: {result['model']}", '',
             f"Audited all **{len(outcomes):,}** saved model responses against the completed receipt inventory. "
             f"The provider declared **{result['totals']['refusals']:,} refusals** and "
             f"**{result['totals']['content_filtered']:,} content-filter signals**.", '',
             '| Cell | Responses | Refusals | Content filtered | Empty answer | Empty content | Thinking only | Provider categories |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for cell, values in result['cells'].items():
        labels = ', '.join(f'{key}: {value}' for key, value in sorted(values['category_counts'].items())) or 'none reported'
        lines.append(f"| {cell} | {values['responses']} | {values['refusals']} | {values['content_filtered']} | "
                     f"{values['empty_answer_text']} | {values['empty_content']} | {values['thinking_only']} | {labels} |")
    lines += ['', f"Non-Python refusals: **{result['non_python_refusals']}**.", '', *result['interpretation'], '',
              'Per-sample classifications, native stop details, and exact raw-receipt hashes are retained in '
              '`provider_outcome_samples.jsonl`; source manifest, completion audit, and evidence hashes are in `provider_outcomes.json`.', '']
    (output_dir / 'provider_outcomes.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--io-workers', type=int, default=16)
    args = parser.parse_args()
    result = audit(args.directory, args.output_dir, io_workers=args.io_workers)
    print(json.dumps({'model': result['model'], 'status': result['status'], 'totals': result['totals'],
                      'non_python_refusals': result['non_python_refusals']}, sort_keys=True))
