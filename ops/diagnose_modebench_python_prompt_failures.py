#!/usr/bin/env python3
"""Additive, post-hoc descriptive audit; no generation or stored-grade changes."""
from __future__ import annotations
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_prompt_ablation_20260911'
LOCAL = BASE / 'local'
SOURCE = LOCAL / 'code_v2/src'
PLAN_SHA = 'f82f028dab3c2dfd2590db91297e0f6eac43101afde561f2ce328d0b493576f5'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
from oat_drgrpo.math_grader import _extract_modebench_candidate
from oat_drgrpo.python_modebench import parse_python_factor_candidate

WORKER = r'''
import json, signal, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from oat_drgrpo.python_modebench import execute_python_factor_candidate
from oat_drgrpo.python_modebench_worker import _timeout
signal.signal(signal.SIGALRM, _timeout)
for line in sys.stdin:
    request = json.loads(line)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.25)
        result = execute_python_factor_candidate(request['candidate'], request['spec'])
        response = {'valid': True, 'canonical_key': result.canonical_key, 'outputs': list(result.outputs)}
    except Exception as error:
        response = {'valid': False, 'error_type': type(error).__name__, 'error_message': str(error)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(response, sort_keys=True), flush=True)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def readlines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    outputs = [BASE / f'LOCAL_PYTHON_FAILURE_DIAGNOSTIC.{suffix}' for suffix in ('json', 'md')]
    draws_path = BASE / 'LOCAL_PYTHON_FAILURE_DRAW_DIAGNOSTICS.jsonl'
    assert not any(p.exists() for p in outputs + [draws_path]), 'Existing additive audit must not be overwritten'
    assert sha(LOCAL / 'plan_v2.json') == PLAN_SHA
    plan = json.loads((LOCAL / 'plan_v2.json').read_text())
    pins = {str(Path(__file__).resolve()): sha(__file__), str(LOCAL / 'plan_v2.json'): PLAN_SHA,
            str(LOCAL / 'completion_integrity_audit.json'): sha(LOCAL / 'completion_integrity_audit.json')}
    for name in ('manifest.json', 'rows.jsonl', 'prompts.jsonl'):
        path = BASE / name
        assert sha(path) == plan['input_sha256'][str(path)]
        pins[str(path)] = sha(path)
    for path, digest in plan['code_sha256'].items():
        assert sha(path) == digest
    pins.update({path: digest for path, digest in plan['code_sha256'].items()
                 if Path(path).name in ('math_grader.py', 'python_modebench.py', 'python_modebench_worker.py')})
    rows = {(r['level'], r['row_index']): r for r in readlines(BASE / 'rows.jsonl') if r['domain'] == 'python_factors'}
    prompts = {(r['level'], r['row_index'], r['arm']): r for r in readlines(BASE / 'prompts.jsonl') if r['domain'] == 'python_factors'}
    audit = json.loads((LOCAL / 'completion_integrity_audit.json').read_text())
    inventories = {r['checkpoint']: r for r in audit['checkpoints_audit']}
    samples = []
    for seed in range(43, 48):
        label = f'qwen05b_E119_python_factors_drgrpo_s{seed}'
        directory = LOCAL / 'results_v2' / label
        result_path, responses_path = directory / 'result.json', directory / 'responses.jsonl'
        result = json.loads(result_path.read_text())
        assert sha(result_path) == inventories[label]['result_sha256']
        assert sha(responses_path) == result['responses_sha256'] == inventories[label]['responses_sha256']
        assert result['identity']['plan_sha256'] == PLAN_SHA
        pins[str(result_path)], pins[str(responses_path)] = sha(result_path), sha(responses_path)
        checkpoint_samples = readlines(responses_path)
        assert len(checkpoint_samples) == 1024
        assert {(s['level'], s['row_index'], s['arm'], s['draw_index']) for s in checkpoint_samples} == {
            (level, row_index, arm, draw) for level, row_index in rows for arm in ('original', 'neutral') for draw in range(8)}
        for sample in checkpoint_samples:
            assert sample['training_seed'] == seed and sample['training_method'] == 'drgrpo'
            row = rows[sample['level'], sample['row_index']]
            prompt = prompts[sample['level'], sample['row_index'], sample['arm']]
            assert sample['row_sha256'] == prompt['row_sha256'] == objsha(row)
            assert sample['messages_sha256'] == prompt['messages_sha256'] == objsha(prompt['messages'])
        samples.extend(checkpoint_samples)
    samples.sort(key=lambda s: (s['training_seed'], s['level'], s['row_index'], s['draw_index'], s['arm']))
    diagnostics, work, work_index = [], [], {}
    for sample in samples:
        row = rows[sample['level'], sample['row_index']]
        candidate = _extract_modebench_candidate(sample['text'], row['answer'])
        d = {k: sample[k] for k in ('checkpoint_label', 'training_seed', 'level', 'row_index', 'pair_id', 'arm',
             'draw_index', 'sampling_seed', 'row_sha256', 'messages_sha256', 'verified', 'canonical_key', 'token_count', 'finish_reason')}
        d.update(text_sha256=objsha(sample['text']), candidate=candidate, parser_accepted=False,
                 literal_constant_body=False, input_independent_body=False,
                 malformed_2_for_prefix=bool(candidate and candidate.startswith('lambda n: 2 for')))
        if candidate is None:
            d['category'] = 'missing_extracted_candidate'
        else:
            try:
                tree = parse_python_factor_candidate(candidate)
            except Exception as error:
                d['error_type'], d['error_message'] = type(error).__name__, str(error)
                d['category'] = ('invalid_python_expression' if str(error) == 'candidate is not one Python expression'
                                 else 'restricted_language_rejection')
            else:
                d['parser_accepted'] = True
                body = tree.body.body
                d['lambda_body_ast_type'] = type(body).__name__
                d['literal_constant_body'] = isinstance(body, ast.Constant)
                d['input_independent_body'] = not any(isinstance(n, ast.Name) and n.id == 'n' for n in ast.walk(body))
                key = (candidate, row['answer'])
                if key not in work_index:
                    work_index[key] = len(work)
                    work.append({'candidate': candidate, 'spec': json.loads(row['answer'])})
                d['worker_result_index'] = work_index[key]
        diagnostics.append(d)
    completed = subprocess.run([sys.executable, '-I', '-B', '-c', WORKER, str(SOURCE)],
                               input=''.join(json.dumps(w) + '\n' for w in work), text=True,
                               capture_output=True, check=True, timeout=120)
    results = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(results) == len(work)
    for d in diagnostics:
        if d['parser_accepted']:
            result = results[d.pop('worker_result_index')]
            if result['valid']:
                d['category'], d['diagnostic_canonical_key'] = 'verified_success', result['canonical_key']
            else:
                d.update(result)
                message = result['error_message']
                d['category'] = ('non_integer_return' if message == 'function output must be an integer' else
                    'wrong_proper_divisor' if message.startswith('function returned ') else 'execution_exception')
        assert (d['category'] == 'verified_success') == d['verified']
        if d['verified']:
            assert d['diagnostic_canonical_key'] == d['canonical_key']

    def summarize(records):
        n = len(records)
        counts = Counter(d['category'] for d in records)
        return {'draws': n, 'verified': sum(d['verified'] for d in records),
                'category_counts': dict(sorted(counts.items())),
                'category_percentages': {k: round(100*v/n, 6) for k, v in sorted(counts.items())},
                'finish_reason_counts': dict(Counter(d['finish_reason'] for d in records)),
                'parser_accepted': sum(d['parser_accepted'] for d in records),
                'literal_constant_body_count': sum(d['literal_constant_body'] for d in records),
                'input_independent_body_count': sum(d['input_independent_body'] for d in records),
                'malformed_2_for_prefix_count': sum(d['malformed_2_for_prefix'] for d in records),
                'lambda_body_ast_type_counts_among_parser_accepted': dict(Counter(d['lambda_body_ast_type'] for d in records if d['parser_accepted'])),
                'error_message_counts': dict(Counter(d['error_message'] for d in records if 'error_message' in d)),
                'most_frequent_extracted_candidates': [{'candidate': k, 'draws': v} for k, v in Counter(d['candidate'] for d in records if d['candidate']).most_common(10)]}

    by_arm = {arm: summarize([d for d in diagnostics if d['arm'] == arm]) for arm in ('original', 'neutral')}
    by_cell = [{'training_seed': seed, 'level': level, 'arm': arm,
                **summarize([d for d in diagnostics if (d['training_seed'], d['level'], d['arm']) == (seed, level, arm)])}
               for seed in range(43, 48) for level in (2, 3) for arm in ('original', 'neutral')]
    examples = []
    for seed in range(43, 48):
        for level in (2, 3):
            first_row = min(idx for lev, idx in rows if lev == level)
            row = rows[level, first_row]
            arms = {}
            for arm in ('original', 'neutral'):
                i = next(i for i, s in enumerate(samples) if (s['training_seed'], s['level'], s['row_index'], s['draw_index'], s['arm']) == (seed, level, first_row, 0, arm))
                arms[arm] = {'text': samples[i]['text'], 'diagnostic': diagnostics[i]}
            examples.append({'training_seed': seed, 'level': level, 'row_index': first_row,
                             'draw_index': 0, 'problem': row['problem'], 'answer_spec': json.loads(row['answer']), 'arms': arms})
    for path, digest in pins.items():
        assert sha(path) == digest, f'Source changed during audit: {path}'
    draws_path.write_text(''.join(json.dumps(d, sort_keys=True) + '\n' for d in diagnostics))
    report = {'schema': 'modebench-python-posthoc-failure-diagnostic-v1', 'status': 'complete',
              'created_at': datetime.now(timezone.utc).isoformat(), 'post_hoc': True,
              'scope': 'All five registered Python Factors DrGRPO seed checkpoints (43-47), both levels and both arms: 5120 existing draws; 2560 per arm.',
              'selection_and_taxonomy': 'Requested after aggregate prompt-ablation results were known. Descriptive taxonomy developed after inspecting saved outputs; not preregistered or a confirmatory hypothesis test.',
              'method': 'Use unchanged frozen math_grader extraction and Python restricted parser. For parser-accepted candidates, unchanged frozen execute_python_factor_candidate runs in a separate Python -I worker with the original SIGALRM 0.25-second handler. Exception messages provide first-failure descriptions. Stored outputs and grades are never edited.',
              'category_definition': {'missing_extracted_candidate': 'Frozen extraction returned None (all observed such outputs ended at the 192-token limit).',
                  'invalid_python_expression': 'Frozen restricted parser raises candidate is not one Python expression.',
                  'restricted_language_rejection': 'Other frozen parser rejections, including language restrictions and character/AST limits; parser precedence is preserved.',
                  'non_integer_return': 'Frozen executor rejects a Boolean/non-integer output.',
                  'wrong_proper_divisor': 'Frozen executor rejects an integer output for one case; only its first failing case is recorded.',
                  'execution_exception': 'Other exception in the unchanged frozen executor.', 'verified_success': 'Frozen executor succeeds and its canonical key equals the stored key.'},
              'fixed_answer_definition': 'literal_constant_body_count requires a parser-accepted lambda with an AST Constant body. input_independent_body_count is the broader static absence of n in an accepted body. Malformed lambda n: 2 for ... strings are not constant functions.',
              'generation_calls': 0, 'stored_grade_disagreements': 0, 'canonical_key_disagreements': 0,
              'worker_unique_candidate_spec_pairs': len(work), 'source_sha256': pins,
              'draw_diagnostics_path': str(draws_path.relative_to(ROOT)), 'draw_diagnostics_sha256': sha(draws_path),
              'by_arm': by_arm, 'by_seed_level_arm': by_cell,
              'example_selection_rule': 'For each seed in ascending 43-47 and each level in ascending 2,3, choose the smallest selected row_index and draw_index 0; show both paired arms. No success/failure or text filter.',
              'deterministic_examples': examples,
              'limitations': 'Counts describe these frozen sampled outputs and are not independent trials across shared prompts/seeds. This diagnostic does not identify an internal training mechanism, estimate a new causal effect, or establish failures for other prompts/checkpoints. Extraction/parser errors take precedence, so syntax inside overlong candidates is not separately classified.'}
    outputs[0].write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    lines = ['# Post-hoc Python Factors failure diagnostic', '', report['scope'], '',
             'This descriptive audit was requested after aggregate results were known. It makes no new generations and changes neither frozen outputs nor verifier rules. Counts use the frozen extractor/parser and unchanged executor in an isolated child process. All 5,120 stored grades and successful canonical keys agree.', '',
             '| Category | Original (n=2,560) | Neutral (n=2,560) |', '|---|---:|---:|']
    for category in sorted(set(by_arm['original']['category_counts']) | set(by_arm['neutral']['category_counts'])):
        lines.append(f"| {category} | {by_arm['original']['category_counts'].get(category, 0)} | {by_arm['neutral']['category_counts'].get(category, 0)} |")
    n = by_arm['neutral']
    lines += ['', f"Neutral responses: {n['malformed_2_for_prefix_count']} extracted candidates start with the malformed `lambda n: 2 for` prefix. Only {n['literal_constant_body_count']} draws contain a parser-accepted literal constant body; malformed `for` responses are not counted as constant functions. {n['finish_reason_counts'].get('length', 0)} neutral draws hit the 192-token limit.", '',
              'Examples follow a fixed rule: smallest selected row index, draw 0, for every seed and level. Full row specifications, paired outputs, and hashes are in the JSON report. The following is the first such pair.', '']
    first = examples[0]
    lines += [f"Seed {first['training_seed']}, level {first['level']}, row {first['row_index']}, draw 0.", '',
              first['problem'], '', 'Original:', '```text', first['arms']['original']['text'], '```', '', 'Neutral:', '```text', first['arms']['neutral']['text'], '```', '',
              'The predominance of syntax failures describes the observed failure surface. It does not establish an internal training mechanism. Categories and this audit are explicitly post hoc.', '',
              f"Machine-readable report: `{outputs[0].name}` (SHA256 `{sha(outputs[0])}`).", f"Per-draw diagnostics: `{draws_path.name}` (SHA256 `{sha(draws_path)}`).", '']
    outputs[1].write_text('\n'.join(lines))
    print(json.dumps({'outputs': {str(p.relative_to(ROOT)): sha(p) for p in outputs + [draws_path]}, 'by_arm': by_arm}, indent=2))


if __name__ == '__main__':
    main()
