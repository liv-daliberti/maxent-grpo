#!/usr/bin/env python3
"""One frozen checkpoint's five-domain recovery cell, with resumable receipts.

The worker may generate and grade; it may not choose what is tested. Prompts,
held-back calibration problems, withdrawn options, the sentence that states a
revised task and the rule that grades a recovery answer all come from the frozen
inputs. Every request is written once to its own receipt keyed by a binding
hash, so an interrupted cell resumes without re-sampling anything.

A recovery answer counts when the original executable verifier accepts it and it
survives the withdrawal under the frozen predicate. Withdrawals are exclusion
only, so that conjunction is exactly validity for the revised task, and no new
verifier semantics enter here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE / 'ops'), str(CODE / 'src')]

from followup_metrics import atomic_new, file_sha, sha  # noqa: E402
import portfolio_withdrawals as pw  # noqa: E402

MAX_TOKENS = 192
MAX_MODEL_LEN = 8192


def grade_fresh(domain, row, text):
    """Grade a new generation on the contract's surface for this domain."""
    from frontier_modebench_contract import grade_response

    return grade_response(1, domain, row, text)


def grade_saved(row, text):
    """Grade a retained response, which is already the environment's own output.

    PantryPlan's retained draws store the projection of the policy's support mask
    rather than the mask itself, so the mask decode must not run again here. The
    other four domains retain the policy's answer verbatim and the two paths
    agree; the worker checks that against every retained grade regardless.
    """
    from oat_drgrpo.math_grader import validated_modebench_outcome_key

    key = validated_modebench_outcome_key(text, row['answer'])
    return {'verified': key is not None, 'canonical_key': key}


def survived(domain, withdrawal, census, key):
    """Whether one verified outcome is still valid after the withdrawal."""
    if key is None:
        return False
    verdict = pw.survives(domain, tuple(withdrawal['option']), key,
                          frozenset(withdrawal['surviving']), census)
    if verdict is None:
        raise ValueError('undecidable outcome under the frozen predicate: ' + key)
    return verdict


def survivors(samples, domain, withdrawal, census):
    return [index for index, sample in enumerate(samples)
            if sample['verified'] and survived(domain, withdrawal, census, sample['canonical_key'])]


def recovery_messages(messages, update, noun, portfolio, attempts):
    """The revised task, the saved portfolio, and the verifier's rejections."""
    text = messages[1]['content'] + '\n\nUPDATED REQUIREMENTS:\n' + update
    text += (f'\n\nPreviously saved {noun}s (some may be invalid now):\n'
             + '\n'.join(f'{i + 1}. {s["text"].strip()}' for i, s in enumerate(portfolio)))
    if attempts:
        text += ('\n\nRecovery attempts rejected by the verifier:\n'
                 + '\n'.join(f'{i + 1}. {s["text"].strip()}\nVerifier: invalid under the '
                             'updated requirements.' for i, s in enumerate(attempts)))
    text += (f'\n\nReturn one valid {noun} for the updated requirements using the required '
             'boxed format.')
    return [messages[0], {'role': 'user', 'content': text}]


def diversity_messages(messages, noun, history):
    shown = '\n'.join(f'{i + 1}. {s["text"].strip()}' for i, s in enumerate(history)) or '(none yet)'
    text = (messages[1]['content'] + '\n\nPrevious responses (including invalid attempts):\n'
            + shown + f'\n\nGive one valid {noun} whose executed outcome differs from every '
            'previous response, where possible. Use the required boxed format.')
    return [messages[0], {'role': 'user', 'content': text}]


def run(planpath, index):
    inputs = json.loads(planpath.read_text())
    if inputs.get('outcomes_read') is not False:
        raise ValueError('frozen inputs must record an outcome-blind freeze')
    for name, digest in inputs['code_sha256'].items():
        if file_sha(CODE / name) != digest:
            raise ValueError('frozen code changed: ' + name)
    if file_sha(inputs['frozen_withdrawals']['path']) != inputs['frozen_withdrawals']['sha256']:
        raise ValueError('frozen withdrawal protocol changed')
    checkpoint = inputs['checkpoints'][index]
    domain = checkpoint['domain']
    tasks = inputs['tasks'][domain]
    noun = inputs['nouns'][domain]
    # The frozen surface must still be the contract's, or this cell would generate
    # on a different interface than the one it was registered for.
    from frontier_modebench_contract import profile_metadata
    if inputs['interface']['template'][domain] != profile_metadata(1, domain)['template_name']:
        raise ValueError('frozen prompt surface no longer matches the contract: ' + domain)
    out = Path(inputs['base']) / 'results' / checkpoint['label'].replace('/', '__')
    out.mkdir(parents=True, exist_ok=True)
    identity = sha({'inputs': file_sha(planpath), 'checkpoint': checkpoint})

    with (out / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'result.json').exists():
            done = json.loads((out / 'result.json').read_text())
            if done['identity'] != identity or done['status'] != 'complete':
                raise ValueError('a completed cell disagrees with this plan')
            print(json.dumps({'event': 'already_complete', 'checkpoint': checkpoint['label']}),
                  flush=True)
            return
        for entry in checkpoint['files']:
            path = Path(checkpoint['model_path']) / entry['name']
            if 'sha256' in entry and file_sha(path) != entry['sha256']:
                raise ValueError('checkpoint weights changed: ' + str(path))
            if not path.is_file():
                raise ValueError('checkpoint file absent: ' + str(path))

        from frontier_modebench_contract import make_messages
        import vllm
        llm = vllm.LLM(model=checkpoint['model_path'], dtype='float16', max_model_len=MAX_MODEL_LEN,
                       gpu_memory_utilization=.65, swap_space=4, enable_prefix_caching=True,
                       enforce_eager=True)
        tok = llm.get_tokenizer()
        runtime = out / 'runtime.json'
        if not runtime.exists():
            atomic_new(runtime, {'identity': identity, 'hostname': os.uname().nodename,
                                 'job': os.environ.get('SLURM_JOB_ID'),
                                 'versions': {name: importlib.metadata.version(name)
                                              for name in ('torch', 'vllm', 'transformers')},
                                 'created_at': datetime.now(timezone.utc).isoformat()})
        seeds, receipts = {}, []

        def call(uid, messages, row, temperature=1.0, n=1):
            seed = 1000000000 + int(sha([checkpoint['label'], uid])[:15], 16) % 1000000000
            for child in range(seed, seed + n):
                if seeds.setdefault(child, uid) != uid:
                    raise ValueError('seed collision')
            rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_tokens = len(tok.encode(rendered, add_special_tokens=False))
            if prompt_tokens + MAX_TOKENS > MAX_MODEL_LEN:
                raise ValueError('saved history exceeds the frozen context; no silent truncation')
            request = {'uid': uid, 'messages': messages, 'row_sha256': sha(row), 'n': n,
                       'temperature': temperature, 'seed': seed, 'max_tokens': MAX_TOKENS,
                       'prompt_tokens': prompt_tokens}
            path = out / 'requests' / (uid + '.json')
            binding = sha([identity, request])
            if path.exists():
                result = json.loads(path.read_text())
                if result['binding'] != binding or len(result['samples']) != n:
                    raise ValueError('retained receipt disagrees: ' + uid)
            else:
                # Level 1 carries no syntax constraint in any domain, so the
                # request surface here is the one the cohort was evaluated on.
                params = vllm.SamplingParams(n=n, temperature=temperature, top_p=1.0,
                                             max_tokens=MAX_TOKENS, seed=seed)
                start = time.monotonic()
                outputs = llm.generate([rendered], [params], use_tqdm=False)
                seconds = time.monotonic() - start
                if len(outputs) != 1 or len(outputs[0].outputs) != n:
                    raise ValueError('generation returned an unexpected shape')
                samples = []
                for offset, sample in enumerate(sorted(outputs[0].outputs, key=lambda s: s.index)):
                    graded = grade_fresh(domain, row, sample.text)
                    samples.append({'text': sample.text, 'verified': graded['verified'],
                                    'canonical_key': graded['canonical_key'],
                                       'output_tokens': len(sample.token_ids),
                                    'finish_reason': sample.finish_reason, 'seed': seed + offset})
                result = {'binding': binding, 'request': request, 'samples': samples,
                          'generation_wall_seconds': seconds,
                          'input_tokens_per_response': prompt_tokens,
                          'logical_input_tokens': prompt_tokens * n}
                atomic_new(path, result)
            receipts.append({'path': str(path), 'sha256': file_sha(path)})
            return result['samples']

        dev = [task for task in tasks if task['split'] == 'dev']
        test = [task for task in tasks if task['split'] == 'test']
        calibration = []
        for temperature in inputs['temperature_grid']:
            values = []
            for task in dev:
                row = task['row']
                census = frozenset(task['census'])
                samples = call(f'dev_t{temperature}_{task["id"]}',
                               make_messages(1, domain, row), row, temperature, 8)
                values.append(sum(bool(survivors(samples, domain, w, census))
                                  for w in task['withdrawals']) / len(task['withdrawals']))
            calibration.append({'temperature': temperature,
                                'survival': sum(values) / len(values),
                                'dev_problems': len(values), 'per_problem': values})
        chosen = min(calibration, key=lambda r: (-r['survival'], abs(r['temperature'] - 1),
                                                 r['temperature']))['temperature']
        if not (out / 'calibration.json').exists():
            atomic_new(out / 'calibration.json', {'identity': identity, 'results': calibration,
                                                  'chosen_temperature': chosen})

        # Only now is a saved portfolio read: the withdrawals and the split were
        # certified and frozen before this worker was submitted.
        saved = inputs['saved_portfolio'][checkpoint['label']]
        if file_sha(saved['path']) != saved['sha256']:
            raise ValueError('saved portfolio draws changed')
        stored = {}
        for line in Path(saved['path']).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('evaluation_kind') != 'fixed_seed_sampled_k_neutral':
                continue
            # The training-run draw file carries every evaluated step, so the
            # terminal step is selected explicitly rather than by file position.
            if record.get('step') != saved['step'] or record.get('draw_index') != saved['draw_index']:
                continue
            for prompt in record['prompts']:
                stored[prompt['prompt_index']] = prompt

        summaries = []
        for task in test:
            row = task['row']
            census = frozenset(task['census'])
            messages = make_messages(1, domain, row)
            prompt = stored[row['prompt_index']]
            ordinary = []
            for text, key, reward in zip(prompt['responses'], prompt['answer_keys'],
                                         prompt['rewards']):
                graded = grade_saved(row, text)
                if graded['verified'] != (reward > 0 and key is not None) or \
                        (graded['verified'] and graded['canonical_key'] != key):
                    raise ValueError('saved grade disagrees with a fresh grade')
                ordinary.append({'text': text, 'verified': graded['verified'],
                                 'canonical_key': graded['canonical_key'],
                                 'output_tokens': len(tok.encode(text, add_special_tokens=False)),
                                 'source': 'saved_reference_draw'})
            if len(ordinary) != 8:
                raise ValueError('saved portfolio is not eight responses')
            portfolios = {'ordinary': ordinary,
                          'temperature': call('test_temp_' + task['id'], messages, row, chosen, 8)}
            diverse = []
            for turn in range(8):
                diverse += call(f'test_diverse_{task["id"]}_{turn}',
                                diversity_messages(messages, noun, diverse), row, 1.0, 1)
            portfolios['diversity_prompt'] = diverse
            record = out / 'portfolios' / f'{task["id"]}.json'
            if not record.exists():
                atomic_new(record, {'identity': identity, 'task': task['id'],
                                    'chosen_temperature': chosen, 'portfolios': portfolios})
            for strategy, samples in portfolios.items():
                for position, withdrawal in enumerate(task['withdrawals']):
                    found = survivors(samples, domain, withdrawal, census)
                    attempts, recovered = [], bool(found)
                    while not recovered and len(attempts) < inputs['max_recovery_calls']:
                        uid = f'recovery_{task["id"]}_{strategy}_{position}_{len(attempts)}'
                        fresh = call(uid, recovery_messages(messages, withdrawal['update'], noun,
                                                            samples, attempts), row, 1.0, 1)
                        attempts += fresh
                        recovered = fresh[0]['verified'] and survived(
                            domain, withdrawal, census, fresh[0]['canonical_key'])
                    summaries.append({
                        'task': task['id'], 'prompt_index': row['prompt_index'],
                        'strategy': strategy, 'withdrawal': withdrawal['option'],
                        'initial_correct': sum(s['verified'] for s in samples),
                        'initial_distinct': len({s['canonical_key'] for s in samples
                                                 if s['verified']}),
                        'saved_usable': len(found), 'zero_call_recovery': bool(found),
                        'recovered': recovered, 'recovery_calls': len(attempts),
                        'recovery_output_tokens': sum(s['output_tokens'] for s in attempts)})
            print(json.dumps({'event': 'problem_complete', 'checkpoint': checkpoint['label'],
                              'task': task['id'], 'requests': len(receipts)}), flush=True)

        atomic_new(out / 'result.json', {
            'schema': 'modebench-recovery-five-domain-result-v1', 'status': 'complete',
            'identity': identity, 'checkpoint': checkpoint, 'domain': domain,
            'calibration': calibration, 'chosen_temperature': chosen, 'records': summaries,
            'request_receipts': receipts,
            'saved_source': {'path': saved['path'], 'sha256': saved['sha256'],
                             'step': saved['step'], 'draw_index': saved['draw_index']},
            'cost_note': 'Receipts retain logical input and output tokens and generation timing. '
                         'Shared prefixes change realized compute, so equal response counts are '
                         'not a compute-cost claim. The ordinary portfolio is a retained draw and '
                         'generates nothing here.'})
        print(json.dumps({'event': 'complete', 'checkpoint': checkpoint['label'],
                          'requests': len(receipts), 'records': len(summaries)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    run(args.inputs.resolve(), args.index)


if __name__ == '__main__':
    main()
