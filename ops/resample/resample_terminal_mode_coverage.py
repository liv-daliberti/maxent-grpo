#!/usr/bin/env python3
"""Re-measure one terminal cell's mode coverage under disjoint sampling streams.

The registered training evaluation seeded draw ``d`` at ``seed_base + d``, and
vLLM 0.8.4 V0 expands an ``n=8`` request with seed ``s`` into children
``s..s+7``. Four consecutive draws therefore share children: the thirty-two
saved responses per prompt carry only eleven distinct nominal streams, with
multiplicities ``[1,2,3,4,4,4,4,4,3,2,1]``. PCMD pools a prompt's four draws, so
it is exactly the statistic that reuse contaminates.

This runs the same evaluation off the training loop, with one difference that
matters: request seeds come from ``modebench_independent_seeds``, the aligned
eight-block policy already registered for the frozen base-model grid. Each
(prompt, draw) gets its own block, so the four draws contribute thirty-two
disjoint streams.

``--mode reproduce`` instead replays the original per-draw seed and asserts the
saved responses come back byte for byte. Nothing about the independent numbers
is believable until that passes on the same GPU model the cell trained on: it is
what separates "we re-measured the cell" from "we ran something adjacent to it".
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'src', ROOT / 'ops', str(Path(__file__).resolve().parent)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SCHEMA = 'pmd-independent-resample-receipt-v1'
ENGINE_CONTRACT = {'vllm_version': '0.8.4', 'engine': 'V0',
                   'parallel_sample_seed_policy': 'request_seed_plus_sample_index'}


def receipt_stamp(cell: dict) -> str:
    """Identify a receipt uniquely, including the level it was evaluated on.

    Without the level, a Level-3 measurement of a cell writes to the same name
    as its Level-1 measurement: the second is skipped as already done, or worse
    overwrites the first. The same policy measured on two test sets is two
    results, not one.
    """
    return '__'.join((str(cell['level']), str(cell['scale']), str(cell['domain']),
                      str(cell['method']), f"s{cell['seed']}"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def gpu_model() -> str:
    import torch
    require(torch.cuda.is_available(), 'no CUDA device visible')
    name = torch.cuda.get_device_name(0).lower()
    for token in ('a100', 'a6000', 'a5000', 'h100', 'l40', 'rtx'):
        if token in name.replace(' ', ''):
            return token
    return name.replace(' ', '_')


def stage_model(cell: dict, staging: Path) -> Path:
    """Materialise the terminal checkpoint on node-local disk.

    Metadata stayed in the run directory when the weights were retired to the
    Hub, so only the weight files are fetched, and each is checked against the
    digest the archive receipt recorded before the local copy was removed.
    """
    weights = cell['weights']
    export = Path(weights['export_dir'])
    target = staging / 'model'
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(export.iterdir()):
        if item.is_file():
            shutil.copyfile(item, target / item.name)
    if weights['location'] == 'local':
        for item in weights['files']:
            source = export / item['relative_path']
            require(source.is_file(), f'local weight file vanished: {source}')
            if not (target / item['relative_path']).is_file():
                shutil.copyfile(source, target / item['relative_path'])
        return target
    from huggingface_hub import hf_hub_download
    for item in weights['files']:
        remote = f'{weights["repo_prefix"]}/{item["relative_path"]}'
        fetched = hf_hub_download(
            repo_id=weights['repo_id'], filename=remote,
            revision=weights['commit_sha'], local_dir=str(staging / 'download'))
        digest = sha_file(Path(fetched))
        require(digest == item['sha256'],
                f'restored weight digest {digest} != archived {item["sha256"]} for {remote}')
        shutil.move(fetched, target / item['relative_path'])
    return target


def load_dataset_prompts(cell: dict) -> list[dict]:
    """Read an admitted evaluation split instead of the run's saved draws.

    Used when a checkpoint is measured on a test set it was never evaluated on,
    so there is no saved draw log to take prompts from. The split is loaded
    through the same ``test_split`` the run trained against, and the reference
    string is the dataset's own answer field -- the same value the training
    evaluation passed to the grader.
    """
    from datasets import load_from_disk, disable_progress_bar
    disable_progress_bar()
    config = cell['eval_config']
    split_name = config.get('test_split', 'multi_answer')
    data = load_from_disk(str(config['eval_data']))
    require(split_name in data, f'{config["eval_data"]} has no {split_name} split')
    rows = data[split_name]
    require(len(rows) > 0, f'empty evaluation split: {config["eval_data"]}')
    return [{'prompt_index': index, 'problem': row['problem'], 'reference': row['answer']}
            for index, row in enumerate(rows)]


def load_prompts(cell: dict) -> list[dict]:
    """Read the terminal prompt set from the run's own saved draw log.

    A step-0 cell evaluates the untrained checkpoint but cannot read its prompts
    from step 0: those records sit at the head of the draw log, outside the
    bounded tail this reads. The evaluation set does not change during a run, so
    such a cell names the step whose record carries the prompts, and the digest
    check below is what proves the set is the right one either way.
    """
    from draw_tail import terminal_draws
    prompt_step = int(cell.get('prompt_step', cell['terminal_step']))
    draws = terminal_draws(Path(cell['draws_path']), prompt_step)
    greedy = terminal_draws(Path(cell['draws_path']), prompt_step,
                            kind='deterministic_greedy_trace_neutral')
    # Only the prompt set is taken from the saved log, and the digest below is
    # what guarantees it is the right one. The number of saved draws is a
    # property of the run that was recorded; the number this job will generate
    # is a property of the requested budget, and a step-0 collection at a wider
    # budget deliberately differs from what the run happened to save.
    require(draws, 'no saved sampled draws at this step to take prompts from')
    rows = [{'prompt_index': int(p['prompt_index']), 'problem': p['prompt'],
             'reference': p['reference']} for p in draws[0]['prompts']]
    rows.sort(key=lambda r: r['prompt_index'])
    digest = sha_text(json.dumps(rows, sort_keys=True, separators=(',', ':')))
    require(digest == cell['prompt_set_sha256'],
            'terminal prompt set no longer matches the manifest digest')
    return rows, draws, greedy


def build_engine(model_dir: Path, cell: dict):
    import vllm
    require(os.environ.get('VLLM_USE_V1') == '0',
            'the child-seed contract requires VLLM_USE_V1=0')
    require(vllm.__version__ == ENGINE_CONTRACT['vllm_version'],
            f'seed contract requires vLLM {ENGINE_CONTRACT["vllm_version"]}, '
            f'found {vllm.__version__}')
    config = cell['eval_config']
    # These are oat's own engine arguments (oat/interface.py), including the
    # KV-cache fraction the run was launched with. Cache size decides how
    # requests batch, and batching decides the reduction order the kernels use,
    # so an engine built more generously is not the engine that produced the
    # saved responses.
    return vllm.LLM(model=str(model_dir), trust_remote_code=True,
                    tensor_parallel_size=1, dtype='bfloat16',
                    gpu_memory_utilization=float(config['vllm_gpu_ratio']),
                    enable_prefix_caching=False,
                    max_model_len=int(config['max_model_len']))


def sampling_params(cell: dict, seed: int, allowed_token_ids, template: str,
                    horizon: int | None = None):
    import vllm
    config = cell['eval_config']
    # A canonical-action run configures its evaluation params to a fixed
    # horizon with EOS ignored, and the mode-coverage request inherits that
    # min_tokens/max_tokens/ignore_eos even though it rebuilds the object. Free
    # decoding to eval_generate_max_length would be a different measurement.
    return vllm.SamplingParams(
        n=int(config['eval_mode_coverage_k']),
        temperature=float(config['eval_mode_coverage_temperature']),
        top_p=1.0,
        max_tokens=int(horizon if horizon else config['eval_generate_max_length']),
        min_tokens=int(horizon) if horizon else 0,
        ignore_eos=bool(horizon),
        stop=None,
        stop_token_ids=None,
        allowed_token_ids=allowed_token_ids,
        include_stop_str_in_output=False,
        seed=int(seed),
    )


def greedy_step_profile(cell: dict, fresh: dict[int, str]) -> dict:
    """Locate the restored checkpoint among the run's saved greedy traces.

    Byte-exact replay is not available: the training evaluation ran inside a
    collocated engine and a standalone one differs slightly in the forward pass,
    which flips a few percent of near-tie token choices. That makes "are these
    the same weights?" a question worth answering separately, and the greedy
    traces answer it. Agreement with the restored checkpoint should climb as the
    saved traces approach the terminal step and peak there; a checkpoint that is
    not the evaluated policy would peak somewhere else or nowhere.
    """
    import json as _json
    profile = {}
    with Path(cell['draws_path']).open() as handle:
        for line in handle:
            try:
                record = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if record.get('evaluation_kind') != 'deterministic_greedy_trace_neutral':
                continue
            saved = {int(p['prompt_index']): p['responses'][0] for p in record['prompts']}
            shared = [i for i in fresh if i in saved]
            if not shared:
                continue
            agree = sum(1 for i in shared if saved[i] == fresh[i])
            profile[int(record['step'])] = {'agreement': agree / len(shared),
                                            'compared': len(shared)}
    terminal = int(cell['terminal_step'])
    if not profile:
        return {'by_step': {}, 'terminal_step': terminal, 'best_step': None,
                'peaks_at_terminal_step': False, 'terminal_agreement': None}
    # Ties go to the terminal step, and the comparison carries a tolerance: the
    # standalone engine already disagrees with the collocated one by a few
    # percent, so demanding a strict argmax would reject correct checkpoints
    # whose last few saved traces sit inside that noise.
    best = max(sorted(profile, reverse=True), key=lambda step: profile[step]['agreement'])
    top = profile[best]['agreement']
    here = profile[terminal]['agreement'] if terminal in profile else None
    tolerance = 0.05
    return {'by_step': profile, 'terminal_step': terminal, 'best_step': best,
            'best_agreement': top, 'tolerance': tolerance,
            'peaks_at_terminal_step': here is not None and here >= top - tolerance,
            'terminal_agreement': here}


def greedy_params(cell: dict, allowed_token_ids, horizon: int | None = None):
    import vllm
    config = cell['eval_config']
    return vllm.SamplingParams(
        n=1, temperature=0.0, top_p=1.0,
        max_tokens=int(horizon if horizon else config['eval_generate_max_length']),
        min_tokens=int(horizon) if horizon else 0,
        ignore_eos=bool(horizon), stop=None, stop_token_ids=None,
        allowed_token_ids=allowed_token_ids, include_stop_str_in_output=False, seed=0)


def draw_seed_plan(cell: dict, rows: list[dict], mode: str) -> dict:
    """Request seeds per (prompt, draw), under the original or independent policy."""
    config = cell['eval_config']
    draws = int(config['eval_mode_coverage_draws'])
    k = int(config['eval_mode_coverage_k'])
    if mode == 'greedy':
        return {'policy': 'deterministic_greedy_trace', 'request_seeds': [[0] for _ in rows],
                'child_seeds': [[[0]] for _ in rows],
                'distinct_child_streams_per_prompt': 1,
                'total_child_seeds': len(rows), 'distinct_child_seeds': 1}
    if mode == 'reproduce':
        base = int(config['eval_mode_coverage_seed'])
        schedule = [[base + d for d in range(draws)] for _ in rows]
        policy = 'original_shared_request_seed_base_plus_draw_index'
    else:
        from modebench_independent_seeds import POLICY, SAMPLES, seed_schedule
        require(SAMPLES == k, f'seed policy blocks are {SAMPLES} wide but K={k}')
        schedule = seed_schedule(cell['domain'], [r['problem'] for r in rows],
                                 list(range(draws)))
        policy = POLICY
    children = [[[seed + i for i in range(k)] for seed in row] for row in schedule]
    flat = [child for row in children for block in row for child in block]
    return {'policy': policy, 'request_seeds': schedule, 'child_seeds': children,
            'distinct_child_streams_per_prompt': len({
                child for block in children[0] for child in block}),
            'total_child_seeds': len(flat), 'distinct_child_seeds': len(set(flat))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cell', type=int, required=True, help='index into manifest cells')
    parser.add_argument('--mode', choices=('independent', 'reproduce', 'greedy'),
                        default='independent',
                        help="'greedy' replays the saved temperature-0 trace, which is "
                             'immune to sampling RNG and so isolates weight identity')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--staging', type=Path, default=Path('/tmp/pmd_resample'))
    parser.add_argument('--keep-staging', action='store_true')
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cell = manifest['cells'][args.cell]
    stamp = receipt_stamp(cell)
    receipt_path = args.output_root / args.mode / f'{stamp}.json.gz'
    require(not receipt_path.exists(), f'receipt already written: {receipt_path}')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    observed_gpu = gpu_model()
    require(observed_gpu == cell['gpu'],
            f'{stamp} trained on {cell["gpu"]} but this job holds {observed_gpu}; '
            'a decoding measurement is only comparable on the same GPU model')

    staging = args.staging / f'{stamp}.{os.getpid()}'
    started = time.time()
    if cell.get('prompt_source') == 'dataset':
        # A different test set than this run was evaluated on: no saved draws to
        # compare against, so reproduce and greedy replay do not apply here.
        require(args.mode == 'independent',
                'dataset-sourced cells support --mode independent only')
        rows, original_draws, greedy_saved = load_dataset_prompts(cell), [], []
    else:
        rows, original_draws, greedy_saved = load_prompts(cell)
    plan = draw_seed_plan(cell, rows, args.mode)

    from oat_drgrpo.templates import TEMPLATE_FACTORY
    from oat_drgrpo.math_grader import extract_normalized_final_answer
    from oat_drgrpo.actor import MATHOracle

    template = cell['eval_config']['prompt_template']
    require(template in TEMPLATE_FACTORY, f'unknown prompt template {template}')
    formatted = [TEMPLATE_FACTORY[template](row['problem']) for row in rows]
    references = [row['reference'] for row in rows]

    model_dir = stage_model(cell, staging)
    # A run never saw a prompt longer than its own prompt_max_length: the
    # training loader drops those rows before the actor is called, and the
    # learner raises rather than silently keeping one (learner/run.py). A
    # cross-level cell reads a harder split than the run trained on, so that
    # split can carry prompts the run's surface never admitted -- at Level 3 and
    # Level 5 some PantryPlan prompts run past the 640 the Pantry runs allow.
    # Applying the same rule here keeps the replay faithful instead of either
    # crashing the cell or widening the context the run was evaluated under.
    # prompt_max_length is a property of the domain, not of the arm, so every
    # run in a domain drops the same prompts and the cells stay comparable.
    prompt_budget = cell['eval_config'].get('prompt_max_length')
    prompt_admission = None
    if prompt_budget is not None:
        from transformers import AutoTokenizer
        budget_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        lengths = [len(budget_tokenizer(text, add_special_tokens=False)['input_ids'])
                   for text in formatted]
        keep = [i for i, n in enumerate(lengths) if n <= int(prompt_budget)]
        prompt_admission = {
            'prompt_max_length': int(prompt_budget),
            'prompts_rendered': len(formatted),
            'prompts_admitted': len(keep),
            'prompts_over_budget': len(formatted) - len(keep),
            'longest_rendered_tokens': max(lengths) if lengths else 0,
            'dropped_prompt_indices': [int(rows[i]['prompt_index'])
                                       for i, n in enumerate(lengths)
                                       if n > int(prompt_budget)],
            'rule': 'training drops rows over prompt_max_length; replayed here',
        }
        require(keep, 'every prompt in this split exceeds the run prompt_max_length')
        if len(keep) != len(formatted):
            rows = [rows[i] for i in keep]
            formatted = [formatted[i] for i in keep]
            references = [references[i] for i in keep]
    llm = build_engine(model_dir, cell)

    canonical_task = cell.get('canonical_action_task', 'none')
    allowed_token_ids = None
    action_space = None
    horizon = None
    if canonical_task != 'none':
        from oat_drgrpo.canonical_actions import resolve_canonical_action_space
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        action_space = resolve_canonical_action_space(tokenizer, canonical_task)
        allowed_token_ids = [int(t) for t in action_space.union_token_ids]
        horizon = int(action_space.horizon)

    oracle = MATHOracle(template=template, verifier_version='fast')
    draws_out = []
    draw_total = 1 if args.mode == 'greedy' else int(cell['eval_config']['eval_mode_coverage_draws'])
    width = 1 if args.mode == 'greedy' else int(cell['eval_config']['eval_mode_coverage_k'])
    for draw_index in range(draw_total):
        # The training loop walks the evaluation set with a DataLoader and hands
        # each batch to the actor as its own generate call, so the engine never
        # sees more than eval_batch_size prompts at once. Continuous batching
        # makes that visible in the arithmetic: submit all 128 together and the
        # requests batch differently and the sampled tokens diverge. Reproducing
        # the saved responses means reproducing the call boundaries too.
        batch = int(cell['eval_config']['eval_batch_size'])
        outputs = []
        for start in range(0, len(rows), batch):
            stop = min(start + batch, len(rows))
            seeds = [plan['request_seeds'][i][draw_index] for i in range(start, stop)]
            if args.mode == 'greedy':
                params = greedy_params(cell, allowed_token_ids, horizon)
            elif len(set(seeds)) == 1:
                # The original neutral path builds one SamplingParams for the
                # whole batch; only the independent policy needs per-prompt ones.
                params = sampling_params(cell, seeds[0], allowed_token_ids, template, horizon)
            else:
                params = [sampling_params(cell, seed, allowed_token_ids, template, horizon)
                          for seed in seeds]
            outputs.extend(llm.generate(formatted[start:stop], params))
        require(len(outputs) == len(rows), 'engine returned the wrong request grid')
        responses, keys, rewards = [], [], []
        flat_responses, flat_refs = [], []
        for index, generated in enumerate(outputs):
            texts = [sample.text.strip() for sample in generated.outputs]
            if action_space is not None:
                from oat_drgrpo.canonical_actions import decode_canonical_action_response
                texts = [decode_canonical_action_response(action_space.task, text,
                                                          references[index])
                         for text in texts]
            require(len(texts) == width, f'prompt {index} returned {len(texts)} samples')
            responses.append(texts)
            flat_responses.extend(texts)
            flat_refs.extend([references[index]] * len(texts))
        reward_tensor, _ = oracle.get_reward([''] * len(flat_responses), flat_responses, flat_refs)
        flat_rewards = [float(v) for v in reward_tensor.tolist()]
        flat_keys = [extract_normalized_final_answer(text, template=template, gt_answer=ref)
                     for text, ref in zip(flat_responses, flat_refs)]
        for index in range(len(rows)):
            lo, hi = index * width, (index + 1) * width
            rewards.append(flat_rewards[lo:hi])
            keys.append([None if k is None else str(k) for k in flat_keys[lo:hi]])
        draws_out.append({
            'draw_index': draw_index,
            'request_seeds': [plan['request_seeds'][i][draw_index] for i in range(len(rows))],
            'prompts': [{'prompt_index': rows[i]['prompt_index'],
                         'responses': responses[i], 'rewards': rewards[i],
                         'answer_keys': keys[i],
                         'child_seeds': plan['child_seeds'][i][draw_index]}
                        for i in range(len(rows))],
        })

    receipt = {
        'schema': SCHEMA, 'mode': args.mode, 'created_at_utc': utc(),
        'host': socket.gethostname(), 'gpu': observed_gpu,
        'elapsed_seconds': round(time.time() - started, 1),
        'engine': {**ENGINE_CONTRACT, 'dtype': 'bfloat16',
                   'generate_call_batch_size': int(cell['eval_config']['eval_batch_size']),
                   'gpu_memory_utilization': float(cell['eval_config']['vllm_gpu_ratio']),
                   'enable_prefix_caching': False, 'tensor_parallel_size': 1},
        # Dataset-sourced cells carry no manifest digest to echo, so the digest
        # of the prompts actually evaluated is recorded either way; it is what a
        # later reader needs to confirm two receipts saw the same problems.
        'cell': {**{k: cell[k] for k in ('scale', 'level', 'domain', 'method', 'seed',
                                         'run_dir', 'terminal_step', 'gpu')},
                 'prompt_set_sha256': sha_text(json.dumps(
                     rows, sort_keys=True, separators=(',', ':'))),
                 'prompt_source': cell.get('prompt_source', 'saved_draws'),
                 'trained_level': cell.get('trained_level', cell.get('level')),
                 'transfer': bool(cell.get('transfer', False)),
                 # What the prompt budget admitted, so a reader can see the
                 # denominator rather than infer it from the digest.
                 'prompt_admission': prompt_admission},
        'eval_config': cell['eval_config'],
        'canonical_action_task': canonical_task, 'canonical_horizon': horizon,
        'seed_plan': {k: plan[k] for k in ('policy', 'distinct_child_streams_per_prompt',
                                           'total_child_seeds', 'distinct_child_seeds')},
        'manifest': {'path': str(args.manifest), 'cell_index': args.cell,
                     'schema': manifest['schema']},
        'draws': draws_out,
    }

    if args.mode in ('reproduce', 'greedy'):
        if args.mode == 'greedy':
            from draw_tail import terminal_draws as _terminal
            baseline = _terminal(Path(cell['draws_path']), int(cell['terminal_step']),
                                 kind='deterministic_greedy_trace_neutral')
            require(len(baseline) == 1, 'expected exactly one saved greedy trace')
        else:
            baseline = original_draws
            # Reproduce mode replays the recorded schedule, so the saved draw
            # count is the thing being reproduced and must match exactly.
            require(len(baseline) == len(draws_out),
                    f'reproduce needs {len(draws_out)} saved draws, found {len(baseline)}')
        mismatches = []
        for draw_index, draw in enumerate(draws_out):
            saved = {int(p['prompt_index']): p for p in baseline[draw_index]['prompts']}
            for fresh in draw['prompts']:
                before = saved[fresh['prompt_index']]
                if list(before['responses']) != list(fresh['responses']):
                    mismatches.append({'draw_index': draw_index,
                                       'prompt_index': fresh['prompt_index']})
        identical_samples = total_samples = 0
        for draw_index, draw in enumerate(draws_out):
            saved = {int(p['prompt_index']): p for p in baseline[draw_index]['prompts']}
            for fresh in draw['prompts']:
                for before, after in zip(saved[fresh['prompt_index']]['responses'],
                                         fresh['responses']):
                    total_samples += 1
                    identical_samples += before == after
        if args.mode == 'greedy':
            receipt['checkpoint_identity'] = greedy_step_profile(
                cell, {int(p['prompt_index']): p['responses'][0]
                       for p in draws_out[0]['prompts']})
        receipt['reproduction'] = {
            'compared_prompts': len(rows) * len(draws_out),
            'mismatched_prompts': len(mismatches),
            'exact': not mismatches,
            'compared_samples': total_samples,
            'identical_samples': identical_samples,
            'sample_agreement': identical_samples / total_samples if total_samples else None,
            'first_mismatches': mismatches[:5],
        }

    # Compress into node-local staging, then rename into place, so a killed job
    # never leaves a half-written receipt on shared storage for the collector to
    # read as complete.
    scratch = Path(staging) / receipt_path.name
    with gzip.GzipFile(filename=str(scratch), mode='wb') as raw:
        raw.write(json.dumps(receipt, sort_keys=True).encode('utf-8'))
    shutil.copyfile(str(scratch), str(receipt_path) + '.partial')
    os.replace(str(receipt_path) + '.partial', str(receipt_path))
    require(receipt_path.is_file() and receipt_path.stat().st_size > 0,
            f'receipt did not land: {receipt_path}')
    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
    summary = {'stamp': stamp, 'mode': args.mode, 'output': str(receipt_path),
               'seconds': receipt['elapsed_seconds'],
               'streams_per_prompt': plan['distinct_child_streams_per_prompt']}
    if 'reproduction' in receipt:
        summary['reproduction_exact'] = receipt['reproduction']['exact']
        summary['sample_agreement'] = round(receipt['reproduction']['sample_agreement'], 5)
        if 'checkpoint_identity' in receipt:
            identity = receipt['checkpoint_identity']
            summary['checkpoint_peaks_at_terminal'] = identity['peaks_at_terminal_step']
            summary['terminal_agreement'] = identity['terminal_agreement']
            summary['best_step'] = identity['best_step']
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
