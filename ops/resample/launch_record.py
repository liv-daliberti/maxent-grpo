"""Recover each run's launch-time evaluation settings from its scheduler record.

The ledgers preserve the ``scontrol show job`` text captured when each run was
submitted, and that text contains the full ``--export`` list. The evaluation
settings the resampler must reproduce -- prompt template, generation budget,
context length, K, draw count and the original mode-coverage seed base -- are
therefore recorded per run rather than inferred from a chain of launchers, and
the requested generic resource names the GPU model the run trained on.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

GRES = re.compile(r'(?:--gres=gpu|[Tt]res[Pp]er[Nn]ode=gres/gpu|gres/gpu):([a-z][a-z0-9]*):\d+')
EXPORT = re.compile(r'--export=([^\s]+)')

EVAL_ENV = {
    'OAT_ZERO_PROMPT_TEMPLATE': 'prompt_template',
    'OAT_ZERO_EVAL_GENERATE_MAX_LENGTH': 'eval_generate_max_length',
    'OAT_ZERO_MAX_MODEL_LEN': 'max_model_len',
    'OAT_ZERO_PROMPT_MAX_LENGTH': 'prompt_max_length',
    'OAT_ZERO_EVAL_BATCH_SIZE': 'eval_batch_size',
    'OAT_ZERO_EVAL_DATA': 'eval_data',
    'OAT_ZERO_EVAL_MODE_COVERAGE_SEED': 'eval_mode_coverage_seed',
    'OAT_ZERO_EVAL_MODE_COVERAGE_K': 'eval_mode_coverage_k',
    'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS': 'eval_mode_coverage_draws',
    'OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE': 'eval_mode_coverage_temperature',
    'OAT_ZERO_VLLM_GPU_RATIO': 'vllm_gpu_ratio',
}
INTEGER = {'eval_generate_max_length', 'max_model_len', 'prompt_max_length',
           'eval_batch_size', 'eval_mode_coverage_seed', 'eval_mode_coverage_k',
           'eval_mode_coverage_draws'}
FLOAT = {'eval_mode_coverage_temperature', 'vllm_gpu_ratio'}


def parse_export(record: str) -> dict[str, str]:
    """Return the ``--export`` environment the job was submitted with."""
    hit = EXPORT.search(record)
    if not hit:
        return {}
    environment: dict[str, str] = {}
    for item in hit.group(1).split(','):
        name, sep, value = item.partition('=')
        if sep:
            environment[name] = value
    return environment


def parse_gpu(record: str) -> str | None:
    hit = GRES.search(record)
    return hit.group(1) if hit else None


def canonical_action_task(record: str) -> str:
    """PantryPlan samples inside a fixed action support; Level 1 otherwise does not."""
    environment = parse_export(record)
    task = environment.get('OAT_ZERO_CANONICAL_ACTION_TASK', 'none')
    if task == 'none' and environment.get('OAT_ZERO_CANONICAL_GRAPH_ACTIONS') == '1':
        return 'graph_coloring'
    return task


def eval_settings(record: str) -> dict[str, object]:
    """Extract the recorded evaluation settings, typed as the args parser would."""
    environment = parse_export(record)
    settings: dict[str, object] = {}
    for name, key in EVAL_ENV.items():
        if name not in environment:
            continue
        raw = environment[name]
        if key in INTEGER:
            settings[key] = int(raw)
        elif key in FLOAT:
            settings[key] = float(raw)
        else:
            settings[key] = raw
    return settings


def launch_records(ledger_dirs) -> dict[str, dict]:
    """Map run_dir to its recorded GPU and evaluation settings."""
    resolved: dict[str, dict] = {}
    for directory in ledger_dirs:
        for path in sorted(Path(directory).glob('*jobs.json')):
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            runs = payload.get('runs')
            if not isinstance(runs, list):
                continue
            for run in runs:
                run_dir = run.get('run_dir')
                record = str(run.get('held_scheduler_record', ''))
                if not run_dir or not record:
                    continue
                entry = {'gpu': run.get('gpu') or parse_gpu(record),
                         'eval_settings': eval_settings(record),
                         'canonical_action_task': canonical_action_task(record),
                         'ledger': path.name}
                previous = resolved.get(str(run_dir))
                if previous is not None:
                    for field in ('gpu', 'eval_settings', 'canonical_action_task'):
                        if previous[field] and entry[field] and previous[field] != entry[field]:
                            raise ValueError(
                                f'ledgers disagree on {field} for {run_dir}: '
                                f'{previous["ledger"]} vs {path.name}')
                    if previous['eval_settings'] and not entry['eval_settings']:
                        continue
                resolved[str(run_dir)] = entry
    return resolved
