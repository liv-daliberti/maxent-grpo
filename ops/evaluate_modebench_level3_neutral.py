"""Isolated neutral interface over the original independent-stream evaluator.

The old evaluator and historical receipt validator are never modified on disk
or in their imported module. This adapter owns a distinct module and receipt
namespace; all model, syntax, seed, grading and sampling code is reused.
"""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'ops'), str(ROOT/'src')]
from modebench_current_contract import make_messages, CURRENT

_spec = importlib.util.spec_from_file_location('_neutral_level3_evaluator', ROOT/'ops/evaluate_modebench_level3_independent.py')
evaluator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluator)
BASE_CODE_IDENTITY = evaluator.code_identity
INTERFACE = 'python_level3_neutral_calibration_v1'
SCHEMA = 'modebench-level3-neutral-calibration-v1'


def frozen_interface(domain, profile=INTERFACE):
    evaluator.require(domain == 'python_factors' and profile == INTERFACE,
                      'neutral calibration accepts only the new Python interface')
    return {**evaluator.original.frozen_interface(domain, 'level2_qwen_r5'),
            'name': INTERFACE, 'prompt_profile': CURRENT, 'prompt_condition': CURRENT,
            'seed_policy': evaluator.POLICY}


def prompt_messages(domain, problem, profile):
    evaluator.require(domain == 'python_factors' and profile == CURRENT, 'unexpected neutral renderer')
    return make_messages(3, domain, {'problem': problem})


def code_identity():
    paths = ['ops/evaluate_modebench_level3_neutral.py', 'ops/modebench_current_contract.py',
             'ops/frontier_modebench_contract.py']
    return {**BASE_CODE_IDENTITY(), **{p:evaluator.file_sha(ROOT/p) for p in paths}}


def validate_task(task, confirm_eval):
    frozen_interface(task['domain'], task.get('interface', INTERFACE))
    evaluator.require(task['level'] == 'level3', 'only Level 3 is recalibrated')
    evaluator.original.validate_task({**task, 'interface':'level2_qwen_r5'}, confirm_eval)
    evaluator.require(all(type(s) is int for s in task['seeds']), 'integer draw labels required')


evaluator.SCHEMA = SCHEMA
evaluator.INTERFACE = INTERFACE
evaluator.frozen_interface = frozen_interface
evaluator.prompt_messages = prompt_messages
evaluator.code_identity = code_identity
evaluator.validate_task = validate_task

if __name__ == '__main__':
    evaluator.main()
