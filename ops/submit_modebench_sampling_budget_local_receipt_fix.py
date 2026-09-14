#!/usr/bin/env python3
"""Operational path-type compatibility fix; sealed collector/plan unchanged."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import prepare_modebench_sampling_budget_local as launcher
LOCAL = ROOT / 'artifacts/modebench_discovery_curves_20260911/local'
original_hash = launcher.file_sha
plan = json.loads((LOCAL / 'plan.json').read_text())
assert original_hash(Path(launcher.__file__)) == plan['code_sha256'][str(LOCAL / 'code/ops/prepare_modebench_sampling_budget_local.py')]
assert original_hash(LOCAL / 'plan.json') == '7973dcdcc881b8dedab242036ce48b5e32a41ce8f505fc0af6b77fc7c0a511fb'
launcher.file_sha = lambda path: original_hash(Path(path))
assert launcher.file_sha(str(LOCAL / 'plan.json')) == original_hash(LOCAL / 'plan.json')
original_verify = launcher.verify_checkpoint

def verify(checkpoint):
    model = original_verify(checkpoint)
    print(json.dumps({'event': 'checkpoint_authenticated', 'checkpoint': checkpoint['label']}), flush=True)
    return model

launcher.verify_checkpoint = verify
launcher.submit()
