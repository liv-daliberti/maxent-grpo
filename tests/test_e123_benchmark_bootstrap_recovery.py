"""E123 bootstrap repair must preserve immutable science and reject other drift."""
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('e123_bootstrap_recovery', ROOT / 'ops/exp_scaling/recover_e123_benchmark_bootstrap_20260910.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_wrapper_binds_real_roots_before_snapshot_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path / 'real repo')
    snap = tmp_path / 'snapshot'
    (snap / 'ops').mkdir(parents=True)
    (snap / 'ops/repo_env.sh').write_text('test "$MAXENT_GRPO_ROOT" = "$OAT_ZERO_REPO_ROOT"\ntest "$CUDA_HOME" = "$OAT_ZERO_REPO_ROOT/var/cuda124_toolkit"\ntest "$PYTHONPYCACHEPREFIX" = "$OAT_ZERO_REPO_ROOT/var/pycache"\n')
    nvcc = recovery.ROOT / 'var/cuda124_toolkit/bin/nvcc'
    nvcc.parent.mkdir(parents=True); nvcc.write_text('#!/bin/sh\nexit 0\n'); nvcc.chmod(0o755)
    script = recovery.wrapper({'snapshot_root': str(snap)}, ['true'])
    result = subprocess.run(['bash'], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_snapshot_drift_only_allows_unregistered_var_cache(tmp_path):
    root = tmp_path / 'snapshot'; root.mkdir()
    registered = root / 'code.py'; registered.write_text('pass\n')
    (root / 'SNAPSHOT_IDENTITY.json').write_text('{}')
    snapshot = {'root': str(root), 'inventory': {'code.py': {'sha256': recovery.auto.launch.digest(registered)}}}
    extra = root / 'var/pycache/junk.pyc'; extra.parent.mkdir(parents=True); extra.write_bytes(b'cache')
    assert recovery.inventory(snapshot) == ['var/pycache/junk.pyc']
    (root / 'unexpected.py').write_text('pass\n')
    with pytest.raises(ValueError, match='outside bootstrap'): recovery.inventory(snapshot)
    (root / 'unexpected.py').unlink(); registered.write_text('changed\n')
    with pytest.raises(ValueError, match='frozen file bytes changed'): recovery.inventory(snapshot)
