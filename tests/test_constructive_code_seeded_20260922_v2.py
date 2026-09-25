"""Regression for ordinary site builtins while retaining isolated seeded startup."""
import os
from pathlib import Path
import subprocess
import pytest
import probe_constructive_code_seeded_20260922 as common
import probe_constructive_code_seeded_20260922_v2 as probe


def test_v2_preserves_original_kernel_isolation():
    common.check_kernel_source_unchanged(probe.SOURCE,common.ORIGINAL)


def test_v2_restores_only_site_builtins_without_site_main():
    source=probe.SOURCE.read_text()
    assert 'site.setquit(); site.setcopyright(); site.sethelper()' in source
    assert '"site.main()' not in source
    assert source.index('sys.path[:]') < source.index('"import site') < source.index('random.seed(0')


def test_actual_pinned_runtime_and_exit_compatibility(tmp_path):
    raw=os.environ.get('SEEDED_SANDBOX_RUNTIME_ROOT')
    if raw is None:
        pytest.skip('set SEEDED_SANDBOX_RUNTIME_ROOT for pinned Python CPU controls')
    binary=tmp_path/'sandbox_v2'
    subprocess.run(['cc','-O2','-Wall','-Wextra','-Werror','-o',str(binary),str(probe.SOURCE)],check=True)
    sandbox=common.SandboxProbe(binary,Path(raw))
    common.run_controls(sandbox)
    controls=probe.run_compatibility_controls(sandbox)
    assert len(controls['standard_exit_semantics'])==9
