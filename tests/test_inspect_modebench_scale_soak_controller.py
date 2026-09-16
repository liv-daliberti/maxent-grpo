"""Scratch process fixtures; never inspect or signal a real remote process."""
import importlib.util
import os
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'artifacts/inspect_modebench_scale_soak_controller_20260912.py'
spec = importlib.util.spec_from_file_location('soak_diagnostic', PATH)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def stat(pid, ticks, state='S'):
    fields = [state, '123'] + ['0'] * 17 + [str(ticks)]
    return str(pid) + ' (comm with ) spaces) ' + ' '.join(fields) + '\n'


def process(proc, pid, ticks, argv, state='S'):
    directory = proc / str(pid)
    directory.mkdir()
    (directory / 'stat').write_text(stat(pid, ticks, state))
    (directory / 'cmdline').write_bytes(b'\0'.join(value.encode() for value in argv) + b'\0')
    (directory / 'cwd').symlink_to(d.ROOT)
    return directory


def test_stat_comm_parentheses_and_tick_offset():
    assert d.process_stat(stat(d.PID, d.TICKS)) == {
        'pid': d.PID, 'state': 'S', 'parent_pid': 123, 'start_ticks': d.TICKS}


def test_script_invocation_distinguishes_code_strings_and_unrelated_paths():
    script = str(d.CONTROLLER)
    actual, warning = d.invoked_script(['/venv/bin/python', '-B', script, '--advance'], d.ROOT)
    assert actual['script'] == script and actual['advancing'] and warning is None
    assert d.invoked_script(['/venv/bin/python', '-c', script, script], d.ROOT) == (None, None)
    assert d.invoked_script(['/bin/bash', '-c', script], d.ROOT) == (None, None)
    assert d.invoked_script(['/venv/bin/python', '/tmp/' + d.CONTROLLER.name], d.ROOT) == (None, None)
    relative, warning = d.invoked_script(['/venv/bin/python', '-u', 'ops/exp_scaling/' + d.CONTROLLER.name], d.ROOT)
    assert relative['script'] == script and warning is None


@pytest.mark.parametrize('args', [['-m', d.CONTROLLER.stem], ['-m' + d.CONTROLLER.stem]])
def test_unresolved_module_is_not_false_absence(args):
    script, warning = d.invoked_script(['/venv/bin/python', *args], d.ROOT)
    assert script is None and warning


def test_snapshot_old_identity_and_scan_real_script_only(tmp_path):
    process(tmp_path, d.PID, d.TICKS, ['/venv/bin/python', '-B', str(d.CONTROLLER), '--advance', '--watch'])
    process(tmp_path, 77, '900', ['/venv/bin/python', '-c', str(d.CONTROLLER)])
    result = d.scan_processes(tmp_path, os.getuid())
    assert result['recorded_process']['status'] == 'recorded_process_present'
    assert result['recorded_process']['start_ticks'] == d.TICKS
    assert [p['pid'] for p in result['matching_controllers']] == [d.PID]
    assert result['scan_errors'] == []


def test_pid_reuse_is_distinct_from_recorded_process(tmp_path):
    process(tmp_path, d.PID, '999', ['/venv/bin/python', '-c', 'pass'])
    result = d.scan_processes(tmp_path, os.getuid())
    assert result['recorded_process']['status'] == 'pid_reused'
    assert result['matching_controllers'] == []


def test_absent_pid_and_zombie_are_reported_truthfully(tmp_path):
    absent = d.scan_processes(tmp_path, os.getuid())
    assert absent['recorded_process']['status'] == 'absent_at_inspection'
    process(tmp_path, d.PID, d.TICKS, [], state='Z')
    zombie = d.scan_processes(tmp_path, os.getuid())
    assert zombie['recorded_process']['state'] == 'Z'
    assert zombie['recorded_process']['status'] == 'recorded_process_present'
    assert zombie['matching_controllers'] == []


def test_missing_cwd_is_not_process_absence(tmp_path):
    directory = process(tmp_path, d.PID, d.TICKS, ['/venv/bin/python', '-B', str(d.CONTROLLER), '--advance'])
    (directory / 'cwd').unlink()
    result = d.scan_processes(tmp_path, os.getuid())
    assert result['recorded_process']['status'] == 'inspection_failed'
    assert result['scan_errors']
