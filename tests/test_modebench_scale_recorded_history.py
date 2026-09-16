"""Historical provenance must preserve literal failures and independent joins."""
from copy import deepcopy
import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path('/n/fs/similarity/maxent-grpo')
SOURCE = ROOT/'artifacts/verify_modebench_scale_recorded_history_20260912.py'
spec = importlib.util.spec_from_file_location('_test_recorded_history', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def cell(index):
    base = m.v4.COMPOSITE/'revision_development'
    proof = m.read(base/'execution_reconciliations'/str(index)/'reconciliation.json')
    terminal = m.read(base/'execution_reconciliations'/str(index)/'terminal_accounting.json')
    row = m.read(m.CS/'execution_reconciliation.json')['original_terminal_rows_by_index'][str(index)]
    command = m.read(base/'plan.json')['cells'][index]['command']
    return row, proof, command, terminal


@pytest.mark.parametrize('index', range(5))
def test_actual_completed_cell_agrees_across_independent_records(index):
    m.completed_cell(index, *cell(index))


@pytest.mark.parametrize('key,value', [
    ('scheduler_success', True), ('exit_cause', 'known'), ('completed_receipts', 3),
    ('scientific_outputs_complete', False), ('exit_code', '0:0'),
    ('stage_plan_sha256', '0'*64), ('job_id_raw', '31254520'),
    ('end_utc', '2026-09-12T00:00:00'),
])
def test_reject_tampered_proof_even_with_complete_saved_receipts(key, value):
    row, proof, command, terminal = cell(2)
    proof[key] = value
    with pytest.raises(ValueError):
        m.completed_cell(2, row, proof, command, terminal)


def test_reject_changed_evaluator_arguments():
    row, proof, command, terminal = cell(0)
    with pytest.raises(ValueError):
        m.completed_cell(0, row, proof, command+['--new-seed', '1'], terminal)


@pytest.mark.parametrize('kind', ['state', 'raw_id', 'duplicate', 'malformed', 'other_cell'])
def test_reject_disagreeing_terminal_record(kind):
    row, proof, command, terminal = cell(2)
    changed = deepcopy(row)
    if kind == 'state': changed[2:4] = ['COMPLETED', '0:0']
    elif kind == 'raw_id': changed[0] = '1'
    elif kind == 'other_cell': changed[1] = '31254520_3'
    elif kind == 'malformed': changed = changed[:-1]
    terminal['stdout'] = '|'.join(changed)+'\n'
    if kind == 'duplicate': terminal['stdout'] *= 2
    with pytest.raises(ValueError):
        m.completed_cell(2, row, proof, command, terminal)


@pytest.mark.parametrize('index', [-1, 5, 6, True])
def test_partial_and_unstarted_cells_cannot_use_complete_cell_path(index):
    with pytest.raises(ValueError):
        m.completed_cell(index, *cell(0))


def test_actual_audit_failure_and_later_readonly_success_stay_distinct():
    summary, pins = m.cs_audit_outer(m.read(m.CS/'execution_reconciliation.json'))
    assert summary['audit_wrapper_returncode'] == 143
    assert summary['audit_wrapper_exit_cause'] == 'unknown'
    assert summary['later_readonly_verification_returncode'] == 0
    assert summary['new_grader_invocations'] == 0
    assert len(pins) >= 8


@pytest.mark.parametrize('relative,key,value', [
    ('actual_audit_v2_execution/audit.exit.json', 'returncode', 0),
    ('actual_audit_v2_execution/audit.exit.json', 'log_sha256', '0'*64),
    ('actual_audit_v2_execution/failure.json', 'no_automatic_regrade', False),
    ('actual_audit_v2_readonly_verification/exit.json', 'returncode', 143),
    ('actual_audit_v2_readonly_verification/exit.json', 'new_grader_invocations', 54400),
    ('actual_audit_v2_readonly_verification/intent.json', 'source_sha256', '0'*64),
    ('actual_audit_v2_readonly_verification/intent.json', 'readonly', False),
    ('actual_audit_v2_readonly_verification/intent.json', 'at_utc', '2026-09-12T00:00:00+00:00'),
])
def test_reject_relabelled_audit_wrapper_or_missing_readonly_evidence(monkeypatch, relative, key, value):
    original = m.read
    def altered(path):
        result = original(path)
        if Path(path) == m.CS/relative: result[key] = value
        return result
    monkeypatch.setattr(m, 'read', altered)
    with pytest.raises(ValueError):
        m.cs_audit_outer(original(m.CS/'execution_reconciliation.json'))


def test_component_has_no_model_grader_scheduler_or_publication_entrypoint():
    tree = ast.parse(SOURCE.read_text())
    forbidden = {'audit_source', 'confirm_domain', 'receipt_scores', 'fit_domain',
                 'fit_revision', 'freeze_domain', '_sweep', 'publish_level', 'atomic_new',
                 'write_text', 'write_bytes', 'unlink', 'mkdir', 'Popen', 'run', 'system'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, 'attr', getattr(node.func, 'id', None))
            assert name not in forbidden
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and isinstance(n.func.value, ast.Name) and n.func.value.id == 'v4'
                   and n.func.attr in {'verify', 'requirements', 'validate_development', 'validate_confirmation'}
                   for n in ast.walk(tree))


@pytest.fixture
def original_join(monkeypatch):
    from types import SimpleNamespace
    plan = m.read(m.v4.COMPOSITE/'revision_development/plan.json')
    proof = m.read(m.CS/'execution_reconciliation.json')
    transport = SimpleNamespace(verify_transport=lambda launcher, path: (plan, {
        'authority': 'actual-test-authority', 'inputs_sha256': {}}))
    helpers = {key: SimpleNamespace(verify_reconciliation=lambda stage, index: cell(index)[1])
               for key in (1, 2)}
    monkeypatch.setattr(m.v4, 'submission', lambda *args: (31254520, None, None, None))
    monkeypatch.setattr(m, 'add_pins', lambda target, values: target.update(values))
    monkeypatch.setattr(m, 'pin_files', lambda *args: None)
    return transport, proof, helpers


def test_original_seven_cell_join_preserves_five_failed_and_two_recovered(original_join):
    transport, proof, helpers = original_join
    value, _ = m.original_development(None, transport, 'actual-test-authority', proof, helpers)
    assert value['cells'] == 7
    assert [v['exit_code'] for v in value['individually_reconciled_cells']] == ['1:0', '1:0', '143:0', '1:0', '1:0']
    assert value['recovered_indices'] == [5, 6]
    assert value['original_runtime6_exists'] is False


@pytest.mark.parametrize('change', ['missing', 'extra', 'wrong_human', 'short_row',
    'node_failure_success', 'cancel_started', 'cancel_allocated', 'cancel_node', 'cancel_tres', 'runtime6'])
def test_original_join_rejects_omitted_or_rewritten_failed_history(original_join, monkeypatch, change):
    transport, proof, helpers = original_join
    rows = proof['original_terminal_rows_by_index']
    if change == 'missing': del rows['6']
    elif change == 'extra': rows['7'] = rows['6'][:]
    elif change == 'wrong_human': rows['4'][1] = '31254520_3'
    elif change == 'short_row': rows['6'].pop()
    elif change == 'node_failure_success': rows['5'][2:4] = ['COMPLETED', '0:0']
    elif change == 'cancel_started': rows['6'][4] = '2026-09-12T12:00:00'
    elif change == 'cancel_allocated': rows['6'][7] = '6'
    elif change == 'cancel_node': rows['6'][6] = 'node202'
    elif change == 'cancel_tres': rows['6'][9] = 'cpu=6'
    else:
        original = Path.exists
        target = m.v4.COMPOSITE/'revision_development/runtime/6.json'
        monkeypatch.setattr(Path, 'exists', lambda path: True if path == target else original(path))
    with pytest.raises(ValueError):
        m.original_development(None, transport, 'actual-test-authority', proof, helpers)


def test_actual_graph_audit_zero_does_not_relabel_failed_scheduler():
    proof = m.read(m.GRAPH/'execution_reconciliation.json')
    summary, pins = m.graph_audit_outer(proof)
    assert proof['scheduler_success'] is False
    assert summary['audit_wrapper_returncode'] == summary['later_readonly_verification_returncode'] == 0
    assert summary['new_grader_invocations'] == 0
    assert len(pins) >= 8


@pytest.mark.parametrize('name,key,value', [
    ('audit.exit.json', 'returncode', 143),
    ('verify.exit.json', 'returncode', 1),
    ('audit.intent.json', 'source_sha256', '0'*64),
    ('verify.intent.json', 'audit_registers_once_before_its_grader_claim', True),
    ('result.json', 'certificate_sha256', '0'*64),
    ('result.json', 'new_grader_invocations', 0),
    ('runtime.json', 'host', 'node202.cs.princeton.edu'),
])
def test_graph_rejects_wrong_actual_process_linkage(monkeypatch, name, key, value):
    original = m.read
    def altered(path):
        result = original(path)
        if Path(path) == m.GRAPH/'actual_audit_execution'/name: result[key] = value
        return result
    monkeypatch.setattr(m, 'read', altered)
    with pytest.raises(ValueError):
        m.graph_audit_outer(original(m.GRAPH/'execution_reconciliation.json'))
