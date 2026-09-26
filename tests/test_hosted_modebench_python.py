"""End-to-end local frozen-worker audit: repair grades without changing evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from test_hosted_modebench_completion import fixture

ROOT = Path(__file__).resolve().parents[1]


def test_serial_python_audit_repairs_only_derived_grades(tmp_path):
    _, samples, _ = fixture(tmp_path, domain='python_factors')
    reference = ROOT / 'artifacts/frontier_modebench_gpt56sol_20260911'
    original_manifest = json.loads((reference / 'manifest.json').read_text())
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    for name, digest in original_manifest['code_sha256'].items():
        source = reference / 'code' / name
        target = tmp_path / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest['code_sha256'][name] = digest
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    preserved = {str(p.relative_to(tmp_path)): p.read_bytes()
                 for folder in ('raw_responses', 'sample_receipts') for p in (tmp_path / folder).glob('*.json')}
    preserved['samples.jsonl'] = (tmp_path / 'samples.jsonl').read_bytes()
    # Explicitly declare this small condition; the CLI still defaults to 15,360.
    script = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'ops'))
import audit_hosted_modebench_python as module
sys.argv = ['audit_hosted_modebench_python.py', '--output-root', sys.argv[2], '--expected-samples', '2']
module.main()
'''
    result = subprocess.run([sys.executable, '-c', script, str(ROOT), str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    audit = json.loads((tmp_path / 'primary_python_regrade_audit.json').read_text())
    assert audit['status'] == 'complete_for_snapshot'
    assert audit['correction_counts'] == {'false_to_true': 2, 'true_to_false': 0, 'verified_key_changes': 0}
    assert audit['unresolved_groups'] == 0
    assert audit['unique_python_row_text_groups'] == 1
    assert len(audit['discrepancies'][0]['confirmations']) == 2
    derived = [json.loads(line) for line in (tmp_path / 'audited_primary_samples.jsonl').read_text().splitlines()]
    assert all(record['verified'] and record['canonical_key'] == 'python_factor:2,2' for record in derived)
    for name, original in preserved.items():
        assert (tmp_path / name).read_bytes() == original, name
