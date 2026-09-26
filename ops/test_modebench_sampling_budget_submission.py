"""Mock the operational wrapper through sbatch and both durable receipts."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as S
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import prepare_modebench_sampling_budget_local as launcher


class SubmissionTest(unittest.TestCase):
    def test_wrapper_reaches_sbatch_and_records_readback(self):
        realbase = ROOT / 'artifacts/modebench_discovery_curves_20260911'
        commands = []
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            local = base / 'local'
            local.mkdir()
            for name in ('plan.json', 'worker.slurm'):
                shutil.copyfile(realbase / 'local' / name, local / name)
            shutil.copyfile(realbase / 'manifest.json', base / 'manifest.json')
            def fake_run(command, **kwargs):
                commands.append(command)
                if command[0] == 'sinfo':
                    return S(stdout='NODELIST STATE GRES\nnode203 idle gpu:a5000:10\n', stderr='', returncode=0)
                if command[0] == 'sbatch':
                    return S(stdout='99999999\n', stderr='', returncode=0)
                if command[0] == 'scontrol':
                    return S(stdout='JobId=99999999 ArrayTaskThrottle=8\n', stderr='', returncode=0)
                raise AssertionError(command)
            with patch.object(launcher, 'BASE', base), patch.object(launcher, 'validate_plan'), \
                 patch.object(launcher, 'verify_checkpoint', return_value=Path('/mock/model')), \
                 patch.object(launcher, 'subprocess') as process, patch.object(launcher, 'file_sha', launcher.file_sha), \
                 contextlib.redirect_stdout(io.StringIO()):
                process.run.side_effect = fake_run
                runpy.run_path(str(ROOT / 'ops/submit_modebench_sampling_budget_local_receipt_fix.py'), run_name='__main__')
            self.assertEqual([command[0] for command in commands], ['sinfo', 'sbatch', 'scontrol'])
            self.assertIn('--array=0-24%8', commands[1])
            intent = json.loads((local / 'submission_intent.json').read_text())
            result = json.loads((local / 'submission_result.json').read_text())
            readback = json.loads((local / 'submission_readback.json').read_text())
            self.assertEqual(intent['checkpoint_indices'], list(range(25)))
            self.assertEqual(result['job_id'], '99999999')
            self.assertEqual(readback['job_id'], '99999999')
            self.assertEqual(len(intent['launcher_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
