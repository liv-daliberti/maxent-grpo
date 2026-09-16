"""Fail-closed checks for RAM recovery; no scheduler or training calls."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import recover_e119_rm47_memory128_20260911 as repair


class RecoverySafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / 'train.out'
        self.log.write_text('post-learning start step=864\n')
        cp = str(self.root / 'run/debug_job31163710/checkpoints/step_00768')
        partial = str(self.root / 'run/debug_job31163710/checkpoints/step_00864')
        self.detail = {'checkpoint_step':768, 'checkpoint':cp, 'current_step':863,
            'saved_counter_validation':{'saved_counters':dict.fromkeys(('global_steps','global_step','prompt_batches_consumed_total'),768)},
            'rejected_checkpoints':{partial:['optimizer: unreadable ZIP directory (BadZipFile)']}}
        self.plan = {'resume_step':768, 'allow_repeat_updates':96,
            'item':{'identity':{'run_dir':str(self.root/'run')},'resources':{'StdOut':str(self.log)}},
            'checkpoint':copy.deepcopy(self.detail), 'checkpoint_files':{'model':{'bytes':1}},
            'partial_path':partial, 'resume_ops':str(self.root/'ops')}

    def checkpoint(self, *, quarantined=False, selection=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(repair.memory,'timing_and_checkpoint',return_value=self.detail))
            stack.enter_context(patch.object(repair,'checkpoint_files',return_value=self.plan['checkpoint_files']))
            stack.enter_context(patch.object(repair.b,'command',return_value=SimpleNamespace(stdout=selection or self.detail['checkpoint'])))
            stack.enter_context(patch.object(repair.recovery,'complete',return_value=False))
            return repair.checkpoint(self.plan,quarantined=quarantined)

    def test_explicit_768_with_partial_864_and_optimizer_restore_is_accepted(self):
        self.assertEqual(self.checkpoint()['checkpoint_step'],768)

    def test_newer_durable_checkpoint_aborts_stale_768_plan(self):
        self.detail['checkpoint_step'] = 864
        with self.assertRaisesRegex(RuntimeError,'Selected checkpoint changed'):
            self.checkpoint()

    def test_unlogged_optimizer_progress_beyond_rollback_bound_aborts(self):
        self.log.write_text('post-learning start step=865\n')
        with self.assertRaisesRegex(RuntimeError,'Unlogged optimizer progress'):
            self.checkpoint()

    def test_wrong_saved_counter_aborts(self):
        self.detail['saved_counter_validation']['saved_counters']['global_steps'] = 767
        with self.assertRaisesRegex(RuntimeError,'Saved counters disagree'):
            self.checkpoint()

    def test_frozen_runtime_must_select_same_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError,'Frozen runtime auto-resume'):
            self.checkpoint(selection='/different/checkpoint')

    def test_after_quarantine_partial_must_be_absent_from_discovery(self):
        with self.assertRaisesRegex(RuntimeError,'Partial checkpoint set changed'):
            self.checkpoint(quarantined=True)
        self.detail['rejected_checkpoints'] = {}
        self.assertEqual(self.checkpoint(quarantined=True)['checkpoint_step'],768)

    def test_model_checkpoint_requires_saved_replay_bank(self):
        directory = self.root / 'checkpoint'; directory.mkdir()
        path = directory / 'mp_rank_00_model_states.pt'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('archive/data.pkl',pickle.dumps({'global_step':768}))
        with self.assertRaisesRegex(RuntimeError,'Replay bank missing'):
            repair.checkpoint_files(directory)

    def test_quarantine_rejects_symlink_and_tracks_exact_inode(self):
        directory = self.root / 'partial'; directory.mkdir()
        for name in ('model.pt','optimizer.pt'):
            (directory/name).write_bytes(b'partial')
        manifest = repair.partial_manifest(directory)
        self.assertEqual(manifest['model.pt']['inode'],(directory/'model.pt').stat().st_ino)
        (directory/'model.pt').unlink()
        (directory/'model.pt').symlink_to(directory/'optimizer.pt')
        with self.assertRaisesRegex(RuntimeError,'Unexpected partial checkpoint contents'):
            repair.partial_manifest(directory)

    def test_load_rejects_expired_inherited_deadline(self):
        oldplan = self.root/'plan.json'; oldtx = self.root/'tx.json'
        oldplan.write_text('{}'); oldtx.write_text(json.dumps({'deadline_utc':'2000-01-01T00:00:00+00:00'}))
        with patch.object(repair.h,'load',return_value=({},{})), patch.object(repair,'OLD_PLAN',oldplan), patch.object(repair,'OLD_TX',oldtx):
            with self.assertRaisesRegex(RuntimeError,'deadline reached'):
                repair.load()


if __name__ == '__main__':
    unittest.main()
