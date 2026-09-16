"""Safety boundaries for the positive checkpoint addendum; no scheduler calls."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops/exp_scaling'))
import reconcile_e119_rm47_checkpoint864_20260911 as addendum

class AddendumTests(unittest.TestCase):
    def test_amend_changes_only_six_checkpoint_fields_without_editing_original(self):
        original={'resume_step':768,'allow_repeat_updates':96,'checkpoint':{'checkpoint_step':768},
                  'checkpoint_files':{'old':1},'partial_path':'partial864','quarantine_path':'archive864',
                  'item':{'job_id':31163710,'resources':{'MinMemoryNode':'116G'}},'old_cpu_job_id':31194078}
        before=copy.deepcopy(original)
        result=addendum.amend(original,{'checkpoint':{'checkpoint_step':864},'checkpoint_files':{'new':1}})
        self.assertEqual(original,before)
        self.assertEqual(result['item'],original['item'])
        self.assertEqual(result['resume_step'],864)
        self.assertEqual(result['allow_repeat_updates'],0)
        self.assertIsNone(result['partial_path'])
        self.assertEqual({k for k in original if original[k]!=result[k]},addendum.CHANGED)

    def test_transaction_prefix_accepts_only_append_only_evolution(self):
        prefix={'hold_intent':True,'old_cpu_stop_requested':True,'new_cpu_job_id':31246560,'events':[{'event':'stopped'}]}
        evolved=copy.deepcopy(prefix);evolved.update(memory_intent=True,actual_resume_step=864)
        evolved['events'].append({'event':'RAM increased'})
        addendum.prefix_valid(evolved,prefix)
        evolved['new_cpu_job_id']=123
        with self.assertRaisesRegex(RuntimeError,'nonce/state prefix'):
            addendum.prefix_valid(evolved,prefix)

    def test_transaction_prefix_rejects_history_rewrite(self):
        prefix={'hold_intent':True,'events':[{'event':'stopped'}]}
        rewritten={'hold_intent':True,'events':[{'event':'other'}]}
        with self.assertRaisesRegex(RuntimeError,'history changed'):
            addendum.prefix_valid(rewritten,prefix)

if __name__=='__main__':unittest.main()
