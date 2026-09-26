"""Safety boundaries for reviewed same-ID E119 memory repair decisions."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "ops/exp_scaling/recover_e119_memory_pressure_20260905.py"
SPEC = importlib.util.spec_from_file_location("e119_memory_pressure_recovery", PATH)
repair = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(repair)


class RecoverySafetyTests(unittest.TestCase):
    def test_reclaimable_file_cache_does_not_qualify(self):
        gib = 2**30
        memory = {"memory.current": 44*gib, "memory.high": 40*gib,
                  "stat.anon": 25*gib, "stat.shmem": 9*gib, "events.high": 10000}
        self.assertFalse(repair.pressure(memory))
        memory["stat.anon"] = 35*gib
        self.assertTrue(repair.pressure(memory))

    def test_truncated_probe_cannot_qualify(self):
        self.assertFalse(repair.pressure({"memory.current": 44*2**30, "memory.high": 40*2**30}))

    def test_existing_holds_and_dependencies_are_preserved(self):
        self.assertTrue(repair.ordinary_pending("JobState=PENDING Priority=1 Dependency=(null) Reason=Resources"))
        for record in ("JobState=PENDING Priority=0 Dependency=(null) Reason=job_requeued_in_held_state",
                       "JobState=PENDING Priority=0 Dependency=(null) Reason=JobHeldAdmin",
                       "JobState=PENDING Priority=1 Dependency=afterok:123 Reason=Dependency",
                       "JobState=RUNNING Priority=1 Dependency=(null) Reason=None"):
            self.assertFalse(repair.ordinary_pending(record))

    def test_only_owned_hold_reasons_can_be_released(self):
        for reason in ("JobHeldUser", "job_requeued_in_held_state"):
            repair.assert_own_hold(f"JobState=PENDING Priority=0 Reason={reason}")
        with self.assertRaises(AssertionError):
            repair.assert_own_hold("JobState=PENDING Priority=0 Reason=JobHeldAdmin")

    def test_near_checkpoint_guards_first_save_and_inflight_boundary(self):
        self.assertTrue(repair.near_checkpoint({"current_step":90, "checkpoint_step":None},96))
        self.assertTrue(repair.near_checkpoint({"current_step":96, "checkpoint_step":None},96))
        self.assertFalse(repair.near_checkpoint({"current_step":96, "checkpoint_step":96},96))
        self.assertFalse(repair.near_checkpoint({"current_step":0, "checkpoint_step":None},96))
        self.assertTrue(repair.near_checkpoint({"current_step":1144, "checkpoint_step":960},192))
        self.assertFalse(repair.near_checkpoint({"current_step":1152, "checkpoint_step":1152},192))

    def test_pantry_uses_effective_96_step_durability(self):
        with patch.object(repair, "field", return_value="/nonexistent/no-log"):
            self.assertEqual(repair.checkpoint_cadence("", {"domain":"pantry_plan"}),96)

    def test_pantry_does_not_widen_previously_allowed_pool(self):
        record="ReqNodeList=node[203,205] Partition=cs Account=allcs"
        with patch.object(repair, "call", return_value="node203\nnode205"):
            self.assertEqual(repair.proposed_nodes(record,{"domain":"pantry_plan"}),"node205")
        with patch.object(repair, "call", return_value="node203"):
            with self.assertRaises(AssertionError):
                repair.proposed_nodes(record,{"domain":"pantry_plan"})

    def test_existing_a100_route_is_retained(self):
        record="ReqNodeList=node302 Partition=mltheory Account=mltheory"
        def answer(*args):
            return "node302" if "hostnames" in args else "NodeName=node302 Gres=gpu:a100:8"
        with patch.object(repair,"call",side_effect=answer):
            self.assertEqual(repair.proposed_nodes(record,{"domain":"pantry_plan"}),"node302")

    def test_complete_submitline_is_compared(self):
        a="JobId=1 SubmitLine=sbatch --export=ALL,SCIENCE=one script.slurm WorkDir=/tmp"
        b="JobId=1 SubmitLine=sbatch --export=ALL,SCIENCE=two script.slurm WorkDir=/tmp"
        self.assertNotEqual(repair.submitline(a),repair.submitline(b))


if __name__ == "__main__":
    unittest.main()
