"""Verifier and frozen-data checks; no model or GPU required."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from oat_drgrpo import noncoding_multi_answer_sata as adapter


ROOT = Path("var/artifacts/noncoding_multi_answer_sata_20260921")


class SATAAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "noncoding_multi_answer_sata_adapter_config.json").read_text())
        cls.tasks = adapter.load_tasks(cls.config)
        cls.raw = json.loads((ROOT / "raw/hf_data_main.json").read_text())

    def test_all_source_labels_and_modes_are_preserved(self):
        for task in self.tasks:
            raw = self.raw[task.record["source_row_index"]]
            self.assertEqual(raw["paragraph"], task.record["paragraph"])
            self.assertEqual(raw["question"], task.record["source_question"])
            keys = set()
            for option in task.record["options"]:
                result = task.verify(option["display"])
                self.assertEqual(result["accepted"], option["text"] in raw["answer groups"])
                self.assertEqual(result["hard_violations"], [])
                if result["accepted"]:
                    self.assertEqual(result["canonical_key"], "reuters_topic:" + option["text"])
                    keys.add(result["canonical_key"])
                else:
                    self.assertIsNone(result["canonical_key"])
            self.assertEqual(len(keys), task.metadata["known_mode_count"])
            self.assertGreaterEqual(len(keys), 2)

    def test_strict_single_letter_and_whitespace(self):
        task = self.tasks[0]
        letter = next(o["display"] for o in task.record["options"] if o["label"])
        self.assertTrue(task.verify(" \n" + letter + "\t")["accepted"])
        for invalid in (letter.lower(), letter + ".", "Answer: " + letter, letter + " B", "[" + letter + "]", "Z", "", "Α", None):
            result = task.verify(invalid)
            self.assertFalse(result["accepted"])
            self.assertIsNone(result["canonical_key"])
            self.assertEqual(result["hard_violations"], [])
            self.assertTrue(result["receipt"]["rejection_reasons"])

    def test_permutation_changes_letters_not_semantic_identity(self):
        task = self.tasks[0]
        changed = deepcopy(task.record)
        changed["options"].reverse()
        for i, o in enumerate(changed["options"]):
            o["display"] = chr(65+i)
        adapter.validate_record(changed)
        other = adapter.Task(changed)
        for o in task.record["options"]:
            o2 = next(p for p in changed["options"] if p["id"] == o["id"])
            self.assertEqual(task.verify(o["display"])["canonical_key"], other.verify(o2["display"])["canonical_key"])

    def test_gold_and_hash_corruption_fail_closed(self):
        r = deepcopy(self.tasks[0].record)
        r["options"][0]["label"] = 1-r["options"][0]["label"]
        with self.assertRaises(ValueError):
            adapter.validate_record(r)
        bad = {**self.config, "records_sha256": "0"*64}
        with self.assertRaises(ValueError):
            adapter.load_tasks(bad)

    def test_test_partition_is_reserved(self):
        self.assertEqual(len(self.tasks), 160)
        self.assertEqual(sum(t.split == "train" for t in self.tasks), 128)
        self.assertEqual(sum(t.split == "dev" for t in self.tasks), 32)
        with self.assertRaises(ValueError):
            adapter.load_tasks({**self.config, "splits": ["test"]})
        # Audit metadata only, not predictions or text inspection on held-out tasks.
        records = [json.loads(line) for line in Path(self.config["records_path"]).read_text().splitlines()]
        self.assertEqual(sum(r["split"] == "test" for r in records), 83)
        groups = {}
        for r in records:
            groups.setdefault(r["document_group"], set()).add(r["split"])
        self.assertTrue(all(len(s) == 1 for s in groups.values()))

    def test_identity_and_prompt_are_fixed(self):
        identity = adapter.dataset_identity(self.config)
        self.assertEqual(identity["records_sha256"], self.config["records_sha256"])
        self.assertEqual(identity["source"]["revision"], "ba43a7ab537adfa3498e3a160a6d1eafbefc95c1")
        self.assertEqual(self.tasks[0].prompt, adapter.load_tasks(self.config)[0].prompt)
        self.assertTrue(self.tasks[0].prompt.endswith("<|im_start|>assistant\n"))


if __name__ == "__main__":
    unittest.main()
