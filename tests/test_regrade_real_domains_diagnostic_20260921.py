import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import regrade_real_domains_diagnostic_20260921 as regrade


class DiagnosticRegradeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "original"
        self.original.mkdir()
        self.config = {"adapter_module": "old", "adapter_config": {}, "task_ids": ["task"], "samples_per_task": 2, "seed": 7, "max_tokens": 16}
        self.prompt = "A real prompt"
        prompt_sha = hashlib.sha256(self.prompt.encode()).hexdigest()
        self.raw_path = self.original / "responses.jsonl"
        rows = [{"task_id": "task", "sample_index": i, "request_seed": 7+i,
                 "prompt_sha256": prompt_sha, "text": text,
                 "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                 "token_ids": [65+i], "token_count": 1, "finish_reason": "stop"}
                for i, text in enumerate(["A", "B"])]
        self.raw_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.original_path = self.original / "evaluation.json"
        self.receipt = {"status": "complete", "config": self.config, "runner_sha256": "1"*64,
                        "config_sha256": "2"*64, "job_id": "123",
                        "task_prompts": [{"task_id": "task", "prompt": self.prompt, "prompt_sha256": prompt_sha}],
                        "artifacts": {"responses": {"path": str(self.raw_path), "sha256": regrade.sha256(self.raw_path)}}}
        self.original_path.write_text(json.dumps(self.receipt))
        self.new_config = self.root / "new_config.json"
        self.new_config.write_text(json.dumps({**self.config, "adapter_module": "hardened", "adapter_config": {"version": 2}}))
        def verify(text):
            return {"accepted": text == "B", "canonical_key": "new:B" if text == "B" else None, "hard_violations": [], "receipt": {"new_verifier": True}}
        self.task = SimpleNamespace(task_id="task", prompt=self.prompt, family="test", split="development", metadata={}, verify=verify)
        self.loader = patch.object(regrade, "load_task_selection", return_value=(SimpleNamespace(__file__=__file__), [self.task], {"manifest_sha256": "3"*64}))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def invoke(self):
        return regrade.run(self.original, self.new_config, self.root / "diagnostic.json")

    def test_regrades_all_original_samples_without_rewriting_them(self):
        old_bytes = self.raw_path.read_bytes()
        old_receipt = self.original_path.read_bytes()
        result = self.invoke()
        self.assertEqual(result["summary"]["samples"], 2)
        self.assertEqual(result["task_results"][0]["mode_counts"], {"new:B": 1})
        self.assertTrue(result["cpu_only"])
        self.assertFalse(result["primary_endpoint"])
        self.assertNotIn("job_id", result)
        self.assertEqual(self.raw_path.read_bytes(), old_bytes)
        self.assertEqual(self.original_path.read_bytes(), old_receipt)

    def test_discloses_prior_verifier_failure_without_assuming_it_passed(self):
        self.receipt["status"] = "audit_fail"
        self.receipt["summary"] = {"hard_violation_count": 3}
        self.original_path.write_text(json.dumps(self.receipt))
        result = self.invoke()
        self.assertEqual(result["generation_provenance"]["original_evaluation_status"], "audit_fail")
        self.assertEqual(result["generation_provenance"]["original_hard_violation_count"], 3)
        self.assertFalse(result["primary_endpoint"])

    def test_rejects_nonterminal_original_receipt(self):
        self.receipt["status"] = "generating"
        self.original_path.write_text(json.dumps(self.receipt))
        with self.assertRaisesRegex(ValueError, "terminal receipt"):
            self.invoke()

    def test_rejects_raw_artifact_tampering(self):
        self.raw_path.write_text(self.raw_path.read_text().replace('"A"', '"C"'))
        with self.assertRaisesRegex(ValueError, "generated samples changed"):
            self.invoke()

    def test_rejects_sampling_change(self):
        cfg = json.loads(self.new_config.read_text())
        cfg["seed"] += 1
        self.new_config.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "generation setting changed"):
            self.invoke()

    def test_rejects_prompt_change(self):
        self.task.prompt = "A different prompt"
        with self.assertRaisesRegex(ValueError, "changed a model prompt"):
            self.invoke()

    def test_rejects_duplicate_requests_even_if_raw_file_is_sealed(self):
        line = self.raw_path.read_text().splitlines()[0]
        self.raw_path.write_text(line + "\n" + line + "\n")
        self.receipt["artifacts"]["responses"]["sha256"] = regrade.sha256(self.raw_path)
        self.original_path.write_text(json.dumps(self.receipt))
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
