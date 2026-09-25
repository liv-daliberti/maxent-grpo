"""Study-level invariants: no seed/task selection or hidden PCMD denominators."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

OPS = Path(__file__).resolve().parents[1]/"ops"
sys.path.insert(0, str(OPS))
import aggregate_real_domains_seeds_20260921 as reducer


def metric(counts, samples=32, qa=False):
    rows = [{"accepted": True, "canonical_key": key, "hard_violations": [], "receipt": {}} for key, count in counts.items() for _ in range(count)]
    rows += [{"accepted": False, "canonical_key": None, "hard_violations": [], "receipt": {}} for _ in range(samples-len(rows))]
    value = reducer.audit.metrics(rows, ["a", "b"] if qa else None)
    value["raw_distinct_at_8"] = sum(len({r["canonical_key"] for r in rows[i:i+8] if r["accepted"]}) for i in range(0, samples, 8))/(samples/8)
    return value


def report(seed, qa=False):
    training = {}
    for index, arm in enumerate(("maxrl", "remax")):
        training[arm] = {"seed": seed, "eval_seed": seed+1000, "job_id": str(seed*10+index), "identity_sha256": f"id-{seed}-{arm}", "checkpoint_sha256": f"seal-{seed}-{arm}", "directory": f"/fake/{seed}/{arm}", "initial_parameters_sha256": f"initial-{seed}", "protocol": {"updates": 32, "model_revision": "pinned", "train_ids": ["train"]}, "sources": {"trainer": "fixed"}, "data": {"manifest_sha256": "fixed"}, "objective": {"arm": arm}, "completed_updates": 32, "train_ids": ["train"]}
    endpoints = {}
    for arm in reducer.ARMS:
        endpoints[arm] = {"seed": 111, "protocol": {"samples_per_task": 32}, "sources": {"evaluator": "fixed"}, "data": {"manifest_sha256": "fixed"}, "prompts": [{"task_id": "x"}, {"task_id": "y"}], "metadata": {"x": {"split": "test"}, "y": {"split": "test"}}, "metrics": {"x": metric({"a": 32}, qa=qa), "y": metric({"a": 16, "b": 16}, qa=qa)}, "job_id": str(seed*100+reducer.ARMS.index(arm)), "receipt_sha256": f"receipt-{seed}-{arm}"}
    return {"seed": seed, "training": training, "endpoints": endpoints, "strata": {"reserved_test": ["x", "y"]}, "report": f"/fake/report-{seed}", "report_sha256": f"report-{seed}", "domain_readiness": "pass", "readiness_interpretation": "pilot_readiness_pass", "terminal_accounting": []}


class SeedAggregationTest(unittest.TestCase):
    def setUp(self):
        self.reports = [report(seed) for seed in (1, 2, 3)]

    def aggregate(self, reports=None, **kwargs):
        return reducer.aggregate_reports(reports or self.reports, "code", **kwargs)

    def test_three_paired_seeds_have_sample_sd_not_prompt_ci(self):
        for seed, item in enumerate(self.reports):
            item["endpoints"]["remax"]["metrics"]["x"] = metric({"a": 32-8*seed})
        result = self.aggregate()
        stats = result["strata"]["reserved_test"]["comparisons"]["remax_minus_maxrl"]["seed_summary"]["accuracy"]
        self.assertEqual(stats["values"], [0, -.125, -.25])
        self.assertEqual(stats["mean"], -.125)
        self.assertEqual(stats["sample_sd"], .125)
        self.assertEqual(stats["range"], [-.25, 0])
        self.assertNotIn("paired_prompt_bootstrap_95_percentile", stats)

    def test_single_seed_requires_explicit_flag(self):
        with self.assertRaisesRegex(ValueError, "exactly three"):
            self.aggregate(self.reports[:1])
        result = self.aggregate(self.reports[:1], allow_single_seed=True)
        self.assertEqual(result["kind"], "single_seed_descriptive_pilot")
        self.assertIsNone(result["strata"]["reserved_test"]["comparisons"]["remax_minus_maxrl"]["seed_summary"]["accuracy"]["sample_sd"])

    def test_two_seeds_never_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "exactly three"):
            self.aggregate(self.reports[:2], allow_single_seed=True)

    def test_duplicate_seed_rejected_even_different_jobs(self):
        self.reports[2]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate paired training seed"):
            self.aggregate()

    def test_duplicate_job_rejected(self):
        self.reports[2]["training"]["maxrl"]["job_id"] = self.reports[0]["training"]["maxrl"]["job_id"]
        with self.assertRaisesRegex(ValueError, "duplicate training scheduler job"):
            self.aggregate()

    def test_unequal_cohorts_rejected(self):
        self.reports[2]["strata"]["reserved_test"] = ["x"]
        with self.assertRaisesRegex(ValueError, "unequal fixed task cohorts"):
            self.aggregate()

    def test_training_protocol_drift_rejected(self):
        self.reports[2]["training"]["remax"]["protocol"]["updates"] = 64
        with self.assertRaisesRegex(ValueError, "training study mismatch"):
            self.aggregate()

    def test_endpoint_protocol_drift_rejected(self):
        self.reports[1]["endpoints"]["maxrl"]["protocol"]["samples_per_task"] = 64
        with self.assertRaisesRegex(ValueError, "endpoint study mismatch"):
            self.aggregate()

    def test_empty_all_six_intersection_is_undefined(self):
        self.reports[0]["endpoints"]["maxrl"]["metrics"]["x"] = metric({"a": 29})
        self.reports[1]["endpoints"]["remax"]["metrics"]["y"] = metric({"a": 29})
        result = self.aggregate()["strata"]["reserved_test"]["comparisons"]["remax_minus_maxrl"]
        self.assertEqual([item["pcmd_pair_common_eligible"]["denominator"] for item in result["per_seed"]], [1, 1, 2])
        sensitivity = result["pcmd_all_seed_common_sensitivity"]
        self.assertEqual(sensitivity["denominator"], 0)
        self.assertEqual(sensitivity["seed_summary"]["values"], [None, None, None])
        self.assertIsNone(sensitivity["seed_summary"]["mean"])

    def test_base_does_not_shrink_remax_vs_maxrl_intersection(self):
        for item in self.reports:
            item["endpoints"]["base"]["metrics"]["x"] = metric({})
        data = self.aggregate()["strata"]["reserved_test"]
        self.assertEqual(data["pcmd_all_trained_endpoint_intersection"], ["x", "y"])
        self.assertEqual(data["pcmd_base_included_intersection"], ["y"])
        self.assertEqual(data["comparisons"]["remax_minus_maxrl"]["pcmd_all_seed_common_sensitivity"]["denominator"], 2)
        self.assertEqual(data["comparisons"]["remax_minus_base"]["pcmd_all_seed_common_sensitivity"]["denominator"], 1)

    def test_no_available_seed_mean_when_one_pcmdenominator_empty(self):
        for task in ("x", "y"):
            self.reports[0]["endpoints"]["remax"]["metrics"][task] = metric({})
        stats = self.aggregate()["strata"]["reserved_test"]["comparisons"]["remax_minus_maxrl"]["pcmd_pair_common_eligible"]
        self.assertEqual(stats["defined_seeds"], 2)
        self.assertIsNone(stats["mean"])
        self.assertEqual(stats["status"], "undefined")

    def test_all_zero_outcomes_remain_valid(self):
        for item in self.reports:
            item["domain_readiness"] = "unknown"
            item["readiness_interpretation"] = "capability_not_established_but_complete_outcome_evidence"
            for arm in reducer.ARMS:
                for task in ("x", "y"):
                    item["endpoints"][arm]["metrics"][task] = metric({})
        result = self.aggregate()
        self.assertEqual(result["status"], "pass")
        comparison = result["strata"]["reserved_test"]["comparisons"]["remax_minus_maxrl"]
        self.assertEqual(comparison["seed_summary"]["accuracy"]["mean"], 0)
        self.assertIsNone(comparison["pcmd_pair_common_eligible"]["mean"])

    def test_qa_known_support_is_reported(self):
        reports = [report(seed, qa=True) for seed in (1, 2, 3)]
        reports[0]["endpoints"]["remax"]["metrics"]["x"] = metric({"a": 16, "b": 16}, qa=True)
        result = reducer.aggregate_reports(reports, "qa")
        self.assertEqual(result["strata"]["reserved_test"]["comparisons"]["remax_minus_maxrl"]["seed_summary"]["annotated_topic_coverage"]["values"], [.25, 0, 0])

    def test_default_study_rejects_development_as_test(self):
        for item in self.reports:
            item["strata"] = {"untrained_development": ["x", "y"]}
        with self.assertRaisesRegex(ValueError, "fixed reserved test"):
            self.aggregate()

    def test_auxiliary_training_identity_must_match(self):
        extras = copy.deepcopy(self.reports)
        for item in extras:
            item["strata"] = {"trained_train": ["train"]}
        extras[0]["training"]["remax"]["identity_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "different training identity"):
            self.aggregate(trained_reports=extras)

    def test_missing_token_or_source_binding_rejected(self):
        for omitted in ("token_text_binding", "frozen_implementation_binding"):
            value = {"status": "pass", "token_text_binding": "pass", "frozen_implementation_binding": "pass"}
            value[omitted] = "unknown"
            with self.assertRaises(ValueError):
                reducer.passed(value, "test")

    def test_protocol_forwards_verified_chain_without_rewriting_runtime_input(self):
        config = {"seed": 3, "eval_seed": 4, "adapter_config": {"runtime_root": "/actual/runtime"}}
        contexts = [Path("/proven/base"), Path("/proven/endpoint")]
        with patch.object(reducer.audit, "comparison_config", return_value=copy.deepcopy(config)) as compare:
            result = reducer.protocol(config, Path("/proven/endpoint"), chain_directories=contexts)
        compare.assert_called_once_with(config, Path("/proven/endpoint"), chain_directories=contexts)
        self.assertEqual(config["seed"], 3)
        self.assertEqual(config["adapter_config"]["runtime_root"], "/actual/runtime")
        self.assertNotIn("seed", result)
        self.assertNotIn("eval_seed", result)

    def test_dataset_identity_forwards_same_verified_chain(self):
        identity = {"manifest_sha256": "fixed", "records_path": "/nested/frozen/records"}
        contexts = [Path("/proven/base"), Path("/proven/endpoint")]
        normalized = {"manifest_sha256": "fixed", "records_path": "/original/records"}
        with patch.object(reducer.audit, "normalized_config", return_value=normalized) as normalize:
            result = reducer.data_identity(identity, Path("/proven/endpoint"), chain_directories=contexts)
        normalize.assert_called_once_with(identity, Path("/proven/endpoint"), chain_directories=contexts)
        self.assertEqual(result["manifest_sha256"], "fixed")
        self.assertEqual(result["records_path"], "/original/records")

    def test_sata_config_digest_uses_exact_adapter_serialization(self):
        config = {"records_path": "/fake/data", "splits": ["train", "test"]}
        value = {"schema_version": "sata_reuters_one_valid_topic_v1", "config": config, "config_sha256": reducer.audit.sha_bytes(json.dumps(config, sort_keys=True).encode()), "records_sha256": "retained"}
        result = reducer.data_identity(value, Path("/nonexistent"))
        self.assertNotIn("config_sha256", result)
        self.assertEqual(result["records_sha256"], "retained")
        value["config_sha256"] = "invalid"
        with self.assertRaisesRegex(ValueError, "dataset configuration digest"):
            reducer.data_identity(value, Path("/nonexistent"))

    def test_identity_file_must_still_match_audited_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = Path(directory)/"identity.json"
            identity.write_text('{"changed": true}')
            arm = {"status": "pass", "token_text_binding": "pass", "frozen_implementation_binding": "pass", "domain": "code", "directory": directory, "identity_sha256": "0"*64}
            with self.assertRaisesRegex(ValueError, "training identity SHA256 mismatch"):
                reducer.load_training(arm, "code")

    def test_incomplete_terminal_updates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"config": {"updates": 32, "adapter_module": "build_constructive_code_hardened_20260921"}}
            result = {"status": "complete", "completed_updates": 16}
            for name, value in (("identity", identity), ("result", result)):
                (root/(name+".json")).write_text(json.dumps(value))
            arm = {"status": "pass", "token_text_binding": "pass", "frozen_implementation_binding": "pass", "domain": "code", "directory": directory, "completed_updates": 16, "identity_sha256": reducer.audit.sha_file(root/"identity.json"), "result_sha256": reducer.audit.sha_file(root/"result.json")}
            with self.assertRaisesRegex(ValueError, "incomplete terminal updates"):
                reducer.load_training(arm, "code")

    def test_input_binding_is_report_not_last_endpoint_context(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixture = report(1)
            flags = {"status": "pass", "token_text_binding": "pass", "frozen_implementation_binding": "pass"}
            arms = [{**flags, "arm": arm, "directory": str(directory/arm/"training"), "identity_sha256": "unused"} for arm in ("maxrl", "remax")]
            endpoints = {}
            for label in reducer.ARMS:
                path = directory/(label+"-evaluation.json")
                value = {"config": {"lora_arm": label}, "lora_checkpoint": {"seal_sha256": fixture["training"].get(label, {}).get("checkpoint_sha256")}}
                path.write_text(json.dumps(value))
                fixture["endpoints"][label]["receipt"] = str(path)
                endpoints[label] = {**flags, "arm": label, "receipt": str(path), "receipt_sha256": reducer.audit.sha_file(path)}
            selected = {"readiness": "pass", "paired_run_integrity_established": True, "capability_established": True, "capability": [], "paired_training": {**flags, "domain": "code", "arms": arms}, "endpoint_comparison": {**flags, "endpoints": endpoints, "strata": {"reserved_test": {"task_ids": ["x", "y"]}}}}
            jobs = [{"job_id": item["job_id"], "state": "COMPLETED", "exit_code": "0:0", "elapsed_seconds": 1, "gpu_count": 1, "allocated_gpu_hours": 1/3600} for item in list(fixture["training"].values())+list(fixture["endpoints"].values())]
            report_path = directory/"summary.json"
            report_path.write_text(json.dumps({"schema": reducer.audit.SCHEMA, "domains": {"code": selected}, "scheduler_accounting": {"jobs": jobs}}))
            expected_sha = reducer.audit.sha_file(report_path)
            with patch.object(reducer, "bound_json", return_value={}), patch.object(reducer.audit, "audit_frozen_sources", return_value="pass"), patch.object(reducer, "load_training", side_effect=lambda arm, domain, **kwargs: fixture["training"][arm["arm"]]), patch.object(reducer, "load_endpoint", side_effect=lambda endpoint, domain, **kwargs: fixture["endpoints"][endpoint["arm"]]):
                loaded = reducer.load_report(report_path, "code")
            self.assertEqual(loaded["report"], str(report_path.resolve()))
            self.assertEqual(loaded["report_sha256"], expected_sha)
            self.assertNotEqual(loaded["report"], endpoints["remax"]["receipt"])
            self.assertNotEqual(loaded["report_sha256"], endpoints["remax"]["receipt_sha256"])

    def test_unknown_readiness_without_paired_integrity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"summary.json"
            path.write_text(json.dumps({"schema": reducer.audit.SCHEMA, "domains": {"qa": {"readiness": "unknown", "paired_run_integrity_established": False}}}))
            with self.assertRaisesRegex(ValueError, "paired integrity"):
                reducer.load_report(path, "qa")


if __name__ == "__main__":
    unittest.main()
