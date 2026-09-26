#!/usr/bin/env python3
"""Aggregate independently audited paired seeds, preserving every fixed prompt.

This CPU-only analysis consumes completed audit reports, rebinds their artifacts,
and recomputes endpoint metrics. It never selects seeds, tasks or checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import summarize_real_domains_pilot_20260921 as audit

SCHEMA = "real-domains-paired-seed-aggregation-20260921-v1"
ARMS = ("base", "maxrl", "remax")
COMPARISONS = {"remax_minus_maxrl": ("maxrl", "remax"),
               "maxrl_minus_base": ("base", "maxrl"),
               "remax_minus_base": ("base", "remax")}
SOURCE_KEYS = ("runner_sha256", "adapter_module_sha256", "model_config_sha256",
               "production_source_sha256", "tasks")
LORA_FIELDS = ("lora_path", "lora_arm", "lora_completed_updates",
               "lora_checkpoint_config_sha256", "max_lora_rank")


def require(condition, message):
    audit.require(condition, message)


def bound_json(path, digest, label):
    path = Path(path)
    require(audit.sha_file(path) == digest, label + " SHA256 mismatch")
    return audit.read_json(path)


def passed(value, label, source=True):
    require(value.get("status") == "pass", label + " audit did not pass")
    require(value.get("token_text_binding") == "pass", label + " exact token/text binding absent")
    if source:
        require(value.get("frozen_implementation_binding") == "pass", label + " frozen source binding absent")


def protocol(config, directory, endpoint=False, *, chain_directories=()):
    # The auditor reverses only launcher-proven frozen paths and independently
    # hashes coding runtime/checker bytes before discounting working locations.
    result = audit.comparison_config(config, Path(directory), chain_directories=chain_directories)
    result.pop("seed", None)
    result.pop("eval_seed", None)
    if endpoint:
        for name in LORA_FIELDS:
            result.pop(name, None)
    return result


def data_identity(identity, directory, *, chain_directories=()):
    result = audit.normalized_config(identity, Path(directory), chain_directories=chain_directories)
    # This digest changes with the frozen path, whose content and source mapping
    # were independently bound; all other dataset digests remain fixed.
    if "config" in identity and "config_sha256" in identity:
        require(identity.get("schema_version") == "sata_reuters_one_valid_topic_v1" and audit.sha_bytes(json.dumps(identity["config"], sort_keys=True).encode()) == identity["config_sha256"], "dataset configuration digest mismatch")
        result.pop("config_sha256")
    return result


def load_training(arm, domain, *, chain_directories=()):
    passed(arm, "training arm")
    require(arm.get("domain", domain) == domain, "training domain mismatch")
    directory = Path(arm["directory"])
    identity = bound_json(directory/"identity.json", arm["identity_sha256"], "training identity")
    result = bound_json(directory/"result.json", arm["result_sha256"], "training result")
    config = identity["config"]
    require(config["adapter_module"] == ("oat_drgrpo.noncoding_multi_answer_sata" if domain == "qa" else "build_constructive_code_hardened_20260921"), "training adapter/domain mismatch")
    require(result.get("status") == "complete", "training is not terminal complete")
    require(type(config["updates"]) is int and config["updates"] > 0, "invalid terminal update target")
    require(result["completed_updates"] == arm["completed_updates"] == config["updates"], "incomplete terminal updates")
    require(identity["arm"] == result["arm"] == arm["arm"], "training arm identity mismatch")
    require(identity["config_sha256"] == result["config_sha256"] == audit.object_sha(config), "training configuration digest mismatch")
    require(config["train_ids"] == arm["train_ids"], "training prompt cohort mismatch")
    require(type(config["seed"]) is int and type(config["eval_seed"]) is int, "invalid training seed")
    require(arm.get("job_id") is not None, "training scheduler job missing")
    require(audit.audit_frozen_sources(directory, identity, training=True) == "pass", "training source bytes no longer bound")
    checkpoint = directory/f"checkpoint-{config['updates']}"/"complete.json"
    seal = bound_json(checkpoint, arm["final_checkpoint_seal_sha256"], "terminal checkpoint seal")
    require(seal["completed_updates"] == config["updates"] and seal["config_sha256"] == identity["config_sha256"], "terminal checkpoint target mismatch")
    return {
        "seed": config["seed"], "eval_seed": config["eval_seed"],
        "job_id": str(arm["job_id"]), "identity_sha256": arm["identity_sha256"],
        "checkpoint_sha256": arm["final_checkpoint_seal_sha256"],
        "initial_parameters_sha256": identity["initial_trainable_parameters_sha256"],
        "protocol": protocol(config, directory, chain_directories=chain_directories),
        "sources": {key: identity[key] for key in SOURCE_KEYS},
        "data": data_identity(identity["dataset_identity"], directory, chain_directories=chain_directories),
        "objective": identity["objective"], "completed_updates": config["updates"],
        "train_ids": config["train_ids"], "directory": str(directory),
    }


def load_endpoint(endpoint, domain, *, chain_directories=()):
    passed(endpoint, "endpoint")
    require(endpoint["domain"] == domain, "endpoint domain mismatch")
    path = Path(endpoint["receipt"])
    receipt = bound_json(path, endpoint["receipt_sha256"], "endpoint receipt")
    require(receipt.get("status") == "complete", "endpoint is not terminal complete")
    require(audit.audit_frozen_sources(path.parent, receipt) == "pass", "endpoint source bytes no longer bound")
    config = receipt["config"]
    require(config["adapter_module"] == ("oat_drgrpo.noncoding_multi_answer_sata" if domain == "qa" else "build_constructive_code_hardened_20260921"), "endpoint adapter/domain mismatch")
    ids, n = config["task_ids"], config["samples_per_task"]
    require(len(ids) == len(set(ids)) and n >= 32 and n % 8 == 0, "endpoint requires unique tasks and complete groups of eight, at least 32 samples")
    require(set(ids) == set(endpoint["task_metrics"]) == set(endpoint["task_metadata"]), "endpoint metric cohort differs from receipt")
    require(receipt["dataset_identity_sha256"] == audit.object_sha(receipt["dataset_identity"]), "endpoint dataset digest mismatch")
    # Bind raw generations as well as decisions. Their exact token decoding was
    # already checked by the input independent audit; no tokenizer is needed here.
    for descriptor in receipt["artifacts"].values():
        audit.artifact_path(descriptor, path.parent)
    attempts = audit.read_rows(audit.artifact_path(receipt["artifacts"]["attempts"], path.parent))
    indexed = audit.unique(attempts, lambda row: (row["task_id"], row["sample_index"]))
    require(set(indexed) == {(task, i) for task in ids for i in range(n)}, "endpoint raw request denominator mismatch")
    gold = audit.qa_records(config)
    reported = audit.unique(receipt["task_results"], lambda row: row["task_id"])
    require(set(reported) == set(ids), "receipt metric cohort mismatch")
    calculated = {}
    for task in ids:
        rows = [indexed[(task, i)] for i in range(n)]
        m = audit.metrics(rows, gold[task]["gold_topic_ids"] if gold is not None else None)
        for key, value in m.items():
            require(key in endpoint["task_metrics"][task] and audit.close(value, endpoint["task_metrics"][task][key]), "audited endpoint metric disagrees with rebound decisions: " + task + "/" + key)
        for key in ("samples", "accepted", "accuracy", "mode_counts", "pcmd", "pcmd_eligible", "pass_at_k", "expected_distinct_valid_modes_at_k"):
            require(audit.close(m[key], reported[task][key]), "receipt metric disagrees with rebound decisions")
        m["raw_distinct_at_8"] = statistics.mean(len({r["canonical_key"] for r in rows[start:start+8] if r["accepted"]}) for start in range(0, n, 8))
        calculated[task] = m
    prompts = {row["task_id"]: row for row in receipt["task_prompts"]}
    require(set(prompts) == set(ids), "endpoint prompt cohort mismatch")
    require(all(endpoint["task_metadata"][task]["split"] == prompts[task]["split"] for task in ids), "endpoint split metadata differs from receipt")
    return {
        "seed": config["seed"], "protocol": protocol(config, path.parent, endpoint=True, chain_directories=chain_directories),
        "sources": {key: receipt[key] for key in ("runner_sha256", "adapter_sha256", "model_config_sha256", "sampling_vocab_upper_bound", "sampling_action_space")},
        "data": data_identity(receipt["dataset_identity"], path.parent, chain_directories=chain_directories),
        "prompts": receipt["task_prompts"], "metrics": calculated,
        "metadata": endpoint["task_metadata"], "receipt_sha256": endpoint["receipt_sha256"],
        "job_id": str(endpoint["job_id"]), "receipt": str(path),
    }


def load_report(path, domain):
    report_path = Path(path)
    report_bytes = report_path.read_bytes()
    report_sha256 = audit.sha_bytes(report_bytes)
    root = json.loads(report_bytes)
    require(root.get("schema") == audit.SCHEMA, "unsupported independent report schema")
    selected = root["domains"][domain]
    require(selected["readiness"] in {"pass", "unknown"}, "domain readiness failed")
    require(selected.get("paired_run_integrity_established") is True, "domain paired integrity is not established")
    require(not any(item.get("status") == "fail" for item in selected.get("capability", [])), "failed domain capability assertion")
    require(selected["readiness"] == "pass" or selected.get("capability_established") is False, "unknown domain readiness is not explained by missing/nonpositive capability")
    pair, comparison = selected["paired_training"], selected["endpoint_comparison"]
    passed(pair, "paired training")
    passed(comparison, "endpoint comparison", source=False)
    require(pair["domain"] == domain and len(pair["arms"]) == 2, "invalid domain pair")
    require([a["arm"] for a in pair["arms"]] == ["maxrl", "remax"], "paired arm order mismatch")
    # Nested endpoint snapshots can freeze the baseline's already-frozen data.
    # Trust mapping contexts only after independently rebinding every source
    # identity; never infer ancestor launchers from path spelling alone.
    contexts = []
    for arm in pair["arms"]:
        passed(arm, "normalization training context")
        directory = Path(arm["directory"])
        identity = bound_json(directory/"identity.json", arm["identity_sha256"], "normalization training identity")
        require(audit.audit_frozen_sources(directory, identity, training=True) == "pass", "unverified training normalization context")
        contexts.append(directory)
    for label in ARMS:
        endpoint = comparison["endpoints"][label]
        passed(endpoint, "normalization endpoint context")
        path = Path(endpoint["receipt"])
        receipt = bound_json(path, endpoint["receipt_sha256"], "normalization endpoint receipt")
        require(audit.audit_frozen_sources(path.parent, receipt) == "pass", "unverified endpoint normalization context")
        contexts.append(path.parent)
    arms = {a["arm"]: load_training(a, domain, chain_directories=contexts) for a in pair["arms"]}
    for key in ("seed", "eval_seed", "protocol", "sources", "data", "completed_updates", "train_ids", "initial_parameters_sha256"):
        require(arms["maxrl"][key] == arms["remax"][key], "paired training mismatch: " + key)
    require(arms["maxrl"]["job_id"] != arms["remax"]["job_id"], "duplicate paired training job")
    endpoints = {label: load_endpoint(comparison["endpoints"][label], domain, chain_directories=contexts) for label in ARMS}
    for key in ("seed", "protocol", "sources", "data", "prompts", "metadata"):
        require(endpoints["base"][key] == endpoints["maxrl"][key] == endpoints["remax"][key], "paired endpoints mismatch: " + key)
    for label in ("maxrl", "remax"):
        receipt = audit.read_json(endpoints[label]["receipt"])
        require(receipt["config"]["lora_arm"] == label, "endpoint LoRA arm mismatch")
        require(receipt["lora_checkpoint"]["seal_sha256"] == arms[label]["checkpoint_sha256"], "endpoint does not use paired terminal checkpoint")
    jobs = {str(item["job_id"]): item for item in root.get("scheduler_accounting", {}).get("jobs", [])}
    required_jobs = {arm["job_id"] for arm in arms.values()} | {endpoint["job_id"] for endpoint in endpoints.values()}
    for job_id in required_jobs:
        require(job_id in jobs, "required domain job lacks scheduler accounting: " + job_id)
        job = jobs[job_id]
        require(job.get("state") == "COMPLETED" and job.get("exit_code") == "0:0", "required domain job is not terminal successful: " + job_id)
        require(job.get("elapsed_seconds", -1) >= 0 and job.get("gpu_count", 0) >= 1 and job.get("allocated_gpu_hours", -1) >= 0, "invalid domain GPU accounting")
    if domain == "qa":
        provenance = root.get("qa_source_provenance", {})
        require(provenance.get("status") == "pass", "QA source provenance did not pass")
        bound_json(provenance["receipt"], provenance["receipt_sha256"], "QA provenance receipt")
        require(provenance["frozen_records_sha256"] == arms["maxrl"]["data"]["records_sha256"], "QA provenance dataset mismatch")
    strata = {name: data["task_ids"] for name, data in comparison["strata"].items()}
    flattened = [task for ids in strata.values() for task in ids]
    require(len(flattened) == len(set(flattened)) and set(flattened) == set(endpoints["base"]["metrics"]), "strata do not partition complete endpoint cohort")
    train_ids = set(arms["maxrl"]["train_ids"])
    for name, ids in strata.items():
        if name == "reserved_test":
            require(not train_ids.intersection(ids) and all(endpoints["base"]["metadata"][task]["split"] in {"test", "heldout"} for task in ids), "reserved test cohort overlaps training or has wrong split")
        elif name.startswith("trained_"):
            require(set(ids) == train_ids, "trained diagnostic must retain complete trained cohort")
        else:
            require(not train_ids.intersection(ids) and all(endpoints["base"]["metadata"][task]["split"] not in {"test", "heldout"} for task in ids), "development stratum has training/test overlap")
    require(audit.sha_file(report_path) == report_sha256, "independent report changed during aggregation")
    return {"seed": arms["maxrl"]["seed"], "training": arms, "endpoints": endpoints,
            "strata": strata, "report": str(report_path.resolve()), "report_sha256": report_sha256,
            "domain_readiness": selected["readiness"],
            "readiness_interpretation": "capability_not_established_but_complete_outcome_evidence" if selected["readiness"] == "unknown" else "pilot_readiness_pass",
            "terminal_accounting": [jobs[job_id] for job_id in sorted(required_jobs)]}


def seed_summary(values):
    # Missing PCMD estimates remain undefined. No available-seed mean silently
    # drops a seed whose eligible prompt intersection is empty.
    defined = [value for value in values if value is not None]
    complete = len(defined) == len(values) and bool(values)
    return {"seeds": len(values), "defined_seeds": len(defined), "values": values,
            "mean": statistics.mean(defined) if complete else None,
            "sample_sd": statistics.stdev(defined) if complete and len(defined) > 1 else None,
            "range": [min(defined), max(defined)] if complete else None,
            "status": "defined" if complete else "undefined"}


def task_value(metric, row):
    if metric == "pass_at_8":
        return row["pass_at_k"]["8"]
    if metric.startswith("expected_distinct_at_"):
        return row["expected_distinct_valid_modes_at_k"][metric.rsplit("_", 1)[-1]]
    return row[metric]


def difference(endpoints, control, treatment, ids, metric):
    if not ids:
        return None
    return statistics.mean(task_value(metric, endpoints[treatment]["metrics"][task]) - task_value(metric, endpoints[control]["metrics"][task]) for task in ids)


def summarize_stratum(reports, name, domain):
    ids = reports[0]["strata"][name]
    require(bool(ids), "empty fixed stratum")
    require(all(report["strata"].get(name) == ids for report in reports), "unequal fixed task cohorts")
    metrics = ["accuracy", "pass_at_8", "expected_distinct_at_8", "expected_distinct_at_32", "raw_distinct_at_8"]
    if domain == "qa":
        metrics.append("annotated_topic_coverage")
    common_trained = [task for task in ids if all(report["endpoints"][arm]["metrics"][task]["accepted"] >= 30 for report in reports for arm in ("maxrl", "remax"))]
    common_with_base = [task for task in common_trained if all(report["endpoints"]["base"]["metrics"][task]["accepted"] >= 30 for report in reports)]
    comparisons = {}
    for label, (control, treatment) in COMPARISONS.items():
        sensitivity_ids = common_trained if label == "remax_minus_maxrl" else common_with_base
        by_seed = []
        for report in reports:
            endpoints = report["endpoints"]
            eligible = [task for task in ids if all(endpoints[arm]["metrics"][task]["accepted"] >= 30 for arm in (control, treatment))]
            values = {metric: difference(endpoints, control, treatment, ids, metric) for metric in metrics}
            by_seed.append({"seed": report["seed"], "prompts": len(ids), "effects": values,
                            "pcmd_pair_common_eligible": {"task_ids": eligible, "denominator": len(eligible), "fixed_cohort_size": len(ids), "effect": difference(endpoints, control, treatment, eligible, "pcmd")},
                            "pcmd_all_seed_common_sensitivity": {"denominator": len(sensitivity_ids), "effect": difference(endpoints, control, treatment, sensitivity_ids, "pcmd")}})
        comparisons[label] = {
            "per_seed": by_seed,
            "seed_summary": {metric: seed_summary([row["effects"][metric] for row in by_seed]) for metric in metrics},
            "pcmd_pair_common_eligible": seed_summary([row["pcmd_pair_common_eligible"]["effect"] for row in by_seed]),
            "pcmd_all_seed_common_sensitivity": {"task_ids": sensitivity_ids, "denominator": len(sensitivity_ids), "fixed_cohort_size": len(ids), "seed_summary": seed_summary([row["pcmd_all_seed_common_sensitivity"]["effect"] for row in by_seed])},
        }
    return {"task_ids": ids, "prompts": len(ids), "comparisons": comparisons,
            "pcmd_all_trained_endpoint_intersection": common_trained,
            "pcmd_base_included_intersection": common_with_base}


def aggregate_reports(reports, domain, allow_single_seed=False, trained_reports=()):
    expected = 1 if allow_single_seed and len(reports) == 1 else 3
    require(len(reports) == expected, "exactly three paired seed reports required; one pilot needs --allow-single-seed")
    require(len({report["seed"] for report in reports}) == len(reports), "duplicate paired training seed")
    require(len({report["report_sha256"] for report in reports}) == len(reports), "duplicate input report")
    jobs = [report["training"][arm]["job_id"] for report in reports for arm in ("maxrl", "remax")]
    require(len(jobs) == len(set(jobs)), "duplicate training scheduler job across seeds")
    reports = sorted(reports, key=lambda report: report["seed"])
    for report in reports[1:]:
        for arm in ("maxrl", "remax"):
            for key in ("protocol", "sources", "data", "objective", "completed_updates", "train_ids"):
                require(report["training"][arm][key] == reports[0]["training"][arm][key], "training study mismatch across seeds: " + key)
        for arm in ARMS:
            for key in ("protocol", "sources", "data", "prompts", "metadata"):
                require(report["endpoints"][arm][key] == reports[0]["endpoints"][arm][key], "endpoint study mismatch across seeds: " + key)
        require(report["strata"] == reports[0]["strata"], "unequal fixed task cohorts")
    if expected == 3:
        require(set(reports[0]["strata"]) == {"reserved_test"}, "three-seed primary reports must evaluate only the fixed reserved test set")
    else:
        require("reserved_test" in reports[0]["strata"] or any(name in reports[0]["strata"] for name in ("dev", "untrained_development")), "pilot lacks an untrained development stratum")
    primary_name = "reserved_test" if "reserved_test" in reports[0]["strata"] else ("dev" if domain == "qa" else "untrained_development")
    strata = {name: summarize_stratum(reports, name, domain) for name in reports[0]["strata"]}
    auxiliary = list(trained_reports)
    if auxiliary:
        require(len(auxiliary) == len(reports), "trained diagnostics need exactly one report per primary seed")
        by_seed = {report["seed"]: report for report in auxiliary}
        require(len(by_seed) == len(auxiliary) and set(by_seed) == {report["seed"] for report in reports}, "trained diagnostic seed selection differs")
        ordered = [by_seed[report["seed"]] for report in reports]
        for original, extra in zip(reports, ordered):
            require(original["training"] == extra["training"], "trained diagnostic uses a different training identity")
            require(len(extra["strata"]) == 1 and all(name.startswith("trained_") for name in extra["strata"]), "auxiliary report must contain only trained diagnostics")
        require(all(extra["strata"] == ordered[0]["strata"] for extra in ordered), "unequal trained diagnostic cohort")
        for extra in ordered[1:]:
            for arm in ARMS:
                for key in ("protocol", "sources", "data", "prompts", "metadata"):
                    require(extra["endpoints"][arm][key] == ordered[0]["endpoints"][arm][key], "trained endpoint protocol mismatch")
        for name in ordered[0]["strata"]:
            require(name not in strata, "duplicate trained diagnostic stratum")
            strata[name] = summarize_stratum(ordered, name, domain)
    inputs = [{"path": report["report"], "sha256": report["report_sha256"], "seed": report["seed"], "role": role, "domain_readiness": report["domain_readiness"], "readiness_interpretation": report["readiness_interpretation"], "terminal_accounting": report["terminal_accounting"]} for role, group in (("primary", reports), ("trained_diagnostic", auxiliary)) for report in group]
    return {"schema": SCHEMA, "status": "pass", "domain": domain,
            "kind": "single_seed_descriptive_pilot" if expected == 1 else "three_paired_seed_descriptive_study",
            "seed_count": expected, "seeds": [report["seed"] for report in reports],
            "primary_stratum": primary_name, "primary_is_reserved_test": primary_name == "reserved_test",
            "trained_diagnostic_available": any(name.startswith("trained_") for name in strata),
            "inputs": inputs, "strata": strata,
            "training_bindings": [{"seed": report["seed"], "endpoint_sampling_seed": report["endpoints"]["base"]["seed"], "training_protocol_sha256": audit.object_sha(report["training"]["maxrl"]["protocol"]), "endpoint_protocol_sha256": audit.object_sha(report["endpoints"]["base"]["protocol"]), "arms": {arm: {key: report["training"][arm][key] for key in ("directory", "job_id", "identity_sha256", "checkpoint_sha256", "completed_updates")} for arm in ("maxrl", "remax")}} for report in reports],
            "interpretation": "This is a statistical aggregation integrity gate, not a favorable capability or pilot-readiness gate. Complete all-zero outcomes remain valid. Every supplied seed and every fixed stratum prompt is retained. Means, sample SDs and ranges describe paired seed effects; no prompt-bootstrap interval is presented as seed uncertainty. Empty PCMD intersections are undefined. No efficacy threshold or favorable subset selection is applied.",
            "pcmd_rule": "At least 30 accepted samples per prompt and endpoint. Pair effects use their two-arm common intersection; sensitivity uses all six trained endpoints (two for explicit single-seed pilot), adding every base endpoint only for comparisons with base.",
            "raw_distinct_at_8_rule": "Mean valid canonical-key count in consecutive disjoint groups of eight by sample_index, averaged equally over fixed prompts; estimator ED@8 is separately reported.",
            "aggregator_sha256": audit.sha_file(__file__), "audit_helper_sha256": audit.sha_file(audit.__file__)}


def markdown(result):
    lines = [f"# {result['domain']} paired-seed aggregation", "", f"Status: {result['status']}; {result['kind']}; seeds {result['seeds']}.", "", f"Primary stratum: `{result['primary_stratum']}`. Reserved test: {result['primary_is_reserved_test']}.", "", result["interpretation"], "", "| Stratum | Metric | Seed effects | Mean | Sample SD | Range |", "|---|---|---|---|---|---|"]
    def fmt(value):
        return "undefined" if value is None else f"{value:.8g}"
    denominator_notes = []
    for name, data in result["strata"].items():
        comparison = data["comparisons"]["remax_minus_maxrl"]
        values = {**comparison["seed_summary"], "pcmd_pair_common": comparison["pcmd_pair_common_eligible"], "pcmd_all_seed_common": comparison["pcmd_all_seed_common_sensitivity"]["seed_summary"]}
        for metric, summary in values.items():
            span = "undefined" if summary["range"] is None else " to ".join(fmt(v) for v in summary["range"])
            lines.append(f"| {name} ({data['prompts']}) | Re:Max − MaxRL {metric} | {', '.join(fmt(v) for v in summary['values'])} | {fmt(summary['mean'])} | {fmt(summary['sample_sd'])} | {span} |")
        denominators = [row["pcmd_pair_common_eligible"]["denominator"] for row in comparison["per_seed"]]
        denominator_notes.append(f"{name}: pair-common PCMD denominators {denominators}/{data['prompts']}; all-trained-endpoint intersection {len(data['pcmd_all_trained_endpoint_intersection'])}/{data['prompts']}.")
    lines.extend(["", *denominator_notes, ""])
    lines.extend([result["pcmd_rule"], "", result["raw_distinct_at_8_rule"], "", "Base comparisons, exact task IDs and source/input bindings are in summary.json.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("code", "qa"), required=True)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--trained-report", action="append", default=[], type=Path)
    parser.add_argument("--allow-single-seed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be a new directory")
    result = aggregate_reports([load_report(path, args.domain) for path in args.report], args.domain, args.allow_single_seed, [load_report(path, args.domain) for path in args.trained_report])
    args.output.mkdir(parents=True)
    (args.output/"summary.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (args.output/"report.md").write_text(markdown(result))
    print(json.dumps({"status": result["status"], "domain": result["domain"], "seeds": result["seeds"], "primary_stratum": result["primary_stratum"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
