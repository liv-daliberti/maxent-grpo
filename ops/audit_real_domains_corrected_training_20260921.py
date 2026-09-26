#!/usr/bin/env python3
"""Strict receipt audit for the explicitly recognized corrected v2 trainer.

Loads a pinned copy of the original audit logic into an isolated Python module;
the original auditor and historical evidence are never edited. This audits
paired training integrity, not endpoint efficacy, GPU accounting or readiness.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

SCHEMA = "real-domains-corrected-training-independent-audit-20260921-v1"
TRAINER_FILENAME = "train_real_domains_pilot_20260921_v2.py"
TRAINER_SHA256 = "e320611aa612d5f2772fe4b45f8e9bbbbad126a5666090b1f97f277049d00535"
TRAINER_SCHEMA = "real-domain-online-maxrl-remax-pilot-20260921-v2-matched-scoring-width"
HELPER_FILENAME = "summarize_real_domains_pilot_20260921.py"
HELPER_SHA256 = "13adb7df5bc19391f92ea1781c09a0477abda0c2a1ecfa4e98cf8d386a80ba71"
BEHAVIOR_CONTRACT = "attention-trimmed rectangle identical to each shuffled live microbatch; microbatch1 required"
ZERO_METRICS = ("logprobs_diff_min", "logprobs_diff_max", "pg_clipfrac")
PRODUCTION_MODULES = frozenset(("oat_drgrpo.learner.grpo", "oat_drgrpo.learner.base", "oat_drgrpo.canonical_replay", "oat_drgrpo.maxrl", "oat_drgrpo.online_canonical_bank", "oat_drgrpo.scoring", "oat_drgrpo.tensor_utils", "oat_drgrpo.args"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_pinned_helper():
    path = Path(__file__).with_name(HELPER_FILENAME)
    require(digest(path) == HELPER_SHA256, "original audit helper source is not the explicitly pinned version")
    spec = importlib.util.spec_from_file_location("_real_domains_corrected_isolated_original_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_corrected_frozen_sources(directory, receipt, training=False, *, helper):
    """Preserve the original implementation/config checks with an exact v2 pin."""
    require(training is True, "corrected-training wrapper cannot audit endpoint protocols")
    directory = Path(directory)
    root = directory.parent if directory.name == "training" else directory
    launcher_path = root / "identity.json"
    require(launcher_path.is_file(), "missing corrected frozen launcher identity")
    launcher = helper.read_json(launcher_path)
    require(launcher.get("schema") == "real-domains-frozen-job-20260921-v1", "unknown frozen launcher schema")
    request = launcher.get("request", {})
    require(request.get("entrypoint") == TRAINER_FILENAME, "frozen launch requested a different trainer")
    require(request.get("arm") == receipt.get("arm") and receipt.get("arm") in {"maxrl", "remax"}, "frozen launch arm differs from training receipt")
    config_path = root / "config.json"
    require(digest(config_path) == launcher["config_sha256"], "frozen launcher config hash mismatch")
    paths = [str(Path(f["snapshot"]).resolve()) for f in launcher["files"]]
    require(len(paths) == len(set(paths)), "duplicate frozen implementation paths")
    files = dict(zip(paths, (f["sha256"] for f in launcher["files"])))

    def verify(path, expected):
        path = Path(path)
        require(digest(path) == expected == files.get(str(path.resolve())), "frozen implementation hash mismatch: " + path.name)

    runner = root / "bundle/ops" / TRAINER_FILENAME
    require(receipt.get("schema") == TRAINER_SCHEMA, "unknown or historical training schema")
    require(receipt.get("runner_sha256") == TRAINER_SHA256, "unknown or altered corrected trainer version")
    verify(runner, TRAINER_SHA256)
    assignments = [n for n in ast.parse(runner.read_text()).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in n.targets)]
    require(len(assignments) == 1, "frozen training defaults not uniquely declared")
    resolved = {**ast.literal_eval(assignments[0].value), **helper.read_json(config_path)}
    require(resolved == receipt["config"], "resolved configuration differs from frozen launch file and sealed defaults")
    require(type(resolved["train_microbatch_size"]) is int and resolved["train_microbatch_size"] == 1,
            "corrected numerical contract requires microbatch size one")
    require(receipt["objective"].get("behavior_scoring") == BEHAVIOR_CONTRACT,
            "missing or altered matched behavior/live scoring contract")
    module = resolved["adapter_module"]
    adapter = root / "bundle" / ("src" if "." in module else "ops") / (module.replace(".", "/") + ".py")
    verify(adapter, receipt["adapter_module_sha256"])
    require(set(receipt.get("production_source_sha256", {})) == PRODUCTION_MODULES, "incomplete or unexpected production source module set")
    for module, expected in receipt["production_source_sha256"].items():
        verify(root / "bundle/src" / (module.replace(".", "/") + ".py"), expected)
    require(digest(Path(resolved["model"]) / "config.json") == receipt["model_config_sha256"],
            "base model configuration hash mismatch")
    return "pass"


def audit_zero_ratio_contract(metrics, updates):
    require(len(metrics) == updates and [r["completed_updates"] for r in metrics] == list(range(1, updates + 1)),
            "incomplete or reordered corrected numerical diagnostics")
    for row in metrics:
        for key in ZERO_METRICS:
            value = row.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value == 0,
                    f"corrected behavior/live numerical contract failed at update {row['completed_updates']}: {key}")
    return {"status": "pass", "updates_checked": updates, "exact_zero_metrics": list(ZERO_METRICS),
            "interpretation": "All reported response-token old/new log-probability extrema and initial clipping fractions equal zero before each optimizer update."}


def audit_pair(maxrl_run, remax_run):
    helper = load_pinned_helper()
    helper.audit_frozen_sources = lambda directory, receipt, training=False: audit_corrected_frozen_sources(
        directory, receipt, training, helper=helper)
    directories = [helper.training_dir(Path(p)) for p in (maxrl_run, remax_run)]
    identities = [helper.read_json(p / "identity.json") for p in directories]
    codecs = [helper.local_codec(identity["config"]) for identity in identities]
    pair = helper.audit_pair(*directories, codecs=codecs)
    require(pair["status"] == "pass" and pair["token_text_binding"] == "pass"
            and pair["frozen_implementation_binding"] == "pass", "incomplete original paired integrity gates")
    contracts = {}
    for directory, identity, arm in zip(directories, identities, pair["arms"]):
        result = helper.read_json(directory / "result.json")
        updates = identity["config"]["updates"]
        seal = helper.read_json(directory / f"checkpoint-{updates}" / "complete.json")
        require(result.get("schema") == seal.get("schema") == TRAINER_SCHEMA, "historical/unknown completion schema")
        contract = audit_zero_ratio_contract(helper.read_rows(directory / "metrics.jsonl"), updates)
        contract.update(behavior_scoring=BEHAVIOR_CONTRACT, train_microbatch_size=1,
                        metrics_sha256=digest(directory / "metrics.jsonl"),
                        training_schema=TRAINER_SCHEMA, runner_sha256=TRAINER_SHA256)
        contracts[arm["arm"]] = contract
    return {
        "schema": SCHEMA, "status": "pass", "paired_training": pair, "corrected_numerical_contract": contracts,
        "analysis_bindings": {"wrapper_sha256": digest(Path(__file__)), "original_helper_sha256": HELPER_SHA256,
                              "trainer_filename": TRAINER_FILENAME, "trainer_sha256": TRAINER_SHA256},
        "scope": "Complete paired training integrity and corrected numerical scoring contract only. Built-in final evaluations are diagnostics; no external endpoint efficacy, scheduler accounting, or larger-study readiness is established.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxrl-run", type=Path, required=True)
    parser.add_argument("--remax-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("corrected audit requires a new output JSON path")
    result = audit_pair(args.maxrl_run, args.remax_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "domain": result["paired_training"]["domain"], "output": str(args.output)}))
