#!/usr/bin/env python3
"""Independent CPU receipt audit for explicitly supplied coding and QA pilots.

No Slurm queries, model inference, or generated-code execution are performed.
QA decisions are regraded from frozen labels; code checker/stability evidence is
verified and bound to raw source. Missing evidence remains unknown.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import gzip
import io
import json
import math
from pathlib import Path
import re
import random
from typing import Any

SCHEMA = "independent-real-domains-pilot-audit-20260921-v1"


HISTORICAL_CODE_REVOCATION = {
    "manifest_sha256": "51e2d88f8a8e699a96da1af307e94936691b0e60fd2a89bc9cde8e497158cfb9",
    "source_evaluation_sha256": "738e6dcb6430f0c990c2286cd441be097296e05c68d6ed86652a92e04b17a363",
    "stress_receipt_sha256": "884e86371558f2372585fd1dd555e3ccbb411309210372b9a2e4e38ceaa64c0a",
    "stress_attempts_sha256": "b8d794790b5d9022a73c0d57ad0cc14bdf6e252528841b8de8c2a6f980e50689",
    "original_accepted_programs": 74,
    "original_accepted_programs_stress_tested": 14,
    "stress_rejected_original_acceptances": 6,
    "source_stress_affected_tasks": ["1038_B", "1408_A"],
    "readiness": "fail",
    "reason": "Historical receipt integrity passed, but fixed source-audit probes exposed false acceptance; historical code results cannot establish readiness or treatment efficacy.",
}


def require_unrevoked_code_manifest(digest):
    require(digest != HISTORICAL_CODE_REVOCATION["manifest_sha256"], "historical code verifier revoked: six of fourteen stress-tested accepted programs failed fixed source-audit probes")


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, f"duplicate request identity: {k}")
        result[k] = row
    return result


def validate_verdict(v):
    require(type(v.get("accepted")) is bool, "acceptance is not boolean")
    require(isinstance(v.get("hard_violations"), list), "missing hard violation list")
    require(not v["hard_violations"], "hard verifier violation")
    require(isinstance(v.get("receipt"), dict), "missing verifier receipt")
    key = v.get("canonical_key")
    require((isinstance(key, str) and bool(key)) if v["accepted"] else key is None, "reward/mode mismatch")


def discovery(n, count, k):
    require(0 <= count <= n and 1 <= k <= n, "invalid discovery denominator")
    return 1.0 if n-count < k else 1-math.comb(n-count, k)/math.comb(n, k)


def metrics(verdicts, gold=None):
    require(bool(verdicts), "cannot summarize zero requested samples")
    for verdict in verdicts:
        validate_verdict(verdict)
    counts = Counter(v["canonical_key"] for v in verdicts if v["accepted"])
    n, accepted = len(verdicts), sum(counts.values())
    result = {
        "samples": n, "accepted": accepted, "accuracy": accepted/n,
        "mode_counts": dict(sorted(counts.items())), "distinct_valid_modes": len(counts),
        "pcmd_eligible": accepted >= 30, "pcmd_accepted_threshold": 30,
        "pcmd": 1-sum(c*(c-1) for c in counts.values())/(accepted*(accepted-1)) if accepted >= 30 else None,
        "pass_at_k": {str(k): discovery(n, accepted, k) for k in (1, 8, 32) if k <= n},
        "expected_distinct_valid_modes_at_k": {str(k): sum(discovery(n, c, k) for c in counts.values()) for k in (1, 8, 32) if k <= n},
    }
    if gold is not None:
        require(len(gold) >= 2 and set(counts) <= set(gold), "observed QA key outside native gold support")
        result.update(known_mode_count=len(gold), annotated_topic_coverage=len(counts)/len(gold), missing_annotated_topics=sorted(set(gold)-set(counts)))
    return result


def aggregate(task_metrics):
    values = list(task_metrics.values())
    require(bool(values), "no task metrics")
    eligible = [v["pcmd"] for v in values if v["pcmd_eligible"]]
    coverage = [v["annotated_topic_coverage"] for v in values if "annotated_topic_coverage" in v]
    common_k = set.intersection(*(set(v["expected_distinct_valid_modes_at_k"]) for v in values))
    return {
        "tasks": len(values), "samples": sum(v["samples"] for v in values),
        "macro_accuracy": sum(v["accuracy"] for v in values)/len(values),
        "tasks_with_multiple_modes": sum(v["distinct_valid_modes"] >= 2 for v in values),
        "pcmd_eligible_tasks": len(eligible), "pcmd_total_tasks": len(values),
        "macro_pcmd_over_eligible": sum(eligible)/len(eligible) if eligible else None,
        "macro_expected_distinct_valid_modes_at_k": {k: sum(v["expected_distinct_valid_modes_at_k"][k] for v in values)/len(values) for k in sorted(common_k, key=int)},
        "macro_annotated_topic_coverage": sum(coverage)/len(coverage) if coverage else None,
    }


def qa_records(config):
    if config.get("adapter_module") != "oat_drgrpo.noncoding_multi_answer_sata":
        return None
    adapter = config["adapter_config"]
    path = Path(adapter["records_path"])
    require(sha_file(path) == adapter["records_sha256"], "QA frozen records hash mismatch")
    return unique(read_rows(path), lambda r: r["id"])


def read_suite(path, descriptor):
    require(sha_file(path) == descriptor["compressed_jsonl_sha256"], "compressed input suite hash mismatch")
    with gzip.open(path,"rt",encoding="ascii") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    identities=[]
    for index,row in enumerate(rows):
        raw=row["stdin"].encode()
        identity={"test_index":index,"input_bytes":len(raw),"input_sha256":sha_bytes(raw)}
        require(all(row[k] == v for k,v in identity.items()), "input test bytes/index/hash mismatch")
        identities.append(identity)
    require(len(rows) == descriptor["test_count"] and object_sha(identities) == descriptor["suite_sha256"], "effective suite identity mismatch")
    return rows


def audit_hardened_reference_ledger(directory, record, admitted):
    directory=Path(directory)
    require(sha_file(directory/"checker.cpp") == record["checker_sha256"], "frozen released checker source hash mismatch")
    require(sha_file(directory/"validator.cpp") == record["validator_sha256"], "frozen released input-validator source hash mismatch")
    read_suite(directory/record.get("suite_file","inputs.jsonl.gz"),record["suite"])
    require(sha_file(directory/"py3_replays.jsonl") == record["references"]["jsonl_sha256"], "human reference-program ledger hash mismatch")
    require(sha_file(directory/"audit_replays.jsonl") == admitted["replays_sha256"], "human reference-admission ledger hash mismatch")
    refs=unique(read_rows(directory/"py3_replays.jsonl"),lambda row:row["submission_sha256"])
    replays=unique(read_rows(directory/"audit_replays.jsonl"),lambda row:row["submission_sha256"])
    require(set(refs) == set(replays), "human reference/replay identity bijection mismatch")
    labels=Counter()
    positives=negatives=0
    for key,ref in refs.items():
        replay=replays[key];label=ref["known_label"]
        require(label in {"correct","incorrect"}, "unknown human reference label")
        require(sha_bytes(ref["code"].encode()) == key, "human reference source hash mismatch")
        require(replay["known_label"] == label and not replay["audit_violations"], "reference label mismatch or hard replay violation")
        for field,value in (("source_problem_id",record["source_problem_id"]),("checker_sha256",record["checker_sha256"]),("suite_id",record["suite_id"]),("suite_sha256",record["suite"]["suite_sha256"]),("task_adapter",record["task_adapter"])):
            require(replay[field] == value, "reference replay uses a different checker/suite/task: "+field)
        labels[label] += 1
        require(type(replay["released_checker_accepted"]) is bool and type(replay["wrapper_accepted"]) is bool, "reference acceptance is not boolean")
        if label == "incorrect":
            require(not replay["released_checker_accepted"] and not replay["wrapper_accepted"], "known human-negative program accepted by hardened released checker")
            negatives += 1
        elif replay["released_checker_accepted"] and replay["wrapper_accepted"]:
            positives += 1
            require(replay["execution"]["executed_tests"] == replay["execution"]["suite_tests"] == record["suite"]["test_count"], "accepted human reference did not execute full hardened suite")
    require(labels == {"correct":12,"incorrect":12} and positives == 12 and negatives == 12, "hardened reference panel does not meet twelve-per-label TPR/TNR gate")
    require(close(admitted["tpr"],positives/12) and admitted["tnr"] == 1.0 and admitted["positive_replays"] == admitted["negative_replays"] == 12, "hardened admission metrics disagree with raw human-reference replays")
    return {"positive_accepted":positives,"positive_total":12,"negative_rejected":negatives,"negative_total":12,"replay_ledger_sha256":admitted["replays_sha256"]}


FIXED_SOURCE_PROBE_SHA256 = "3e7c34e2d8b2e3bb6a5d4b329525bffaeced7e8fb2791f2d09f257bca65513cb"
HARDENING_SCHEMA = "constructive-code-hardened-20260921-v1"


VERIFIER_CRITICAL_PATHS = {
    "ops/build_constructive_code_wider_20260921.py",
    "ops/audit_constructive_code_sources.py",
    "ops/evaluate_constructive_code_pilot_20260921.py",
    "ops/evaluate_constructive_code_v3_coder_viability.py",
    "ops/replay_constructive_code_review_slate.py",
    "ops/replay_constructive_code_v2.py",
    "ops/constructive_code_sandbox.c",
    "src/oat_drgrpo/constructive_code_wider_adapters_20260921.py",
    "src/oat_drgrpo/constructive_code_reserved_adapters_20260921.py",
    "src/oat_drgrpo/constructive_code_holdout_extension_20260921.py",
    "src/oat_drgrpo/constructive_code_adapters.py",
    "src/oat_drgrpo/constructive_code_sandbox.py",
    "src/oat_drgrpo/constructive_code.py",
    "third_party/testlib/testlib.h",
}
PINNED_TESTLIB_SHA256 = "bb323e3c89285214966076e0d23d5a295c5f6126da7ff198c1276ddb95ecb1a0"


def relocated_source_identity(mapping):
    normalized={}
    for name,digest in mapping.items():
        path=Path(name);parts=path.parts
        if path.is_absolute():
            anchors=[i for i,part in enumerate(parts) if part in {"src","ops","third_party"}]
            require(bool(anchors), "unrecognized verifier source location")
            path=Path(*parts[anchors[-1]:])
        require(not path.is_absolute() and ".." not in path.parts and path.parts[0] in {"src","ops","third_party"}, "invalid relative verifier source identity")
        key=path.as_posix()
        require(key not in normalized and re.fullmatch(r"[0-9a-f]{64}",digest), "duplicate or invalid verifier source identity")
        normalized[key]=digest
    return normalized


def audit_hardening_sources(root, quality):
    root=Path(root)
    sources=relocated_source_identity(quality["canonicalizer_source_files"])
    support=relocated_source_identity(quality["verifier_support_files"])
    require(not set(sources)&set(support), "overlapping canonicalizer/support source identities")
    sources.update(support)
    require(VERIFIER_CRITICAL_PATHS <= set(sources), "hardened source identity omits verifier-critical helper files")
    source_root=root.parent.parent if root.parent.name == "data" and root.parent.parent.name == "bundle" else Path(__file__).resolve().parents[1]
    for relative,digest in sources.items():
        path=source_root/relative
        if relative=="third_party/testlib/testlib.h" and source_root.name=="bundle" and not path.exists():
            launcher=read_json(source_root.parent/"identity.json")
            candidates=[row for row in launcher["files"] if Path(row["source"]).as_posix().endswith("/third_party/testlib/testlib.h")]
            require(len(candidates)==1 and candidates[0]["sha256"]==digest, "frozen testlib relocation is not uniquely byte-bound")
            path=Path(candidates[0]["snapshot"])
            require(path.resolve()==(source_root/"testlib/testlib.h").resolve(), "frozen testlib relocation has unexpected destination")
        require(sha_file(path) == digest, "active frozen verifier support bytes differ from source-quality receipt: "+relative)
    require(quality["pinned_testlib_sha256"] == sources["third_party/testlib/testlib.h"] == PINNED_TESTLIB_SHA256, "hardened checker uses unpinned testlib")
    require(sha_file(source_root/"ops/build_constructive_code_hardened_20260921.py") == quality["loader_guard_sha256"], "active hardened loader guard hash mismatch")
    post=quality.get("post_audit_provenance_verification")
    if post is not None:
        executed=Path(post["executed_hardener_source"])
        require(not executed.is_absolute() and ".." not in executed.parts, "invalid executed-source snapshot path")
        require(sha_file(root/executed) == post["executed_hardener_sha256"] == quality["builder_sha256"], "original executed hardener source snapshot hash mismatch")
        require(post["canonicalizer_sources_byte_identical_to_full_replay"] is True and post["support_sources_unchanged_since_before_hardener_execution"] is True, "post-audit source identity does not assert unchanged verification behavior")
    else:
        require(quality["builder_sha256"] == quality["loader_guard_sha256"], "executed and current hardeners differ without explicit provenance supplement")
    if "coordinator_source_sha256" in quality:
        require(sha_file(source_root/"ops/build_constructive_code_hardened_larger_20260921.py") == quality["coordinator_source_sha256"], "larger replay coordinator source hash mismatch")
        sealing=quality["post_replay_pool_sealing"]
        require(sha_file(source_root/"ops/seal_constructive_code_hardened_larger_20260921.py") == sealing["sealer_source_sha256"], "post-replay sealer source hash mismatch")
        require(sealing["task_adapter_problem_key_and_canonical_record_hashes_revalidated"] is True, "larger replay post-validation is missing")
    return {"status":"pass","relative_source_sha256":sources,"loader_guard_sha256":quality["loader_guard_sha256"],"executed_builder_sha256":quality["builder_sha256"],"post_audit_supplement_declared":post is not None}


def hardening_quality(root, manifest, dataset_identity):
    root=Path(root);quality=read_json(root/"hardening_quality.json")
    require(sha_file(root/"hardening_quality.json") == manifest["hardening_quality_sha256"] == dataset_identity["hardening_quality_sha256"], "hardened global quality receipt hash mismatch")
    require(quality["schema"] == "constructive-code-hardening-quality-20260921-v1" and quality["status"] == "pass" and quality["hardening_schema"] == HARDENING_SCHEMA, "hardened global quality gate is not passing")
    policy=quality["policy"]
    require(policy["required_tpr"] == policy["required_tnr"] == 1.0 and policy["positive_replays"] == policy["negative_replays"] == 12, "hardened quality policy is weaker than twelve-per-label exact TPR/TNR")
    require(policy["append_only"] is True and all(policy[k] is False for k in ("checker_modified","prompt_modified","policy_outputs_used_for_test_selection")), "hardened quality policy permits changed prompts/checkers or model-conditioned probes")
    require(quality["fixed_probe_manifest_sha256"] == manifest["fixed_probe_manifest_sha256"] == dataset_identity["fixed_probe_manifest_sha256"] == sha_file(root/"fixed_source_probe_fixture.json") == FIXED_SOURCE_PROBE_SHA256, "fixed source-audit probe provenance mismatch")
    require(quality["source_manifest_sha256"] == manifest["source_manifest_sha256"] == dataset_identity["hardening_source_manifest_sha256"] == sha_file(root/"pre_hardening_manifest.json"), "pre-hardening source manifest hash mismatch")
    fixture=read_json(root/"fixed_source_probe_fixture.json")
    require(fixture["model_sampling_used_in_probe_selection"] is False, "source probes selected using policy samples")
    items=unique(quality["tasks"],lambda row:row["task_id"])
    passed={k for k,v in items.items() if v["status"] == "pass"}
    require(passed == set(quality["admitted_problem_ids"]) == set(manifest["admitted_problem_ids"]), "hardened quality admitted-task set mismatch")
    require(quality["source_candidates"] == len(items) and quality["admitted"] == len(passed) and quality["quarantined"] == len(items)-len(passed), "hardened admission/quarantine denominator mismatch")
    for task in passed:
        row=items[task]
        require(all(row[k] == 12 for k in ("positive_replays","negative_replays","positive_accepted","negative_rejected")) and row["tpr"] == row["tnr"] == 1.0 and not row["violations"], "hardened admitted task violates exact reference gate")
    return quality,items,unique(fixture["cases"],lambda row:row["task_id"])


def audit_hardening_task(root, directory, record, admitted, quality, item, probes):
    root,directory=Path(root),Path(directory);h=record["hardening"]
    require(h["schema"] == HARDENING_SCHEMA and h["source_manifest_sha256"] == quality["source_manifest_sha256"] and h["fixed_probe_manifest_sha256"] == FIXED_SOURCE_PROBE_SHA256, "task hardening provenance identity mismatch")
    old=read_json(root/"pre_hardening_manifest.json")
    old_items=unique(old["tasks"],lambda row:row["source_problem_id"])
    require(h["original_task_record_sha256"] == item["original_task_record_sha256"] == old_items[record["source_problem_id"]]["task_record_sha256"], "task original-source record binding mismatch")
    restored={k:v for k,v in record.items() if k not in ("hardening","task_record_sha256")}
    restored.update(suite_id=h["original_suite_id"],suite=h["original_suite"],suite_file=h["original_suite_file"],admission_status="admitted",admission_audit_sha256=h["original_admission_sha256"])
    require(object_sha(restored) == h["original_task_record_sha256"], "hardened task changes fields beyond registered input-suite/admission changes")
    require(sha_file(directory/"pre_hardening_admission_audit.json") == h["original_admission_sha256"] and sha_file(directory/"pre_hardening_audit_replays.jsonl") == h["original_replays_sha256"], "original admission provenance bytes mismatch")
    old_rows=read_suite(directory/h["original_suite_file"],h["original_suite"])
    new_rows=read_suite(directory/record["suite_file"],record["suite"])
    task=record["source_problem_id"];probe=probes.get(task)
    expected_additions=[probe["stdin"]] if probe else []
    require([r["stdin"] for r in new_rows] == [r["stdin"] for r in old_rows]+expected_additions, "hardened suite does not preserve original inputs and append only fixed source probes")
    require([r["stdin"] for r in h["added_inputs"]] == expected_additions, "task source-probe addition declaration mismatch")
    if probe:
        require(probe["checker_sha256"] == record["checker_sha256"] and probe["frozen_reward_suite_sha256"] == h["original_suite"]["suite_sha256"] and sha_bytes(probe["stdin"].encode()) == probe["input_sha256"], "fixed source probe does not bind original checker/suite/input")
    require(sha_bytes(record["statement"].encode()) == record["statement_sha256"] == h["statement_sha256"] and record["checker_sha256"] == h["checker_sha256"] and record["references"]["jsonl_sha256"] == h["reference_ledger_sha256"], "hardened prompt/checker/human-reference identity changed")
    bindings={"task_id":task,"task_record_sha256":record["task_record_sha256"],"admission_audit_sha256":record["admission_audit_sha256"],"reference_replays_sha256":admitted["replays_sha256"],"reference_ledger_sha256":record["references"]["jsonl_sha256"],"checker_sha256":record["checker_sha256"],"statement_sha256":record["statement_sha256"],"original_checker_sha256":h["checker_sha256"],"original_statement_sha256":h["statement_sha256"],"original_suite_id":h["original_suite_id"],"original_suite_sha256":h["original_suite"]["suite_sha256"],"original_suite_file_sha256":h["original_suite"]["compressed_jsonl_sha256"],"hardened_suite_id":record["suite_id"],"hardened_suite_sha256":record["suite"]["suite_sha256"],"hardened_suite_file_sha256":record["suite"]["compressed_jsonl_sha256"],"original_input_count":len(old_rows),"hardened_input_count":len(new_rows),"added_input_sha256s":[sha_bytes(value.encode()) for value in expected_additions]}
    for key,value in bindings.items():
        require(item[key] == value, "global/task hardened quality binding mismatch: "+key)
    require(item["audit_method"] == admitted["audit_method"], "global/task audit method mismatch")
    if admitted["audit_method"] == "reused_exact_hardened_verification_contract_24_reference_replays":
        audit_reused_hardening_task(root,directory,record,admitted,quality)
    else:
        require(admitted["audit_method"] == "full_independent_12_positive_12_negative_replay", "unknown hardened admission audit method")
    for key in ("original_inputs_preserved_in_order","original_checker_unchanged","original_prompt_unchanged"):
        require(item[key] is True and admitted[key] is True, "hardened invariant flag is not true: "+key)
    require(relocated_source_identity(admitted["canonicalizer_source_files"]) == relocated_source_identity(quality["canonicalizer_source_files"]), "canonicalizer source bytes changed across reference admission audits")



REUSED_CONTRACT_FIELDS = ("source_problem_id","problem_key","task_adapter","witness_family","limits","checker_sha256","validator_sha256","statement","statement_sha256","references","suite_id","suite")


def audit_reused_hardening_task(root,directory,record,admitted,quality):
    """Recompute equivalence; a signed-looking reuse declaration alone is insufficient."""
    root,directory=Path(root),Path(directory)
    origin=admitted["replay_reuse_origin"];old_root=Path(origin["root"])
    require(old_root.resolve()!=root.resolve(), "replay reuse provenance cycle")
    for filename,field in (("manifest.json","manifest_sha256"),("hardening_quality.json","hardening_quality_sha256")):
        require(sha_file(old_root/filename)==origin[field], "replay origin root seal mismatch: "+field)
    old_manifest=read_json(old_root/"manifest.json");old_quality=read_json(old_root/"hardening_quality.json")
    require(old_manifest["hardening_quality_sha256"]==origin["hardening_quality_sha256"] and old_manifest["tasks_sha256"]==object_sha(old_manifest["tasks"]), "replay origin manifest binding mismatch")
    old_item=unique(old_manifest["tasks"],lambda row:row["source_problem_id"])[record["source_problem_id"]]
    old_dir=old_root/old_item["relative_path"];old=read_json(old_dir/"task.json")
    require(old["task_record_sha256"]==old_item["task_record_sha256"]==origin["task_record_sha256"]==object_sha({k:v for k,v in old.items() if k!="task_record_sha256"}), "replay origin task record hash mismatch")
    require(record["task_record_sha256"]==object_sha({k:v for k,v in record.items() if k!="task_record_sha256"}), "replay target task record hash mismatch")
    for field in REUSED_CONTRACT_FIELDS:
        require(old[field]==record[field], "reused execution contract changed: "+field)
    require(sha_file(old_dir/"admission_audit.json")==origin["admission_audit_sha256"]==old["admission_audit_sha256"], "replay origin admission hash mismatch")
    old_admitted=read_json(old_dir/"admission_audit.json")
    require(old_admitted["audit_method"]=="full_independent_12_positive_12_negative_replay", "replay origin was not a full independent reference replay")
    restored={k:v for k,v in admitted.items() if k!="replay_reuse_origin"};restored["audit_method"]=old_admitted["audit_method"]
    require(restored==old_admitted and origin["original_status"]==old_admitted["status"], "replay reuse changed original counts, verdicts, runtime or audit details")
    for field in ("canonicalizer_source_files","verifier_support_files"):
        require(relocated_source_identity(old_quality[field])==relocated_source_identity(quality[field]), "replay origin verifier source contract changed")
    require(sha_file(old_dir/"audit_replays.jsonl")==sha_file(directory/"audit_replays.jsonl")==origin["reference_replays_sha256"]==admitted["replays_sha256"], "replay reuse changed reference verdict bytes")
    suite=read_suite(directory/record["suite_file"],record["suite"])
    require(read_suite(old_dir/old["suite_file"],old["suite"])==suite, "replay reuse changed ordered input records")
    contract={"problem_id":record["source_problem_id"],"suite_id":record["suite_id"],"suite_sha256":record["suite"]["suite_sha256"],"inputs_in_order_sha256":object_sha([r["stdin"] for r in suite]),"checker_sha256":record["checker_sha256"],"reference_ledger_sha256":record["references"]["jsonl_sha256"],"limits":record["limits"],"statement_sha256":record["statement_sha256"],"witness_family":record["witness_family"],"canonicalizer_sources":relocated_source_identity(quality["canonicalizer_source_files"]),"verifier_support_sources":relocated_source_identity(quality["verifier_support_files"])}
    require(contract==origin["verification_contract"] and object_sha(contract)==origin["verification_contract_sha256"], "replay equivalence contract hash mismatch")
    reuse=read_json(root/"replay_reuse_manifest.json")
    require(sha_file(root/"replay_reuse_manifest.json")==quality["replay_reuse_manifest_sha256"], "replay reuse manifest hash mismatch")
    require(reuse["schema"]=="constructive-code-exact-reference-replay-reuse-v1" and reuse["policy_model_outputs_used"] is False, "invalid replay reuse policy")
    require(reuse["initial_manifest_sha256"]==origin["manifest_sha256"] and reuse["initial_quality_sha256"]==origin["hardening_quality_sha256"], "replay reuse initial source mismatch")
    sealing=quality["post_replay_pool_sealing"];review=read_json(root/"independent_replay_reuse_audit.json")
    require(sha_file(root/"independent_replay_reuse_audit.json")==sealing["independent_reuse_audit_sha256"], "independent replay reuse review hash mismatch")
    reviews=unique(review["tasks"],lambda row:row["task_id"])
    require(review["status"]=="pass" and review["initial_manifest_sha256"]==origin["manifest_sha256"] and review["replay_reuse_manifest_sha256"]==quality["replay_reuse_manifest_sha256"], "independent replay reuse review origin mismatch")
    require(set(reviews)==set(reuse["reused_problem_ids"]) and len(reuse["reused_problem_ids"])==len(reviews)==sealing["reused_initial_source_panels"], "replay reuse panel identity bijection mismatch")
    require(sealing["fresh_source_panels"]+len(reviews)==quality["source_candidates"], "replay reuse/fresh panel denominator mismatch")
    reviewed=reviews[record["source_problem_id"]]
    for field,value in (("old_task_record_sha256",old["task_record_sha256"]),("new_task_record_sha256",record["task_record_sha256"]),("admission_audit_sha256",sha_file(directory/"admission_audit.json")),("reference_replays_sha256",admitted["replays_sha256"]),("task_adapter",record["task_adapter"]),("problem_key",record["problem_key"]),("witness_family",record["witness_family"])):
        require(reviewed[field]==value, "independent replay reuse per-task binding mismatch: "+field)
    require(reviewed["original_audit_counts_status_violations_preserved"] is True, "independent review does not affirm preserved verdicts")
    return {"status":"pass","origin_manifest_sha256":origin["manifest_sha256"],"reference_replays_sha256":origin["reference_replays_sha256"],"verification_contract_sha256":origin["verification_contract_sha256"],"post_reuse_validation":True}


def code_records(config, dataset_identity):
    if config.get("adapter_module") not in {"build_constructive_code_wider_20260921", "build_constructive_code_hardened_20260921"}:
        return None
    root = Path(config["adapter_config"]["slate_root"])
    manifest = read_json(root/"manifest.json")
    require_unrevoked_code_manifest(dataset_identity["manifest_sha256"])
    require(sha_file(root/"manifest.json") == dataset_identity["manifest_sha256"], "coding frozen manifest hash mismatch")
    require(manifest["status"] == "audited" and manifest["tasks_sha256"] == object_sha(manifest["tasks"]), "coding admission manifest integrity mismatch")
    require(manifest.get("hardening_schema") == "constructive-code-hardened-20260921-v1", "coding readiness requires the source-audit hardened verifier; historical suites are insufficient")
    quality, quality_items, probes = hardening_quality(root,manifest,dataset_identity)
    audit_hardening_sources(root,quality)
    by_id = {r["source_problem_id"]:r for r in manifest["tasks"]}
    wanted = config.get("task_ids", config.get("train_ids",[])+config.get("eval_ids",[]))
    records = {}
    for task in wanted:
        require(task in manifest["admitted_problem_ids"], "coding task not admitted")
        item = by_id[task]; directory = root/item["relative_path"]
        record = read_json(directory/"task.json")
        require(record["task_record_sha256"] == item["task_record_sha256"] == object_sha({k:v for k,v in record.items() if k != "task_record_sha256"}), "coding task record integrity mismatch")
        require(record["source_problem_id"]==task and record["split"]==item.get("split",record["split"]), "coding task index/source split mismatch")
        require(sha_file(directory/"admission_audit.json") == record["admission_audit_sha256"], "coding admission audit hash mismatch")
        admitted = read_json(directory/"admission_audit.json")
        require(admitted["status"] == "pass" and not admitted["violations"], "nonpassing coding admission audit")
        require(admitted["negative_replays"] == 12 and admitted["tnr"] == 1.0, "hardened coding admission must reject all twelve human-negative programs")
        require(admitted["positive_replays"] == 12 and admitted["tpr"] == 1.0, "hardened coding admission must accept all twelve human-positive programs")
        audit_hardening_task(root,directory,record,admitted,quality,quality_items[task],probes)
        audit_hardened_reference_ledger(directory,record,admitted)
        records[task] = record
    return records


def inspect_verdict(text, verdict, task_id, qa=None, code_contract=None):
    validate_verdict(verdict)
    receipt = verdict["receipt"]
    if qa is not None:
        require(task_id in qa, "unknown QA task")
        row = qa[task_id]
        choices = {o["display"]: o for o in row["options"]}
        label = text.strip()
        option = choices.get(label) if re.fullmatch(r"[A-Z]", label) else None
        accepted = option is not None and option["label"] == 1
        key = option["id"] if accepted else None
        require((verdict["accepted"], verdict["canonical_key"]) == (accepted, key), "QA receipt disagrees with independently regraded native labels")
        require(receipt.get("response_sha256") == sha_bytes(text.encode()), "QA receipt/raw text mismatch")
        return
    if code_contract is not None:
        record = code_contract[task_id]
        for key, expected in (("source_problem_id", task_id), ("checker_sha256", record["checker_sha256"]), ("suite_id", record["suite_id"]), ("suite_sha256", record["suite"]["suite_sha256"])):
            require(receipt.get(key) == expected, "code receipt differs from frozen checker/suite/task: "+key)
        if verdict["accepted"]:
            require(receipt["execution"]["suite_tests"] == record["suite"]["test_count"], "accepted code suite size differs from frozen data")
    require(receipt.get("emitted_text_sha256") == sha_bytes(text.encode()), "code receipt/raw text mismatch")
    code, stripped = text, False
    for opening in ("```python\n", "```python3\n", "```\n"):
        if text.startswith(opening) and text.endswith("\n```"):
            code, stripped = text[len(opening):-4], True
            break
    require(receipt.get("executed_source_sha256") == sha_bytes(code.encode()), "executed code/source hash mismatch")
    require(receipt.get("fence_stripped") == stripped, "code extraction rule mismatch")
    require(receipt.get("accepted") == verdict["accepted"] and receipt.get("canonical_key") == verdict["canonical_key"], "outer/code verdict disagreement")
    if verdict["accepted"]:
        require(receipt.get("stability_recheck_required") is True, "accepted code lacks stability check")
        for replay in (receipt, receipt.get("stability_recheck") or {}):
            require(replay.get("accepted") is True and replay.get("canonical_key") == verdict["canonical_key"], "code recheck changed acceptance or mode")
            require(not replay.get("hard_violations") and replay.get("terminal_worker_record") is True, "unclean code recheck")
            require(replay.get("released_checker_accepted") is True and replay.get("wrapper_accepted") is True, "code checker agreement missing")
            execution = replay.get("execution") or {}
            require(type(execution.get("suite_tests")) is int and execution["suite_tests"] > 0 and execution.get("executed_tests") == execution["suite_tests"], "code did not execute full suite")
            require(replay.get("executed_source_sha256") == sha_bytes(code.encode()), "recheck source hash changed")
            for field in ("source_problem_id", "checker_sha256", "suite_id", "suite_sha256"):
                require(replay.get(field) == receipt.get(field), "stability recheck identity drift: "+field)


def local_codec(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    return tokenizer


def check_tokens(row, codec, prompt=None, prompt_tokens=None):
    require(isinstance(row.get("token_ids"), list) and all(type(t) is int and t >= 0 for t in row["token_ids"]), "invalid raw response token IDs")
    if "token_count" in row:
        require(row["token_count"] == len(row["token_ids"]), "response token count mismatch")
    if codec is not None:
        decoded = codec.decode(row["token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        require(decoded == row["text"], "raw response tokens do not decode to emitted text")
        if prompt is not None and prompt_tokens is not None:
            require(codec.encode(prompt, add_special_tokens=False) == prompt_tokens, "raw prompt tokens do not encode frozen prompt")


def artifact_path(descriptor, base):
    path = Path(descriptor["path"])
    path = path if path.is_absolute() else base/path
    require(sha_file(path) == descriptor["sha256"], f"sidecar hash mismatch: {path.name}")
    return path


def audit_frozen_sources(directory, receipt, training=False):
    """Check implementation and resolved config bytes against launch/run seals."""
    directory = Path(directory)
    root = directory.parent if directory.name == "training" else directory
    launcher_path = root/"identity.json"
    if not launcher_path.exists():
        return "unknown"
    launcher = read_json(launcher_path)
    if launcher.get("schema") != "real-domains-frozen-job-20260921-v1":
        return "unknown"
    config_path = root/"config.json"
    require(sha_file(config_path) == launcher["config_sha256"], "frozen launcher config hash mismatch")
    if not training:
        require(receipt["config_sha256"] == sha_file(config_path), "evaluation config seal mismatch")
    files = {str(Path(f["snapshot"]).resolve()):f["sha256"] for f in launcher["files"]}
    def verify(path,digest):
        path=Path(path)
        require(sha_file(path) == digest == files[str(path.resolve())], "frozen implementation hash mismatch: "+path.name)
    runner=root/"bundle/ops"/("train_real_domains_pilot_20260921.py" if training else "evaluate_real_domains_20260921.py")
    verify(runner,receipt["runner_sha256"])
    raw_config = read_json(config_path)
    if training:
        assignments = [n for n in ast.parse(runner.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == "DEFAULTS" for t in n.targets)]
        require(len(assignments) == 1, "frozen training defaults not uniquely declared")
        raw_config = {**ast.literal_eval(assignments[0].value), **raw_config}
    require(raw_config == receipt["config"], "resolved configuration differs from frozen launch file and sealed defaults")
    module=receipt["config"]["adapter_module"]
    adapter=root/"bundle"/("src" if "." in module else "ops")/(module.replace(".","/")+".py")
    verify(adapter,receipt["adapter_module_sha256" if training else "adapter_sha256"])
    for module,digest in receipt.get("production_source_sha256",{}).items():
        verify(root/"bundle/src"/(module.replace(".","/")+".py"),digest)
    require(sha_file(Path(receipt["config"]["model"])/"config.json") == receipt["model_config_sha256"], "base model configuration hash mismatch")
    return "pass"


def audit_evaluation(path, codec=None):
    path = Path(path)
    if path.is_dir():
        path = path/"evaluation.json"
    receipt = read_json(path)
    require(receipt.get("status") == "complete", "evaluation is not complete")
    config = receipt["config"]
    frozen_sources = audit_frozen_sources(path.parent, receipt)
    require(receipt.get("dataset_identity_sha256") == object_sha(receipt["dataset_identity"]), "evaluation dataset identity hash mismatch")
    responses = unique(read_rows(artifact_path(receipt["artifacts"]["responses"], path.parent)), lambda r: (r["task_id"], r["sample_index"]))
    attempts = unique(read_rows(artifact_path(receipt["artifacts"]["attempts"], path.parent)), lambda r: (r["task_id"], r["sample_index"]))
    ids, n = config["task_ids"], config["samples_per_task"]
    require(len(ids) == len(set(ids)), "duplicate configured evaluation task")
    expected = {(task, i) for task in ids for i in range(n)}
    require(set(responses) == set(attempts) == expected, "response/attempt bijection or requested denominator mismatch")
    prompts = unique(receipt["task_prompts"], lambda r: r["task_id"])
    require(set(prompts) == set(ids), "task prompt selection mismatch")
    qa = qa_records(config)
    code_contract = code_records(config, receipt["dataset_identity"])
    require(qa is not None or code_contract is not None, "unsupported domain adapter; verifier quality cannot be established")
    grouped = defaultdict(list)
    for key, raw in responses.items():
        attempt, prompt = attempts[key], prompts[key[0]]
        require(sha_bytes(prompt["prompt"].encode()) == prompt["prompt_sha256"] == raw["prompt_sha256"], "evaluation prompt hash mismatch")
        require(sha_bytes(raw["text"].encode()) == raw["text_sha256"], "response text hash mismatch")
        require(raw["request_seed"] == config["seed"]+10000*ids.index(key[0])+key[1], "evaluation seed mismatch")
        for name in ("task_id", "sample_index", "request_seed", "prompt_sha256", "text_sha256", "token_count", "finish_reason"):
            require(attempt[name] == raw[name], f"response/attempt binding mismatch: {name}")
        check_tokens(raw, codec)
        inspect_verdict(raw["text"], attempt, key[0], qa, code_contract)
        grouped[key[0]].append(attempt)
    per_task = {task: metrics(grouped[task], qa[task]["gold_topic_ids"] if qa else None) for task in ids}
    reported = unique(receipt["task_results"], lambda r: r["task_id"])
    require(set(reported) == set(per_task), "reported task coverage mismatch")
    for task, recomputed in per_task.items():
        for name in ("samples", "accepted", "accuracy", "mode_counts", "pcmd_eligible", "pcmd", "expected_distinct_valid_modes_at_k", "pass_at_k"):
            require(close(recomputed[name], reported[task][name]), f"reported evaluation metric mismatch: {task}/{name}")
    metadata = {}
    for task in ids:
        metadata[task] = {k: prompts[task].get(k, "unknown") for k in ("split", "family")}
        if qa is not None:
            require(prompts[task].get("split", qa[task].get("split")) == qa[task].get("split"), "QA prompt split differs from frozen record")
            if qa[task].get("split")=="test":
                require(config["adapter_config"].get("allow_test") is True, "QA test evaluation lacks allow_test")
        if code_contract is not None:
            require(prompts[task].get("split")==code_contract[task]["split"], "coding prompt split differs from frozen record")
            if code_contract[task]["split"] in {"test","heldout"}:
                require(config["adapter_config"].get("allow_heldout") is True, "coding test evaluation lacks allow_heldout")
    by_split = {split: aggregate({t: per_task[t] for t in ids if metadata[t]["split"] == split}) for split in sorted({v["split"] for v in metadata.values()})}
    return {"status": "pass", "kind": "capability_only", "domain": "qa" if qa is not None else "code", "frozen_implementation_binding": frozen_sources, "receipt": str(path), "receipt_sha256": sha_file(path), "job_id": receipt.get("job_id"), "task_metrics": per_task, "task_metadata": metadata, "summary": aggregate(per_task), "by_split": by_split, "token_text_binding": "pass" if codec else "unknown", "correctness_evidence": "native QA labels regraded" if qa else "sealed checker and independent full-suite stability receipts; programs not reexecuted by auditor"}


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (float, int)) and isinstance(b, (float, int)) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    return a == b


def training_dir(path):
    path = Path(path)
    return path/"training" if (path/"training").is_dir() else path


def _normalize_config_once(config, directory):
    """Reverse one launcher's explicitly recorded frozen-data path rewrites."""
    root = directory.parent if directory.name == "training" else directory
    launcher = root/"identity.json"
    replacements = []
    if launcher.exists():
        frozen = read_json(launcher)
        if frozen.get("schema") == "real-domains-frozen-job-20260921-v1":
            for i, original in enumerate(frozen["request"].get("freeze_data_roots", [])):
                replacements.append((str((root/"bundle/data"/str(i)).resolve()), str(Path(original).resolve())))
    def visit(value):
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        if isinstance(value, str):
            for source, destination in replacements:
                if value == source or value.startswith(source+"/"):
                    return destination+value[len(source):]
        return value
    return visit(config)


def normalized_config(config,directory,*,chain_directories=()):
    """Resolve only explicit freeze maps from independently verified launch contexts."""
    directories=list(dict.fromkeys(str(Path(d).resolve()) for d in (directory,*chain_directories)))
    current=json.loads(json.dumps(config));seen={object_sha(current)}
    while True:
        changed=False
        for context in directories:
            candidate=_normalize_config_once(current,Path(context))
            if candidate==current:
                continue
            fingerprint=object_sha(candidate)
            require(fingerprint not in seen, "cyclic frozen-data path provenance")
            seen.add(fingerprint);current=candidate;changed=True
        if not changed:
            return current
        require(len(seen)<=1+len(directories)*(len(directories)+1), "frozen-data path provenance does not converge")


PINNED_RUNTIME_IMAGE_SHA256 = "6d036dfa4a6e216d71e2ddae4cb673c0ff588d2c8ff8af3ed2ab7b3fed309437"
PINNED_RUNTIME_FILES = {
    "usr/lib64/ld-linux-x86-64.so.2":"438c546d8e8cc48496bf3a95f753051afd9db66a629a74e31a9ded71586b56e0",
    "usr/local/bin/python3.10":"590a8c6d6f33dd13991b43285f0acb8999f4ce338eacbdda2faec0f25ca2a0b6",
    "usr/local/lib/libpython3.10.so.1.0":"988df48b3ba1c6e2dec55b332c046fe3fec744f487b05cb686d08ed6336f2ab5",
}
CODE_LOCATION_FIELDS = ("build_root","runtime_root","scratch_root","launcher")


def audit_code_runtime_locations(config):
    """Bind executable bytes before discounting role-specific working locations."""
    adapter=config["adapter_config"];slate=Path(adapter["slate_root"])
    manifest=read_json(slate/"manifest.json")
    require(manifest["tasks_sha256"]==object_sha(manifest["tasks"]), "runtime task index hash mismatch")
    items=unique(manifest["tasks"],lambda row:row["source_problem_id"])
    wanted=config.get("task_ids",config.get("train_ids",[])+config.get("eval_ids",[]))
    require(wanted and wanted==adapter["problem_ids"], "runtime executable selection differs from configured tasks")
    launcher_hash=sha_file(adapter["launcher"])
    require(sha_file(adapter["image"])==PINNED_RUNTIME_IMAGE_SHA256, "runtime image bytes differ from pinned image")
    actual_runtime={name:sha_file(Path(adapter["runtime_root"])/name) for name in PINNED_RUNTIME_FILES}
    require(actual_runtime==PINNED_RUNTIME_FILES, "runtime executable bytes differ from pinned runtime")
    require(all(Path(adapter[field]).is_dir() for field in ("build_root","runtime_root","scratch_root")), "coding work location is not a directory")
    checkers={}
    for task in wanted:
        item=items[task];directory=slate/item["relative_path"];record=read_json(directory/"task.json")
        require(record["task_record_sha256"]==item["task_record_sha256"]==object_sha({k:v for k,v in record.items() if k!="task_record_sha256"}), "runtime task record hash mismatch")
        require(sha_file(directory/"admission_audit.json")==record["admission_audit_sha256"], "runtime admission hash mismatch")
        admitted=read_json(directory/"admission_audit.json")
        require(admitted["status"]=="pass" and admitted["runtime"]["launcher_sha256"]==launcher_hash, "actual launcher differs from admitted executable")
        require(admitted["runtime"]["runtime"]["image_sha256"]==PINNED_RUNTIME_IMAGE_SHA256 and admitted["runtime"]["runtime"]["critical_file_sha256"]==actual_runtime, "admitted/runtime identity disagreement")
        checkers[task]=sha_file(Path(adapter["build_root"])/task.lower()/"checker")
        require(checkers[task]==admitted["checker_build"]["binary_sha256"] and admitted["checker_build"]["source_sha256"]==record["checker_sha256"], "actual checker differs from admitted executable")
    return {"launcher_sha256":launcher_hash,"image_sha256":PINNED_RUNTIME_IMAGE_SHA256,"critical_runtime_sha256":actual_runtime,"checker_binary_sha256":checkers}


def comparison_config(config,directory,*,chain_directories=()):
    normalized=normalized_config(config,directory,chain_directories=chain_directories)
    if config.get("adapter_module")=="build_constructive_code_hardened_20260921":
        require("_verified_code_execution" not in config, "reserved audit-only configuration field supplied")
        normalized["_verified_code_execution"]=audit_code_runtime_locations(config)
        normalized["adapter_config"]={**normalized["adapter_config"]}
        for field in CODE_LOCATION_FIELDS:
            normalized["adapter_config"][field]="<verified-role-location:"+field+">"
    return normalized


def endpoint_stratum(domain,split,trained,*,allow_reserved=False):
    require(domain in {"code","qa"}, "endpoint includes unknown domain")
    if split in {"test","heldout"}:
        require(not trained, "reserved test overlaps training IDs")
        require(allow_reserved and (domain=="code" or split=="test"), "reserved test requires validated cohort permission")
        return "reserved_test"
    if domain=="code" and split=="development":
        return "trained_development" if trained else "untrained_development"
    if domain=="code" and split=="validation":
        require(not trained, "validation prompt overlaps training IDs")
        return "validation"
    require(split in ({"train","dev"} if domain=="qa" else {"train"}), "endpoint includes unknown stratum")
    require(not trained or split=="train", "trained prompt split contradiction")
    return "trained_train" if trained else "untrained_train" if split=="train" else "dev"


def endpoint_cohort(config,metadata,train_ids,domain):
    """Admit complete reserved-test, complete trained-support, or frozen pilot cohorts."""
    ids=set(metadata);train=set(train_ids);adapter=config["adapter_config"]
    require(ids and train, "empty endpoint or training cohort")
    if domain=="qa":
        records=qa_records(config)
        require(records is not None, "QA cohort adapter mismatch")
        source_splits={key:row.get("split") for key,row in records.items()}
    elif domain=="code":
        manifest=read_json(Path(adapter["slate_root"])/"manifest.json")
        require(manifest["tasks_sha256"]==object_sha(manifest["tasks"]), "endpoint cohort manifest hash mismatch")
        admitted=set(manifest["admitted_problem_ids"])
        source_splits={}
        for row in manifest["tasks"]:
            if row["source_problem_id"] not in admitted or row["status"]!="admitted":
                continue
            if "split" in row:
                split=row["split"]
            else:
                record=read_json(Path(adapter["slate_root"])/row["relative_path"]/"task.json")
                require(record["task_record_sha256"]==row["task_record_sha256"]==object_sha({k:v for k,v in record.items() if k!="task_record_sha256"}), "endpoint cohort task record hash mismatch")
                require(record["source_problem_id"]==row["source_problem_id"], "endpoint cohort task identity mismatch")
                split=record["split"]
            source_splits[row["source_problem_id"]]=split
        require(set(source_splits)==admitted, "endpoint cohort admission status mismatch")
    else:
        raise AuditError("endpoint includes unknown domain")
    require(ids<=set(source_splits) and train<=set(source_splits), "endpoint or training task absent from frozen source")
    require(all(source_splits[t] in {"train","development"} for t in train), "reserved or validation source task used for training")
    require(all(metadata[t]["split"]==source_splits[t] for t in ids), "endpoint cohort label differs from frozen source")
    reserved={t for t,split in source_splits.items() if split in {"test","heldout"}}
    if ids&reserved:
        require(ids==reserved and not ids&train, "reserved primary must contain the full frozen test set and no training IDs")
        if domain=="qa":
            splits=adapter.get("splits",["train","dev"]);splits=[splits] if isinstance(splits,str) else splits
            require(adapter.get("allow_test") is True and "test" in splits, "QA reserved primary requires allow_test and test split")
        else:
            require(adapter.get("allow_heldout") is True, "coding reserved primary requires allow_heldout")
            if "admitted_test_problem_ids" in manifest:
                require(reserved==set(manifest["admitted_test_problem_ids"]), "reserved test manifest membership mismatch")
        return "reserved_test_primary"
    if ids==train:
        return "trained_support_diagnostic"
    require(train<=ids, "non-test endpoint omits trained prompts")
    if domain=="qa":
        dev={t for t,split in source_splits.items() if split=="dev"}
        require(ids==train|dev and dev, "QA mixed pilot must retain all trained and all dev prompts")
    else:
        development={t for t,split in source_splits.items() if split=="development"}
        require(ids==development and train<=development, "coding mixed pilot must retain the complete admitted development cohort")
    return "pilot_mixed_development"


def audit_training_diagnostics(rows,result,group_size):
    """Recompute binary advantage support and report measured resources."""
    active=mixed_positive=0
    stages={key:0.0 for key in ("generation_seconds","verification_seconds","learning_seconds","total_update_seconds")}
    for row in rows:
        successes=row["fresh_successes"]
        mixed=0<successes<group_size
        expected_min=-1.0 if mixed else 0.0
        expected_max=group_size/successes-1.0 if mixed else 0.0
        for key,expected in (("adv_min",expected_min),("adv_max",expected_max),("adv_mean",0.0)):
            value=row[key]
            require(isinstance(value,(int,float)) and math.isfinite(value) and math.isclose(value,expected,rel_tol=1e-6,abs_tol=1e-6), "fresh binary MaxRL advantage diagnostic mismatch: "+key)
        gradient=row["actual_gradient_norm"]
        require(isinstance(gradient,(int,float)) and math.isfinite(gradient) and gradient>=0, "nonfinite actual training gradient")
        active+=mixed
        mixed_positive+=mixed and gradient>0
        for key in stages:
            value=row[key]
            require(isinstance(value,(int,float)) and math.isfinite(value) and value>=0, "invalid training stage duration")
            stages[key]+=value
    allocated=result["peak_gpu_allocated_bytes"];reserved=result["peak_gpu_reserved_bytes"]
    require(type(allocated) is int and type(reserved) is int and 0<allocated<=reserved, "invalid peak GPU memory receipt")
    hours=result["allocated_gpu_hours_during_runner"]
    require(isinstance(hours,(int,float)) and math.isfinite(hours) and hours*3600>=stages["total_update_seconds"], "runner duration shorter than measured updates")
    return {"status":"pass","groups_with_nonzero_recomputed_fresh_advantages":active,"mixed_groups_with_positive_actual_gradient":mixed_positive,"interpretation":"Positive actual gradients in MaxRL, whose replay gradient is zero, establish active fresh updates; Re:Max actual gradients include replay.","stage_seconds":stages,"mean_update_seconds":stages["total_update_seconds"]/len(rows),"runner_gpu_hours":hours,"peak_gpu_allocated_bytes":allocated,"peak_gpu_reserved_bytes":reserved,"accounting_note":"Runner duration excludes allocation overhead; scheduler parent allocation is authoritative for the budget."}


def audit_training(path, codec=None):
    directory = training_dir(path)
    identity, result = read_json(directory/"identity.json"), read_json(directory/"result.json")
    config, arm = identity["config"], identity["arm"]
    frozen_sources = audit_frozen_sources(directory,identity,training=True)
    require(arm in {"maxrl", "remax"}, "unrecognized training arm")
    require(result.get("status") == "complete", "training arm is not complete")
    require(result["arm"] == arm, "result arm mismatch")
    require(identity["config_sha256"] == object_sha(config) == result["config_sha256"], "training config hash mismatch")
    updates = config["updates"]
    require(result["completed_updates"] == updates, "training stopped before requested updates")
    require(identity.get("resume") is None and identity["objective"].get("initial_bank") == "empty", "paired pilot did not start from empty bank")
    require(identity["objective"].get("compute_only") == (arm == "maxrl"), "replay arm registration mismatch")
    require(identity["objective"].get("extra_reward_or_entropy_terms") is False, "unregistered objective terms")
    require(config["train_ids"] and config["eval_ids"] and not set(config["train_ids"]) & set(config["eval_ids"]), "train/eval prompt overlap")
    require(len(config["train_ids"]) == len(set(config["train_ids"])) and len(config["eval_ids"]) == len(set(config["eval_ids"])), "duplicate train/eval IDs")
    candidates = unique(read_rows(directory/"candidates.jsonl"), lambda r: r["request_id"])
    verified = unique(read_rows(directory/"verified.jsonl"), lambda r: r["request_id"])
    expected = {}
    for step in range(updates):
        task = config["train_ids"][step % len(config["train_ids"])]
        for sample in range(config["group_size"]):
            expected[f"train:{step}:{task}:{sample}"] = ("train", step, task, sample, config["seed"]+1000000+step*10000)
    phases = (["initial"] if config["initial_evaluation"] else []) + ["final"]
    for phase in phases:
        step = 0 if phase == "initial" else updates
        for ti, task in enumerate(config["eval_ids"]):
            for sample in range(config["eval_samples"]):
                expected[f"{phase}:{step}:{task}:{sample}"] = (phase, step, task, sample, config["eval_seed"]+ti*10000)
    require(set(candidates) == set(verified) == set(expected), "candidate/verified/request bijection mismatch")
    tasks = unique(identity["tasks"], lambda r: r["task_id"])
    require(set(tasks) == set(config["train_ids"]+config["eval_ids"]), "training prompt identity list mismatch")
    qa, grouped, prompt_tokens = qa_records(config), defaultdict(list), {}
    code_contract = code_records(config, identity.get("dataset_identity", {}))
    require(qa is not None or code_contract is not None, "unsupported training adapter; verifier quality cannot be established")
    source_tasks=qa if qa is not None else code_contract
    for task in config["train_ids"]:
        split=source_tasks[task].get("split")
        require(split in {"train","development"} or split is None and frozen_sources=="unknown", "reserved or validation source task used for training")
    for task,row in tasks.items():
        split=source_tasks[task].get("split")
        if split is not None:
            require(row["split"]==split, "training task split differs from frozen source")
    for request, raw in candidates.items():
        ver = verified[request]
        require({k:v for k,v in ver.items() if k != "verdict"} == raw, "verified sample differs from original raw candidate")
        phase, step, task, sample, seed = expected[request]
        require((raw["phase"],raw["step"],raw["task_id"],raw["sample_index"]) == (phase,step,task,sample), "request field mismatch")
        require(raw["batch_seed"] == seed+(sample//config["generation_batch_size"])*config["generation_batch_size"] and raw["batch_offset"] == sample%config["generation_batch_size"], "training/evaluation sampling seed mismatch")
        require(raw["prompt_sha256"] == tasks[task]["prompt_sha256"], "training prompt hash mismatch")
        require(len(raw["prompt_token_ids"]) == tasks[task]["prompt_tokens"], "training prompt token count mismatch")
        require(all(type(t) is int and t >= 0 for t in raw["prompt_token_ids"]), "invalid prompt tokens")
        if task in prompt_tokens:
            require(prompt_tokens[task] == raw["prompt_token_ids"], "prompt token IDs changed within task")
        prompt_tokens[task] = raw["prompt_token_ids"]
        if codec:
            rendered = codec.decode(raw["prompt_token_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            require(sha_bytes(rendered.encode()) == raw["prompt_sha256"], "raw prompt tokens do not bind rendered prompt")
        require(0 < len(raw["token_ids"]) <= config["max_new_tokens"], "training response exceeds frozen token cap")
        check_tokens(raw, codec)
        inspect_verdict(raw["text"], ver["verdict"], task, qa, code_contract)
        grouped[(phase, task)].append(ver["verdict"])
    rows = read_rows(directory/"metrics.jsonl")
    require([r["completed_updates"] for r in rows] == list(range(1, updates+1)), "missing/duplicate/reordered optimizer updates")
    entries, cursor, replay_positive = {}, 0, 0
    for step, metric in enumerate(rows):
        task = config["train_ids"][step%len(config["train_ids"])]
        samples = [verified[f"train:{step}:{task}:{i}"] for i in range(config["group_size"])]
        added = 0
        for sample in samples:
            verdict = sample["verdict"]
            if not verdict["accepted"]:
                continue
            entry = entries.setdefault(task, {"prompt": sample["prompt_token_ids"], "modes": {}})
            key = verdict["canonical_key"]
            if key in entry["modes"]:
                entry["modes"][key]["fresh_count"] += 1
            elif len(entry["modes"]) < config["replay_capacity"]:
                entry["modes"][key] = {"token_ids": sample["token_ids"], "fresh_count": 1, "request_id": sample["request_id"], "text_sha256": sha_bytes(sample["text"].encode())}
                added += 1
        bank_tasks = sorted(entries)
        replay_task = bank_tasks[cursor%len(bank_tasks)] if bank_tasks else None
        if bank_tasks:
            cursor += 1
        replay_modes = len(entries[replay_task]["modes"]) if replay_task else 0
        successes = sum(s["verdict"]["accepted"] for s in samples)
        checks = {"task_id": task, "optimizer_updates": step+1, "fresh_successes": successes, "fresh_tokens": sum(len(s["token_ids"]) for s in samples), "mixed_group": int(0 < successes < len(samples)), "new_modes": added, "bank_tasks": len(entries), "bank_modes": sum(len(e["modes"]) for e in entries.values()), "replay_task_id": replay_task, "replay_modes": replay_modes}
        for key, expected_value in checks.items():
            require(metric.get(key) == expected_value, f"training/replay metric mismatch at step {step}: {key}")
        gradient = metric.get("canonical_replay_applied_score_gradient_l2")
        require(isinstance(gradient, (int,float)) and math.isfinite(gradient) and gradient >= 0, "missing/nonfinite replay gradient diagnostic")
        require(gradient == 0 if arm == "maxrl" or not replay_modes else gradient > 0, "replay gradient applied to wrong arm or absent from treatment")
        replay_positive += gradient > 0
    checkpoint = directory/f"checkpoint-{updates}"
    seal = read_json(checkpoint/"complete.json")
    require(seal["arm"] == arm and seal["config_sha256"] == identity["config_sha256"] and seal["completed_updates"] == updates, "checkpoint arm/config/update seal mismatch")
    require(seal["bank_sha256"] == sha_file(checkpoint/"bank.json"), "checkpoint bank hash mismatch")
    require(seal["training_state_sha256"] == sha_file(checkpoint/"training.pt"), "checkpoint optimizer/RNG state hash mismatch")
    require(bool(seal["adapter_files"]), "checkpoint has no adapter weights")
    actual_files = {p.relative_to(checkpoint/"adapter").as_posix() for p in (checkpoint/"adapter").rglob("*") if p.is_file()}
    require(actual_files == set(seal["adapter_files"]), "checkpoint adapter file set mismatch")
    for name, digest in seal["adapter_files"].items():
        require(sha_file(checkpoint/"adapter"/name) == digest, "checkpoint adapter content mismatch")
    require(read_json(checkpoint/"bank.json") == {"capacity": config["replay_capacity"], "entries": entries, "cursor": cursor}, "bank does not reconstruct from first raw verified policy exemplars")
    evaluations = {}
    for phase in phases:
        per_task = {task: metrics(grouped[(phase,task)], qa[task]["gold_topic_ids"] if qa else None) for task in config["eval_ids"]}
        phase_receipt = read_json(directory/f"{phase}.json")
        require(phase_receipt["completed_updates"] == (0 if phase == "initial" else updates), "evaluation checkpoint step mismatch")
        reported = unique(phase_receipt["tasks"], lambda r:r["task_id"])
        require(set(reported) == set(per_task), "training evaluation task mismatch")
        for task, recalc in per_task.items():
            for key in ("samples", "accepted", "mode_counts", "pcmd", "pcmd_eligible"):
                require(close(recalc[key], reported[task][key]), f"training phase metric mismatch: {phase}/{task}/{key}")
            for k, value in recalc["expected_distinct_valid_modes_at_k"].items():
                if k == "1":
                    continue
                require(close(value, reported[task][f"distinct_valid_at_{k}"]), "reported expected distinct metric mismatch")
        require(phase_receipt["tasks"] == result["evaluation"][phase], "final result differs from phase sidecar")
        evaluations[phase] = {"task_metrics": per_task, "summary": aggregate(per_task)}
    submission = directory.parent/"submission.json" if directory.name == "training" else directory/"submission.json"
    job_id = read_json(submission).get("job_id") if submission.exists() else None
    return {"status": "pass", "domain": "qa" if qa is not None else "code", "frozen_implementation_binding": frozen_sources, "job_id": job_id, "directory": str(directory), "identity_sha256": sha_file(directory/"identity.json"), "result_sha256": sha_file(directory/"result.json"), "arm": arm, "completed_updates": updates, "initial_trainable_parameters_sha256": identity.get("initial_trainable_parameters_sha256"), "bank_reconstructed_from_empty": True, "bank_modes": {task: sorted(entry["modes"]) for task, entry in entries.items()}, "train_ids": config["train_ids"], "mixed_groups": sum(row["mixed_group"] for row in rows), "all_correct_groups": sum(row["fresh_successes"] == config["group_size"] for row in rows), "all_wrong_groups": sum(row["fresh_successes"] == 0 for row in rows), "replay_gradient_positive_updates": replay_positive, "final_checkpoint_seal_sha256": sha_file(checkpoint/"complete.json"), "token_text_binding": "pass" if codec else "unknown", "optimizer_state": "sealed bytes verified; pickle intentionally not deserialized", "training_diagnostics": audit_training_diagnostics(rows,result,config["group_size"]) if frozen_sources == "pass" else {"status":"unknown"}, "evaluation": evaluations, "_identity": identity, "_config": comparison_config(config, directory)}


def audit_pair(maxrl_dir, remax_dir, codecs=None):
    codecs = codecs or (None, None)
    arms = [audit_training(p,c) for p,c in zip((maxrl_dir,remax_dir), codecs)]
    control, treatment = arms
    require([a["arm"] for a in arms] == ["maxrl", "remax"], "pair arm ordering mismatch")
    require(control["_config"] == treatment["_config"], "paired configurations differ beyond proven frozen paths and byte-bound runtime locations")
    require(control["initial_trainable_parameters_sha256"] and control["initial_trainable_parameters_sha256"] == treatment["initial_trainable_parameters_sha256"], "initial LoRA parameter hashes differ or are absent")
    for key in ("tasks", "production_source_sha256", "runner_sha256", "adapter_module_sha256", "model_config_sha256"):
        require(control["_identity"].get(key) == treatment["_identity"].get(key), f"paired identity mismatch: {key}")
    comparison = paired_effects(control["evaluation"]["final"]["task_metrics"], treatment["evaluation"]["final"]["task_metrics"])
    for arm in arms:
        arm.pop("_identity"); arm.pop("_config")
    return {"status": "pass", "domain": control["domain"], "frozen_implementation_binding": "pass" if all(a["frozen_implementation_binding"] == "pass" for a in arms) else "unknown", "paired_summary": comparison["paired_summary"], "arms": arms, "paired_final_remax_minus_maxrl": comparison["per_task"], "treatment_efficacy": "pilot estimates only; no confirmatory superiority claim", "token_text_binding": "pass" if all(a["token_text_binding"] == "pass" for a in arms) else "unknown"}


def paired_effects(control, treatment):
    require(set(control) == set(treatment), "paired metric task IDs differ")
    deltas = {}
    for task, c in control.items():
        t = treatment[task]
        d = {"accuracy": t["accuracy"]-c["accuracy"], "distinct_valid_modes": t["distinct_valid_modes"]-c["distinct_valid_modes"], "expected_distinct_at_k": {k:t["expected_distinct_valid_modes_at_k"][k]-v for k,v in c["expected_distinct_valid_modes_at_k"].items()}}
        d["pcmd_both_eligible"] = c["pcmd_eligible"] and t["pcmd_eligible"]
        d["pcmd"] = t["pcmd"]-c["pcmd"] if d["pcmd_both_eligible"] else None
        if "annotated_topic_coverage" in c:
            d["annotated_topic_coverage"] = t["annotated_topic_coverage"]-c["annotated_topic_coverage"]
        deltas[task] = d
    def effect(values):
        if not values:
            return {"prompts": 0, "mean": None, "paired_prompt_bootstrap_95_percentile": None}
        rng = random.Random(20260921)
        means = sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(2000))
        return {"prompts": len(values), "mean": sum(values)/len(values), "paired_prompt_bootstrap_95_percentile": [means[49], means[1949]], "scope": "descriptive prompt bootstrap from one paired seed; not seed-level uncertainty"}
    summary = {"accuracy": effect([d["accuracy"] for d in deltas.values()]), "pcmd_common_eligible": effect([d["pcmd"] for d in deltas.values() if d["pcmd"] is not None])}
    for k in ("8", "32"):
        summary["expected_distinct_at_"+k] = effect([d["expected_distinct_at_k"][k] for d in deltas.values() if k in d["expected_distinct_at_k"]])
    summary["annotated_topic_coverage"] = effect([d["annotated_topic_coverage"] for d in deltas.values() if "annotated_topic_coverage" in d])
    return {"paired_summary": summary, "per_task": deltas}


def audit_endpoints(paths, pair, codecs=None):
    """Compare explicitly supplied base, MaxRL and Re:Max evaluation receipts."""
    require(pair.get("status") == "pass", "endpoint comparison needs independently passing paired training")
    require(len(paths) == 3, "endpoint comparison requires base, MaxRL and Re:Max")
    codecs = codecs or (None, None, None)
    paths = [Path(p)/"evaluation.json" if Path(p).is_dir() else Path(p) for p in paths]
    receipts = [read_json(p) for p in paths]
    evaluated = [audit_evaluation(p,c) for p,c in zip(paths,codecs)]
    labels = ("base", "maxrl", "remax")
    identities, configs = [], []
    chain_directories=[p.parent for p in paths]+[Path(arm["directory"]) for arm in pair["arms"]]
    for i,(path,receipt) in enumerate(zip(paths,receipts)):
        config = receipt["config"]
        # Runtime checks consume actual receipt locations, before any path normalization.
        normalized = comparison_config(config,path.parent,chain_directories=chain_directories)
        if i == 0:
            require(not config.get("lora_path") and "lora_checkpoint" not in receipt, "base endpoint unexpectedly loads LoRA")
        else:
            arm = pair["arms"][i-1]
            require(config["lora_arm"] == labels[i] == arm["arm"], "endpoint arm mismatch")
            checkpoint = Path(config["lora_path"]).parent
            seal = read_json(checkpoint/"complete.json")
            require(sha_file(checkpoint/"complete.json") == arm["final_checkpoint_seal_sha256"] == receipt["lora_checkpoint"]["seal_sha256"], "endpoint did not load the audited final checkpoint")
            require(seal == receipt["lora_checkpoint"]["seal"], "endpoint checkpoint seal receipt mismatch")
            require(seal["completed_updates"] == config["lora_completed_updates"] == arm["completed_updates"] and seal["config_sha256"] == config["lora_checkpoint_config_sha256"], "endpoint update/config identity mismatch")
            for filename,field in (("bank.json","bank_sha256"),("training.pt","training_state_sha256")):
                require(sha_file(checkpoint/filename) == seal[field], "endpoint frozen checkpoint data mismatch")
            files = {p.relative_to(checkpoint/"adapter").as_posix():sha_file(p) for p in (checkpoint/"adapter").rglob("*") if p.is_file()}
            require(files == seal["adapter_files"] == receipt["lora_files"], "endpoint adapter weights mismatch")
            for field in ("lora_path","lora_arm","lora_completed_updates","lora_checkpoint_config_sha256","max_lora_rank"):
                normalized.pop(field,None)
        configs.append(normalized)
        identities.append({k:receipt.get(k) for k in ("task_prompts","runner_sha256","adapter_sha256","model_config_sha256","sampling_vocab_upper_bound","sampling_action_space")})
    require(configs[0] == configs[1] == configs[2], "endpoint sampling/model/data configuration mismatch")
    require(identities[0] == identities[1] == identities[2], "endpoint prompts or policy implementation identities differ")
    tasks = list(evaluated[0]["task_metrics"])
    require(set(pair["arms"][0]["train_ids"]) == set(pair["arms"][1]["train_ids"]), "paired training task selection differs")
    train_ids = set(pair["arms"][0]["train_ids"])
    require(all(e["domain"]==pair["domain"] for e in evaluated), "endpoint and training domains differ")
    cohort=endpoint_cohort(receipts[0]["config"],evaluated[0]["task_metadata"],train_ids,pair["domain"])
    groups = defaultdict(list)
    for task in tasks:
        split = evaluated[0]["task_metadata"][task]["split"]
        groups[endpoint_stratum(pair["domain"],split,task in train_ids,allow_reserved=cohort=="reserved_test_primary")].append(task)
    strata = {}
    for group,ids in groups.items():
        selected = [{t:e["task_metrics"][t] for t in ids} for e in evaluated]
        strata[group] = {"task_ids":ids,"arms":{label:aggregate(m) for label,m in zip(labels,selected)},"remax_minus_maxrl":paired_effects(selected[1],selected[2]),"maxrl_minus_base":paired_effects(selected[0],selected[1]),"remax_minus_base":paired_effects(selected[0],selected[2])}
    per_trained = {}
    for task in tasks:
        if task not in train_ids:
            continue
        entry = {label:e["task_metrics"][task] for label,e in zip(labels,evaluated)}
        for label,arm,e in zip(labels[1:],pair["arms"],evaluated[1:]):
            bank = set(arm["bank_modes"].get(task,[]))
            counts = e["task_metrics"][task]["mode_counts"]
            entry[label]["discovered_bank_modes"] = sorted(bank)
            entry[label]["bank_topics_observed_at_endpoint"] = sorted(bank & set(counts))
            entry[label]["bank_topic_observed_fraction"] = len(bank & set(counts))/len(bank) if bank else None
            entry[label]["bank_topic_response_mass"] = sum(counts.get(k,0) for k in bank)/e["task_metrics"][task]["samples"]
        per_trained[task] = entry
    common_multi_bank = [t for t in train_ids & set(tasks) if all(len(a["bank_modes"].get(t,[])) >= 2 for a in pair["arms"])]
    return {"status":"pass","domain":pair["domain"],"endpoint_cohort":cohort,"kind":"descriptive_three_endpoint_comparison","endpoints":dict(zip(labels,evaluated)),"strata":strata,"trained_prompt_table":per_trained,"common_multi_bank_task_ids":sorted(common_multi_bank),"common_multi_bank_interpretation":"exploratory post-treatment subset only; never the primary effect denominator","replay_support_interpretation":"Only fresh verified modes actually discovered into each arm's bank are replay targets. Other known QA topics were not promised replay coverage; total coding witness support is unknown.","training_regime":"mechanistic ceiling/zero-fresh-gradient smoke" if all(a["mixed_groups"] == 0 for a in pair["arms"]) else "single-seed training pilot","token_text_binding":"pass" if all(e["token_text_binding"] == "pass" for e in evaluated) else "unknown"}


def audit_qa_provenance(path, config):
    """Bind the external original-source audit to every frozen dataset row."""
    path = Path(path); evidence = read_json(path)
    require(evidence["schema"] == "sata-reuters-independent-source-comparison-v3", "source provenance audit version not supported")
    records = qa_records(config); require(records is not None,"source audit requires SATA adapter")
    manifest = read_json(config["adapter_config"]["manifest_path"])
    require(evidence["sata_release_sha256"] == manifest["source"]["raw_sha256"], "source audit/release identity mismatch")
    require(evidence["archive_sha256"] == sha_file(path.parent/"reuters_uci.zip"), "original Reuters archive hash mismatch")
    source_rows = unique(evidence["rows"],lambda r:r["source_index"])
    require(len(source_rows) == evidence["sata_news_rows"] and not evidence["unmatched"] and not evidence["original_gold_used_as_distractor"], "incomplete or contradictory source audit")
    original_splits = defaultdict(set)
    for row in records.values():
        source = source_rows[row["source_row_index"]]
        require(source["paragraph_sha256"] == sha_bytes(row["paragraph"].encode()), "provenance paragraph binding mismatch")
        gold = {o["text"] for o in row["options"] if o["label"] == 1}
        negative = {o["text"] for o in row["options"] if o["label"] == 0}
        require(gold == set(source["sata_gold"]) and negative == set(source["sata_distractors"]), "provenance native topic labels mismatch")
        require(source["match_count"] == len(source["matches"]) > 0, "missing original Reuters match")
        require(any(gold <= set(m["original_topics"]) for m in source["matches"]), "no original match supports all native gold topics")
        for match in source["matches"]:
            require(not negative & set(match["original_topics"]), "original positive topic used as distractor")
            original_splits[match["reuters_id"]].add(row["split"])
    require(all(len(v) == 1 for v in original_splits.values()), "original Reuters article leaks across splits")
    return {"status":"pass","receipt":str(path),"receipt_sha256":sha_file(path),"original_archive_sha256":evidence["archive_sha256"],"sata_release_sha256":evidence["sata_release_sha256"],"frozen_records_sha256":config["adapter_config"]["records_sha256"],"frozen_rows_checked":len(records),"source_news_rows":len(source_rows),"unique_source_matches":evidence["uniquely_matched"],"ambiguous_source_matches":evidence["multiple_matches"],"original_articles_crossing_splits":[],"scope":"External source matching receipt bound to archive/release/frozen rows; source matching algorithm is not rerun by this auditor."}


def accounting(paths):
    jobs, sources = {}, []
    for name in paths:
        path = Path(name); content = path.read_text(); sources.append({"path":str(path), "sha256":sha_file(path)})
        if path.suffix.lower() == ".json":
            data = json.loads(content)
            if isinstance(data, dict) and "stdout" in data:
                rows = list(csv.DictReader(io.StringIO(data["stdout"]), delimiter="|"))
            else:
                rows = data if isinstance(data,list) else [data] if "job_id" in data or "JobIDRaw" in data else data.get("jobs", data.get("records", []))
        else:
            rows = list(csv.DictReader(io.StringIO(content), delimiter="|"))
        require(isinstance(rows, list), "unrecognized provided accounting receipt")
        for row in rows:
            job = str(row.get("JobIDRaw", row.get("JobID", row.get("job_id", ""))))
            if not re.fullmatch(r"\d+", job):
                continue  # Parent allocations only: do not double-count batch/extern steps.
            elapsed = row.get("ElapsedRaw", row.get("elapsed_seconds"))
            tres = row.get("AllocTRES", row.get("alloc_tres", ""))
            gpu = row.get("gpu_count")
            if gpu is None:
                match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", str(tres))
                gpu = int(match.group(1)) if match else None
            require(elapsed is not None and gpu is not None, f"incomplete GPU accounting for job {job}")
            record = {"job_id":job, "elapsed_seconds":float(elapsed), "gpu_count":int(gpu), "allocated_gpu_hours":float(elapsed)*int(gpu)/3600, "state":row.get("State",row.get("state")), "exit_code":row.get("ExitCode",row.get("exit_code"))}
            require(record["elapsed_seconds"] >= 0 and record["gpu_count"] >= 0, "negative scheduler accounting")
            require(job not in jobs or jobs[job] == record, f"conflicting accounting snapshots for job {job}")
            jobs[job] = record
    return {"status":"provided_receipts" if jobs else "unknown", "jobs":list(jobs.values()), "sources":sources, "allocated_gpu_hours":sum(j["allocated_gpu_hours"] for j in jobs.values()) if jobs else None, "online_scheduler_queried":False}


def captured(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ImportError as error:
        return {"status":"unknown", "reason":f"missing local tokenizer dependency: {error}"}
    except FileNotFoundError as error:
        return {"status":"unknown", "reason":f"missing evidence: {error.filename}"}
    except (AuditError, KeyError, TypeError, ValueError) as error:
        return {"status":"fail", "reason":f"{type(error).__name__}: {error}"}


def report(evaluations, pairs, scheduler=None, endpoints=None, provenance=None):
    endpoints = endpoints or {}
    domains = {}
    for domain in ("code", "qa"):
        caps = evaluations.get(domain, [])
        pair = pairs.get(domain, {"status":"unknown", "reason":"no paired training directories supplied"})
        endpoint = endpoints.get(domain)
        domain_mismatch = any(v["status"] == "pass" and v.get("domain") != domain for v in caps+[pair])
        failures = domain_mismatch or any(v["status"] == "fail" for v in caps+[pair]+([endpoint] if endpoint else []))
        capable = any(v["status"] == "pass" and v["summary"]["tasks_with_multiple_modes"] > 0 and v["token_text_binding"] == "pass" and v.get("frozen_implementation_binding") == "pass" for v in caps)
        paired = pair["status"] == "pass" and pair["token_text_binding"] == "pass" and pair.get("frozen_implementation_binding") == "pass"
        state = "fail" if failures else "pass" if capable and paired else "unknown"
        if (endpoint is None or endpoint["status"] != "pass") and state == "pass":
            state = "unknown"
        domains[domain] = {"endpoint_comparison":endpoint, "readiness":state,"capability":caps,"paired_training":pair,"capability_established":capable,"paired_run_integrity_established":paired}
    overall = "fail" if any(v["readiness"] == "fail" for v in domains.values()) else "pass" if all(v["readiness"] == "pass" for v in domains.values()) else "unknown"
    scheduler = scheduler or accounting([])
    required_jobs = set()
    missing_job_ids = False
    for domain in domains.values():
        passed = [r for r in domain["capability"] if r["status"] == "pass"]
        if domain["paired_training"]["status"] == "pass":
            passed += domain["paired_training"]["arms"]
        if domain["endpoint_comparison"] and domain["endpoint_comparison"]["status"] == "pass":
            passed += list(domain["endpoint_comparison"]["endpoints"].values())
        for row in passed:
            if row.get("job_id") is None:
                missing_job_ids = True
            else:
                required_jobs.add(str(row["job_id"]))
    jobs = {j["job_id"]: j for j in scheduler["jobs"]}
    account_complete = bool(required_jobs) and not missing_job_ids and required_jobs <= set(jobs) and all(str(jobs[j].get("state", "")).startswith("COMPLETED") and str(jobs[j].get("exit_code", "")) in {"0", "0:0"} for j in required_jobs)
    scheduler["required_pilot_job_ids"] = sorted(required_jobs)
    scheduler["missing_pilot_job_ids"] = sorted(required_jobs-set(jobs))
    scheduler["pilot_accounting_complete"] = account_complete
    budget_ok = scheduler["allocated_gpu_hours"] is not None and scheduler["allocated_gpu_hours"] <= 200
    if overall == "pass" and not (account_complete and budget_ok):
        overall = "unknown"
    if scheduler["allocated_gpu_hours"] is not None and scheduler["allocated_gpu_hours"] > 200:
        overall = "fail"
    if provenance is not None and provenance["status"] != "pass":
        overall = "fail" if provenance["status"] == "fail" else "unknown" if overall == "pass" else overall
    return {"schema":SCHEMA,"historical_code_quality_revocation":HISTORICAL_CODE_REVOCATION,"auditor_sha256":sha_file(__file__),"qa_source_provenance":provenance,"readiness_gate":overall,"readiness_scope":"Passing this gate means instrumented pilots are ready for larger comparisons; it does not establish treatment superiority","domains":domains,"scheduler_accounting":scheduler,"interpretation":["Capability is not treatment efficacy.","Coding modes are canonical executable output behavior, not algorithms.","QA modes are annotated native topics; hierarchical categories can both be valid.","PCMD is reported only with at least 30 accepted samples; paired PCMD differences use common eligible prompts.","Expected distinct@8/32 uses without-replacement subsampling from all attempts, including failures.","Missing paired runs or token bindings remain unknown; no favorable result is inferred."]}


def markdown(summary):
    lines = [f"Pilot readiness: **{summary['readiness_gate']}**.", "", summary["readiness_scope"]+".", "", "| Domain | Capability | Paired integrity | Readiness |", "|---|---|---|---|"]
    for domain, data in summary["domains"].items():
        lines.append(f"| {domain} | {data['capability_established']} | {data['paired_run_integrity_established']} | {data['readiness']} |")
    lines.extend(["", "Historical code evidence was revoked: six of fourteen accepted programs tested on fixed source-audit probes failed (74 total original acceptances). The original raw records remain preserved; receipt integrity alone did not establish verifier quality. Only the hardened verifier and fresh primary runs can establish code readiness."])
    for domain, data in summary["domains"].items():
        for cap in data["capability"]:
            if cap["status"] == "pass":
                s=cap["summary"]
                lines.extend(["", f"{domain} capability: {s['tasks']} prompts, {s['samples']} attempts, accuracy {s['macro_accuracy']:.3f}; {s['tasks_with_multiple_modes']} prompts produced multiple valid modes. PCMD eligible: {s['pcmd_eligible_tasks']}/{s['pcmd_total_tasks']}."])
            else:
                lines.extend(["", f"{domain} capability {cap['status']}: {cap.get('reason','')}"])
        pair=data["paired_training"]
        if pair["status"] != "pass":
            lines.extend(["",f"{domain} paired evidence {pair['status']}: {pair.get('reason','')}"])
        else:
            lines.extend(["",f"{domain} paired updates: {pair['arms'][0]['completed_updates']} per arm. Initial LoRA identity, raw-token bindings, bank reconstruction, gradients and checkpoint seals checked."])
            for arm in pair["arms"]:
                lines.append(f"{arm['arm']} fresh groups: {arm['mixed_groups']} mixed, {arm['all_correct_groups']} all-correct, {arm['all_wrong_groups']} all-wrong; final bank {sum(map(len, arm['bank_modes'].values()))} modes over {len(arm['bank_modes'])} prompts.")
                diagnostic=arm.get("training_diagnostics",{})
                if diagnostic.get("status")=="pass":
                    lines.append(f"{arm['arm']} fresh-advantage groups: {diagnostic['groups_with_nonzero_recomputed_fresh_advantages']}; positive actual gradients on mixed groups: {diagnostic['mixed_groups_with_positive_actual_gradient']}. Mean update {diagnostic['mean_update_seconds']:.2f}s; peak allocated/reserved GPU memory {diagnostic['peak_gpu_allocated_bytes']/2**30:.2f}/{diagnostic['peak_gpu_reserved_bytes']/2**30:.2f} GiB. Scheduler allocation remains the budget authority.")
            internal=pair["arms"][0]["evaluation"]["final"]["summary"]
            lines.append(f"Built-in final evaluation: {internal['tasks']} prompts, {internal['samples']} samples total; execution diagnostic only. Treatment interpretation uses the separate complete external endpoints.")
            effects = pair["paired_summary"]
            for metric in ("accuracy", "expected_distinct_at_8", "expected_distinct_at_32", "annotated_topic_coverage", "pcmd_common_eligible"):
                effect = effects[metric]
                if effect["mean"] is not None:
                    lo,hi = effect["paired_prompt_bootstrap_95_percentile"]
                    lines.append(f"Re:Max minus MaxRL {metric}: {effect['mean']:+.4f} (descriptive prompt bootstrap 95% interval {lo:+.4f} to {hi:+.4f}; {effect['prompts']} prompts).")
    for domain,data in summary["domains"].items():
        endpoint = data.get("endpoint_comparison")
        if endpoint is None:
            continue
        if endpoint["status"] != "pass":
            lines.extend(["",f"{domain} external endpoint audit {endpoint['status']}: {endpoint.get('reason','')}"])
            continue
        lines.extend(["",f"{domain} endpoint interpretation: **{endpoint['training_regime']}**. Primary summaries retain the complete frozen endpoint cohort, with trained and untrained development prompts distinguished. Coding development prompts are not the reserved larger-study test set.","", "| Stratum | Arm | Prompts | Accuracy | E[distinct@8] | E[distinct@32] | PCMD (eligible) | Annotated support observed |","|---|---|---:|---:|---:|---:|---:|---:|"])
        def number(v):
            return "NA" if v is None else f"{v:.4f}"
        for group,stratum in endpoint["strata"].items():
            for arm,a in stratum["arms"].items():
                lines.append(f"| {group} | {arm} | {a['tasks']} | {a['macro_accuracy']:.4f} | {number(a['macro_expected_distinct_valid_modes_at_k'].get('8'))} | {number(a['macro_expected_distinct_valid_modes_at_k'].get('32'))} | {number(a['macro_pcmd_over_eligible'])} ({a['pcmd_eligible_tasks']}/{a['tasks']}) | {number(a['macro_annotated_topic_coverage'])} |")
        for group,stratum in endpoint["strata"].items():
            for comparison in ("remax_minus_maxrl", "maxrl_minus_base"):
                for metric in ("accuracy","expected_distinct_at_8","expected_distinct_at_32","pcmd_common_eligible","annotated_topic_coverage"):
                    e=stratum[comparison]["paired_summary"][metric]
                    if e["mean"] is not None:
                        lo,hi=e["paired_prompt_bootstrap_95_percentile"]
                        lines.extend(["",f"{domain} {group}, {comparison}, {metric}: {e['mean']:+.4f} (descriptive paired-prompt 95% interval {lo:+.4f} to {hi:+.4f}; n={e['prompts']})."])
        lines.extend(["", "Trained-prompt descriptive table. QA support is observed native topics / known listed support; total coding witness support is unknown. Bank is endpoint-observed bank modes / that arm's discovered bank; base has no bank.","","| Prompt | Arm | Accuracy | E[distinct@8] | E[distinct@32] | PCMD | Support | Bank |","|---|---|---:|---:|---:|---:|---:|---:|"])
        for task,arms in endpoint["trained_prompt_table"].items():
            for arm,m in arms.items():
                support=f"{m['distinct_valid_modes']}/{m.get('known_mode_count','?')}"
                bank="NA" if arm == "base" else f"{len(m['bank_topics_observed_at_endpoint'])}/{len(m['discovered_bank_modes'])}"
                lines.append(f"| {task} | {arm} | {m['accuracy']:.4f} | {number(m['expected_distinct_valid_modes_at_k'].get('8'))} | {number(m['expected_distinct_valid_modes_at_k'].get('32'))} | {number(m['pcmd'])} | {support} | {bank} |")
        lines.extend(["",endpoint["replay_support_interpretation"], "",f"Common multi-bank subset has {len(endpoint['common_multi_bank_task_ids'])} prompts: {endpoint['common_multi_bank_interpretation']}."])
    provenance=summary.get("qa_source_provenance")
    if provenance:
        lines.extend(["",f"QA original-source provenance: {provenance['status']}. "+(f"{provenance['frozen_rows_checked']} frozen records bound to the Reuters/source-label audit with no original article crossing splits. {provenance['scope']}" if provenance['status']=='pass' else provenance.get('reason',''))])
    hours=summary["scheduler_accounting"]["allocated_gpu_hours"]
    lines.extend(["",f"Allocated GPU-hours from supplied scheduler receipts: {hours if hours is not None else 'unknown'}.","",*summary["interpretation"]])
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation",action="append",default=[],metavar="DOMAIN=DIR_OR_JSON")
    parser.add_argument("--pair",action="append",default=[],metavar="DOMAIN=MAXRL_DIR,REMAX_DIR")
    parser.add_argument("--endpoints",action="append",default=[],metavar="DOMAIN=BASE_DIR,MAXRL_DIR,REMAX_DIR")
    parser.add_argument("--qa-provenance",type=Path)
    parser.add_argument("--accounting",action="append",default=[],type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--skip-tokenizer",action="store_true",help="Bindings stay unknown; readiness cannot pass")
    args=parser.parse_args(); evaluations=defaultdict(list); pairs={}; endpoints={}; qa_config=None
    codecs = {}
    def get_codec(config):
        if args.skip_tokenizer:
            return None
        key = config["model"]
        if key not in codecs:
            codecs[key] = local_codec(config)
        return codecs[key]
    for value in args.evaluation:
        domain,name=value.split("=",1); require(domain in {"code","qa"},"unknown domain")
        path=Path(name); path=path/"evaluation.json" if path.is_dir() else path
        def evaluate():
            return audit_evaluation(path,get_codec(read_json(path)["config"]))
        evaluations[domain].append(captured(evaluate))
        if domain == "qa" and path.exists():
            qa_config=read_json(path)["config"]
    for value in args.pair:
        domain,names=value.split("=",1); require(domain in {"code","qa"} and domain not in pairs,"unknown/duplicate pair domain")
        a,b=map(Path,names.split(",",1))
        def pair():
            ca,cb=(get_codec(read_json(training_dir(p)/"identity.json")["config"]) for p in (a,b))
            return audit_pair(a,b,(ca,cb))
        pairs[domain]=captured(pair)
        if domain == "qa" and (training_dir(a)/"identity.json").exists():
            qa_config=read_json(training_dir(a)/"identity.json")["config"]
    for value in args.endpoints:
        domain,names=value.split("=",1); require(domain in {"code","qa"} and domain not in endpoints,"unknown/duplicate endpoint domain")
        paths=[Path(name) for name in names.split(",")]
        def endpoint():
            configs=[read_json(p/"evaluation.json" if p.is_dir() else p)["config"] for p in paths]
            return audit_endpoints(paths,pairs.get(domain,{}),[get_codec(c) for c in configs])
        endpoints[domain]=captured(endpoint)
    provenance=captured(audit_qa_provenance,args.qa_provenance,qa_config) if args.qa_provenance else None
    summary=report(evaluations,pairs,accounting(args.accounting),endpoints,provenance)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
    (args.output/"report.md").write_text(markdown(summary))
    print(json.dumps({"readiness_gate":summary["readiness_gate"],"output":str(args.output)}))
    return 1 if summary["readiness_gate"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
