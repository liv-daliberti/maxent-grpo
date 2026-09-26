#!/usr/bin/env python3
"""Record the independently reviewed 710_C input-validator incompatibility.

Original checker, validator, source inputs, and prior failed audit remain intact.
This source-only action never reads policy outputs or changes reward predicates.
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import re
import build_constructive_code_wider_20260921 as b

EXPECTED = {
    "checker_sha256": "171098383fbbed6a43e62f9567c2da266ac73f86d0e961362ed4a1be795dd067",
    "validator_sha256": "2eb2f2fccbe83d3b48a0ca8c971a39044d46cb70796b6ebba33d184d8fc0922a",
    "source_inputs_sha256": "52545eef45378d779575cd9e7676c9a326729d39fe8b39c3b24bcc903d6c0751",
}


def resolve(root: Path) -> None:
    task_dir = root / "710_c"
    record = json.loads((task_dir / "task.json").read_text())
    audit = json.loads((task_dir / "admission_audit.json").read_text())
    if record["checker_sha256"] != EXPECTED["checker_sha256"] or record["validator_sha256"] != EXPECTED["validator_sha256"]:
        raise ValueError("source identities differ from independent review")
    if b.digest(task_dir / "checker.cpp") != EXPECTED["checker_sha256"] or b.digest(task_dir / "validator.cpp") != EXPECTED["validator_sha256"] or b.digest(task_dir / "inputs.jsonl.gz") != EXPECTED["source_inputs_sha256"]:
        raise ValueError("source bytes differ from independent review")
    if audit["status"] != "fail" or audit["violations"] != ["released validator disagrees on independently valid inputs; requires source review"] or audit["tpr"] != 1.0 or audit["tnr"] != 1.0 or audit["positive_replays"] != 12 or audit["negative_replays"] != 12:
        raise ValueError("admission failure differs from independent review")
    raw = [json.loads(line) for line in gzip.open(task_dir / "inputs.jsonl.gz", "rt")]
    admitted = [json.loads(line) for line in gzip.open(task_dir / record["suite_file"], "rt")]
    if len(raw) != 16 or [r["stdin"] for r in raw] != [r["stdin"] for r in admitted]:
        raise ValueError("source inputs were changed or dropped")
    checks = []
    for row in raw:
        text = row["stdin"]
        if re.fullmatch(r"\s*[0-9]+\s*", text) is None or not 1 <= int(text) <= 49 or int(text) % 2 != 1:
            raise ValueError("source input violates original single odd integer constraints")
        if row["input_sha256"] != b.raw_sha256(text) or row["input_bytes"] != len(text.encode()):
            raise ValueError("source input byte binding drift")
        checks.append({"source_test_index": row["test_index"], "input_sha256": row["input_sha256"], "n": int(text), "constraint_pass": True})
    for row in audit["input_audit"]["checks"]:
        if not row["retained"] or row["independent_error"] is not None or row["validator_returncode"] != 3 or "Illegal pattern" not in row["validator_message"]:
            raise ValueError("source validator error differs from reviewed incompatibility")
    review = {"status": "resolved_input_validator_incompatibility", **EXPECTED, "problem_id": "710_C", "source_url": "https://codeforces.com/problemset/problem/710/C", "reason": "Generated input validator uses unsupported PCRE noncapturing-group and anchor syntax in pinned testlib; every source input fails pattern parsing.", "independent_contract": "Exactly one ordinary whitespace-delimited decimal integer n, odd and in[1,49]. Leading/trailing whitespace is permitted; no additional nonwhitespace tokens. This is a statement-constraint validator, not a claimed byte-identical repair of strict readLine/readEof.", "independent_review": "cited_data read-only review 2026-09-21 independently checked all16 raw/admitted byte sequences, input hashes and original constraints; unchanged output checker passes12 known-correct and12 known-incorrect programs.", "all_source_input_bytes_retained": True, "output_checker_modified": False, "model_sampling_performed": False, "checks": checks, "pre_review_admission_sha256": b.digest(task_dir / "admission_audit.json")}
    b.write_json(task_dir / "pre_review_admission_audit.json", audit)
    b.write_json(task_dir / "validator_compatibility_review.json", review)
    audit.update({"status": "pass", "violations": [], "resolved_validator_incompatibility": review, "validator_compatibility_review_sha256": b.digest(task_dir / "validator_compatibility_review.json")})
    b.write_json(task_dir / "admission_audit.json", audit)
    record.update({"admission_status": "admitted", "admission_audit_sha256": b.digest(task_dir / "admission_audit.json"), "validator_compatibility_review_sha256": b.digest(task_dir / "validator_compatibility_review.json")})
    record["task_record_sha256"] = b.canonical_hash({k:v for k,v in record.items() if k != "task_record_sha256"})
    b.write_json(task_dir / "task.json", record)
    manifest = json.loads((root / "manifest.json").read_text())
    for row in manifest["tasks"]:
        if row["source_problem_id"] == "710_C":
            row.update({"status": "admitted", "task_record_sha256": record["task_record_sha256"], "admission_audit_sha256": record["admission_audit_sha256"]})
    manifest.update({"tasks_sha256": b.canonical_hash(manifest["tasks"]), "admitted_problem_ids": [r["source_problem_id"] for r in manifest["tasks"] if r["status"] == "admitted"]})
    b.write_json(root / "manifest.json", manifest)
    summary = json.loads((root / "admission_summary.json").read_text())
    summary.update({"admitted": len(manifest["admitted_problem_ids"]), "admitted_problem_ids": manifest["admitted_problem_ids"]})
    summary["tasks"] = [{k:v for k,v in audit.items() if k not in ("input_audit", "canonicalizer_audit", "checker_build")} if r["source_problem_id"] == "710_C" else r for r in summary["tasks"]]
    b.write_json(root / "admission_summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    resolve(parser.parse_args().root)
