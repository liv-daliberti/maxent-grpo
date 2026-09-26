#!/usr/bin/env python3
"""Merge audited source cohorts into explicit train/validation/test pools.

Initial pilot inputs and manifests are never mutated. Split assignment uses only
prospective reservations and source admission, without reading policy outcomes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import build_constructive_code_wider_20260921 as b

VALIDATION_IDS = ("1554_D", "534_A")


def merge(roots: list[Path], output: Path):
    if output.exists():
        raise FileExistsError(output)
    source = []; tasks = []; seen_ids = set(); seen_statements = set()
    for root in roots:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["schema_version"] != b.SCHEMA or manifest["status"] != "audited" or manifest["tasks_sha256"] != b.canonical_hash(manifest["tasks"]):
            raise ValueError("source cohort lacks intact completed admission")
        if b.digest(root / "split_reservation.json") != manifest["split_reservation_sha256"]:
            raise ValueError("source cohort split binding drift")
        source.append((root, manifest))
        for summary in manifest["tasks"]:
            row = json.loads((root / summary["relative_path"] / "task.json").read_text())
            if row["source_problem_id"] in seen_ids or row["normalized_statement_sha256"] in seen_statements:
                raise ValueError("task identity or statement alias duplicated across cohorts")
            if row["task_record_sha256"] != summary["task_record_sha256"] or row["task_record_sha256"] != b.canonical_hash({k:v for k,v in row.items() if k != "task_record_sha256"}):
                raise ValueError("source task record identity drift")
            if b.digest(root / summary["relative_path"] / "admission_audit.json") != row["admission_audit_sha256"]:
                raise ValueError("source admission audit binding drift")
            seen_ids.add(row["source_problem_id"]);seen_statements.add(row["normalized_statement_sha256"])
            tasks.append((root, summary, row))
    admitted = {r["source_problem_id"] for _,_,r in tasks if r["admission_status"] == "admitted"}
    if not set(VALIDATION_IDS) <= admitted:
        raise ValueError("prospectively selected validation tasks are not admitted")
    output.mkdir(parents=True)
    provenance = []; summaries = []; repairs = []
    for index, (root, manifest) in enumerate(source):
        provenance.append({"path": str(root.resolve()), "manifest_sha256": b.digest(root / "manifest.json"), "tasks_sha256": manifest["tasks_sha256"], "cohort": manifest.get("cohort", "initial")})
        target = output / "source_manifests" / str(index);target.mkdir(parents=True)
        for filename in ("manifest.json", "admission_summary.json", "statement_repairs.json"):
            shutil.copy2(root / filename, target / filename)
        repairs.extend(json.loads((root / "statement_repairs.json").read_text())["repairs"])
    for root, summary, record in tasks:
        task_id = record["source_problem_id"]
        folder = output / summary["relative_path"]
        shutil.copytree(root / summary["relative_path"], folder)
        split = "test" if task_id in (*b.NEW_HELDOUT, *b.EXTENSION_HELDOUT) else "validation" if task_id in VALIDATION_IDS else "train"
        record.update({"source_task_record_sha256": record["task_record_sha256"], "split": split})
        record["task_record_sha256"] = b.canonical_hash({k:v for k,v in record.items() if k != "task_record_sha256"})
        b.write_json(folder / "task.json", record)
        summaries.append({**summary, "split": split, "task_record_sha256": record["task_record_sha256"]})
    shutil.copy2(roots[0] / "split_reservation.json", output / "split_reservation.json")
    shutil.copy2(b.EXTENSION_RESERVATION, output / "extension_reservation.json")
    b.write_json(output / "statement_repairs.json", {"schema_version": b.SCHEMA, "repairs": repairs})
    admitted_summaries = [r for r in summaries if r["status"] == "admitted"]
    pools = {split: [r["source_problem_id"] for r in admitted_summaries if r["split"] == split] for split in ("train", "validation", "test")}
    manifest = {"schema_version": b.SCHEMA, "status": "audited", "cohort": "larger_study_explicit_splits", "tasks": summaries, "tasks_sha256": b.canonical_hash(summaries), "admitted_problem_ids": [r["source_problem_id"] for r in admitted_summaries], "default_training_problem_ids": pools["train"], "validation_problem_ids": pools["validation"], "admitted_test_problem_ids": pools["test"], "source_revisions": source[0][1]["source_revisions"], "source_manifests": provenance, "split_reservation_sha256": b.digest(output / "split_reservation.json"), "extension_reservation_sha256": b.digest(output / "extension_reservation.json"), "statement_repairs_sha256": b.digest(output / "statement_repairs.json"), "protected_unused_original_heldout_ids": list(b.ORIGINAL_HELDOUT), "split_selection_used_policy_outcomes": False, "split_rule": "All23 initial admitted tasks remain train candidates. Two smallest normalized-statement SHA256s among11 admitted train reserves are clean validation; other9 reserves join training. Admitted prospective heldout and structural replacements are test-only.", "pool_counts": {k:len(v) for k,v in pools.items()}, "family_counts": {split:dict(Counter(r["family"] for r in admitted_summaries if r["split"]==split)) for split in pools}, "larger_ready_source_gate": len(pools["train"])>=32 and len(pools["validation"])==2 and len(pools["test"])>=16}
    b.write_json(output / "manifest.json", manifest)
    b.write_json(output / "split_manifest.json", {"schema": "constructive-code-larger-splits-20260921-v1", "manifest_sha256": b.digest(output / "manifest.json"), "train_ids": pools["train"], "validation_ids": pools["validation"], "heldout_ids": pools["test"], "protected_unused_original_heldout_ids": list(b.ORIGINAL_HELDOUT), "selection_used_policy_outcomes": False, "excluded_candidates": [r for r in summaries if r["status"] != "admitted"]})
    print(json.dumps({"counts":manifest["pool_counts"],"larger_ready_source_gate":manifest["larger_ready_source_gate"],"manifest_sha256":b.digest(output/'manifest.json')},sort_keys=True))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,action="append",required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();merge(args.root,args.output)
