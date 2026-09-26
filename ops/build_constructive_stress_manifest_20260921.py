#!/usr/bin/env python3
"""Freeze independent audit-only counterexamples; never modify reward suites."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import build_constructive_code_wider_20260921 as b
from replay_constructive_code_review_slate import TestCase
from oat_drgrpo.constructive_code_wider_adapters_20260921 import input_cases

SCHEMA = "constructive-code-out-of-reward-stress-20260921-v1"
PROBES = {"1016_D": "2 3\n0 0\n0 1 1\n", "1360_G": "1\n3 4 3 2\n", "1408_A": "1\n3\n1 3 2\n2 2 1\n3 1 3\n", "1323_A": "1\n3\n1 1 1\n", "1038_B": "2\n"}


def probe_task(task, text):
    data = text.encode(); digest = b.raw_sha256(text)
    return replace(task, suite_id="out_of_reward_source_coverage_diagnostic", suite_sha256=b.canonical_hash([digest]), tests=(TestCase(0, data, digest),))


def build(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    base = b.configure_replay(); runtime = b._runtime(base, args)
    tasks, _, _, manifest = b.load_admitted_tasks(args.slate_root, args.build_root, list(PROBES))
    cases = []
    for task in tasks:
        task_dir = args.slate_root / task.problem_id.lower()
        record = json.loads((task_dir / "task.json").read_text())
        refs = [json.loads(line) for line in (task_dir / "py3_replays.jsonl").read_text().splitlines()]
        old = [json.loads(line) for line in (task_dir / "audit_replays.jsonl").read_text().splitlines()]
        stdin = PROBES[task.problem_id]; input_cases(task.problem_id, stdin)
        modified = probe_task(task, stdin); checks = []
        for label, expected in (("correct", True), ("incorrect", False)):
            previously_accepted = next(row for row in old if row["known_label"] == label and row["wrapper_accepted"])
            source = next(row for row in refs if row["submission_sha256"] == previously_accepted["submission_sha256"])
            replay = base._replay_submission(task=modified, submission=base.Submission(source["code"], label, source["submission_sha256"]), launcher=args.launcher, runtime_root=args.runtime_root, scratch_root=args.scratch_root)
            if replay["released_checker_accepted"] != expected or replay["wrapper_accepted"] != expected:
                raise ValueError(f"{task.problem_id}: source reference disagrees with diagnostic expectation")
            checks.append({"known_label": label, "source_program_sha256": source["submission_sha256"], "source_program": source["code"], "frozen_reward_suite_accepted": True, "expected_stress_accepted": expected, "replay": replay})
        contest, index = task.problem_id.split("_")
        cases.append({"task_id": task.problem_id, "stdin": stdin, "input_sha256": b.raw_sha256(stdin), "input_bytes": len(stdin.encode()), "checker_sha256": task.checker_sha256, "frozen_reward_suite_sha256": task.suite_sha256, "source_statement_sha256": record["statement_sha256"], "source_url": f"https://codeforces.com/problemset/problem/{contest}/{index}", "source_program_provenance": record["provenance"], "probe_provenance": "Independent human-authored counterexample from original task constraints, discovered by inspecting a source-labeled incorrect program that passed the finite source suite; not a released dataset input.", "original_statement_constraints_pass": True, "reference_checks": checks})
        print(task.problem_id, "positive accepted / negative rejected", flush=True)
    result = {"schema": SCHEMA, "status": "reference_audit_pass", "reward_data_modified": False, "model_sampling_used_in_probe_selection": False, "probe_selection": "Only source-reference failures, chosen before reading generated policy programs; no selection by treatment results.", "interpretation": "Small targeted stress diagnostic. These constructed probes are outside training reward and outside primary benchmark metrics; they do not establish total program correctness.", "source_slate_manifest_sha256": b.digest(args.slate_root / "manifest.json"), "cases": cases, "runtime": runtime}
    b.write_json(args.output, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slate-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image", type=Path, default=b.ROOT / "var/images/python-3.10-slim-c1e4e6c01eb4.sqsh")
    for field in ("runtime-root", "launcher", "scratch-root", "build-root"):
        p.add_argument("--" + field, type=Path, required=True)
    build(p.parse_args())
