#!/usr/bin/env python3
"""Materialize and CPU-audit a wider, prospectively selected constructive slate.

Only selected metadata/submission columns and CodeContests-O input rows are
retrieved at pinned revisions. No reference programs enter a policy prompt.
Historical datasets, checkers, and adapters remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
SOURCE_ROOT = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from audit_constructive_code_sources import PLUS_REPO, PLUS_REVISION, OVERLAY_REPO, OVERLAY_REVISION, candidate_key, raw_sha256
from materialize_constructive_code_review_slate import PLUS_PAYLOAD_COLUMNS, OVERLAY_PAYLOAD_COLUMNS, _overlay_inputs, _source_sizes, build_overlay_suite
from materialize_constructive_code_plus_suites import _read_selected_parquet_rows
from materialize_constructive_code_v2 import select_heldout_python3, _write_suite

SCHEMA = "constructive-code-wider-slate-20260921-v1"
INDEX_SHA256 = "00ae7293af83314b8c49a68d867ac07a4c4b81b580fd2174494c88fbf35a2602"
DEFAULT_OUTPUT = ROOT / "var/data/constructive_code_wider_20260921"
DEFAULT_INDEX = ROOT / "var/artifacts/constructive_code_candidate_source_index.json"
TASKS = {
    "1454_A": ("ordered_sequence", "wider_1454_a_v1"),
    "1513_A": ("ordered_sequence", "wider_1513_a_v1"),
    "1569_A": ("ordered_sequence", "wider_1569_a_v1"),
    "1352_B": ("unordered_set", "wider_1352_b_multiset_v1"),
    "1016_D": ("assignment", "wider_1016_d_v1"),
    "1360_G": ("assignment", "wider_1360_g_v1"),
    "361_A": ("assignment", "wider_361_a_v1"),
    "244_A": ("assignment", "wider_244_a_anchored_v1"),
    "1102_B": ("unordered_partition", "status_label_partition_v1"),
    "1408_A": ("assignment", "multi_case_implicit_assignment_v1"),
    "988_A": ("unordered_set", "status_integer_set_v1"),
    "1399_D": ("unordered_partition", "multi_case_label_partition_v1"),
    "1323_A": ("unordered_set", "wider_1323_a_v1"),
    "1073_A": ("ordered_sequence", "wider_1073_a_v1"),
    "1380_A": ("ordered_sequence", "wider_1380_a_v1"),
    "1095_C": ("unordered_set", "wider_1095_c_multiset_v1"),
    "1352_G": ("ordered_sequence", "wider_1352_g_v1"),
    "482_A": ("ordered_sequence", "fixed_integer_sequence_v1"),
    "1339_B": ("ordered_sequence", "wider_1339_b_v1"),
    "1371_D": ("assignment", "wider_1371_d_v1"),
    "1038_B": ("unordered_partition", "wider_1038_b_v1"),
    "1051_B": ("unordered_partition", "status_pair_partition_v1"),
    "545_B": ("ordered_sequence", "wider_545_b_v1"),
    "1352_F": ("ordered_sequence", "wider_1352_f_v1"),
}
ORIGINAL_HELDOUT = ("361_B", "1294_C", "149_C")
ANCHORS = ("988_A", "1399_D")
SUITE_ID = "codecontests_o_wider_20260921"
REFERENCE_PER_LABEL = 12
NEW_HELDOUT = ("1023_C", "1047_A", "1088_A", "1093_B", "1325_A", "1332_B", "1360_F", "1430_A", "1436_B", "1450_A", "1497_C2", "1549_A", "1559_B", "1606_A", "544_B", "710_C")
TRAIN_RESERVE = ("1096_A", "1269_A", "1326_A", "1463_B", "1511_B", "1554_D", "1594_A", "1608_A", "1617_B", "199_A", "472_A", "534_A")
RESERVATION = ROOT / "var/data/constructive_code_holdout_reservation_20260921.json"
RESERVATION_SHA256 = "751c9ff51d07c5ab8eec69096d8ce0d741d6810cb3e6941fea1ccce7c49516e4"
from oat_drgrpo.constructive_code_reserved_adapters_20260921 import FAMILIES as RESERVED_FAMILIES, ADAPTER_IDS as RESERVED_ADAPTER_IDS
EXTRA_TASKS = {problem: (RESERVED_FAMILIES[problem], RESERVED_ADAPTER_IDS[problem]) for problem in (*TRAIN_RESERVE, *NEW_HELDOUT)}
from oat_drgrpo.constructive_code_holdout_extension_20260921 import FAMILIES as EXT_FAMILIES, ADAPTER_IDS as EXT_ADAPTER_IDS
EXTENSION_HELDOUT = tuple(EXT_FAMILIES)
EXTENSION_TASKS = {p: (EXT_FAMILIES[p], EXT_ADAPTER_IDS[p]) for p in EXTENSION_HELDOUT}
ALL_TASKS = {**TASKS, **EXTRA_TASKS, **EXTENSION_TASKS}
EXTENSION_RESERVATION = ROOT / "var/data/constructive_code_holdout_extension_reservation_20260921.json"
EXTENSION_RESERVATION_SHA256 = "5f924af8760e5e714fccbb7f37228ab24e1398fe409707bf6224bc6d81b813dc"
# Transcriptions checked against the original Codeforces problem pages. Preserve
# both source and effective bytes. No hints, solutions, or input cases are added.
STATEMENT_REPAIRS = {
    "1016_D": (("109", "10^9", 3),),
    "482_A": (("105", "10^5", 1), ("|pn - 1 - pn|", "|p_{n-1} - p_n|", 1)),
    "545_B": (("105", "10^5", 1),),
    "199_A": (("109", "10^9", 1),),
    "472_A": (("106", "10^6", 1),),
    "544_B": (("n2", "n^2", 1),),
    "710_C": (("n2", "n^2", 2),),
}


def repair_statement(problem: str, source_statement: str) -> tuple[str, dict[str, Any]]:
    statement = source_statement
    replacements = []
    for old, new, count in STATEMENT_REPAIRS.get(problem, ()):
        if statement.count(old) != count:
            raise ValueError(f"{problem}: source statement repair precondition drift")
        statement = statement.replace(old, new)
        replacements.append({"original": old, "replacement": new, "count": count})
    contest, problem_index = problem.split("_")
    return statement, {"source_statement_sha256": raw_sha256(source_statement), "effective_statement_sha256": raw_sha256(statement), "replacements": replacements, "source_url": f"https://codeforces.com/problemset/problem/{contest}/{problem_index}", "verification": "original Codeforces rendered mathematical notation checked 2026-09-21" if replacements else "unmodified pinned source statement"}


def freeze_statement_repairs(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "audited":
        raise ValueError("statement provenance finalized only after source audit completes")
    repairs = []
    for summary in manifest["tasks"]:
        path = output / summary["relative_path"] / "task.json"
        record = json.loads(path.read_text())
        source = record.get("source_statement", record["statement"])
        statement, provenance = repair_statement(record["source_problem_id"], source)
        record.update({"source_statement": source, "statement": statement, "statement_sha256": raw_sha256(statement), "statement_provenance": provenance})
        record["task_record_sha256"] = canonical_hash({k:v for k,v in record.items() if k != "task_record_sha256"})
        write_json(path, record); summary["task_record_sha256"] = record["task_record_sha256"]
        if provenance["replacements"]:
            repairs.append({"source_problem_id": record["source_problem_id"], **provenance})
    reservation = json.loads(RESERVATION.read_text())
    if canonical_hash(reservation) != RESERVATION_SHA256:
        raise ValueError("prospective heldout reservation drift")
    write_json(output / "split_reservation.json", reservation)
    write_json(output / "statement_repairs.json", {"schema_version": SCHEMA, "repairs": repairs, "model_sampling_performed": False})
    manifest.update({"tasks_sha256": canonical_hash(manifest["tasks"]), "split_reservation_sha256": digest(output / "split_reservation.json"), "statement_repairs_sha256": digest(output / "statement_repairs.json")})
    write_json(output / "manifest.json", manifest)



def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows))


def _index(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if digest(path) != INDEX_SHA256:
        raise ValueError("pinned candidate index SHA drift")
    index = json.loads(path.read_text())
    by_id = {row["source_problem_id"]: row for row in index["records"]}
    if set(ALL_TASKS) - set(by_id):
        raise ValueError("shortlist missing from source index")
    normalized = [by_id[key]["codecontests_plus"]["statement_normalized_sha256"] for key in ALL_TASKS]
    if len(set(normalized)) != len(ALL_TASKS):
        raise ValueError("development tasks contain statement aliases")
    heldout = {by_id[key]["codecontests_plus"]["statement_normalized_sha256"] for key in ORIGINAL_HELDOUT}
    if set(normalized) & heldout:
        raise ValueError("development aliases a preserved held-out problem")
    return index, by_id


def _fetch_source_kind(kind: str, selected: Mapping[str, Any], sizes: Mapping[str, int], output: Path) -> None:
    repo, revision, columns = (PLUS_REPO, PLUS_REVISION, PLUS_PAYLOAD_COLUMNS) if kind == "codecontests_plus" else (OVERLAY_REPO, OVERLAY_REVISION, OVERLAY_PAYLOAD_COLUMNS)
    groups = defaultdict(list)
    for problem, metadata in selected.items():
        cache = output / "source_cache" / f"{problem}.{kind}.json"
        if not cache.exists():
            groups[metadata[kind]["source_shard"]].append(problem)
    def fetch(shard: str, ids: list[str]):
        rows = _read_selected_parquet_rows(repo, revision, shard, sizes[shard], columns, [selected[key][kind]["source_row_index"] for key in ids], attempts=3)
        for row in rows:
            matches = [key for key in ids if row["_source_row_index"] == selected[key][kind]["source_row_index"]]
            if len(matches) != 1:
                raise ValueError("ambiguous pinned source row")
            key = matches[0]
            metadata = selected[key]
            if raw_sha256(row["checker"]) != metadata[kind]["checker_raw_sha256"]:
                raise ValueError(f"{key}: checker source drift")
            if kind == "codecontests_plus" and candidate_key(row) != metadata["problem_key"]:
                raise ValueError(f"{key}: Plus source identity drift")
            if kind == "codecontests_o" and row["name"] != metadata[kind]["name"]:
                raise ValueError(f"{key}: overlay source identity drift")
            write_json(output / "source_cache" / f"{key}.{kind}.json", row)
            print(f"[wider-fetch] source={kind} task={key}", flush=True)
    # Submission columns can be hundreds of MB per row group; bound memory and
    # avoid the multi-GB unselected Plus test-input and reference-output columns.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fetch, shard, ids): shard for shard, ids in sorted(groups.items())}
        for future in as_completed(futures):
            future.result()


def fetch_and_materialize(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi
    index, by_id = _index(args.index)
    active_tasks = {"initial": TASKS, "reserved": EXTRA_TASKS, "heldout-extension": EXTENSION_TASKS}[args.cohort]
    selected = {key: by_id[key] for key in active_tasks}
    args.output.mkdir(parents=True, exist_ok=True)
    aliases = defaultdict(list)
    for row in index["records"]:
        aliases[row["codecontests_plus"]["statement_normalized_sha256"]].append(row["source_problem_id"])
    reservation = json.loads(RESERVATION.read_text())
    if canonical_hash(reservation) != RESERVATION_SHA256:
        raise ValueError("prospective split reservation drift")
    write_json(args.output / "split_reservation.json", reservation)
    if args.cohort == "heldout-extension":
        extension = json.loads(EXTENSION_RESERVATION.read_text())
        if canonical_hash(extension) != EXTENSION_RESERVATION_SHA256:
            raise ValueError("prospective heldout extension reservation drift")
        write_json(args.output / "extension_reservation.json", extension)
    api = HfApi()
    for kind, repo, revision in (("codecontests_plus", PLUS_REPO, PLUS_REVISION), ("codecontests_o", OVERLAY_REPO, OVERLAY_REVISION)):
        info = api.dataset_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError("source revision drift")
        _fetch_source_kind(kind, selected, _source_sizes(info), args.output)
    summaries = []
    for problem, (family, adapter) in active_tasks.items():
        source = selected[problem]
        plus = json.loads((args.output / "source_cache" / f"{problem}.codecontests_plus.json").read_text())
        overlay = json.loads((args.output / "source_cache" / f"{problem}.codecontests_o.json").read_text())
        task_dir = args.output / problem.lower()
        task_dir.mkdir(exist_ok=True)
        (task_dir / "checker.cpp").write_text(plus["checker"])
        (task_dir / "validator.cpp").write_text(plus["validator"])
        records = []
        counts = {}
        for label in ("correct", "incorrect"):
            chosen, count = select_heldout_python3(plus.get(label + "_submissions"), set(), REFERENCE_PER_LABEL)
            counts[label] = count
            records.extend({"known_label": label, **row} for row in chosen)
        records.sort(key=lambda row: (row["known_label"], row["submission_sha256"]))
        write_jsonl(task_dir / "py3_replays.jsonl", records)
        inputs = _overlay_inputs(overlay)
        suite = _write_suite(task_dir / "inputs.jsonl.gz", inputs)
        source_statement = source["statement"]
        statement, statement_provenance = repair_statement(problem, source_statement)
        if "<image>" in statement or "<img" in statement.lower():
            raise ValueError(f"unexpected image placeholder in shortlist {problem}")
        task = {
            "schema_version": SCHEMA, "source_problem_id": problem, "problem_key": source["problem_key"], "title": source["title"],
            "source_statement": source_statement, "statement_provenance": statement_provenance, "statement": statement, "statement_sha256": raw_sha256(statement), "normalized_statement_sha256": source["codecontests_plus"]["statement_normalized_sha256"],
            "split": "test" if problem in (*NEW_HELDOUT, *EXTENSION_HELDOUT) else "development", "continuity_anchor": problem in ANCHORS,
            "witness_family": family, "task_adapter": adapter,
            "checker_sha256": digest(task_dir / "checker.cpp"), "validator_sha256": digest(task_dir / "validator.cpp"),
            "limits": {"time_milliseconds": int(plus["time_limit"]), "memory_megabytes": int(plus["memory_limit"])},
            "suite_id": SUITE_ID, "suite": suite,
            "references": {"jsonl_sha256": digest(task_dir / "py3_replays.jsonl"), "per_label": REFERENCE_PER_LABEL, "counts": counts},
            "provenance": {"codecontests_plus": source["codecontests_plus"], "codecontests_o": source["codecontests_o"], "plus_revision": PLUS_REVISION, "overlay_revision": OVERLAY_REVISION},
            "admission_status": "pending_cpu_audit",
        }
        task["task_record_sha256"] = canonical_hash(task)
        write_json(task_dir / "task.json", task)
        summaries.append({"source_problem_id": problem, "relative_path": problem.lower(), "task_record_sha256": task["task_record_sha256"], "family": family, "split": task["split"], "status": "pending_cpu_audit"})
        print(f"[wider-materialize] task={problem} tests={suite['test_count']} references={len(records)}", flush=True)
    write_json(args.output / "manifest.json", {"schema_version": SCHEMA, "status": "pending_cpu_audit", "generated_at": datetime.now(timezone.utc).isoformat(), "cohort": args.cohort, "split_reservation_sha256": digest(args.output / "split_reservation.json"), "index_sha256": INDEX_SHA256, "tasks": summaries, "tasks_sha256": canonical_hash(summaries), "source_revisions": {"codecontests_plus": PLUS_REVISION, "codecontests_o": OVERLAY_REVISION}, "original_heldout_ids": list(ORIGINAL_HELDOUT), "sampling_performed": False})



def configure_replay():
    import replay_constructive_code_v2 as base
    from oat_drgrpo import constructive_code_wider_adapters_20260921 as wider
    base.canonicalize_task_witness = wider.canonicalize_task_witness
    return base


def _task_from_record(base: Any, task_dir: Path, record: Mapping[str, Any], build_root: Path):
    unsigned = {key: value for key, value in record.items() if key != "task_record_sha256"}
    if canonical_hash(unsigned) != record["task_record_sha256"]:
        raise ValueError("task record SHA drift")
    if digest(task_dir / "checker.cpp") != record["checker_sha256"] or digest(task_dir / "py3_replays.jsonl") != record["references"]["jsonl_sha256"]:
        raise ValueError("task checker/reference SHA drift")
    problem = record["source_problem_id"]
    checker = build_root / problem.lower() / "checker"
    build = base._compile_checker(task_dir / "checker.cpp", checker, record["checker_sha256"])
    limits = record["limits"]
    cpu = max(1, math.ceil(limits["time_milliseconds"] * 3 / 1000))
    task = base.Task(problem_id=problem, problem_key=record["problem_key"], adapter_id=record["task_adapter"], witness_family=record["witness_family"], suite_id=record["suite_id"], suite_sha256=record["suite"]["suite_sha256"], checker_sha256=record["checker_sha256"], checker_binary=checker, tests=base._load_tests(task_dir / record.get("suite_file", "inputs.jsonl.gz"), record["suite"]), submissions=(), limits=base.SandboxLimits(cpu_seconds=cpu, wall_seconds=float(cpu + 2), memory_bytes=limits["memory_megabytes"] * 1024**2, output_bytes=16 * 1024**2, file_count=32, source_bytes=256 * 1024))
    return task, build


def _runtime(base: Any, args: Any) -> dict[str, Any]:
    from oat_drgrpo.constructive_code_sandbox import verify_runtime_root, PINNED_IMAGE_SHA256
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    launcher_hash = base.build_launcher(base.SANDBOX_SOURCE, args.launcher)
    if args.runtime_root.exists():
        runtime = {"reused": True, "critical_file_sha256": verify_runtime_root(args.runtime_root), "image_sha256": digest(args.image)}
        if runtime["image_sha256"] != PINNED_IMAGE_SHA256:
            raise ValueError("pinned runtime image SHA drift")
    else:
        runtime = asdict(base.prepare_runtime(args.image, args.runtime_root))
    return {"launcher_sha256": launcher_hash, "runtime": runtime}


# Small test-only witnesses check semantic distinctions and label/serialization
# quotients against the released checker. These are NOT training/eval inputs.
CANONICAL_PROBES = {
    "1454_A": ("1\n3\n", "2 3 1\n", "3 1 2\n", " 2  3 1 \n"),
    "1513_A": ("1\n4 1\n", "1 3 2 4\n", "1 4 2 3\n", "1 3 2 4 \n"),
    "1569_A": ("1\n4\nabba\n", "1 2\n", "3 4\n", "1  2 \n"),
    "1352_B": ("1\n12 3\n", "YES\n2 2 8\n", "YES\n2 4 6\n", "YES\n8 2 2\n"),
    "1016_D": ("2 2\n0 0\n0 0\n", "YES\n0 0\n0 0\n", "YES\n1 1\n1 1\n", "YES\n0  0\n0  0\n"),
    "1360_G": ("1\n2 2 1 1\n", "YES\n10\n01\n", "YES\n01\n10\n", "YES\n10\n01\n"),
    "361_A": ("2 4\n", "4 0\n0 4\n", "2 2\n2 2\n", " 4 0\n0 4\n"),
    "244_A": ("2 2\n1 2\n", "1 3\n2 4\n", "1 4\n2 3\n", "3 1\n4 2\n"),
    "1102_B": ("4 2\n1 2 1 2\n", "YES\n1 1 2 2\n", "YES\n1 2 2 1\n", "YES\n2 2 1 1\n"),
    "1408_A": ("1\n3\n1 1 1\n2 2 2\n3 3 3\n", "1 2 3\n", "1 3 2\n", "1  2 3\n"),
    "988_A": ("4 2\n1 2 3 4\n", "YES\n1 2\n", "YES\n3 4\n", "YES\n2 1\n"),
    "1399_D": ("1\n5\n00110\n", "2\n1 2 1 2 1\n", "2\n1 2 2 1 1\n", "2\n2 1 2 1 2\n"),
    "1323_A": ("1\n3\n2 4 6\n", "2\n2 3\n", "1\n1\n", "2\n3 2\n"),
    "1073_A": ("4\nabba\n", "YES\nab\n", "YES\nba\n", "YES\nab\n"),
    "1380_A": ("1\n4\n1 4 2 3\n", "YES\n1 2 3\n", "YES\n1 2 4\n", "YES\n1 2 3\n"),
    "1095_C": ("10 4\n", "YES\n1 1 4 4\n", "YES\n2 2 2 4\n", "YES\n4 4 1 1\n"),
    "1352_G": ("1\n4\n", "3 1 4 2\n", "2 4 1 3\n", "3 1 4 2\n"),
    "482_A": ("4 2\n", "1 3 2 4\n", "4 2 3 1\n", "1  3 2 4\n"),
    "1339_B": ("1\n4\n1 2 3 4\n", "2 3 1 4\n", "3 2 4 1\n", "2 3 1 4\n"),
    "1371_D": ("1\n2 2\n", "0\n10\n01\n", "0\n01\n10\n", "0\n10\n01\n"),
    "1038_B": ("4\n", "Yes\n1 2\n3 1 3 4\n", "Yes\n1 4\n3 1 2 3\n", "Yes\n3 4 3 1\n1 2\n"),
    "1051_B": ("1 4\n", "YES\n1 2\n3 4\n", "YES\n1 4\n2 3\n", "YES\n4 3\n2 1\n"),
    "545_B": ("00\n11\n", "01\n", "10\n", "01\n"),
    "1352_F": ("1\n1 1 1\n", "0011\n", "1100\n", "0011\n"),
}


CANONICAL_PROBES.update({'1023_C': ('6 4\n(()())\n', '(())', '()()', '(())'), '1047_A': ('9\n', '1 1 7', '1 4 4', '7 1 1'), '1088_A': ('10\n', '6 3', '8 2', '6 3'), '1093_B': ('1\nabc\n', 'abc', 'acb', 'abc'), '1325_A': ('1\n12\n', '1 11', '4 8', '11 1'), '1332_B': ('1\n3\n6 10 15\n', '2\n1 2 2', '2\n1 1 2', '2\n2 1 1'), '1360_F': ('1\n1 2\naa\n', 'aa', 'ab', 'aa'), '1430_A': ('1\n15\n', '5 0 0', '0 3 0', '5 0 0'), '1436_B': ('1\n2\n', '1 1\n1 1', '1 4\n4 1', '1 1\n1 1'), '1450_A': ('1\n3\nabc\n', 'abc', 'acb', 'abc'), '1497_C2': ('1\n12 3\n', '4 4 4', '6 3 3', '4 4 4'), '1549_A': ('1\n7\n', '2 3', '2 6', '2 3'), '1559_B': ('1\n3\n???\n', 'RBR', 'BRB', 'RBR'), '1606_A': ('1\nab\n', 'aa', 'bb', 'aa'), '544_B': ('2 1\n', 'YES\nLS\nSS', 'YES\nSS\nSL', 'YES\nLS\nSS'), '710_C': ('3\n', '8 1 6\n3 5 7\n4 9 2', '4 3 8\n9 5 1\n2 7 6', '8 1 6\n3 5 7\n4 9 2'), '1096_A': ('1\n1 6\n', '1 2', '2 4', '1 2'), '1269_A': ('1\n', '9 8', '25 24', '9 8'), '1326_A': ('1\n2\n', '23', '29', '23'), '1463_B': ('1\n2\n4 4\n', '4 4', '2 4', '4 4'), '1511_B': ('1\n2 2 1\n', '12 15', '14 21', '12 15'), '1554_D': ('1\n2\n', 'ab', 'ac', 'ab'), '1594_A': ('1\n3\n', '1 2', '-2 3', '1 2'), '1608_A': ('1\n3\n', '2 3 4', '3 4 5', '2 3 4'), '1617_B': ('1\n12\n', '5 6 1', '3 8 1', '6 5 1'), '199_A': ('8\n', '0 0 8', '0 3 5', '8 0 0'), '472_A': ('20\n', '4 16', '8 12', '16 4'), '534_A': ('4\n', '4\n3 1 4 2', '4\n2 4 1 3', '4\n3 1 4 2')})

CANONICAL_PROBES.update({'1196_C': ('1\n1\n0 0 1 1 1 1\n', '1 0 0', '1 1 2', '1  0 0'), '1092_A': ('1\n7 3\n', 'aaabbcc', 'aabbbcc', 'abcabac'), '1413_A': ('1\n2\n1 1\n', '1 -1', '2 -2', '1  -1'), '1520_C': ('1\n3\n', '2 9 7\n4 6 3\n1 8 5', '1 4 2\n8 6 9\n5 3 7', '2 9 7\n4 6 3\n1 8 5')})

def _canonicalizer_audit(base: Any, task: Any, scratch: Path) -> dict[str, Any]:
    from oat_drgrpo.constructive_code import ReleasedCheckerDecision, sha256_bytes
    stdin, first, second, equivalent = CANONICAL_PROBES[task.problem_id]
    rows = []
    for name, text in (("first", first), ("different", second), ("equivalent", equivalent)):
        result = base._run_checker(task.checker_binary, stdin.encode(), text.encode(), scratch)
        if not result.accepted:
            raise ValueError(f"released checker rejected canonicalizer {name} probe: {result.message}")
        decision = ReleasedCheckerDecision(task.checker_sha256, sha256_bytes(stdin.encode()), sha256_bytes(text.encode()), True, result.returncode)
        witness = base.canonicalize_task_witness(problem_id=task.problem_id, adapter_id=task.adapter_id, input_data=stdin, output=text, decision=decision)
        rows.append({"kind": name, "input": stdin, "output": text, "canonical_key": witness.canonical_key, "canonical_json": witness.canonical_json})
    if rows[0]["canonical_key"] == rows[1]["canonical_key"] or rows[0]["canonical_key"] != rows[2]["canonical_key"]:
        raise ValueError("canonicalizer merged distinct witnesses or split equivalent witnesses")
    invalid = base._run_checker(task.checker_binary, stdin.encode(), b"invalid\n", scratch)
    if invalid.accepted or invalid.timed_out:
        raise ValueError("released checker failed malformed-output negative probe")
    return {"status": "pass", "test_only_not_dataset_inputs": True, "examples": rows, "malformed_output_rejected": True}


def _prepare_inputs(base: Any, task_dir: Path, record: dict[str, Any], build_root: Path) -> dict[str, Any]:
    from oat_drgrpo.constructive_code_wider_adapters_20260921 import input_cases
    raw_file = task_dir / "inputs.jsonl.gz"
    if digest(raw_file) != record.get("source_suite", record["suite"])["compressed_jsonl_sha256"]:
        raise ValueError("original source suite bytes drifted")
    rows = [json.loads(line) for line in gzip.open(raw_file, "rt")]
    validator = build_root / record["source_problem_id"].lower() / "validator"
    build = base._compile_checker(task_dir / "validator.cpp", validator, record["validator_sha256"])
    retained, checks = [], []
    for row in rows:
        independent_error = None
        try:
            input_cases(record["source_problem_id"], row["stdin"])
        except ValueError as error:
            independent_error = str(error)
        try:
            completed = subprocess.run([str(validator)], input=row["stdin"], text=True, capture_output=True, timeout=3)
            validator_returncode, validator_message = completed.returncode, (completed.stdout + completed.stderr)[-1000:]
        except subprocess.TimeoutExpired:
            validator_returncode, validator_message = -999, "validator timeout"
        checks.append({"source_test_index": row["test_index"], "input_sha256": row["input_sha256"], "independent_error": independent_error, "validator_returncode": validator_returncode, "validator_message": validator_message, "retained": independent_error is None})
        if independent_error is None:
            retained.append(row["stdin"])
    if not retained:
        raise ValueError("no source inputs satisfy original task constraints")
    record["source_suite"] = record.get("source_suite", record["suite"])
    record["suite"] = _write_suite(task_dir / "admitted_inputs.jsonl.gz", retained)
    record["suite_file"] = "admitted_inputs.jsonl.gz"
    audit = {"source_inputs": len(rows), "retained_inputs": len(retained), "invalid_source_inputs": len(rows) - len(retained), "selection_rule": "retain exactly source inputs passing independent statement-constraint parser; never filter on policy/model success", "validator_build": build, "validator_disagreements": sum(row["retained"] and row["validator_returncode"] != 0 for row in checks), "checks": checks}
    write_json(task_dir / "input_audit.json", audit)
    return audit



VALIDATOR_COMPATIBILITY_545 = {
    "validator_sha256": "e4e76532b7fda0695430bd2227cc649522b2dc0604ca67328a4ec43c7d403cb7",
    "checker_sha256": "853ea21db59193b48b790083ec189eae108c1724f28ededbd978188df5be63cb",
    "source_inputs_compressed_sha256": "caf3a53ed8975ed373cfb917ae1f266ae98819f27f83de97b41417b5433245c6",
    "reason": "Released input validator interprets ^ and $ literally under pinned testlib, rejecting ordinary binary lines and accepting literal-anchor nonbinary lines.",
    "independent_contract": "Exactly two lines; each binary-only; equal lengths; length in [1,100000]. All original source inputs retained byte-for-byte.",
    "independent_review": "compute_fit read-only review 2026-09-21: original validator rejects 00\\n11\\n with code3 and accepts ^00$\\n^11$\\n with code0; all61 source inputs satisfy the original problem constraints.",
}


def _known_validator_compatibility(record: Mapping[str, Any], input_audit: Mapping[str, Any]) -> dict[str, Any] | None:
    if record["source_problem_id"] != "545_B" or not input_audit["validator_disagreements"]:
        return None
    expected = VALIDATOR_COMPATIBILITY_545
    source_suite = record.get("source_suite", record["suite"])
    if record["validator_sha256"] != expected["validator_sha256"] or record["checker_sha256"] != expected["checker_sha256"] or source_suite["compressed_jsonl_sha256"] != expected["source_inputs_compressed_sha256"]:
        raise ValueError("545_B source-reviewed validator exception identity drift")
    if input_audit["source_inputs"] != 61 or input_audit["retained_inputs"] != 61 or input_audit["invalid_source_inputs"] != 0:
        raise ValueError("545_B source-reviewed validator exception input count drift")
    if any(row["independent_error"] is not None or not row["retained"] or row["validator_returncode"] != 3 or "doesn't correspond to pattern \"^[01]{1,100000}$\"" not in row["validator_message"] for row in input_audit["checks"]):
        raise ValueError("545_B validator failure differs from reviewed literal-anchor issue")
    return dict(expected)


def resolve_validator_review(args: argparse.Namespace) -> None:
    """Resolve only the independently reviewed 545_B input-validator defect."""
    from oat_drgrpo.constructive_code_wider_adapters_20260921 import input_cases
    root = args.output; task_dir = root / "545_b"
    manifest = json.loads((root / "manifest.json").read_text())
    record = json.loads((task_dir / "task.json").read_text())
    audit = json.loads((task_dir / "admission_audit.json").read_text())
    expected_violation = "released validator disagrees on independently valid inputs; requires source review"
    if audit["status"] != "fail" or audit["violations"] != [expected_violation] or audit["tpr"] != 1 or audit["tnr"] != 1:
        raise ValueError("545_B failed admission differs from the independently reviewed case")
    compatibility = _known_validator_compatibility(record, audit["input_audit"])
    source_rows = [json.loads(line) for line in gzip.open(task_dir / "inputs.jsonl.gz", "rt")]
    admitted_rows = [json.loads(line) for line in gzip.open(task_dir / record["suite_file"], "rt")]
    if [r["stdin"] for r in source_rows] != [r["stdin"] for r in admitted_rows] or digest(task_dir / "inputs.jsonl.gz") != compatibility["source_inputs_compressed_sha256"]:
        raise ValueError("545_B inputs changed during validator review")
    for row in source_rows:
        input_cases("545_B", row["stdin"])
    base = configure_replay()
    validator = args.build_root / "545_b_review" / "validator"
    build = base._compile_checker(task_dir / "validator.cpp", validator, compatibility["validator_sha256"])
    probes = []
    for text, expected_code in (("00\n11\n", 3), ("^00$\n^11$\n", 0)):
        completed = subprocess.run([str(validator)], input=text, text=True, capture_output=True, timeout=3)
        if completed.returncode != expected_code:
            raise ValueError("545_B literal-anchor incompatibility failed reproduction")
        probes.append({"stdin": text, "returncode": completed.returncode, "stderr": completed.stderr})
    review = {"status": "resolved_input_validator_incompatibility", **compatibility, "all_source_inputs_rechecked": 61, "all_source_input_bytes_retained": True, "output_checker_modified": False, "probes": probes, "validator_build": build, "pre_review_admission_sha256": digest(task_dir / "admission_audit.json")}
    write_json(task_dir / "pre_review_admission_audit.json", audit)
    write_json(task_dir / "validator_compatibility_review.json", review)
    audit.update({"status": "pass", "violations": [], "resolved_validator_incompatibility": review, "validator_compatibility_review_sha256": digest(task_dir / "validator_compatibility_review.json")})
    write_json(task_dir / "admission_audit.json", audit)
    record.update({"admission_status": "admitted", "admission_audit_sha256": digest(task_dir / "admission_audit.json"), "validator_compatibility_review_sha256": digest(task_dir / "validator_compatibility_review.json")})
    record["task_record_sha256"] = canonical_hash({k:v for k,v in record.items() if k != "task_record_sha256"})
    write_json(task_dir / "task.json", record)
    for summary in manifest["tasks"]:
        if summary["source_problem_id"] == "545_B":
            summary.update({"status": "admitted", "task_record_sha256": record["task_record_sha256"], "admission_audit_sha256": record["admission_audit_sha256"]})
    manifest.update({"admitted_problem_ids": [r["source_problem_id"] for r in manifest["tasks"] if r["status"] == "admitted"], "tasks_sha256": canonical_hash(manifest["tasks"])})
    write_json(root / "manifest.json", manifest)
    summary = json.loads((root / "admission_summary.json").read_text())
    summary.update({"admitted": len(manifest["admitted_problem_ids"]), "admitted_problem_ids": manifest["admitted_problem_ids"]})
    summary["tasks"] = [{k:v for k,v in audit.items() if k not in ("input_audit", "canonicalizer_audit", "checker_build")} if r["source_problem_id"] == "545_B" else r for r in summary["tasks"]]
    write_json(root / "admission_summary.json", summary)


def audit_slate(args: argparse.Namespace) -> None:
    import evaluate_constructive_code_v3_coder_viability as prior
    base = configure_replay()
    if digest(base.TESTLIB_ROOT / "testlib.h") != base.TESTLIB_SHA256:
        raise ValueError("pinned testlib source drift")
    manifest = json.loads((args.output / "manifest.json").read_text())
    runtime = _runtime(base, args)
    results = []
    for summary in manifest["tasks"]:
        problem = summary["source_problem_id"]
        task_dir = args.output / summary["relative_path"]
        record = json.loads((task_dir / "task.json").read_text())
        result = {"source_problem_id": problem, "family": record["witness_family"], "status": "fail", "violations": []}
        try:
            input_audit = _prepare_inputs(base, task_dir, record, args.build_root)
            record["task_record_sha256"] = canonical_hash({k:v for k,v in record.items() if k != "task_record_sha256"})
            write_json(task_dir / "task.json", record)
            task, checker_build = _task_from_record(base, task_dir, record, args.build_root)
            canonical_audit = _canonicalizer_audit(base, task, args.scratch_root)
            refs = [json.loads(line) for line in (task_dir / "py3_replays.jsonl").read_text().splitlines()]
            replays = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(base._replay_submission, task=task, submission=base.Submission(row["code"], row["known_label"], row["submission_sha256"]), launcher=args.launcher, runtime_root=args.runtime_root, scratch_root=args.scratch_root): row for row in refs}
                for future in as_completed(futures):
                    ref = futures[future]
                    try:
                        replay = future.result()
                        violations = prior._hard_replay_violations(replay)
                        # A deliberately incorrect source program may legitimately
                        # exceed its execution bound; that is an audited rejection.
                        if ref["known_label"] == "incorrect":
                            violations = [v for v in violations if v != "candidate execution-bound violation"]
                        replay["audit_violations"] = violations
                    except Exception as error:
                        replay = {"source_problem_id": problem, "known_label": ref["known_label"], "submission_sha256": ref["submission_sha256"], "released_checker_accepted": False, "wrapper_accepted": False, "behavior_key": None, "audit_violations": [f"worker exception: {type(error).__name__}: {error}"]}
                    replays.append(replay)
            replays.sort(key=lambda row: (row["known_label"], row["submission_sha256"]))
            write_jsonl(task_dir / "audit_replays.jsonl", replays)
            correct = [r for r in replays if r["known_label"] == "correct"]
            incorrect = [r for r in replays if r["known_label"] == "incorrect"]
            tpr = sum(r["released_checker_accepted"] and r["wrapper_accepted"] for r in correct) / len(correct)
            tnr = sum(not r["released_checker_accepted"] for r in incorrect) / len(incorrect)
            violations = [f"{r['submission_sha256']}: {v}" for r in replays for v in r["audit_violations"]]
            compatibility = _known_validator_compatibility(record, input_audit)
            if input_audit["validator_disagreements"] and compatibility is None:
                violations.append("released validator disagrees on independently valid inputs; requires source review")
            if tpr < 0.9 or tnr < 0.9:
                violations.append("known-source TPR or TNR below 0.9")
            result.update({"status": "pass" if not violations else "fail", "violations": violations, "tpr": tpr, "tnr": tnr, "positive_replays": len(correct), "negative_replays": len(incorrect), "reference_distinct_modes": len({r["behavior_key"] for r in correct if r["behavior_key"]}), "input_audit": input_audit, "resolved_validator_incompatibility": compatibility, "canonicalizer_audit": canonical_audit, "checker_build": checker_build, "replays_sha256": digest(task_dir / "audit_replays.jsonl")})
        except Exception as error:
            result["violations"].append(f"{type(error).__name__}: {error}")
        write_json(task_dir / "admission_audit.json", result)
        record["admission_status"] = "admitted" if result["status"] == "pass" else "quarantined"
        record["admission_audit_sha256"] = digest(task_dir / "admission_audit.json")
        record["task_record_sha256"] = canonical_hash({k:v for k,v in record.items() if k != "task_record_sha256"})
        write_json(task_dir / "task.json", record)
        summary.update({"status": record["admission_status"], "task_record_sha256": record["task_record_sha256"], "admission_audit_sha256": record["admission_audit_sha256"]})
        results.append(result)
        manifest.update({"status": "audit_in_progress", "tasks_sha256": canonical_hash(manifest["tasks"])})
        write_json(args.output / "manifest.json", manifest)
        print(f"[wider-audit] task={problem} status={result['status']} tpr={result.get('tpr')} tnr={result.get('tnr')} violations={result['violations'][:2]}", flush=True)
    manifest.update({"status": "audited", "tasks_sha256": canonical_hash(manifest["tasks"]), "admitted_problem_ids": [r["source_problem_id"] for r in results if r["status"] == "pass"], "runtime": runtime, "audit_completed_at": datetime.now(timezone.utc).isoformat()})
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "admission_summary.json", {"schema_version": SCHEMA, "candidates": len(results), "admitted": len(manifest["admitted_problem_ids"]), "admitted_problem_ids": manifest["admitted_problem_ids"], "tasks": [{k:v for k,v in r.items() if k not in ("input_audit", "canonicalizer_audit", "checker_build")} for r in results], "runtime": runtime, "model_sampling_performed": False})


def load_admitted_tasks(slate_root: Path, build_root: Path, problem_ids: Sequence[str] | None = None, *, allow_heldout: bool = False):
    base = configure_replay()
    manifest = json.loads((slate_root / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA or manifest["status"] != "audited" or manifest["tasks_sha256"] != canonical_hash(manifest["tasks"]):
        raise ValueError("wider slate is not an intact completed admission audit")
    if manifest.get("split_reservation_sha256") != digest(slate_root / "split_reservation.json"):
        raise ValueError("prospective split reservation binding drift")
    if manifest.get("extension_reservation_sha256") and digest(slate_root / "extension_reservation.json") != manifest["extension_reservation_sha256"]:
        raise ValueError("heldout extension reservation binding drift")
    wanted = list(problem_ids) if problem_ids is not None else list(manifest.get("default_training_problem_ids", [p for p in manifest["admitted_problem_ids"] if allow_heldout or p not in (*NEW_HELDOUT, *EXTENSION_HELDOUT)]))
    if not allow_heldout and set(wanted) & set((*ORIGINAL_HELDOUT, *NEW_HELDOUT, *EXTENSION_HELDOUT)):
        raise ValueError("heldout tasks require explicit final-evaluation authorization")
    if not wanted or len(wanted) != len(set(wanted)) or set(wanted) - set(manifest["admitted_problem_ids"]):
        raise ValueError("requested task is duplicate, absent, or not admitted")
    by_id = {r["source_problem_id"]: r for r in manifest["tasks"]}
    tasks, public, builds = [], [], []
    for problem in wanted:
        summary = by_id[problem]; task_dir = slate_root / summary["relative_path"]
        record = json.loads((task_dir / "task.json").read_text())
        if record["task_record_sha256"] != summary["task_record_sha256"] or digest(task_dir / "admission_audit.json") != record["admission_audit_sha256"]:
            raise ValueError("task admission binding drift")
        audit = json.loads((task_dir / "admission_audit.json").read_text())
        if audit["status"] != "pass" or audit["violations"]:
            raise ValueError("nonpassing task admission audit")
        task, build = _task_from_record(base, task_dir, record, build_root)
        tasks.append(task); builds.append(build)
        public.append({key: record[key] for key in ("source_problem_id", "problem_key", "statement", "statement_sha256", "witness_family", "task_adapter", "split", "continuity_anchor", "checker_sha256")})
    return tasks, public, builds, manifest


@dataclass
class PolicyTask:
    task_id: str
    prompt: str
    family: str
    split: str
    metadata: dict[str, Any]
    _task: Any
    _base: Any
    _runtime_args: Any

    def verify(self, text: str) -> dict[str, Any]:
        from types import SimpleNamespace
        import evaluate_constructive_code_v3_coder_viability as prior
        import evaluate_constructive_code_pilot_20260921 as pilot
        code, stripped = prior._strip_exact_surrounding_fence(text)
        candidate = {"row_index": 0, "sample_index": 0, "request_seed": 0, "emitted_text_sha256": raw_sha256(text), "executed_source_sha256": raw_sha256(code), "fence_stripped": stripped, "token_count": 0, "finish_reason": "adapter_not_supplied", "code": code}
        try:
            attempt = pilot._verify_candidate(self._base, self._runtime_args, candidate, self._task)
        except Exception as error:
            attempt = pilot._attempt(candidate, self._task, None, f"{type(error).__name__}: {error}")
        # Bounded rejection of an initially wrong program is a reward outcome,
        # not a broken harness. An accepted program failing its recheck remains
        # an integrity violation and cannot be softened.
        hard = list(attempt["hard_violations"])
        if not attempt.get("stability_recheck_required"):
            hard = [value for value in hard if value != "candidate execution-bound violation"]
        return {"accepted": attempt["accepted"], "canonical_key": attempt["canonical_key"], "hard_violations": hard, "receipt": attempt}


def load_tasks(adapter_config: Mapping[str, Any]) -> list[PolicyTask]:
    from types import SimpleNamespace
    import evaluate_constructive_code_v3_coder_viability as prior
    base = configure_replay()
    values = {key: Path(adapter_config[key]) for key in ("slate_root", "build_root", "image", "runtime_root", "launcher", "scratch_root")}
    args = SimpleNamespace(**values)
    runtime = _runtime(base, args)
    tasks, public, builds, manifest = load_admitted_tasks(args.slate_root, args.build_root, adapter_config.get("problem_ids"), allow_heldout=bool(adapter_config.get("allow_heldout", False)))
    return [PolicyTask(task_id=row["source_problem_id"], prompt=prior._prompt(row["statement"]), family=row["witness_family"], split=row["split"], metadata={**{k:v for k,v in row.items() if k != "statement"}, "suite_id": task.suite_id, "suite_sha256": task.suite_sha256, "checker_build": build, "runtime": runtime}, _task=task, _base=base, _runtime_args=args) for task,row,build in zip(tasks,public,builds)]


def dataset_identity(adapter_config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(adapter_config["slate_root"])
    manifest = json.loads((root / "manifest.json").read_text())
    return {"schema_version": SCHEMA, "manifest_sha256": digest(root / "manifest.json"), "selected_problem_ids": adapter_config.get("problem_ids", manifest.get("default_training_problem_ids", [p for p in manifest.get("admitted_problem_ids", []) if adapter_config.get("allow_heldout", False) or p not in (*NEW_HELDOUT, *EXTENSION_HELDOUT)])), "split_reservation_sha256": manifest.get("split_reservation_sha256"), "source_revisions": manifest["source_revisions"], "normalized_statement_split_unit": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fetch", "audit", "finalize", "resolve-validator"), required=True)
    parser.add_argument("--cohort", choices=("initial", "reserved", "heldout-extension"), default="initial")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", type=Path, default=ROOT / "var/images/python-3.10-slim-c1e4e6c01eb4.sqsh")
    parser.add_argument("--runtime-root", type=Path, default=Path("/tmp/constructive_wider_20260921/runtime"))
    parser.add_argument("--launcher", type=Path, default=Path("/tmp/constructive_wider_20260921/launcher"))
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp/constructive_wider_20260921/scratch"))
    parser.add_argument("--build-root", type=Path, default=Path("/tmp/constructive_wider_20260921/build"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "fetch":
        fetch_and_materialize(args)
    elif args.phase == "audit":
        audit_slate(args)
        freeze_statement_repairs(args.output)
    elif args.phase == "resolve-validator":
        resolve_validator_review(args)
        freeze_statement_repairs(args.output)
    else:
        freeze_statement_repairs(args.output)


if __name__ == "__main__":
    main()
