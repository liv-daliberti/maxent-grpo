#!/usr/bin/env python3
"""Bounded calibration harness for a prospectively named E47 successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
E47 = ROOT / "var/artifacts/e47_math_strategy_calibration_v1"
DEFAULT_OUTPUT = ROOT / "var/artifacts/e47r_reasoned_math_strategy_calibration_v1"
PROTOCOL_ID = os.environ.get("E47_BOUNDED_PROTOCOL_ID", "E47R-CAL")
SCHEMA_PREFIX = os.environ.get(
    "E47_BOUNDED_SCHEMA_PREFIX", "e47r_reasoned_equivalence"
)
sys.path.insert(0, str(ROOT / "src"))

from oat_drgrpo.math_strategy_canonicalizer import MathStrategyCanonicalizer

from e47_calibration import _post_slurm_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    Path(temporary).replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    Path(temporary).replace(path)


def _prompt_tokens(problem_id: str) -> list[int]:
    return list(hashlib.sha256(problem_id.encode("utf-8")).digest())


def _schedule(
    *,
    injections: dict[str, dict[str, Any]],
    key: dict[str, str],
    policy_rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    by_id = {row["sample_id"]: row for row in injections.values()}
    anchor = by_id[key["anchor"]]
    exact = by_id[key["exact_duplicate"]]
    formatting = by_id[key["format_variant"]]
    lexical = by_id[key["lexical_paraphrase"]]
    policy = list(policy_rows)
    random.Random(471100).shuffle(policy)
    rounds = [
        [anchor, exact] + policy[:14],
        [formatting, lexical] + policy[14:28],
    ]
    for start in range(28, len(policy), 16):
        rounds.append(policy[start : start + 16])
    return [round_rows for round_rows in rounds if round_rows]


def run(output: Path, endpoint_path: Path, workers: int, timeout: int) -> None:
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    problems = {
        row["problem_id"]: row for row in _read_jsonl(E47 / "problems.jsonl")
    }
    policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(E47 / "validated_policy.blinded.jsonl"):
        policy[row["problem_id"]].append(row)
    injections: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read_jsonl(E47 / "injections.blinded.jsonl"):
        injections[row["problem_id"]][row["sample_id"]] = row
    keys: dict[str, dict[str, str]] = defaultdict(dict)
    for row in _read_jsonl(E47 / "private/injection_key.jsonl"):
        keys[row["problem_id"]][row["injection_kind"]] = row["sample_id"]
    result_root = output / "problems"
    result_root.mkdir(parents=True, exist_ok=True)

    def run_problem(problem_id: str) -> str:
        result_path = result_root / f"{problem_id}.json"
        if result_path.is_file():
            return problem_id
        records: list[dict[str, Any]] = []
        records_lock = threading.Lock()

        def transport(payload: dict[str, Any]) -> dict[str, Any]:
            response = _post_slurm_json(endpoint, payload, timeout)
            with records_lock:
                records.append(
                    {
                        "seed": payload["seed"],
                        "messages": payload["messages"],
                        "response": response,
                    }
                )
            return response

        canonicalizer = MathStrategyCanonicalizer(
            endpoint=f"slurm-relay://{PROTOCOL_ID.lower()}/v1",
            timeout_seconds=timeout,
            max_workers=1,
            max_item_chars=4000,
            transport=transport,
        )
        outcomes: dict[str, str | None] = {}
        round_records = []
        rounds = _schedule(
            injections=injections[problem_id],
            key=keys[problem_id],
            policy_rows=policy[problem_id],
        )
        for round_index, rows in enumerate(rounds):
            texts = [str(row["text"]) for row in rows]
            result_keys, diagnostics = canonicalizer.canonicalize(
                prompt_token_ids=[_prompt_tokens(problem_id)] * len(rows),
                prompt_texts=[str(problems[problem_id]["problem"])] * len(rows),
                response_texts=texts,
                task_reward_positive=[True] * len(rows),
                active_mask=[True] * len(rows),
                num_samples=len(rows),
            )
            for row, strategy_key in zip(rows, result_keys):
                outcomes[row["sample_id"]] = strategy_key
            round_records.append(
                {
                    "round": round_index,
                    "sample_ids": [row["sample_id"] for row in rows],
                    "outcome_keys": result_keys,
                    "diagnostics": diagnostics.__dict__,
                }
            )
        _write_json(
            result_path,
            {
                "schema": f"{SCHEMA_PREFIX}_problem_v1",
                "protocol_id": PROTOCOL_ID,
                "problem_id": problem_id,
                "rounds": round_records,
                "outcomes": outcomes,
                "canonicalizer_state": canonicalizer.state_dict(),
                "judge_records": records,
            },
        )
        return problem_id

    pending = [
        problem_id
        for problem_id in sorted(problems)
        if not (result_root / f"{problem_id}.json").is_file()
    ]
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(run_problem, problem_id): problem_id
            for problem_id in pending
        }
        for future in as_completed(futures):
            problem_id = futures[future]
            try:
                future.result()
                print(f"bounded calibration {problem_id}", flush=True)
            except Exception as exc:
                errors.append((problem_id, repr(exc)))
                print(f"ERROR {problem_id}: {exc}", file=sys.stderr, flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} bounded problems failed: {errors[:3]}")
    print(f"completed {len(pending)} pending bounded problems")


def analyze(output: Path) -> None:
    known_invalid_regressions = {
        "s_09ea047c9f33e571",
        "s_af55e947400d8999",
    }
    injection_key: dict[str, dict[str, str]] = defaultdict(dict)
    for row in _read_jsonl(E47 / "private/injection_key.jsonl"):
        injection_key[row["problem_id"]][row["injection_kind"]] = row["sample_id"]
    valid_rows = _read_jsonl(E47 / "validated_policy.blinded.jsonl")
    policy_ids = {row["sample_id"] for row in valid_rows}
    by_kind: dict[str, list[bool]] = defaultdict(list)
    policy_outcomes: dict[str, str | None] = {}
    diagnostics = Counter()
    per_problem = {}
    for problem_id in sorted(injection_key):
        record = json.loads(
            (output / "problems" / f"{problem_id}.json").read_text(
                encoding="utf-8"
            )
        )
        outcomes = record["outcomes"]
        anchor_key = outcomes.get(injection_key[problem_id]["anchor"])
        control_keys = {
            kind: outcomes.get(injection_key[problem_id][kind])
            for kind in (
                "exact_duplicate",
                "format_variant",
                "lexical_paraphrase",
            )
        }
        family_key = anchor_key or next(
            (value for value in control_keys.values() if value is not None),
            None,
        )
        problem_false = {}
        for kind in ("exact_duplicate", "format_variant", "lexical_paraphrase"):
            control_key = control_keys[kind]
            # Ambiguous/unassigned controls are false negatives and reduce
            # coverage; they are not a rewarded false-new event. A false new
            # requires two admitted keys for the same injected family.
            false_new = (
                family_key is not None
                and control_key is not None
                and control_key != family_key
            )
            by_kind[kind].append(false_new)
            problem_false[kind] = false_new
        for sample_id, strategy_key in outcomes.items():
            if sample_id in policy_ids:
                policy_outcomes[sample_id] = strategy_key
        for round_record in record["rounds"]:
            diagnostics.update(round_record["diagnostics"])
        per_problem[problem_id] = {
            "false_new": problem_false,
            "strategy_support": len(
                {
                    key
                    for sample_id, key in outcomes.items()
                    if sample_id in policy_ids and key is not None
                }
            ),
        }
    all_controls = [value for values in by_kind.values() for value in values]
    report = {
        "schema": f"{SCHEMA_PREFIX}_calibration_report_v1",
        "protocol_id": PROTOCOL_ID,
        "injection_false_new": {
            "false_new": sum(all_controls),
            "comparisons": len(all_controls),
            "rate": sum(all_controls) / len(all_controls),
            "by_kind": {
                kind: {
                    "false_new": sum(values),
                    "comparisons": len(values),
                    "rate": sum(values) / len(values),
                }
                for kind, values in sorted(by_kind.items())
            },
        },
        "diagnostics": dict(diagnostics),
        "policy_validator_positive": len(valid_rows),
        "policy_canonicalized": sum(
            key is not None for key in policy_outcomes.values()
        ),
        "known_invalid_regression": {
            "sample_ids": sorted(known_invalid_regressions),
            "admitted_ids": sorted(
                sample_id
                for sample_id in known_invalid_regressions
                if policy_outcomes.get(sample_id) is not None
            ),
        },
        "semantic_regressions": json.loads(
            (output / "semantic_regressions.json").read_text(
                encoding="utf-8"
            )
        ),
        "per_problem": per_problem,
        "manual_audit": {"status": "pending"},
        "gate_status": "pending_manual_audit",
    }
    _write_json(output / "analysis.json", report)
    _write_json(output / "policy_outcomes.json", policy_outcomes)
    print(json.dumps(report, indent=2, sort_keys=True))


def audit_packet(output: Path) -> None:
    problems = {
        row["problem_id"]: row for row in _read_jsonl(E47 / "problems.jsonl")
    }
    valid = _read_jsonl(E47 / "validated_policy.blinded.jsonl")
    text = {row["sample_id"]: row["text"] for row in valid}
    problem_for = {row["sample_id"]: row["problem_id"] for row in valid}
    outcomes = json.loads(
        (output / "policy_outcomes.json").read_text(encoding="utf-8")
    )
    by_problem_key: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample_id, key in outcomes.items():
        if key is not None:
            by_problem_key[problem_for[sample_id]][key].append(sample_id)
    same = []
    different = []
    for problem_id, by_key in by_problem_key.items():
        for ids in by_key.values():
            if len(ids) >= 2:
                same.append((problem_id, ids[0], ids[1], "judge_same"))
        keys = sorted(by_key)
        for left_index, left_key in enumerate(keys):
            for right_key in keys[left_index + 1 :]:
                different.append(
                    (
                        problem_id,
                        by_key[left_key][0],
                        by_key[right_key][0],
                        "judge_different",
                    )
                )
    rng = random.Random(471880)
    rng.shuffle(same)
    rng.shuffle(different)
    selected = same[:25] + different[:25]
    rng.shuffle(selected)
    packet = []
    hidden = []
    for index, (problem_id, left, right, source) in enumerate(selected):
        pair_id = f"b_{index:03d}_{hashlib.sha256((left+right).encode()).hexdigest()[:12]}"
        packet.append(
            {
                "pair_id": pair_id,
                "problem_id": problem_id,
                "problem": problems[problem_id]["problem"],
                "solution_a": text[left],
                "solution_b": text[right],
                "human_label": "",
                "human_reason": "",
            }
        )
        hidden.append(
            {
                "pair_id": pair_id,
                "pair_source": source,
                "judge_label": "same" if source == "judge_same" else "different",
            }
        )
    _write_jsonl(output / "manual_audit_packet.jsonl", packet)
    _write_jsonl(output / "private/manual_audit_key.jsonl", hidden)
    print(f"wrote {len(packet)} bounded audit pairs")


def audit_score(output: Path) -> None:
    packet = _read_jsonl(output / "manual_audit_packet.jsonl")
    labels_path = output / "manual_audit_labels.json"
    audit_provenance = {"kind": "inline_packet_labels"}
    if labels_path.is_file():
        labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels_payload, dict) or not isinstance(
            labels_payload.get("labels"), list
        ):
            raise RuntimeError("invalid bounded manual-audit labels file")
        labels = {
            str(row["pair_id"]): row for row in labels_payload["labels"]
        }
        if set(labels) != {row["pair_id"] for row in packet}:
            raise RuntimeError("bounded audit labels do not match the packet")
        for row in packet:
            label = labels[row["pair_id"]]
            row["human_label"] = str(label.get("label", ""))
            row["human_reason"] = str(label.get("reason", ""))
        audit_provenance = {
            key: value
            for key, value in labels_payload.items()
            if key != "labels"
        }
        audit_provenance["labels_sha256"] = hashlib.sha256(
            labels_path.read_bytes()
        ).hexdigest()
    key = {
        row["pair_id"]: row
        for row in _read_jsonl(output / "private/manual_audit_key.jsonl")
    }
    for row in packet:
        if row.get("human_label") not in {"same", "different", "uncertain"}:
            raise RuntimeError(f"unlabeled pair {row['pair_id']}")
    binary = [row for row in packet if row["human_label"] != "uncertain"]
    same = [row for row in binary if row["human_label"] == "same"]
    different = [row for row in binary if row["human_label"] == "different"]
    false_new = sum(key[row["pair_id"]]["judge_label"] == "different" for row in same)
    false_merge = sum(key[row["pair_id"]]["judge_label"] == "same" for row in different)
    manual = {
        "status": "complete",
        "audit_provenance": audit_provenance,
        "labeled_pairs": len(packet),
        "uncertain_pairs": len(packet) - len(binary),
        "same_pairs": len(same),
        "same_false_new": false_new,
        "same_false_new_rate": false_new / len(same) if same else None,
        "different_pairs": len(different),
        "different_false_merge": false_merge,
        "different_false_merge_rate": (
            false_merge / len(different) if different else None
        ),
    }
    report_path = output / "analysis.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["manual_audit"] = manual
    injection = report["injection_false_new"]
    diagnostics = report["diagnostics"]
    checks = {
        "exact_duplicate_false_new_zero": (
            injection["by_kind"]["exact_duplicate"]["false_new"] == 0
        ),
        "overall_false_new_at_most_0_05": injection["rate"] <= 0.05,
        "lexical_false_new_at_most_0_10": (
            injection["by_kind"]["lexical_paraphrase"]["rate"] <= 0.10
        ),
        "canonicalized_positive_fraction_at_least_0_50": (
            report["policy_canonicalized"]
            / max(1, report["policy_validator_positive"])
            >= 0.50
        ),
        "known_invalid_regressions_rejected": not report.get(
            "known_invalid_regression", {}
        ).get("admitted_ids", []),
        "manual_same_false_new_at_most_0_05": (
            manual["same_false_new_rate"] is not None
            and manual["same_false_new_rate"] <= 0.05
        ),
        "manual_different_false_merge_at_most_0_20": (
            (
                manual["different_false_merge_rate"] is not None
                and manual["different_false_merge_rate"] <= 0.20
            )
            or (
                manual["different_pairs"] == 0
                and bool(
                    report.get("semantic_regressions", {})
                    .get("checks", {})
                    .get("different_proofs_separated", False)
                )
            )
        ),
        "semantic_regressions_pass": bool(
            report.get("semantic_regressions", {}).get("pass", False)
        ),
        "no_structural_failure": diagnostics.get("judge_calls", 0) > 0,
    }
    report["gate_checks"] = checks
    report["gate_status"] = "pass" if all(checks.values()) else "fail"
    _write_json(report_path, report)
    print(json.dumps({"gate_status": report["gate_status"], **checks}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("run", "analyze", "audit-packet", "audit-score")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--endpoint", type=Path, default=E47 / "qwen72_endpoint.json"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.command == "run":
        run(args.output, args.endpoint, args.workers, args.timeout)
    elif args.command == "analyze":
        analyze(args.output)
    elif args.command == "audit-packet":
        audit_packet(args.output)
    else:
        audit_score(args.output)


if __name__ == "__main__":
    main()
