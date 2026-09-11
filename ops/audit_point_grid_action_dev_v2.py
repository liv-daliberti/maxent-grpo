#!/usr/bin/env python3
"""Independently replay-audit the frozen Point grid-action v2 receipt."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _hash_tree(root: Path) -> str:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = "./" + path.relative_to(root).as_posix()
        lines.append(f"{_sha256(path)}  {relative}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-parent", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--admission-audit", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("fresh Point grid v2 audit output is required")
    source_root = (args.source_parent / "src").resolve()
    sys.path.insert(0, str(source_root))
    os.environ["OAT_ZERO_REPO_ROOT"] = str(Path.cwd().resolve())
    os.environ["OAT_ZERO_MAZE_WORKER_PYTHON"] = str(args.worker_python.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")

    from datasets import load_from_disk
    from oat_drgrpo.point_maze_grid import (
        adapt_point_maze_spec,
        parse_point_grid_program,
        parse_point_grid_spec,
    )
    from oat_drgrpo.point_maze_grid_process import PointGridVerifierProcess

    evaluator_path = args.execution_root / "evaluate_point_grid_viability_dev_v2.py"
    module_spec = importlib.util.spec_from_file_location(
        "frozen_point_grid_v2_evaluator", evaluator_path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load frozen Point evaluator helpers")
    evaluator = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(evaluator)

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    dataset_identity = json.loads(
        (args.data_root / "identity.json").read_text(encoding="utf-8")
    )
    rows = load_from_disk(str(args.data_root / "dev"))["multi_answer"].to_list()
    errors: list[str] = []
    checks = {
        "job_identity_matches": receipt.get("job_id") == identity.get("job_id"),
        "source_hash_matches": (
            receipt.get("source_hash") == identity.get("source_hash")
            == _hash_tree(source_root)
        ),
        "execution_hash_matches": (
            receipt.get("execution_hash") == identity.get("execution_hash")
            == _hash_tree(args.execution_root)
        ),
        "protocol_hash_matches": (
            _sha256(args.protocol) == receipt.get("protocol_sha256")
            == identity.get("protocol_sha256")
        ),
        "admission_hash_matches": (
            _sha256(args.admission_audit) == receipt.get("admission_audit_sha256")
            == identity.get("admission_audit_sha256")
        ),
        "dataset_identity_matches": (
            receipt.get("dataset_identity_sha256") == _canonical(dataset_identity)
        ),
        "development_rows_match": receipt.get("data_split_sha256") == _canonical(rows),
        "attempt_count_is_256": len(receipt.get("attempts", [])) == 256,
        "information_boundary_declared": receipt.get("information_boundary") == {
            "certified_route_programs_loaded": False,
            "development_only": True,
            "evaluation_prompts_loaded": False,
            "guided_decoder_action_mask": "wall legality and no revisits only",
            "guided_decoder_filters_by_goal": False,
            "guided_decoder_labels_route_identity": False,
            "route_catalogue_loaded": False,
        },
    }

    attempts_by_row = {index: [] for index in range(len(rows))}
    for attempt in receipt.get("attempts", []):
        row_index = int(attempt["row_index"])
        if row_index not in attempts_by_row:
            errors.append(f"out-of-range row index {row_index}")
            continue
        attempts_by_row[row_index].append(attempt)

    replay_prompt_results = []
    verifier = PointGridVerifierProcess(
        timeout_seconds=25.0,
        worker_python=args.worker_python,
    )
    try:
        for row_index, row in enumerate(rows):
            raw_spec = adapt_point_maze_spec(json.loads(row["answer"]))
            parsed_spec = parse_point_grid_spec(raw_spec)
            minimum = evaluator._shortest_steps(raw_spec)
            maximum = min(minimum + 4, raw_spec["max_actions"])
            action_space = set(evaluator._simple_path_choices(raw_spec, minimum, maximum))
            goal_count = evaluator._goal_choice_count(raw_spec, sorted(action_space))
            attempts = sorted(
                attempts_by_row[row_index], key=lambda item: int(item["sample_index"])
            )
            if [int(item["sample_index"]) for item in attempts] != list(range(1, 65)):
                errors.append(f"row {row_index} sample indices differ from 1..64")
            keys = []
            cache = {}
            for attempt in attempts:
                text = str(attempt["text"])
                if text not in action_space:
                    errors.append(f"row {row_index} text outside frozen legal space")
                try:
                    parse_point_grid_program(text, parsed_spec)
                except ValueError:
                    errors.append(f"row {row_index} legal-space text failed parser")
                if text not in cache:
                    validation = verifier.validate(text, raw_spec)
                    cache[text] = validation.canonical_key if validation else None
                key = cache[text]
                keys.append(key)
                if key != attempt.get("canonical_key") or bool(key) != bool(
                    attempt.get("verified")
                ):
                    errors.append(
                        f"row {row_index} sample {attempt['sample_index']} replay mismatch"
                    )
            prefix = [key for key in keys[:16] if key]
            full = [key for key in keys if key]
            counts = Counter(full)
            replayed = {
                "verified_in_prefix": len(prefix),
                "verified_in_full_sample": len(full),
                "distinct_keys_in_full_sample": len(counts),
                "legal_simple_path_action_count": len(action_space),
                "unlabeled_goal_ending_action_count": goal_count,
                "sampled_length_bounds": [minimum, maximum],
            }
            replay_prompt_results.append(replayed)
            claimed = receipt["prompt_results"][row_index]
            for field, value in replayed.items():
                if claimed.get(field) != value:
                    errors.append(f"row {row_index} summary mismatch for {field}")
    finally:
        verifier.close()

    summary = {
        "prompt_count": len(rows),
        "prefix_success_prompts": sum(
            item["verified_in_prefix"] > 0 for item in replay_prompt_results
        ),
        "multimode_prompts": sum(
            item["distinct_keys_in_full_sample"] >= 2
            for item in replay_prompt_results
        ),
        "verified_completions": sum(
            item["verified_in_full_sample"] for item in replay_prompt_results
        ),
    }
    checks["summary_replays_exactly"] = summary == receipt.get("summary")
    checks["frozen_decision_is_pass"] = (
        receipt.get("status") == "pass"
        and summary["prefix_success_prompts"] >= 2
        and summary["multimode_prompts"] >= 1
    )
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    payload = {
        "schema_version": "point-grid-action-development-audit-v2",
        "status": "pass" if not errors else "fail",
        "receipt_sha256": _sha256(args.receipt),
        "identity_sha256": _sha256(args.identity),
        "checks": checks,
        "replayed_summary": summary,
        "replayed_prompt_results": replay_prompt_results,
        "errors": errors,
    }
    _atomic(args.output, payload)
    print(json.dumps({"status": payload["status"], "summary": summary}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
