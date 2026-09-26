#!/usr/bin/env python3
"""Summarize the live, prospectively generated E50 route-bank corpus.

This is a read-only diagnostic.  It applies the same conservative signature
and exact-answer eligibility function used by E50I/J/K to every atomically
checkpointed corpus row.  The resulting JSON is for monitoring only and does
not grant training authority; E50G remains the sole final selector.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile
import os
from datetime import datetime
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
I_SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50i_third_conditioned_teacher_contingency.py"
)
ARTIFACTS = ROOT / "var/artifacts"
F_ROOT = ARTIFACTS / "e50f_conditioned_teacher_route_calibration_v1"
H_ROOT = ARTIFACTS / "e50h_second_conditioned_teacher_corpus_v1"
I_ROOT = ARTIFACTS / "e50i_third_conditioned_teacher_contingency_v1"
J_ROOT = ARTIFACTS / "e50j_fourth_conditioned_teacher_contingency_v1"
K_ROOT = ARTIFACTS / "e50k_fifth_conditioned_teacher_contingency_v1"
G_ROOT = (
    ARTIFACTS / "e50g_safe_signature_teacher_route_calibration_v1"
)


def _load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rows(final_path: pathlib.Path, partial_path: pathlib.Path) -> list[dict[str, Any]]:
    return _read_jsonl(final_path if final_path.is_file() else partial_path)


def _target(
    *,
    result_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    attempted_key: str,
    eligible_key: str,
    fallback: int | None,
) -> int | None:
    if result_path.is_file():
        result = _read_json(result_path)
        attempted = result.get(attempted_key)
        if isinstance(attempted, list):
            return len(attempted)
        generated = result.get("generation_problem_count")
        if isinstance(generated, int):
            return generated
    if checkpoint_path.is_file():
        checkpoint = _read_json(checkpoint_path)
        eligible = checkpoint.get(eligible_key)
        if isinstance(eligible, list):
            return 50 - len(eligible)
    return fallback


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ARTIFACTS / "e50_live_route_bank_summary.json",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    helper = _load_module("e50_live_route_bank_helper", I_SCRIPT)
    from oat_drgrpo.math_grader import boxed_reward_fn

    problems = helper.E50C._load_jsonl(helper.E50C.E47_PROBLEMS)
    for order, problem in enumerate(problems):
        problem["problem_order"] = order

    stage_specs = [
        {
            "stage": "E50F",
            "label": "First teacher pair",
            "rows": _rows(
                F_ROOT / "private/conditioned_teacher_records.jsonl",
                F_ROOT / "private/conditioned_teacher_records.partial.jsonl",
            ),
            "target": 50,
        },
        {
            "stage": "E50H",
            "label": "Second teacher pair",
            "rows": _rows(
                H_ROOT / "private/second_conditioned_records.jsonl",
                H_ROOT / "private/second_conditioned_records.partial.jsonl",
            ),
            "target": 50,
        },
        {
            "stage": "E50I",
            "label": "Third pair",
            "rows": _rows(
                I_ROOT / "private/third_conditioned_records.jsonl",
                I_ROOT / "private/third_conditioned_records.partial.jsonl",
            ),
            "target": _target(
                result_path=I_ROOT / "result.json",
                checkpoint_path=I_ROOT / "private/checkpoint_identity.json",
                attempted_key="attempted_problem_orders",
                eligible_key="prethird_eligible_problem_orders",
                fallback=None,
            ),
        },
        {
            "stage": "E50J",
            "label": "Fourth pair",
            "rows": _rows(
                J_ROOT / "private/fourth_conditioned_records.jsonl",
                J_ROOT / "private/fourth_conditioned_records.partial.jsonl",
            ),
            "target": _target(
                result_path=J_ROOT / "result.json",
                checkpoint_path=J_ROOT / "private/checkpoint_identity.json",
                attempted_key="attempted_problem_orders",
                eligible_key="prefourth_eligible_problem_orders",
                fallback=None,
            ),
        },
        {
            "stage": "E50K",
            "label": "Fifth pair",
            "rows": _rows(
                K_ROOT / "private/fifth_conditioned_records.jsonl",
                K_ROOT / "private/fifth_conditioned_records.partial.jsonl",
            ),
            "target": _target(
                result_path=K_ROOT / "result.json",
                checkpoint_path=K_ROOT / "private/checkpoint_identity.json",
                attempted_key="attempted_problem_orders",
                eligible_key="prefifth_eligible_problem_orders",
                fallback=None,
            ),
        },
    ]
    for stage in stage_specs:
        rows = stage.pop("rows")
        by_order = {int(row["problem_order"]): row for row in rows}
        if len(by_order) != len(rows):
            raise RuntimeError(f"{stage['stage']} duplicate problem order")
        stage["by_order"] = by_order

    first_eligible_stage: dict[int, str] = {}
    eligible_orders = []
    for order, problem in enumerate(problems):
        prior: list[dict[str, Any]] = []
        for stage in stage_specs:
            record = stage["by_order"].get(order)
            if record is None:
                continue
            eligible = helper._eligible(
                record=record,
                problem=problem,
                boxed_reward_fn=boxed_reward_fn,
                prior_records=tuple(prior),
            )
            if eligible and order not in first_eligible_stage:
                first_eligible_stage[order] = stage["stage"]
            prior.append(record)
        if order in first_eligible_stage:
            eligible_orders.append(order)

    cumulative = 0
    stages = []
    for stage in stage_specs:
        new_eligible = sum(
            source == stage["stage"]
            for source in first_eligible_stage.values()
        )
        cumulative += new_eligible
        by_order = stage["by_order"]
        stages.append(
            {
                "stage": stage["stage"],
                "label": stage["label"],
                "generated": len(by_order),
                "target": stage["target"],
                "error_count": sum(
                    bool(row.get("error")) for row in by_order.values()
                ),
                "new_eligible_count": new_eligible,
                "cumulative_eligible_count": cumulative,
            }
        )

    g_result_path = G_ROOT / "result.json"
    g_result = _read_json(g_result_path) if g_result_path.is_file() else {}
    selected = g_result.get("selected_source_indices")
    if not isinstance(selected, list):
        selected = []
    payload = {
        "schema": "e50_live_route_bank_summary_v1",
        "claim_scope": (
            "read-only live diagnostic; E50G alone grants training authority"
        ),
        "generated_at": datetime.now().astimezone().isoformat(),
        "problem_count": len(problems),
        "provisional_eligible_problem_count": len(eligible_orders),
        "provisional_eligible_problem_orders": eligible_orders,
        "stages": stages,
        "e50g": {
            "result_present": g_result_path.is_file(),
            "pass": g_result.get("pass"),
            "signature_candidate_count": g_result.get(
                "signature_candidate_count"
            ),
            "sound_bound_signature_menu_count": g_result.get(
                "sound_bound_signature_menu_count"
            ),
            "bidirectionally_executable_count": g_result.get(
                "bidirectionally_executable_count"
            ),
            "selected_problem_count": len(selected),
            "selected_source_indices": selected,
            "checks": g_result.get("checks") or {},
        },
    }
    _atomic_json(args.out.resolve(), payload)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
