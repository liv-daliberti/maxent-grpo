#!/usr/bin/env python3
"""Trace-certify and conservatively veto E49H's frozen curated route pairs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from datasets import load_from_disk


ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
E49E_PATH = HERE / "materialize_e49e_trace_bank_data.py"
E49J_PATH = HERE / "calibrate_e49j_mathir_signature_veto.py"
V2B_PATH = HERE / "repair_e49e_singleton_gaps_v2b.py"


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e49e = _load_module("e49h_e49e_trace_validator", E49E_PATH)
e49j = _load_module("e49h_e49j_pair_veto", E49J_PATH)
v2b = _load_module("e49h_v2b_nonleak_validator", V2B_PATH)
v2b._configure()

SCHEMA = "e49h_curated_pair_certification_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_rows(source: pathlib.Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for split in ("train", "eval"):
        loaded = load_from_disk(str(source / split))
        dataset = loaded[next(iter(loaded))]
        for index, raw in enumerate(dataset):
            row = dict(raw)
            row_id = e49e.base._row_id(split, index, row)
            if row_id in rows:
                raise RuntimeError(f"duplicate source row: {row_id}")
            rows[row_id] = {
                "row_id": row_id,
                "split": split,
                "index": index,
                "problem": str(row["problem"]),
                "reference_answer": str(row["answer"]),
            }
    if len(rows) != 100:
        raise RuntimeError("E49H source is not the frozen 100-row toy set")
    return rows


def _row_cache(output: pathlib.Path, row_id: str) -> pathlib.Path:
    return (
        output
        / "request_cache"
        / _sha256_bytes(row_id.encode("utf-8"))[:20]
    )


def _sound_cache(
    output: pathlib.Path,
    row_id: str,
    strategy_id: str,
    role: str,
    seed: int,
    menu_sha256: str,
) -> pathlib.Path:
    return _row_cache(output, row_id) / (
        f"sound-{strategy_id}-{role}-{seed}-{menu_sha256[:16]}.json"
    )


def _veto_cache(
    output: pathlib.Path,
    row_id: str,
    role: str,
    seed: int,
    menu_sha256: str,
) -> pathlib.Path:
    return _row_cache(output, row_id) / (
        f"veto-{role}-{seed}-{menu_sha256[:16]}.json"
    )


def _cached_sound(
    *,
    path: pathlib.Path,
    row: dict[str, Any],
    menu: Any,
    strategy_id: str,
    role: str,
    seed: int,
    endpoint: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("trace_contract_version")
            != e49e.TRACE_CONTRACT_VERSION
            or record.get("candidate_menu_sha256") != menu.sha256
            or record.get("strategy_id") != strategy_id
            or record.get("role") != role
            or record.get("seed") != seed
        ):
            raise RuntimeError(f"E49H soundness cache changed: {path}")
        return record
    record = e49e._sound_request(
        endpoint=endpoint,
        model=model,
        problem=row["problem"],
        reference_answer=row["reference_answer"],
        menu=menu,
        strategy_id=strategy_id,
        role=role,
        seed=seed,
        timeout=timeout,
    )
    _write_json(path, record)
    return record


def _route_payload(
    menu: Any,
    strategy_id: str,
    sound_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    strategy = menu.strategy(strategy_id)
    if strategy is None:
        raise RuntimeError(f"missing strategy after parsing: {strategy_id}")
    action_map = {
        action.action_id: action.operation for action in menu.actions
    }
    return {
        "plan": strategy.plan,
        "action_combo": strategy.action_combo,
        "actions": [
            {
                "action_id": action_id,
                "operation": action_map[action_id],
            }
            for action_id in strategy.action_ids
        ],
        "independent_execution_traces": [
            audit["assessment"] for audit in sound_audits
        ],
    }


def _pair_payload(
    row: dict[str, Any],
    menu: Any,
    audits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "schema": "e49h_curated_pair_for_hardened_veto_v1",
        "pair_id": (
            "E49H_" + _sha256_bytes(row["row_id"].encode("utf-8"))[:20]
        ),
        "problem": row["problem"],
        "reference_answer": row["reference_answer"],
        "route_a": _route_payload(menu, "S1", audits["S1"]),
        "route_b": _route_payload(menu, "S2", audits["S2"]),
    }


def _cached_veto(
    *,
    path: pathlib.Path,
    pair: dict[str, Any],
    role: str,
    seed: int,
    swapped: bool,
    endpoint: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("schema") != e49j.SCHEMA
            or record.get("packet_row_sha256")
            != _canonical_sha256(pair)
            or record.get("role") != role
            or record.get("seed") != seed
            or record.get("display_swapped") != swapped
        ):
            raise RuntimeError(f"E49H hardened-veto cache changed: {path}")
        return record
    record = e49j._request_one(
        pair=pair,
        endpoint=endpoint,
        model=model,
        role=role,
        seed=seed,
        swapped=swapped,
        timeout=timeout,
    )
    _write_json(path, record)
    return record


def _load_and_validate(
    *,
    contracts_path: pathlib.Path,
    source: pathlib.Path,
    expected_train: int = 12,
    expected_eval: int = 10,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    if (
        not isinstance(contracts, dict)
        or len(contracts) != expected_train + expected_eval
        or sum(key.startswith("train:") for key in contracts)
        != expected_train
        or sum(key.startswith("eval:") for key in contracts)
        != expected_eval
    ):
        raise RuntimeError("E49H frozen cohort identity changed")
    source_rows = _source_rows(source)
    menus = {}
    for row_id, payload in contracts.items():
        row = source_rows.get(row_id)
        if row is None:
            raise RuntimeError(f"E49H row missing from source: {row_id}")
        menu = e49e._menu_from_payload(payload)
        if (
            len(menu.strategies) != 2
            or [strategy.strategy_id for strategy in menu.strategies]
            != ["S1", "S2"]
            or any(
                e49e._extract_closed_candidate(menu, strategy_id) is None
                for strategy_id in ("S1", "S2")
            )
            or not v2b.v2._proposal_is_nonleaking_v2(
                menu,
                problem=row["problem"],
                reference_answer=row["reference_answer"],
            )
        ):
            raise RuntimeError(f"E49H contract failed local gates: {row_id}")
        menus[row_id] = menu
    return contracts, source_rows, menus


def _verify_calibration(path: pathlib.Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = report.get("checks") or {}
    if (
        report.get("schema")
        != "e49j_restricted_mathir_calibration_report_v1"
        or report.get("pass") is not True
        or checks.get("all_requests_complete") is not True
        or checks.get("false_new_exactly_zero") is not True
        or checks.get("all_hidden_equivalent_controls_rejected") is not True
        or checks.get("at_least_three_of_four_true_distinct_recovered")
        is not True
    ):
        raise RuntimeError("E49J calibration did not authorize E49H")
    return report


def run(args: argparse.Namespace) -> None:
    final_records = args.output / "candidate_records.jsonl"
    summary_path = args.output / "run_summary.json"
    if final_records.exists() or summary_path.exists():
        raise RuntimeError("E49H final evidence already exists")
    _verify_calibration(args.e49j_calibration)
    contracts, source_rows, menus = _load_and_validate(
        contracts_path=args.contracts,
        source=args.source,
        expected_train=args.expected_train,
        expected_eval=args.expected_eval,
    )
    endpoint, model = e49j.base._endpoint(args.endpoint)

    sound_work = []
    for row_id in sorted(contracts):
        for strategy_id in ("S1", "S2"):
            for seed, role in zip(
                e49e.SOUNDNESS_SEEDS,
                e49e.SOUNDNESS_ROLES,
                strict=True,
            ):
                sound_work.append(
                    (row_id, strategy_id, role, seed)
                )

    sound_by_row: dict[
        str, dict[str, list[dict[str, Any]]]
    ] = {
        row_id: {"S1": [], "S2": []} for row_id in contracts
    }

    def execute_sound(item):
        row_id, strategy_id, role, seed = item
        menu = menus[row_id]
        return (
            row_id,
            strategy_id,
            _cached_sound(
                path=_sound_cache(
                    args.output,
                    row_id,
                    strategy_id,
                    role,
                    seed,
                    menu.sha256,
                ),
                row=source_rows[row_id],
                menu=menu,
                strategy_id=strategy_id,
                role=role,
                seed=seed,
                endpoint=endpoint,
                model=model,
                timeout=args.timeout,
            ),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute_sound, item) for item in sound_work]
        for future in as_completed(futures):
            row_id, strategy_id, audit = future.result()
            sound_by_row[row_id][strategy_id].append(audit)

    sound_pass = {}
    pair_payloads = {}
    for row_id in sorted(contracts):
        for strategy_id in ("S1", "S2"):
            sound_by_row[row_id][strategy_id].sort(
                key=lambda audit: audit["seed"]
            )
        sound_pass[row_id] = all(
            len(sound_by_row[row_id][strategy_id]) == 2
            and all(
                audit.get("pass") is True
                for audit in sound_by_row[row_id][strategy_id]
            )
            for strategy_id in ("S1", "S2")
        )
        if sound_pass[row_id]:
            pair_payloads[row_id] = _pair_payload(
                source_rows[row_id],
                menus[row_id],
                sound_by_row[row_id],
            )

    veto_by_row: dict[str, list[dict[str, Any]]] = {
        row_id: [] for row_id in contracts
    }
    veto_work = [
        (row_id, role, seed, swapped)
        for row_id in sorted(pair_payloads)
        for role, seed, swapped in e49j.ASSESSMENTS
    ]

    def execute_veto(item):
        row_id, role, seed, swapped = item
        menu = menus[row_id]
        return (
            row_id,
            _cached_veto(
                path=_veto_cache(
                    args.output,
                    row_id,
                    role,
                    seed,
                    menu.sha256,
                ),
                pair=pair_payloads[row_id],
                role=role,
                seed=seed,
                swapped=swapped,
                endpoint=endpoint,
                model=model,
                timeout=args.timeout,
            ),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute_veto, item) for item in veto_work]
        for future in as_completed(futures):
            row_id, audit = future.result()
            veto_by_row[row_id].append(audit)

    records = []
    for row_id in sorted(contracts):
        veto_by_row[row_id].sort(key=lambda audit: audit["seed"])
        complete_veto = bool(
            len(veto_by_row[row_id]) == 4
            and all(
                audit.get("completed_invalid") is False
                for audit in veto_by_row[row_id]
            )
        )
        distinct_vote_count = sum(
            audit.get("distinct_vote") is True
            for audit in veto_by_row[row_id]
        )
        hardened_distinct = bool(
            complete_veto and distinct_vote_count == 4
        )
        row = source_rows[row_id]
        records.append(
            {
                "schema": (
                    f"{args.experiment}_curated_pair_certification_v1"
                ),
                "row_id": row_id,
                "split": row["split"],
                "problem_sha256": _sha256_bytes(
                    row["problem"].encode("utf-8")
                ),
                "reference_answer_sha256": _sha256_bytes(
                    row["reference_answer"].encode("utf-8")
                ),
                "contract_sha256": _canonical_sha256(
                    contracts[row_id]
                ),
                "menu": json.loads(menus[row_id].canonical_json),
                "menu_sha256": menus[row_id].sha256,
                "sound_audits": sound_by_row[row_id],
                "sound_pass": sound_pass[row_id],
                "pair_payload": pair_payloads.get(row_id),
                "hardened_veto_audits": veto_by_row[row_id],
                "hardened_veto_complete": complete_veto,
                "hardened_distinct_vote_count": distinct_vote_count,
                "hardened_distinct": hardened_distinct,
                "pass": bool(sound_pass[row_id] and hardened_distinct),
            }
        )
    _write_jsonl(final_records, records)
    passing = [record for record in records if record["pass"]]
    summary = {
        "schema": f"{args.experiment}_curated_route_run_summary_v1",
        "contracts_sha256": _sha256(args.contracts),
        "endpoint_sha256": _sha256(args.endpoint),
        "e49j_calibration_sha256": _sha256(args.e49j_calibration),
        "candidate_count": len(records),
        "sound_request_count": len(sound_work),
        "veto_request_count": len(veto_work),
        "sound_pair_count": sum(record["sound_pass"] for record in records),
        "passing_pair_count": len(passing),
        "passing_train_count": sum(
            record["split"] == "train" for record in passing
        ),
        "passing_eval_count": sum(
            record["split"] == "eval" for record in passing
        ),
        "all_requests_complete": all(
            all(
                len(record["sound_audits"][strategy_id]) == 2
                for strategy_id in ("S1", "S2")
            )
            and (
                not record["sound_pass"]
                or record["hardened_veto_complete"]
            )
            for record in records
        ),
        "candidate_records_sha256": _sha256(final_records),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def preflight(args: argparse.Namespace) -> None:
    contracts, _, menus = _load_and_validate(
        contracts_path=args.contracts,
        source=args.source,
        expected_train=args.expected_train,
        expected_eval=args.expected_eval,
    )
    payload = {
        "schema": f"{args.experiment}_curated_route_preflight_v1",
        "candidate_count": len(contracts),
        "train_candidate_count": sum(
            row_id.startswith("train:") for row_id in contracts
        ),
        "eval_candidate_count": sum(
            row_id.startswith("eval:") for row_id in contracts
        ),
        "all_contracts_closed_and_nonleaking": len(menus) == len(contracts),
        "contracts_sha256": _sha256(args.contracts),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--contracts", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", type=pathlib.Path)
    parser.add_argument("--e49j-calibration", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--experiment",
        choices=("e49h", "e49k", "e49l", "e49n", "e49p"),
        default="e49h",
    )
    parser.add_argument("--expected-train", type=int, default=12)
    parser.add_argument("--expected-eval", type=int, default=10)
    args = parser.parse_args()
    if args.action == "preflight":
        preflight(args)
        return
    if (
        args.endpoint is None
        or args.e49j_calibration is None
        or args.output is None
    ):
        raise RuntimeError(
            "run requires --endpoint, --e49j-calibration, and --output"
        )
    run(args)


if __name__ == "__main__":
    main()
