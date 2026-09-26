#!/usr/bin/env python3
"""Recover fully visible routes from one-of-two execution false negatives."""

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


HERE = pathlib.Path(__file__).resolve().parent
BASE_PATH = HERE / "certify_e49h_curated_routes.py"


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_module("e49q_e49h_certifier", BASE_PATH)
e49j = base.e49j
SCHEMA = "e49q_recovered_pair_certification_v1"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: pathlib.Path, payload: Any) -> None:
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


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
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


def _validated_candidates(
    evidence_paths: list[pathlib.Path],
    calibration: pathlib.Path,
) -> list[dict[str, Any]]:
    expected_calibration_sha256 = _sha256(calibration)
    candidates = []
    seen_identities = set()
    for evidence in evidence_paths:
        identity_path = evidence / "frozen_identity.json"
        summary_path = evidence / "run_summary.json"
        records_path = evidence / "candidate_records.jsonl"
        contracts_path = evidence / "curated_contracts.json"
        for required in (
            identity_path,
            summary_path,
            records_path,
            contracts_path,
        ):
            if not required.is_file():
                raise RuntimeError(
                    f"incomplete E49Q source evidence: {required}"
                )
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            identity.get("requests_at_freeze") != 0
            or identity.get("contracts_sha256") != _sha256(contracts_path)
            or summary.get("all_requests_complete") is not True
            or summary.get("contracts_sha256") != _sha256(contracts_path)
            or summary.get("candidate_records_sha256")
            != _sha256(records_path)
            or summary.get("e49j_calibration_sha256")
            != expected_calibration_sha256
        ):
            raise RuntimeError(
                f"E49Q source evidence identity failed: {evidence}"
            )
        identity_sha256 = _sha256(identity_path)
        if identity_sha256 in seen_identities:
            raise RuntimeError("duplicate E49Q source evidence identity")
        seen_identities.add(identity_sha256)
        for record in _read_jsonl(records_path):
            if (
                record.get("menu_sha256")
                != _canonical_sha256(record["menu"])
                or len(record.get("sound_audits") or {}) != 2
            ):
                raise RuntimeError("E49Q candidate record changed")
            candidate_id = hashlib.sha256(
                (
                    f"{identity_sha256}\0{record['row_id']}\0"
                    f"{record['menu_sha256']}"
                ).encode("utf-8")
            ).hexdigest()[:24]
            candidates.append(
                {
                    "candidate_id": f"E49Q_{candidate_id}",
                    "source_evidence": str(evidence),
                    "source_identity_sha256": identity_sha256,
                    "source_record_sha256": _canonical_sha256(record),
                    "record": record,
                }
            )
    return candidates


def _eligible(record: dict[str, Any]) -> bool:
    if record.get("pass") is True:
        return False
    audits_by_strategy = record["sound_audits"]
    return all(
        len(audits_by_strategy[strategy_id]) == 2
        and all(
            audit.get("completed_invalid") is False
            for audit in audits_by_strategy[strategy_id]
        )
        and any(
            audit.get("pass") is True
            for audit in audits_by_strategy[strategy_id]
        )
        for strategy_id in ("S1", "S2")
    )


def _cache_path(
    output: pathlib.Path,
    candidate_id: str,
    role: str,
    seed: int,
) -> pathlib.Path:
    return output / "request_cache" / candidate_id / f"{role}-{seed}.json"


def _cached_request(
    *,
    path: pathlib.Path,
    pair: dict[str, Any],
    endpoint: str,
    model: str,
    role: str,
    seed: int,
    swapped: bool,
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
            raise RuntimeError(f"E49Q cache changed: {path}")
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


def run(args: argparse.Namespace) -> None:
    final_path = args.output / "recovered_records.jsonl"
    summary_path = args.output / "run_summary.json"
    if final_path.exists() or summary_path.exists():
        raise RuntimeError("fresh E49Q output is required")
    base._verify_calibration(args.e49j_calibration)
    candidates = _validated_candidates(
        args.evidence,
        args.e49j_calibration,
    )
    source_rows = base._source_rows(args.source)
    endpoint, model = e49j.base._endpoint(args.endpoint)
    eligible = []
    for candidate in candidates:
        record = candidate["record"]
        if not _eligible(record):
            continue
        menu = base.e49e._menu_from_payload(record["menu"])
        pair = base._pair_payload(
            source_rows[record["row_id"]],
            menu,
            record["sound_audits"],
        )
        pair["schema"] = "e49q_recovered_pair_for_mathir_v1"
        pair["pair_id"] = candidate["candidate_id"]
        eligible.append({**candidate, "menu": menu, "pair": pair})

    work = [
        (candidate, role, seed, swapped)
        for candidate in eligible
        for role, seed, swapped in e49j.ASSESSMENTS
    ]

    def execute(item):
        candidate, role, seed, swapped = item
        audit = _cached_request(
            path=_cache_path(
                args.output,
                candidate["candidate_id"],
                role,
                seed,
            ),
            pair=candidate["pair"],
            endpoint=endpoint,
            model=model,
            role=role,
            seed=seed,
            swapped=swapped,
            timeout=args.timeout,
        )
        return candidate["candidate_id"], audit

    audits_by_candidate = {
        candidate["candidate_id"]: [] for candidate in eligible
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, item) for item in work]
        for future in as_completed(futures):
            candidate_id, audit = future.result()
            audits_by_candidate[candidate_id].append(audit)

    rows = []
    for candidate in eligible:
        audits = sorted(
            audits_by_candidate[candidate["candidate_id"]],
            key=lambda row: row["seed"],
        )
        complete = bool(
            len(audits) == 4
            and all(
                audit.get("completed_invalid") is False for audit in audits
            )
        )
        votes = sum(
            audit.get("distinct_vote") is True for audit in audits
        )
        record = candidate["record"]
        rows.append(
            {
                "schema": SCHEMA,
                "candidate_id": candidate["candidate_id"],
                "row_id": record["row_id"],
                "split": record["split"],
                "source_evidence": candidate["source_evidence"],
                "source_identity_sha256": candidate[
                    "source_identity_sha256"
                ],
                "source_record_sha256": candidate[
                    "source_record_sha256"
                ],
                "menu": record["menu"],
                "menu_sha256": record["menu_sha256"],
                "sound_audits": record["sound_audits"],
                "at_least_one_exact_execution_per_route": True,
                "pair_payload": candidate["pair"],
                "mathir_audits": audits,
                "mathir_complete": complete,
                "mathir_distinct_vote_count": votes,
                "pass": bool(complete and votes == 4),
            }
        )
    _write_jsonl(final_path, rows)
    passing = [row for row in rows if row["pass"]]
    summary = {
        "schema": "e49q_recovered_route_run_summary_v1",
        "source_evidence_identity_sha256s": sorted(
            {
                candidate["source_identity_sha256"]
                for candidate in candidates
            }
        ),
        "e49j_calibration_sha256": _sha256(args.e49j_calibration),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "request_count": len(work),
        "all_requests_complete": all(
            row["mathir_complete"] for row in rows
        ),
        "passing_candidate_count": len(passing),
        "passing_train_row_count": len(
            {row["row_id"] for row in passing if row["split"] == "train"}
        ),
        "passing_eval_row_count": len(
            {row["row_id"] for row in passing if row["split"] == "eval"}
        ),
        "recovered_records_sha256": _sha256(final_path),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def preflight(args: argparse.Namespace) -> None:
    base._verify_calibration(args.e49j_calibration)
    candidates = _validated_candidates(
        args.evidence,
        args.e49j_calibration,
    )
    eligible = [
        candidate for candidate in candidates
        if _eligible(candidate["record"])
    ]
    payload = {
        "schema": "e49q_recovered_route_preflight_v1",
        "source_candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "eligible_train_row_count": len(
            {
                candidate["record"]["row_id"]
                for candidate in eligible
                if candidate["record"]["split"] == "train"
            }
        ),
        "eligible_eval_row_count": len(
            {
                candidate["record"]["row_id"]
                for candidate in eligible
                if candidate["record"]["split"] == "eval"
            }
        ),
        "request_count": 4 * len(eligible),
        "all_four_execution_traces_present": all(
            all(
                len(candidate["record"]["sound_audits"][strategy_id]) == 2
                for strategy_id in ("S1", "S2")
            )
            for candidate in eligible
        ),
        "at_least_one_exact_execution_per_route": all(
            _eligible(candidate["record"]) for candidate in eligible
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument(
        "--evidence",
        type=pathlib.Path,
        action="append",
        required=True,
    )
    parser.add_argument("--endpoint", type=pathlib.Path)
    parser.add_argument("--e49j-calibration", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.action == "preflight":
        preflight(args)
        return
    if args.endpoint is None or args.output is None:
        raise RuntimeError("run requires --endpoint and --output")
    run(args)


if __name__ == "__main__":
    main()
