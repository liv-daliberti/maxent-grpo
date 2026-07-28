#!/usr/bin/env python3
"""Confirm singleton observed routes, then test literal 0.5B executability."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49ac_confirmed_singleton_observed_route_discovery_20260726.md"
)
BASE_AB_SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49ab_all_observed_persistent_pairwise_route_discovery.py"
)
AB_ROOT = (
    ROOT
    / "var/artifacts/"
    "e49ab_all_observed_persistent_pairwise_route_discovery_v1"
)
AB_RESULT = AB_ROOT / "result.json"
AB_CLUSTERS = AB_ROOT / "cluster_records.jsonl"
E49T_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e49t_natural_menu_math_toy_05b_3ep_v1_identity.json"
)
ENDPOINT_RECORD = (
    ROOT / "var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
)
PAIRWISE_SOURCE = (
    ROOT
    / "var/artifacts/source_snapshots/"
    "e49b_math_strategy_ae06b1023ade504c7b8a07e901874e5b563b9e8fad618e5fefe519337458ebfc/"
    "src/oat_drgrpo/math_strategy_canonicalizer.py"
)
MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
SCHEMA = "e49ac_confirmed_singleton_observed_route_discovery_v1"
RELATION_SEEDS = (470731, 470732)
FORCED_SAMPLE_COUNT = 64
MIN_FORCED_SUCCESSES = 2
UNFORCED_SAMPLE_COUNT = 64
MIN_NATURAL_PER_ROUTE = 2
MIN_NATURAL_TOTAL = 8
SEED = 490831


def _load_module(path: pathlib.Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AB = _load_module(BASE_AB_SCRIPT, "e49ac_base_e49ab")
AA = AB.BASE


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _write_jsonl(
    path: pathlib.Path, rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _neutral_problem(problem: str, menu: Any) -> str:
    from oat_drgrpo.math_strategy_menu import (
        MENU_END,
        MENU_START,
        strategy_menu_natural_response_instructions,
    )

    return (
        problem.rstrip()
        + f"\n\n{MENU_START}\n"
        + menu.canonical_json
        + f"\n{MENU_END}"
        + "\n\nStrategy IDs and listing order are labels only; neither "
        "strategy is preferred. Choose independently on each attempt."
        + strategy_menu_natural_response_instructions(menu)
    )


def _natural_support(route_counts: dict[str, int]) -> bool:
    return (
        len(route_counts) == 2
        and min(route_counts.values()) >= MIN_NATURAL_PER_ROUTE
        and sum(route_counts.values()) >= MIN_NATURAL_TOTAL
    )


def _problem_rank(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        -min(row["unforced_route_success_counts"].values()),
        -int(row["unforced_accepted_count"]),
        -int(row["minimum_cluster_size"]),
        int(row["source_index"]),
    )


def _verify_activation() -> dict[str, Any]:
    result = json.loads(AB_RESULT.read_text(encoding="utf-8"))
    if (
        result.get("schema")
        != "e49ab_all_observed_persistent_pairwise_route_discovery_v1"
        or result.get("pass") is True
        or int(result.get("bidirectionally_executable_count", 0)) >= 10
    ):
        raise RuntimeError(
            "E49AC activates only after final E49AB yields fewer than ten "
            "bidirectionally executable menus"
        )
    if result.get("cluster_records_sha256") != _sha256(AB_CLUSTERS):
        raise RuntimeError("E49AC E49AB cluster-record identity mismatch")
    if int(result.get("all_observed_response_count", 0)) != 1550:
        raise RuntimeError("E49AC requires E49AB's complete 1550-response replay")
    return result


def _candidate_maps(
    full_rows: Any,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    candidates = AB._all_candidates(full_rows)
    if len(candidates) != 74:
        raise RuntimeError("E49AC expected exactly 74 frozen candidates")
    by_index = {int(row["source_index"]): row for row in candidates}
    return candidates, by_index


def _singleton_candidate(
    candidate: dict[str, Any],
    cluster_record: dict[str, Any],
) -> dict[str, Any]:
    exemplar_by_id = {
        str(row["sample_id"]): row for row in candidate["exemplars"]
    }
    if len(exemplar_by_id) != len(candidate["exemplars"]):
        raise RuntimeError("E49AC observed a truncated response-SHA collision")
    members: dict[str, list[str]] = defaultdict(list)
    for outcome in cluster_record.get("outcomes", []):
        sample_id = str(outcome["sample_id"])
        key = outcome.get("strategy_key")
        if sample_id not in exemplar_by_id:
            raise RuntimeError("E49AC cluster outcome is outside frozen cohort")
        if key is not None:
            members[str(key)].append(sample_id)
    ordered = sorted(
        members.items(), key=lambda item: (-len(item[1]), item[0])
    )
    base = {
        key: candidate[key]
        for key in ("source_index", "row_id", "subject", "level")
    }
    if len(ordered) < 2:
        return {
            **base,
            "candidate_pair": False,
            "emitted_key_count": len(ordered),
        }
    retained = []
    for index, (strategy_key, member_ids) in enumerate(
        ordered[:2], start=1
    ):
        ordered_ids = sorted(
            member_ids,
            key=lambda sample_id: exemplar_by_id[sample_id][
                "response_sha256"
            ],
        )
        retained.append(
            {
                "cluster_id": f"C{index}",
                "strategy": (
                    "independently confirmed observed component "
                    f"{strategy_key}"
                ),
                "strategy_key": strategy_key,
                "member_ids": ordered_ids,
            }
        )
    return {
        **base,
        "candidate_pair": True,
        "emitted_key_count": len(ordered),
        "clusters": retained,
        "minimum_cluster_size": min(
            len(row["member_ids"]) for row in retained
        ),
        "combined_cluster_size": sum(
            len(row["member_ids"]) for row in retained
        ),
    }


def _confirm_candidate(
    *,
    pairwise_module: Any,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    singleton: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    base = {
        key: singleton[key]
        for key in (
            "source_index",
            "row_id",
            "subject",
            "level",
            "emitted_key_count",
            "minimum_cluster_size",
            "combined_cluster_size",
            "clusters",
        )
    }
    exemplar_by_id = {
        str(row["sample_id"]): row for row in candidate["exemplars"]
    }
    try:
        representatives = [
            (
                f"ROUTE_{index}",
                str(
                    exemplar_by_id[cluster["member_ids"][0]][
                        "text"
                    ]
                ),
            )
            for index, cluster in enumerate(
                singleton["clusters"], start=1
            )
        ]
        pair = [("PAIR_0000", "ROUTE_1", "ROUTE_2")]
        relations = []
        for seed in RELATION_SEEDS:
            judge = pairwise_module.MathStrategyCanonicalizer(
                endpoint=endpoint,
                model=model,
                timeout_seconds=timeout,
                max_workers=1,
                permutation_seeds=(470721, 470722),
                max_item_chars=4000,
                missing_ids_are_ambiguous=True,
            )
            captured: list[dict[str, Any]] = []
            original_post = judge._post

            def capture_post(payload: dict[str, Any]) -> dict[str, Any]:
                response = original_post(payload)
                content = str(
                    (
                        ((response.get("choices") or [{}])[0]).get(
                            "message"
                        )
                        or {}
                    ).get("content")
                    or ""
                )
                parsed_content = content.strip()
                if parsed_content.startswith("```"):
                    parsed_content = re.sub(
                        r"^```(?:json)?\s*", "", parsed_content
                    )
                    parsed_content = re.sub(
                        r"\s*```$", "", parsed_content
                    )
                captured.append(
                    {
                        "response_id": response.get("id"),
                        "finish_reason": (
                            (response.get("choices") or [{}])[0]
                        ).get("finish_reason"),
                        "content_sha256": hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                        "assessment": json.loads(parsed_content),
                    }
                )
                return response

            judge._post = capture_post
            relation = judge._judge_relations(
                problem=candidate["problem"],
                base_items=representatives,
                pairs=pair,
                seed=seed,
            )["PAIR_0000"]
            if len(captured) != 1:
                raise RuntimeError(
                    "E49AC direct relation audit did not issue exactly one call"
                )
            relations.append(
                {
                    "seed": seed,
                    "relation": relation,
                    **captured[0],
                }
            )
        return {
            **base,
            "relations": relations,
            "singleton_confirmation_pass": all(
                row["relation"] == "different" for row in relations
            ),
        }
    except Exception as exc:
        return {
            **base,
            "relations": [],
            "singleton_confirmation_pass": False,
            "error": str(exc),
        }


def _failure_result(
    *,
    output: pathlib.Path,
    result_path: pathlib.Path,
    failure: str,
    candidates: list[dict[str, Any]],
    singleton_records: list[dict[str, Any]],
    confirmed_records: list[dict[str, Any]],
    menu_records: list[dict[str, Any]],
    ab_result: dict[str, Any],
) -> None:
    singleton_path = output / "singleton_confirmation_records.jsonl"
    menu_path = output / "menu_records.jsonl"
    payload = {
        "schema": SCHEMA,
        "pass": False,
        "failure": failure,
        "candidate_count": len(candidates),
        "singleton_pair_count": sum(
            row.get("candidate_pair") is True for row in singleton_records
        ),
        "singleton_confirmed_count": len(confirmed_records),
        "double_audited_menu_count": sum(
            row.get("double_audit_pass") is True for row in menu_records
        ),
        "all_observed_response_count": 1550,
        "identity": {
            "protocol_sha256": _sha256(PROTOCOL),
            "script_sha256": _sha256(SCRIPT),
            "base_e49ab_script_sha256": _sha256(BASE_AB_SCRIPT),
            "e49ab_result_sha256": _sha256(AB_RESULT),
            "e49ab_cluster_records_sha256": _sha256(AB_CLUSTERS),
            "e49ab_result_schema": ab_result["schema"],
            "pairwise_source_sha256": _sha256(PAIRWISE_SOURCE),
            "e49t_identity_sha256": _sha256(E49T_IDENTITY),
        },
        "singleton_confirmation_records_sha256": _sha256(singleton_path),
        "menu_records_sha256": _sha256(menu_path),
    }
    _write_json(result_path, payload)
    print(result_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh E49AC result required: {result_path}")
    ab_result = _verify_activation()
    AA._verify_inputs()

    from datasets import load_from_disk

    endpoint, judge_model = AA._endpoint(ENDPOINT_RECORD)
    full_rows = load_from_disk(str(AA.FULL / "train"))["train"]
    candidates, candidate_by_index = _candidate_maps(full_rows)
    cluster_rows = _load_jsonl(AB_CLUSTERS)
    if (
        len(cluster_rows) != 74
        or {int(row["source_index"]) for row in cluster_rows}
        != set(candidate_by_index)
    ):
        raise RuntimeError("E49AC requires all 74 E49AB cluster records")

    singleton_records = []
    for cluster_record in cluster_rows:
        source_index = int(cluster_record["source_index"])
        singleton_records.append(
            _singleton_candidate(
                candidate_by_index[source_index], cluster_record
            )
        )
    pair_candidates = [
        row for row in singleton_records if row["candidate_pair"]
    ]
    pairwise_module = AA._load_pairwise_module()
    confirmed_records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _confirm_candidate,
                pairwise_module=pairwise_module,
                endpoint=endpoint,
                model=judge_model,
                candidate=candidate_by_index[int(row["source_index"])],
                singleton=row,
                timeout=args.timeout,
            )
            for row in pair_candidates
        ]
        for future in as_completed(futures):
            record = future.result()
            confirmed_records.append(record)
            print(
                json.dumps(
                    {
                        "phase": "confirm",
                        "source_index": record["source_index"],
                        "pass": record["singleton_confirmation_pass"],
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    confirmed_by_index = {
        int(row["source_index"]): row for row in confirmed_records
    }
    for row in singleton_records:
        if int(row["source_index"]) in confirmed_by_index:
            row.update(confirmed_by_index[int(row["source_index"])])
    singleton_records.sort(key=lambda row: int(row["source_index"]))
    singleton_path = output / "singleton_confirmation_records.jsonl"
    _write_jsonl(singleton_path, singleton_records)
    confirmed_records = [
        row
        for row in singleton_records
        if row.get("singleton_confirmation_pass") is True
    ]

    menu_records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                AB._bounded_menu_candidate,
                endpoint=endpoint,
                model=judge_model,
                candidate=candidate_by_index[int(record["source_index"])],
                cluster_record=record,
                timeout=args.timeout,
            )
            for record in confirmed_records
        ]
        for future in as_completed(futures):
            record = future.result()
            menu_records.append(record)
            print(
                json.dumps(
                    {
                        "phase": "menu",
                        "source_index": record["source_index"],
                        "double_audit_pass": record["double_audit_pass"],
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    menu_records.sort(key=lambda row: int(row["source_index"]))
    menu_path = output / "menu_records.jsonl"
    _write_jsonl(menu_path, menu_records)
    accepted = [
        row for row in menu_records if row["double_audit_pass"]
    ]
    accepted.sort(
        key=lambda row: (
            -int(row["minimum_cluster_size"]),
            -int(row["combined_cluster_size"]),
            int(row["source_index"]),
        )
    )
    if not accepted:
        _failure_result(
            output=output,
            result_path=result_path,
            failure="no_double_audited_confirmed_singleton_menu",
            candidates=candidates,
            singleton_records=singleton_records,
            confirmed_records=confirmed_records,
            menu_records=menu_records,
            ab_result=ab_result,
        )
        return

    # Import the corrected successor answer grader before placing the frozen
    # E49T canonicalizer at the front of oat_drgrpo's package path.
    from oat_drgrpo.math_grader import boxed_reward_fn
    from oat_drgrpo.templates import apply_qwen_math_template
    import oat_drgrpo
    import vllm

    identity = json.loads(E49T_IDENTITY.read_text(encoding="utf-8"))
    frozen_source = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e49t_natural_menu_{identity['source_hash']}"
        / "src"
    )
    if not frozen_source.is_dir():
        raise RuntimeError("E49AC frozen E49T source snapshot is missing")
    oat_drgrpo.__path__.insert(0, str(frozen_source / "oat_drgrpo"))
    from oat_drgrpo.math_strategy_canonicalizer import (
        MathStrategyCanonicalizer,
    )

    cases = []
    unforced_cases = []
    for record in accepted:
        source = dict(full_rows[int(record["source_index"])])
        menu = AA._parse_menu(record["menu"])
        for strategy in menu.strategies:
            cases.append(
                {
                    "source_index": int(record["source_index"]),
                    "row_id": str(record["row_id"]),
                    "problem": AA._forced_problem(
                        str(source["problem"]), menu, strategy
                    ),
                    "answer": str(source["answer"]),
                    "menu": menu,
                    "strategy": strategy,
                }
            )
        unforced_cases.append(
            {
                "source_index": int(record["source_index"]),
                "row_id": str(record["row_id"]),
                "problem": _neutral_problem(str(source["problem"]), menu),
                "answer": str(source["answer"]),
                "menu": menu,
            }
        )
    llm = vllm.LLM(
        model=str(MODEL),
        dtype="bfloat16",
        max_model_len=3072,
        gpu_memory_utilization=0.80,
        swap_space=8.0,
        enable_prefix_caching=True,
        seed=SEED,
    )
    forced_outputs = llm.generate(
        [apply_qwen_math_template(case["problem"]) for case in cases],
        vllm.SamplingParams(
            n=FORCED_SAMPLE_COUNT,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            seed=SEED,
        ),
    )
    unforced_outputs = llm.generate(
        [
            apply_qwen_math_template(case["problem"])
            for case in unforced_cases
        ],
        vllm.SamplingParams(
            n=UNFORCED_SAMPLE_COUNT,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            seed=SEED + 1,
        ),
    )
    flat_tokens = []
    flat_problems = []
    flat_responses = []
    flat_positive = []
    private_forced = []
    for case_index, (case, request_output) in enumerate(
        zip(cases, forced_outputs, strict=True)
    ):
        if len(request_output.outputs) != FORCED_SAMPLE_COUNT:
            raise RuntimeError("E49AC forced sample count mismatch")
        for sample_index, sample in enumerate(request_output.outputs):
            text = str(sample.text)
            _, reward = boxed_reward_fn(text, case["answer"], fast=False)
            positive = float(reward) > 0
            flat_tokens.append(list(request_output.prompt_token_ids))
            flat_problems.append(case["problem"])
            flat_responses.append(text)
            flat_positive.append(positive)
            private_forced.append(
                {
                    "case_index": case_index,
                    "source_index": case["source_index"],
                    "row_id": case["row_id"],
                    "strategy_id": case["strategy"].strategy_id,
                    "action_combo": case["strategy"].action_combo,
                    "sample_index": sample_index,
                    "answer_positive": positive,
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "response": text,
                }
            )
    flat_unforced_tokens = []
    flat_unforced_problems = []
    flat_unforced_responses = []
    flat_unforced_positive = []
    private_unforced = []
    for case_index, (case, request_output) in enumerate(
        zip(unforced_cases, unforced_outputs, strict=True)
    ):
        if len(request_output.outputs) != UNFORCED_SAMPLE_COUNT:
            raise RuntimeError("E49AC unforced sample count mismatch")
        for sample_index, sample in enumerate(request_output.outputs):
            text = str(sample.text)
            _, reward = boxed_reward_fn(text, case["answer"], fast=False)
            positive = float(reward) > 0
            flat_unforced_tokens.append(list(request_output.prompt_token_ids))
            flat_unforced_problems.append(case["problem"])
            flat_unforced_responses.append(text)
            flat_unforced_positive.append(positive)
            private_unforced.append(
                {
                    "case_index": case_index,
                    "source_index": case["source_index"],
                    "row_id": case["row_id"],
                    "sample_index": sample_index,
                    "answer_positive": positive,
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "response": text,
                }
            )
    forced_path = output / "private/forced_responses.jsonl"
    unforced_path = output / "private/unforced_responses.jsonl"
    _write_jsonl(forced_path, private_forced)
    _write_jsonl(unforced_path, private_unforced)

    judge = MathStrategyCanonicalizer(
        endpoint=endpoint,
        model=judge_model,
        timeout_seconds=args.timeout,
        max_workers=args.workers,
        permutation_seeds=(470721, 470722),
        max_item_chars=4000,
        missing_ids_are_ambiguous=True,
        allow_unstructured_menu_inference=True,
    )
    keys, diagnostics = judge.canonicalize(
        prompt_token_ids=flat_tokens,
        prompt_texts=flat_problems,
        response_texts=flat_responses,
        task_reward_positive=flat_positive,
        active_mask=[True] * len(flat_responses),
        num_samples=FORCED_SAMPLE_COUNT,
    )
    unforced_judge = MathStrategyCanonicalizer(
        endpoint=endpoint,
        model=judge_model,
        timeout_seconds=args.timeout,
        max_workers=args.workers,
        permutation_seeds=(470721, 470722),
        max_item_chars=4000,
        missing_ids_are_ambiguous=True,
        allow_unstructured_menu_inference=True,
    )
    unforced_keys, unforced_diagnostics = unforced_judge.canonicalize(
        prompt_token_ids=flat_unforced_tokens,
        prompt_texts=flat_unforced_problems,
        response_texts=flat_unforced_responses,
        task_reward_positive=flat_unforced_positive,
        active_mask=[True] * len(flat_unforced_responses),
        num_samples=UNFORCED_SAMPLE_COUNT,
    )
    case_results = []
    for case_index, case in enumerate(cases):
        start = case_index * FORCED_SAMPLE_COUNT
        stop = start + FORCED_SAMPLE_COUNT
        expected_key = judge._menu_strategy_key(
            case["menu"], case["strategy"].strategy_id
        )
        expected = sum(
            keys[index] == expected_key for index in range(start, stop)
        )
        wrong = sum(
            keys[index] is not None and keys[index] != expected_key
            for index in range(start, stop)
        )
        case_results.append(
            {
                "source_index": case["source_index"],
                "row_id": case["row_id"],
                "menu_sha256": case["menu"].sha256,
                "strategy_id": case["strategy"].strategy_id,
                "action_combo": case["strategy"].action_combo,
                "sample_count": FORCED_SAMPLE_COUNT,
                "answer_success_count": sum(flat_positive[start:stop]),
                "forced_route_success_count": expected,
                "wrong_route_success_count": wrong,
                "minimally_executable": expected >= MIN_FORCED_SUCCESSES,
            }
        )
    problem_results = []
    for order, (record, unforced_case) in enumerate(
        zip(accepted, unforced_cases, strict=True)
    ):
        route_cases = [
            row
            for row in case_results
            if row["source_index"] == int(record["source_index"])
        ]
        if len(route_cases) != 2:
            raise RuntimeError("E49AC accepted menu lost one route")
        start = order * UNFORCED_SAMPLE_COUNT
        stop = start + UNFORCED_SAMPLE_COUNT
        unforced_route_counts = {
            strategy.strategy_id: sum(
                key
                == unforced_judge._menu_strategy_key(
                    unforced_case["menu"], strategy.strategy_id
                )
                for key in unforced_keys[start:stop]
            )
            for strategy in unforced_case["menu"].strategies
        }
        natural_support = _natural_support(unforced_route_counts)
        problem_results.append(
            {
                "candidate_order": order,
                "source_index": int(record["source_index"]),
                "row_id": str(record["row_id"]),
                "menu": record["menu"],
                "menu_sha256": record["menu_sha256"],
                "route_sources": record["route_sources"],
                "minimum_cluster_size": record["minimum_cluster_size"],
                "combined_cluster_size": record["combined_cluster_size"],
                "bidirectionally_executable": all(
                    row["minimally_executable"] for row in route_cases
                )
                and natural_support,
                "route_success_counts": {
                    row["strategy_id"]: row["forced_route_success_count"]
                    for row in route_cases
                },
                "answer_success_counts": {
                    row["strategy_id"]: row["answer_success_count"]
                    for row in route_cases
                },
                "unforced_route_success_counts": unforced_route_counts,
                "unforced_accepted_count": sum(
                    unforced_route_counts.values()
                ),
                "natural_support_pass": natural_support,
            }
        )
    problem_results.sort(key=_problem_rank)
    selected = [
        row["source_index"]
        for row in problem_results
        if row["bidirectionally_executable"]
    ][:10]
    payload = {
        "schema": SCHEMA,
        "pass": len(selected) == 10,
        "candidate_count": len(candidates),
        "singleton_pair_count": len(pair_candidates),
        "singleton_confirmed_count": len(confirmed_records),
        "double_audited_menu_count": len(accepted),
        "bidirectionally_executable_count": sum(
            row["bidirectionally_executable"] for row in problem_results
        ),
        "selected_source_indices": selected,
        "wrong_route_success_count": sum(
            row["wrong_route_success_count"] for row in case_results
        ),
        "all_observed_response_count": 1550,
        "forced_sample_count_per_route": FORCED_SAMPLE_COUNT,
        "minimum_forced_successes_per_route": MIN_FORCED_SUCCESSES,
        "unforced_sample_count_per_problem": UNFORCED_SAMPLE_COUNT,
        "minimum_natural_successes_per_route": MIN_NATURAL_PER_ROUTE,
        "minimum_natural_accepted_total": MIN_NATURAL_TOTAL,
        "forced_canonicalizer_diagnostics": {
            field: float(getattr(diagnostics, field))
            for field in diagnostics.__dataclass_fields__
        },
        "unforced_canonicalizer_diagnostics": {
            field: float(getattr(unforced_diagnostics, field))
            for field in unforced_diagnostics.__dataclass_fields__
        },
        "identity": {
            "protocol_sha256": _sha256(PROTOCOL),
            "script_sha256": _sha256(SCRIPT),
            "base_e49ab_script_sha256": _sha256(BASE_AB_SCRIPT),
            "e49ab_result_sha256": _sha256(AB_RESULT),
            "e49ab_cluster_records_sha256": _sha256(AB_CLUSTERS),
            "pairwise_source_sha256": _sha256(PAIRWISE_SOURCE),
            "e49t_identity_sha256": _sha256(E49T_IDENTITY),
        },
        "singleton_confirmation_records_sha256": _sha256(singleton_path),
        "menu_records_sha256": _sha256(menu_path),
        "forced_responses_sha256": _sha256(forced_path),
        "unforced_responses_sha256": _sha256(unforced_path),
        "cases": case_results,
        "problems": problem_results,
    }
    _write_json(result_path, payload)
    print(result_path)


if __name__ == "__main__":
    main()
