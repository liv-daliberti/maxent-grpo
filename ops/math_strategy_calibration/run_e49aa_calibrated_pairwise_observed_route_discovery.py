#!/usr/bin/env python3
"""Find executable hard-MATH routes with the calibrated E47W pairwise veto."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
FULL = ROOT / "var/data/math12k_384_math500"
MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
E49Y_ROOT = ROOT / "var/artifacts/e49y_observed_route_discovery_v1"
NATURAL_RESPONSES = E49Y_ROOT / "private/natural_responses.jsonl"
SAMPLING_SUMMARY = E49Y_ROOT / "sampling_summary.jsonl"
E49Y_RESULT = E49Y_ROOT / "result.json"
E49T_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e49t_natural_menu_math_toy_05b_3ep_v1_identity.json"
)
ENDPOINT_RECORD = (
    ROOT / "var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
)
E47W_ANALYSIS = (
    ROOT
    / "var/artifacts/e47w_pairwise_math_strategy_calibration_v1/"
    "analysis.json"
)
PAIRWISE_SOURCE = (
    ROOT
    / "var/artifacts/source_snapshots/"
    "e49b_math_strategy_ae06b1023ade504c7b8a07e901874e5b563b9e8fad618e5fefe519337458ebfc/"
    "src/oat_drgrpo/math_strategy_canonicalizer.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49aa_calibrated_pairwise_observed_route_discovery_20260726.md"
)
EXPECTED = {
    "natural": "ab6c5c081919caaf358c65ff63e394b0fc6587dc98fb69318a4c77b5aa76401f",
    "summary": "e6203f7dec2a4b751d6d47e090c3c9884974f9dc3535e2dcfc61fef7edac7fc3",
    "e47w": "3ad930355d4ef3cc30f153035db3cfdca5bf8290566b00fe3383111c4c7455d6",
    "pairwise": "91a1a8fdc3b29fa49b1089154f5955ec2d9e759cc85e94b35bc7d101d81bf988",
    "endpoint": "964ce3bda1d2cac7dddf2065451fd2b456433478356df9d493e98f9e23c66ac3",
}
FORCED_SAMPLE_COUNT = 8
MAX_EXEMPLARS = 16
MAX_EXEMPLAR_CHARS = 6000
SEED = 490811

sys.path.insert(0, str(ROOT / "ops/math_strategy_calibration"))
from run_e49w_bottom_up_route_calibration import (  # noqa: E402
    _audit,
    _audit_passes,
    _endpoint,
    _forced_problem,
    _parse_menu,
)
from run_e49y_observed_route_discovery import (  # noqa: E402
    _propose_observed_menu,
)


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


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
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


def _load_pairwise_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "e49aa_frozen_pairwise_canonicalizer", PAIRWISE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen E47W canonicalizer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_inputs() -> None:
    observed = {
        "natural": _sha256(NATURAL_RESPONSES),
        "summary": _sha256(SAMPLING_SUMMARY),
        "e47w": _sha256(E47W_ANALYSIS),
        "pairwise": _sha256(PAIRWISE_SOURCE),
        "endpoint": _sha256(ENDPOINT_RECORD),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"E49AA frozen input mismatch: {observed}")
    e49y = json.loads(E49Y_RESULT.read_text(encoding="utf-8"))
    if (
        e49y.get("schema") != "e49y_observed_route_discovery_v1"
        or e49y.get("pass") is True
    ):
        raise RuntimeError("E49AA activates only after final E49Y failure")
    e47w = json.loads(E47W_ANALYSIS.read_text(encoding="utf-8"))
    if (
        e47w.get("schema") != "e47w_pairwise_veto_calibration_report_v1"
        or e47w.get("gate_status") != "pass"
        or not all(e47w.get("gate_checks", {}).values())
    ):
        raise RuntimeError("frozen E47W pairwise calibration is not passing")


def _candidates(full_rows: Any) -> list[dict[str, Any]]:
    summaries = _load_jsonl(SAMPLING_SUMMARY)
    if (
        len(summaries) != 211
        or sum(int(row["validator_positive_count"]) for row in summaries)
        != 1647
    ):
        raise RuntimeError("E49AA sampling summary drifted")
    eligible_indices = {
        int(row["source_index"])
        for row in summaries
        if int(row["validator_positive_count"]) >= 4
    }
    if len(eligible_indices) != 74:
        raise RuntimeError("E49AA expected exactly 74 candidate rows")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(NATURAL_RESPONSES):
        if (
            int(row["source_index"]) in eligible_indices
            and row.get("answer_positive") is True
            and len(str(row["response"])) <= MAX_EXEMPLAR_CHARS
        ):
            grouped[int(row["source_index"])].append(
                {
                    "sample_id": str(row["response_sha256"])[:20],
                    "sample_index": int(row["sample_index"]),
                    "text": str(row["response"]),
                    "response_sha256": str(row["response_sha256"]),
                }
            )
    candidates = []
    for source_index in sorted(eligible_indices):
        exemplars = sorted(
            grouped[source_index],
            key=lambda row: (len(row["text"]), row["response_sha256"]),
        )[:MAX_EXEMPLARS]
        if len(exemplars) < 4:
            raise RuntimeError("E49AA candidate lost its minimum exemplars")
        source = dict(full_rows[source_index])
        candidates.append(
            {
                "source_index": source_index,
                "row_id": str(source["unique_id"]),
                "problem": str(source["problem"]),
                "answer": str(source["answer"]),
                "subject": str(source["subject"]),
                "level": int(source["level"]),
                "exemplars": exemplars,
            }
        )
    return candidates


def _cluster_candidate(
    module: Any,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    base = {
        key: candidate[key]
        for key in ("source_index", "row_id", "subject", "level")
    }
    try:
        judge = module.MathStrategyCanonicalizer(
            endpoint=endpoint,
            model=model,
            timeout_seconds=timeout,
            max_workers=2,
            permutation_seeds=(470721, 470722),
            max_item_chars=4000,
            missing_ids_are_ambiguous=True,
        )
        exemplars = candidate["exemplars"]
        keys, diagnostics = judge.canonicalize(
            prompt_token_ids=[
                [490811, int(candidate["source_index"]) + 1]
                for _ in exemplars
            ],
            prompt_texts=[candidate["problem"]] * len(exemplars),
            response_texts=[row["text"] for row in exemplars],
            task_reward_positive=[True] * len(exemplars),
            active_mask=[True] * len(exemplars),
            num_samples=len(exemplars),
        )
        members: dict[str, list[str]] = defaultdict(list)
        for exemplar, key in zip(exemplars, keys, strict=True):
            if key is not None:
                members[str(key)].append(str(exemplar["sample_id"]))
        eligible = [
            (key, member_ids)
            for key, member_ids in members.items()
            if len(member_ids) >= 2
        ]
        eligible.sort(key=lambda item: (-len(item[1]), item[0]))
        state = judge.state_dict()
        representative_groups = list(state["representatives"].values())
        descriptions = (
            representative_groups[0] if len(representative_groups) == 1 else {}
        )
        clusters = [
            {
                "cluster_id": f"C{index}",
                "strategy": (
                    descriptions.get(key, {}).get("description")
                    or f"calibrated pairwise component {key}"
                ),
                "strategy_key": key,
                "member_ids": member_ids,
            }
            for index, (key, member_ids) in enumerate(
                eligible[:2], start=1
            )
        ]
        return {
            **base,
            "diagnostics": {
                field: float(getattr(diagnostics, field))
                for field in diagnostics.__dataclass_fields__
            },
            "clusters": clusters,
            "cluster_eligible": len(clusters) == 2,
            "minimum_cluster_size": (
                min(len(row["member_ids"]) for row in clusters)
                if len(clusters) == 2
                else 0
            ),
            "combined_cluster_size": sum(
                len(row["member_ids"]) for row in clusters
            ),
            "outcome_keys": keys,
        }
    except Exception as exc:
        return {**base, "cluster_eligible": False, "error": str(exc)}


def _menu_candidate(
    *,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    cluster_record: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    retained_ids = {
        sample_id
        for cluster in cluster_record["clusters"]
        for sample_id in cluster["member_ids"]
    }
    audit_candidate = {
        **candidate,
        "exemplars": [
            row
            for row in candidate["exemplars"]
            if row["sample_id"] in retained_ids
        ],
    }
    base = {
        key: cluster_record[key]
        for key in (
            "source_index",
            "row_id",
            "subject",
            "level",
            "minimum_cluster_size",
            "combined_cluster_size",
        )
    }
    try:
        proposal, generation = _propose_observed_menu(
            endpoint=endpoint,
            model=model,
            candidate=audit_candidate,
            clusters=cluster_record["clusters"],
            timeout=timeout,
        )
        menu = _parse_menu(proposal["menu"])
        audits = [
            _audit(
                endpoint=endpoint,
                model=model,
                candidate=audit_candidate,
                proposal=proposal,
                audit_index=index,
                timeout=timeout,
            )
            for index in range(2)
        ]
        return {
            **base,
            "generation": generation,
            "menu": json.loads(menu.canonical_json),
            "menu_sha256": menu.sha256,
            "route_sources": proposal["route_sources"],
            "audits": audits,
            "double_audit_pass": all(
                _audit_passes(proposal, audit) for audit in audits
            ),
        }
    except Exception as exc:
        return {**base, "double_audit_pass": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh E49AA result required: {result_path}")
    _verify_inputs()

    from datasets import load_from_disk
    from oat_drgrpo.math_grader import boxed_reward_fn
    from oat_drgrpo.templates import apply_qwen_math_template

    endpoint, judge_model = _endpoint(ENDPOINT_RECORD)
    full_rows = load_from_disk(str(FULL / "train"))["train"]
    candidates = _candidates(full_rows)
    candidate_by_index = {
        int(row["source_index"]): row for row in candidates
    }
    pairwise_module = _load_pairwise_module()

    cluster_records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _cluster_candidate,
                pairwise_module,
                endpoint,
                judge_model,
                candidate,
                args.timeout,
            )
            for candidate in candidates
        ]
        for future in as_completed(futures):
            record = future.result()
            cluster_records.append(record)
            print(
                json.dumps(
                    {
                        "phase": "cluster",
                        "source_index": record["source_index"],
                        "eligible": record["cluster_eligible"],
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    cluster_records.sort(key=lambda row: int(row["source_index"]))
    cluster_path = output / "cluster_records.jsonl"
    _write_jsonl(cluster_path, cluster_records)
    cluster_eligible = [
        row for row in cluster_records if row["cluster_eligible"]
    ]

    menu_records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _menu_candidate,
                endpoint=endpoint,
                model=judge_model,
                candidate=candidate_by_index[int(record["source_index"])],
                cluster_record=record,
                timeout=args.timeout,
            )
            for record in cluster_eligible
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
        payload = {
            "schema": "e49aa_calibrated_pairwise_observed_route_discovery_v1",
            "pass": False,
            "failure": "no_double_audited_pairwise_observed_menu",
            "candidate_count": len(candidates),
            "cluster_eligible_count": len(cluster_eligible),
            "double_audited_menu_count": 0,
            "cluster_records_sha256": _sha256(cluster_path),
            "menu_records_sha256": _sha256(menu_path),
        }
        _write_json(result_path, payload)
        print(result_path)
        return

    import vllm

    cases = []
    for record in accepted:
        source = dict(full_rows[int(record["source_index"])])
        menu = _parse_menu(record["menu"])
        for strategy in menu.strategies:
            cases.append(
                {
                    "source_index": int(record["source_index"]),
                    "row_id": str(record["row_id"]),
                    "problem": _forced_problem(
                        str(source["problem"]), menu, strategy
                    ),
                    "answer": str(source["answer"]),
                    "menu": menu,
                    "strategy": strategy,
                }
            )
    llm = vllm.LLM(
        model=str(MODEL),
        dtype="bfloat16",
        max_model_len=2048,
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
    flat_tokens = []
    flat_problems = []
    flat_responses = []
    flat_positive = []
    private_forced = []
    for case_index, (case, request_output) in enumerate(
        zip(cases, forced_outputs, strict=True)
    ):
        if len(request_output.outputs) != FORCED_SAMPLE_COUNT:
            raise RuntimeError("E49AA forced sample count mismatch")
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

    identity = json.loads(E49T_IDENTITY.read_text(encoding="utf-8"))
    frozen_source = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e49t_natural_menu_{identity['source_hash']}"
        / "src"
    )
    import oat_drgrpo

    oat_drgrpo.__path__.insert(0, str(frozen_source / "oat_drgrpo"))
    from oat_drgrpo.math_strategy_canonicalizer import (
        MathStrategyCanonicalizer,
    )

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
                "minimally_executable": expected >= 1,
            }
        )
    problem_results = []
    for order, record in enumerate(accepted):
        route_cases = [
            row
            for row in case_results
            if row["source_index"] == int(record["source_index"])
        ]
        if len(route_cases) != 2:
            raise RuntimeError("E49AA accepted menu lost one route")
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
                ),
                "route_success_counts": {
                    row["strategy_id"]: row["forced_route_success_count"]
                    for row in route_cases
                },
                "answer_success_counts": {
                    row["strategy_id"]: row["answer_success_count"]
                    for row in route_cases
                },
            }
        )
    selected = [
        row["source_index"]
        for row in problem_results
        if row["bidirectionally_executable"]
    ][:10]
    forced_path = output / "private/forced_responses.jsonl"
    _write_jsonl(forced_path, private_forced)
    payload = {
        "schema": "e49aa_calibrated_pairwise_observed_route_discovery_v1",
        "pass": len(selected) == 10,
        "candidate_count": len(candidates),
        "cluster_eligible_count": len(cluster_eligible),
        "double_audited_menu_count": len(accepted),
        "bidirectionally_executable_count": sum(
            row["bidirectionally_executable"] for row in problem_results
        ),
        "selected_source_indices": selected,
        "wrong_route_success_count": sum(
            row["wrong_route_success_count"] for row in case_results
        ),
        "canonicalizer_diagnostics": {
            field: float(getattr(diagnostics, field))
            for field in diagnostics.__dataclass_fields__
        },
        "identity": {
            **EXPECTED,
            "protocol_sha256": _sha256(PROTOCOL),
            "script_sha256": _sha256(SCRIPT),
            "e49y_result_sha256": _sha256(E49Y_RESULT),
            "e49t_identity_sha256": _sha256(E49T_IDENTITY),
        },
        "cluster_records_sha256": _sha256(cluster_path),
        "menu_records_sha256": _sha256(menu_path),
        "forced_responses_sha256": _sha256(forced_path),
        "cases": case_results,
        "problems": problem_results,
    }
    _write_json(result_path, payload)
    print(result_path)


if __name__ == "__main__":
    main()
