#!/usr/bin/env python3
"""Discover dual hard-MATH routes that Qwen2.5-0.5B actually writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
import tempfile
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
E49T_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e49t_natural_menu_math_toy_05b_3ep_v1_identity.json"
)
ENDPOINT_RECORD = (
    ROOT / "var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/e49y_observed_route_discovery_20260726.md"
)
SAMPLE_COUNT = 64
FORCED_SAMPLE_COUNT = 8
SEED = 490791
MAX_EXEMPLARS = 16
MAX_EXEMPLAR_CHARS = 6000
SYSTEM = (
    "You are a conservative mathematical solution-route auditor. "
    "Return valid JSON only."
)
PARTITION_MODE = "global_exact_relation"

sys.path.insert(0, str(ROOT / "ops/math_strategy_calibration"))
from run_e49w_bottom_up_route_calibration import (  # noqa: E402
    _audit,
    _audit_passes,
    _endpoint,
    _forced_problem,
    _parse_menu,
    _post,
    _proposal_passes_local_contract,
    _proposal_schema,
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _partition_schema(sample_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["clusters"],
        "properties": {
            "clusters": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["cluster_id", "strategy", "member_ids"],
                    "properties": {
                        "cluster_id": {
                            "type": "string",
                            "enum": [f"C{index}" for index in range(1, 9)],
                        },
                        "strategy": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 512,
                        },
                        "member_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "string",
                                "enum": sample_ids,
                            },
                        },
                    },
                },
            }
        },
    }


def _exact_assignment_partition_schema(
    sample_ids: list[str],
) -> dict[str, Any]:
    cluster_ids = [f"C{index}" for index in range(1, 9)]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assignments", "cluster_strategies"],
        "properties": {
            "assignments": {
                "type": "object",
                "additionalProperties": False,
                "required": sample_ids,
                "properties": {
                    sample_id: {
                        "type": "string",
                        "enum": cluster_ids,
                    }
                    for sample_id in sample_ids
                },
            },
            "cluster_strategies": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["cluster_id", "strategy"],
                    "properties": {
                        "cluster_id": {
                            "type": "string",
                            "enum": cluster_ids,
                        },
                        "strategy": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 512,
                        },
                    },
                },
            },
        },
    }


def _validate_partition(
    partition: dict[str, Any],
    sample_ids: list[str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    clusters = partition.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("partition returned no clusters")
    if [row.get("cluster_id") for row in clusters] != [
        f"C{index}" for index in range(1, len(clusters) + 1)
    ]:
        raise ValueError("partition cluster IDs are not consecutive")
    assignment: dict[str, str] = {}
    normalized = []
    for cluster in clusters:
        member_ids = cluster.get("member_ids")
        if (
            not isinstance(member_ids, list)
            or len(member_ids) != len(set(member_ids))
        ):
            raise ValueError("partition cluster contains duplicate members")
        for sample_id in member_ids:
            if sample_id in assignment:
                raise ValueError("partition assigned one response twice")
            assignment[sample_id] = str(cluster["cluster_id"])
        normalized.append(
            {
                "cluster_id": str(cluster["cluster_id"]),
                "strategy": str(cluster["strategy"]).strip(),
                "member_ids": list(member_ids),
            }
        )
    if set(assignment) != set(sample_ids):
        raise ValueError("partition did not assign every supplied response")
    return assignment, normalized


def _validate_exact_assignment_partition(
    partition: dict[str, Any],
    sample_ids: list[str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    assignments = partition.get("assignments")
    descriptions = partition.get("cluster_strategies")
    if not isinstance(assignments, dict) or set(assignments) != set(sample_ids):
        raise ValueError("exact partition did not assign every response")
    if not isinstance(descriptions, list) or not descriptions:
        raise ValueError("exact partition returned no cluster descriptions")
    description_by_id: dict[str, str] = {}
    for row in descriptions:
        cluster_id = str(row.get("cluster_id", ""))
        strategy = str(row.get("strategy", "")).strip()
        if not cluster_id or not strategy or cluster_id in description_by_id:
            raise ValueError("exact partition cluster descriptions invalid")
        description_by_id[cluster_id] = strategy
    assignment = {
        sample_id: str(assignments[sample_id]) for sample_id in sample_ids
    }
    used = set(assignment.values())
    if used != set(description_by_id):
        raise ValueError(
            "exact partition descriptions do not match used clusters"
        )
    normalized = [
        {
            "cluster_id": cluster_id,
            "strategy": description_by_id[cluster_id],
            "member_ids": [
                sample_id
                for sample_id in sample_ids
                if assignment[sample_id] == cluster_id
            ],
        }
        for cluster_id in sorted(
            used, key=lambda value: int(value.removeprefix("C"))
        )
    ]
    return assignment, normalized


def _same_relation(
    assignment: dict[str, str],
    sample_ids: list[str],
) -> tuple[bool, ...]:
    return tuple(
        assignment[left] == assignment[right]
        for left_index, left in enumerate(sample_ids)
        for right in sample_ids[left_index + 1 :]
    )


def _partition_once(
    *,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    pass_index: int,
    timeout: int,
) -> dict[str, Any]:
    exemplars = list(candidate["exemplars"])
    order = list(range(len(exemplars)))
    random.Random(SEED + 1000 + pass_index).shuffle(order)
    ordered = [exemplars[index] for index in order]
    rendered = "\n\n".join(
        f"RESPONSE {row['sample_id']}:\n{row['text']}" for row in ordered
    )
    sample_ids = [str(row["sample_id"]) for row in exemplars]
    exact_assignment = PARTITION_MODE == "consensus_intersection"
    output_instruction = (
        "Assign every response ID through the required assignments object, "
        "then describe every used label in cluster_strategies."
        if exact_assignment
        else (
            "Use C1..Cn consecutively in the order each cluster first "
            "appears below."
        )
    )
    prompt = f"""Partition these independently answer-validated solutions to
one hard MATH problem by genuinely distinct executed solution strategy.

Every response must appear in exactly one cluster. Two responses belong
together when their decisive equation, identity, theorem, construction,
counted set, substitution, or search method is the same after routine
algebra. Wording, notation, step order, extra checks, and expanded versus
factored forms do not create a new route. Split only when the responses
actually execute different central methods. {output_instruction} Do not
invent a route.

PROBLEM:
{candidate['problem']}

CORRECT NATURAL RESPONSES:
{rendered}
"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 3072,
        "seed": SEED + 100 + pass_index,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "observed_route_partition",
                "strict": True,
                "schema": (
                    _exact_assignment_partition_schema(sample_ids)
                    if exact_assignment
                    else _partition_schema(sample_ids)
                ),
            },
        },
    }
    response, partition = _post(endpoint, payload, timeout=timeout)
    if exact_assignment:
        assignment, clusters = _validate_exact_assignment_partition(
            partition, sample_ids
        )
    else:
        assignment, clusters = _validate_partition(partition, sample_ids)
    return {
        "pass_index": pass_index,
        "seed": payload["seed"],
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "request_order": [str(row["sample_id"]) for row in ordered],
        "clusters": clusters,
        "assignment": assignment,
    }


def _stable_clusters(
    *,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    timeout: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    passes = [
        _partition_once(
            endpoint=endpoint,
            model=model,
            candidate=candidate,
            pass_index=pass_index,
            timeout=timeout,
        )
        for pass_index in range(2)
    ]
    sample_ids = [str(row["sample_id"]) for row in candidate["exemplars"]]
    if PARTITION_MODE == "consensus_intersection":
        description_maps = [
            {
                str(cluster["cluster_id"]): str(cluster["strategy"])
                for cluster in partition["clusters"]
            }
            for partition in passes
        ]
        consensus: dict[tuple[str, str], list[str]] = {}
        for sample_id in sample_ids:
            signature = (
                passes[0]["assignment"][sample_id],
                passes[1]["assignment"][sample_id],
            )
            consensus.setdefault(signature, []).append(sample_id)
        groups = [
            {
                "cluster_id": f"{signature[0]}__{signature[1]}",
                "partition_labels": list(signature),
                "strategy": (
                    f"partition 1: {description_maps[0][signature[0]]}; "
                    f"partition 2: {description_maps[1][signature[1]]}"
                ),
                "member_ids": members,
            }
            for signature, members in consensus.items()
            if len(members) >= 2
        ]
        pairs = [
            (left, right)
            for left_index, left in enumerate(groups)
            for right in groups[left_index + 1 :]
            if (
                left["partition_labels"][0]
                != right["partition_labels"][0]
                and left["partition_labels"][1]
                != right["partition_labels"][1]
            )
        ]
        if not pairs:
            raise ValueError(
                "no two consensus groups are distinct in both partitions"
            )
        pairs.sort(
            key=lambda pair: (
                -min(
                    len(pair[0]["member_ids"]),
                    len(pair[1]["member_ids"]),
                ),
                -(
                    len(pair[0]["member_ids"])
                    + len(pair[1]["member_ids"])
                ),
                min(pair[0]["member_ids"]),
                min(pair[1]["member_ids"]),
            )
        )
        selected = list(pairs[0])
        selected.sort(
            key=lambda cluster: (
                -len(cluster["member_ids"]),
                min(cluster["member_ids"]),
            )
        )
        return selected, passes
    relations = [
        _same_relation(partition["assignment"], sample_ids)
        for partition in passes
    ]
    if relations[0] != relations[1]:
        raise ValueError("two route partitions disagree pairwise")
    first = passes[0]["clusters"]
    eligible = [cluster for cluster in first if len(cluster["member_ids"]) >= 2]
    if len(eligible) < 2:
        raise ValueError("fewer than two stable clusters have two members")
    eligible.sort(
        key=lambda cluster: (
            -len(cluster["member_ids"]),
            min(cluster["member_ids"]),
        )
    )
    return eligible[:2], passes


def _propose_observed_menu(
    *,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    clusters: list[dict[str, Any]],
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_id = {
        str(row["sample_id"]): row for row in candidate["exemplars"]
    }
    retained_ids = [
        sample_id
        for cluster in clusters
        for sample_id in cluster["member_ids"]
    ]
    retained_exemplars = [by_id[sample_id] for sample_id in retained_ids]
    rendered_clusters = "\n\n".join(
        f"ROUTE {strategy_id} — partition description: {cluster['strategy']}\n"
        + "\n\n".join(
            f"EXEMPLAR {sample_id}:\n{by_id[sample_id]['text']}"
            for sample_id in cluster["member_ids"]
        )
        for strategy_id, cluster in zip(("S1", "S2"), clusters, strict=True)
    )
    prompt = f"""Convert exactly two already-observed solution clusters into
a finite S1/S2 action menu for this hard MATH problem.

Both routes are observed; proposed routes are forbidden. S1 must describe
only the supplied S1 cluster and S2 only the supplied S2 cluster. Cite one to
three actual exemplar IDs for each route.

Build one shared action vocabulary A1..An. Each action must be a concise,
problem-specific operation actually executed by the cited route. Each combo
must contain every decisive step needed to derive the answer. Do not use
vague operations, import a hidden theorem, combine routes, or include/hint
the final answer. Preserve the central distinction found by the partition.

PROBLEM:
{candidate['problem']}

OBSERVED ROUTE CLUSTERS:
{rendered_clusters}
"""
    sample_ids = [str(row["sample_id"]) for row in retained_exemplars]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 3072,
        "seed": SEED + 200,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "e49y_observed_route_menu",
                "strict": True,
                "schema": _proposal_schema(sample_ids),
            },
        },
    }
    response, proposal = _post(endpoint, payload, timeout=timeout)
    if not _proposal_passes_local_contract(proposal, set(sample_ids)):
        raise ValueError("observed-route menu failed local finite contract")
    if any(
        source["source_kind"] != "observed"
        for source in proposal["route_sources"]
    ):
        raise ValueError("E49Y menu attempted to introduce a proposed route")
    menu = _parse_menu(proposal["menu"])
    return proposal, {
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "seed": payload["seed"],
        "menu_sha256": menu.sha256,
    }


def _build_route_candidate(
    *,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    base = {
        "source_index": candidate["source_index"],
        "row_id": candidate["row_id"],
        "problem_sha256": _sha256_text(candidate["problem"]),
        "reference_answer_sha256": _sha256_text(candidate["answer"]),
        "subject": candidate["subject"],
        "level": candidate["level"],
        "validator_positive_count": candidate["validator_positive_count"],
        "selected_exemplar_ids": [
            row["sample_id"] for row in candidate["exemplars"]
        ],
    }
    try:
        clusters, partitions = _stable_clusters(
            endpoint=endpoint,
            model=model,
            candidate=candidate,
            timeout=timeout,
        )
        retained_ids = {
            sample_id
            for cluster in clusters
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
        proposal, generation = _propose_observed_menu(
            endpoint=endpoint,
            model=model,
            candidate=audit_candidate,
            clusters=clusters,
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
        passed = all(_audit_passes(proposal, audit) for audit in audits)
        cluster_sizes = [len(cluster["member_ids"]) for cluster in clusters]
        return {
            **base,
            "stable_partitions": partitions,
            "retained_clusters": clusters,
            "minimum_cluster_size": min(cluster_sizes),
            "combined_cluster_size": sum(cluster_sizes),
            "generation": generation,
            "menu": json.loads(menu.canonical_json),
            "menu_sha256": menu.sha256,
            "route_sources": proposal["route_sources"],
            "audits": audits,
            "double_audit_pass": passed,
        }
    except Exception as exc:
        return {**base, "double_audit_pass": False, "error": str(exc)}


def _candidates_from_private_rows(
    *,
    rows: list[tuple[int, dict[str, Any]]],
    private_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[int, list[dict[str, Any]]] = {}
    for private in private_rows:
        by_source.setdefault(int(private["source_index"]), []).append(private)
    candidates = []
    public_sampling = []
    for source_index, row in rows:
        observed = sorted(
            by_source.get(source_index, []),
            key=lambda item: int(item["sample_index"]),
        )
        if len(observed) != SAMPLE_COUNT:
            raise RuntimeError(
                f"source {source_index} has {len(observed)} frozen samples"
            )
        if [int(item["sample_index"]) for item in observed] != list(
            range(SAMPLE_COUNT)
        ):
            raise RuntimeError(
                f"source {source_index} frozen sample indices are invalid"
            )
        if any(str(item["row_id"]) != str(row["unique_id"]) for item in observed):
            raise RuntimeError(
                f"source {source_index} frozen row identity mismatch"
            )
        positives = []
        for private in observed:
            text = str(private["response"])
            response_sha256 = _sha256_text(text)
            if response_sha256 != str(private["response_sha256"]):
                raise RuntimeError(
                    f"source {source_index} frozen response hash mismatch"
                )
            if bool(private["answer_positive"]):
                positives.append(
                    {
                        "sample_id": response_sha256[:20],
                        "sample_index": int(private["sample_index"]),
                        "text": text,
                        "response_sha256": response_sha256,
                    }
                )
        public_sampling.append(
            {
                "source_index": source_index,
                "row_id": str(row["unique_id"]),
                "level": int(row["level"]),
                "validator_positive_count": len(positives),
            }
        )
        eligible = sorted(
            (
                positive
                for positive in positives
                if len(positive["text"]) <= MAX_EXEMPLAR_CHARS
            ),
            key=lambda positive: (
                len(positive["text"]),
                positive["response_sha256"],
            ),
        )[:MAX_EXEMPLARS]
        if len(positives) >= 4 and len(eligible) >= 4:
            candidates.append(
                {
                    "source_index": source_index,
                    "row_id": str(row["unique_id"]),
                    "problem": str(row["problem"]),
                    "answer": str(row["answer"]),
                    "subject": str(row["subject"]),
                    "level": int(row["level"]),
                    "validator_positive_count": len(positives),
                    "exemplars": eligible,
                }
            )
    return candidates, public_sampling


def main() -> None:
    global PARTITION_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=(
            ROOT / "var/artifacts/e49y_observed_route_discovery_v1"
        ),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--partition-mode",
        choices=("global_exact_relation", "consensus_intersection"),
        default="global_exact_relation",
    )
    parser.add_argument(
        "--reuse-natural-root",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--expected-natural-sha256",
        default=None,
    )
    parser.add_argument(
        "--protocol",
        type=pathlib.Path,
        default=PROTOCOL,
    )
    parser.add_argument(
        "--result-schema",
        default="e49y_observed_route_discovery_v1",
    )
    args = parser.parse_args()
    PARTITION_MODE = args.partition_mode
    output = args.output_root.resolve()
    protocol = args.protocol.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh route result required: {result_path}")

    sys.path.insert(0, str(ROOT / "src"))
    from datasets import load_from_disk
    from oat_drgrpo.math_grader import boxed_reward_fn
    from oat_drgrpo.templates import apply_qwen_math_template

    endpoint, judge_model = _endpoint(ENDPOINT_RECORD)
    full = load_from_disk(str(FULL / "train"))["train"]
    rows = [
        (source_index, dict(row))
        for source_index, row in enumerate(full)
        if int(row["level"]) >= 4
    ]
    if len(rows) != 211:
        raise RuntimeError(f"expected 211 hard rows, found {len(rows)}")
    llm = None
    if args.reuse_natural_root is not None:
        source_private = (
            args.reuse_natural_root.resolve()
            / "private/natural_responses.jsonl"
        )
        if args.expected_natural_sha256 is None:
            raise RuntimeError(
                "--reuse-natural-root requires --expected-natural-sha256"
            )
        actual_sha256 = _sha256(source_private)
        if actual_sha256 != args.expected_natural_sha256:
            raise RuntimeError(
                "frozen natural corpus hash mismatch: "
                f"{actual_sha256} != {args.expected_natural_sha256}"
            )
        private_rows = [
            json.loads(line)
            for line in source_private.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        import vllm

        prompts = [
            apply_qwen_math_template(str(row["problem"])) for _, row in rows
        ]
        llm = vllm.LLM(
            model=str(MODEL),
            dtype="bfloat16",
            max_model_len=2048,
            gpu_memory_utilization=0.80,
            swap_space=8.0,
            enable_prefix_caching=True,
            seed=SEED,
        )
        outputs = llm.generate(
            prompts,
            vllm.SamplingParams(
                n=SAMPLE_COUNT,
                temperature=1.0,
                top_p=1.0,
                max_tokens=1024,
                seed=SEED,
            ),
        )
        if len(outputs) != len(rows):
            raise RuntimeError("natural generation count mismatch")
        private_rows = []
        for (source_index, row), request_output in zip(
            rows, outputs, strict=True
        ):
            if len(request_output.outputs) != SAMPLE_COUNT:
                raise RuntimeError("natural sample count mismatch")
            for sample_index, sample in enumerate(request_output.outputs):
                text = str(sample.text)
                _, reward = boxed_reward_fn(
                    text, str(row["answer"]), fast=False
                )
                private_rows.append(
                    {
                        "source_index": source_index,
                        "row_id": str(row["unique_id"]),
                        "sample_index": sample_index,
                        "answer_positive": float(reward) > 0.0,
                        "response_sha256": _sha256_text(text),
                        "response": text,
                    }
                )
    candidates, public_sampling = _candidates_from_private_rows(
        rows=rows,
        private_rows=private_rows,
    )
    private_path = output / "private/natural_responses.jsonl"
    _write_jsonl(private_path, private_rows)
    _write_jsonl(output / "sampling_summary.jsonl", public_sampling)
    print(
        json.dumps(
            {
                "hard_rows": len(rows),
                "eligible_route_rows": len(candidates),
                "validator_positive": sum(
                    row["validator_positive_count"] for row in public_sampling
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    route_records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _build_route_candidate,
                endpoint=endpoint,
                model=judge_model,
                candidate=candidate,
                timeout=args.timeout,
            ): candidate["source_index"]
            for candidate in candidates
        }
        for future in as_completed(futures):
            record = future.result()
            route_records.append(record)
            print(
                json.dumps(
                    {
                        "source_index": record["source_index"],
                        "row_id": record["row_id"],
                        "double_audit_pass": record["double_audit_pass"],
                        "error": record.get("error"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    route_records.sort(key=lambda record: int(record["source_index"]))
    route_records_path = output / "route_records.jsonl"
    _write_jsonl(route_records_path, route_records)
    accepted = [
        record for record in route_records if record["double_audit_pass"]
    ]
    accepted.sort(
        key=lambda record: (
            -int(record["minimum_cluster_size"]),
            -int(record["combined_cluster_size"]),
            int(record["source_index"]),
        )
    )
    if not accepted:
        _write_json(
            result_path,
            {
                "schema": args.result_schema,
                "pass": False,
                "failure": "no_double_audited_observed_route_menu",
                "hard_row_count": len(rows),
                "eligible_route_row_count": len(candidates),
                "double_audited_menu_count": 0,
                "partition_mode": PARTITION_MODE,
                "natural_responses_sha256": _sha256(private_path),
            },
        )
        print(result_path)
        return

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

    source_by_index = {source_index: row for source_index, row in rows}
    cases = []
    for record in accepted:
        row = source_by_index[int(record["source_index"])]
        menu = _parse_menu(record["menu"])
        for strategy in menu.strategies:
            cases.append(
                {
                    "source_index": record["source_index"],
                    "row_id": record["row_id"],
                    "problem": _forced_problem(
                        str(row["problem"]), menu, strategy
                    ),
                    "answer": str(row["answer"]),
                    "menu": menu,
                    "strategy": strategy,
                }
            )
    forced_prompts = [
        apply_qwen_math_template(case["problem"]) for case in cases
    ]
    if llm is None:
        import vllm

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
        forced_prompts,
        vllm.SamplingParams(
            n=FORCED_SAMPLE_COUNT,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            seed=SEED + 1,
        ),
    )
    flat_prompt_tokens = []
    flat_problem_texts = []
    flat_responses = []
    flat_answer_positive = []
    forced_private_rows = []
    for case_index, (case, request_output) in enumerate(
        zip(cases, forced_outputs, strict=True)
    ):
        if len(request_output.outputs) != FORCED_SAMPLE_COUNT:
            raise RuntimeError("E49Y forced sample count mismatch")
        prompt_tokens = list(request_output.prompt_token_ids)
        for sample_index, sample in enumerate(request_output.outputs):
            text = str(sample.text)
            _, reward = boxed_reward_fn(text, case["answer"], fast=False)
            answer_positive = float(reward) > 0.0
            flat_prompt_tokens.append(prompt_tokens)
            flat_problem_texts.append(case["problem"])
            flat_responses.append(text)
            flat_answer_positive.append(answer_positive)
            forced_private_rows.append(
                {
                    "case_index": case_index,
                    "source_index": case["source_index"],
                    "row_id": case["row_id"],
                    "strategy_id": case["strategy"].strategy_id,
                    "action_combo": case["strategy"].action_combo,
                    "sample_index": sample_index,
                    "answer_positive": answer_positive,
                    "response_sha256": _sha256_text(text),
                    "response": text,
                }
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
        prompt_token_ids=flat_prompt_tokens,
        prompt_texts=flat_problem_texts,
        response_texts=flat_responses,
        task_reward_positive=flat_answer_positive,
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
        expected = sum(keys[index] == expected_key for index in range(start, stop))
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
                "answer_success_count": sum(
                    flat_answer_positive[start:stop]
                ),
                "forced_route_success_count": expected,
                "wrong_route_success_count": wrong,
                "minimally_executable": expected >= 1,
            }
        )

    problem_results = []
    for record in accepted:
        route_cases = [
            row
            for row in case_results
            if row["source_index"] == record["source_index"]
        ]
        if len(route_cases) != 2:
            raise RuntimeError("E49Y accepted problem lost one forced route")
        problem_results.append(
            {
                "candidate_order": len(problem_results),
                "source_index": record["source_index"],
                "row_id": record["row_id"],
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
    forced_private_path = output / "private/forced_responses.jsonl"
    _write_jsonl(forced_private_path, forced_private_rows)
    payload = {
        "schema": args.result_schema,
        "pass": len(selected) == 10,
        "decision_rule": {
            "natural_samples_per_problem": SAMPLE_COUNT,
            "minimum_validator_positive_for_clustering": 4,
            "minimum_stable_cluster_members": 2,
            "forced_route_minimum_successes_of_8": 1,
            "required_bidirectional_problems": 10,
            "candidate_order": (
                "minimum_cluster_size_desc_combined_size_desc_source_index"
            ),
            "selection": "first_10_passing_candidates",
        },
        "identity": {
            "full_source_manifest_sha256": _sha256(
                FULL / "MATERIALIZATION_MANIFEST.json"
            ),
            "e49t_identity_sha256": _sha256(E49T_IDENTITY),
            "endpoint_record_sha256": _sha256(ENDPOINT_RECORD),
            "protocol_sha256": _sha256(protocol),
            "script_sha256": _sha256(SCRIPT),
            "helper_e49w_script_sha256": _sha256(
                ROOT
                / "ops/math_strategy_calibration/"
                "run_e49w_bottom_up_route_calibration.py"
            ),
            "canonicalizer_sha256": _sha256(
                frozen_source
                / "oat_drgrpo/math_strategy_canonicalizer.py"
            ),
            "math_grader_sha256": _sha256(
                ROOT / "src/oat_drgrpo/math_grader.py"
            ),
            "model_revision": MODEL.name,
            "seed": SEED,
            "partition_mode": PARTITION_MODE,
            "reused_natural_root": (
                str(args.reuse_natural_root.resolve())
                if args.reuse_natural_root is not None
                else None
            ),
        },
        "hard_row_count": len(rows),
        "natural_response_count": len(private_rows),
        "validator_positive_count": sum(
            row["validator_positive_count"] for row in public_sampling
        ),
        "eligible_route_row_count": len(candidates),
        "stable_double_audited_menu_count": len(accepted),
        "forced_case_count": len(case_results),
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
        "natural_responses_sha256": _sha256(private_path),
        "forced_responses_sha256": _sha256(forced_private_path),
        "route_records_sha256": _sha256(route_records_path),
        "cases": case_results,
        "problems": problem_results,
    }
    _write_json(result_path, payload)
    print(result_path)


if __name__ == "__main__":
    main()
