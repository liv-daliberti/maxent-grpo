#!/usr/bin/env python3
"""Replay every validated E49Y response through one persistent bank/problem."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
BASE_SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49aa_calibrated_pairwise_observed_route_discovery.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49ab_all_observed_persistent_pairwise_route_discovery_20260726.md"
)
E49AA_RESULT = (
    ROOT
    / "var/artifacts/"
    "e49aa_calibrated_pairwise_observed_route_discovery_v1/result.json"
)
SCHEMA = "e49ab_all_observed_persistent_pairwise_route_discovery_v1"
BATCH_SIZE = 16


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location(
        "e49ab_base_e49aa", BASE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load E49AA pipeline")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
ORIGINAL_MENU_CANDIDATE = BASE._menu_candidate


def _verify_activation() -> None:
    result = json.loads(E49AA_RESULT.read_text(encoding="utf-8"))
    if (
        result.get("schema")
        != "e49aa_calibrated_pairwise_observed_route_discovery_v1"
        or result.get("pass") is True
    ):
        raise RuntimeError("E49AB activates only after final E49AA failure")


def _all_candidates(full_rows: Any) -> list[dict[str, Any]]:
    summaries = BASE._load_jsonl(BASE.SAMPLING_SUMMARY)
    eligible_indices = {
        int(row["source_index"])
        for row in summaries
        if int(row["validator_positive_count"]) >= 4
    }
    if len(eligible_indices) != 74:
        raise RuntimeError("E49AB expected exactly 74 candidate problems")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in BASE._load_jsonl(BASE.NATURAL_RESPONSES):
        if (
            int(row["source_index"]) in eligible_indices
            and row.get("answer_positive") is True
        ):
            text = str(row["response"])
            if len(text) > 4000:
                raise RuntimeError("E49AB positive response exceeds 4000 chars")
            grouped[int(row["source_index"])].append(
                {
                    "sample_id": str(row["response_sha256"])[:20],
                    "sample_index": int(row["sample_index"]),
                    "text": text,
                    "response_sha256": str(row["response_sha256"]),
                }
            )
    candidates = []
    for source_index in sorted(eligible_indices):
        exemplars = sorted(
            grouped[source_index], key=lambda row: row["response_sha256"]
        )
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
    if sum(len(row["exemplars"]) for row in candidates) != 1550:
        raise RuntimeError(
            "E49AB did not retain all 1550 candidate-problem positives"
        )
    return candidates


def _persistent_cluster_candidate(
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
        fields = module.MathStrategyDiagnostics.__dataclass_fields__
        aggregate = {field: 0.0 for field in fields}
        keyed_members: dict[str, list[str]] = defaultdict(list)
        outcome_rows = []
        exemplars = candidate["exemplars"]
        prompt_tokens = [490821, int(candidate["source_index"]) + 1]
        for batch_start in range(0, len(exemplars), BATCH_SIZE):
            batch = exemplars[batch_start : batch_start + BATCH_SIZE]
            keys, diagnostics = judge.canonicalize(
                prompt_token_ids=[prompt_tokens] * len(batch),
                prompt_texts=[candidate["problem"]] * len(batch),
                response_texts=[row["text"] for row in batch],
                task_reward_positive=[True] * len(batch),
                active_mask=[True] * len(batch),
                num_samples=len(batch),
            )
            for field in aggregate:
                aggregate[field] += float(getattr(diagnostics, field))
            for exemplar, key in zip(batch, keys, strict=True):
                outcome_rows.append(
                    {
                        "sample_id": exemplar["sample_id"],
                        "strategy_key": key,
                    }
                )
                if key is not None:
                    keyed_members[str(key)].append(exemplar["sample_id"])
        eligible = [
            (key, members)
            for key, members in keyed_members.items()
            if len(members) >= 2
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
                    or f"calibrated persistent component {key}"
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
            "diagnostics": aggregate,
            "batch_count": (len(exemplars) + BATCH_SIZE - 1) // BATCH_SIZE,
            "response_count": len(exemplars),
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
            "outcomes": outcome_rows,
        }
    except Exception as exc:
        return {**base, "cluster_eligible": False, "error": str(exc)}


def _bounded_menu_candidate(**kwargs: Any) -> dict[str, Any]:
    record = dict(kwargs["cluster_record"])
    record["clusters"] = [
        {**cluster, "member_ids": list(cluster["member_ids"][:3])}
        for cluster in record["clusters"]
    ]
    return ORIGINAL_MENU_CANDIDATE(
        **{**kwargs, "cluster_record": record}
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    known, _ = parser.parse_known_args()
    _verify_activation()

    BASE.SCRIPT = SCRIPT
    BASE.PROTOCOL = PROTOCOL
    BASE._candidates = _all_candidates
    BASE._cluster_candidate = _persistent_cluster_candidate
    BASE._menu_candidate = _bounded_menu_candidate
    BASE.main()

    result_path = known.output_root.resolve() / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["schema"] = SCHEMA
    result.setdefault("identity", {})["e49aa_result_sha256"] = BASE._sha256(
        E49AA_RESULT
    )
    result["identity"]["base_e49aa_script_sha256"] = BASE._sha256(BASE_SCRIPT)
    result["all_observed_response_count"] = 1550
    result["batch_size"] = BATCH_SIZE
    BASE._write_json(result_path, result)


if __name__ == "__main__":
    main()
