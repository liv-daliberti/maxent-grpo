#!/usr/bin/env python3
"""Score finite-menu route coverage in E49T sampled evaluation traces.

The ordinary MATH sampled evaluator collapses every correct derivation to the
same answer key.  This sidecar replays only answer-positive responses through
the exact frozen E49T two-pass menu inference contract and reports how many
certified menu routes were actually executed.  It never changes training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import sys
import tempfile
from collections.abc import Iterable
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PREFIX = "e49t_natural_menu_math_toy_05b_3ep_v1"
EXPECTED_ENDPOINT_RECORD_SHA256 = (
    "964ce3bda1d2cac7dddf2065451fd2b456433478356df9d493e98f9e23c66ac3"
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


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest_rows(prefix: str) -> list[dict[str, str]]:
    path = ROOT / "var/artifacts" / f"{prefix}_comparative_jobs.tsv"
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError("empty E49T comparative manifest")
    header = lines[0].split("\t")
    rows = [
        dict(zip(header, line.split("\t"), strict=True))
        for line in lines[1:]
        if line.strip()
    ]
    if {row["arm"] for row in rows} != {
        "grpo",
        "online_canonical_haarnoja",
    }:
        raise RuntimeError("E49T comparative manifest does not bind two arms")
    return rows


def _debug_root(run_stamp: str, job_id: str) -> pathlib.Path:
    # The manifest run stamp omits the xdr model/algorithm prefix.  Resolve the
    # one exact matching run instead of reconstructing a fragile path.
    matches = sorted(
        (ROOT / "var/data").glob(
            "xdr_qwen25_0p5b_instruct_*_" + run_stamp
        )
    )
    if len(matches) != 1:
        # Current launchers use the arm before the run stamp and the arm/seed
        # inside it, so a suffix match is the authoritative fallback.
        matches = sorted(
            path
            for path in (ROOT / "var/data").glob(
                "xdr_qwen25_0p5b_instruct_*"
            )
            if path.name.endswith(run_stamp)
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"could not resolve exactly one run for {run_stamp}: {matches}"
        )
    debug = matches[0] / f"debug_job{job_id}"
    if not debug.is_dir():
        raise RuntimeError(f"missing E49T debug directory: {debug}")
    return debug


def _load_frozen_modules(identity: dict[str, Any]) -> tuple[Any, Any]:
    source_hash = str(identity["source_hash"])
    source = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e49t_natural_menu_{source_hash}"
        / "src"
    )
    if not source.is_dir():
        raise RuntimeError(f"missing frozen E49T source snapshot: {source}")
    sys.path.insert(0, str(source))
    canonicalizer = importlib.import_module(
        "oat_drgrpo.math_strategy_canonicalizer"
    )
    menu = importlib.import_module("oat_drgrpo.math_strategy_menu")
    return canonicalizer, menu


def _endpoint(
    identity: dict[str, Any], record_path: pathlib.Path
) -> tuple[str, str]:
    if _sha256(record_path) != identity["judge_endpoint_record_sha256"]:
        raise RuntimeError("E49T judge endpoint record identity changed")
    if _sha256(record_path) != EXPECTED_ENDPOINT_RECORD_SHA256:
        raise RuntimeError("unexpected E49T endpoint record")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "model": "qwen2.5-72b",
        "node": "node302",
        "port": 8770,
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "enforce_eager": True,
        "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("E49T endpoint configuration changed")
    return (
        f"http://{record['node']}:{int(record['port'])}/v1",
        str(record["model"]),
    )


def _sampled_records(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    sampled = [
        row
        for row in rows
        if row.get("evaluation_kind") == "fixed_seed_sampled_k_neutral"
        and int(row.get("sample_count", 0)) == 8
        and row.get("benchmark") == "math"
    ]
    observed_steps = [int(row["step"]) for row in sampled]
    if len(observed_steps) != len(set(observed_steps)):
        raise RuntimeError("duplicate E49T sampled evaluation step")
    return sorted(sampled, key=lambda row: int(row["step"]))


def _score_record(
    *,
    record: dict[str, Any],
    canonicalizer_module: Any,
    menu_module: Any,
    endpoint: str,
    model: str,
) -> dict[str, Any]:
    prompts = record.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 50:
        raise RuntimeError("E49T route scoring requires all 50 eval prompts")
    judge = canonicalizer_module.MathStrategyCanonicalizer(
        endpoint=endpoint,
        model=model,
        timeout_seconds=600,
        max_workers=4,
        permutation_seeds=(470721, 470722),
        max_item_chars=4000,
        missing_ids_are_ambiguous=True,
        allow_unstructured_menu_inference=True,
    )

    prompt_token_ids: list[list[int]] = []
    prompt_texts: list[str] = []
    responses: list[str] = []
    task_positive: list[bool] = []
    active: list[bool] = []
    menus = []
    for prompt_index, prompt in enumerate(prompts):
        text = str(prompt["prompt"])
        menu = menu_module.parse_strategy_menu(text)
        if menu is None:
            raise RuntimeError("sampled E49T prompt lost its frozen menu")
        row_responses = prompt.get("responses")
        rewards = prompt.get("rewards")
        if (
            not isinstance(row_responses, list)
            or not isinstance(rewards, list)
            or len(row_responses) != 8
            or len(rewards) != 8
        ):
            raise RuntimeError("sampled E49T prompt is not a complete K=8 group")
        menus.append(menu)
        prompt_token_ids.extend([[prompt_index + 1]] * 8)
        prompt_texts.extend([text] * 8)
        responses.extend(str(value) for value in row_responses)
        task_positive.extend(float(value) > 0.0 for value in rewards)
        active.extend([True] * 8)

    keys, diagnostics = judge.canonicalize(
        prompt_token_ids=prompt_token_ids,
        prompt_texts=prompt_texts,
        response_texts=responses,
        task_reward_positive=task_positive,
        active_mask=active,
        num_samples=8,
    )

    prompt_rows = []
    correct_total = 0
    accepted_total = 0
    strategy_sum = 0
    normalized_coverage_sum = 0.0
    dual_prompt_count = 0
    dual_full_coverage = 0
    for prompt_index, menu in enumerate(menus):
        start = prompt_index * 8
        stop = start + 8
        positive_indices = [
            index
            for index in range(start, stop)
            if task_positive[index]
        ]
        accepted_keys = {
            str(keys[index])
            for index in positive_indices
            if keys[index] is not None
        }
        correct_count = len(positive_indices)
        accepted_count = sum(
            keys[index] is not None for index in positive_indices
        )
        support = len(menu.strategies)
        coverage = len(accepted_keys) / support
        correct_total += correct_count
        accepted_total += accepted_count
        strategy_sum += len(accepted_keys)
        normalized_coverage_sum += coverage
        if support >= 2:
            dual_prompt_count += 1
            dual_full_coverage += int(len(accepted_keys) == support)
        prompt_rows.append(
            {
                "prompt_index": prompt_index,
                "menu_sha256": menu.sha256,
                "menu_support": support,
                "correct_response_count": correct_count,
                "accepted_correct_response_count": accepted_count,
                "observed_strategy_count": len(accepted_keys),
                "normalized_strategy_coverage": coverage,
                "strategy_keys": sorted(accepted_keys),
                "response_sha256": [
                    hashlib.sha256(
                        responses[index].encode("utf-8")
                    ).hexdigest()
                    for index in range(start, stop)
                ],
            }
        )
    if dual_prompt_count != 10:
        raise RuntimeError(
            f"E49T expected 10 dual eval prompts, saw {dual_prompt_count}"
        )
    return {
        "step": int(record["step"]),
        "seed": int(record["seed"]),
        "sample_count": 8,
        "prompt_count": 50,
        "correct_response_count": correct_total,
        "accepted_correct_response_count": accepted_total,
        "accepted_correct_fraction": (
            accepted_total / correct_total if correct_total else 0.0
        ),
        "mean_observed_strategy_count_all_prompts": strategy_sum / 50,
        "mean_normalized_strategy_coverage_all_prompts": (
            normalized_coverage_sum / 50
        ),
        "dual_prompt_count": dual_prompt_count,
        "dual_full_coverage_count": dual_full_coverage,
        "dual_full_coverage_rate": dual_full_coverage / dual_prompt_count,
        "diagnostics": {
            field: float(getattr(diagnostics, field))
            for field in diagnostics.__dataclass_fields__
        },
        "prompts": prompt_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--endpoint-record",
        type=pathlib.Path,
        default=(
            ROOT
            / "var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=(
            ROOT / "var/artifacts/e49t_toy_route_coverage_v1/result.json"
        ),
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="'all' or a comma-separated list such as 0,50,100,150",
    )
    parser.add_argument(
        "--arms",
        default="grpo,online_canonical_haarnoja",
        help="comma-separated subset of grpo,online_canonical_haarnoja",
    )
    args = parser.parse_args()

    identity_path = (
        ROOT / "var/artifacts" / f"{args.stamp_prefix}_identity.json"
    )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("schema") != args.stamp_prefix:
        raise RuntimeError("unexpected E49T identity schema")
    canonicalizer_module, menu_module = _load_frozen_modules(identity)
    endpoint, model = _endpoint(identity, args.endpoint_record.resolve())
    requested_steps = None
    if args.steps != "all":
        requested_steps = {
            int(value) for value in args.steps.split(",") if value.strip()
        }
    requested_arms = {
        value.strip() for value in args.arms.split(",") if value.strip()
    }
    valid_arms = {"grpo", "online_canonical_haarnoja"}
    if not requested_arms or not requested_arms <= valid_arms:
        raise RuntimeError(f"invalid requested E49T arms: {requested_arms}")

    arms = {}
    score_cache: dict[str, dict[str, Any]] = {}
    for row in _manifest_rows(args.stamp_prefix):
        if row["arm"] not in requested_arms:
            continue
        debug = _debug_root(row["run_stamp"], row["job_id"])
        sampled = _sampled_records(
            _load_jsonl(debug / "eval_mode_coverage_draws.jsonl")
        )
        if requested_steps is not None:
            sampled = [
                record
                for record in sampled
                if int(record["step"]) in requested_steps
            ]
        evaluations = []
        for record in sampled:
            signature = hashlib.sha256(
                json.dumps(
                    {
                        "step": record.get("step"),
                        "prompts": record.get("prompts"),
                        "sample_count": record.get("sample_count"),
                        "seed": record.get("seed"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if signature not in score_cache:
                score_cache[signature] = _score_record(
                    record=record,
                    canonicalizer_module=canonicalizer_module,
                    menu_module=menu_module,
                    endpoint=endpoint,
                    model=model,
                )
            evaluations.append(score_cache[signature])
        arms[row["arm"]] = {
            "job_id": row["job_id"],
            "run_stamp": row["run_stamp"],
            "evaluations": evaluations,
        }

    payload = {
        "schema": "e49t_toy_route_coverage_v1",
        "stamp_prefix": args.stamp_prefix,
        "identity_sha256": _sha256(identity_path),
        "endpoint_record_sha256": _sha256(args.endpoint_record.resolve()),
        "frozen_source_hash": identity["source_hash"],
        "judge_contract": {
            "model": model,
            "permutation_seeds": [470721, 470722],
            "temperature": 0.0,
            "finite_menu_only": True,
            "unstructured_menu_inference": True,
        },
        "arms": arms,
    }
    _write_json(args.out.resolve(), payload)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
