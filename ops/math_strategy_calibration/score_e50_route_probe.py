#!/usr/bin/env python3
"""Sample and score exact finite-menu route support on ten no-gradient probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
BASE_MODEL = (
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
EXPECTED_ENDPOINT_SHA256 = (
    "964ce3bda1d2cac7dddf2065451fd2b456433478356df9d493e98f9e23c66ac3"
)
SAMPLES = 64
SEED = 500722


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


def _load_frozen_modules() -> tuple[Any, Any]:
    identity = json.loads(E49T_IDENTITY.read_text(encoding="utf-8"))
    frozen_source = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e49t_natural_menu_{identity['source_hash']}"
        / "src"
    )
    import oat_drgrpo

    frozen_package = frozen_source / "oat_drgrpo"
    oat_drgrpo.__path__.insert(0, str(frozen_package))
    canonicalizer = importlib.import_module(
        "oat_drgrpo.math_strategy_canonicalizer"
    )
    menu = importlib.import_module("oat_drgrpo.math_strategy_menu")
    expected = {
        "canonicalizer": (
            "1dc83a2cd9da4cb092dbde747a647b416107846f985683104a98cbe9b022282b"
        ),
        "menu": (
            "95b061310c646c853aa6241b67a3b5b34663f28a7b0e004ed49418d3c7fb1fed"
        ),
    }
    observed = {
        "canonicalizer": _sha256(pathlib.Path(canonicalizer.__file__)),
        "menu": _sha256(pathlib.Path(menu.__file__)),
    }
    if observed != expected:
        raise RuntimeError(
            f"route-probe frozen module identity mismatch: {observed}"
        )
    return canonicalizer, menu


def _endpoint(
    *,
    record_path: pathlib.Path,
    expected_sha256: str,
    expected_node: str,
    expected_port: int,
) -> tuple[str, str]:
    if _sha256(record_path) != expected_sha256:
        raise RuntimeError("route-probe endpoint record changed")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "model": "qwen2.5-72b",
        "node": expected_node,
        "port": expected_port,
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "enforce_eager": True,
        "structured_output_backend": "guidance",
        "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("unexpected route-probe judge endpoint")
    return (
        f"http://{record['node']}:{int(record['port'])}/v1",
        str(record["model"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-root", type=pathlib.Path, default=BASE_MODEL)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--expected-data-schema", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--require-baseline-gate", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--endpoint-record", type=pathlib.Path, default=ENDPOINT_RECORD
    )
    parser.add_argument(
        "--expected-endpoint-sha256", default=EXPECTED_ENDPOINT_SHA256
    )
    parser.add_argument("--expected-endpoint-node", default="node302")
    parser.add_argument("--expected-endpoint-port", type=int, default=8770)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    model_root = args.model_root.resolve()
    out = args.out.resolve()
    endpoint_record = args.endpoint_record.resolve()
    if out.exists():
        raise RuntimeError(f"fresh route-probe output required: {out}")
    manifest_path = data_root / "MATERIALIZATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != args.expected_data_schema
        or manifest.get("row_counts", {}).get("route_probe") != 10
        or manifest.get("multi_support_counts", {}).get("train") != 10
    ):
        raise RuntimeError("route-probe data manifest is not eligible")
    if not (model_root / "config.json").is_file():
        raise RuntimeError(f"route-probe model is incomplete: {model_root}")

    from datasets import load_from_disk
    from oat_drgrpo.math_grader import boxed_reward_fn
    from oat_drgrpo.templates import apply_qwen_math_template
    import vllm

    dataset_dict = load_from_disk(str(data_root / "route_probe"))
    rows = dataset_dict[next(iter(dataset_dict))]
    if len(rows) != 10:
        raise RuntimeError("route-probe split must contain ten rows")
    prompts = [apply_qwen_math_template(str(row["problem"])) for row in rows]
    llm = vllm.LLM(
        model=str(model_root),
        dtype="bfloat16",
        max_model_len=3072,
        gpu_memory_utilization=0.80,
        swap_space=8.0,
        enable_prefix_caching=True,
        seed=SEED,
    )
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            n=SAMPLES,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            seed=SEED,
        ),
    )
    if len(outputs) != 10:
        raise RuntimeError("route-probe generation count mismatch")

    prompt_tokens = []
    prompt_texts = []
    responses = []
    positives = []
    private_rows = []
    for prompt_index, (row, request_output) in enumerate(
        zip(rows, outputs, strict=True)
    ):
        if len(request_output.outputs) != SAMPLES:
            raise RuntimeError("route-probe sample count mismatch")
        for sample_index, sample in enumerate(request_output.outputs):
            text = str(sample.text)
            _, reward = boxed_reward_fn(
                text, str(row["answer"]), fast=False
            )
            positive = float(reward) > 0.0
            prompt_tokens.append(list(request_output.prompt_token_ids))
            prompt_texts.append(str(row["problem"]))
            responses.append(text)
            positives.append(positive)
            private_rows.append(
                {
                    "prompt_index": prompt_index,
                    "row_id": str(row["unique_id"]),
                    "sample_index": sample_index,
                    "answer_positive": positive,
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "response": text,
                }
            )
    private_path = out.parent / "private/route_probe_responses.jsonl"
    _write_jsonl(private_path, private_rows)

    canonicalizer_module, menu_module = _load_frozen_modules()
    endpoint, judge_model = _endpoint(
        record_path=endpoint_record,
        expected_sha256=args.expected_endpoint_sha256,
        expected_node=args.expected_endpoint_node,
        expected_port=args.expected_endpoint_port,
    )
    judge = canonicalizer_module.MathStrategyCanonicalizer(
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
        prompt_token_ids=prompt_tokens,
        prompt_texts=prompt_texts,
        response_texts=responses,
        task_reward_positive=positives,
        active_mask=[True] * len(responses),
        num_samples=SAMPLES,
    )
    prompt_results = []
    for prompt_index, row in enumerate(rows):
        menu = menu_module.parse_strategy_menu(str(row["problem"]))
        if menu is None or len(menu.strategies) != 2:
            raise RuntimeError("route-probe row lost its dual finite menu")
        start = prompt_index * SAMPLES
        stop = start + SAMPLES
        route_counts = {
            strategy.strategy_id: sum(
                key
                == judge._menu_strategy_key(menu, strategy.strategy_id)
                for key in keys[start:stop]
            )
            for strategy in menu.strategies
        }
        accepted = sum(route_counts.values())
        observed = sum(count > 0 for count in route_counts.values())
        natural_support = (
            min(route_counts.values()) >= 2 and accepted >= 8
        )
        prompt_results.append(
            {
                "prompt_index": prompt_index,
                "row_id": str(row["unique_id"]),
                "menu_sha256": menu.sha256,
                "answer_positive_count": sum(positives[start:stop]),
                "accepted_route_count": accepted,
                "route_success_counts": route_counts,
                "observed_strategy_count": observed,
                "normalized_route_coverage": observed / 2,
                "both_routes_observed": observed == 2,
                "natural_support_pass": natural_support,
            }
        )
    dual_full = sum(row["both_routes_observed"] for row in prompt_results)
    natural_support_count = sum(
        row["natural_support_pass"] for row in prompt_results
    )
    all_accepted = all(
        row["accepted_route_count"] >= 1 for row in prompt_results
    )
    baseline_gate = natural_support_count >= 8 and all_accepted
    if args.require_baseline_gate and not baseline_gate:
        status = "fail"
    else:
        status = "pass"
    payload = {
        "schema": "e50_route_probe_result_v1",
        "label": args.label,
        "status": status,
        "baseline_gate": baseline_gate,
        "require_baseline_gate": args.require_baseline_gate,
        "sample_count_per_prompt": SAMPLES,
        "prompt_count": 10,
        "dual_full_coverage_count": dual_full,
        "dual_full_coverage_rate": dual_full / 10,
        "natural_support_prompt_count": natural_support_count,
        "natural_support_prompt_rate": natural_support_count / 10,
        "natural_support_min_per_route": 2,
        "natural_support_min_total_accepted": 8,
        "all_prompts_have_accepted_route": all_accepted,
        "mean_normalized_route_coverage": sum(
            row["normalized_route_coverage"] for row in prompt_results
        )
        / 10,
        "accepted_route_response_count": sum(
            row["accepted_route_count"] for row in prompt_results
        ),
        "answer_positive_response_count": sum(positives),
        "diagnostics": {
            field: float(getattr(diagnostics, field))
            for field in diagnostics.__dataclass_fields__
        },
        "identity": {
            "data_manifest_sha256": _sha256(manifest_path),
            "model_config_sha256": _sha256(model_root / "config.json"),
            "model_root": str(model_root),
            "endpoint_record_sha256": _sha256(endpoint_record),
            "e49t_identity_sha256": _sha256(E49T_IDENTITY),
            "canonicalizer_sha256": _sha256(
                pathlib.Path(canonicalizer_module.__file__)
            ),
            "menu_source_sha256": _sha256(
                pathlib.Path(menu_module.__file__)
            ),
            "script_sha256": _sha256(SCRIPT),
            "seed": SEED,
        },
        "private_responses_sha256": _sha256(private_path),
        "prompts": prompt_results,
    }
    _write_json(out, payload)
    print(out)


if __name__ == "__main__":
    main()
