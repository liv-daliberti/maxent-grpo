#!/usr/bin/env python3
"""Pilot unanimous finite-menu route inference on frozen E49S step-zero outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile

from oat_drgrpo.math_strategy_canonicalizer import MathStrategyCanonicalizer


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49t_unstructured_menu_inference_pilot_20260724.md"
)
SCRIPT = pathlib.Path(__file__).resolve()


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint-record", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("fresh E49T pilot output required")
    if (
        not PROTOCOL.is_file()
        or "FROZEN BEFORE PILOT CALLS"
        not in PROTOCOL.read_text(encoding="utf-8")
    ):
        raise RuntimeError("E49T pilot protocol is not frozen")
    endpoint_record = json.loads(
        args.endpoint_record.read_text(encoding="utf-8")
    )
    expected = {
        "model": "qwen2.5-72b",
        "node": "node105",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "enforce_eager": True,
        "structured_output_backend": "guidance",
        "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
    }
    if any(endpoint_record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("unexpected E49T endpoint")
    records = json.loads(args.eval_results.read_text(encoding="utf-8"))
    cohort = [
        record
        for record in records
        if record.get("scores") == [1]
        and isinstance(record.get("output"), list)
        and len(record["output"]) == 1
    ]
    if len(records) != 50 or not cohort:
        raise RuntimeError("E49T source cohort is malformed or empty")
    prompts = [str(record["problem"]) for record in cohort]
    responses = [str(record["output"][0]) for record in cohort]
    prompt_ids = [[index + 1] for index in range(len(cohort))]
    strict = MathStrategyCanonicalizer(
        endpoint="http://unused",
        allow_unstructured_menu_inference=False,
        transport=lambda _: (_ for _ in ()).throw(
            AssertionError("strict parser unexpectedly called judge")
        ),
    )
    strict_keys, strict_diagnostics = strict.canonicalize(
        prompt_token_ids=prompt_ids,
        prompt_texts=prompts,
        response_texts=responses,
        task_reward_positive=[True] * len(cohort),
        active_mask=[True] * len(cohort),
        num_samples=1,
    )
    endpoint = (
        f"http://{endpoint_record['node']}:{int(endpoint_record['port'])}/v1"
    )
    inferred = MathStrategyCanonicalizer(
        endpoint=endpoint,
        model="qwen2.5-72b",
        timeout_seconds=600,
        max_workers=4,
        max_item_chars=4000,
        allow_unstructured_menu_inference=True,
    )
    inferred_keys, diagnostics = inferred.canonicalize(
        prompt_token_ids=prompt_ids,
        prompt_texts=prompts,
        response_texts=responses,
        task_reward_positive=[True] * len(cohort),
        active_mask=[True] * len(cohort),
        num_samples=1,
    )
    decisions = [
        {
            "cohort_index": index,
            "reference": str(record["reference"]),
            "response_sha256": hashlib.sha256(
                responses[index].encode("utf-8")
            ).hexdigest(),
            "strict_key": strict_keys[index],
            "inferred_key": inferred_keys[index],
        }
        for index, record in enumerate(cohort)
    ]
    accepted = sum(key is not None for key in inferred_keys)
    payload = {
        "schema": "e49t_unstructured_menu_inference_pilot_v1",
        "pass": (
            accepted > 0
            and diagnostics.judge_format_failure_rows == 0
            and diagnostics.inferred_unstructured_rows == accepted
        ),
        "cohort_size": len(cohort),
        "strict_accepted_count": sum(
            key is not None for key in strict_keys
        ),
        "strict_rejected_contract_count": (
            strict_diagnostics.rejected_contract_rows
        ),
        "inferred_accepted_count": accepted,
        "inferred_acceptance_fraction": accepted / len(cohort),
        "inferred_rejected_count": len(cohort) - accepted,
        "diagnostics": diagnostics.__dict__,
        "decisions": decisions,
        "identities": {
            "protocol_sha256": _sha256(PROTOCOL),
            "pilot_script_sha256": _sha256(SCRIPT),
            "canonicalizer_sha256": _sha256(
                ROOT / "src/oat_drgrpo/math_strategy_canonicalizer.py"
            ),
            "eval_results_sha256": _sha256(args.eval_results),
            "endpoint_record_sha256": _sha256(args.endpoint_record),
        },
    }
    _write(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
