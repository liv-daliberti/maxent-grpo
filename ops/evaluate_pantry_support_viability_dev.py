#!/usr/bin/env python3
"""Development-only Pantry support-action viability with guided decoding."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
import re
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _support_prompt(problem: str) -> str:
    problem = re.sub(
        r"\nReturn only ingredient_id=grams pairs.*$",
        "",
        str(problem),
        flags=re.DOTALL,
    ).rstrip()
    return (
        "<|im_start|>system\n"
        "Choose only the ingredient support for this pantry problem. "
        "A trusted environment will find quantities on exactly that support. "
        "Return only space-separated ingredient IDs and no prose.\n"
        "<|im_end|>\n<|im_start|>user\n"
        + problem
        + "\n\nChoose 2 to 4 distinct ingredient IDs. Output only those IDs "
        "in alphabetical order, separated by one space."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def _support_choices(spec: dict[str, Any]) -> list[str]:
    from oat_drgrpo.pantry_plan import parse_pantry_plan_spec

    parsed = parse_pantry_plan_spec(spec)
    ingredient_ids = sorted(row.ingredient_id for row in parsed.ingredients)
    return [
        " ".join(support)
        for width in range(parsed.min_ingredients, parsed.max_ingredients + 1)
        for support in combinations(ingredient_ids, width)
    ]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-split-root", type=Path, required=True)
    parser.add_argument("--split", default="multi_answer")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--admission-audit", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--prefix-count", type=int, default=16)
    parser.add_argument("--minimum-prefix-success-prompts", type=int, required=True)
    parser.add_argument("--minimum-multimode-prompts", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--assistant-prefix", default="")
    parser.add_argument(
        "--prompt-repair",
        choices=("none", "point_maze_v2", "pantry_plan_v3"),
        default="none",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.domain != "pantry_plan" or args.assistant_prefix or args.prompt_repair != "none":
        raise ValueError("development adapter requires PantryPlan with no legacy prompt repair")
    if not 0 < args.prefix_count <= args.sample_count:
        raise ValueError("prefix count must be in [1, sample count]")
    if args.output.exists():
        raise FileExistsError(f"fresh viability receipt required: {args.output}")

    import vllm
    from datasets import load_from_disk
    from oat_drgrpo.pantry_support_action import validate_pantry_support_action
    from vllm.sampling_params import GuidedDecodingParams

    admission = json.loads(args.admission_audit.read_text(encoding="utf-8"))
    if admission.get("status") != "pass":
        raise ValueError("deterministic admission audit is not passing")
    identity_path = args.data_root / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    dataset_dict = load_from_disk(str(args.data_split_root))
    if set(dataset_dict) != {args.split}:
        raise ValueError("development split names differ from the frozen contract")
    rows = dataset_dict[args.split].to_list()
    if not rows:
        raise ValueError("development split is empty")

    llm = vllm.LLM(
        model=str(args.model.resolve()),
        dtype="bfloat16",
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=0.82,
        swap_space=16.0,
        enable_prefix_caching=True,
    )
    outputs = []
    action_spaces: list[list[str]] = []
    for row_index, row in enumerate(rows):
        spec = json.loads(str(row["answer"]))
        choices = _support_choices(spec)
        action_spaces.append(choices)
        params = vllm.SamplingParams(
            n=int(args.sample_count),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_tokens=int(args.max_tokens),
            seed=int(args.seed) + row_index,
            guided_decoding=GuidedDecodingParams(
                choice=choices,
                backend="xgrammar",
            ),
        )
        outputs.extend(llm.generate([_support_prompt(str(row["problem"]))], params))
    if len(outputs) != len(rows):
        raise RuntimeError("vLLM returned the wrong number of prompt outputs")

    prompt_results: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []
    for row_index, (row, output, action_space) in enumerate(
        zip(rows, outputs, action_spaces)
    ):
        if len(output.outputs) != args.sample_count:
            raise RuntimeError(
                f"prompt {row_index} returned {len(output.outputs)} samples"
            )
        spec = json.loads(str(row["answer"]))
        keys: list[str | None] = []
        validation_cache: dict[str, str | None] = {}
        for sample_index, sample in enumerate(output.outputs, start=1):
            text = str(sample.text).strip()
            if text not in validation_cache:
                validation = validate_pantry_support_action(text, spec)
                validation_cache[text] = (
                    validation.canonical_key if validation is not None else None
                )
            key = validation_cache[text]
            keys.append(key)
            all_attempts.append(
                {
                    "row_index": row_index,
                    "sample_index": sample_index,
                    "text": text,
                    "token_count": len(sample.token_ids),
                    "verified": key is not None,
                    "canonical_key": key,
                }
            )
        prefix_keys = [key for key in keys[: args.prefix_count] if key is not None]
        full_keys = [key for key in keys if key is not None]
        counts = Counter(full_keys)
        prompt_results.append(
            {
                "row_index": row_index,
                "family": str(row.get("answer_mode_family", "")),
                "instance_fingerprint": str(row.get("instance_fingerprint", "")),
                "verified_in_prefix": len(prefix_keys),
                "verified_in_full_sample": len(full_keys),
                "syntactic_action_space_size": len(action_space),
                "distinct_keys_in_prefix": len(set(prefix_keys)),
                "distinct_keys_in_full_sample": len(counts),
                "canonical_key_counts": dict(sorted(counts.items())),
            }
        )

    prefix_success_prompts = sum(
        row["verified_in_prefix"] > 0 for row in prompt_results
    )
    multimode_prompts = sum(
        row["distinct_keys_in_full_sample"] >= 2 for row in prompt_results
    )
    passed = (
        prefix_success_prompts >= args.minimum_prefix_success_prompts
        and multimode_prompts >= args.minimum_multimode_prompts
    )
    if not math.isfinite(float(prefix_success_prompts + multimode_prompts)):
        raise RuntimeError("nonfinite viability counts")
    payload = {
        "schema_version": "pantry-support-action-development-viability-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            "development_interface_signal_pass"
            if passed
            else "development_interface_signal_fail"
        ),
        "domain": args.domain,
        "job_id": int(args.job_id),
        "model": str(args.model.resolve()),
        "model_config_sha256": _sha256_file(args.model / "config.json"),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "protocol_sha256": _sha256_file(args.protocol),
        "admission_audit_sha256": _sha256_file(args.admission_audit),
        "dataset_identity_sha256": _canonical_sha256(identity),
        "data_split_root": str(args.data_split_root.resolve()),
        "data_split_sha256": _canonical_sha256(rows),
        "sampling": {
            "seed": args.seed,
            "sample_count": args.sample_count,
            "prefix_count": args.prefix_count,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "action_adapter": "pantry_support_projected_v1",
            "guided_action_space": "all prompt-local ingredient combinations",
            "prompt_template": "qwen_pantry_support_action",
            "assistant_prefix": args.assistant_prefix,
            "prompt_repair": args.prompt_repair,
        },
        "criteria": {
            "minimum_prefix_success_prompts": (
                args.minimum_prefix_success_prompts
            ),
            "minimum_multimode_prompts": args.minimum_multimode_prompts,
        },
        "summary": {
            "prompt_count": len(rows),
            "prefix_success_prompts": prefix_success_prompts,
            "multimode_prompts": multimode_prompts,
            "verified_completions": sum(
                row["verified_in_full_sample"] for row in prompt_results
            ),
        },
        "prompt_results": prompt_results,
        "attempts": all_attempts,
        "information_boundary": {
            "development_only": True,
            "evaluation_prompts_loaded": False,
            "certified_feasible_supports_in_context": False,
            "reference_allocation_in_context": False,
            "action_space_contains_all_support_combinations": True,
            "quantity_projection_uses_only_prompt_local_constraints": True,
        },
    }
    _atomic_json(args.output, payload)
    print(
        "[pantry-support-dev] "
        f"domain={args.domain} status={payload['status']} "
        f"prefix_success={prefix_success_prompts}/{len(rows)} "
        f"multimode={multimode_prompts}/{len(rows)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
