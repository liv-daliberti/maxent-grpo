#!/usr/bin/env python3
"""Development-only PointMaze grid viability with a legal simple-path mask."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
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
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _shortest_steps(spec: dict[str, Any]) -> int:
    maze = spec["maze_map"]
    start = tuple(spec["reset_cell"])
    goal = tuple(spec["goal_cell"])
    queue = deque([(start, 0)])
    observed = {start}
    for_pop = ((-1, 0), (0, 1), (1, 0), (0, -1))
    while queue:
        cell, distance = queue.popleft()
        if cell == goal:
            return distance
        for delta in for_pop:
            nxt = cell[0] + delta[0], cell[1] + delta[1]
            if nxt not in observed and maze[nxt[0]][nxt[1]] == 0:
                observed.add(nxt)
                queue.append((nxt, distance + 1))
    raise ValueError("development maze has no grid path")


def _prompt(spec: dict[str, Any], minimum: int, maximum: int) -> str:
    rendered: list[str] = []
    for row_index, row in enumerate(spec["maze_map"]):
        cells = []
        for column_index, value in enumerate(row):
            coordinate = [row_index, column_index]
            if coordinate == spec["reset_cell"]:
                cells.append("S")
            elif coordinate == spec["goal_cell"]:
                cells.append("G")
            else:
                cells.append("#" if value else ".")
        rendered.append("".join(cells))
    return (
        "<|im_start|>system\n"
        "Plan a cardinal grid path. Return only uppercase direction tokens "
        "separated by one space and no prose.\n"
        "<|im_end|>\n<|im_start|>user\n"
        "Move from S to G without entering #. Each token moves exactly one "
        "grid cell: N up, E right, S down, W left.\n"
        + "\n".join(rendered)
        + f"\nUse between {minimum} and {maximum} tokens. Never revisit a cell. "
        "End exactly at G."
        "\n<|im_end|>\n<|im_start|>assistant\n"
    )


def _regex(minimum: int, maximum: int) -> str:
    return (
        r"(N|E|S|W)( (N|E|S|W)){"
        + f"{minimum - 1},{maximum - 1}"
        + "}"
    )


def _simple_path_choices(
    spec: dict[str, Any], minimum: int, maximum: int
) -> list[str]:
    """Enumerate every wall-legal non-self-intersecting path, not just solutions."""

    maze = spec["maze_map"]
    start = tuple(spec["reset_cell"])
    directions = ((-1, 0, "N"), (0, 1, "E"), (1, 0, "S"), (0, -1, "W"))
    choices: list[str] = []

    def visit(cell, observed, tokens):
        if len(tokens) >= minimum:
            choices.append(" ".join(tokens))
        if len(tokens) == maximum:
            return
        for delta_row, delta_column, token in directions:
            nxt = cell[0] + delta_row, cell[1] + delta_column
            if maze[nxt[0]][nxt[1]] == 0 and nxt not in observed:
                observed.add(nxt)
                tokens.append(token)
                visit(nxt, observed, tokens)
                tokens.pop()
                observed.remove(nxt)

    visit(start, {start}, [])
    return sorted(choices)


def _goal_choice_count(spec: dict[str, Any], choices: list[str]) -> int:
    current_start = tuple(spec["reset_cell"])
    goal = tuple(spec["goal_cell"])
    deltas = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
    count = 0
    for choice in choices:
        current = current_start
        for token in choice.split():
            delta = deltas[token]
            current = current[0] + delta[0], current[1] + delta[1]
        count += current == goal
    return count


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
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.prefix_count <= args.sample_count:
        raise ValueError("prefix count must be in [1, sample count]")
    if args.output.exists():
        raise FileExistsError(f"fresh development receipt required: {args.output}")

    import vllm
    from datasets import load_from_disk
    from oat_drgrpo.point_maze_grid import (
        adapt_point_maze_spec,
        parse_point_grid_program,
        parse_point_grid_spec,
    )
    from oat_drgrpo.point_maze_grid_process import PointGridVerifierProcess
    from vllm.sampling_params import GuidedDecodingParams

    admission = json.loads(args.admission_audit.read_text(encoding="utf-8"))
    if admission.get("status") != "pass":
        raise ValueError("PointMaze v1 admission audit is not passing")
    identity = json.loads((args.data_root / "identity.json").read_text())
    dataset = load_from_disk(str(args.data_split_root))
    if set(dataset) != {args.split}:
        raise ValueError("development split names differ from the contract")
    rows = dataset[args.split].to_list()
    if len(rows) != 4:
        raise ValueError("Point grid development pilot requires exactly four maps")
    adapted_specs = [adapt_point_maze_spec(json.loads(row["answer"])) for row in rows]
    length_bounds = [
        (_shortest_steps(spec), min(_shortest_steps(spec) + 4, spec["max_actions"]))
        for spec in adapted_specs
    ]
    action_spaces = [
        _simple_path_choices(spec, *bounds)
        for spec, bounds in zip(adapted_specs, length_bounds)
    ]

    llm = vllm.LLM(
        model=str(args.model.resolve()),
        dtype="bfloat16",
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=0.82,
        swap_space=16.0,
        enable_prefix_caching=True,
    )
    outputs = []
    for row_index, (spec, bounds, choices) in enumerate(
        zip(adapted_specs, length_bounds, action_spaces)
    ):
        parameters = vllm.SamplingParams(
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
        outputs.extend(llm.generate([_prompt(spec, *bounds)], parameters))
    if len(outputs) != len(rows):
        raise RuntimeError("vLLM returned the wrong number of outputs")

    prompt_results: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    verifier = PointGridVerifierProcess()
    try:
        for row_index, (row, output, raw_spec, bounds, choices) in enumerate(
            zip(rows, outputs, adapted_specs, length_bounds, action_spaces)
        ):
            if len(output.outputs) != args.sample_count:
                raise RuntimeError("vLLM returned the wrong sample count")
            parsed_spec = parse_point_grid_spec(raw_spec)
            cache: dict[str, str | None] = {}
            keys: list[str | None] = []
            syntactically_valid = 0
            grid_valid = 0
            for sample_index, sample in enumerate(output.outputs, start=1):
                text = str(sample.text).strip()
                syntactically_valid += 1
                if text not in cache:
                    try:
                        parse_point_grid_program(text, parsed_spec)
                        grid_valid += 1
                    except ValueError:
                        cache[text] = None
                    else:
                        validation = verifier.validate(text, raw_spec)
                        cache[text] = (
                            validation.canonical_key
                            if validation is not None
                            else None
                        )
                key = cache[text]
                keys.append(key)
                attempts.append(
                    {
                        "row_index": row_index,
                        "sample_index": sample_index,
                        "text": text,
                        "token_count": len(sample.token_ids),
                        "verified": key is not None,
                        "canonical_key": key,
                    }
                )
            prefix_keys = [key for key in keys[: args.prefix_count] if key]
            full_keys = [key for key in keys if key]
            counts = Counter(full_keys)
            prompt_results.append(
                {
                    "row_index": row_index,
                    "family": str(row.get("answer_mode_family", "")),
                    "legacy_instance_fingerprint": str(
                        row.get("instance_fingerprint", "")
                    ),
                    "grid_spec_sha256": raw_spec["spec_sha256"],
                    "sampled_length_bounds": list(bounds),
                    "legal_simple_path_action_count": len(choices),
                    "unlabeled_goal_ending_action_count": _goal_choice_count(
                        raw_spec, choices
                    ),
                    "syntactically_valid": syntactically_valid,
                    "unique_grid_valid_programs": grid_valid,
                    "verified_in_prefix": len(prefix_keys),
                    "verified_in_full_sample": len(full_keys),
                    "distinct_keys_in_prefix": len(set(prefix_keys)),
                    "distinct_keys_in_full_sample": len(counts),
                    "canonical_key_counts": dict(sorted(counts.items())),
                }
            )
    finally:
        verifier.close()

    prefix_success = sum(row["verified_in_prefix"] > 0 for row in prompt_results)
    multimode = sum(row["distinct_keys_in_full_sample"] >= 2 for row in prompt_results)
    passed = (
        prefix_success >= args.minimum_prefix_success_prompts
        and multimode >= args.minimum_multimode_prompts
    )
    if not math.isfinite(float(prefix_success + multimode)):
        raise RuntimeError("nonfinite development counts")
    payload = {
        "schema_version": "point-grid-action-development-viability-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "decision": (
            "development_interface_signal_pass"
            if passed
            else "development_interface_signal_fail"
        ),
        "domain": "point_maze_grid_development",
        "job_id": int(args.job_id),
        "model": str(args.model.resolve()),
        "model_config_sha256": _sha256_file(args.model / "config.json"),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "protocol_sha256": _sha256_file(args.protocol),
        "admission_audit_sha256": _sha256_file(args.admission_audit),
        "dataset_identity_sha256": _canonical_sha256(identity),
        "data_split_sha256": _canonical_sha256(rows),
        "sampling": {
            "seed": args.seed,
            "sample_count": args.sample_count,
            "prefix_count": args.prefix_count,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "guided_decoding": "all wall-legal simple paths, endpoint unfiltered",
            "prompt_template": "qwen_point_grid_development_v2",
        },
        "criteria": {
            "minimum_prefix_success_prompts": args.minimum_prefix_success_prompts,
            "minimum_multimode_prompts": args.minimum_multimode_prompts,
        },
        "summary": {
            "prompt_count": len(rows),
            "prefix_success_prompts": prefix_success,
            "multimode_prompts": multimode,
            "verified_completions": sum(
                row["verified_in_full_sample"] for row in prompt_results
            ),
        },
        "prompt_results": prompt_results,
        "attempts": attempts,
        "information_boundary": {
            "development_only": True,
            "evaluation_prompts_loaded": False,
            "certified_route_programs_loaded": False,
            "route_catalogue_loaded": False,
            "guided_decoder_filters_by_goal": False,
            "guided_decoder_labels_route_identity": False,
            "guided_decoder_action_mask": "wall legality and no revisits only",
        },
    }
    _atomic_json(args.output, payload)
    print(
        "[point-grid-dev] "
        f"status={payload['status']} prefix_success={prefix_success}/4 "
        f"multimode={multimode}/4 output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
