#!/usr/bin/env python3
"""Run one frozen ConstructiveCode v6 paired-smoke or Stage-B arm."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.canonical_replay import (  # noqa: E402
    CanonicalReplayInverseController,
    CanonicalReplayLikelihoodController,
    canonical_replay_split_mass_balance_loss,
)
from oat_drgrpo.interactive_episode_objective import (  # noqa: E402
    add_verified_advantage_outside_centering,
    drgrpo_task_advantages,
)
from oat_drgrpo.online_canonical_bank import OnlineCanonicalBank  # noqa: E402
from oat_drgrpo.semantic_shannon import SemanticShannonTracker  # noqa: E402


CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
ARMS = (CONTROL, TREATMENT)
PAIRED_MODE = "paired_smoke"
STAGE_B_MODE = "stage_b"
MODES = (PAIRED_MODE, STAGE_B_MODE)
DEVELOPMENT_PROBLEMS = ("359_B", "988_A", "1399_D")
TRAIN_PROBLEMS = ("327_B", "659_C", "1283_C", "1102_B")
EVALUATION_PROBLEMS = ("361_B", "1294_C", "149_C")
DEVELOPMENT_FAMILIES = (
    "ordered_sequence",
    "unordered_set",
    "unordered_partition",
)
TRAIN_FAMILIES = (
    "ordered_sequence",
    "unordered_set",
    "assignment",
    "unordered_partition",
)
SAMPLES = 16
RESPONSE_TOKENS = 1024
REPLAY_CAPACITY = 16
OVERLAY_SUITE = "codecontests_o_corner_cases_v2"
PLUS_SUITE = "codecontests_plus_5x_v2"
MODEL_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
SYSTEM_MESSAGE = (
    "Write a complete Python 3 program that solves the problem. Return only "
    "the program source, without Markdown fences or explanation."
)
PADDING_STATEMENT = (
    "Compute-only padding row. Return a complete Python 3 program that exits."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def tree_sha256(root: Path) -> str:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append((path.relative_to(root).as_posix(), sha256_file(path)))
    if not records:
        raise ValueError(f"empty tree cannot be hashed: {root}")
    return canonical_sha256(records)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def prompt(statement: str) -> str:
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("ConstructiveCode public statement must be nonempty")
    return (
        "<|im_start|>system\n"
        + SYSTEM_MESSAGE
        + "<|im_end|>\n<|im_start|>user\n"
        + statement
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def strip_exact_surrounding_fence(text: str) -> tuple[str, bool]:
    for opening in ("```python\n", "```python3\n", "```\n"):
        if text.startswith(opening) and text.endswith("\n```"):
            return text[len(opening) : -4], True
    return text, False


def hard_replay_violations(replay: Mapping[str, Any]) -> list[str]:
    violations = []
    execution = replay.get("execution")
    if not isinstance(execution, Mapping):
        return ["missing execution record"]
    timings = execution.get("candidate_invocation_wall_seconds")
    checker_seconds = execution.get("checker_wall_seconds")
    if (
        not isinstance(timings, list)
        or not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
            for value in timings
        )
        or not isinstance(checker_seconds, (int, float))
        or not math.isfinite(checker_seconds)
        or checker_seconds < 0
    ):
        violations.append("nonfinite execution latency")
    if replay.get("released_checker_accepted") != replay.get("wrapper_accepted"):
        violations.append("checker-wrapper disagreement")
    failure = execution.get("first_failure")
    if isinstance(failure, Mapping):
        stage = failure.get("stage")
        if stage == "candidate" and any(
            bool(failure.get(key))
            for key in ("timed_out", "output_limited", "sandbox_violation")
        ):
            violations.append("candidate execution-bound violation")
        elif stage == "released_checker" and bool(failure.get("timed_out")):
            violations.append("released checker timeout")
        elif stage == "task_adapter":
            violations.append("task canonicalizer rejected checker acceptance")
    if replay.get("released_checker_accepted") is True and replay.get(
        "behavior_key"
    ) is None:
        violations.append("accepted candidate lacks behavior key")
    return violations


def _replay_modules():
    import evaluate_constructive_code_v6_coder_viability as v6

    replay_v5, base, materialize = v6._replay_modules()
    return v6, replay_v5, base, materialize


def _selected_suites(
    gate: Mapping[str, Any], problem_keys: Mapping[str, str]
) -> dict[str, str]:
    if (
        gate.get("status") != "pass"
        or gate.get("expected_replay_count") != 960
        or gate.get("observed_replay_count") != 960
        or gate.get("violations") not in ([], None)
        or gate.get("checker_equivalence_violations") not in ([], None)
    ):
        raise ValueError("ConstructiveCode v6 gate is not an exact pass")
    task_rows = gate.get("task_results")
    suite_rows = gate.get("suite_results")
    if not isinstance(task_rows, list) or len(task_rows) != 10:
        raise ValueError("ConstructiveCode v6 gate lacks ten task results")
    if not isinstance(suite_rows, list):
        raise ValueError("ConstructiveCode v6 gate lacks suite results")
    task_status = {
        str(row.get("problem_key")): row.get("status")
        for row in task_rows
        if isinstance(row, Mapping)
    }
    suite_status = {
        (str(row.get("problem_key")), str(row.get("suite_id"))): row.get(
            "status"
        )
        for row in suite_rows
        if isinstance(row, Mapping)
    }
    selected = {}
    for problem_id, problem_key in problem_keys.items():
        if task_status.get(problem_key) != "pass":
            raise ValueError(f"v6 task did not pass: {problem_id}")
        if suite_status.get((problem_key, OVERLAY_SUITE)) == "pass":
            selected[problem_id] = OVERLAY_SUITE
        elif suite_status.get((problem_key, PLUS_SUITE)) == "pass":
            selected[problem_id] = PLUS_SUITE
        else:
            raise ValueError(f"v6 task has no admitted suite: {problem_id}")
    return selected


def load_frozen_tasks(
    *,
    problem_ids: Sequence[str],
    expected_split: Mapping[str, str],
    slate_root: Path,
    v1_root: Path,
    build_root: Path,
    gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], Any]:
    _v5, _replay_v5, base, materialize = _replay_modules()
    source_manifest = base._validate_source_manifest(slate_root)
    base._validate_v1_logical_root(source_manifest)
    if base.sha256_file(base.TESTLIB_ROOT / "testlib.h") != base.TESTLIB_SHA256:
        raise ValueError("pinned testlib.h hash drift")
    summaries = {
        str(row["source_problem_id"]): row for row in source_manifest["tasks"]
    }
    if set(summaries) != set(materialize.V5_TASKS):
        raise ValueError("ConstructiveCode v6 task manifest drift")
    if tuple(problem_ids) != tuple(dict.fromkeys(problem_ids)):
        raise ValueError("ConstructiveCode task request contains duplicates")
    if any(problem_id not in summaries for problem_id in problem_ids):
        raise ValueError("ConstructiveCode task request left the v6 slate")

    records = {}
    problem_keys = {}
    for problem_id in problem_ids:
        task_dir = slate_root / str(summaries[problem_id]["relative_path"])
        record = base._load_json(task_dir / "task.json")
        records[problem_id] = (task_dir, record)
        problem_keys[problem_id] = str(record["problem_key"])
    selected = _selected_suites(gate, problem_keys)

    registered = set(base.registered_task_adapters())
    tasks = {}
    public_rows = {}
    builds = []
    for problem_id in problem_ids:
        task_dir, record = records[problem_id]
        summary = summaries[problem_id]
        family, adapter = materialize.V5_TASKS[problem_id]
        unsigned = {
            key: value for key, value in record.items() if key != "task_record_sha256"
        }
        stored_sha = record.get("task_record_sha256")
        if (
            record.get("schema_version") != base.TASK_RECORD_SCHEMA
            or record.get("source_problem_id") != problem_id
            or record.get("witness_family") != family
            or record.get("task_adapter") != adapter
            or stored_sha != base._canonical_sha256(unsigned)
            or stored_sha != summary.get("task_record_sha256")
            or (problem_id, adapter) not in registered
            or materialize.V5_SPLIT_ASSIGNMENT.get(problem_id)
            != expected_split.get(problem_id)
        ):
            raise ValueError(f"v6 task identity or split drift: {problem_id}")
        language = record.get("language_contract")
        if not isinstance(language, Mapping) or (
            language.get("accepted_labels") != sorted(base.PYTHON3_LABELS)
            or language.get("runtime") != "Python 3.10.20"
        ):
            raise ValueError(f"v6 language contract drift: {problem_id}")
        v1_hashes = base._v1_hashes(
            v1_root / str(summary["relative_path"]), problem_id
        )
        if language.get(base.V1_LEDGER_COUNT_FIELD) != len(v1_hashes):
            raise ValueError(f"v6 logical v1 ledger drift: {problem_id}")
        replay_path = task_dir / "py3_replays.jsonl"
        if base._sha256_path(replay_path) != record.get("replays", {}).get(
            "jsonl_sha256"
        ):
            raise ValueError(f"v6 replay ledger drift: {problem_id}")
        base._validate_submission_records(base._load_jsonl(replay_path), v1_hashes)
        checker_sha = str(record.get("checker_sha256") or "")
        checker_binary = build_root / problem_id.lower() / "checker"
        build = base._compile_checker(
            task_dir / "checker.cpp", checker_binary, checker_sha
        )
        build["source_problem_id"] = problem_id
        builds.append(build)
        raw_limits = record.get("limits")
        if not isinstance(raw_limits, Mapping):
            raise ValueError(f"v6 limits missing: {problem_id}")
        time_ms = raw_limits.get("time_milliseconds")
        memory_mb = raw_limits.get("memory_megabytes")
        if (
            isinstance(time_ms, bool)
            or not isinstance(time_ms, int)
            or time_ms <= 0
            or isinstance(memory_mb, bool)
            or not isinstance(memory_mb, int)
            or memory_mb <= 0
        ):
            raise ValueError(f"v6 limits malformed: {problem_id}")
        cpu_seconds = max(
            1, math.ceil(time_ms * base.PYTHON_CPU_MULTIPLIER / 1000)
        )
        limits = base.SandboxLimits(
            cpu_seconds=cpu_seconds,
            wall_seconds=float(cpu_seconds + 2),
            memory_bytes=memory_mb * 1024 * 1024,
            output_bytes=base.OUTPUT_LIMIT_BYTES,
            file_count=32,
            source_bytes=256 * 1024,
        )
        suites = record.get("suites")
        suite_id = selected[problem_id]
        if not isinstance(suites, Mapping) or set(suites) != base.REQUIRED_SUITE_IDS:
            raise ValueError(f"v6 suite metadata drift: {problem_id}")
        suite = suites[suite_id]
        tasks[problem_id] = base.Task(
            problem_id=problem_id,
            problem_key=str(record["problem_key"]),
            adapter_id=adapter,
            witness_family=family,
            suite_id=suite_id,
            suite_sha256=str(suite["suite_sha256"]),
            checker_sha256=checker_sha,
            checker_binary=checker_binary,
            tests=base._load_tests(task_dir / base.SUITE_FILES[suite_id], suite),
            submissions=(),
            limits=limits,
        )
        statement = record.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"v6 public statement missing: {problem_id}")
        public_rows[problem_id] = {
            "source_problem_id": problem_id,
            "problem_key": str(record["problem_key"]),
            "witness_family": family,
            "statement": statement,
            "statement_sha256": hashlib.sha256(statement.encode()).hexdigest(),
            "suite_id": suite_id,
            "suite_sha256": str(suite["suite_sha256"]),
            "checker_sha256": checker_sha,
        }
    if tuple(tasks) != tuple(problem_ids) or tuple(public_rows) != tuple(problem_ids):
        raise RuntimeError("ConstructiveCode v6 task order drift")
    return tasks, public_rows, builds, source_manifest


def request_namespace(
    *, mode: str, seed: int, phase: str, update: int, task_index: int, draw: int
) -> int:
    if mode not in MODES or phase not in {"train", "eval_greedy", "eval_sample"}:
        raise ValueError("invalid ConstructiveCode request namespace")
    mode_offset = 0 if mode == PAIRED_MODE else 2_000_000_000
    phase_offset = {
        "train": 0,
        "eval_greedy": 500_000_000,
        "eval_sample": 1_000_000_000,
    }[phase]
    value = (
        int(seed)
        + mode_offset
        + phase_offset
        + int(update) * 1_000_000
        + int(task_index) * 10_000
        + int(draw) * 100
    )
    if value < 0 or value >= 2**63:
        raise ValueError("ConstructiveCode request namespace overflow")
    return value


def _completion_ids(sequence: Sequence[int], prompt_width: int, eos_id: int) -> tuple[int, ...]:
    generated = [int(value) for value in sequence[prompt_width:]]
    if eos_id in generated:
        generated = generated[: generated.index(eos_id) + 1]
    if not generated or len(generated) > RESPONSE_TOKENS:
        raise RuntimeError("ConstructiveCode generation left its token rectangle")
    return tuple(generated)


def generate_candidates(
    *,
    model: Any,
    tokenizer: Any,
    statement: str,
    count: int,
    namespace: int,
    do_sample: bool,
    max_model_len: int,
) -> list[dict[str, Any]]:
    import torch

    if count <= 0:
        raise ValueError("ConstructiveCode generation count must be positive")
    rendered = prompt(statement)
    prompt_ids = tuple(tokenizer.encode(rendered, add_special_tokens=False))
    if not prompt_ids or len(prompt_ids) + RESPONSE_TOKENS > max_model_len:
        raise ValueError("ConstructiveCode prompt left the frozen context bound")
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        [rendered] * count,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].cuda()
    attention = encoded["attention_mask"].cuda()
    prompt_width = int(input_ids.shape[1])
    if prompt_width != len(prompt_ids):
        raise RuntimeError("identical ConstructiveCode prompts padded unexpectedly")
    torch.manual_seed(int(namespace))
    torch.cuda.manual_seed_all(int(namespace))
    invalid_tokens = list(range(len(tokenizer), int(model.config.vocab_size)))
    was_cache = bool(model.config.use_cache)
    model.config.use_cache = True
    model.eval()
    try:
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=do_sample,
                temperature=1.0 if do_sample else None,
                top_p=1.0 if do_sample else None,
                top_k=0 if do_sample else None,
                max_new_tokens=RESPONSE_TOKENS,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=int(tokenizer.eos_token_id),
                suppress_tokens=invalid_tokens,
                use_cache=True,
            )
    finally:
        model.config.use_cache = was_cache
    if int(sequences.shape[0]) != count:
        raise RuntimeError("ConstructiveCode generation returned wrong row count")
    candidates = []
    for slot, sequence in enumerate(sequences.detach().cpu().tolist()):
        response_ids = _completion_ids(
            sequence, prompt_width, int(tokenizer.eos_token_id)
        )
        emitted = tokenizer.decode(
            list(response_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        code, stripped = strip_exact_surrounding_fence(emitted)
        candidates.append(
            {
                "slot": slot,
                "request_namespace": int(namespace),
                "request_stream_offset": slot,
                "prompt_token_ids": prompt_ids,
                "response_token_ids": response_ids,
                "emitted_text_sha256": hashlib.sha256(emitted.encode()).hexdigest(),
                "executed_source_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "fence_stripped": stripped,
                "token_count": len(response_ids),
                "finish_reason": (
                    "stop" if response_ids[-1] == tokenizer.eos_token_id else "length"
                ),
                "code": code,
            }
        )
    return candidates


def execute_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    task: Any,
    base: Any,
    launcher: Path,
    runtime_root: Path,
    scratch_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    exceptions: dict[int, str] = {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for candidate in candidates:
            slot = int(candidate["slot"])
            submission = base.Submission(
                str(candidate["code"]),
                "model_sample",
                str(candidate["executed_source_sha256"]),
            )
            futures[
                executor.submit(
                    base._replay_submission,
                    task=task,
                    submission=submission,
                    launcher=launcher,
                    runtime_root=runtime_root,
                    scratch_root=scratch_root,
                )
            ] = slot
        for future in as_completed(futures):
            slot = futures[future]
            try:
                completed[slot] = future.result()
            except BaseException as error:
                exceptions[slot] = f"{type(error).__name__}: {error}"
    wall_seconds = time.monotonic() - started
    attempts = []
    for candidate in candidates:
        slot = int(candidate["slot"])
        replay = completed.get(slot)
        violations = (
            ["worker exception: " + exceptions.get(slot, "missing result")]
            if replay is None
            else hard_replay_violations(replay)
        )
        if violations:
            raise RuntimeError(
                f"ConstructiveCode hard worker violation at slot {slot}: {violations}"
            )
        assert replay is not None
        accepted = bool(
            replay.get("released_checker_accepted") is True
            and replay.get("wrapper_accepted") is True
            and isinstance(replay.get("behavior_key"), str)
        )
        attempts.append(
            {
                **{
                    key: candidate[key]
                    for key in (
                        "slot",
                        "request_namespace",
                        "request_stream_offset",
                        "emitted_text_sha256",
                        "executed_source_sha256",
                        "fence_stripped",
                        "token_count",
                        "finish_reason",
                    )
                },
                "code": str(candidate["code"]),
                "terminal_worker_record": True,
                "accepted": accepted,
                "canonical_key": replay.get("behavior_key"),
                "replay": replay,
                "batch_execution_wall_seconds": wall_seconds,
            }
        )
    return attempts


def _fixed_rectangle(
    *,
    prompt_token_ids: Sequence[int],
    response_rows: Sequence[Sequence[int]],
    pad_token_id: int,
    device: Any,
) -> tuple[Any, Any, Any, int]:
    import torch

    prompt_ids = tuple(int(value) for value in prompt_token_ids)
    if not prompt_ids or any(value < 0 for value in prompt_ids):
        raise ValueError("ConstructiveCode fixed rectangle has invalid prompt")
    if not response_rows:
        raise ValueError("ConstructiveCode fixed rectangle has no rows")
    prompt_length = len(prompt_ids)
    width = prompt_length + RESPONSE_TOKENS
    input_ids = torch.full(
        (len(response_rows), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros_like(input_ids)
    response_mask = torch.zeros(
        (len(response_rows), RESPONSE_TOKENS),
        dtype=torch.bool,
        device=device,
    )
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    for row_index, raw_response in enumerate(response_rows):
        response = tuple(int(value) for value in raw_response)
        if not response or len(response) > RESPONSE_TOKENS or any(
            value < 0 for value in response
        ):
            raise ValueError("ConstructiveCode response left fixed rectangle")
        input_ids[row_index, :prompt_length] = prompt_tensor
        input_ids[
            row_index, prompt_length : prompt_length + len(response)
        ] = torch.tensor(response, dtype=torch.long, device=device)
        attention[row_index, : prompt_length + len(response)] = 1
        response_mask[row_index, : len(response)] = True
    return input_ids, attention, response_mask, prompt_length


def _selected_logprobs(
    *, model: Any, input_ids: Any, attention: Any, prompt_length: int, valid_vocab: int
) -> Any:
    import torch

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention,
        use_cache=False,
        logits_to_keep=RESPONSE_TOKENS + 1,
    )
    logits = outputs.logits[:, :RESPONSE_TOKENS, :].float()
    if int(logits.shape[1]) != RESPONSE_TOKENS:
        raise RuntimeError("Qwen did not return the fixed response logit rectangle")
    if valid_vocab <= 0 or valid_vocab > int(logits.shape[-1]):
        raise ValueError("ConstructiveCode valid vocabulary bound is invalid")
    if valid_vocab < int(logits.shape[-1]):
        logits[..., valid_vocab:] = -torch.inf
    labels = input_ids[:, prompt_length : prompt_length + RESPONSE_TOKENS]
    return torch.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)


def detached_policy_scores(
    *,
    model: Any,
    tokenizer: Any,
    prompt_token_ids: Sequence[int],
    response_rows: Sequence[Sequence[int]],
) -> tuple[Any, Any]:
    import torch

    rows = []
    masks = []
    model.eval()
    with torch.no_grad():
        for response in response_rows:
            input_ids, attention, mask, prompt_length = _fixed_rectangle(
                prompt_token_ids=prompt_token_ids,
                response_rows=[response],
                pad_token_id=int(tokenizer.pad_token_id),
                device="cuda",
            )
            logps = _selected_logprobs(
                model=model,
                input_ids=input_ids,
                attention=attention,
                prompt_length=prompt_length,
                valid_vocab=len(tokenizer),
            )
            rows.append(logps.detach().cpu())
            masks.append(mask.detach().cpu())
    return torch.cat(rows, dim=0), torch.cat(masks, dim=0)


def backward_on_policy(
    *,
    model: Any,
    tokenizer: Any,
    prompt_token_ids: Sequence[int],
    response_rows: Sequence[Sequence[int]],
    behavior_logps: Any,
    response_masks: Any,
    advantages: Any,
) -> dict[str, float]:
    import torch

    if len(response_rows) != SAMPLES:
        raise ValueError("ConstructiveCode policy traversal requires 16 rows")
    total_loss = 0.0
    max_abs_diff = 0.0
    model.train()
    for index, response in enumerate(response_rows):
        input_ids, attention, mask, prompt_length = _fixed_rectangle(
            prompt_token_ids=prompt_token_ids,
            response_rows=[response],
            pad_token_id=int(tokenizer.pad_token_id),
            device="cuda",
        )
        live = _selected_logprobs(
            model=model,
            input_ids=input_ids,
            attention=attention,
            prompt_length=prompt_length,
            valid_vocab=len(tokenizer),
        )
        old = behavior_logps[index : index + 1].to(live.device)
        active = mask.to(live.dtype)
        if not torch.equal(mask.cpu(), response_masks[index : index + 1]):
            raise RuntimeError("ConstructiveCode policy response mask drift")
        logprob_difference = live - old
        ratio = torch.exp(logprob_difference)
        advantage = advantages[index].to(live.device, dtype=live.dtype)
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
        token_loss = -torch.minimum(unclipped, clipped) * active
        loss = token_loss.sum() / float(SAMPLES * RESPONSE_TOKENS)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite ConstructiveCode policy loss")
        loss.backward()
        total_loss += float(loss.detach().item())
        max_abs_diff = max(
            max_abs_diff,
            float((logprob_difference.detach().abs() * active).max().item()),
        )
    return {
        "policy_loss": total_loss,
        "behavior_live_logprob_abs_diff_max": max_abs_diff,
        "policy_score_rows": float(SAMPLES),
        "policy_response_token_slots": float(SAMPLES * RESPONSE_TOKENS),
        "policy_score_passes": 2.0,
    }


def replay_backward(
    *,
    model: Any,
    tokenizer: Any,
    replay_groups: Sequence[Any],
    padding_prompt_ids: Sequence[int],
    padding_response_ids: Sequence[int],
    compute_only: bool,
    balance_controller: CanonicalReplayInverseController,
    mass_controller: CanonicalReplayLikelihoodController,
) -> dict[str, float]:
    import torch

    if len(replay_groups) > 1:
        raise ValueError("ConstructiveCode replay schedule permits one prompt")
    if replay_groups:
        group = replay_groups[0]
        active_responses = list(group.response_token_ids)
        if not 1 <= len(active_responses) <= REPLAY_CAPACITY:
            raise RuntimeError("ConstructiveCode replay bank left capacity")
        replay_prompt = tuple(group.prompt_token_ids)
    else:
        active_responses = []
        replay_prompt = tuple(padding_prompt_ids)
    response_rows = [*active_responses]
    response_rows.extend(
        [tuple(padding_response_ids)] * (REPLAY_CAPACITY - len(response_rows))
    )
    detached, masks = detached_policy_scores(
        model=model,
        tokenizer=tokenizer,
        prompt_token_ids=replay_prompt,
        response_rows=response_rows,
    )
    active_count = len(active_responses)
    balance_alpha = float(balance_controller.current_alpha)
    mass_alpha = float(mass_controller.current_alpha)
    if active_count:
        token_counts = masks[:active_count].sum(dim=1).to(detached.dtype)
        mode_scores = (
            detached[:active_count] * masks[:active_count].to(detached.dtype)
        ).sum(dim=1) / token_counts
        split = canonical_replay_split_mass_balance_loss(
            mode_scores.double(), [active_count]
        )
        raw = (
            split.mass_score_gradients * mass_alpha
            + split.balance_score_gradients * balance_alpha
        ).float()
        mass_gradient_l2 = float(
            torch.linalg.vector_norm(split.mass_score_gradients).item()
        )
        balance_gradient_l2 = float(
            torch.linalg.vector_norm(split.balance_score_gradients).item()
        )
        raw_weighted_loss = float(
            (
                split.mass_loss * mass_alpha
                + split.balance_loss * balance_alpha
            ).item()
            * (SAMPLES - 1)
            / SAMPLES
            / SAMPLES
        )
        actuator_loss = float(split.mass_loss.item())
        balance_loss = float(split.balance_loss.item())
        normalized_entropy = float(split.normalized_entropy.item())
        balance_eligible = int(split.balance_eligible_groups)
        actuator_groups = int(split.actuator_groups)
    else:
        raw = torch.zeros(0, dtype=torch.float32)
        mass_gradient_l2 = balance_gradient_l2 = 0.0
        raw_weighted_loss = actuator_loss = balance_loss = 0.0
        normalized_entropy = 1.0
        balance_eligible = actuator_groups = 0
    raw_gradient = torch.zeros(REPLAY_CAPACITY, dtype=torch.float32)
    if active_count:
        raw_gradient[:active_count] = raw
    applied_gradient = (
        torch.zeros_like(raw_gradient) if compute_only else raw_gradient
    )
    backward_scale = (SAMPLES - 1) / SAMPLES / SAMPLES
    backward_gradient = applied_gradient * backward_scale
    # E58 scores both detached and live replay rows under temporary eval mode;
    # gradients remain enabled for the live pass.
    model.eval()
    live_score_sum = 0.0
    for index, response in enumerate(response_rows):
        input_ids, attention, mask, prompt_length = _fixed_rectangle(
            prompt_token_ids=replay_prompt,
            response_rows=[response],
            pad_token_id=int(tokenizer.pad_token_id),
            device="cuda",
        )
        live = _selected_logprobs(
            model=model,
            input_ids=input_ids,
            attention=attention,
            prompt_length=prompt_length,
            valid_vocab=len(tokenizer),
        )
        live_score = (live * mask.to(live.dtype)).sum() / mask.sum()
        loss = live_score * backward_gradient[index].to(live.device)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite ConstructiveCode replay loss")
        loss.backward()
        live_score_sum += float(live_score.detach().item())
    if balance_eligible:
        balance_diagnostics = balance_controller.observe(normalized_entropy)
        balance_diagnostics["canonical_replay_observation_skipped"] = 0.0
        balance_diagnostics["canonical_replay_global_eligibility_weight"] = float(
            balance_eligible
        )
    else:
        balance_diagnostics = balance_controller.idle_diagnostics()
    if actuator_groups:
        mass_diagnostics = mass_controller.observe(actuator_loss)
        mass_diagnostics["canonical_replay_mass_observation_skipped"] = 0.0
        mass_diagnostics[
            "canonical_replay_mass_global_eligibility_weight"
        ] = float(actuator_groups)
    else:
        mass_diagnostics = mass_controller.idle_diagnostics()
    return {
        "canonical_replay_available_groups": float(bool(active_count)),
        "canonical_replay_available_modes": float(active_count),
        "canonical_replay_capacity": float(REPLAY_CAPACITY),
        "canonical_replay_global_scheduler_active": 1.0,
        "canonical_replay_global_groups_per_step": 1.0,
        "canonical_replay_schedule_used_global": float(bool(active_count)),
        "canonical_replay_balance_loss": balance_loss,
        "canonical_replay_actuator_loss": actuator_loss,
        "canonical_replay_raw_weighted_loss": raw_weighted_loss,
        "canonical_replay_weighted_loss": (
            0.0 if compute_only else raw_weighted_loss
        ),
        "canonical_replay_compute_only": float(compute_only),
        "canonical_replay_score_passes": 2.0,
        "canonical_replay_score_rows": float(REPLAY_CAPACITY),
        "canonical_replay_live_score_sum": live_score_sum,
        "canonical_replay_normalized_model_entropy": normalized_entropy,
        "canonical_replay_alpha_used": balance_alpha,
        "canonical_replay_mass_alpha_used": mass_alpha,
        "canonical_replay_eligible_groups": float(balance_eligible),
        "canonical_replay_actuator_groups": float(actuator_groups),
        "canonical_replay_actuator_modes": float(active_count),
        "canonical_replay_reward_estimator_scale": (SAMPLES - 1) / SAMPLES,
        "canonical_replay_objective_scale": 1.0 / SAMPLES,
        "canonical_replay_mass_score_gradient_l2": float(
            mass_gradient_l2
        ),
        "canonical_replay_balance_score_gradient_l2": balance_gradient_l2,
        "canonical_replay_applied_score_gradient_l2": float(
            torch.linalg.vector_norm(applied_gradient).item()
        ),
        "canonical_replay_raw_score_gradient_l2": float(
            torch.linalg.vector_norm(raw_gradient).item()
        ),
        "canonical_replay_score_gradient_sum": float(raw_gradient.sum().item()),
        "canonical_replay_applied_score_gradient_sum": float(
            applied_gradient.sum().item() / SAMPLES
        ),
        **balance_diagnostics,
        **mass_diagnostics,
    }


def _tensor_rms(values: Any) -> float:
    import torch

    return float(torch.sqrt(torch.mean(values.float().square())).item())


def evaluation_coordinate(
    *,
    model: Any,
    tokenizer: Any,
    tasks: Mapping[str, Any],
    public_rows: Mapping[str, Mapping[str, Any]],
    base: Any,
    launcher: Path,
    runtime_root: Path,
    scratch_root: Path,
    execution_workers: int,
    seed: int,
    update: int,
    max_model_len: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_summaries = []
    ledger_tasks = []
    for task_index, problem_id in enumerate(EVALUATION_PROBLEMS):
        task = tasks[problem_id]
        row = public_rows[problem_id]
        greedy = generate_candidates(
            model=model,
            tokenizer=tokenizer,
            statement=str(row["statement"]),
            count=1,
            namespace=request_namespace(
                mode=STAGE_B_MODE,
                seed=seed,
                phase="eval_greedy",
                update=update,
                task_index=task_index,
                draw=0,
            ),
            do_sample=False,
            max_model_len=max_model_len,
        )
        greedy_attempt = execute_candidates(
            candidates=greedy,
            task=task,
            base=base,
            launcher=launcher,
            runtime_root=runtime_root,
            scratch_root=scratch_root,
            workers=execution_workers,
        )[0]
        draws = []
        ledger_draws = []
        for draw in range(4):
            candidates = generate_candidates(
                model=model,
                tokenizer=tokenizer,
                statement=str(row["statement"]),
                count=8,
                namespace=request_namespace(
                    mode=STAGE_B_MODE,
                    seed=seed,
                    phase="eval_sample",
                    update=update,
                    task_index=task_index,
                    draw=draw,
                ),
                do_sample=True,
                max_model_len=max_model_len,
            )
            attempts = execute_candidates(
                candidates=candidates,
                task=task,
                base=base,
                launcher=launcher,
                runtime_root=runtime_root,
                scratch_root=scratch_root,
                workers=execution_workers,
            )
            accepted = [attempt for attempt in attempts if attempt["accepted"]]
            draws.append(
                {
                    "draw": draw,
                    "mean8": len(accepted) / 8.0,
                    "pass8": float(bool(accepted)),
                    "distinct8": float(
                        len({attempt["canonical_key"] for attempt in accepted})
                    ),
                }
            )
            ledger_draws.append(
                [
                    {key: value for key, value in attempt.items() if key != "code"}
                    for attempt in attempts
                ]
            )
        task_summaries.append(
            {
                "source_problem_id": problem_id,
                "witness_family": row["witness_family"],
                "greedy": float(greedy_attempt["accepted"]),
                "draws": draws,
            }
        )
        ledger_tasks.append(
            {
                "source_problem_id": problem_id,
                "greedy": {
                    key: value
                    for key, value in greedy_attempt.items()
                    if key != "code"
                },
                "draws": ledger_draws,
            }
        )
    evaluation_task_count = len(EVALUATION_PROBLEMS)
    evaluation_draw_count = 4
    evaluation_program_count = evaluation_task_count * (1 + 8 * evaluation_draw_count)
    greedy_value = sum(row["greedy"] for row in task_summaries) / evaluation_task_count
    metric = {
        "schema": "constructive-code-v6-stage-b-evaluation-metric-v1",
        "trainer/global_step": update,
        "misc/prompt_consumed": update * SAMPLES,
        "misc/prompt_epoch": update / 4.0,
        "eval/multi_answer/accuracy": greedy_value,
        "eval/multi_answer/sampled_mean_at_8": sum(
            draw["mean8"] for row in task_summaries for draw in row["draws"]
        )
        / (evaluation_task_count * evaluation_draw_count),
        "eval/multi_answer/sampled_any_correct_at_8": sum(
            draw["pass8"] for row in task_summaries for draw in row["draws"]
        )
        / (evaluation_task_count * evaluation_draw_count),
        "eval/multi_answer/sampled_distinct_correct_at_8": sum(
            draw["distinct8"] for row in task_summaries for draw in row["draws"]
        )
        / (evaluation_task_count * evaluation_draw_count),
        "evaluation_programs": evaluation_program_count,
        "evaluation_tasks": evaluation_task_count,
        "evaluation_draws_per_task": evaluation_draw_count,
        "task_summaries": task_summaries,
    }
    for draw in range(4):
        for key, metric_key in (
            ("mean8", "sampled_mean_at_8"),
            ("pass8", "sampled_any_correct_at_8"),
            ("distinct8", "sampled_distinct_correct_at_8"),
        ):
            metric[f"eval/multi_answer/{metric_key}_draw_{draw}"] = sum(
                row["draws"][draw][key] for row in task_summaries
            ) / evaluation_task_count
    ledger = {
        "schema": "constructive-code-v6-stage-b-evaluation-ledger-v1",
        "update": update,
        "seed": seed,
        "programs": evaluation_program_count,
        "tasks": ledger_tasks,
    }
    return metric, ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--slate-root", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--gate-identity", type=Path, required=True)
    parser.add_argument("--curation-manifest", type=Path, required=True)
    parser.add_argument("--viability-receipt", type=Path, required=True)
    parser.add_argument("--paired-audit", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-candidates", type=Path, required=True)
    parser.add_argument("--output-evaluations", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--execution-workers", type=int, default=16)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    return parser.parse_args()


def _identity_job(identity: Mapping[str, Any], args: argparse.Namespace) -> int:
    if args.mode == PAIRED_MODE:
        raw = identity.get("jobs", {}).get(args.arm)
    else:
        raw = identity.get("cells", {}).get(f"{args.arm}/s{args.seed}")
        if isinstance(raw, Mapping):
            raw = raw.get("job_id")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("ConstructiveCode job is absent from frozen identity")
    return raw


def main() -> None:
    args = parse_args()
    expected_seed = 78101 if args.mode == PAIRED_MODE else None
    if (
        (expected_seed is not None and args.seed != expected_seed)
        or (args.mode == STAGE_B_MODE and args.seed not in {43, 44, 45, 46, 47})
        or args.learning_rate != 2e-7
        or args.max_model_len != 8192
        or args.execution_workers != 16
    ):
        raise ValueError("ConstructiveCode frozen run contract drift")
    fresh = [args.output_receipt, args.output_metrics, args.output_candidates]
    if args.output_evaluations is not None:
        fresh.append(args.output_evaluations)
    if any(path.exists() for path in fresh):
        raise FileExistsError("fresh ConstructiveCode run outputs are required")
    required = (
        args.model / "config.json",
        args.slate_root / "manifest.json",
        args.image,
        args.gate_audit,
        args.gate_identity,
        args.curation_manifest,
        args.viability_receipt,
        args.protocol,
        args.identity,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.mode == STAGE_B_MODE and (
        args.paired_audit is None
        or not args.paired_audit.is_file()
        or args.output_evaluations is None
    ):
        raise ValueError("Stage B requires paired audit and evaluation ledger")
    if args.mode == PAIRED_MODE and (
        args.paired_audit is not None or args.output_evaluations is not None
    ):
        raise ValueError("paired smoke cannot load Stage-B-only artifacts")

    gate = json.loads(args.gate_audit.read_text(encoding="utf-8"))
    curation = json.loads(args.curation_manifest.read_text(encoding="utf-8"))
    viability = json.loads(args.viability_receipt.read_text(encoding="utf-8"))
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    expected_split_assignment = {
        **{key: "train" for key in TRAIN_PROBLEMS},
        **{key: "development" for key in DEVELOPMENT_PROBLEMS},
        **{key: "evaluation" for key in EVALUATION_PROBLEMS},
    }
    if (
        curation.get("schema_version")
        != "constructive-code-slate-v6-curation-v1"
        or curation.get("status") != "admitted_pre_model"
        or curation.get("split_assignment") != expected_split_assignment
        or curation.get("tasks_sha256")
        != canonical_sha256(curation.get("tasks"))
        or len(curation.get("tasks", [])) != 10
        or identity.get("v6_manifest_sha256")
        != sha256_file(args.curation_manifest)
        or viability.get("status") != "pass"
        or viability.get("decision") != "eligible_for_paired_online_training_smoke"
        or viability.get("summary", {}).get("terminal_worker_records") != 192
        or viability.get("summary", {}).get("prefix_success_tasks", 0) < 1
        or viability.get("summary", {}).get("multimode_tasks", 0) < 1
        or viability.get("hard_violations") not in ([], None)
        or _identity_job(identity, args) != int(args.job_id)
    ):
        raise ValueError("ConstructiveCode viability or job identity drift")
    paired_audit = None
    if args.mode == STAGE_B_MODE:
        assert args.paired_audit is not None
        paired_audit = json.loads(args.paired_audit.read_text(encoding="utf-8"))
        if (
            paired_audit.get("status") != "pass"
            or paired_audit.get("decision")
            != "eligible_for_ten_constructive_code_stage_b_jobs"
        ):
            raise ValueError("ConstructiveCode paired smoke did not authorize Stage B")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("ConstructiveCode training requires a BF16 GPU")
    _v5, _replay_v5, base, _materialize = _replay_modules()
    args.runtime_root.parent.mkdir(parents=True, exist_ok=True)
    args.build_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    launcher_sha = base.build_launcher(base.SANDBOX_SOURCE, args.launcher)
    runtime_identity = base.prepare_runtime(args.image, args.runtime_root)
    if args.mode == PAIRED_MODE:
        problem_ids = DEVELOPMENT_PROBLEMS
        expected_split = {key: "development" for key in problem_ids}
    else:
        problem_ids = (*TRAIN_PROBLEMS, *EVALUATION_PROBLEMS)
        expected_split = {
            **{key: "train" for key in TRAIN_PROBLEMS},
            **{key: "evaluation" for key in EVALUATION_PROBLEMS},
        }
    tasks, public_rows, checker_builds, source_manifest = load_frozen_tasks(
        problem_ids=problem_ids,
        expected_split=expected_split,
        slate_root=args.slate_root,
        v1_root=args.v1_root,
        build_root=args.build_root,
        gate=gate,
    )
    training_ids = DEVELOPMENT_PROBLEMS if args.mode == PAIRED_MODE else TRAIN_PROBLEMS
    expected_families = (
        DEVELOPMENT_FAMILIES if args.mode == PAIRED_MODE else TRAIN_FAMILIES
    )
    if tuple(public_rows[key]["witness_family"] for key in training_ids) != expected_families:
        raise RuntimeError("ConstructiveCode frozen family order drift")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    initial_model_hash = tree_sha256(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    semantic = SemanticShannonTracker(
        coefficient=0.10,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_signed_advantage=True,
        success_conditioned_signed_cap=0.05,
        open_set_inverse_adaptation=True,
        open_set_warmup_steps=64,
        open_set_ema_decay=0.90,
    )
    canonical = OnlineCanonicalBank(
        entropy_alpha=0.0,
        novelty_beta=0.50,
        pseudocount=1.0,
        surprisal_clip=5.0,
        retain_exemplars=True,
        replay_capacity=REPLAY_CAPACITY,
        global_replay_groups_per_step=1,
        global_replay_bootstrap_steps=0,
    )
    balance_controller = CanonicalReplayInverseController(
        base_alpha=0.10, warmup_steps=64, ema_decay=0.90
    )
    mass_controller = CanonicalReplayLikelihoodController(
        base_alpha=0.10, warmup_steps=64, ema_decay=0.90
    )
    padding_prompt_ids = tuple(
        tokenizer.encode(prompt(PADDING_STATEMENT), add_special_tokens=False)
    )
    padding_response_ids = (int(tokenizer.eos_token_id),)
    total_updates = len(DEVELOPMENT_PROBLEMS) if args.mode == PAIRED_MODE else 48
    training_attempts = 0
    verified_candidates = 0
    multimode_updates = 0
    optimizer_steps = 0
    all_metrics = []
    started_at = datetime.now(timezone.utc)

    if args.mode == STAGE_B_MODE:
        assert args.output_evaluations is not None
        metric, ledger = evaluation_coordinate(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            public_rows=public_rows,
            base=base,
            launcher=args.launcher,
            runtime_root=args.runtime_root,
            scratch_root=args.scratch_root,
            execution_workers=args.execution_workers,
            seed=args.seed,
            update=0,
            max_model_len=args.max_model_len,
        )
        append_jsonl(args.output_metrics, [metric])
        append_jsonl(args.output_evaluations, [ledger])

    for update in range(1, total_updates + 1):
        task_index = (update - 1) % len(training_ids)
        problem_id = training_ids[task_index]
        task = tasks[problem_id]
        row = public_rows[problem_id]
        candidates = generate_candidates(
            model=model,
            tokenizer=tokenizer,
            statement=str(row["statement"]),
            count=SAMPLES,
            namespace=request_namespace(
                mode=args.mode,
                seed=args.seed,
                phase="train",
                update=update,
                task_index=task_index,
                draw=0,
            ),
            do_sample=True,
            max_model_len=args.max_model_len,
        )
        attempts = execute_candidates(
            candidates=candidates,
            task=task,
            base=base,
            launcher=args.launcher,
            runtime_root=args.runtime_root,
            scratch_root=args.scratch_root,
            workers=args.execution_workers,
        )
        for attempt in attempts:
            append_jsonl(
                args.output_candidates,
                [
                    {
                        "schema": "constructive-code-v6-training-candidate-v1",
                        "mode": args.mode,
                        "arm": args.arm,
                        "seed": args.seed,
                        "update": update,
                        "source_problem_id": problem_id,
                        "problem_key": row["problem_key"],
                        "suite_id": row["suite_id"],
                        **attempt,
                    }
                ],
            )
        training_attempts += len(attempts)
        rewards = torch.tensor(
            [[float(attempt["accepted"]) for attempt in attempts]],
            dtype=torch.float32,
        )
        task_advantages = drgrpo_task_advantages(rewards).flatten()
        prompt_rows = [candidate["prompt_token_ids"] for candidate in candidates]
        response_rows = [candidate["response_token_ids"] for candidate in candidates]
        keys = [
            str(attempt["canonical_key"]) if attempt["accepted"] else None
            for attempt in attempts
        ]
        reward_values = [float(attempt["accepted"]) for attempt in attempts]
        semantic_raw, semantic_diag = (
            semantic.score_success_conditioned_signed_advantages_and_update(
                prompt_token_ids=prompt_rows,
                answer_keys=keys,
                task_rewards=reward_values,
                active_mask=[True] * SAMPLES,
                num_samples=SAMPLES,
            )
        )
        canonical_raw, canonical_diag = canonical.score_and_update(
            prompt_token_ids=prompt_rows,
            outcome_keys=keys,
            task_rewards=reward_values,
            active_mask=[True] * SAMPLES,
            num_samples=SAMPLES,
            response_token_ids=response_rows,
        )
        raw_exploration = torch.tensor(
            [
                (float(semantic_value) + float(canonical_value))
                * (SAMPLES - 1)
                / SAMPLES
                for semantic_value, canonical_value in zip(
                    semantic_raw, canonical_raw
                )
            ],
            dtype=torch.float32,
        )
        applied_exploration = (
            raw_exploration
            if args.arm == TREATMENT
            else torch.zeros_like(raw_exploration)
        )
        advantages = add_verified_advantage_outside_centering(
            task_advantages, applied_exploration
        )
        behavior_logps, response_masks = detached_policy_scores(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=candidates[0]["prompt_token_ids"],
            response_rows=response_rows,
        )
        replay_groups = canonical.scheduled_global_replay_groups(min_modes=1)
        optimizer.zero_grad(set_to_none=True)
        policy_diag = backward_on_policy(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=candidates[0]["prompt_token_ids"],
            response_rows=response_rows,
            behavior_logps=behavior_logps,
            response_masks=response_masks,
            advantages=advantages,
        )
        replay_diag = replay_backward(
            model=model,
            tokenizer=tokenizer,
            replay_groups=replay_groups,
            padding_prompt_ids=padding_prompt_ids,
            padding_response_ids=padding_response_ids,
            compute_only=args.arm == CONTROL,
            balance_controller=balance_controller,
            mass_controller=mass_controller,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(grad_norm.detach().item())):
            raise RuntimeError("nonfinite ConstructiveCode gradient norm")
        optimizer.step()
        optimizer_steps += 1
        accepted_count = sum(attempt["accepted"] for attempt in attempts)
        distinct_keys = len(
            {attempt["canonical_key"] for attempt in attempts if attempt["accepted"]}
        )
        verified_candidates += accepted_count
        multimode_updates += int(distinct_keys >= 2)
        metric = {
            "schema": "constructive-code-v6-training-metric-v1",
            "mode": args.mode,
            "arm": args.arm,
            "seed": args.seed,
            "trainer/global_step": update,
            "misc/prompt_consumed": update * SAMPLES,
            "misc/prompt_epoch": update / len(training_ids),
            "source_problem_id": problem_id,
            "witness_family": row["witness_family"],
            "verified_candidates": accepted_count,
            "distinct_verified_keys": distinct_keys,
            "task_advantage_rms": _tensor_rms(task_advantages),
            "raw_exploration_advantage_rms": _tensor_rms(raw_exploration),
            "applied_exploration_advantage_rms": _tensor_rms(applied_exploration),
            "online_canonical_separate_base_advantage_rms": _tensor_rms(
                task_advantages
            ),
            "online_canonical_separate_combined_advantage_rms": _tensor_rms(
                applied_exploration
            ),
            "online_canonical_advantage_applied_after_task_centering": float(
                args.arm == TREATMENT
            ),
            "online_canonical_new_outcome_row_fraction": float(
                canonical_diag.new_outcome_row_fraction
            ),
            "online_canonical_novelty_advantage_rms": float(
                canonical_diag.novelty_advantage_rms
            ),
            "online_canonical_tracked_prompts": float(
                canonical.tracked_prompt_count
            ),
            "online_canonical_tracked_outcomes": float(
                canonical.tracked_outcome_count
            ),
            "online_canonical_mean_support_per_prompt": float(
                canonical.mean_support_per_prompt
            ),
            "online_canonical_support_at_least_two_prompt_fraction": float(
                canonical.support_at_least_two_prompt_fraction
            ),
            "verified_discovery_cumulative_outcomes": float(
                canonical.tracked_outcome_count
            ),
            "verified_discovery_tracked_prompts": float(
                canonical.tracked_prompt_count
            ),
            "semantic_shannon_success_conditioned_signed_advantage_active": float(
                args.arm == TREATMENT
            ),
            "semantic_shannon_success_conditioned_signed_effective_advantage_rms": float(
                semantic_diag.effective_advantage_rms
            ),
            "grad_norm": float(grad_norm.detach().item()),
            "optimizer_step": optimizer_steps,
            **policy_diag,
            **replay_diag,
        }
        semantic_controller = semantic.state_dict().get("open_set_controller", {})
        for source, target in (
            ("entropy_ema", "semantic_shannon_success_conditioned_signed_open_set_entropy_ema"),
            ("reference_entropy", "semantic_shannon_success_conditioned_signed_open_set_reference_entropy"),
            ("current_coefficient", "semantic_shannon_success_conditioned_signed_open_set_next_coefficient"),
            ("observation_count", "semantic_shannon_success_conditioned_signed_open_set_observations"),
        ):
            value = semantic_controller.get(source)
            if value is not None:
                metric[target] = float(value)
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in metric.values()
        ):
            raise RuntimeError("nonfinite ConstructiveCode training metric")
        all_metrics.append(metric)
        append_jsonl(args.output_metrics, [metric])
        print(
            "[constructive-v6-train] "
            f"mode={args.mode} arm={args.arm} seed={args.seed} "
            f"update={update}/{total_updates} task={problem_id} "
            f"verified={accepted_count}/16 modes={distinct_keys} "
            f"loss={policy_diag['policy_loss']:.6f}",
            flush=True,
        )
        if args.mode == STAGE_B_MODE:
            assert args.output_evaluations is not None
            eval_metric, eval_ledger = evaluation_coordinate(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                public_rows=public_rows,
                base=base,
                launcher=args.launcher,
                runtime_root=args.runtime_root,
                scratch_root=args.scratch_root,
                execution_workers=args.execution_workers,
                seed=args.seed,
                update=update,
                max_model_len=args.max_model_len,
            )
            append_jsonl(args.output_metrics, [eval_metric])
            append_jsonl(args.output_evaluations, [eval_ledger])

    ended_at = datetime.now(timezone.utc)
    payload = {
        "schema": "constructive-code-v6-run-receipt-v1",
        "status": "complete",
        "mode": args.mode,
        "arm": args.arm,
        "seed": args.seed,
        "job_id": int(args.job_id),
        "generated_at": ended_at.isoformat(),
        "started_at": started_at.isoformat(),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "initial_model_tree_sha256": initial_model_hash,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "protocol_sha256": sha256_file(args.protocol),
        "identity_sha256": sha256_file(args.identity),
        "gate_audit_sha256": sha256_file(args.gate_audit),
        "gate_identity_sha256": sha256_file(args.gate_identity),
        "curation_manifest_sha256": sha256_file(args.curation_manifest),
        "viability_receipt_sha256": sha256_file(args.viability_receipt),
        "paired_audit_sha256": (
            None if args.paired_audit is None else sha256_file(args.paired_audit)
        ),
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "checker_builds": checker_builds,
        "runtime": {
            "launcher_binary_sha256": launcher_sha,
            "runtime_identity": asdict(runtime_identity),
        },
        "tasks": {
            "training": list(training_ids),
            "evaluation": (
                [] if args.mode == PAIRED_MODE else list(EVALUATION_PROBLEMS)
            ),
            "public_rows_sha256": canonical_sha256(public_rows),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "clip_epsilon": 0.2,
            "max_grad_norm": 1.0,
            "steps": optimizer_steps,
        },
        "mechanism": {
            "compute_only_control": args.arm == CONTROL,
            "semantic_coefficient": 0.10,
            "novelty_beta": 0.50,
            "replay_mass_alpha": 0.10,
            "replay_balance_alpha": 0.10,
            "replay_capacity": REPLAY_CAPACITY,
            "warmup_steps": 64,
            "ema_decay": 0.90,
            "reward_estimator_factor": 15.0 / 16.0,
            "per_rollout_replay_factor": 1.0 / 16.0,
            "gold_support_feedback": False,
            "coefficient_projection": False,
        },
        "counts": {
            "updates": optimizer_steps,
            "training_requests": training_attempts,
            "training_terminal_executions": training_attempts,
            "verified_training_candidates": verified_candidates,
            "multimode_updates": multimode_updates,
            "policy_score_rows": optimizer_steps * SAMPLES,
            "policy_response_token_slots": optimizer_steps
            * SAMPLES
            * RESPONSE_TOKENS,
            "replay_score_rows_per_pass": optimizer_steps * REPLAY_CAPACITY,
            "replay_score_passes": 2,
            "evaluation_coordinates": 0 if args.mode == PAIRED_MODE else 49,
            "evaluation_programs": (
                0
                if args.mode == PAIRED_MODE
                else 49 * len(EVALUATION_PROBLEMS) * (1 + 4 * 8)
            ),
        },
        "metrics_sha256": sha256_file(args.output_metrics),
        "candidate_ledger_sha256": sha256_file(args.output_candidates),
        "evaluation_ledger_sha256": (
            None
            if args.output_evaluations is None
            else sha256_file(args.output_evaluations)
        ),
        "information_boundary": {
            "loaded_problem_ids": list(problem_ids),
            "evaluation_rows_loaded": args.mode == STAGE_B_MODE,
            "reference_programs_in_context": False,
            "canonical_keys_in_context": False,
            "checker_source_in_context": False,
            "test_inputs_in_context": False,
            "verifier_feedback_before_terminal": False,
            "evaluation_feedback_to_training": False,
        },
        "controller_states": {
            "semantic": semantic.state_dict(),
            "canonical_bank": canonical.state_dict(),
            "replay_balance": balance_controller.state_dict(),
            "replay_mass": mass_controller.state_dict(),
        },
    }
    atomic_json(args.output_receipt, payload)
    print(
        "[constructive-v6-train] complete "
        f"mode={args.mode} arm={args.arm} seed={args.seed} "
        f"verified={verified_candidates}/{training_attempts} "
        f"multimode_updates={multimode_updates}/{optimizer_steps} "
        f"receipt={args.output_receipt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
