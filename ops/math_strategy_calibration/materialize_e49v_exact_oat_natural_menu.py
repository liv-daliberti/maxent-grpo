#!/usr/bin/env python3
"""Materialize the exact 384-train/MATH-500 E49V finite-menu dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oat_drgrpo.math_strategy_menu import (  # noqa: E402
    MENU_END,
    MENU_SCHEMA,
    MENU_START,
    parse_strategy_menu,
    strategy_menu_natural_response_instructions,
)


SOURCE = ROOT / "var/data/math12k_384_math500"
TOY = ROOT / "var/data/e49t_natural_menu_math_toy"
TOY_CERTIFICATION = (
    ROOT
    / "var/artifacts/e49s_deterministic_mathir_repairs_v1/"
    "advancement_decision.json"
)
TOY_ADVANCEMENT = (
    ROOT
    / "var/artifacts/"
    "e49t_natural_menu_math_toy_advancement_v1.json"
)
TOY_ADVANCEMENT_SCHEMA = "e49t_natural_menu_math_toy_advancement_v1"
OUTPUT_SCHEMA = "e49v_exact_oat_natural_menu_materialization_v1"
OUTPUT_LABEL = "E49V"
EXPECTED_MULTI_SUPPORT = {"train": 10, "eval": 10}
EXPECTED_MULTI_TOTAL = 20
EXPECTED_SINGLETON_TOTAL = 864
BASE_SCRIPT: pathlib.Path | None = None
MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49v_exact_oat_natural_menu_full_20260726.md"
)
SCRIPT = pathlib.Path(__file__).resolve()
GENERATION_SEEDS = (490801, 490802, 490803)
AUDIT_SEEDS = (490811, 490812)
AUDIT_ROLES = ("literal_execution", "hidden_step_attack")
AUDIT_CONTRACT = "e49v_singleton_literal_execution_v1"
EXPECTED_ENDPOINT_SHA256 = (
    "964ce3bda1d2cac7dddf2065451fd2b456433478356df9d493e98f9e23c66ac3"
)
_APPEND_LOCK = threading.Lock()


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tree_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


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


def _append_jsonl(path: pathlib.Path, payload: Any) -> None:
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _load_jsonl(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["row_id"])] = record
    return records


def _endpoint(record_path: pathlib.Path) -> tuple[str, str]:
    if _sha256(record_path) != EXPECTED_ENDPOINT_SHA256:
        raise RuntimeError("E49V endpoint record differs from frozen E49T")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "model": "qwen2.5-72b",
        "node": "node302",
        "port": 8770,
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "enforce_eager": True,
        "structured_output_backend": "guidance",
        "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("E49V endpoint configuration changed")
    return (
        f"http://{record['node']}:{int(record['port'])}/v1",
        str(record["model"]),
    )


def _post(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: int,
) -> tuple[dict[str, Any], str]:
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    error: Exception | None = None
    for attempt in range(3):
        try:
            with opener.open(request, timeout=timeout) as response:
                body = response.read()
            decoded = json.loads(body.decode("utf-8"))
            choices = decoded.get("choices") or []
            if not choices:
                raise ValueError("Qwen72 returned no choices")
            content = str(
                (choices[0].get("message") or {}).get("content") or ""
            )
            return decoded, content
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            error = RuntimeError(
                f"HTTP {exc.code} {exc.reason}: {detail[:1200]}"
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            error = exc
        if attempt < 2:
            time.sleep(2**attempt)
    raise RuntimeError(f"Qwen72 request failed after retries: {error}")


def _generation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "actions", "strategies"],
        "properties": {
            "schema": {"type": "string", "enum": [MENU_SCHEMA]},
            "actions": {
                "type": "array",
                "minItems": 2,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action_id", "operation"],
                    "properties": {
                        "action_id": {
                            "type": "string",
                            "enum": [f"A{i}" for i in range(1, 7)],
                        },
                        "operation": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 320,
                        },
                    },
                },
            },
            "strategies": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["strategy_id", "action_ids", "plan"],
                    "properties": {
                        "strategy_id": {"type": "string", "enum": ["S1"]},
                        "action_ids": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 6,
                            "items": {
                                "type": "string",
                                "enum": [f"A{i}" for i in range(1, 7)],
                            },
                        },
                        "plan": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 640,
                        },
                    },
                },
            },
        },
    }


def _audit_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assessment"],
        "properties": {
            "assessment": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "status",
                    "failure_code",
                    "derived_answer",
                    "matches_reference_answer",
                    "all_actions_executed",
                    "sufficient_without_hidden_step",
                    "leaks_final_answer",
                    "brief_execution",
                ],
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["valid", "invalid", "ambiguous"],
                    },
                    "failure_code": {
                        "type": "string",
                        "enum": [
                            "none",
                            "mathematical_error",
                            "answer_mismatch",
                            "missing_action",
                            "hidden_decisive_step",
                            "vague_or_circular",
                            "domain_or_case_gap",
                            "final_answer_leak",
                            "other",
                        ],
                    },
                    "derived_answer": {"type": "string", "minLength": 1},
                    "matches_reference_answer": {"type": "boolean"},
                    "all_actions_executed": {"type": "boolean"},
                    "sufficient_without_hidden_step": {"type": "boolean"},
                    "leaks_final_answer": {"type": "boolean"},
                    "brief_execution": {"type": "string", "minLength": 1},
                },
            }
        },
    }


def _parse_menu(payload: dict[str, Any]) -> Any:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    menu = parse_strategy_menu(
        f"x\n{MENU_START}\n{canonical}\n{MENU_END}"
    )
    if menu is None or len(menu.strategies) != 1:
        raise RuntimeError("E49V generated singleton menu did not parse")
    strategy = menu.strategies[0]
    if strategy.action_ids != tuple(
        action.action_id for action in menu.actions
    ):
        raise RuntimeError(
            "E49V singleton must execute every generated action in order"
        )
    return menu


def _generate(
    *,
    endpoint: str,
    model: str,
    problem: str,
    seed: int,
    feedback_codes: list[str],
    timeout: int,
) -> tuple[Any, dict[str, Any]]:
    feedback = ""
    if feedback_codes:
        feedback = (
            "\nA prior answer-blind audit rejected a proposal under these "
            "failure categories only: "
            + ", ".join(sorted(set(feedback_codes)))
            + ". Replace or repair the route without guessing the final "
            "answer."
        )
    user = f"""Create exactly ONE conservative finite action combo for the
MATH problem below. The combo must be a mathematically sound and sufficient
route that a small language model can literally execute.

Return 2 to 6 consecutive actions A1..An and exactly one strategy S1 whose
action_ids are every action A1..An in that order. Each operation must be
problem-specific, objective, and checkable. Include every decisive theorem,
identity, construction, substitution, case split, bound, or calculation
needed to derive the answer. Do not use vague steps such as "solve",
"reason", "simplify as needed", or "check". Do not state, encode, hint at, or
leak the final numerical or symbolic answer. The plan must describe the same
exact action sequence and must not introduce an unlisted step.

PROBLEM:
{problem}
{feedback}"""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You design executable mathematical proof plans. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "e49v_singleton_menu",
                "strict": True,
                "schema": _generation_schema(),
            },
        },
    }
    response, content = _post(endpoint, payload, timeout=timeout)
    parsed = json.loads(content)
    menu = _parse_menu(parsed)
    return menu, {
        "seed": seed,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "problem_sha256": _sha256_text(problem),
        "request_sha256": _sha256_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        ),
        "answer_was_not_in_generation_payload": True,
    }


def _audit(
    *,
    endpoint: str,
    model: str,
    problem: str,
    answer: str,
    menu: Any,
    seed: int,
    role: str,
    timeout: int,
) -> dict[str, Any]:
    role_instruction = {
        "literal_execution": (
            "Literally execute the exact ordered actions. Do not repair or "
            "silently add a theorem, identity, case, or calculation."
        ),
        "hidden_step_attack": (
            "Actively search for a hidden decisive step, vague shortcut, "
            "domain/case gap, mathematical error, or semantic answer leak. "
            "Do not give the proposal the benefit of the doubt."
        ),
    }[role]
    user = f"""Audit this singleton finite-menu route.

{role_instruction}

Work the problem independently, then execute A1..An exactly as written.
Return valid only when all actions materially execute, the listed actions are
sufficient without any hidden decisive operation, the derived answer is
mathematically equivalent to the auditor-only reference answer, and neither
the action text nor plan reveals the final answer. A problem constant that
also appears in the answer is not automatically a leak; a leak means the
requested result itself is disclosed rather than derived. If uncertain,
return ambiguous. Record the answer actually derived and a concise execution,
but do not rewrite or improve the menu.

PROBLEM:
{problem}

AUDITOR-ONLY REFERENCE ANSWER:
{answer}

MENU:
{menu.canonical_json}
"""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative mathematical route auditor. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "e49v_singleton_audit",
                "strict": True,
                "schema": _audit_schema(),
            },
        },
    }
    response, content = _post(endpoint, payload, timeout=timeout)
    parsed = json.loads(content)
    assessment = parsed.get("assessment")
    if not isinstance(assessment, dict):
        raise RuntimeError("E49V singleton audit omitted assessment")
    passed = (
        assessment.get("status") == "valid"
        and assessment.get("failure_code") == "none"
        and assessment.get("matches_reference_answer") is True
        and assessment.get("all_actions_executed") is True
        and assessment.get("sufficient_without_hidden_step") is True
        and assessment.get("leaks_final_answer") is False
        and bool(str(assessment.get("derived_answer") or "").strip())
        and bool(str(assessment.get("brief_execution") or "").strip())
    )
    return {
        "audit_contract": AUDIT_CONTRACT,
        "role": role,
        "seed": seed,
        "pass": passed,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "assessment": assessment,
    }


def _build_generated(
    *,
    endpoint: str,
    model: str,
    row_id: str,
    problem: str,
    answer: str,
    timeout: int,
    prior_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generation_round = (
        int(prior_record.get("generation_round", 0)) + 1
        if isinstance(prior_record, dict)
        else 0
    )
    attempts = []
    feedback_codes: list[str] = []
    generation_seeds = tuple(
        seed + generation_round * 1000 for seed in GENERATION_SEEDS
    )
    for generation_seed in generation_seeds:
        try:
            menu, generation = _generate(
                endpoint=endpoint,
                model=model,
                problem=problem,
                seed=generation_seed,
                feedback_codes=feedback_codes,
                timeout=timeout,
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        _audit,
                        endpoint=endpoint,
                        model=model,
                        problem=problem,
                        answer=answer,
                        menu=menu,
                        seed=seed,
                        role=role,
                        timeout=timeout,
                    )
                    for seed, role in zip(
                        AUDIT_SEEDS, AUDIT_ROLES, strict=True
                    )
                ]
                audits = [future.result() for future in futures]
            attempts.append(
                {
                    "generation": generation,
                    "menu": json.loads(menu.canonical_json),
                    "menu_sha256": menu.sha256,
                    "audits": audits,
                }
            )
            if all(audit["pass"] for audit in audits):
                return {
                    "schema": "e49v_generated_singleton_record_v1",
                    "row_id": row_id,
                    "problem_sha256": _sha256_text(problem),
                    "answer_sha256": _sha256_text(answer),
                    "menu": json.loads(menu.canonical_json),
                    "menu_sha256": menu.sha256,
                    "attempts": attempts,
                    "audit_contract": AUDIT_CONTRACT,
                    "generation_round": generation_round,
                    "pass": True,
                }
            feedback_codes = [
                str(audit["assessment"].get("failure_code") or "other")
                for audit in audits
                if not audit["pass"]
            ]
        except (
            json.JSONDecodeError,
            RuntimeError,
            ValueError,
        ) as exc:
            attempts.append(
                {"generation_seed": generation_seed, "error": str(exc)}
            )
            feedback_codes = ["other"]
    return {
        "schema": "e49v_generated_singleton_record_v1",
        "row_id": row_id,
        "problem_sha256": _sha256_text(problem),
        "answer_sha256": _sha256_text(answer),
        "attempts": attempts,
        "audit_contract": AUDIT_CONTRACT,
        "generation_round": generation_round,
        "pass": False,
    }


def _record_valid(
    record: dict[str, Any],
    *,
    problem: str,
    answer: str,
) -> bool:
    if (
        record.get("schema") != "e49v_generated_singleton_record_v1"
        or record.get("pass") is not True
        or record.get("problem_sha256") != _sha256_text(problem)
        or record.get("answer_sha256") != _sha256_text(answer)
        or record.get("audit_contract") != AUDIT_CONTRACT
        or not isinstance(record.get("menu"), dict)
    ):
        return False
    try:
        menu = _parse_menu(record["menu"])
    except (RuntimeError, ValueError, json.JSONDecodeError):
        return False
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    accepted = attempts[-1]
    audits = accepted.get("audits")
    generation = accepted.get("generation")
    if (
        menu.sha256 != record.get("menu_sha256")
        or accepted.get("menu_sha256") != menu.sha256
        or not isinstance(generation, dict)
        or generation.get("problem_sha256") != _sha256_text(problem)
        or generation.get("answer_was_not_in_generation_payload") is not True
        or not isinstance(audits, list)
        or len(audits) != 2
        or {
            (audit.get("seed"), audit.get("role"))
            for audit in audits
        }
        != set(zip(AUDIT_SEEDS, AUDIT_ROLES, strict=True))
        or any(
            audit.get("audit_contract") != AUDIT_CONTRACT
            or audit.get("pass") is not True
            or audit.get("finish_reason") != "stop"
            or not str(audit.get("response_id") or "")
            for audit in audits
        )
        or len({audit["response_id"] for audit in audits}) != 2
    ):
        return False
    return True


def _row_id(split: str, index: int, row: dict[str, Any]) -> str:
    stable = str(row.get("unique_id") or "")
    if not stable:
        stable = _sha256_text(
            str(row["problem"]) + "\0" + str(row["answer"])
        )[:20]
    return f"{split}:{index:04d}:{stable}"


def _source_key(split: str, row: dict[str, Any]) -> str:
    if split == "train":
        unique_id = str(row.get("unique_id") or "")
        if not unique_id:
            raise RuntimeError("E49V train source row lacks unique_id")
        return unique_id
    return _sha256_text(
        str(row["problem"]) + "\0" + str(row["answer"])
    )


def _toy_overlays() -> dict[str, dict[str, dict[str, Any]]]:
    overlays: dict[str, dict[str, dict[str, Any]]] = {
        "train": {},
        "eval": {},
    }
    for split in ("train", "eval"):
        dataset_dict = load_from_disk(str(TOY / split))
        dataset = dataset_dict[next(iter(dataset_dict))]
        for row in dataset:
            original = {
                **dict(row),
                "problem": str(row["original_problem"]),
            }
            key = _source_key(split, original)
            if key in overlays[split]:
                raise RuntimeError("duplicate E49T toy overlay source key")
            menu = parse_strategy_menu(str(row["problem"]))
            if (
                menu is None
                or menu.sha256 != row["strategy_menu_sha256"]
            ):
                raise RuntimeError("E49T toy overlay menu identity mismatch")
            overlays[split][key] = {
                "problem": str(row["problem"]),
                "answer": str(row["answer"]),
                "original_problem": str(row["original_problem"]),
                "menu": menu,
                "origin": str(row["strategy_menu_origin"]),
            }
    if {split: len(rows) for split, rows in overlays.items()} != {
        "train": 50,
        "eval": 50,
    }:
        raise RuntimeError("E49V requires all 100 E49T toy overlays")
    return overlays


def _embed(problem: str, menu: Any) -> str:
    embedded = (
        f"{problem}\n\n{MENU_START}\n"
        f"{menu.canonical_json}\n{MENU_END}"
        + strategy_menu_natural_response_instructions(menu)
    )
    reparsed = parse_strategy_menu(embedded)
    if reparsed is None or reparsed.sha256 != menu.sha256:
        raise RuntimeError("E49V menu changed during embedding")
    return embedded


def _prompt_token_length(tokenizer: Any, problem: str) -> int:
    rendered = (
        "<|im_start|>system\n"
        "Please reason step by step, and put your final answer within "
        "\\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + problem
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def _verify_advancement() -> dict[str, Any]:
    result = json.loads(TOY_ADVANCEMENT.read_text(encoding="utf-8"))
    if (
        result.get("schema") != TOY_ADVANCEMENT_SCHEMA
        or result.get("complete_evidence") is not True
        or result.get("advance_to_exact_oat_full") is not True
        or not all(result.get("checks", {}).values())
    ):
        raise RuntimeError(
            f"{OUTPUT_LABEL} toy has not passed the full advancement gate"
        )
    return result


def materialize(
    *,
    output_root: pathlib.Path,
    evidence_root: pathlib.Path,
    endpoint_record: pathlib.Path,
    workers: int,
    timeout: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(f"fresh E49V output root required: {output_root}")
    _verify_advancement()
    endpoint, model = _endpoint(endpoint_record)
    overlays = _toy_overlays()
    train_dict = load_from_disk(str(SOURCE / "train"))
    eval_dict = load_from_disk(str(SOURCE / "eval"))
    train_name = next(iter(train_dict))
    eval_name = next(iter(eval_dict))
    splits = {
        "train": train_dict[train_name],
        "eval": eval_dict[eval_name],
    }
    if {split: len(data) for split, data in splits.items()} != {
        "train": 384,
        "eval": 500,
    }:
        raise RuntimeError("E49V source is not exact OAT 384/MATH-500")

    work = []
    overlay_matches = {"train": 0, "eval": 0}
    observed_overlay_keys = {"train": set(), "eval": set()}
    for split, dataset in splits.items():
        for index, row in enumerate(dataset):
            key = _source_key(split, row)
            overlay = overlays[split].get(key)
            if overlay is not None:
                if (
                    overlay["original_problem"] != str(row["problem"])
                    or overlay["answer"] != str(row["answer"])
                ):
                    raise RuntimeError("E49V toy overlay source binding failed")
                overlay_matches[split] += 1
                observed_overlay_keys[split].add(key)
                continue
            work.append(
                (
                    _row_id(split, index, row),
                    str(row["problem"]),
                    str(row["answer"]),
                )
            )
    if overlay_matches != {"train": 50, "eval": 50} or any(
        observed_overlay_keys[split] != set(overlays[split])
        for split in ("train", "eval")
    ):
        raise RuntimeError("E49V did not bind all 100 toy rows exactly once")
    if len(work) != 784:
        raise RuntimeError(f"E49V expected 784 generated rows, saw {len(work)}")

    record_path = evidence_root / "private/generated_records.jsonl"
    existing = _load_jsonl(record_path)
    for row_id, problem, answer in work:
        record = existing.get(row_id)
        if record is not None and (
            record.get("problem_sha256") != _sha256_text(problem)
            or record.get("answer_sha256") != _sha256_text(answer)
        ):
            raise RuntimeError(f"E49V resume identity drift for {row_id}")
    pending = [
        (row_id, problem, answer)
        for row_id, problem, answer in work
        if not _record_valid(
            existing.get(row_id, {}),
            problem=problem,
            answer=answer,
        )
    ]
    print(
        f"E49V singleton menus total=784 complete={784-len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _build_generated,
                endpoint=endpoint,
                model=model,
                row_id=row_id,
                problem=problem,
                answer=answer,
                timeout=timeout,
                prior_record=existing.get(row_id),
            ): (row_id, problem, answer)
            for row_id, problem, answer in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            _append_jsonl(record_path, record)
            existing[record["row_id"]] = record
            print(
                f"[{completed}/{len(pending)}] {record['row_id']} "
                f"pass={record['pass']}",
                flush=True,
            )

    failures = [
        row_id
        for row_id, problem, answer in work
        if not _record_valid(
            existing.get(row_id, {}),
            problem=problem,
            answer=answer,
        )
    ]
    if failures:
        summary = {
            "schema": "e49v_exact_oat_generation_summary_v1",
            "pass": False,
            "generated_required": 784,
            "generated_passing": 784 - len(failures),
            "failures": failures,
        }
        _write_json(evidence_root / "generation_summary.json", summary)
        raise RuntimeError(
            f"E49V has {len(failures)} failed singleton menus"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    output_datasets = {}
    support_counts = {"train": 0, "eval": 0}
    prompt_lengths = []
    for split, dataset in splits.items():
        rows = []
        for index, raw in enumerate(dataset):
            row = dict(raw)
            original = str(row["problem"])
            key = _source_key(split, row)
            overlay = overlays[split].get(key)
            if overlay is not None:
                problem = overlay["problem"]
                menu = overlay["menu"]
                origin = "certified_e49t_overlay:" + overlay["origin"]
            else:
                record = existing[_row_id(split, index, row)]
                menu = _parse_menu(record["menu"])
                problem = _embed(original, menu)
                origin = "e49v_answer_blind_singleton_double_audit"
            support_counts[split] += int(len(menu.strategies) >= 2)
            prompt_length = _prompt_token_length(tokenizer, problem)
            if prompt_length > 2048:
                raise RuntimeError(
                    f"E49V prompt exceeds 2048 tokens: {split}:{index} "
                    f"tokens={prompt_length}"
                )
            prompt_lengths.append(
                {
                    "split": split,
                    "index": index,
                    "tokens": prompt_length,
                    "menu_sha256": menu.sha256,
                }
            )
            row["original_problem"] = original
            row["problem"] = problem
            row["strategy_menu_sha256"] = menu.sha256
            row["strategy_menu_origin"] = origin
            rows.append(row)
        output_datasets[split] = Dataset.from_list(rows)
    if support_counts != EXPECTED_MULTI_SUPPORT:
        raise RuntimeError(
            f"{OUTPUT_LABEL} dual support changed from its passing toy"
        )

    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        DatasetDict({train_name: output_datasets["train"]}).save_to_disk(
            str(staging / "train")
        )
        DatasetDict({eval_name: output_datasets["eval"]}).save_to_disk(
            str(staging / "eval")
        )
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "source": str(SOURCE.relative_to(ROOT)),
            "source_train_tree_sha256": _tree_sha256(SOURCE / "train"),
            "source_eval_tree_sha256": _tree_sha256(SOURCE / "eval"),
            "train_rows": 384,
            "eval_rows": 500,
            "menu_count": 884,
            "overlay_menu_count": 100,
            "generated_singleton_menu_count": 784,
            "singleton_menu_count": EXPECTED_SINGLETON_TOTAL,
            "multi_strategy_menu_count": EXPECTED_MULTI_TOTAL,
            "multi_support_counts": support_counts,
            "zero_support_rows": 0,
            "all_generated_singletons_double_audited": True,
            "audit_contract": AUDIT_CONTRACT,
            "audit_seeds": list(AUDIT_SEEDS),
            "audit_roles": list(AUDIT_ROLES),
            "generation_seeds": list(GENERATION_SEEDS),
            "retry_generation_seed_stride": 1000,
            "prompt_token_limit": 2048,
            "max_prompt_tokens": max(
                row["tokens"] for row in prompt_lengths
            ),
            "source_exact_order_preserved": True,
            "train_tree_sha256": _tree_sha256(staging / "train"),
            "eval_tree_sha256": _tree_sha256(staging / "eval"),
            "generated_records_sha256": _sha256(record_path),
            "toy_data_manifest_sha256": _sha256(
                TOY / "MATERIALIZATION_MANIFEST.json"
            ),
            "toy_certification_sha256": _sha256(TOY_CERTIFICATION),
            "toy_advancement_sha256": _sha256(TOY_ADVANCEMENT),
            "endpoint_record_sha256": _sha256(endpoint_record),
            "protocol_sha256": _sha256(PROTOCOL),
            "materializer_sha256": _sha256(SCRIPT),
            "base_materializer_sha256": (
                _sha256(BASE_SCRIPT) if BASE_SCRIPT is not None else None
            ),
        }
        _write_json(staging / "MATERIALIZATION_MANIFEST.json", manifest)
        os.replace(staging, output_root)
    finally:
        if staging.exists():
            for item in sorted(staging.rglob("*"), reverse=True):
                if item.is_file():
                    item.unlink()
                else:
                    item.rmdir()
            staging.rmdir()

    _write_json(
        evidence_root / "prompt_length_audit.json",
        {
            "schema": "e49v_prompt_length_audit_v1",
            "pass": True,
            "limit": 2048,
            "max": max(row["tokens"] for row in prompt_lengths),
            "rows": prompt_lengths,
        },
    )
    summary = {
        "schema": "e49v_exact_oat_generation_summary_v1",
        "pass": True,
        "output_root": str(output_root.relative_to(ROOT)),
        "generated_required": 784,
        "generated_passing": 784,
        "overlay_count": 100,
        "manifest_sha256": _sha256(
            output_root / "MATERIALIZATION_MANIFEST.json"
        ),
    }
    _write_json(evidence_root / "generation_summary.json", summary)
    return summary


def audit_only(
    *,
    output_root: pathlib.Path,
    evidence_root: pathlib.Path,
    endpoint_record: pathlib.Path,
) -> dict[str, Any]:
    manifest_path = output_root / "MATERIALIZATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema": OUTPUT_SCHEMA,
        "train_rows": 384,
        "eval_rows": 500,
        "menu_count": 884,
        "overlay_menu_count": 100,
        "generated_singleton_menu_count": 784,
        "singleton_menu_count": EXPECTED_SINGLETON_TOTAL,
        "multi_strategy_menu_count": EXPECTED_MULTI_TOTAL,
        "multi_support_counts": EXPECTED_MULTI_SUPPORT,
        "zero_support_rows": 0,
        "all_generated_singletons_double_audited": True,
        "source_exact_order_preserved": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("E49V materialization manifest contract mismatch")
    if (
        manifest.get("source_train_tree_sha256")
        != _tree_sha256(SOURCE / "train")
        or manifest.get("source_eval_tree_sha256")
        != _tree_sha256(SOURCE / "eval")
        or manifest.get("train_tree_sha256")
        != _tree_sha256(output_root / "train")
        or manifest.get("eval_tree_sha256")
        != _tree_sha256(output_root / "eval")
        or manifest.get("generated_records_sha256")
        != _sha256(evidence_root / "private/generated_records.jsonl")
        or manifest.get("endpoint_record_sha256")
        != _sha256(endpoint_record)
        or manifest.get("toy_advancement_sha256")
        != _sha256(TOY_ADVANCEMENT)
    ):
        raise RuntimeError("E49V frozen artifact identity mismatch")
    return {
        "schema": "e49v_exact_oat_audit_only_v1",
        "pass": True,
        "manifest_sha256": _sha256(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=ROOT / "var/data/e49v_exact_oat_natural_menu_full",
    )
    parser.add_argument(
        "--evidence-root",
        type=pathlib.Path,
        default=(
            ROOT
            / "var/artifacts/e49v_exact_oat_natural_menu_full_v1"
        ),
    )
    parser.add_argument(
        "--endpoint-record",
        type=pathlib.Path,
        default=(
            ROOT
            / "var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise RuntimeError("E49V workers must be in 1..8")
    if args.audit_only:
        result = audit_only(
            output_root=args.output_root.resolve(),
            evidence_root=args.evidence_root.resolve(),
            endpoint_record=args.endpoint_record.resolve(),
        )
    else:
        result = materialize(
            output_root=args.output_root.resolve(),
            evidence_root=args.evidence_root.resolve(),
            endpoint_record=args.endpoint_record.resolve(),
            workers=args.workers,
            timeout=args.timeout,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
