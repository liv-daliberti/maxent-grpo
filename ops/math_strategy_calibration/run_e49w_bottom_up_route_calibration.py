#!/usr/bin/env python3
"""Build bottom-up hard-MATH menus and test both routes with Qwen2.5-0.5B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
E47 = ROOT / "var/artifacts/e47_math_strategy_calibration_v1"
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
    / "paper/preregistration/"
    "e49w_bottom_up_05b_route_menu_calibration_20260726.md"
)
CALIBRATION_SCHEMA = "e49w_bottom_up_05b_route_calibration_v1"
BASE_SCRIPT: pathlib.Path | None = None
SAMPLE_COUNT = 8
SEED = 490771
MAX_EXEMPLARS = 12
MAX_EXEMPLAR_CHARS = 6000
SYSTEM = (
    "You are a conservative mathematical proof-route auditor. "
    "Return valid JSON only."
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


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _endpoint(record_path: pathlib.Path) -> tuple[str, str]:
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
        raise RuntimeError("E49W requires the frozen E49T Qwen72 endpoint")
    return (
        f"http://{record['node']}:{int(record['port'])}/v1",
        str(record["model"]),
    )


def _post(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
                decoded = json.loads(response.read().decode("utf-8"))
            choices = decoded.get("choices") or []
            content = str(
                ((choices[0] if choices else {}).get("message") or {}).get(
                    "content"
                )
                or ""
            )
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("structured response was not an object")
            return decoded, parsed
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            error = RuntimeError(
                f"HTTP {exc.code} {exc.reason}: {detail[:1000]}"
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
    raise RuntimeError(f"Qwen72 request failed after 3 attempts: {error}")


def _menu_schema() -> dict[str, Any]:
    action_ids = [f"A{index}" for index in range(1, 9)]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "actions", "strategies"],
        "properties": {
            "schema": {
                "type": "string",
                "enum": ["math_strategy_action_menu_v1"],
            },
            "actions": {
                "type": "array",
                "minItems": 2,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action_id", "operation"],
                    "properties": {
                        "action_id": {
                            "type": "string",
                            "enum": action_ids,
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
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["strategy_id", "action_ids", "plan"],
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "enum": ["S1", "S2"],
                        },
                        "action_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 6,
                            "items": {
                                "type": "string",
                                "enum": action_ids,
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


def _proposal_schema(sample_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["menu", "route_sources"],
        "properties": {
            "menu": _menu_schema(),
            "route_sources": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "strategy_id",
                        "source_kind",
                        "exemplar_ids",
                    ],
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "enum": ["S1", "S2"],
                        },
                        "source_kind": {
                            "type": "string",
                            "enum": ["observed", "proposed"],
                        },
                        "exemplar_ids": {
                            "type": "array",
                            "maxItems": 3,
                            "items": {
                                "type": "string",
                                "enum": sample_ids,
                            },
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
        "required": ["strategy_assessments", "source_assessments", "pair"],
        "properties": {
            "strategy_assessments": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "strategy_id",
                        "status",
                        "failure_code",
                        "derived_answer",
                        "matches_reference_answer",
                        "all_actions_sufficient",
                        "missing_decisive_step",
                        "menu_reveals_final_answer",
                        "actions_concrete_for_small_model",
                    ],
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "enum": ["S1", "S2"],
                        },
                        "status": {
                            "type": "string",
                            "enum": ["sound", "unsound", "ambiguous"],
                        },
                        "failure_code": {
                            "type": "string",
                            "enum": [
                                "none",
                                "algebra_error",
                                "invalid_method",
                                "domain_or_case_error",
                                "missing_decisive_step",
                                "vague_or_circular",
                                "answer_mismatch",
                                "answer_leak",
                                "other",
                            ],
                        },
                        "derived_answer": {
                            "type": "string",
                            "maxLength": 128,
                        },
                        "matches_reference_answer": {"type": "boolean"},
                        "all_actions_sufficient": {"type": "boolean"},
                        "missing_decisive_step": {"type": "boolean"},
                        "menu_reveals_final_answer": {"type": "boolean"},
                        "actions_concrete_for_small_model": {"type": "boolean"},
                    },
                },
            },
            "source_assessments": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["strategy_id", "binding_status"],
                    "properties": {
                        "strategy_id": {
                            "type": "string",
                            "enum": ["S1", "S2"],
                        },
                        "binding_status": {
                            "type": "string",
                            "enum": [
                                "observed_bound",
                                "proposed_not_applicable",
                                "invalid",
                            ],
                        },
                    },
                },
            },
            "pair": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "relation",
                    "shared_core",
                    "s1_decisive_operation",
                    "s2_decisive_operation",
                ],
                "properties": {
                    "relation": {
                        "type": "string",
                        "enum": ["distinct", "equivalent", "ambiguous"],
                    },
                    "shared_core": {"type": "string", "maxLength": 256},
                    "s1_decisive_operation": {
                        "type": "string",
                        "maxLength": 256,
                    },
                    "s2_decisive_operation": {
                        "type": "string",
                        "maxLength": 256,
                    },
                },
            },
        },
    }


def _load_candidates() -> list[dict[str, Any]]:
    problems = {
        row["problem_id"]: row
        for row in _read_jsonl(E47 / "problems.jsonl")
    }
    summary = json.loads(
        (E47 / "validation_summary.json").read_text(encoding="utf-8")
    )
    positives: dict[str, int] = summary["per_problem"]
    selected_ids = sorted(
        (problem_id for problem_id, count in positives.items() if count >= 2),
        key=lambda problem_id: (-int(positives[problem_id]), problem_id),
    )
    if len(selected_ids) != 19:
        raise RuntimeError(
            f"E49W expected 19 E47 candidates, found {len(selected_ids)}"
        )
    responses: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(E47 / "validated_policy.blinded.jsonl"):
        responses.setdefault(str(row["problem_id"]), []).append(row)
    candidates = []
    for rank, problem_id in enumerate(selected_ids):
        eligible = sorted(
            (
                row
                for row in responses.get(problem_id, [])
                if len(str(row["text"])) <= MAX_EXEMPLAR_CHARS
            ),
            key=lambda row: (len(str(row["text"])), str(row["sample_id"])),
        )[:MAX_EXEMPLARS]
        if len(eligible) < 2:
            raise RuntimeError(f"E49W candidate {problem_id} lost exemplars")
        problem = problems[problem_id]
        if int(problem.get("level", 0)) != 5:
            raise RuntimeError("E49W candidate cohort is no longer level 5")
        candidates.append(
            {
                "candidate_rank": rank,
                "problem_id": problem_id,
                "row_id": str(problem["unique_id"]),
                "problem": str(problem["problem"]),
                "answer": str(problem["answer"]),
                "subject": str(problem["subject"]),
                "level": int(problem["level"]),
                "validator_positive_count": int(positives[problem_id]),
                "exemplars": eligible,
            }
        )
    return candidates


def _parse_menu(payload: dict[str, Any]) -> Any:
    from oat_drgrpo.math_strategy_menu import (
        MENU_END,
        MENU_START,
        parse_strategy_menu,
    )

    embedded = (
        f"x\n{MENU_START}\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + f"\n{MENU_END}"
    )
    menu = parse_strategy_menu(embedded)
    if menu is None or len(menu.strategies) != 2:
        raise ValueError("E49W proposal did not parse as a two-route menu")
    return menu


def _proposal_passes_local_contract(
    proposal: dict[str, Any],
    sample_ids: set[str],
) -> bool:
    return not _proposal_local_contract_failures(proposal, sample_ids)


def _proposal_local_contract_failures(
    proposal: dict[str, Any],
    sample_ids: set[str],
) -> list[str]:
    failures: list[str] = []
    try:
        menu = _parse_menu(proposal["menu"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"menu_parse:{type(exc).__name__}:{exc}")
        return failures
    sources = proposal.get("route_sources")
    if not isinstance(sources, list) or len(sources) != 2:
        failures.append("route_sources_not_exactly_two")
        return failures
    by_id = {
        str(row.get("strategy_id")): row
        for row in sources
        if isinstance(row, dict)
    }
    if set(by_id) != {"S1", "S2"}:
        failures.append("route_source_ids_not_exactly_s1_s2")
        return failures
    if by_id["S1"].get("source_kind") != "observed":
        failures.append("s1_not_observed")
    for strategy in menu.strategies:
        row = by_id[strategy.strategy_id]
        exemplars = row.get("exemplar_ids")
        if not isinstance(exemplars, list):
            failures.append(
                f"{strategy.strategy_id.casefold()}_exemplars_not_list"
            )
            continue
        if len(set(exemplars)) != len(exemplars):
            failures.append(
                f"{strategy.strategy_id.casefold()}_duplicate_exemplar"
            )
        if not set(exemplars) <= sample_ids:
            failures.append(
                f"{strategy.strategy_id.casefold()}_unknown_exemplar"
            )
        if row.get("source_kind") == "observed" and not exemplars:
            failures.append(
                f"{strategy.strategy_id.casefold()}_observed_without_exemplar"
            )
        if row.get("source_kind") == "proposed" and exemplars:
            failures.append(
                f"{strategy.strategy_id.casefold()}_proposed_with_exemplar"
            )
    return failures


def _propose(
    *,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rendered = "\n\n".join(
        f"EXEMPLAR {row['sample_id']}:\n{row['text']}"
        for row in candidate["exemplars"]
    )
    prompt = f"""Construct exactly two finite solution routes for this hard
MATH problem, working bottom-up from correct Qwen2.5-0.5B derivations.

S1 must describe a route materially executed by at least one supplied
exemplar. S2 should describe a second genuinely distinct supplied route when
one exists. If every exemplar uses the same central method, S2 may instead be
a newly proposed route, but it must be concise, sound, and realistically
executable by a 0.5B model when explicitly requested.

Build one shared action vocabulary A1..An and exactly S1,S2. Every action must
be a concrete, problem-specific mathematical operation. Each combo must
contain every decisive step needed to derive the answer, with no vague
"solve", "reason", "simplify as needed", hidden theorem, or merely decorative
check. Distinct routes must differ in a central identity, construction,
counted set, substitution, theorem, or search method—not wording or routine
algebra. Do not include, encode, or hint the final answer anywhere in an
action or plan. Cite up to three supplied exemplar IDs for each observed
route; proposed routes must have no exemplar IDs.

PROBLEM:
{candidate['problem']}

VALIDATOR-POSITIVE 0.5B EXEMPLARS:
{rendered}
"""
    sample_ids = [str(row["sample_id"]) for row in candidate["exemplars"]]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 3072,
        "seed": SEED + candidate["candidate_rank"],
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "e49w_bottom_up_menu",
                "strict": True,
                "schema": _proposal_schema(sample_ids),
            },
        },
    }
    response, proposal = _post(endpoint, payload, timeout=timeout)
    if not _proposal_passes_local_contract(proposal, set(sample_ids)):
        raise ValueError("E49W proposal failed its local finite-menu contract")
    return proposal, {
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "seed": payload["seed"],
    }


def _audit(
    *,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    proposal: dict[str, Any],
    audit_index: int,
    timeout: int,
) -> dict[str, Any]:
    exemplars = {
        str(row["sample_id"]): str(row["text"])
        for row in candidate["exemplars"]
    }
    cited = sorted(
        {
            sample_id
            for source in proposal["route_sources"]
            for sample_id in source["exemplar_ids"]
        }
    )
    rendered = "\n\n".join(
        f"EXEMPLAR {sample_id}:\n{exemplars[sample_id]}"
        for sample_id in cited
    )
    role = (
        "literal route soundness, completeness, and exemplar binding"
        if audit_index == 0
        else "adversarial route equivalence, answer leakage, and hidden-step veto"
    )
    prompt = f"""Independently audit a proposed finite route menu. Your role is
{role}.

Literally execute every action in each exact combo. Do not repair a route or
silently import a missing identity. A route passes only if its listed actions
are sufficient to derive the auditor-only answer. Check that operations are
concrete enough for a small model, that no action or plan reveals the final
answer, and that every observed route is genuinely executed by its cited
exemplar. A proposed route has no exemplar binding.

Treat the pair as equivalent unless you can name a distinct decisive
operation actually present on each side. Notation, ordering routine algebra,
factoring versus expanding the same equation, or adding a redundant check do
not establish novelty. Return ambiguous when unsure.

PROBLEM:
{candidate['problem']}

AUDITOR-ONLY REFERENCE ANSWER:
{candidate['answer']}

MENU AND SOURCE CLAIMS:
{json.dumps(proposal, sort_keys=True)}

CITED EXEMPLARS:
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
        "seed": SEED + 100 + audit_index,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "e49w_bottom_up_menu_audit",
                "strict": True,
                "schema": _audit_schema(),
            },
        },
    }
    response, assessment = _post(endpoint, payload, timeout=timeout)
    return {
        "audit_index": audit_index,
        "role": role,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "seed": payload["seed"],
        "assessment": assessment,
    }


def _audit_passes(
    proposal: dict[str, Any],
    audit: dict[str, Any],
) -> bool:
    if (
        audit.get("finish_reason") != "stop"
        or not str(audit.get("response_id") or "")
    ):
        return False
    assessment = audit.get("assessment")
    if not isinstance(assessment, dict):
        return False
    strategies = assessment.get("strategy_assessments")
    if not isinstance(strategies, list) or {
        row.get("strategy_id") for row in strategies if isinstance(row, dict)
    } != {"S1", "S2"}:
        return False
    if any(
        row.get("status") != "sound"
        or row.get("failure_code") != "none"
        or row.get("matches_reference_answer") is not True
        or row.get("all_actions_sufficient") is not True
        or row.get("missing_decisive_step") is not False
        or row.get("menu_reveals_final_answer") is not False
        or row.get("actions_concrete_for_small_model") is not True
        or not str(row.get("derived_answer") or "").strip()
        for row in strategies
    ):
        return False
    sources = assessment.get("source_assessments")
    if not isinstance(sources, list):
        return False
    actual = {
        str(row.get("strategy_id")): row.get("binding_status")
        for row in sources
        if isinstance(row, dict)
    }
    expected = {
        str(row["strategy_id"]): (
            "observed_bound"
            if row["source_kind"] == "observed"
            else "proposed_not_applicable"
        )
        for row in proposal["route_sources"]
    }
    if actual != expected:
        return False
    pair = assessment.get("pair")
    return bool(
        isinstance(pair, dict)
        and pair.get("relation") == "distinct"
        and str(pair.get("s1_decisive_operation") or "").strip()
        and str(pair.get("s2_decisive_operation") or "").strip()
        and str(pair["s1_decisive_operation"]).strip().casefold()
        != str(pair["s2_decisive_operation"]).strip().casefold()
    )


def _build_candidate(
    *,
    endpoint: str,
    model: str,
    candidate: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    base = {
        "candidate_rank": candidate["candidate_rank"],
        "problem_id": candidate["problem_id"],
        "row_id": candidate["row_id"],
        "problem_sha256": _sha256_text(candidate["problem"]),
        "reference_answer_sha256": _sha256_text(candidate["answer"]),
        "subject": candidate["subject"],
        "level": candidate["level"],
        "validator_positive_count": candidate["validator_positive_count"],
        "selected_exemplar_ids": [
            str(row["sample_id"]) for row in candidate["exemplars"]
        ],
    }
    try:
        proposal, generation = _propose(
            endpoint=endpoint,
            model=model,
            candidate=candidate,
            timeout=timeout,
        )
        menu = _parse_menu(proposal["menu"])
        audits = [
            _audit(
                endpoint=endpoint,
                model=model,
                candidate=candidate,
                proposal=proposal,
                audit_index=index,
                timeout=timeout,
            )
            for index in range(2)
        ]
        passed = all(_audit_passes(proposal, audit) for audit in audits)
        return {
            **base,
            "generation": generation,
            "menu": json.loads(menu.canonical_json),
            "menu_sha256": menu.sha256,
            "route_sources": proposal["route_sources"],
            "audits": audits,
            "double_audit_pass": passed,
        }
    except Exception as exc:
        return {**base, "double_audit_pass": False, "error": str(exc)}


def _embed(problem: str, menu: Any) -> str:
    from oat_drgrpo.math_strategy_menu import (
        MENU_END,
        MENU_START,
        strategy_menu_natural_response_instructions,
    )

    return (
        problem.rstrip()
        + f"\n\n{MENU_START}\n"
        + menu.canonical_json
        + f"\n{MENU_END}"
        + strategy_menu_natural_response_instructions(menu)
    )


def _forced_problem(problem: str, menu: Any, strategy: Any) -> str:
    embedded = _embed(problem, menu)
    marker = "</strategy_menu_json>"
    menu_end = embedded.index(marker) + len(marker)
    return (
        embedded[:menu_end]
        + "\n\nROUTE-CAPABILITY AUDIT: execute exactly "
        + f"{strategy.strategy_id}: {strategy.action_combo}. "
        + "State that strategy ID and combo, then materially execute every "
        + "listed action in order as a natural mathematical derivation. Do "
        + "not use, mention, mix, or switch to any other route. Finish with "
        + "the final answer inside \\boxed{}. A correct answer reached by a "
        + "different route does not pass this audit."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=(
            ROOT
            / "var/artifacts/e49w_bottom_up_05b_route_calibration_v1"
        ),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh E49W result required: {result_path}")

    sys.path.insert(0, str(ROOT / "src"))
    identity = json.loads(E49T_IDENTITY.read_text(encoding="utf-8"))
    frozen_source = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"e49t_natural_menu_{identity['source_hash']}"
        / "src"
    )
    if not frozen_source.is_dir():
        raise RuntimeError("E49W frozen E49T source snapshot is missing")
    endpoint, judge_model = _endpoint(ENDPOINT_RECORD)
    candidates = _load_candidates()

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _build_candidate,
                endpoint=endpoint,
                model=judge_model,
                candidate=candidate,
                timeout=args.timeout,
            ): candidate["candidate_rank"]
            for candidate in candidates
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                json.dumps(
                    {
                        "candidate_rank": record["candidate_rank"],
                        "problem_id": record["problem_id"],
                        "double_audit_pass": record["double_audit_pass"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    records.sort(key=lambda row: int(row["candidate_rank"]))
    _write_jsonl(output / "menu_records.jsonl", records)
    accepted_records = [
        record for record in records if record["double_audit_pass"]
    ]
    if not accepted_records:
        payload = {
            "schema": CALIBRATION_SCHEMA,
            "pass": False,
            "failure": "no_double_audited_menu",
            "candidate_count": len(records),
            "double_audited_menu_count": 0,
        }
        _write_json(result_path, payload)
        print(result_path)
        return

    from oat_drgrpo.math_grader import boxed_reward_fn
    import oat_drgrpo

    # Keep exact answer validation on the corrected successor grader, while
    # route admission remains the byte-frozen E49T calibrated canonicalizer.
    oat_drgrpo.__path__.insert(0, str(frozen_source / "oat_drgrpo"))
    from oat_drgrpo.math_strategy_canonicalizer import (
        MathStrategyCanonicalizer,
    )
    from oat_drgrpo.math_strategy_menu import parse_strategy_menu
    from oat_drgrpo.templates import apply_qwen_math_template
    import vllm

    candidate_by_id = {
        candidate["problem_id"]: candidate for candidate in candidates
    }
    cases = []
    for record in accepted_records:
        candidate = candidate_by_id[record["problem_id"]]
        menu = _parse_menu(record["menu"])
        for strategy in menu.strategies:
            cases.append(
                {
                    "candidate_rank": record["candidate_rank"],
                    "problem_id": record["problem_id"],
                    "row_id": record["row_id"],
                    "problem": _forced_problem(
                        candidate["problem"], menu, strategy
                    ),
                    "answer": candidate["answer"],
                    "menu": menu,
                    "strategy": strategy,
                }
            )

    prompts = [apply_qwen_math_template(case["problem"]) for case in cases]
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
    if len(outputs) != len(cases):
        raise RuntimeError("E49W vLLM output count mismatch")

    flat_prompt_tokens = []
    flat_problem_texts = []
    flat_responses = []
    flat_answer_positive = []
    private_rows = []
    for case_index, (case, request_output) in enumerate(
        zip(cases, outputs, strict=True)
    ):
        if len(request_output.outputs) != SAMPLE_COUNT:
            raise RuntimeError("E49W vLLM sample count mismatch")
        prompt_tokens = list(request_output.prompt_token_ids)
        for sample_index, sample in enumerate(request_output.outputs):
            text = str(sample.text)
            _, reward = boxed_reward_fn(text, case["answer"], fast=False)
            answer_positive = float(reward) > 0.0
            flat_prompt_tokens.append(prompt_tokens)
            flat_problem_texts.append(case["problem"])
            flat_responses.append(text)
            flat_answer_positive.append(answer_positive)
            private_rows.append(
                {
                    "case_index": case_index,
                    "candidate_rank": case["candidate_rank"],
                    "problem_id": case["problem_id"],
                    "row_id": case["row_id"],
                    "strategy_id": case["strategy"].strategy_id,
                    "action_combo": case["strategy"].action_combo,
                    "sample_index": sample_index,
                    "answer_positive": answer_positive,
                    "response": text,
                    "response_sha256": _sha256_text(text),
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
        num_samples=SAMPLE_COUNT,
    )

    case_results = []
    for case_index, case in enumerate(cases):
        start = case_index * SAMPLE_COUNT
        stop = start + SAMPLE_COUNT
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
                "candidate_rank": case["candidate_rank"],
                "problem_id": case["problem_id"],
                "row_id": case["row_id"],
                "menu_sha256": case["menu"].sha256,
                "strategy_id": case["strategy"].strategy_id,
                "action_combo": case["strategy"].action_combo,
                "sample_count": SAMPLE_COUNT,
                "answer_success_count": sum(
                    flat_answer_positive[start:stop]
                ),
                "forced_route_success_count": expected,
                "wrong_route_success_count": wrong,
                "minimally_executable": expected >= 1,
            }
        )

    problem_results = []
    for record in accepted_records:
        routes = [
            row
            for row in case_results
            if row["problem_id"] == record["problem_id"]
        ]
        if len(routes) != 2:
            raise RuntimeError("E49W candidate lost one forced route")
        problem_results.append(
            {
                "candidate_rank": record["candidate_rank"],
                "problem_id": record["problem_id"],
                "row_id": record["row_id"],
                "menu": record["menu"],
                "menu_sha256": record["menu_sha256"],
                "route_sources": record["route_sources"],
                "bidirectionally_executable": all(
                    row["minimally_executable"] for row in routes
                ),
                "route_success_counts": {
                    row["strategy_id"]: row["forced_route_success_count"]
                    for row in routes
                },
                "answer_success_counts": {
                    row["strategy_id"]: row["answer_success_count"]
                    for row in routes
                },
            }
        )
    problem_results.sort(key=lambda row: int(row["candidate_rank"]))
    selected = [
        row["problem_id"]
        for row in problem_results
        if row["bidirectionally_executable"]
    ][:10]
    private_path = output / "private/responses.jsonl"
    _write_jsonl(private_path, private_rows)
    menu_records_path = output / "menu_records.jsonl"
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "pass": len(selected) == 10,
        "decision_rule": {
            "candidate_order": "validator_positive_desc_then_problem_id",
            "route_minimum_successes_of_8": 1,
            "required_bidirectional_problems": 10,
            "selection": "first_10_passing_candidates_in_frozen_order",
        },
        "identity": {
            "e47_manifest_sha256": _sha256(E47 / "manifest.json"),
            "e47_problems_sha256": _sha256(E47 / "problems.jsonl"),
            "e47_validated_policy_sha256": _sha256(
                E47 / "validated_policy.blinded.jsonl"
            ),
            "e47_validation_summary_sha256": _sha256(
                E47 / "validation_summary.json"
            ),
            "e49t_identity_sha256": _sha256(E49T_IDENTITY),
            "endpoint_record_sha256": _sha256(ENDPOINT_RECORD),
            "protocol_sha256": _sha256(PROTOCOL),
            "script_sha256": _sha256(SCRIPT),
            "base_script_sha256": (
                _sha256(BASE_SCRIPT) if BASE_SCRIPT is not None else None
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
        },
        "candidate_count": len(candidates),
        "double_audited_menu_count": len(accepted_records),
        "forced_case_count": len(case_results),
        "sample_count_per_route": SAMPLE_COUNT,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "bidirectionally_executable_count": sum(
            row["bidirectionally_executable"] for row in problem_results
        ),
        "selected_problem_ids": selected,
        "wrong_route_success_count": sum(
            row["wrong_route_success_count"] for row in case_results
        ),
        "canonicalizer_diagnostics": {
            field: float(getattr(diagnostics, field))
            for field in diagnostics.__dataclass_fields__
        },
        "menu_records_sha256": _sha256(menu_records_path),
        "private_responses_sha256": _sha256(private_path),
        "cases": case_results,
        "problems": problem_results,
    }
    _write_json(result_path, payload)
    print(result_path)


if __name__ == "__main__":
    main()
