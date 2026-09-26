"""Source-only replacement heldout contracts frozen before any policy access."""
from __future__ import annotations
from collections import Counter
import json
from typing import Any
from .constructive_code import CanonicalWitness, ConstructiveCodeError, ReleasedCheckerDecision, WITNESS_SCHEMA_VERSION, sha256_bytes
from .constructive_code_wider_adapters_20260921 import Cursor, _string
from . import constructive_code_adapters as old

FAMILIES = {"1196_C": "assignment", "1092_A": "assignment", "1413_A": "assignment", "1520_C": "assignment"}
ADAPTER_IDS = {p: f"holdout_{p.lower()}_v1" for p in FAMILIES}
ADAPTER_IDS["1092_A"] = "holdout_1092_a_frequency_v1"


def input_cases(problem: str, text: str) -> list[dict[str, Any]]:
    c = Cursor(text)
    count = c.integer(1, {"1196_C": 100000, "1092_A": 100, "1413_A": 1000, "1520_C": 100}[problem])
    cases = []
    for _ in range(count):
        if problem == "1196_C":
            n = c.integer(1, 100000)
            case = {"n": n, "robots": [[*c.ints(2, -100000, 100000), *c.ints(4, 0, 1)] for _ in range(n)]}
        elif problem == "1092_A":
            n = c.integer(1, 100); case = {"n": n, "k": c.integer(1, min(n, 26))}
        elif problem == "1413_A":
            n = c.integer(2, 100); a = c.ints(n, -100, 100)
            if n % 2 or any(x == 0 for x in a):
                raise ConstructiveCodeError("seal inputs must have even length and nonzero energies")
            case = {"n": n, "a": a}
        else:
            case = {"n": c.integer(1, 100)}
        cases.append(case)
    c.end()
    if problem == "1196_C" and sum(case["n"] for case in cases) > 100000:
        raise ConstructiveCodeError("total robot count exceeds source bound")
    return cases


def canonical_value(problem: str, input_text: str, output_text: str) -> Any:
    c = Cursor(output_text); result = []
    for case in input_cases(problem, input_text):
        if problem == "1196_C":
            status = c.integer(0, 1)
            value = {"point": c.ints(2)} if status else {"status": "impossible"}
        elif problem == "1092_A":
            # No task constraint depends on character positions: quotient all
            # string permutations, retaining frequencies of the named letters.
            counts = Counter(_string(c, case["n"], "abcdefghijklmnopqrstuvwxyz"[:case["k"]]))
            value = [counts[chr(97 + i)] for i in range(case["k"])]
        elif problem == "1413_A":
            value = c.ints(case["n"])
        else:
            first = c.integer()
            value = {"status": "impossible"} if first == -1 else [first, *c.ints(case["n"] ** 2 - 1)]
        result.append(value)
    c.end(); return result


def canonicalize_task_witness(*, problem_id: str, adapter_id: str, input_data: str | bytes, output: str | bytes, decision: ReleasedCheckerDecision) -> CanonicalWitness:
    if ADAPTER_IDS.get(problem_id) != adapter_id:
        raise ConstructiveCodeError("unregistered heldout task/adapter")
    ib, it = old._trusted_text(input_data, "input", 16 * 1024**2)
    ob, ot = old._trusted_text(output, "output", 16 * 1024**2)
    if not decision.accepted or decision.input_sha256 != sha256_bytes(ib) or decision.output_sha256 != sha256_bytes(ob):
        raise ConstructiveCodeError("canonicalization requires exact accepted checker binding")
    family = FAMILIES[problem_id]
    payload = {"adapter": adapter_id, "family": family, "schema_version": WITNESS_SCHEMA_VERSION, "value": canonical_value(problem_id, it, ot)}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return CanonicalWitness(family, decision.checker_sha256, decision.input_sha256, decision.output_sha256, text, f"constructive_witness:{family}:v1:{sha256_bytes(text.encode('ascii'))}")
