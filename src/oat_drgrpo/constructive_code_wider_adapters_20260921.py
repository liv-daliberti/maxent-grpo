"""Task-bound witness identities and independent input checks for the wider slate.

A released checker decision must accept and bind the exact input/output bytes.
Canonicalization removes only stated serialization or anonymous-label freedoms.
It is not a replacement correctness oracle and does not label algorithms.
"""
from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from .constructive_code import CanonicalWitness, ConstructiveCodeError, ReleasedCheckerDecision, WITNESS_SCHEMA_VERSION, sha256_bytes
from . import constructive_code_adapters as old

FAMILIES = {
    "1454_A": "ordered_sequence", "1513_A": "ordered_sequence", "1569_A": "ordered_sequence",
    "1352_B": "unordered_set", "1016_D": "assignment", "1360_G": "assignment",
    "361_A": "assignment", "244_A": "assignment", "1323_A": "unordered_set",
    "1073_A": "ordered_sequence", "1380_A": "ordered_sequence", "1095_C": "unordered_set",
    "1352_G": "ordered_sequence", "1339_B": "ordered_sequence", "1371_D": "assignment",
    "1038_B": "unordered_partition", "545_B": "ordered_sequence", "1352_F": "ordered_sequence",
}
ADAPTER_IDS = {problem: f"wider_{problem.lower()}_v1" for problem in FAMILIES}
ADAPTER_IDS.update({"1352_B": "wider_1352_b_multiset_v1", "1095_C": "wider_1095_c_multiset_v1", "244_A": "wider_244_a_anchored_v1"})


class Cursor:
    def __init__(self, text: str):
        self.tokens = text.split()
        self.position = 0
        if len(self.tokens) > 2_000_000:
            raise ConstructiveCodeError("too many input/output tokens")

    def token(self) -> str:
        if self.position >= len(self.tokens):
            raise ConstructiveCodeError("input/output ended early")
        result = self.tokens[self.position]
        self.position += 1
        return result

    def integer(self, low: int | None = None, high: int | None = None) -> int:
        raw = self.token()
        if re.fullmatch(r"[+-]?[0-9]+", raw) is None:
            raise ConstructiveCodeError("non-integer token")
        value = int(raw)
        if low is not None and value < low or high is not None and value > high:
            raise ConstructiveCodeError("integer outside task constraints")
        return value

    def ints(self, count: int, low: int | None = None, high: int | None = None) -> list[int]:
        if not 0 <= count <= 2_000_000:
            raise ConstructiveCodeError("invalid collection length")
        return [self.integer(low, high) for _ in range(count)]

    def end(self) -> None:
        if self.position != len(self.tokens):
            raise ConstructiveCodeError("trailing input/output tokens")


def _string(cursor: Cursor, n: int, alphabet: str) -> str:
    text = cursor.token()
    if len(text) != n or any(char not in alphabet for char in text):
        raise ConstructiveCodeError("input string violates length/alphabet")
    return text


def input_cases(problem: str, text: str) -> list[dict[str, Any]]:
    from . import constructive_code_holdout_extension_20260921 as extension
    if problem in extension.FAMILIES:
        return extension.input_cases(problem, text)
    from . import constructive_code_reserved_adapters_20260921 as reserved
    if problem in reserved.FAMILIES:
        return reserved.input_cases(problem, text)
    """Parse every selected source input and check statement-level constraints."""
    c = Cursor(text)
    multicase = {"1454_A": 100, "1513_A": 100, "1569_A": 1000, "1352_B": 1000, "1360_G": 1000, "1408_A": 100, "1399_D": 20000, "1323_A": 100, "1380_A": 200, "1352_G": 100, "1339_B": 10000, "1371_D": 100, "1352_F": 1000}
    count = c.integer(1, multicase[problem]) if problem in multicase else 1
    cases = []
    for _ in range(count):
        if problem in ("1454_A", "1352_G"):
            case = {"n": c.integer(2, 100 if problem == "1454_A" else 1000)}
        elif problem == "1513_A":
            n = c.integer(1, 100); case = {"n": n, "k": c.integer(0, n)}
        elif problem == "1569_A":
            n = c.integer(1, 50); case = {"n": n, "s": _string(c, n, "ab")}
        elif problem in ("1352_B", "1095_C"):
            case = {"n": c.integer(1, 10**9), "k": c.integer(1, 100 if problem == "1352_B" else 200000)}
        elif problem == "1016_D":
            n, m = c.integer(2, 100), c.integer(2, 100)
            case = {"n": n, "m": m, "a": c.ints(n, 0, 10**9), "b": c.ints(m, 0, 10**9)}
        elif problem == "1360_G":
            n, m = c.integer(1, 50), c.integer(1, 50)
            case = {"n": n, "m": m, "a": c.integer(1, m), "b": c.integer(1, n)}
        elif problem == "361_A":
            case = {"n": c.integer(1, 100), "k": c.integer(1, 1000)}
        elif problem == "244_A":
            n, k = c.integer(1, 30), c.integer(1, 30)
            anchors = c.ints(k, 1, n * k)
            if len(set(anchors)) != k:
                raise ConstructiveCodeError("duplicate child anchor")
            case = {"n": n, "k": k, "anchors": anchors}
        elif problem in ("1102_B", "988_A"):
            high = 5000 if problem == "1102_B" else 100
            n = c.integer(1, high); k = c.integer(1, n)
            case = {"n": n, "k": k, "a": c.ints(n, 1, high)}
        elif problem == "1408_A":
            n = c.integer(3, 100)
            case = {"n": n, "a": c.ints(n, 1, 100), "b": c.ints(n, 1, 100), "c": c.ints(n, 1, 100)}
            if any(len({case['a'][i], case['b'][i], case['c'][i]}) != 3 for i in range(n)):
                raise ConstructiveCodeError("circle choices are not distinct at a position")
        elif problem == "1399_D":
            n = c.integer(1, 200000); case = {"n": n, "s": _string(c, n, "01")}
        elif problem == "1323_A":
            n = c.integer(1, 100); case = {"n": n, "a": c.ints(n, 1, 100)}
        elif problem == "1073_A":
            n = c.integer(1, 1000); case = {"n": n, "s": _string(c, n, "abcdefghijklmnopqrstuvwxyz")}
        elif problem == "1380_A":
            n = c.integer(3, 1000); p = c.ints(n, 1, n)
            if len(set(p)) != n:
                raise ConstructiveCodeError("input is not a permutation")
            case = {"n": n, "p": p}
        elif problem == "482_A":
            n = c.integer(2, 100000); case = {"n": n, "k": c.integer(1, n - 1)}
        elif problem == "1339_B":
            n = c.integer(3, 100000); case = {"n": n, "a": c.ints(n, -10**9, 10**9)}
        elif problem == "1371_D":
            n = c.integer(1, 300); case = {"n": n, "k": c.integer(0, n * n)}
        elif problem == "1038_B":
            case = {"n": c.integer(1, 45000)}
        elif problem == "1051_B":
            left = c.integer(1, 10**18); right = c.integer(left, 10**18)
            if (right - left + 1) % 2 or right - left + 1 > 300000:
                raise ConstructiveCodeError("pairing interval violates task constraints")
            case = {"l": left, "r": right}
        elif problem == "545_B":
            if len(text.splitlines()) != 2 or any(not line or set(line) - {"0", "1"} for line in text.splitlines()):
                raise ConstructiveCodeError("binary strings require exactly two nonempty binary lines")
            s = c.token()
            if not 1 <= len(s) <= 100000 or set(s) - {"0", "1"}:
                raise ConstructiveCodeError("invalid first binary string")
            case = {"s": s, "t": _string(c, len(s), "01")}
        elif problem == "1352_F":
            numbers = c.ints(3, 0, 100)
            if sum(numbers) == 0 or (numbers[1] == 0 and numbers[0] > 0 and numbers[2] > 0):
                raise ConstructiveCodeError("reconstruction input has no promised solution")
            case = {"counts": numbers}
        else:
            raise ConstructiveCodeError("unregistered wider input parser")
        cases.append(case)
    c.end()
    if problem in ("1399_D", "1339_B") and sum(case["n"] for case in cases) > (200000 if problem == "1399_D" else 100000):
        raise ConstructiveCodeError("aggregate input length exceeds problem bound")
    if problem == "1371_D" and sum(case["n"] ** 2 for case in cases) > 100000:
        raise ConstructiveCodeError("aggregate matrix size exceeds problem bound")
    return cases


def _status(c: Cursor) -> bool:
    status = c.token().upper()
    if status not in ("YES", "NO"):
        raise ConstructiveCodeError("invalid status token")
    return status == "YES"


def canonical_value(problem: str, input_text: str, output_text: str) -> Any:
    cases = input_cases(problem, input_text)
    c = Cursor(output_text)
    results = []
    for case in cases:
        if problem == "1454_A":
            value = c.ints(case["n"])
        elif problem in ("1513_A", "1352_G"):
            first = c.integer()
            value = {"status": "impossible"} if first == -1 else [first, *c.ints(case["n"] - 1)]
        elif problem == "1569_A":
            pair = c.ints(2)
            value = {"status": "impossible"} if pair == [-1, -1] else pair
        elif problem in ("1352_B", "1095_C"):
            value = {"status": "ok", "multiset": sorted(c.ints(case["k"]))} if _status(c) else {"status": "impossible"}
        elif problem == "1016_D":
            value = [c.ints(case["m"]) for _ in range(case["n"])] if _status(c) else {"status": "impossible"}
        elif problem == "1360_G":
            value = [_string(c, case["m"], "01") for _ in range(case["n"])] if _status(c) else {"status": "impossible"}
        elif problem == "361_A":
            value = [c.ints(case["n"]) for _ in range(case["n"])]
        elif problem == "244_A":
            # Child order is named by input position and MUST NOT be sorted.
            value = [sorted(c.ints(case["n"])) for _ in range(case["k"])]
        elif problem == "1323_A":
            n = c.integer()
            value = {"status": "impossible"} if n == -1 else sorted(c.ints(n))
        elif problem == "1073_A":
            value = {"status": "ok", "substring": c.token()} if _status(c) else {"status": "impossible"}
        elif problem == "1380_A":
            value = c.ints(3) if _status(c) else {"status": "impossible"}
        elif problem == "1339_B":
            value = c.ints(case["n"])
        elif problem == "1371_D":
            c.integer()  # The checker verifies this redundant optimal score.
            value = [_string(c, case["n"], "01") for _ in range(case["n"])]
        elif problem == "1038_B":
            value = sorted([sorted(c.ints(c.integer())), sorted(c.ints(c.integer()))]) if _status(c) else {"status": "impossible"}
        elif problem in ("545_B", "1352_F"):
            value = c.token()
            if problem == "545_B" and value == "impossible":
                value = {"status": "impossible"}
        else:
            raise ConstructiveCodeError("unregistered wider canonicalizer")
        results.append(value)
    c.end()
    return results


def canonicalize_task_witness(*, problem_id: str, adapter_id: str, input_data: str | bytes, output: str | bytes, decision: ReleasedCheckerDecision) -> CanonicalWitness:
    from . import constructive_code_holdout_extension_20260921 as extension
    if problem_id in extension.FAMILIES:
        return extension.canonicalize_task_witness(problem_id=problem_id, adapter_id=adapter_id, input_data=input_data, output=output, decision=decision)
    from . import constructive_code_reserved_adapters_20260921 as reserved
    if problem_id in reserved.FAMILIES:
        return reserved.canonicalize_task_witness(problem_id=problem_id, adapter_id=adapter_id, input_data=input_data, output=output, decision=decision)
    if (problem_id, adapter_id) in old.registered_task_adapters():
        # Existing tasks preserve the exact old equivalence/key contract.
        input_cases(problem_id, input_data.decode() if isinstance(input_data, bytes) else input_data)
        return old.canonicalize_task_witness(problem_id=problem_id, adapter_id=adapter_id, input_data=input_data, output=output, decision=decision)
    if ADAPTER_IDS.get(problem_id) != adapter_id:
        raise ConstructiveCodeError("unregistered task/adapter identity")
    ib, it = old._trusted_text(input_data, "input", 16 * 1024 * 1024)
    ob, ot = old._trusted_text(output, "output", 16 * 1024 * 1024)
    if not decision.accepted or decision.input_sha256 != sha256_bytes(ib) or decision.output_sha256 != sha256_bytes(ob):
        raise ConstructiveCodeError("canonicalization requires exact accepted checker binding")
    family = FAMILIES[problem_id]
    payload = {"adapter": adapter_id, "family": family, "schema_version": WITNESS_SCHEMA_VERSION, "value": canonical_value(problem_id, it, ot)}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return CanonicalWitness(family=family, checker_sha256=decision.checker_sha256, input_sha256=decision.input_sha256, output_sha256=decision.output_sha256, canonical_json=text, canonical_key=f"constructive_witness:{family}:v1:{sha256_bytes(text.encode('ascii'))}")
