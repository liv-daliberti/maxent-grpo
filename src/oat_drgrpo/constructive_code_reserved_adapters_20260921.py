"""Audited semantic contracts for prospectively reserved larger-study problems.

These problems are not eligible for policy sampling merely because an adapter
exists. Split and source-admission manifests control their use.
"""
from __future__ import annotations

import json
from typing import Any

from .constructive_code import CanonicalWitness, ConstructiveCodeError, ReleasedCheckerDecision, WITNESS_SCHEMA_VERSION, sha256_bytes
from .constructive_code_wider_adapters_20260921 import Cursor, _string, _status
from . import constructive_code_adapters as old

FAMILIES = {
    "1023_C": "ordered_sequence", "1047_A": "unordered_set", "1088_A": "ordered_sequence",
    "1093_B": "ordered_sequence", "1325_A": "unordered_set", "1332_B": "unordered_partition",
    "1360_F": "ordered_sequence", "1430_A": "assignment", "1436_B": "assignment",
    "1450_A": "ordered_sequence", "1497_C2": "unordered_set", "1549_A": "unordered_set",
    "1559_B": "ordered_sequence", "1606_A": "ordered_sequence", "544_B": "assignment", "710_C": "assignment",
    "1096_A": "ordered_sequence", "1269_A": "ordered_sequence", "1326_A": "ordered_sequence",
    "1463_B": "ordered_sequence", "1511_B": "ordered_sequence", "1554_D": "ordered_sequence",
    "1594_A": "ordered_sequence", "1608_A": "ordered_sequence", "1617_B": "assignment",
    "199_A": "unordered_set", "472_A": "unordered_set", "534_A": "ordered_sequence",
}
ADAPTER_IDS = {problem: f"reserved_{problem.lower()}_v1" for problem in FAMILIES}
COMPOSITES = {n for n in range(4, 1001) if any(n % d == 0 for d in range(2, int(n**0.5) + 1))}


def _prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def input_cases(problem: str, text: str) -> list[dict[str, Any]]:
    c = Cursor(text)
    multi = {"1093_B": 100, "1325_A": 100, "1332_B": 1000, "1360_F": 100, "1430_A": 1000, "1436_B": 10, "1450_A": 100, "1497_C2": 10000, "1549_A": 1000, "1559_B": 100, "1606_A": 1000, "1096_A": 1000, "1326_A": 400, "1463_B": 1000, "1511_B": 285, "1554_D": 500, "1594_A": 10000, "1608_A": 100, "1617_B": 100000}
    one_integer = {"1047_A": (3, 10**9), "1088_A": (1, 100), "1325_A": (2, 10**9), "1430_A": (1, 1000), "1436_B": (2, 100), "1549_A": (5, 10**9), "710_C": (1, 49), "1269_A": (1, 10**7), "1326_A": (1, 100000), "1554_D": (1, 100000), "1594_A": (1, 10**18), "1608_A": (1, 1000), "1617_B": (10, 10**9), "199_A": (0, 10**9 - 1), "472_A": (12, 10**6), "534_A": (1, 5000)}
    count = c.integer(1, multi[problem]) if problem in multi else 1
    cases = []
    for _ in range(count):
        if problem in one_integer:
            n = c.integer(*one_integer[problem]); case = {"n": n}
            if problem == "710_C" and n % 2 != 1:
                raise ConstructiveCodeError("matrix dimension must be odd")
            if problem == "1549_A" and not _prime(n):
                raise ConstructiveCodeError("promised input prime is composite")
            if problem == "199_A":
                a, b = 0, 1
                while a < n:
                    a, b = b, a + b
                if a != n:
                    raise ConstructiveCodeError("promised input Fibonacci number is invalid")
        elif problem == "1023_C":
            n = c.integer(2, 200000); k = c.integer(2, n)
            s = _string(c, n, "()")
            balance = 0
            for char in s:
                balance += 1 if char == "(" else -1
                if balance < 0:
                    raise ConstructiveCodeError("input bracket prefix is unbalanced")
            if balance or n % 2 or k % 2:
                raise ConstructiveCodeError("input bracket lengths/balance invalid")
            case = {"n": n, "k": k, "s": s}
        elif problem in ("1093_B", "1606_A"):
            s = c.token(); alphabet = "ab" if problem == "1606_A" else "abcdefghijklmnopqrstuvwxyz"
            maximum = 100 if problem == "1606_A" else 1000
            if not 1 <= len(s) <= maximum or set(s) - set(alphabet):
                raise ConstructiveCodeError("string violates task constraints")
            case = {"s": s}
        elif problem == "1332_B":
            n = c.integer(1, 1000); values = c.ints(n, 4, 1000)
            if any(value not in COMPOSITES for value in values):
                raise ConstructiveCodeError("input array contains a noncomposite number")
            case = {"n": n, "a": values}
        elif problem == "1360_F":
            n, m = c.integer(1, 10), c.integer(1, 10)
            case = {"n": n, "m": m, "a": [_string(c, m, "abcdefghijklmnopqrstuvwxyz") for _ in range(n)]}
        elif problem in ("1450_A", "1559_B"):
            n = c.integer(1, 200 if problem == "1450_A" else 100)
            case = {"n": n, "s": _string(c, n, "BR?" if problem == "1559_B" else "abcdefghijklmnopqrstuvwxyz")}
        elif problem == "1497_C2":
            n = c.integer(3, 10**9); case = {"n": n, "k": c.integer(3, n)}
        elif problem == "544_B":
            n = c.integer(1, 100); case = {"n": n, "k": c.integer(0, n * n)}
        elif problem == "1096_A":
            left = c.integer(1, 998244353); right = c.integer(left, 998244353)
            if 2 * left > right:
                raise ConstructiveCodeError("input range violates promised existence")
            case = {"l": left, "r": right}
        elif problem == "1463_B":
            n = c.integer(2, 50); case = {"n": n, "a": c.ints(n, 1, 10**9)}
        elif problem == "1511_B":
            a, b = c.integer(1, 9), c.integer(1, 9)
            case = {"a": a, "b": b, "c": c.integer(1, min(a, b))}
        else:
            raise ConstructiveCodeError("unregistered reserved input parser")
        cases.append(case)
    c.end()
    for name, field, maximum in (("1332_B", "n", 10000), ("1497_C2", "k", 100000), ("1326_A", "n", 100000), ("1554_D", "n", 300000), ("1608_A", "n", 10000)):
        if problem == name and sum(case[field] for case in cases) > maximum:
            raise ConstructiveCodeError("aggregate case size exceeds task bound")
    if problem == "1511_B" and len({(x['a'],x['b'],x['c']) for x in cases}) != len(cases):
        raise ConstructiveCodeError("digit-length input cases must be distinct")
    return cases


def canonical_value(problem: str, input_text: str, output_text: str) -> Any:
    c = Cursor(output_text); values = []
    for case in input_cases(problem, input_text):
        if problem in ("1047_A", "199_A"):
            value = sorted(c.ints(3))
        elif problem in ("1325_A", "1549_A", "472_A"):
            value = sorted(c.ints(2))
        elif problem in ("1088_A", "1430_A"):
            first = c.integer(); count = 2 if problem == "1088_A" else 3
            value = {"status": "impossible"} if first == -1 else [first, *c.ints(count - 1)]
        elif problem == "1332_B":
            count = c.integer(1, 11); labels = c.ints(case["n"], 1, count)
            groups = [[i + 1 for i, label in enumerate(labels) if label == group] for group in range(1, count + 1)]
            if any(not group for group in groups):
                raise ConstructiveCodeError("partition uses an empty group")
            value = sorted(groups)
        elif problem in ("1436_B", "710_C"):
            value = [c.ints(case["n"]) for _ in range(case["n"])]
        elif problem == "1497_C2":
            value = sorted(c.ints(case["k"]))
        elif problem == "544_B":
            value = [_string(c, case["n"], "SL") for _ in range(case["n"])] if _status(c) else {"status": "impossible"}
        elif problem in ("1096_A", "1269_A", "1511_B", "1594_A"):
            value = c.ints(2)
        elif problem in ("1463_B", "1608_A"):
            value = c.ints(case["n"])
        elif problem == "1617_B":
            a, b, d = c.ints(3); value = {"pair": sorted([a, b]), "gcd": d}
        elif problem == "534_A":
            value = c.ints(c.integer())
        elif problem in ("1023_C", "1093_B", "1360_F", "1450_A", "1559_B", "1606_A", "1326_A", "1554_D"):
            value = c.token()
            if value == "-1":
                value = {"status": "impossible"}
        else:
            raise ConstructiveCodeError("unregistered reserved canonicalizer")
        values.append(value)
    c.end(); return values


def canonicalize_task_witness(*, problem_id: str, adapter_id: str, input_data: str | bytes, output: str | bytes, decision: ReleasedCheckerDecision) -> CanonicalWitness:
    if ADAPTER_IDS.get(problem_id) != adapter_id:
        raise ConstructiveCodeError("unregistered reserved task/adapter")
    ib, it = old._trusted_text(input_data, "input", 16 * 1024**2)
    ob, ot = old._trusted_text(output, "output", 16 * 1024**2)
    if not decision.accepted or decision.input_sha256 != sha256_bytes(ib) or decision.output_sha256 != sha256_bytes(ob):
        raise ConstructiveCodeError("canonicalization requires exact accepted checker binding")
    family = FAMILIES[problem_id]
    payload = {"adapter": adapter_id, "family": family, "schema_version": WITNESS_SCHEMA_VERSION, "value": canonical_value(problem_id, it, ot)}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return CanonicalWitness(family, decision.checker_sha256, decision.input_sha256, decision.output_sha256, text, f"constructive_witness:{family}:v1:{sha256_bytes(text.encode('ascii'))}")
