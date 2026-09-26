#!/usr/bin/env python3
"""Reproduce the frozen E50G rewrite false-new diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import tempfile
from collections import Counter, defaultdict
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50g_full_injection_rewrite_audit_20260726.md"
)
SIGNATURE_SOURCE = (
    ROOT / "ops/math_strategy_calibration/safe_math_strategy_signatures.py"
)
INJECTIONS = (
    ROOT
    / "var/artifacts/e47_math_strategy_calibration_v1/"
    "injections.blinded.jsonl"
)
KEY = (
    ROOT
    / "var/artifacts/e47_math_strategy_calibration_v1/private/"
    "injection_key.jsonl"
)
MANIFEST = (
    ROOT / "var/artifacts/e47_math_strategy_calibration_v1/manifest.json"
)
EXPECTED = {
    "signature_source": (
        "f3455c3bd575a9c7e8cbb0b4bc1c5bad5fd7545418cca4e1a86423334aae08cc"
    ),
    "injections": (
        "4ebf26f12cc2aabdd535b02f901373e6a2b27f9a36d8e8ee48751fac6549faf5"
    ),
    "key": (
        "8163c1131f4085d20431e93c47de136ae06c7ae0eca483ec8c2e28ac7d2fd131"
    ),
    "manifest": (
        "778d656d1a79255e2b2e2f6847c6ca29a877c73f9ddcdaa3c44df565c2f820b0"
    ),
}
KINDS = ("exact_duplicate", "format_variant", "lexical_paraphrase")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise RuntimeError(f"fresh rewrite audit output required: {out}")

    observed = {
        "signature_source": _sha256(SIGNATURE_SOURCE),
        "injections": _sha256(INJECTIONS),
        "key": _sha256(KEY),
        "manifest": _sha256(MANIFEST),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"E50G rewrite audit input drift: {observed}")
    spec = importlib.util.spec_from_file_location(
        "e50g_frozen_safe_signatures", SIGNATURE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen E50G signature source")
    signatures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signatures)

    texts = {
        str(row["sample_id"]): str(row["text"])
        for row in _load_jsonl(INJECTIONS)
    }
    by_problem: dict[str, dict[str, str]] = defaultdict(dict)
    for row in _load_jsonl(KEY):
        if row.get("validator_reward") != 1:
            raise RuntimeError("E50G injection control lost exact validity")
        by_problem[str(row["problem_id"])][str(row["injection_kind"])] = str(
            row["sample_id"]
        )

    rows = []
    for problem_id in sorted(by_problem):
        family = by_problem[problem_id]
        if set(family) != {"anchor", *KINDS}:
            raise RuntimeError(f"incomplete E50G family: {problem_id}")
        anchor = {"actions": [texts[family["anchor"]]]}
        anchor_hits = signatures.signature_hits(anchor)
        anchor_signature = signatures.decisive_signature(anchor)
        for kind in KINDS:
            rewrite = {"actions": [texts[family[kind]]]}
            rewrite_hits = signatures.signature_hits(rewrite)
            rewrite_signature = signatures.decisive_signature(rewrite)
            distinct, left, right = signatures.safe_distinct_pair(
                anchor, rewrite
            )
            if left != anchor_signature or right != rewrite_signature:
                raise RuntimeError("E50G signature API inconsistency")
            rows.append(
                {
                    "problem_id": problem_id,
                    "injection_kind": kind,
                    "anchor_sample_id": family["anchor"],
                    "rewrite_sample_id": family[kind],
                    "anchor_signature_hits": list(anchor_hits),
                    "rewrite_signature_hits": list(rewrite_hits),
                    "anchor_decisive_signature": anchor_signature,
                    "rewrite_decisive_signature": rewrite_signature,
                    "signature_changed": (
                        anchor_hits != rewrite_hits
                        or anchor_signature != rewrite_signature
                    ),
                    "false_new": bool(distinct),
                }
            )

    kind_counts = Counter(row["injection_kind"] for row in rows)
    false_new = sum(row["false_new"] for row in rows)
    signature_changes = sum(row["signature_changed"] for row in rows)
    checks = {
        "exactly_fifty_problems": len(by_problem) == 50,
        "exactly_150_comparisons": len(rows) == 150,
        "exactly_fifty_per_rewrite_kind": kind_counts
        == Counter({kind: 50 for kind in KINDS}),
        "zero_false_new": false_new == 0,
        "zero_signature_changes": signature_changes == 0,
    }
    result = {
        "schema": "e50g_full_injection_rewrite_audit_v1",
        "claim_scope": (
            "retrospective frozen-code whole-cohort diagnostic; "
            "does not independently authorize training"
        ),
        "pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "problem_count": len(by_problem),
            "comparison_count": len(rows),
            "comparison_count_by_kind": dict(sorted(kind_counts.items())),
            "false_new_count": false_new,
            "signature_change_count": signature_changes,
            "single_signature_anchor_count": sum(
                signatures.decisive_signature(
                    {"actions": [texts[family["anchor"]]]}
                )
                is not None
                for family in by_problem.values()
            ),
        },
        "identity": {
            **observed,
            "protocol_sha256": _sha256(PROTOCOL),
            "script_sha256": _sha256(SCRIPT),
        },
        "comparisons": rows,
    }
    _write_json(out, result)
    print(json.dumps({key: result[key] for key in ("pass", "checks", "counts")}))


if __name__ == "__main__":
    main()
