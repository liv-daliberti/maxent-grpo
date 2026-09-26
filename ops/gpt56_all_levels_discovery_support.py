#!/usr/bin/env python3
"""Extend the immutable 160-prompt support certificate with 80 selected L1 rows.

Reads benchmark inputs and frozen executable verifiers, never model outcomes.
The L1 Pantry witness surface is the native six-bit mask, not an allocation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
SEED = 20260911


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identity(row):
    return row["level"], row["domain"], row["row_index"]


def pair_id(row):
    return f"L{row['level']}_{row['domain']}_{row['row_index']:03d}"


def audit_selection(rows, source_rows):
    """Verify exact source rows and outcome-independent first-16 SHA ordering."""
    require(len(rows) == 80 and len({identity(r) for r in rows}) == 80,
            "Expected 80 unique L1 rows")
    expected = {}
    for domain in DOMAINS:
        candidates = [r for r in source_rows if identity(r)[:2] == (1, domain)]
        require(len(candidates) == 128 and len({identity(r) for r in candidates}) == 128,
                "Expected 128 unique current native-source candidates per L1 domain")
        ranked = sorted(candidates, key=lambda r: (
            hashlib.sha256(f"{SEED}\0{r['level']}\0{r['domain']}\0{r['row_index']}".encode()).hexdigest(),
            identity(r)))
        expected.update({identity(r): r for r in ranked[:16]})
    require({identity(r) for r in rows} == set(expected), "L1 selected identities differ from frozen SHA ranking")
    for row in rows:
        require(row == expected[identity(row)], "Selected L1 row differs from current native source")


def python_witnesses(spec):
    """Construct the two externally certified return-vector programs independently."""
    cases = spec["cases"]
    require(cases and len(set(cases)) == len(cases), "Python cases must be distinct")
    divisors = [[d for d in range(2, n) if n % d == 0] for n in cases]
    require(all(divisors), "Python certificate needs a proper divisor for every case")
    result = {}
    for endpoint in (0, -1):
        outputs = [values[endpoint] for values in divisors]
        program = "lambda n:" + "".join(f"{d} if n=={n} else " for n, d in zip(cases[:-1], outputs[:-1])) + str(outputs[-1])
        result["python_factor:" + ",".join(map(str, outputs))] = "\\boxed{" + program + "}"
    require(len(result) == 2, "Expected two distinct certified Python vectors")
    digest = hashlib.sha256("\n".join(sorted(result)).encode()).hexdigest()
    require(len(result) == spec["num_externally_certified_modes"] and digest == spec["certified_mode_key_sha256"],
            "Reconstructed Python witnesses disagree with source certificate")
    return result


def certify_surface_row(row, grade, enumerate_mathir):
    """Certify finite valid witnesses without claiming total hosted support."""
    spec = json.loads(row["answer"]) if isinstance(row["answer"], str) else row["answer"]
    domain = row["domain"]
    if domain == "python_factors":
        witnesses = python_witnesses(spec)
        field = "num_externally_certified_modes"
        rationale = ("Two independently reconstructed smallest/largest proper-divisor return vectors "
                     "are executed by the frozen external Python verifier; metadata's larger num_modes "
                     "is not used as an exhaustive support claim.")
    elif domain == "mathir":
        witnesses = {v.canonical_key: "\\boxed{" + ";".join(v.action_ids) + "}"
                     for v in enumerate_mathir(spec)}
        field = "valid_mode_count"
        rationale = ("Distinct bounded action-menu state paths are enumerated by the frozen MathIR "
                     "module and each route is checked through the frozen hosted adapter. The "
                     "reference retains the conservative certified-lower-bound convention.")
        digest = hashlib.sha256("\n".join(sorted(witnesses)).encode()).hexdigest()
        require(digest == spec["valid_mode_key_sha256"], "MathIR witness keys differ from source certificate")
    elif domain == "pantry_plan":
        witnesses = {}
        for number in range(64):
            mask = f"{number:06b}"
            outcome = grade(row["level"], domain, row, mask)
            if outcome["verified"]:
                require(outcome["canonical_key"] not in witnesses, "Two distinct Pantry masks share a mode")
                witnesses[outcome["canonical_key"]] = mask
        field = "certified_mode_count"
        support_keys = sorted(key.removeprefix("pantry_plan:pantry-v1:") for key in witnesses)
        digest = hashlib.sha256("\n".join(support_keys).encode()).hexdigest()
        require(digest == spec["certified_support_sha256"], "Pantry support masks differ from source certificate")
        rationale = ("All 64 six-bit ingredient masks are passed through the original L1 trusted "
                     "quantity projection and executable verifier. Distinct successful ingredient "
                     "supports are witnessed on the actual hosted surface. The reference retains "
                     "the conservative certified-lower-bound convention.")
    else:
        raise ValueError("Unsupported L1 surface domain")
    require(len(witnesses) == spec[field] and len(witnesses) > 0, "Witness count differs from source certificate")
    records = {}
    for key, text in sorted(witnesses.items()):
        outcome = grade(row["level"], domain, row, text)
        require(outcome["verified"] and outcome["canonical_key"] == key, "Frozen hosted verifier rejects witness")
        records[key] = {"text": text, "graded_text": outcome["graded_text"], "canonical_key": key}
    keys = sorted(witnesses)
    return {"pair_id": pair_id(row), "level": row["level"], "domain": domain,
            "row_index": row["row_index"], "row_sha256": object_sha(row),
            "support_count": len(keys), "support_kind": "certified_lower_bound",
            "source_field": "answer." + field, "declared_answer_mode_count": spec[field],
            "rationale": rationale, "canonical_keys": keys,
            "key_sha256": hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
            "witnesses": records, "actual_hosted_surface_verified": True}


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_reference(rows_path, all_rows_path, source_rows_path, legacy_path, code_root, enumerator_path, prior_report_path):
    source_manifest_path = Path(source_rows_path).parent / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    require(source_manifest["model"] == "gpt-5.6-sol", "Native source is a different model")
    require(sha(source_rows_path) == source_manifest["artifact_sha256"][Path(source_rows_path).name],
            "Native source rows differ from original manifest")
    rows = read_rows(rows_path)
    all_rows = read_rows(all_rows_path)
    audit_selection(rows, read_rows(source_rows_path))
    require(len(all_rows) == 240 and len({identity(r) for r in all_rows}) == 240, "Expected 240 distinct all-level rows")
    require(Counter(identity(r)[:2] for r in all_rows) == Counter({(l, d): 16 for l in (1, 2, 3) for d in DOMAINS}),
            "All-level cohort must have 16 rows per cell")
    by_id = {pair_id(r): r for r in all_rows}
    require(all(by_id[pair_id(row)] == row for row in rows), "L1 rows differ from combined cohort")
    legacy = json.loads(Path(legacy_path).read_text())
    refs = dict(legacy["references"])
    require(len(refs) == 160 and all(r["level"] in (2, 3) for r in refs.values()), "Expected 160 original L2/L3 references")
    require(set(refs) == {pair_id(r) for r in all_rows if r["level"] in (2, 3)}, "Legacy support identities differ from retained cohort")
    prior_rows_binding = legacy["sources"]["new_rows"]
    require(sha(prior_rows_binding["path"]) == prior_rows_binding["sha256"], "Old support source rows changed")
    prior_rows = {pair_id(r): r for r in read_rows(prior_rows_binding["path"])}
    # Some old certificates bind their source row by explicit hash; others bind
    # a source rows file. Preserve every old reference object unchanged.
    for key, ref in refs.items():
        if "row_sha256" in ref:
            require(ref["row_sha256"] == object_sha(by_id[key]), "Retained row differs from old support certificate")
        else:
            require(key in prior_rows and prior_rows[key] == by_id[key], "Retained row differs from old bound source rows")
    require(not any(name.startswith("oat_drgrpo") for name in sys.modules), "Use a fresh process for frozen verifier imports")
    code_root = Path(code_root).resolve()
    code_manifest = json.loads((code_root.parent / "manifest.json").read_text())
    for name, expected in code_manifest["code_sha256"].items():
        require(sha(code_root / name) == expected, "Frozen source changed: " + name)
    for name, expected in source_manifest["code_sha256"].items():
        if name.startswith("src/oat_drgrpo/") or name == "ops/frontier_modebench_contract.py":
            require(code_manifest["code_sha256"].get(name) == expected,
                    "New verifier differs from original native hosted contract: " + name)
    require(sha(enumerator_path) == legacy["sources"]["support_helper"]["sha256"], "Graph/Countdown helper differs from frozen prior certificate")
    sys.path.insert(0, str(code_root / "src"))
    contract = load_file(code_root / "ops/frontier_modebench_contract.py", "_l1_frozen_hosted_contract")
    mathir = importlib.import_module("oat_drgrpo.mathir")
    grader = importlib.import_module("oat_drgrpo.math_grader")
    helper = load_file(enumerator_path, "_frozen_graph_countdown_support")
    require(contract.profile_metadata(1, "pantry_plan")["response_surface"] == "six_bit_ingredient_support",
            "L1 Pantry native response surface differs from six-bit mask contract")
    for row in sorted(rows, key=identity):
        if row["domain"] in ("graph_coloring", "countdown"):
            ref = helper.certify_row(row, grader.validated_modebench_outcome_key)
            ref.update(row_sha256=object_sha(row), actual_hosted_surface_verified=True)
        else:
            ref = certify_surface_row(row, contract.grade_response, mathir.enumerate_mathir_action_menu_validations)
        require(ref["pair_id"] not in refs, "Duplicate support identity")
        refs[ref["pair_id"]] = ref
        print(json.dumps({"certified": ref["pair_id"], "count": ref["support_count"], "kind": ref["support_kind"]}), flush=True)
    for name, module in list(sys.modules.items()):
        if name.startswith("oat_drgrpo") and getattr(module, "__file__", None):
            require(Path(module.__file__).resolve().is_relative_to(code_root), "Verifier import escaped frozen source: " + name)
    cells = defaultdict(list)
    for ref in refs.values():
        cells[f"level{ref['level']}/{ref['domain']}"].append(ref)
    summaries = {cell: {"prompts": len(rs), "mean_support_count": sum(r["support_count"] for r in rs) / len(rs),
                       "min_support_count": min(r["support_count"] for r in rs),
                       "max_support_count": max(r["support_count"] for r in rs),
                       "support_kinds": sorted({r["support_kind"] for r in rs})} for cell, rs in sorted(cells.items())}
    return {"schema": "gpt56-all-levels-discovery-support-v1", "status": "complete",
            "sources": {"new_rows": binding(rows_path), "all_rows": binding(all_rows_path),
                        "native_source_rows": binding(source_rows_path), "native_source_manifest": binding(source_manifest_path),
                        "legacy_support": binding(legacy_path),
                        "prior_rows": binding(prior_rows_binding["path"]), "prior_analysis_report": binding(prior_report_path),
                        "frozen_manifest": binding(code_root.parent / "manifest.json"),
                        "frozen_grader": binding(grader.__file__), "frozen_hosted_contract": binding(contract.__file__),
                        "graph_countdown_helper": binding(enumerator_path), "support_helper": binding(__file__)},
            "outcomes_read": False, "selection_from_outcomes": False, "selected_rows": len(refs),
            "selection_seed": SEED, "new_L1_rows": 80, "legacy_references_preserved_unchanged": 160,
            "references": refs, "cells": summaries,
            "support_kind_definitions": legacy["support_kind_definitions"],
            "extension_analysis_contract": legacy["extension_analysis_contract"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--all-rows", type=Path, required=True)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--legacy-support", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--enumerator-helper", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite existing support certificate")
    result = build_reference(args.rows, args.all_rows, args.source_rows, args.legacy_support, args.code_root, args.enumerator_helper, args.prior_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    copies = args.output.parent / "support_code"
    copies.mkdir(exist_ok=False)
    for path in (Path(__file__), args.enumerator_helper, ROOT / "tests/test_gpt56_all_levels_discovery_support.py"):
        shutil.copy2(path, copies / path.name)
    manifest = {"schema": "gpt56-all-levels-support-certificate-manifest-v1", "status": "complete",
                "certificate": binding(args.output), "sources": result["sources"],
                "code_copies": [binding(p) for p in sorted(copies.iterdir())],
                "verified_new_witness_count": sum(r["support_count"] for r in result["references"].values() if r["level"] == 1),
                "countdown_extra_unary_witnesses": 16, "outcomes_read": False}
    (args.output.parent / "support_certificate_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "cells": result["cells"]}, indent=2))


if __name__ == "__main__":
    main()
