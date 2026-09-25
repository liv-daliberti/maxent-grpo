#!/usr/bin/env python3
"""Compare fixed-token HF/vLLM diagnostics; this is not a task evaluation."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics

from evaluate_real_domains_20260921 import atomic_json, object_sha, sha256
from diagnose_real_domains_vllm_20260921 import adapter_identity, validate_config


def require(value, message):
    if not value:
        raise ValueError(message)


def statistics_of(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(v) for v in values), "finite nonempty values required")
    return {"n": len(values), "mean": statistics.mean(values),
            "mean_absolute": statistics.mean(abs(v) for v in values),
            "max_absolute": max(abs(v) for v in values),
            "minimum": min(values), "maximum": max(values)}


def load_bound(directory, backend):
    directory = Path(directory).resolve()
    config = json.loads((directory / "config.json").read_text())
    validate_config(config)
    identity = json.loads((directory / "identity.json").read_text())
    result = json.loads((directory / "diagnostic.json").read_text())
    require(result["status"] == "complete", "diagnostic incomplete")
    metadata = result if backend == "hf" else result["metadata"]
    require(metadata["config_sha256"] == identity["config_sha256"] == sha256(directory / "config.json"), "config binding failed")
    for row in identity["files"]:
        require(sha256(Path(row["snapshot"])) == row["sha256"], "frozen launch bytes changed")
    runner = "diagnose_real_domains_likelihood_20260921.py" if backend == "hf" else "diagnose_real_domains_vllm_20260921.py"
    require(metadata["runner_sha256"] == sha256(directory / "bundle/ops" / runner), "executed scorer source differs")
    adapters = {c["name"]: adapter_identity(c.get("adapter"), Path(config["model"]).resolve()) for c in config["checkpoints"]}
    return config, result, adapters, {
        "directory": str(directory), "result_sha256": sha256(directory / "diagnostic.json"),
        "config_sha256": sha256(directory / "config.json"), "identity_sha256": sha256(directory / "identity.json"),
        "runner_sha256": metadata["runner_sha256"], "frozen_files_verified": len(identity["files"]),
        "adapter_seals": {k: v["seal_sha256"] if v else None for k, v in adapters.items()},
    }


def compare(hf_directory, vllm_directory):
    hc, hf, ha, hb = load_bound(hf_directory, "hf")
    vc, vl, va, vb = load_bound(vllm_directory, "vllm")
    require(hc["model"] == vc["model"] and hc["rows"] == vc["rows"], "fixed model/rows differ")
    require(hb["adapter_seals"] == vb["adapter_seals"], "checkpoint seals differ between engines")
    for name in ha:
        require((ha[name]["adapter_files"] if ha[name] else None) == (va[name]["adapter_files"] if va[name] else None), "adapter weights differ")
    rows = {row["row_id"]: row for row in hc["rows"]}
    checkpoints = [c["name"] for c in hc["checkpoints"]]
    require(checkpoints == [c["name"] for c in vc["checkpoints"]], "checkpoint order differs")
    H = {(s["checkpoint"], s["adapter_precision"], s["row_id"]): s for s in hf["scores"]}
    V = {(s["checkpoint"], s["row_id"]): s for s in vl["scores"]}
    eh = {(c, p, r) for c in checkpoints for p in (("native",) if c == "base" else ("native", "bfloat16")) for r in rows}
    ev = {(c, r) for c in checkpoints for r in rows}
    require(len(H) == len(hf["scores"]) and set(H) == eh, "HF score coverage mismatch")
    require(len(V) == len(vl["scores"]) and set(V) == ev, "vLLM score coverage mismatch")
    for key, s in H.items():
        n = len(rows[key[2]]["response_token_ids"])
        for field, total in (("token_logprobs", "sum_logprob"), ("raw_token_logprobs", "raw_sum_logprob")):
            require(len(s[field]) == n and all(math.isfinite(v) for v in s[field]), "HF response token mismatch")
            require(abs(sum(s[field]) - s[total]) < 1e-8, "HF sum mismatch")
    for key, s in V.items():
        require(s["row_sha256"] == object_sha(rows[key[1]]) and s["full_token_echo_verified"], "vLLM original token binding failed")
        require(len(s["token_logprobs"]) == len(rows[key[1]]["response_token_ids"]), "vLLM response token count differs")
        require(all(math.isfinite(v) for v in s["token_logprobs"]) and abs(sum(s["token_logprobs"]) - s["sum_logprob"]) < 1e-8, "vLLM score sum mismatch")

    def values(engine, checkpoint, row_id):
        if engine == "vllm":
            return V[checkpoint, row_id]["token_logprobs"]
        return H[checkpoint, "native" if checkpoint == "base" else engine, row_id]["raw_token_logprobs"]

    def mean(engine, checkpoint, row_id):
        return statistics.mean(values(engine, checkpoint, row_id))

    engines = ("native", "bfloat16", "vllm")
    records = []
    for row_id, original in rows.items():
        record = {k: original.get(k) for k in ("row_id", "task_id", "canonical_key", "origins")}
        record.update(response_tokens=len(original["response_token_ids"]), row_sha256=object_sha(original))
        record["raw_mean_logprob"] = {e: {c: mean(e, c, row_id) for c in checkpoints} for e in engines}
        record["checkpoint_minus_base"] = {e: {c: mean(e, c, row_id) - mean(e, "base", row_id) for c in checkpoints if c != "base"} for e in engines}
        record["remax_minus_maxrl"] = {e: {str(step): mean(e, f"remax_{step}", row_id) - mean(e, f"maxrl_{step}", row_id) for step in (16, 32)} for e in engines}
        records.append(record)
    contrasts = {
        e: {"checkpoint_minus_base": {c: statistics_of(mean(e, c, r) - mean(e, "base", r) for r in rows) for c in checkpoints if c != "base"},
            "remax_minus_maxrl": {str(step): statistics_of(mean(e, f"remax_{step}", r) - mean(e, f"maxrl_{step}", r) for r in rows) for step in (16, 32)}} for e in engines}
    drift = {}
    for first, second in (("vllm", "native"), ("vllm", "bfloat16"), ("bfloat16", "native")):
        drift[f"{first}_minus_{second}"] = {
            "raw_mean_logprob": {c: statistics_of(mean(first, c, r) - mean(second, c, r) for r in rows) for c in checkpoints},
            "raw_token_logprob": {c: statistics_of(a-b for r in rows for a,b in zip(values(first,c,r), values(second,c,r))) for c in checkpoints},
            "checkpoint_minus_base": {c: statistics_of((mean(first,c,r)-mean(first,"base",r))-(mean(second,c,r)-mean(second,"base",r)) for r in rows) for c in checkpoints if c != "base"},
            "remax_minus_maxrl": {str(step): statistics_of((mean(first,f"remax_{step}",r)-mean(first,f"maxrl_{step}",r))-(mean(second,f"remax_{step}",r)-mean(second,f"maxrl_{step}",r)) for r in rows) for step in (16,32)},
        }
    mask_differences = [a-b for s in H.values() for a,b in zip(s["token_logprobs"],s["raw_token_logprobs"])]
    return {
        "schema": "real-domains-fixed-token-engine-parity-20260921-v1", "status": "pass",
        "meaning_of_pass": "Original config/source/checkpoint/token bindings and arithmetic passed; numerical engines are NOT asserted equivalent.",
        "bindings": {"hf": hb, "vllm": vb}, "rows": len(rows), "tasks": len({r["task_id"] for r in rows.values()}),
        "normalization": "raw full model vocabulary for all engine comparisons; each row weighted equally, then per-token mean within row",
        "hf_masked_minus_raw_token_logprob": statistics_of(mask_differences),
        "hf_reported_max_inaccessible_probability": max(s["max_inaccessible_probability"] for s in H.values()),
        "contrasts": contrasts, "engine_drift": drift, "per_row": records,
        "limitations": ["Fixed bank exemplars are post-treatment discoveries, not independent held-out responses.",
                        "An exemplar likelihood is not total native mode probability or coverage.",
                        "These means and drift summaries provide no uncertainty over training seeds.",
                        "Engine and dtype differences cannot alone establish why sampled ED32 changed."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--vllm-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("comparison output already exists")
    result = compare(args.hf_dir, args.vllm_dir)
    result["analysis_source_sha256"] = sha256(Path(__file__))
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "rows": result["rows"], "output": str(args.output)}))
