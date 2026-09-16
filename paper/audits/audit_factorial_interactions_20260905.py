"""Recompute descriptive paired contrasts from frozen paper endpoint artifacts.

Run from any directory with Python's standard library:
    python paper/audits/audit_factorial_interactions_20260905.py

This reads no live metrics, scheduler state, or continuation ledgers. The only
output is the sibling JSON audit; it does not update the manuscript or figures.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("drgrpo", "maxrl", "replay_drgrpo", "replay_maxrl")
METRICS = ("pass8", "distinct8")
T975_DF4 = 2.7764451051977987
EXPECTED_SEEDS = {"qwen05b": list(range(43, 48)), "falcon1b": list(range(55, 60)), "qwen3b": list(range(70, 75))}
COEFFICIENTS = {
    "maxrl_without_replay": {"maxrl": 1, "drgrpo": -1},
    "replay_on_drgrpo": {"replay_drgrpo": 1, "drgrpo": -1},
    "replay_on_maxrl": {"replay_maxrl": 1, "maxrl": -1},
    "factorial_interaction": {"replay_maxrl": 1, "maxrl": -1, "replay_drgrpo": -1, "drgrpo": 1},
    "replay_on_maxrl_minus_maxrl_without_replay": {"replay_maxrl": 1, "maxrl": -2, "drgrpo": 1},
    "replay_maxrl_minus_replay_drgrpo": {"replay_maxrl": 1, "replay_drgrpo": -1},
}
SOURCES: dict[str, str] = {}


def load(relative: str) -> dict:
    raw = (ROOT / relative).read_bytes()
    SOURCES[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def summarize(values: dict[int, float]) -> dict:
    assert len(values) == 5, "Intervals are restricted to complete five-seed blocks."
    mean = statistics.mean(values.values())
    standard_error = statistics.stdev(values.values()) / math.sqrt(5)
    radius = T975_DF4 * standard_error
    return {"n": 5, "mean": mean, "standard_error": standard_error,
            "student_t_95_unadjusted": [mean - radius, mean + radius],
            "per_seed": values}


def analyze(campaign: str, model: str, level: int, domain: str, methods: dict, source: str) -> dict:
    expected = EXPECTED_SEEDS[model]
    available = {arm: sorted(methods[arm]) for arm in ARMS}
    common = sorted(set.intersection(*(set(available[arm]) for arm in ARMS)))
    for arm in ARMS:
        assert set(available[arm]) <= set(expected), (campaign, model, domain, arm)
        for row in methods[arm].values():
            assert all(math.isfinite(row[metric]) for metric in METRICS)
            assert 0 <= row["pass8"] <= 1
            assert row["pass8"] <= row["distinct8"] <= 8
    out = {"campaign": campaign, "model": model, "level": level, "domain": domain,
           "source": source, "expected_seeds": expected, "available_seeds_by_arm": available,
           "common_seeds": common, "n": len(common), "complete": common == expected}
    if not out["complete"]:
        out["status"] = "Excluded from interaction summary: incomplete frozen four-arm block."
        return out
    out["status"] = "Complete frozen four-arm block; post-hoc descriptive audit."
    out["contrasts"] = {
        name: {metric: summarize({seed: sum(weight * methods[arm][seed][metric]
                                           for arm, weight in weights.items())
                                  for seed in common}) for metric in METRICS}
        for name, weights in COEFFICIENTS.items()
    }
    return out


def main() -> None:
    e118_source = "paper/figures/e118_all_scale_factorial_progress.json"
    level2_source = "paper/figures/modebench_level_admission.json"
    qwen_source = "paper/results/e118_qwen05b_factorial.json"
    e118, level2, qwen = load(e118_source), load(level2_source), load(qwen_source)
    assert e118["target_step"] == level2["target_step"] == qwen["target_step"] == 3072
    blocks = []
    for model, cells in sorted(e118["cells"].items()):
        for domain, cell in sorted(cells.items()):
            methods = {}
            for arm in ARMS:
                seeds = cell["method_seeds"][arm]
                assert len(seeds) == len(set(seeds)), (model, domain, arm, "duplicate seed")
                for metric in METRICS:
                    assert len(cell["methods"][arm][metric]) == len(seeds)
                methods[arm] = {seed: {metric: cell["methods"][arm][metric][i] for metric in METRICS}
                                for i, seed in enumerate(seeds)}
                if model == "qwen05b":
                    for seed, row in methods[arm].items():
                        assert row == qwen["domains"][domain]["per_method"][arm][str(seed)]
            blocks.append(analyze("E118", model, 1, domain, methods, e118_source))
    graph = level2["partial_treatment"]
    assert graph["complete_block"] and graph["domain"] == "graph_coloring"
    methods = {arm: {int(seed): row for seed, row in graph["methods"][arm]["per_seed"].items()} for arm in ARMS}
    blocks.append(analyze("E119", "qwen05b", 2, graph["domain"], methods, level2_source))
    for domain, cell in sorted(level2["terminal_progress_by_domain"].items()):
        if domain == graph["domain"]:
            continue
        assert not cell["complete_block"], "Another complete Level-2 block needs endpoint input."
        blocks.append({"campaign": "E119", "model": "qwen05b", "level": 2, "domain": domain,
                       "source": level2_source, "expected_seeds": EXPECTED_SEEDS["qwen05b"],
                       "available_seeds_by_arm": cell["terminal_seeds_by_arm"],
                       "common_seeds": cell["four_arm_matched_seeds"], "n": cell["four_arm_n"],
                       "complete": False, "status": "Excluded: incomplete frozen four-arm block; no per-seed endpoints copied."})
    for block in blocks:
        if block["campaign"] == "E118" and block["model"] == "qwen05b":
            for name in ("replay_on_drgrpo", "replay_on_maxrl", "factorial_interaction"):
                for metric in METRICS:
                    observed = block["contrasts"][name][metric]
                    saved = qwen["domains"][block["domain"]]["contrasts"][name][metric]
                    assert observed["mean"] == saved["mean"]
                    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(observed["student_t_95_unadjusted"], saved["student_t_95"]))
    complete = [block for block in blocks if block["complete"]]
    audit = {
        "schema": "frozen-paper-paired-factorial-interaction-audit-v1",
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources_sha256": SOURCES, "target_step": 3072,
        "scope": "Frozen paper artifacts only; complete common five-seed four-arm blocks. No live run refresh or outcome-based block selection.",
        "interpretation": "Post-hoc descriptive contrasts, not additional confirmatory claims. Positive replay simple effects need not imply positive interaction. Bounded metrics can create diminishing returns near a ceiling.",
        "uncertainty": {"unit": "Paired training-seed difference after frozen prompt/draw averaging",
                        "formula": "mean +/- t_(0.975,4) * sample_stdev / sqrt(5)",
                        "t_critical": T975_DF4, "degrees_of_freedom": 4,
                        "conditional_on": "128 frozen evaluation prompts and four fixed evaluation draw streams; no resampling over new prompts or draw seeds",
                        "multiplicity": "Unadjusted approximate Student-t intervals; no p-values or simultaneous-coverage claim"},
        "contrast_coefficients": COEFFICIENTS, "complete_block_count": len(complete),
        "complete_block_count_by_campaign": {campaign: sum(b["campaign"] == campaign for b in complete) for campaign in ("E118", "E119")},
        "cross_check": "All five Qwen0.5B E118 domain endpoints and existing simple-effect/interaction summaries agree with the independently saved factorial result artifact.",
        "blocks": blocks,
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"Saved {output.relative_to(ROOT)}: {len(complete)} complete blocks.")
    for block in complete:
        summary = block["contrasts"]["factorial_interaction"]
        print(block["campaign"], block["model"], f"L{block['level']}", block["domain"],
              *(f"{metric}={summary[metric]['mean']:+.6f} {summary[metric]['student_t_95_unadjusted']}" for metric in METRICS))


if __name__ == "__main__":
    main()
