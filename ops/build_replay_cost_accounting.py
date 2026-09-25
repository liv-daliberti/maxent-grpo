#!/usr/bin/env python3
"""Descriptive replay component accounting from retained diagnostic rows.

This estimates bank token storage and a single likelihood forward/backward
subtotal, 6|theta| T_replay, relative to a hypothetical reference forward,
2|theta| T_fresh. It does not measure total training compute or the cost
difference between replay and its zero-derivative control. The maintained
replay implementation also has a detached scoring forward, which is excluded
from this subtotal, as are generation and other system costs.

The ratio is 3 T_replay / T_fresh. Its occupancy approximation 3 E|B| / G
requires comparable sequence lengths and need not agree exactly. Fresh-group
tokens are reconstructed using a replay-derived prompt-length estimate.

The three cohorts contribute 75 replay runs (E78/E79/E80r1). scan() averages
retained logging rows without deduplicating repeated steps or truncating the
horizon; these summaries are not audited cumulative per-update workloads.
No new training or runtime profiling is performed. See the manuscript's
compute-accounting appendix for the limits of the comparison.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "var/artifacts"
OUT_JSON = ROOT / "paper/results/replay_cost_accounting.json"
MACROS = ROOT / "paper/results/replay_cost_macros.tex"
TABLE = ROOT / "paper/results/replay_cost_table_body.tex"

# The three paired cohorts, one per family; only replay arms enter this summary.
COHORTS = (
    ("e78", "Qwen2.5-0.5B", "e78_verified_replay_only_05b_jobs.json"),
    ("e79", "Falcon3-1B", "e79_falcon1b_aligned_verified_replay_jobs.json"),
    ("e80r1", "Qwen2.5-3B", "e80r1_qwen3b_aligned_verified_replay_jobs.json"),
)

DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")

# Token ids are stored as int32 in the checkpointed bank.
BYTES_PER_TOKEN = 4
# bf16 weights: the reference copy is a copy, so its size is the model's.
BYTES_PER_PARAM = 2

RT_PROMPT = "train/canonical_replay_realized_prompt_tokens"
RT_RESPONSE = "train/canonical_replay_realized_response_tokens"
RT_MODES = "train/canonical_replay_actuator_modes"
FRESH_LEN = "actor/response_tok_len"
ROLLOUTS = "train/agg_eff_rollouts"


def llama_parameter_count(config: dict) -> int:
    """Exact parameter count for a Llama-family config, for an archived run.

    Falcon3-1B's checkpoints were archived to stubs, so its resident-state cost
    is derived from the architecture rather than from a file size. The two
    cohorts whose weights survive are checked against their real byte counts.
    """
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    inter = config["intermediate_size"]
    vocab = config["vocab_size"]
    head_dim = config.get("head_dim", hidden // heads)
    attention = (
        hidden * heads * head_dim          # q
        + 2 * hidden * kv_heads * head_dim  # k, v
        + heads * head_dim * hidden        # o
    )
    mlp = 3 * hidden * inter
    per_layer = attention + mlp + 2 * hidden   # two RMSNorms
    embeddings = vocab * hidden
    if not config.get("tie_word_embeddings", False):
        embeddings += vocab * hidden
    return embeddings + layers * per_layer + hidden


def checkpoint_dir(run_dir: Path) -> Path | None:
    jobs = sorted(
        item for item in run_dir.iterdir()
        if item.is_dir() and item.name.startswith("debug_job")
    )
    for job in reversed(jobs):
        saved = job / "saved_models"
        if saved.is_dir():
            steps = sorted(item for item in saved.iterdir() if item.is_dir())
            if steps:
                return steps[-1]
    return None


def reference_bytes(run_dir: Path) -> tuple[int, str]:
    """Bytes a frozen reference copy of this policy would occupy."""
    step = checkpoint_dir(run_dir)
    if step is None:
        raise SystemExit(f"no checkpoint under {run_dir}")
    shards = sum(
        item.stat().st_size for item in step.iterdir() if item.suffix == ".safetensors"
    )
    if shards:
        return shards, "safetensors"
    index = step / "model.safetensors.index.json"
    if index.is_file():
        total = json.loads(index.read_text())["metadata"]["total_size"]
        return int(total), "safetensors-index"
    config = json.loads((step / "config.json").read_text())
    return llama_parameter_count(config) * BYTES_PER_PARAM, "config-derived"


def metrics_path(run_dir: Path) -> Path:
    jobs = sorted(
        item for item in run_dir.iterdir()
        if item.is_dir() and item.name.startswith("debug_job")
    )
    for job in reversed(jobs):
        candidate = job / "train_metrics.jsonl"
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no train_metrics.jsonl under {run_dir}")


def scan(path: Path) -> dict:
    """One streaming pass over a run's metrics.

    Lines are filtered by substring before they are parsed, because the file is
    tens of megabytes of JSON per run and only the replay rows are wanted.
    """
    prompt_tokens = response_tokens = modes = 0.0
    fresh_length_total = 0.0
    rollouts = set()
    updates = 0
    cooccurring = 0
    last_fresh: float | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if RT_PROMPT not in line and FRESH_LEN not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fresh = row.get(FRESH_LEN)
            if isinstance(fresh, (int, float)) and math.isfinite(fresh):
                last_fresh = float(fresh)
            raw_prompt = row.get(RT_PROMPT)
            if raw_prompt is None:
                continue
            if last_fresh is None:
                continue
            if fresh is not None:
                cooccurring += 1
            updates += 1
            prompt_tokens += float(raw_prompt)
            response_tokens += float(row.get(RT_RESPONSE, 0.0))
            modes += float(row.get(RT_MODES, 0.0))
            fresh_length_total += last_fresh
            group = row.get(ROLLOUTS)
            if isinstance(group, (int, float)) and math.isfinite(group):
                rollouts.add(float(group))
    if not updates or modes <= 0:
        raise SystemExit(f"no replay rows in {path}")
    return {
        "updates": updates,
        "cooccurring_rows": cooccurring,
        "group_sizes": sorted(rollouts),
        "replay_prompt_tokens_per_update": prompt_tokens / updates,
        "replay_response_tokens_per_update": response_tokens / updates,
        "mean_bank_occupancy": modes / updates,
        # One replayed row carries its prompt once, so the prompt length is the
        # replicated prompt tokens divided by the rows that replicated it.
        "prompt_length": prompt_tokens / modes,
        "exemplar_length": response_tokens / modes,
        "fresh_response_length": fresh_length_total / updates,
    }



LABELS = {
    "graph_coloring": "Graph",
    "countdown": "Countdown",
    "python_factors": "Python",
    "mathir": "MathIR",
    "pantry_plan": "Pantry",
}


def emit_paper(families, summary, cells) -> None:
    """Macros and the table body the appendix inputs; never hand edited."""
    occupancy = cells("mean_bank_occupancy")
    ratio = cells("flop_ratio")
    order = ["Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B"]
    flat = [ratio[f][d]["mean"] for f in order for d in DOMAINS]
    below = [value for value in flat if value < 1.0]
    lines = []
    for family in order:
        state = summary[family]
        gigabytes = state["reference_bytes"] / 1e9
        kilobytes = state["bank_bytes_mean"] / 1e3
        occupancies = "  &  ".join(
            f"{occupancy[family][d]['mean']:.2f}" for d in DOMAINS
        )
        ratios = "  &  ".join(f"{ratio[family][d]['mean']:.2f}" for d in DOMAINS)
        lines.append(
            f"  {family}  &  {occupancies}  &  {gigabytes:.2f}  &  "
            f"{kilobytes:.0f}  &  {state['state_ratio']:,.0f} \\\\"
        )
        lines.append(f"  \\quad ratio  &  {ratios}  &  &  &  \\\\")
    TABLE.write_text(
        "% Generated by ops/build_replay_cost_accounting.py; do not hand edit.\n"
        + "\n".join(lines)
        + "\n  \\bottomrule\n",
        encoding="utf-8",
    )
    smallest = min(summary[f]["state_ratio"] for f in order)
    largest = max(summary[f]["state_ratio"] for f in order)
    MACROS.write_text(
        "% Generated by ops/build_replay_cost_accounting.py; do not hand edit.\n"
        f"\\newcommand{{\\MDcostRuns}}{{{sum(families[f]['runs'] for f in order)}}}\n"
        f"\\newcommand{{\\MDcostOccMin}}{{{min(flat) * 16 / 3:.2f}}}\n"
        f"\\newcommand{{\\MDcostOccMax}}{{{max(flat) * 16 / 3:.2f}}}\n"
        f"\\newcommand{{\\MDcostRatioMin}}{{{min(flat):.2f}}}\n"
        f"\\newcommand{{\\MDcostRatioMax}}{{{max(flat):.2f}}}\n"
        f"\\newcommand{{\\MDcostCellsBelow}}{{{len(below)}}}\n"
        f"\\newcommand{{\\MDcostCellsTotal}}{{{len(flat)}}}\n"
        f"\\newcommand{{\\MDcostStateMin}}{{{smallest:,.0f}}}\n"
        f"\\newcommand{{\\MDcostStateMax}}{{{largest:,.0f}}}\n"
        f"\\newcommand{{\\MDcostBankKB}}{{{max(summary[f]['bank_bytes_mean'] for f in order) / 1e3:.0f}}}\n",
        encoding="utf-8",
    )


def main() -> int:
    records: list[dict] = []
    families: dict[str, dict] = {}
    for tag, family, ledger in COHORTS:
        payload = json.loads((ARTIFACTS / ledger).read_text(encoding="utf-8"))
        runs = [run for run in payload["runs"] if run["arm"] == "replay"]
        if not runs:
            raise SystemExit(f"{tag}: no replay arm in ledger")
        ref_bytes, ref_source = reference_bytes(Path(runs[0]["run_dir"]))
        families[family] = {
            "tag": tag,
            "ledger": ledger,
            "ledger_sha256": hashlib.sha256(
                (ARTIFACTS / ledger).read_bytes()
            ).hexdigest(),
            "model": payload.get("model"),
            "train_rows": int(payload["train_rows"]),
            "replay_weight": payload.get("replay_weight"),
            "reference_bytes": ref_bytes,
            "reference_bytes_source": ref_source,
            "runs": len(runs),
        }
        for run in runs:
            run_dir = Path(run["run_dir"])
            measured = scan(metrics_path(run_dir))
            group = measured["group_sizes"]
            if len(group) != 1:
                raise SystemExit(f"{run_dir}: group size not constant: {group}")
            group_size = group[0]
            replay_tokens = (
                measured["replay_prompt_tokens_per_update"]
                + measured["replay_response_tokens_per_update"]
            )
            fresh_tokens = group_size * (
                measured["prompt_length"] + measured["fresh_response_length"]
            )
            records.append({
                "family": family,
                "domain": run["domain"],
                "seed": run["seed"],
                "run_dir": str(run_dir),
                "group_size": group_size,
                **measured,
                "replay_tokens_per_update": replay_tokens,
                "fresh_tokens_per_update": fresh_tokens,
                # forward+backward on the bank against a forward on the group
                "flop_ratio": 3.0 * replay_tokens / fresh_tokens,
                # the bank holds each prompt once and one response per key
                "bank_bytes": BYTES_PER_TOKEN * families[family]["train_rows"] * (
                    measured["prompt_length"]
                    + measured["mean_bank_occupancy"] * measured["exemplar_length"]
                ),
            })
            print(
                f"  {family:<13} {run['domain']:<15} s{run['seed']} "
                f"|B|={measured['mean_bank_occupancy']:5.2f} "
                f"flop={records[-1]['flop_ratio']:.3f}",
                flush=True,
            )

    def cells(field: str) -> dict:
        table: dict[str, dict[str, dict]] = {}
        for family in families:
            table[family] = {}
            for domain in DOMAINS:
                values = [
                    r[field] for r in records
                    if r["family"] == family and r["domain"] == domain
                ]
                if values:
                    table[family][domain] = {
                        "mean": statistics.fmean(values),
                        "min": min(values),
                        "max": max(values),
                        "seeds": len(values),
                    }
        return table

    summary = {}
    for family, meta in families.items():
        rows = [r for r in records if r["family"] == family]
        bank = statistics.fmean(r["bank_bytes"] for r in rows)
        summary[family] = {
            "mean_bank_occupancy": statistics.fmean(
                r["mean_bank_occupancy"] for r in rows
            ),
            "flop_ratio_mean": statistics.fmean(r["flop_ratio"] for r in rows),
            "flop_ratio_min": min(r["flop_ratio"] for r in rows),
            "flop_ratio_max": max(r["flop_ratio"] for r in rows),
            "bank_bytes_mean": bank,
            "reference_bytes": meta["reference_bytes"],
            "state_ratio": meta["reference_bytes"] / bank,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "generator": "ops/build_replay_cost_accounting.py",
                "families": families,
                "domains": list(DOMAINS),
                "bytes_per_token": BYTES_PER_TOKEN,
                "runs": records,
                "by_family_domain": {
                    "mean_bank_occupancy": cells("mean_bank_occupancy"),
                    "flop_ratio": cells("flop_ratio"),
                    "exemplar_length": cells("exemplar_length"),
                    "fresh_response_length": cells("fresh_response_length"),
                },
                "by_family": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    emit_paper(families, summary, cells)
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)} from {len(records)} replay runs")
    print(f"wrote {MACROS.relative_to(ROOT)} and {TABLE.relative_to(ROOT)}")
    for family, row in summary.items():
        print(
            f"  {family:<13} E|B|={row['mean_bank_occupancy']:.2f} "
            f"flop x{row['flop_ratio_mean']:.3f} "
            f"state {row['state_ratio']:.0f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
