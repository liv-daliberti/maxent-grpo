#!/usr/bin/env python3
"""Summarize reference-KL measurements and descriptive replay occupancy.

Terminal PCMD is a promptwise pair-collision complement computed from
verified responses. The control and replay arms use the existing 32-stream
resample; KL arms use their own disjoint evaluation draws.

Initial-reference PCMD describes the conditional target of the idealized
categorical stationary solution. It is not a bound on neural or transient
PCMD. Replay occupancy is the mean number of replayed keys per logged
update, averaged over runs. Its transform 1 - 1 / mean occupancy is a
descriptive summary, not a measured mean fixed-bank limit or a bound on
held-out neural diversity. Legacy JSON field names containing 'ceiling'
are retained for data compatibility; the paper labels these as summaries.
"""
from __future__ import annotations

import json, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
try:
    from paper_domain_typography import format_domain_names
except ModuleNotFoundError:
    from ops.paper_domain_typography import format_domain_names

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
from mode_diversity import mode_diversity  # noqa: E402

LEDGERS = ("var/artifacts/e129_drgrpo_reference_kl_05b_jobs.json",
           "var/artifacts/e129x_reference_kl_high_beta_05b_jobs.json",
           "var/artifacts/e129y_reference_kl_knee_05b_jobs.json")
RESAMPLE = "paper/results/mode_diversity_terminal_resampled_partial.json"
COST = "paper/results/replay_cost_accounting.json"
# Each replay objective has its own measured occupancy. The categorical
# small-P coefficient motivates a discovery hypothesis, not an identified
# cause of the empirical occupancy differences.
REMAX_GLOB = "var/data/xdr_qwen25_0p5b_instruct_maxrl_verified_replay_e118r2_*"
# Hosted diversity supplies additional descriptions of candidate references.
# It is not a ceiling on subsequent neural training with reference KL.
HOSTED = "paper/results/mode_diversity_hosted_cohort_20260917.json"
OUT_JSON = ROOT / "paper/results/reference_kl_comparison.json"
MACROS = ROOT / "paper/results/reference_kl_macros.tex"
TABLE = ROOT / "paper/results/reference_kl_table_body.tex"
TERMINAL_STEP = 3072
# The resample's own bar for calling a cell reportable.
MIN_DEFINED = 30
# Superscripts distinguish coefficient cells with fewer than five seeds.
SEED_TARGET = 5
LABEL = {"graph_coloring": "Graph", "countdown": "Countdown",
         "python_factors": "Python", "mathir": "MathIR", "pantry_plan": "Pantry"}
ORDER = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")


def cell(run_dir: str, step: int) -> dict | None:
    """PMD and pass@8 at one checkpoint, or None when it has not been written."""
    directory = Path(run_dir)
    if not directory.is_dir():
        return None
    jobs = [item for item in sorted(directory.iterdir()) if item.is_dir()]
    if not jobs:
        return None
    draws = jobs[-1] / "eval_mode_coverage_draws.jsonl"
    if not draws.is_file():
        return None
    pooled: dict[int, Counter] = defaultdict(Counter)
    passed: list[float] = []
    present = False
    with draws.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"step"' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row["step"]) != step:
                continue
            present = True
            for prompt in row["prompts"]:
                verified = [key for key, reward
                            in zip(prompt["answer_keys"], prompt["rewards"])
                            if key is not None and float(reward) > 0]
                pooled[int(prompt["prompt_index"])].update(verified)
                passed.append(1.0 if verified else 0.0)
    if not present:
        return None
    values = [v for v in (mode_diversity(c) for c in pooled.values()) if v is not None]
    if not values:
        return None
    return {"pmd": statistics.fmean(values), "pass8": statistics.fmean(passed),
            "defined": len(values), "prompts": len(pooled)}


def remax_occupancy() -> dict[str, float]:
    """Mean actuator modes per update on the Re:Max arm, by domain."""
    total: dict[str, list[float]] = defaultdict(list)
    for run in sorted(ROOT.glob(REMAX_GLOB)):
        name = run.name
        domain = next((d for d in ORDER if d.split("_")[0] in name), None)
        if domain is None:
            continue
        jobs = [j for j in sorted(run.iterdir()) if j.is_dir()]
        if not jobs:
            continue
        metrics = jobs[-1] / "train_metrics.jsonl"
        if not metrics.is_file():
            continue
        modes = updates = 0.0
        with metrics.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "canonical_replay_actuator_modes" not in line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = row.get("train/canonical_replay_actuator_modes")
                if isinstance(value, (int, float)):
                    modes += float(value); updates += 1
        if updates:
            total[domain].append(modes / updates)
    return {d: statistics.fmean(v) for d, v in total.items() if v}


def frozen_correctness() -> dict[str, float]:
    """Mean per-sample correctness of the frozen model, from each run's pass 0.

    This is mu(C) in Proposition "The retained conditional is the reference's",
    and it is what sets how much coefficient a domain can carry: the stationary
    correctness satisfies logit P* = logit mu(C) + c_G/beta, so the coefficient
    at which that falls to one half is beta* = c_G / (-logit mu(C)).
    """
    seen: dict[str, list[float]] = defaultdict(list)
    for name in LEDGERS:
        payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
        for run in payload["runs"]:
            directory = Path(run["run_dir"])
            if not directory.is_dir():
                continue
            jobs = [j for j in sorted(directory.iterdir()) if j.is_dir()]
            if not jobs:
                continue
            metrics = jobs[-1] / "train_metrics.jsonl"
            if not metrics.is_file():
                continue
            with metrics.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if "sampled_mean_at_8" not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("trainer/global_step") == 0 or row.get("trainer/step") == 0:
                        value = row.get("eval/multi_answer/sampled_mean_at_8")
                        if isinstance(value, (int, float)):
                            seen[run["domain"]].append(float(value))
                        break
    return {d: statistics.fmean(v) for d, v in seen.items() if v}


def main() -> int:
    terminal: dict[tuple[str, float], list[dict]] = defaultdict(list)
    frozen: dict[str, list[float]] = defaultdict(list)
    frozen_support: dict[str, list[int]] = defaultdict(list)
    frozen_pass8: dict[str, list[float]] = defaultdict(list)
    for name in LEDGERS:
        payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
        for run in payload["runs"]:
            end = cell(run["run_dir"], TERMINAL_STEP)
            if end:
                terminal[(run["domain"], float(run["beta"]))].append(end)
            start = cell(run["run_dir"], 0)
            # Initial PCMD requires at least MIN_DEFINED eligible prompts.
            # A rare defined prompt does not make the domain summary reportable.
            if start and start["defined"] >= MIN_DEFINED:
                frozen[run["domain"]].append(start["pmd"])
                frozen_support[run["domain"]].append(start["defined"])
            # Accuracy needs no support guard: pass@8 is defined on every
            # prompt, so the frozen policy's correctness is measurable even
            # where its success-conditional breadth is not.
            if start:
                frozen_pass8[run["domain"]].append(start["pass8"])

    resample = json.loads((ROOT / RESAMPLE).read_text(encoding="utf-8"))
    reference: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in resample["cells"]:
        if item["scale"] == "qwen05b":
            reference[(item["domain"], item["method"])].append(item["resampled"])

    occupancy = json.loads((ROOT / COST).read_text(encoding="utf-8"))
    occupancy = occupancy["by_family_domain"]["mean_bank_occupancy"]["Qwen2.5-0.5B"]
    remax_bank = remax_occupancy()
    mu = frozen_correctness()
    C_G = 15 / 16   # the Dr.GRPO coefficient at G = 16

    betas = sorted({b for _, b in terminal})
    rows = {}
    for domain in ORDER:
        seen = [b for b in betas if (domain, b) in terminal]
        if not seen:
            continue
        bank = occupancy[domain]["mean"]
        control = reference.get((domain, "drgrpo"), [])
        replay = reference.get((domain, "replay_drgrpo"), [])
        maxrl = reference.get((domain, "maxrl"), [])
        remax = reference.get((domain, "replay_maxrl"), [])
        remax_occ = remax_bank.get(domain)
        rows[domain] = {
            "frozen_pmd": statistics.fmean(frozen[domain]) if frozen[domain] else None,
            "frozen_defined": (statistics.fmean(frozen_support[domain])
                               if frozen_support[domain] else 0),
            "frozen_pass8": (statistics.fmean(frozen_pass8[domain])
                             if frozen_pass8[domain] else None),
            "bank_occupancy": bank,
            "replay_ceiling": 1.0 - 1.0 / bank if bank > 1 else None,
            "control": {"pmd": statistics.fmean(c["pmd"] for c in control),
                        "pass8": statistics.fmean(c["pass8"] for c in control)} if control else None,
            "replay": {"pmd": statistics.fmean(r["pmd"] for r in replay),
                       "pass8": statistics.fmean(r["pass8"] for r in replay)} if replay else None,
            "maxrl": {"pmd": statistics.fmean(m["pmd"] for m in maxrl),
                      "pass8": statistics.fmean(m["pass8"] for m in maxrl)} if maxrl else None,
            "remax": {"pmd": statistics.fmean(r["pmd"] for r in remax),
                      "pass8": statistics.fmean(r["pass8"] for r in remax)} if remax else None,
            "remax_bank_occupancy": remax_occ,
            "remax_ceiling": (1.0 - 1.0 / remax_occ) if remax_occ and remax_occ > 1 else None,
            "frozen_correctness": mu.get(domain),
            "beta_star": (C_G / -math.log(mu[domain] / (1 - mu[domain])))
                         if mu.get(domain) and 0 < mu[domain] < 0.5 else None,
            "kl": {f"{b}": {"pmd": statistics.fmean(x["pmd"] for x in terminal[(domain, b)]),
                            "pass8": statistics.fmean(x["pass8"] for x in terminal[(domain, b)]),
                            "seeds": len(terminal[(domain, b)])}
                   for b in seen},
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"generator": "ops/build_reference_kl_comparison.py",
         "estimator": "PCMD = 1 - sum_m n_m(n_m-1)/(K(K-1)) over a prompt's verified responses",
         "terminal_step": TERMINAL_STEP, "betas": betas, "domains": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    def num(value, places=3):
        return f"{value:.{places}f}" if value is not None else "---"

    lines = []
    for domain in ORDER:
        row = rows.get(domain)
        if not row:
            continue
        cells = [num(row["control"]["pmd"] if row["control"] else None),
                 num(row["maxrl"]["pmd"] if row["maxrl"] else None)]
        for b in betas:
            entry = row["kl"].get(f"{b}")
            mark = (f"$^{{{entry['seeds']}}}$"
                    if entry and entry["seeds"] < SEED_TARGET else "")
            cells.append(num(entry["pmd"] if entry else None) + mark)
        cells.append(num(row["replay"]["pmd"] if row["replay"] else None))
        cells.append(num(row["remax"]["pmd"] if row["remax"] else None))
        # Initial PCMD and two descriptive occupancy transforms.
        cells.append(num(row["frozen_pmd"], 2))
        cells.append(num(row["replay_ceiling"], 2))
        cells.append(num(row["remax_ceiling"], 2))
        cells.append(num(row["beta_star"], 2))
        lines.append(f"  {LABEL[domain]}  &  " + "  &  ".join(cells) + r" \\")
    TABLE.write_text(format_domain_names("% Generated by ops/build_reference_kl_comparison.py; do not hand edit.\n"
                     + "\n".join(lines) + "\n  \\bottomrule\n"), encoding="utf-8")

    hosted = json.loads((ROOT / HOSTED).read_text(encoding="utf-8"))
    macro = sorted((m["label"], m["macro_pmd"]) for m in hosted["models"])
    worst, best = min(macro, key=lambda x: x[1]), max(macro, key=lambda x: x[1])
    # The largest coefficient every domain carries, so anchor and memory can be
    # read on one common set rather than on whichever domains happen to have
    # the coefficient being quoted.
    shared = [b for b in betas
              if all(f"{b}" in rows[d]["kl"] for d in rows)]
    # "Matched breadth" means the coefficient whose mean PCMD is nearest the
    # memory arm's, not the largest one available: the comparison being drawn
    # is like-for-like diversity, and the largest shared coefficient need not
    # sit anywhere near it.
    remax_pmd = statistics.fmean(rows[d]["remax"]["pmd"] for d in rows)
    matched = (min(shared,
                   key=lambda b: abs(statistics.fmean(
                       rows[d]["kl"][f"{b}"]["pmd"] for d in rows) - remax_pmd))
               if shared else None)
    matched = f"{matched}" if matched is not None else None
    if matched:
        kl_pmd = statistics.fmean(rows[d]["kl"][matched]["pmd"] for d in rows)
        kl_pass = statistics.fmean(rows[d]["kl"][matched]["pass8"] for d in rows)
        rx_pmd = statistics.fmean(rows[d]["remax"]["pmd"] for d in rows)
        rx_pass = statistics.fmean(rows[d]["remax"]["pass8"] for d in rows)
        rd_pmd = statistics.fmean(rows[d]["replay"]["pmd"] for d in rows)
        rd_pass = statistics.fmean(rows[d]["replay"]["pass8"] for d in rows)
    covered = [d for d in rows if rows[d]["bank_occupancy"] >= 2.0]

    # Derive point-estimate comparison summaries from the same cells as the table.
    ARM = {"replay": "Re:Dr", "remax": "Re:Max"}

    def names(domains):
        """A domain list that reads as prose: ``A``, ``A and B``, ``A, B and C``."""
        labels = [LABEL[d] for d in domains]
        if len(labels) < 2:
            return "".join(labels)
        return ", ".join(labels[:-1]) + " and " + labels[-1]
    kl_cells = [(d, b, e) for d, r in rows.items() for b, e in r["kl"].items()]
    thin = [c for c in kl_cells if c[2]["seeds"] < SEED_TARGET]

    def best_kl(row):
        return max(row["kl"].values(), key=lambda e: e["pmd"], default=None)

    def best_memory(row):
        held = [row[a] for a in ARM if row[a]]
        return max(held, key=lambda e: e["pmd"]) if held else None

    # Classify coefficients by whether they exceed the larger replay PCMD
    # estimate and also match that arm's pass@8 estimate. These comparisons
    # do not estimate significance or equalize total computation.
    free, priced, knee = [], [], None
    for d, r in rows.items():
        if not (r["replay"] and best_memory(r)):
            continue
        rival = best_memory(r)
        over = [(b, e) for b, e in r["kl"].items() if e["pmd"] > rival["pmd"]]
        if not over:
            continue
        kept = [(b, e) for b, e in over if e["pass8"] >= rival["pass8"]]
        (free if kept else priced).append(d)
        for b, e in over:
            cost = rival["pass8"] - e["pass8"]
            if knee is None or cost > knee[3]:
                knee = (d, b, e, cost)
    # Largest KL PCMD among cells exceeding the larger replay PCMD estimate
    # without a lower pass@8 point estimate for that comparator.
    win = None
    for d in free:
        r = rows[d]
        rival = best_memory(r)
        for b, e in r["kl"].items():
            if e["pmd"] > rival["pmd"] and e["pass8"] >= rival["pass8"]:
                if win is None or e["pmd"] > win[2]["pmd"]:
                    win = (d, b, e, rival)

    # Domains no coefficient reaches: the memory arms' own cases.
    held = [d for d, r in rows.items() if r["replay"] and best_memory(r)]
    memory_only = [d for d in held if d not in free and d not in priced]
    # Smallest positive mean initial single-draw correctness. This differs
    # from pass@8 and is used only in a descriptive stationary plug-in.
    mus = [(d, r["frozen_correctness"]) for d, r in rows.items()
           if r["frozen_correctness"]]
    mu_low = min(mus, key=lambda x: x[1])

    memory_cells = [(d, a, r[a]) for d, r in rows.items() for a in ARM if r[a]]
    top_memory = max(memory_cells, key=lambda x: x[2]["pmd"])
    frozen_cells = [(d, r["frozen_pmd"]) for d, r in rows.items()
                    if r["frozen_pmd"] is not None]
    top_frozen = max(frozen_cells, key=lambda x: x[1])

    # Reportable hosted reference descriptions, with their own eligibility sets.
    hosted_cells = [c for m in hosted["models"] for c in m["cells"]
                    if c.get("reportable")]
    hosted_top = max(hosted_cells, key=lambda c: c["pmd"])
    MACROS.write_text(
        format_domain_names("% Generated by ops/build_reference_kl_comparison.py; do not hand edit.\n"
        f"\\newcommand{{\\MDklDomains}}{{{len(rows)}}}\n"
        f"\\newcommand{{\\MDklBetaCount}}{{{len(betas)}}}\n"
        f"\\newcommand{{\\MDklBetaList}}{{{', '.join(str(b) for b in betas)}}}\n"
        f"\\newcommand{{\\MDklSeedsMax}}{{{max((e['seeds'] for r in rows.values() for e in r['kl'].values()), default=0)}}}\n"
        f"\\newcommand{{\\MDklCoveredDomains}}{{{len(covered)}}}\n"
        f"\\newcommand{{\\MDklHostedModels}}{{{len(macro)}}}\n"
        f"\\newcommand{{\\MDklHostedLo}}{{{worst[1]:.2f}}}\n"
        f"\\newcommand{{\\MDklHostedHi}}{{{best[1]:.2f}}}\n"
        f"\\newcommand{{\\MDklHostedHiModel}}{{{best[0]}}}\n"
        f"\\newcommand{{\\MDklHostedRange}}{{{worst[1]:.2f}--{best[1]:.2f}}}\n"
        f"\\newcommand{{\\MDklHostedCells}}{{{len(hosted_cells)}}}\n"
        f"\\newcommand{{\\MDklHostedCellsHalf}}"
        f"{{{sum(1 for c in hosted_cells if c['pmd'] >= 0.50)}}}\n"
        f"\\newcommand{{\\MDklHostedCellMax}}{{{hosted_top['pmd']:.2f}}}\n"
        f"\\newcommand{{\\MDklSeedsMin}}"
        f"{{{min((e['seeds'] for *_, e in kl_cells), default=0)}}}\n"
        f"\\newcommand{{\\MDklThinCells}}{{{len(thin)}}}\n"
        f"\\newcommand{{\\MDklTotalCells}}{{{len(kl_cells)}}}\n"
        f"\\newcommand{{\\MDklFreeDomains}}{{{len(free)}}}\n"
        f"\\newcommand{{\\MDklFreeNames}}{{{names(free)}}}\n"
        f"\\newcommand{{\\MDklPricedDomains}}{{{len(priced)}}}\n"
        f"\\newcommand{{\\MDklPricedNames}}"
        f"{{{names(priced)}}}\n"
        f"\\newcommand{{\\MDklMemoryDomains}}{{{len(memory_only)}}}\n"
        f"\\newcommand{{\\MDklMemoryNames}}"
        f"{{{names(memory_only)}}}\n"
        f"\\newcommand{{\\MDklBetaStarLo}}"
        f"{{{min(r['beta_star'] for r in rows.values() if r['beta_star']):.2f}}}\n"
        f"\\newcommand{{\\MDklBetaStarHi}}"
        f"{{{max(r['beta_star'] for r in rows.values() if r['beta_star']):.2f}}}\n"
        f"\\newcommand{{\\MDklMuLow}}{{{mu_low[1]:.4f}}}\n"
        f"\\newcommand{{\\MDklMuLowDomain}}{{{LABEL[mu_low[0]]}}}\n"
        f"\\newcommand{{\\MDklTopMemoryArm}}{{{ARM[top_memory[1]]}}}\n"
        f"\\newcommand{{\\MDklTopMemoryDomain}}{{{LABEL[top_memory[0]]}}}\n"
        f"\\newcommand{{\\MDklTopMemoryPmd}}{{{top_memory[2]['pmd']:.3f}}}\n"
        f"\\newcommand{{\\MDklTopFrozenDomain}}{{{LABEL[top_frozen[0]]}}}\n"
        f"\\newcommand{{\\MDklTopFrozenPmd}}{{{top_frozen[1]:.2f}}}\n"
        + ("" if win is None else
           f"\\newcommand{{\\MDklWinDomain}}{{{LABEL[win[0]]}}}\n"
           f"\\newcommand{{\\MDklWinBeta}}{{{win[1]}}}\n"
           f"\\newcommand{{\\MDklWinPmd}}{{{win[2]['pmd']:.3f}}}\n"
           f"\\newcommand{{\\MDklWinPass}}{{{win[2]['pass8']:.3f}}}\n"
           f"\\newcommand{{\\MDklWinSeeds}}{{{win[2]['seeds']}}}\n"
           f"\\newcommand{{\\MDklWinRivalPmd}}{{{win[3]['pmd']:.3f}}}\n"
           f"\\newcommand{{\\MDklWinRivalPass}}{{{win[3]['pass8']:.3f}}}\n"
           f"\\newcommand{{\\MDklWinFrozenPmd}}"
           f"{{{rows[win[0]]['frozen_pmd']:.2f}}}\n")
        + ("" if knee is None else
           f"\\newcommand{{\\MDklKneeDomain}}{{{LABEL[knee[0]]}}}\n"
           f"\\newcommand{{\\MDklKneeBeta}}{{{knee[1]}}}\n"
           f"\\newcommand{{\\MDklKneePmd}}{{{knee[2]['pmd']:.3f}}}\n"
           f"\\newcommand{{\\MDklKneePass}}{{{knee[2]['pass8']:.3f}}}\n"
           f"\\newcommand{{\\MDklKneeSeeds}}{{{knee[2]['seeds']}}}\n"
           f"\\newcommand{{\\MDklKneeRivalPmd}}"
           f"{{{best_memory(rows[knee[0]])['pmd']:.3f}}}\n"
           f"\\newcommand{{\\MDklKneeRivalPass}}"
           f"{{{best_memory(rows[knee[0]])['pass8']:.3f}}}\n")
        + (f"\\newcommand{{\\MDklMatchedBeta}}{{{matched.lstrip('0')}}}\n"
           f"\\newcommand{{\\MDklMatchedDomains}}{{{len(rows)}}}\n"
           f"\\newcommand{{\\MDklMatchedKLpmd}}{{{kl_pmd:.3f}}}\n"
           f"\\newcommand{{\\MDklMatchedKLpass}}{{{kl_pass:.3f}}}\n"
           f"\\newcommand{{\\MDklMatchedRemaxPmd}}{{{rx_pmd:.3f}}}\n"
           f"\\newcommand{{\\MDklMatchedRemaxPass}}{{{rx_pass:.3f}}}\n"
           f"\\newcommand{{\\MDklMatchedRedrPmd}}{{{rd_pmd:.3f}}}\n"
           f"\\newcommand{{\\MDklMatchedRedrPass}}{{{rd_pass:.3f}}}\n"
           f"\\newcommand{{\\MDklMatchedGain}}{{{rx_pass - kl_pass:.3f}}}\n"
           if matched else "")),
        encoding="utf-8")

    print(f"wrote {OUT_JSON.relative_to(ROOT)}, {TABLE.name}, {MACROS.name}")
    hdr = f"{'domain':<10}{'E|B|':>6}{'1-1/E|B|':>10}{'frozen':>8}{'ctrl':>7}"
    hdr += "".join(f"{'b='+str(b):>8}" for b in betas)
    hdr += f"{'Re:Dr':>8}{'E|B|rm':>7}{'ceil_rm':>9}{'Re:Max':>8}"
    print(hdr)
    for domain in ORDER:
        row = rows.get(domain)
        if not row: continue
        line = f"{LABEL[domain]:<10}{row['bank_occupancy']:>6.2f}{(row['replay_ceiling'] or 0):>10.3f}"
        line += f"{(row['frozen_pmd'] or 0):>8.3f}{(row['control']['pmd'] if row['control'] else 0):>7.3f}"
        for b in betas:
            e = row["kl"].get(f"{b}")
            line += f"{e['pmd']:>8.3f}" if e else f"{'-':>8}"
        line += f"{(row['replay']['pmd'] if row['replay'] else 0):>8.3f}"
        line += f"{(row['remax_bank_occupancy'] or 0):>7.2f}{(row['remax_ceiling'] or 0):>9.3f}"
        line += f"{(row['remax']['pmd'] if row['remax'] else 0):>8.3f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
