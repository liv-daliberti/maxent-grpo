#!/usr/bin/env python3
"""Split-sample sign test for winner-take-all among verified modes.

The naive test (rank modes and measure their starting log-odds from the SAME
draws) is biased below 1/2 by regression to the mean: a mode over-sampled at
step t is both more likely to be ranked ahead and more likely to fall back at
t+1, with no self-reinforcement needed. Pooled over 110k pairs that artifact
alone produced ~0.48 agreement in both arms.

Here the draws at step t are split by index parity into two independent halves:

    RANK  half (even indices)  -> decides sign(q_c - q_d), i.e. which mode leads
    BASE  half (odd indices)   -> gives q_c(t), q_d(t), the starting log-odds
    step t+1 (all draws)       -> gives q_c(t+1), q_d(t+1)

Ranking noise and starting-point noise are then independent, so a deviation from
1/2 is signal. Counts are Laplace-smoothed so extinction is a defined move.

Reward filter: canonical keys are recorded for incorrect draws too, so only
draws with reward == 1.0 contribute.
"""
from __future__ import annotations
import argparse, collections, json, math
from pathlib import Path

GREEDY = "deterministic_greedy_trace_neutral"

def split_counts(records, step, smooth=0.5):
    """prompt -> (rank_counter, base_counter, all_counter) over CORRECT draws."""
    rank = collections.defaultdict(collections.Counter)
    base = collections.defaultdict(collections.Counter)
    allc = collections.defaultdict(collections.Counter)
    for r in records:
        if r.get("step") != step or r.get("evaluation_kind") == GREEDY:
            continue
        for p in r.get("prompts", []):
            keys, rew = p.get("answer_keys") or [], p.get("rewards") or []
            if len(keys) != len(rew):
                continue
            i = p["prompt_index"]
            for j, (k, w) in enumerate(zip(keys, rew)):
                if k is None or w != 1.0:
                    continue
                allc[i][k] += 1
                (rank if j % 2 == 0 else base)[i][k] += 1
    return rank, base, allc

def analyse(path, min_correct=4, smooth=0.5):
    recs = [json.loads(l) for l in open(path)]
    steps = sorted({r["step"] for r in recs})
    if len(steps) < 2:
        return None
    cache = {s: split_counts(recs, s) for s in steps}
    agree = pairs = 0
    ext_a = ext_p = 0
    for lo, hi in zip(steps[:-1], steps[1:]):
        rank, base, _ = cache[lo]
        _, _, nxt = cache[hi]
        for i, cr in rank.items():
            cb, cn = base.get(i), nxt.get(i)
            if not cb or not cn:
                continue
            nr, nb, nn = sum(cr.values()), sum(cb.values()), sum(cn.values())
            if nr < min_correct or nb < min_correct or nn < min_correct:
                continue
            ks = sorted(set(cr) | set(cb))
            if len(ks) < 2:
                continue
            K = len(ks)
            for a in range(K):
                for b in range(a + 1, K):
                    c, d = ks[a], ks[b]
                    if cr[c] == cr[d]:
                        continue              # RANK half gives no ordering
                    qc0 = (cb.get(c, 0) + smooth) / (nb + smooth * K)
                    qd0 = (cb.get(d, 0) + smooth) / (nb + smooth * K)
                    qc1 = (cn.get(c, 0) + smooth) / (nn + smooth * K)
                    qd1 = (cn.get(d, 0) + smooth) / (nn + smooth * K)
                    drift = math.log(qc1 / qd1) - math.log(qc0 / qd0)
                    if drift == 0:
                        continue
                    ok = math.copysign(1, drift) == math.copysign(1, cr[c] - cr[d])
                    agree += ok; pairs += 1
                    if cn.get(c, 0) == 0 or cn.get(d, 0) == 0:
                        ext_a += ok; ext_p += 1
    if not pairs:
        return None
    return {"pairs": pairs, "agree": agree, "rate": agree / pairs,
            "extinct_pairs": ext_p, "extinct_rate": (ext_a / ext_p) if ext_p else None}

def wilson(k, n, z=1.96):
    if not n: return (None, None)
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (c - h, c + h)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default="var/artifacts/e78_verified_replay_only_05b_jobs.json")
    ap.add_argument("--min-correct", type=int, default=4)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    runs = json.load(open(a.ledger))["runs"]
    out = []
    for r in runs:
        h = list(Path(r["run_dir"]).glob("debug_job*/eval_mode_coverage_draws.jsonl"))
        if not h: continue
        res = analyse(h[0], min_correct=a.min_correct)
        if res:
            res.update({k: r[k] for k in ("arm", "domain", "seed")})
            out.append(res)
    by = collections.defaultdict(list)
    for r in out: by[r["arm"]].append(r)
    print(f"{'arm':9} {'runs':5} {'pairs':>9} {'agree':>8} {'95% CI':>18} {'extinct':>9} {'ext_rate':>9}")
    print("-" * 70)
    for arm in sorted(by):
        rs = by[arm]
        n = sum(x["pairs"] for x in rs); k = sum(x["agree"] for x in rs)
        ep = sum(x["extinct_pairs"] for x in rs)
        ea = sum((x["extinct_rate"] or 0)*x["extinct_pairs"] for x in rs)
        lo, hi = wilson(k, n)
        print(f"{arm:9} {len(rs):<5} {n:>9} {k/max(n,1):>8.3f} {f'[{lo:.3f}, {hi:.3f}]':>18} {ep:>9} {ea/max(ep,1):>9.3f}")
    print("\n0.5 = no winner-take-all. RANK and BASE halves are disjoint draws,")
    print("so regression to the mean no longer pushes the estimate below 0.5.")
    if a.output:
        a.output.write_text(json.dumps({"runs": out}, indent=2) + "\n")
        print(f"\nwrote {a.output}")

if __name__ == "__main__":
    main()
