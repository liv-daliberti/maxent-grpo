#!/usr/bin/env python3
"""Measure the idealized mode-collapse claims against real Qwen2.5-0.5B runs.

Each test takes a claim from the categorical analysis and measures the sign or
count it implies for neural training. Existing E78 evaluation logs only; no
training, no GPU.

T1 ORDERING (Thm winner-take-all). Among correct modes that coexist at a
   checkpoint, does the more frequent one gain relative log-odds by the next
   checkpoint? Split-sample: the RANK half of the draws (even indices) fixes
   which mode leads, the BASE half (odd indices) fixes the starting log-odds.
   Sharing draws between the two biases the estimate below 1/2 by regression to
   the mean, which is why the naive version reads ~.48 under both objectives.

T2 EXTINCTION. When exactly one mode of a coexisting pair is absent at the next
   checkpoint, is it the rarer one? Ranking again comes from the RANK half.

T3 COEXISTENCE. Prompts carrying at least two correct modes, by checkpoint.
   This is the quantity collapse destroys, and it sets what T1/T2 condition on.

T4 TRANSIENT (Thm replay finite threshold). Under replay, does verified
   diversity dip below both endpoints before recovering?

Reward filter: canonical keys are recorded for incorrect draws too, so only
draws with reward == 1.0 contribute.
"""
from __future__ import annotations
import argparse, collections, json, math
from pathlib import Path

GREEDY = "deterministic_greedy_trace_neutral"
SMOOTH = 0.5

def halves(records, step):
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

def pcmd(counter):
    m = sum(counter.values())
    if m < 2:
        return None
    return 1 - sum(n * (n - 1) for n in counter.values()) / (m * (m - 1))

def analyse(path, min_correct=4):
    recs = [json.loads(l) for l in open(path)]
    steps = sorted({r["step"] for r in recs})
    if len(steps) < 3:
        return None
    H = {s: halves(recs, s) for s in steps}
    t1 = [0, 0]; t2 = [0, 0]
    coex = {}
    div = {}
    for s in steps:
        _, _, a = H[s]
        coex[s] = sum(1 for c in a.values() if sum(c.values()) >= min_correct and len(c) >= 2)
        vals = [pcmd(c) for c in a.values() if sum(c.values()) >= min_correct]
        vals = [v for v in vals if v is not None]
        div[s] = sum(vals) / len(vals) if vals else None
    for lo, hi in zip(steps[:-1], steps[1:]):
        rank, base, _ = H[lo]
        _, _, nxt = H[hi]
        for i, cr in rank.items():
            cb, cn = base.get(i), nxt.get(i)
            if not cb or not cn:
                continue
            nb, nn = sum(cb.values()), sum(cn.values())
            if sum(cr.values()) < min_correct or nb < min_correct or nn < min_correct:
                continue
            ks = sorted(set(cr) | set(cb))
            K = len(ks)
            if K < 2:
                continue
            for x in range(K):
                for y in range(x + 1, K):
                    c, d = ks[x], ks[y]
                    if cr[c] == cr[d]:
                        continue
                    lead_c = cr[c] > cr[d]
                    qc0 = (cb.get(c, 0) + SMOOTH) / (nb + SMOOTH * K)
                    qd0 = (cb.get(d, 0) + SMOOTH) / (nb + SMOOTH * K)
                    qc1 = (cn.get(c, 0) + SMOOTH) / (nn + SMOOTH * K)
                    qd1 = (cn.get(d, 0) + SMOOTH) / (nn + SMOOTH * K)
                    drift = math.log(qc1 / qd1) - math.log(qc0 / qd0)
                    if drift != 0:
                        t1[0] += (drift > 0) == lead_c; t1[1] += 1
                    gone_c, gone_d = cn.get(c, 0) == 0, cn.get(d, 0) == 0
                    if gone_c != gone_d:                 # exactly one died
                        t2[0] += gone_d if lead_c else gone_c
                        t2[1] += 1
    ds = [div[s] for s in steps if div[s] is not None]
    transient = None
    if len(ds) >= 3:
        transient = min(ds[1:-1]) < min(ds[0], ds[-1])
    return {"steps": len(steps),
            "t1_agree": t1[0], "t1_pairs": t1[1],
            "t2_agree": t2[0], "t2_pairs": t2[1],
            "coex_first": coex[steps[0]], "coex_last": coex[steps[-1]],
            "coex_mean": sum(coex.values()) / len(coex),
            "div_first": div[steps[0]], "div_last": div[steps[-1]],
            "transient_dip": transient}

def wilson(k, n, z=1.96):
    if not n: return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
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

    def block(rows, label):
        k1 = sum(x["t1_agree"] for x in rows); n1 = sum(x["t1_pairs"] for x in rows)
        k2 = sum(x["t2_agree"] for x in rows); n2 = sum(x["t2_pairs"] for x in rows)
        l1, h1 = wilson(k1, n1); l2, h2 = wilson(k2, n2)
        cf = sum(x["coex_first"] for x in rows) / len(rows)
        cl = sum(x["coex_last"] for x in rows) / len(rows)
        tr = sum(1 for x in rows if x["transient_dip"]); 
        print(f"{label:22} T1 {k1/max(n1,1):.3f} [{l1:.3f},{h1:.3f}] n={n1:<7} "
              f"T2 {k2/max(n2,1):.3f} [{l2:.3f},{h2:.3f}] n={n2:<6} "
              f"coex {cf:5.1f}->{cl:4.1f}  dip {tr}/{len(rows)}")

    by_arm = collections.defaultdict(list)
    for r in out: by_arm[r["arm"]].append(r)
    print("T1 = P(more frequent correct mode gains log-odds); T2 = P(the rarer mode is the one that dies)")
    print("coex = prompts with >=2 correct modes, first -> last checkpoint; dip = runs with a transient diversity minimum\n")
    for arm in sorted(by_arm):
        block(by_arm[arm], f"{arm} (all domains)")
    print()
    for arm in sorted(by_arm):
        for dom in sorted({r["domain"] for r in by_arm[arm]}):
            rows = [r for r in by_arm[arm] if r["domain"] == dom]
            block(rows, f"  {arm}/{dom}")
    if a.output:
        a.output.write_text(json.dumps({"runs": out}, indent=2) + "\n")
        print(f"\nwrote {a.output}")

if __name__ == "__main__":
    main()
