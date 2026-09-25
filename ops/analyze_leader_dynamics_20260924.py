#!/usr/bin/env python3
"""Leader test: does winner-take-all act on verified modes, and does replay damp it?

Theory prediction (winner-take-all / self-reinforcement): a correct mode that is
already more frequent gains relative log-odds against a rarer correct mode. This
is the prediction that separates the replay mechanism from "RL just sharpens",
because sharpening alone does not say WHICH correct mode wins.

Two tests per run, from existing evaluation logs only. No GPU, no training.

  (A) Leader retention. Per prompt, the most frequent correct mode at the early
      step; does it still lead at the final step? Compared against a
      frequency-matched null: under no self-reinforcement, the chance that mode
      c leads later is its early share q_c, so the expected retention rate is
      the mean early leader share, not 1/m.

  (B) Pair sign test. For every pair of correct modes (c,d) present early,
      does sign(delta log(q_c/q_d)) match sign(q_c - q_d)? Winner-take-all
      predicts agreement above 1/2. This is the sharper test: it uses every
      pair, not just the leader.

Reward filter: canonical keys are recorded for every draw including incorrect
ones, so keys are taken only where reward == 1.0 (see the paper's note that
unfiltered keys inflate mode counts).
"""
from __future__ import annotations
import argparse, collections, json, math, os, random
from pathlib import Path

GREEDY = "deterministic_greedy_trace_neutral"   # excluded: 1 deterministic draw

def correct_mode_counts(records, step):
    """prompt_index -> Counter(canonical_key) over CORRECT sampled draws."""
    per = collections.defaultdict(collections.Counter)
    for r in records:
        if r.get("step") != step or r.get("evaluation_kind") == GREEDY:
            continue
        for p in r.get("prompts", []):
            keys, rew = p.get("answer_keys") or [], p.get("rewards") or []
            if len(keys) != len(rew):
                continue                      # refuse to guess at misalignment
            for k, w in zip(keys, rew):
                if k is not None and w == 1.0:
                    per[p["prompt_index"]][k] += 1
    return per

def analyse_run(path, min_correct=4, smooth=0.5, mode="consecutive"):
    """Sign test on relative log-odds drift between correct modes.

    For each prompt and each ordered pair of correct modes (c,d) present at step
    t with q_c != q_d, ask whether the change in log(q_c/q_d) from t to the next
    evaluated step carries the sign of (q_c - q_d). Winner-take-all predicts
    agreement above 1/2.

    Counts are Laplace-smoothed by `smooth` so that a mode going EXTINCT is a
    defined, large move in the predicted direction. Excluding extinctions (as a
    naive filter does) removes the strongest winner-take-all evidence and biases
    the test toward 1/2.

    mode="consecutive" pools every adjacent checkpoint pair, which measures the
    local drift the theory describes and uses all 33 checkpoints. mode="endpoint"
    compares the first trained checkpoint against the last.
    """
    recs = [json.loads(l) for l in open(path)]
    steps = sorted({r["step"] for r in recs})
    if len(steps) < 2:
        return None
    counts = {s: correct_mode_counts(recs, s) for s in steps}
    if mode == "endpoint":
        span = [(steps[1], steps[-1])] if len(steps) > 2 else [(steps[0], steps[-1])]
    else:
        span = list(zip(steps[:-1], steps[1:]))
    agree = pairs = 0
    kept = tot = 0
    null_share = []
    extinct_pairs = extinct_agree = 0
    for lo, hi in span:
        a, b = counts[lo], counts.get(hi, {})
        for i, ca in a.items():
            na = sum(ca.values())
            if na < min_correct or len(ca) < 2:
                continue
            cb = b.get(i) or collections.Counter()
            nb = sum(cb.values())
            top = ca.most_common()
            # leader retention (only meaningful with a strict early leader)
            if not (len(top) > 1 and top[0][1] == top[1][1]) and nb >= min_correct:
                null_share.append(top[0][1] / na)
                tot += 1
                bt = cb.most_common()
                if bt and bt[0][0] == top[0][0] and not (len(bt) > 1 and bt[0][1] == bt[1][1]):
                    kept += 1
            if nb == 0:
                continue                       # no measurement at hi, not extinction
            ks = list(ca)
            for ci in range(len(ks)):
                for di in range(ci + 1, len(ks)):
                    c, d = ks[ci], ks[di]
                    if ca[c] == ca[d]:
                        continue
                    qc0 = (ca[c] + smooth) / (na + smooth * len(ks))
                    qd0 = (ca[d] + smooth) / (na + smooth * len(ks))
                    qc1 = (cb.get(c, 0) + smooth) / (nb + smooth * len(ks))
                    qd1 = (cb.get(d, 0) + smooth) / (nb + smooth * len(ks))
                    ok = math.copysign(1, math.log(qc1 / qd1) - math.log(qc0 / qd0)) == math.copysign(1, ca[c] - ca[d])
                    agree += ok
                    pairs += 1
                    if cb.get(c, 0) == 0 or cb.get(d, 0) == 0:
                        extinct_pairs += 1
                        extinct_agree += ok
    if not pairs and not tot:
        return None
    return {"steps": len(steps), "mode": mode, "prompts": tot,
            "leader_kept": kept, "leader_rate": (kept / tot) if tot else None,
            "null_rate": (sum(null_share) / len(null_share)) if null_share else None,
            "pairs": pairs, "pair_agree": agree,
            "pair_rate": (agree / pairs) if pairs else None,
            "extinct_pairs": extinct_pairs,
            "extinct_rate": (extinct_agree / extinct_pairs) if extinct_pairs else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default="var/artifacts/e78_verified_replay_only_05b_jobs.json")
    ap.add_argument("--min-correct", type=int, default=4)
    ap.add_argument("--mode", choices=("consecutive","endpoint"), default="consecutive")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    runs = json.load(open(a.ledger))["runs"]
    out = []
    for r in runs:
        hits = list(Path(r["run_dir"]).glob("debug_job*/eval_mode_coverage_draws.jsonl"))
        if not hits:
            continue
        res = analyse_run(hits[0], min_correct=a.min_correct, mode=a.mode)
        if res:
            res.update({k: r[k] for k in ("arm", "domain", "seed")})
            out.append(res)
    by = collections.defaultdict(list)
    for r in out:
        by[r["arm"]].append(r)
    print(f"{'arm':9} {'runs':5} {'prompts':8} {'leader':>8} {'null':>8} {'pairs':>8} {'agree':>8} {'extinct':>9} {'ext_agree':>10}")
    print("-" * 82)
    for arm in sorted(by):
        rs = by[arm]
        P = sum(x["prompts"] for x in rs); K = sum(x["leader_kept"] for x in rs)
        N = sum(x["null_rate"] * x["prompts"] for x in rs) / max(P, 1)
        pr = sum(x["pairs"] for x in rs); ag = sum(x["pair_agree"] for x in rs)
        ep = sum(x["extinct_pairs"] for x in rs)
        ea = sum((x["extinct_rate"] or 0) * x["extinct_pairs"] for x in rs)
        print(f"{arm:9} {len(rs):<5} {P:<8} {K/max(P,1):>8.3f} {N:>8.3f} {pr:>8} {ag/max(pr,1):>8.3f} "
              f"{ep:>9} {ea/max(ep,1):>10.3f}")
    print("\nleader = observed leader-retention rate; null = frequency-matched expectation;")
    print("pair_agree = P(sign(dlog odds) == sign(early gap)); 0.5 is no winner-take-all.")
    if a.output:
        a.output.write_text(json.dumps({"runs": out}, indent=2) + "\n")
        print(f"\nwrote {a.output}")

if __name__ == "__main__":
    main()
