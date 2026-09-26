#!/usr/bin/env python3
"""Monitor the 4-arm x 5-seed real-domains coding campaign (2026-09-24).

This campaign lives in var/artifacts/real_domains_pilot_20260921/, not in the
exp_scaling cohort registry, so ops/exp_scaling/campaign_stats.py cannot see it
(same situation E124 was in). Reads run directories plus sacct; no GPU, no
cluster mutation. Safe to run repeatedly, but sacct is one call, so respect the
60s floor between invocations.
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

EXE = Path("/n/fs/similarity/maxent-grpo/var/artifacts/real_domains_pilot_20260921/code_4arm_execution_20260924")
UPDATES = 128
# arm -> (maxrl_task_objective, replay applied). canonical_replay_compute_only
# is the recorded discriminator for the replay factor; the objective factor is
# NOT recoverable from the result.json "objective" receipt, whose "fresh" string
# is hardcoded to binary_maxrl_advantages for every arm. Key on `arm` instead.
MATRIX = {"maxrl": (True, False), "remax": (True, True),
          "drgrpo": (False, False), "redr": (False, True)}

def sacct():
    try:
        out = subprocess.run(["sacct", "-X", "-n", "-P", "-S", "2026-09-24",
                              "--format=JobIDRaw,JobName,State,Elapsed,NodeList"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    d = {}
    for line in out.strip().splitlines():
        f = line.split("|")
        if len(f) >= 5 and f[1].startswith("code4arm"):
            d[f[0]] = {"name": f[1], "state": f[2], "elapsed": f[3], "node": f[4]}
    return d

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()
    jobs, rows = sacct(), []
    for d in sorted(EXE.glob("training_*")):
        if not d.is_dir() or d.name == "training_requests" or d.name.endswith("_work"):
            continue   # _work trees are per-run sandboxes, not runs
        arm, seed = d.name.replace("training_", "").rsplit("_s", 1)
        r = {"arm": arm, "seed": int(seed), "dir": d.name,
             "job": None, "state": "NOT_SUBMITTED", "elapsed": "-", "node": "",
             "updates": 0, "bank_modes": None, "compute_only": None, "replay_ok": None}
        sub = d / "submission.json"
        if sub.exists():
            j = json.loads(sub.read_text()).get("job_id")
            if j is not None:
                r["job"] = j
                r.update({k: jobs.get(str(j), {}).get(k, r[k]) for k in ("state", "elapsed", "node")})
        m = d / "training" / "metrics.jsonl"
        if m.exists():
            last = None
            with m.open() as fh:
                for n, line in enumerate(fh, 1):
                    last = line
                r["updates"] = n if last else 0
            if last:
                try:
                    rec = json.loads(last)
                    r["bank_modes"] = rec.get("bank_modes")
                    co = rec.get("canonical_replay_compute_only")
                    r["compute_only"] = co
                    if co is not None:                 # expected: replay applied <=> compute_only == 0
                        r["replay_ok"] = (co == 0.0) == MATRIX[arm][1]
                except Exception:
                    pass
        if (d / "training" / "result.json").exists():
            r["state"] = r["state"] if r["state"] not in ("", "NOT_SUBMITTED") else "COMPLETED"
        rows.append(r)
    if args.json:
        print(json.dumps(rows, indent=2)); return
    print(f"4-arm x 5-seed coding campaign  ({EXE.name})\n")
    print(f"{'arm':7} {'seed':6} {'job':10} {'state':11} {'elapsed':9} {'node':8} {'updates':>9}  {'bank':>5}  replay")
    print("-" * 82)
    for r in sorted(rows, key=lambda x: (x["arm"], x["seed"])):
        flag = {True: "ok", False: "MISMATCH", None: "-"}[r["replay_ok"]]
        print(f"{r['arm']:7} {r['seed']:<6} {str(r['job']):10} {r['state']:11} {r['elapsed']:9} "
              f"{r['node'] or '-':8} {r['updates']:>4}/{UPDATES:<4} {str(r['bank_modes'] or '-'):>5}  {flag}")
    print()
    by = {}
    for r in rows:
        by[r["state"]] = by.get(r["state"], 0) + 1
    print("states :", ", ".join(f"{k}={v}" for k, v in sorted(by.items())))
    done = sum(r["updates"] for r in rows)
    print(f"updates: {done}/{len(rows)*UPDATES} ({100*done/max(len(rows)*UPDATES,1):.1f}%)")
    bad = [f"{r['arm']}/s{r['seed']}" for r in rows if r["replay_ok"] is False]
    if bad:
        print("REPLAY FACTOR MISMATCH:", ", ".join(bad))

if __name__ == "__main__":
    main()
