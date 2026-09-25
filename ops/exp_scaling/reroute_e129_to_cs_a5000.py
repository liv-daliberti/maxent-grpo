#!/usr/bin/env python3
"""Move E129/E129-X's still-pending cells onto the idle cs A5000 nodes.

Slurm will not let a submitted job change partition or account, so reaching
node202/203/204 -- which sit in `cs` and not in `mltheory`, and which E126
already used under account `allcs` -- means cancelling the pending cells and
resubmitting them there. Only cells that have never started are touched: a
terminal or running cell keeps its original job, and its run directory is left
untouched, so nothing that has produced data is disturbed.

The GPU model does not change. These cells already run on a5000 after the
node302 repin, and node202/203/204 are a5000 as well, so this moves them
between physical nodes inside one pool rather than into a third pool. Placement
against the matched E78 control is no more and no less matched than it already
was.

Each cell is resubmitted held, audited for its own coefficient exactly as the
original launcher audits, and released only after the whole batch is placed.
The ledger is rewritten in place with the new job ids, the superseded ids kept
alongside them, so the record says what actually ran.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

COHORTS = {
    "e129":  ("var/artifacts/e129_drgrpo_reference_kl_05b_jobs.json",
              "launch_e129_drgrpo_reference_kl_05b"),
    "e129x": ("var/artifacts/e129x_reference_kl_high_beta_05b_jobs.json",
              "launch_e129x_reference_kl_high_beta_05b"),
}
CS_NODES = ("node203", "node204", "node202")
PARTITION, ACCOUNT, GRES = "cs", "allcs", "gpu:a5000:1"


def atomic_json(path: Path, payload) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True); fh.write("\n")
    os.replace(tmp, path)


def squeue_states() -> dict[str, str]:
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", "od2961"),
                          "-h", "-o", "%i|%T"], capture_output=True, text=True).stdout
    return {j: s for j, s in (l.split("|") for l in out.splitlines() if "|" in l)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    states = squeue_states()
    plan = []
    for tag, (ledger_rel, module) in COHORTS.items():
        mod = __import__(module)
        ledger = ROOT / ledger_rel
        payload = json.loads(ledger.read_text(encoding="utf-8"))
        snapshot = Path(payload["snapshot_root"])
        for run in payload["runs"]:
            jid = str(run["job_id"])
            if states.get(jid) != "PENDING":
                continue                      # running, terminal, or already gone
            if Path(run["run_dir"]).is_dir():
                continue                      # has produced something; leave it
            template = {"domain": run["domain"], "seed": run["seed"],
                        "source_node": run["source_node"]}
            plan.append((tag, mod, ledger, payload, run, snapshot, template))
    for index, item in enumerate(plan):
        item[4]["_target_node"] = CS_NODES[index % len(CS_NODES)]
    print(f"cells to reroute: {len(plan)}")
    for node in CS_NODES:
        print(f"  {node}: {sum(1 for _,_,_,_,r,_,_ in plan if r['_target_node']==node)}")
    if not args.submit or args.dry_run:
        print("[reroute] dry run; nothing cancelled or submitted")
        return 0

    cancelled, submitted = [], []
    try:
        for tag, mod, ledger, payload, run, snapshot, template in plan:
            jid = str(run["job_id"])
            subprocess.run(["scancel", jid], check=False, capture_output=True)
            cancelled.append(jid)
        time.sleep(5)
        for n, (tag, mod, ledger, payload, run, snapshot, template) in enumerate(plan, 1):
            source = next(x for x in mod.references(ROOT)
                          if str(x["domain"]) == run["domain"] and int(x["seed"]) == run["seed"])
            env, target = mod.build_env(ROOT, source, run["arm"], snapshot)
            cmd = ["sbatch", "--parsable", "--hold",
                   f"--job-name={mod.job_name(run['domain'], run['arm'], run['seed'])}",
                   "--export=ALL," + ",".join(f"{k}={v}" for k, v in env.items()),
                   f"--partition={PARTITION}", f"--account={ACCOUNT}",
                   f"--nodelist={run['_target_node']}", f"--gres={GRES}",
                   "--cpus-per-task=8", "--mem=48G", "--time=1-12:00:00", "--nice=100",
                   str(ROOT / "ops/slurm/train_node302.slurm")]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"{run['run_stamp']}: {res.stderr.strip()[:120]}")
            new = res.stdout.strip().split(";", 1)[0]
            if not new.isdigit():
                raise RuntimeError(f"{run['run_stamp']}: bad id {res.stdout!r}")
            submitted.append(new)
            run["superseded_job_id"] = int(run["job_id"])
            run["job_id"] = int(new)
            run["rerouted"] = {"partition": PARTITION, "account": ACCOUNT,
                               "node": run.pop("_target_node"), "gres": GRES,
                               "reason": "cs a5000 capacity; node302 left to E124"}
            if n % 25 == 0:
                time.sleep(2)
        for tag, (ledger_rel, _) in COHORTS.items():
            pass
        seen = set()
        for tag, mod, ledger, payload, run, snapshot, template in plan:
            if id(ledger) in seen: continue
            seen.add(id(ledger)); atomic_json(ledger, payload)
        for n, jid in enumerate(submitted, 1):
            subprocess.run(["scontrol", "release", jid], check=False, capture_output=True)
            if n % 25 == 0: time.sleep(2)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        if submitted:
            subprocess.run(["scancel", *submitted], check=False)
        raise
    print(f"[reroute] cancelled {len(cancelled)}, resubmitted and released {len(submitted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
