#!/usr/bin/env python3
"""Submit E119: the complete Qwen-0.5B Level-2 Dr.GRPO/MaxRL/replay factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_b3a_replay_ablation as base
import launch_e78_verified_replay_only_05b as e78


DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
ARMS = ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
SEEDS = (43, 44, 45, 46, 47)
PASSES = 8
TRAIN_ROWS = 384
EVAL_ROWS = 128
TARGET_STEPS = PASSES * TRAIN_ROWS
CHECKPOINT_INTERVAL = 192
REPLAY_WEIGHT = 0.10
DATA_ROOT = "var/data/modebench_harder_v2_matched_r5"
IDENTITY_SHA256 = "2af80f4d31a44482574b314cc37ef84bde48c53ecc2ff78dadf97571f0d73fb2"
BASELINE_REPORT = f"{DATA_ROOT}/admission_fairness_report.json"
REPEAT_REPORT = "var/results/modebench_level2_r5_frozen_repeat1/admission_fairness_report.json"
PROTOCOL = "paper/preregistration/e119_level2_qwen05b_factorial_20260901.md"
LEDGER = "var/artifacts/e119_level2_qwen05b_factorial_jobs.json"
MODEL_TAG = "qwen25_0p5b_instruct"
DOMAIN_DIR = {"pantry_plan": "pantry"} | {domain: domain for domain in DOMAINS if domain != "pantry_plan"}
DOMAIN_TAGS = {"graph_coloring": "graph", "countdown": "countdown", "python_factors": "python", "mathir": "mathir", "pantry_plan": "pantry"}
PROMPTS = {
    "graph_coloring": "qwen_boxed",
    "countdown": "qwen_level2_countdown",
    "python_factors": "qwen_level2_python_factors",
    "mathir": "qwen_level2_mathir",
    "pantry_plan": "qwen_level2_pantry",
}
SYNTAX = {
    "graph_coloring": "none",
    "countdown": "countdown_legal_v3",
    "python_factors": "domain_legal_v1",
    "mathir": "domain_legal_v1",
    "pantry_plan": "domain_legal_v1",
}
VARIANTS = {
    "drgrpo": "grpo_compute_matched",
    "replay_drgrpo": "verified_first_replay_rehearsal_only",
    "maxrl": "maxrl_compute_matched",
    "replay_maxrl": "maxrl_verified_replay",
}


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admitted(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass" or report.get("decision") != "admit_all_domains_for_treatment_training":
        raise SystemExit(f"E119 requires an all-admit report: {path}")
    if set(report.get("domain_decisions", {}).values()) != {"admit"}:
        raise SystemExit(f"E119 report has a non-admitted domain: {path}")


def templates(repo: Path) -> list[dict[str, Any]]:
    runs = e78.references(repo)
    selected = [run for run in runs if str(run["domain"]) in DOMAINS and int(run["seed"]) in SEEDS]
    if len(selected) != len(DOMAINS) * len(SEEDS):
        raise SystemExit("E119 requires exactly 25 E78 schedule templates")
    return selected


def objective(arm: str) -> dict[str, str]:
    replay_live = arm in {"replay_drgrpo", "replay_maxrl"}
    maxrl = arm in {"maxrl", "replay_maxrl"}
    return {
        "OAT_ZERO_VARIANT": VARIANTS[arm],
        "OAT_ZERO_MAXRL_TASK_OBJECTIVE": "1" if maxrl else "0",
        "OAT_ZERO_CRITIC_TYPE": "drgrpo",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": "0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE": "verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA": repr(REPLAY_WEIGHT),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA": repr(REPLAY_WEIGHT),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY": "16",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0" if replay_live else "1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY": "0",
        "OAT_ZERO_MAXENT_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_DUAL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_INVERSE_ADAPTATION": "0",
        "OAT_ZERO_POLICY_ENTROPY_COEF": "0.0",
        "OAT_ZERO_SEED_ENTROPY_ALPHA": "0.0",
        "OAT_ZERO_BETA": "0.0",
        "OAT_ZERO_DAPO_ENABLED": "0",
        "OAT_ZERO_RLEP_REPLAY_COUNT": "0",
        "OAT_ZERO_DIAYN_NUM_OPTIONS": "0",
    }


def run_stamp(domain: str, arm: str, seed: int) -> str:
    return f"e119_level2_{DOMAIN_TAGS[domain]}_{arm}_s{seed}"


def save_path(repo: Path, domain: str, arm: str, seed: int) -> Path:
    return repo / "var/data" / f"xdr_{MODEL_TAG}_{VARIANTS[arm]}_{run_stamp(domain, arm, seed)}"


def environment(repo: Path, template: dict[str, Any], arm: str, snapshot: Path) -> tuple[dict[str, str], Path]:
    domain = str(template["domain"]); seed = int(template["seed"])
    target = save_path(repo, domain, arm, seed)
    env = base.build_export_vars(repo, template, target, "b1b")
    data = repo / DATA_ROOT / DOMAIN_DIR[domain]
    env.update({
        "SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, arm, seed),
        "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"), "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
        "OAT_ZERO_PROMPT_DATA": str(data / "train"), "OAT_ZERO_EVAL_DATA": str(data / "eval"),
        "OAT_ZERO_PROMPT_TEMPLATE": PROMPTS[domain], "OAT_ZERO_TEST_SPLIT": "multi_answer",
        "OAT_ZERO_MODEBENCH_DOMAIN": domain, "OAT_ZERO_MODEBENCH_SYNTAX_PROFILE": SYNTAX[domain],
        "OAT_ZERO_PROMPT_MAX_LENGTH": "1024", "OAT_ZERO_GENERATE_MAX_LENGTH": "192",
        "OAT_ZERO_EVAL_GENERATE_MAX_LENGTH": "192", "OAT_ZERO_MAX_MODEL_LEN": "2048",
        "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS), "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
        "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES), "OAT_ZERO_NUM_PPO_EPOCHS": "1",
        "OAT_ZERO_LEARNING_RATE": repr(2e-7), "OAT_ZERO_BETA": "0.0",
        "OAT_ZERO_NUM_SAMPLES": "16", "OAT_ZERO_TEMPERATURE": "1.0", "OAT_ZERO_TOP_P": "1.0",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL), "OAT_ZERO_EVAL_MODE_COVERAGE_K": "8",
        "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE": "1.0", "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": "4",
        "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL), "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
        "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL), "OAT_ZERO_MAX_SAVE_NUM": "2",
        "OAT_ZERO_MAX_RESUME_NUM": "1", "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS": "1",
        "OAT_ZERO_AUTO_RESUME": "1", "OAT_ZERO_WATCHDOG_REQUEUE": "1", "OAT_ZERO_USE_WB": "0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_USE_V1": "0",
    })
    env.update(objective(arm))
    return env, target


def command(repo: Path, template: dict[str, Any], arm: str, env: dict[str, str]) -> list[str]:
    domain = str(template["domain"]); seed = int(template["seed"])
    short = {"drgrpo":"d", "replay_drgrpo":"rd", "maxrl":"m", "replay_maxrl":"rm"}[arm]
    exports = ",".join(f"{key}={value}" for key, value in env.items())
    return ["sbatch", "--parsable", "--hold", f"--job-name=e119-{DOMAIN_TAGS[domain][:6]}-{short}-s{seed}",
            f"--export=ALL,{exports}", "--partition=mltheory", "--account=mltheory", "--nodelist=node105",
            "--gres=gpu:a5000:1", "--cpus-per-task=8", "--mem=64G", "--time=1-12:00:00", "--nice=0",
            "--requeue", "--chdir=" + str(repo), str(Path(env["OAT_ZERO_OPS_SNAPSHOT_ROOT"]) / "slurm/train_node302.slurm")]


def audit(job_id: str, cell: dict[str, Any]) -> str:
    result = subprocess.run(["scontrol", "show", "job", "-dd", "-o", job_id], capture_output=True, text=True, check=False)
    if result.returncode: raise RuntimeError(result.stderr.strip())
    record = result.stdout; arm = str(cell["arm"]); domain = str(cell["domain"])
    required = ("JobState=PENDING", "Reason=JobHeldUser", "Account=mltheory", "ReqNodeList=node105",
                f"OAT_ZERO_PROMPT_DATA={root()/DATA_ROOT/DOMAIN_DIR[domain]/'train'}",
                f"OAT_ZERO_EVAL_DATA={root()/DATA_ROOT/DOMAIN_DIR[domain]/'eval'}",
                f"OAT_ZERO_MODEBENCH_SYNTAX_PROFILE={SYNTAX[domain]}", f"OAT_ZERO_PROMPT_TEMPLATE={PROMPTS[domain]}",
                f"OAT_ZERO_MAXRL_TASK_OBJECTIVE={'1' if arm in {'maxrl','replay_maxrl'} else '0'}",
                f"OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY={'0' if arm in {'replay_drgrpo','replay_maxrl'} else '1'}",
                "OAT_ZERO_NUM_SAMPLES=16", "OAT_ZERO_NUM_PROMPT_EPOCH=8", "OAT_ZERO_MAX_TRAIN=384",
                "OAT_ZERO_LEARNING_RATE=2e-07", "OAT_ZERO_BETA=0.0", f"OAT_ZERO_SEED={cell['seed']}")
    missing = [value for value in required if value not in record]
    if missing: raise RuntimeError(f"held E119 job {job_id} lacks {missing}")
    return record


def main() -> int:
    repo = root(); parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path); args = parser.parse_args()
    if args.submit and args.dry_run: raise SystemExit("choose --submit or --dry-run")
    identity = repo / DATA_ROOT / "identity.json"; protocol = repo / PROTOCOL; ledger = repo / LEDGER
    required = (identity, protocol, repo / BASELINE_REPORT, repo / REPEAT_REPORT)
    if any(not path.is_file() for path in required): raise SystemExit(f"E119 missing required input: {[str(p) for p in required if not p.is_file()]}")
    if digest(identity) != IDENTITY_SHA256: raise SystemExit("E119 Level-2 identity drift")
    admitted(repo / BASELINE_REPORT); admitted(repo / REPEAT_REPORT)
    if args.submit and ledger.exists(): raise SystemExit(f"refusing duplicate E119 submission: {ledger}")
    snapshot = e78.snapshot_util.ensure_snapshot(repo, args.snapshot_root)
    planned = []
    for template in templates(repo):
        for arm in ARMS:
            env, target = environment(repo, template, arm, snapshot)
            if target.exists(): raise SystemExit(f"refusing existing E119 run directory: {target}")
            planned.append({"domain":str(template["domain"]), "arm":arm, "seed":int(template["seed"]),
                            "run_stamp":run_stamp(str(template["domain"]),arm,int(template["seed"])),
                            "run_dir":str(target), "command":command(repo,template,arm,env)})
    if len(planned) != 100: raise SystemExit(f"E119 expected 100 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned: print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(f"[e119] dry_run=True cells=100 snapshot={snapshot}"); return 0
    submitted=[]; records=[]
    try:
        for cell in planned:
            result=subprocess.run(cell["command"],capture_output=True,text=True,check=False)
            if result.returncode: raise RuntimeError(f"submission failed for {cell['run_stamp']}: {result.stderr.strip()}")
            job_id=result.stdout.strip().split(";",1)[0]
            if not job_id.isdigit(): raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id); held=audit(job_id,cell)
            records.append({key:cell[key] for key in ("domain","arm","seed","run_stamp","run_dir")} | {"job_id":int(job_id),"held_scheduler_record":held})
        payload={"schema":"e119_level2_qwen05b_factorial_jobs_v1","protocol":str(protocol),"protocol_sha256":digest(protocol),
                 "launcher":str(Path(__file__).resolve()),"launcher_sha256":digest(Path(__file__)),"identity":str(identity),
                 "identity_sha256":digest(identity),"baseline_admission_report":str(repo/BASELINE_REPORT),
                 "baseline_admission_sha256":digest(repo/BASELINE_REPORT),"repeat_admission_report":str(repo/REPEAT_REPORT),
                 "repeat_admission_sha256":digest(repo/REPEAT_REPORT),"snapshot_root":str(snapshot),
                 "model":"Qwen2.5-0.5B-Instruct","domains":list(DOMAINS),"arms":list(ARMS),"seeds":list(SEEDS),
                 "train_rows":TRAIN_ROWS,"eval_rows":EVAL_ROWS,"passes":PASSES,"target_steps":TARGET_STEPS,
                 "checkpoint_interval_steps":CHECKPOINT_INTERVAL,"replay_weight":REPLAY_WEIGHT,"runs":records,
                 "released":False,"outcomes_inspected_before_release":False}
        e78.atomic_json(ledger,payload)
        for job_id in submitted:
            release=subprocess.run(["scontrol","release",job_id],capture_output=True,text=True,check=False)
            if release.returncode: raise RuntimeError(f"release failed for {job_id}: {release.stderr.strip()}")
        payload["released"]=True; e78.atomic_json(ledger,payload)
    except Exception:
        e78.cancel(submitted)
        if ledger.exists(): ledger.unlink()
        raise
    print(f"[e119] cells={len(records)} released={len(submitted)} snapshot={snapshot} ledger={ledger}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
