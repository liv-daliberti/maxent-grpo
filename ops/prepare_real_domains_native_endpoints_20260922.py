#!/usr/bin/env python3
"""Freeze the authorized fixed coding128 native-HF endpoints; never submit jobs.

The plan phase records thirteen prospective jobs before endpoint outcomes exist.
Materialization requires the paired initialization identities for base, and both
complete terminal runs plus the selected sealed checkpoint for trained policies.
Every runnable job is a new immutable source/data/provenance/checkpoint snapshot.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil

import prepare_real_domains_run_20260921 as original_launcher
import prepare_real_domains_endpoints_20260921 as original_endpoints

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = "evaluate_real_domains_native_hf_20260922.py"
TRAINER = "train_real_domains_pilot_20260921_v2.py"
TRAINER_SHA256 = "e320611aa612d5f2772fe4b45f8e9bbbbad126a5666090b1f97f277049d00535"
TRAINER_SCHEMA = "real-domain-online-maxrl-remax-pilot-20260921-v2-matched-scoring-width"
MUTABLE = original_endpoints.MUTABLE
PRODUCTION_MODULES = (
    "oat_drgrpo.learner.grpo", "oat_drgrpo.learner.base", "oat_drgrpo.canonical_replay",
    "oat_drgrpo.maxrl", "oat_drgrpo.online_canonical_bank", "oat_drgrpo.scoring",
    "oat_drgrpo.tensor_utils", "oat_drgrpo.args",
)


def digest(path):
    return original_launcher.digest(Path(path))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_plan(plan):
    """Reject silent cohort, denominator, trajectory or budget changes."""
    if plan.get("schema") != "corrected-code128-development-diagnostic-plan-20260921-v1":
        raise ValueError("unsupported original plan schema")
    train = plan["dataset"]["train_ids"]
    dev = plan["dataset"]["untrained_development_ids"]
    evaluation = plan["native_hf_evaluation"]
    if len(train) != 8 or len(dev) != 13 or len(set(train + dev)) != 21 or evaluation["global_task_order"] != train + dev:
        raise ValueError("plan must preserve the fixed eight training and thirteen development tasks")
    if plan["dataset"]["reserved_test_overlap"] or plan["reserved_test_prompts_or_responses_used"]:
        raise ValueError("reserved tests are outside this diagnostic")
    expected = {("base", 0), ("maxrl", 32), ("remax", 32), ("maxrl", 64), ("remax", 64), ("maxrl", 128), ("remax", 128)}
    seen = set()
    names = set()
    shards = []
    for policy in evaluation["schedule"]:
        key = (policy["arm"], policy["step"])
        if key not in expected or key in seen:
            raise ValueError("duplicate or unplanned policy/checkpoint")
        seen.add(key)
        intermediate = policy["step"] in (32, 64)
        expected_counts = {task: (128 if task in train else 32) if intermediate else (512 if task in train else 128) for task in train + dev}
        observed = {}
        policy_cap = 0.0
        for shard in policy["job_shards"]:
            name = shard["name"]
            if name in names or Path(name).name != name or name in (".", ".."):
                raise ValueError("endpoint names must be unique safe directory names")
            names.add(name)
            tasks = shard["task_ids"]
            if len(set(tasks)) != len(tasks) or tasks != [t for t in train + dev if t in tasks]:
                raise ValueError("shards must retain global task order")
            counts = shard.get("samples_per_task_by_id", {t: shard.get("samples_per_task") for t in tasks})
            if set(counts) != set(tasks) or any(type(n) is not int or n <= 0 or n % 4 for n in counts.values()):
                raise ValueError("complete four-response batches and exact sample denominators required")
            if set(observed) & set(tasks):
                raise ValueError("duplicate task within policy shards")
            observed.update(counts)
            cap = shard["gpu_hour_cap"]
            if cap != (2.5 if intermediate else 3.0):
                raise ValueError("unplanned shard allocation cap")
            policy_cap += cap
            shards.append({"name": name, "arm": key[0], "checkpoint_step": key[1], "purpose": policy["purpose"], "task_ids": tasks, "samples_per_task_by_id": counts, "gpu_hour_cap": cap, "total_completions": sum(counts.values())})
        if observed != expected_counts or sum(observed.values()) != policy["total_completions"] or policy_cap != policy["allocation_cap_gpu_hours"]:
            raise ValueError("policy sample denominator or allocation differs from fixed plan")
    if seen != expected or len(shards) != 13 or sum(s["total_completions"] for s in shards) != 23040 or sum(s["gpu_hour_cap"] for s in shards) != 37:
        raise ValueError("expected thirteen shards, 23040 completions and 37 evaluation GPU hours")
    if evaluation["seed"] != 119411 or evaluation["generation_batch_size"] != 4 or evaluation["max_new_tokens"] != 1024 or any(evaluation[k] != v for k, v in {"temperature": 1.0, "top_p": 1.0, "top_k": 0}.items()):
        raise ValueError("fixed native sampler contract changed")
    if plan["budget"]["total_new_hard_envelope_gpu_hours"] != 47 or plan["budget"]["training_hard_caps_gpu_hours"] != 6 or plan["budget"]["retry_or_validation_reserve_gpu_hours"] != 4:
        raise ValueError("fixed 47-hour allocation envelope changed")
    if plan["proposed_trainer"]["sha256"] != TRAINER_SHA256:
        raise ValueError("corrected trainer source pin changed")
    return shards


def source_contract(plan):
    quality = json.loads((Path(plan["dataset"]["slate_root"]) / "hardening_quality.json").read_text())
    pins = {**quality["canonicalizer_source_files"], **quality["verifier_support_files"]}
    pins["ops/" + TRAINER] = TRAINER_SHA256
    names = ["ops/" + ENTRYPOINT, "ops/prepare_real_domains_native_endpoints_20260922.py", "ops/prepare_real_domains_run_20260921.py", "ops/prepare_real_domains_endpoints_20260921.py", "ops/evaluate_real_domains_20260921.py", "ops/repo_env.sh"]
    names.extend("src/" + name.replace(".", "/") + ".py" for name in PRODUCTION_MODULES)
    for name in names:
        pins[name] = digest(ROOT / name)
    for name, sha in pins.items():
        if digest(ROOT / name) != sha:
            raise ValueError("critical source differs from original admission/trainer: " + name)
    return pins


def model_manifest(model):
    model = Path(model).resolve()
    files = [{"relative_path": p.relative_to(model).as_posix(), "path": str(p), "sha256": digest(p), "size_bytes": p.stat().st_size} for p in sorted(model.rglob("*")) if p.is_file() and ".cache" not in p.parts]
    if not files or not any(row["relative_path"].endswith(".safetensors") for row in files):
        raise ValueError("model manifest requires full local model weights")
    return {"schema": "native-hf-model-files-20260922-v1", "model_root": str(model), "model_revision": model.name, "files": files}


def prepare_schedule(plan_path, output, maxrl_run, remax_run, run_parent):
    plan_path, output, run_parent = Path(plan_path).resolve(), Path(output).resolve(), Path(run_parent).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    shards = validate_plan(plan)
    pins = source_contract(plan)
    slate = Path(plan["dataset"]["slate_root"])
    if digest(slate / "manifest.json") != plan["dataset"]["manifest_sha256"] or digest(slate / "hardening_quality.json") != plan["dataset"]["quality_sha256"]:
        raise ValueError("dataset bytes differ from original plan")
    model = model_manifest(plan["model"])
    output.mkdir(parents=True)
    protocol = output / "protocol"
    protocol.mkdir()
    shutil.copyfile(plan_path, protocol / "original_plan.json")
    if digest(protocol / "original_plan.json") != digest(plan_path):
        raise ValueError("original plan changed while copying")
    write(protocol / "source_contract.json", pins)
    write(protocol / "model_files.json", model)
    runs = {"maxrl": str(Path(maxrl_run).resolve()), "remax": str(Path(remax_run).resolve())}
    schedule = {"schema": "native-hf-endpoint-schedule-20260922-v1", "status": "PROSPECTIVE_FIXED_SCHEDULE_NO_JOBS_SUBMITTED", "created_at": datetime.now(timezone.utc).isoformat(), "original_plan_path": str(plan_path), "original_plan_sha256": digest(plan_path), "plan_path": str(protocol / "original_plan.json"), "plan_sha256": digest(protocol / "original_plan.json"), "source_contract_path": str(protocol / "source_contract.json"), "source_contract_sha256": digest(protocol / "source_contract.json"), "model_manifest_path": str(protocol / "model_files.json"), "model_manifest_sha256": digest(protocol / "model_files.json"), "training_runs": runs, "run_parent": str(run_parent), "endpoint_cap_gpu_hours": 37.0, "training_cap_gpu_hours": 6.0, "validation_retry_reserve_gpu_hours": 4.0, "total_cap_gpu_hours": 47.0, "total_completions": 23040, "shards": shards}
    for shard in shards:
        write(output / "prospective" / (shard["name"] + ".json"), {"schema": "native-hf-prospective-endpoint-20260922-v1", "plan_sha256": schedule["plan_sha256"], "source_contract_sha256": schedule["source_contract_sha256"], "model_manifest_sha256": schedule["model_manifest_sha256"], "training_runs": runs, "global_task_order": plan["native_hf_evaluation"]["global_task_order"], "eval_seed": 119411, "run": str(run_parent / shard["name"]), **shard})
    write(output / "schedule.json", schedule)
    return schedule


def verify_schedule(path):
    schedule = json.loads(Path(path).read_text())
    if schedule.get("schema") != "native-hf-endpoint-schedule-20260922-v1":
        raise ValueError("unknown schedule schema")
    for prefix in ("plan", "source_contract", "model_manifest"):
        if digest(schedule[prefix + "_path"]) != schedule[prefix + "_sha256"]:
            raise ValueError(prefix + " drift")
    plan = json.loads(Path(schedule["plan_path"]).read_text())
    if schedule["shards"] != validate_plan(plan):
        raise ValueError("schedule differs from original plan")
    for relative, sha in json.loads(Path(schedule["source_contract_path"]).read_text()).items():
        if digest(ROOT / relative) != sha:
            raise ValueError("source drift after prospective freeze: " + relative)
    return schedule, plan


def frozen_training_run(path, schedule, arm, checked_external=None):
    """Keep bundle custody checks while admitting only explicitly bound model files.

    External cached weights were added to the training identity before submission.
    They must form the exact model-manifest file set, with identical source and
    snapshot paths; arbitrary external files remain forbidden. Both arms share
    one read-and-hash cache within this validation call, never across calls.
    """
    path = Path(path).resolve()
    identity = json.loads((path / "identity.json").read_text())
    config = json.loads((path / "config.json").read_text())
    bundle = path / "bundle"
    if identity.get("schema") != "real-domains-frozen-job-20260921-v1" or identity["request"].get("entrypoint") != TRAINER or identity["request"].get("arm") != arm:
        raise ValueError("frozen training entrypoint or arm mismatch")
    if digest(path / "config.json") != identity["config_sha256"]:
        raise ValueError("frozen training config checksum mismatch")
    manifest = json.loads(Path(schedule["model_manifest_path"]).read_text())
    allowed = {str(Path(row["path"]).absolute()): row["sha256"] for row in manifest["files"]}
    checked_external = {} if checked_external is None else checked_external
    files, seen_external, manifests = {}, {}, []
    for row in identity["files"]:
        snapshot = Path(row["snapshot"]).absolute()
        resolved = snapshot.resolve()
        if resolved in files:
            raise ValueError("duplicate frozen training file")
        if resolved.is_relative_to(bundle):
            if digest(resolved) != row["sha256"]:
                raise ValueError("frozen training dependency checksum mismatch")
        else:
            if str(snapshot) != row["source"]:
                raise ValueError("external model binding source/snapshot mismatch")
            kind = row.get("binding_kind")
            if kind == "external_content_addressed_model_file" and allowed.get(str(snapshot)) == row["sha256"]:
                seen_external[str(snapshot)] = row["sha256"]
            elif kind == "prospective_shared_model_manifest" and row["sha256"] == schedule["model_manifest_sha256"]:
                manifests.append(str(snapshot))
            else:
                raise ValueError("unregistered external frozen training dependency")
            key = (str(snapshot), row["sha256"])
            if key not in checked_external:
                if digest(snapshot) != row["sha256"]:
                    raise ValueError("external model checksum mismatch")
                checked_external[key] = True
        files[resolved] = row["sha256"]
    if seen_external != allowed or len(manifests) != 1:
        raise ValueError("training identity must bind exact complete model manifest")
    replacements = [(str(Path(source).resolve()), str(bundle / "data" / str(i))) for i, source in enumerate(identity["request"].get("freeze_data_roots", []))]
    if config != original_launcher.rewrite_paths(identity["request"]["config"], replacements):
        raise ValueError("frozen training request/config mismatch")
    return {"root": path, "identity": identity, "config": config, "files": files}


def training_pair(schedule, plan, require_terminal):
    runs = {}
    normalized = []
    checked_external = {}
    for arm in ("maxrl", "remax"):
        root = Path(schedule["training_runs"][arm])
        run = frozen_training_run(root, schedule, arm, checked_external)
        path = root / "training" / "identity.json"
        identity = json.loads(path.read_text())
        cfg = identity["config"]
        if identity.get("schema") != TRAINER_SCHEMA or identity.get("arm") != arm or identity.get("runner_sha256") != TRAINER_SHA256 or identity.get("config_sha256") != canonical(cfg) or identity.get("input_config_sha256") != digest(root / "config.json"):
            raise ValueError("training identity schema/source/arm/config mismatch")
        if any(cfg.get(k) != value for k, value in run["config"].items()):
            raise ValueError("training configuration differs from frozen input")
        if cfg["train_ids"] != plan["dataset"]["train_ids"] or cfg["eval_ids"] != plan["dataset"]["untrained_development_ids"] or cfg["updates"] != 128 or cfg["seed"] != 88411 or cfg["train_microbatch_size"] != 1 or cfg["model_revision"] != plan["model_revision"] or cfg["model"] != plan["model"]:
            raise ValueError("training pair differs from fixed coding128 protocol")
        if not identity.get("initial_trainable_parameters_sha256"):
            raise ValueError("both training initialization hashes are required before baseline")
        if set(identity.get("production_source_sha256", {})) != set(PRODUCTION_MODULES):
            raise ValueError("incomplete production source identity")
        for name, sha in identity["production_source_sha256"].items():
            relative = "src/" + name.replace(".", "/") + ".py"
            if sha != original_endpoints.source(run, relative) or sha != digest(ROOT / relative):
                raise ValueError("production source drift")
        for relative, sha in json.loads(Path(schedule["source_contract_path"]).read_text()).items():
            if relative in ("ops/" + ENTRYPOINT, "ops/prepare_real_domains_native_endpoints_20260922.py"):
                continue
            frozen_relative = "testlib/testlib.h" if relative == "third_party/testlib/testlib.h" else "repo_env.sh" if relative == "ops/repo_env.sh" else relative
            if original_endpoints.source(run, frozen_relative) != sha:
                raise ValueError("training source differs from prospective endpoint: " + relative)
        quality = Path(cfg["adapter_config"]["slate_root"]) / "hardening_quality.json"
        if digest(quality) != plan["dataset"]["quality_sha256"] or digest(quality.parent / "manifest.json") != plan["dataset"]["manifest_sha256"]:
            raise ValueError("training dataset differs from fixed plan")
        norm = copy.deepcopy(cfg)
        for key in MUTABLE:
            norm["adapter_config"].pop(key, None)
        norm["adapter_config"]["slate_root"] = {"manifest_sha256": plan["dataset"]["manifest_sha256"], "quality_sha256": plan["dataset"]["quality_sha256"]}
        normalized.append(norm)
        result_path = root / "training" / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else None
        if require_terminal and (not result or result.get("status") != "complete" or result.get("arm") != arm or result.get("completed_updates") != 128 or result.get("config_sha256") != identity["config_sha256"]):
            raise ValueError("trained endpoints require both complete terminal128 runs")
        runs[arm] = {"root": root, "frozen": run, "identity": identity, "identity_path": path, "result_path": result_path if result is not None else None}
    if normalized[0] != normalized[1] or runs["maxrl"]["identity"]["initial_trainable_parameters_sha256"] != runs["remax"]["identity"]["initial_trainable_parameters_sha256"]:
        raise ValueError("paired training configuration or initial weights mismatch")
    return runs


def verify_checkpoint(path, arm, step, training_identity):
    path = Path(path).resolve()
    seal = json.loads((path / "complete.json").read_text())
    if path.name != f"checkpoint-{step}" or seal.get("schema") != TRAINER_SCHEMA or seal.get("arm") != arm or type(seal.get("completed_updates")) is not int or seal["completed_updates"] != step or seal.get("config_sha256") != training_identity["config_sha256"]:
        raise ValueError("selected checkpoint schema/arm/step/config mismatch")
    if step not in (32, 64, 128):
        raise ValueError("checkpoint96 is a recovery artifact, not an endpoint")
    if digest(path / "bank.json") != seal["bank_sha256"] or digest(path / "training.pt") != seal["training_state_sha256"]:
        raise ValueError("checkpoint bank/training-state seal mismatch")
    actual = {p.relative_to(path / "adapter").as_posix(): digest(p) for p in sorted((path / "adapter").rglob("*")) if p.is_file()}
    if actual != seal["adapter_files"]:
        raise ValueError("checkpoint adapter seal mismatch")
    adapter = json.loads((path / "adapter" / "adapter_config.json").read_text())
    cfg = training_identity["config"]
    if adapter.get("r") != cfg["lora_rank"] or adapter.get("lora_alpha") != cfg["lora_alpha"] or set(adapter.get("target_modules", [])) != set(cfg["lora_target_modules"]):
        raise ValueError("checkpoint LoRA architecture mismatch")
    return {"path": str(path), "seal_sha256": digest(path / "complete.json"), "arm": arm, "step": step, "seal": seal}


def freeze_job(request_path, output):
    """Versioned no-submit copy of the established frozen-job launcher contract."""
    request = json.loads(Path(request_path).read_text())
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    validation = request.get("config", {}).get("validation_only") is True
    valid_time = (type(request["time_limit_minutes"]) is int and 1 <= request["time_limit_minutes"] <= 15) if validation else request["time_limit_minutes"] in (150, 180)
    valid_partition = request["partition"] in ({"all", "lowprio"} if validation else {"lowprio"})
    if request["entrypoint"] != ENTRYPOINT or request.get("gpu_count", 1) != 1 or not valid_partition or not valid_time:
        raise ValueError("native endpoint must retain one A6000 and registered bounded allocation")
    bundle = output / "bundle"
    bundle.mkdir(parents=True)
    files = []
    def snapshot(source, destination):
        source, destination = Path(source), Path(destination)
        sha = digest(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(destination) != sha or digest(source) != sha:
            raise ValueError("source changed while freezing: " + str(source))
        files.append({"source": str(source), "snapshot": str(destination), "sha256": sha})
    sources = sorted(list((ROOT / "ops").glob("*.py")) + list((ROOT / "ops").glob("*.c")) + list((ROOT / "src").rglob("*.py")))
    for source in sources:
        snapshot(source, bundle / source.relative_to(ROOT))
    snapshot(ROOT / "ops/repo_env.sh", bundle / "repo_env.sh")
    snapshot(ROOT / "third_party/testlib/testlib.h", bundle / "testlib/testlib.h")
    replacements = []
    for index, name in enumerate(request["freeze_data_roots"]):
        source_root = Path(name).resolve()
        if not source_root.is_dir():
            raise NotADirectoryError(source_root)
        destination = bundle / "data" / str(index)
        replacements.append((str(source_root), str(destination)))
        for source in sorted(source_root.rglob("*")):
            if source.is_file():
                snapshot(source, destination / source.relative_to(source_root))
    config = original_launcher.rewrite_paths(request["config"], replacements)
    write(output / "config.json", config)
    identity = {"schema": "real-domains-frozen-job-20260921-v1", "prepared_at": datetime.now(timezone.utc).isoformat(), "request": request, "request_sha256": digest(request_path), "config_sha256": digest(output / "config.json"), "files": files, "job_id": None, "allocated_gpu_hour_ceiling": request["time_limit_minutes"] / 60}
    write(output / "identity.json", identity)
    python = ROOT / "var/seed_paper_eval/paper310/bin/python"
    q = shlex.quote
    script = ["#!/usr/bin/env bash", "set -euo pipefail", f"export OAT_ZERO_REPO_ROOT={q(str(ROOT))}", f"source {q(str(bundle / 'repo_env.sh'))}", f"export OAT_ZERO_SOURCE_ROOT={q(str(bundle / 'src'))}", f"export OAT_ZERO_TESTLIB_ROOT={q(str(bundle / 'testlib'))}", f"export OAT_ZERO_SANDBOX_SOURCE={q(str(bundle / 'ops/constructive_code_sandbox.c'))}", f"export PYTHONPATH={q(str(bundle / 'ops') + ':' + str(bundle / 'src'))}", f"export LD_LIBRARY_PATH={q(str(python.parent.parent / 'lib'))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}", "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4", f"{q(str(python))} - {q(str(output / 'identity.json'))} {q(str(output / 'config.json'))} <<'PYVERIFY'", "import hashlib,json,pathlib,sys", "def digest(p):", "    h=hashlib.sha256()", "    with pathlib.Path(p).open('rb') as f:", "        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)", "    return h.hexdigest()", "identity=json.loads(pathlib.Path(sys.argv[1]).read_text())", "assert digest(sys.argv[2])==identity['config_sha256'], 'configuration drift'", "for row in identity['files']:", "    assert digest(row['snapshot'])==row['sha256'], row['snapshot']", "config=json.loads(pathlib.Path(sys.argv[2]).read_text())", "assert digest(config['model_manifest_path'])==config['model_manifest_sha256'], 'model manifest drift'", "print('Frozen source/data/checkpoint/provenance verified; native evaluator verifies full model bytes before loading.',flush=True)", "PYVERIFY", "exec " + shlex.join([str(python), str(bundle / "ops" / ENTRYPOINT), "--config", str(output / "config.json"), "--output", str(output / "evaluation.json")]), ""]
    (output / "run.slurm").write_text("\n".join(script))
    argv = ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1", "--gres=gpu:a6000:1", "--cpus-per-task=8", "--mem=64G", f"--time={request['time_limit_minutes']}", "--account=mltheory", "--partition=" + request["partition"], "--job-name=" + request["job_name"], f"--output={output}/slurm-%j.out", f"--error={output}/slurm-%j.err", str(output / "run.slurm")]
    write(output / "submission_intent.json", {"argv": argv, "authorized_gpu_hour_ceiling": request["time_limit_minutes"] / 60})
    return {"run": str(output), "frozen_identity_sha256": digest(output / "identity.json"), "config_sha256": digest(output / "config.json"), "gpu_hour_cap": request["time_limit_minutes"] / 60, "submitted": False}


def materialize(schedule_path, name):
    schedule_path = Path(schedule_path).resolve()
    schedule, plan = verify_schedule(schedule_path)
    matches = [row for row in schedule["shards"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError("unknown endpoint shard")
    shard = matches[0]
    parent = schedule_path.parent
    staging = parent / "materialized" / name
    destination = Path(schedule["run_parent"]) / name
    if staging.exists() or destination.exists():
        raise FileExistsError("each endpoint materialization must use its new planned destination")
    runs = training_pair(schedule, plan, shard["arm"] != "base")
    checkpoint = None
    if shard["arm"] != "base":
        selected = runs[shard["arm"]]
        checkpoint = verify_checkpoint(selected["root"] / "training" / f"checkpoint-{shard['checkpoint_step']}", shard["arm"], shard["checkpoint_step"], selected["identity"])
    provenance_dir = staging / "provenance"
    provenance_dir.mkdir(parents=True)
    checkpoint_root = staging / "selected_checkpoint"
    if checkpoint:
        # A containing root preserves checkpoint-N in relocated runtime paths.
        # Hard links do not alter sealed originals; the runnable bundle is a
        # byte-checked independent copy and the seal is checked again below.
        checkpoint_root.mkdir()
        staged_checkpoint = checkpoint_root / Path(checkpoint["path"]).name
        shutil.copytree(checkpoint["path"], staged_checkpoint, copy_function=os.link)
    provenance = {"runs": {}, "terminal_updates": 128}
    for arm, run in runs.items():
        row = {"original_run_root": str(run["root"])}
        for prefix, source in (("identity", run["identity_path"]), ("result", run["result_path"]), ("frozen_identity", run["root"] / "identity.json")):
            if source is None:
                row[prefix + "_path"] = row[prefix + "_sha256"] = None
                continue
            target = provenance_dir / arm / (prefix + ".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            sha = digest(source)
            shutil.copyfile(source, target)
            if digest(target) != sha or digest(source) != sha:
                raise ValueError("training provenance changed during snapshot")
            row[prefix + "_path"] = str(target)
            row[prefix + "_sha256"] = sha
        provenance["runs"][arm] = row
    config = {"schema": "native-hf-real-domains-evaluation-20260922-v1", "training_config": copy.deepcopy(runs["maxrl"]["identity"]["config"]), "arm": shard["arm"], "checkpoint_step": shard["checkpoint_step"], "global_task_order": plan["native_hf_evaluation"]["global_task_order"], "task_ids": shard["task_ids"], "samples_per_task_by_id": shard["samples_per_task_by_id"], "eval_seed": 119411, "checkpoint_path": str(staged_checkpoint) if checkpoint else None, "training_provenance": provenance, "expected_initial_trainable_parameters_sha256": runs["maxrl"]["identity"]["initial_trainable_parameters_sha256"], "max_gpu_hours": shard["gpu_hour_cap"], "cohort_ids": {"train": plan["dataset"]["train_ids"], "development": plan["dataset"]["untrained_development_ids"]}, "plan_path": schedule["plan_path"], "plan_sha256": schedule["plan_sha256"], "model_manifest_path": schedule["model_manifest_path"], "model_manifest_sha256": schedule["model_manifest_sha256"]}
    # Always use the original source data; relocated training paths remain in the
    # immutable identity copies and do not leak mutable verifier caches across jobs.
    config["training_config"]["adapter_config"]["slate_root"] = plan["dataset"]["slate_root"]
    for key in MUTABLE:
        if key in config["training_config"]["adapter_config"]:
            config["training_config"]["adapter_config"][key] = str(destination.parent / (name + "_work") / key)
    request = {"entrypoint": ENTRYPOINT, "config": config, "freeze_data_roots": [plan["dataset"]["slate_root"], str(provenance_dir), str(parent / "protocol")] + ([str(checkpoint_root)] if checkpoint else []), "time_limit_minutes": int(shard["gpu_hour_cap"] * 60), "gpu_count": 1, "partition": "lowprio", "job_name": name, "endpoint_provenance": {"schedule_path": str(schedule_path), "schedule_sha256": digest(schedule_path), "plan_sha256": schedule["plan_sha256"], "source_contract_sha256": schedule["source_contract_sha256"], "preparer_sha256": digest(Path(__file__)), "selected_checkpoint": checkpoint, "shard": shard}}
    request_path = staging / "request.json"
    write(request_path, request)
    result = freeze_job(request_path, destination)
    copied = json.loads((destination / "config.json").read_text())
    # Compare the actual runnable copies to the prospective critical-source
    # contract, closing the interval between schedule validation and copying.
    for relative, expected in json.loads(Path(schedule["source_contract_path"]).read_text()).items():
        frozen_relative = "testlib/testlib.h" if relative == "third_party/testlib/testlib.h" else "repo_env.sh" if relative == "ops/repo_env.sh" else relative
        if digest(destination / "bundle" / frozen_relative) != expected:
            raise ValueError("critical source changed while freezing endpoint: " + relative)
    if digest(copied["plan_path"]) != schedule["plan_sha256"] or digest(copied["model_manifest_path"]) != schedule["model_manifest_sha256"]:
        raise ValueError("protocol/model manifest changed during endpoint freezing")
    if checkpoint:
        receipt = verify_checkpoint(copied["checkpoint_path"], shard["arm"], shard["checkpoint_step"], runs[shard["arm"]]["identity"])
        if receipt["seal_sha256"] != checkpoint["seal_sha256"]:
            raise ValueError("checkpoint changed during endpoint freezing")
    result.update({"schema": "native-hf-endpoint-materialization-20260922-v1", "request": str(request_path), "request_sha256": digest(request_path), "schedule_sha256": digest(schedule_path), "name": name, "total_completions": shard["total_completions"]})
    write(staging / "materialization.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    for field in ("plan", "output", "maxrl-run", "remax-run", "run-parent"):
        plan.add_argument("--" + field, type=Path, required=True)
    validation = commands.add_parser("freeze-validation")
    validation.add_argument("--request", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    selected = commands.add_parser("materialize")
    selected.add_argument("--schedule", type=Path, required=True)
    selected.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.command == "freeze-validation":
        request = json.loads(args.request.read_text())
        if request.get("config", {}).get("validation_only") is not True:
            raise ValueError("direct request freezing is restricted to tiny explicit validation jobs")
        result = freeze_job(args.request, args.output)
    elif args.command == "plan":
        result = prepare_schedule(args.plan, args.output, args.maxrl_run, args.remax_run, args.run_parent)
    else:
        result = materialize(args.schedule, args.name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
