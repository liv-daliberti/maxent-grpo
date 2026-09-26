#!/usr/bin/env python3
"""Independent audit of fixed coding128 native-HF evaluation receipts.

No inference or code execution occurs here. Raw tokens, checkpoint/source seals,
the hardened checker evidence and all requested denominators are checked before
metrics are recomputed. Missing evidence raises; ineligible PCMD remains null.

Derived from the frozen auditor with ONE change: the expected training
allocation is declared on the command line instead of hard-coded to
180 minutes / 3.0 GPU-hours. The seeded128 runs were amended to 420 / 6.5
during training, before any endpoint existed, because the 3.0 h budget
truncated the slower arm; every scientific parameter the auditor checks
(cohort ids, seed, updates, group size, learning rate, LoRA rank, replay
alpha, checkpoint cadence) is unchanged and still enforced. The check is
kept rather than removed so the allocation cannot drift silently, and the
declared value is recorded in the emitted summary.
"""
from __future__ import annotations

import argparse
import ast
import importlib
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import audit_real_domains_corrected_training_20260921 as training_audit

SCHEMA = "independent-native-hf-code128-audit-20260922-v1"
TRAIN_IDS = ["1454_A", "1569_A", "361_A", "1408_A", "988_A", "1323_A", "1380_A", "1038_B"]
DEV_IDS = ["1513_A", "1352_B", "1016_D", "1360_G", "244_A", "1102_B", "1095_C", "1352_G", "482_A", "1339_B", "1371_D", "1051_B", "545_B"]
GLOBAL_IDS = TRAIN_IDS + DEV_IDS
RESERVED_IDS = {"1023_C", "1093_B", "1325_A", "1430_A", "1436_B", "1450_A", "1549_A", "1606_A", "544_B", "710_C", "1092_A", "1413_A", "1520_C"}
EVALUATOR_SHA256 = "3c3b0a3f90595f595bc8bdd5c0ab36558d776668eb15b452ff7a9ae527d2d012"
EVALUATOR_FILE = "evaluate_real_domains_native_hf_20260922.py"
EVALUATOR_SCHEMA = "native-hf-real-domains-evaluation-20260922-v1"
METRICS_HELPER_SHA256 = "f0f315ae45fb1245b778ee14978ecd880bf5c2035a8dad49d8f62227cd9b975e"
SAMPLING_SEED = 119411
TRAINING_SEED = 88411


DECLARED_ALLOCATION = {"minutes": None, "gpu_hours": None}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return training_audit.digest(path)


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def unique(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, f"duplicate identity: {identity}")
        result[identity] = row
    return result


def same(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, float):
        return type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    return type(actual) is type(expected) and actual == expected


def discovery(n, count, k):
    """Product-form finite-sample estimator, independent of evaluator comb()."""
    require(type(n) is type(count) is type(k) is int and 0 <= count <= n and 1 <= k <= n, "invalid discovery denominator")
    if n - count < k:
        return 1.0
    return 1.0 - math.prod((n - count - i) / (n - i) for i in range(k))


def metrics(rows):
    require(bool(rows), "empty requested denominator")
    for row in rows:
        require(type(row.get("accepted")) is bool, "nonboolean acceptance")
        require(isinstance(row.get("hard_violations"), list) and not row["hard_violations"], "hard verifier violation")
        key = row.get("canonical_key")
        require(isinstance(key, str) and bool(key) if row["accepted"] else key is None, "reward/mode mismatch")
    counts = Counter(row["canonical_key"] for row in rows if row["accepted"])
    n, m = len(rows), sum(counts.values())
    eligible = m >= 30
    return {"samples": n, "accepted": m, "accuracy": m / n,
            "mode_counts": dict(sorted(counts.items())), "distinct_valid_modes": len(counts),
            "pcmd_eligible": eligible, "pcmd_accepted_threshold": 30,
            "pcmd": sum(a * b for ka, a in counts.items() for kb, b in counts.items() if ka != kb) / (m * (m - 1)) if eligible else None,
            "pass_at_k": {str(k): discovery(n, m, k) for k in (1, 8, 32) if k <= n},
            "expected_distinct_valid_modes_at_k": {str(k): sum(discovery(n, c, k) for c in counts.values()) for k in (1, 8, 32) if k <= n},
            "hard_violation_count": 0}


def aggregate(tasks):
    values = list(tasks.values())
    require(bool(values), "empty task cohort")
    ks = sorted(set.intersection(*(set(v["pass_at_k"]) for v in values)), key=int)
    eligible = [t for t in tasks if tasks[t]["pcmd_eligible"]]
    return {"tasks": len(values), "samples": sum(v["samples"] for v in values),
            "accepted": sum(v["accepted"] for v in values),
            "macro_accuracy": sum(v["accuracy"] for v in values) / len(values),
            "tasks_with_multiple_modes": sum(v["distinct_valid_modes"] >= 2 for v in values),
            "pcmd_eligible_task_ids": eligible, "pcmd_eligible_tasks": len(eligible), "pcmd_total_tasks": len(values),
            "macro_pcmd_over_eligible_tasks": sum(tasks[t]["pcmd"] for t in eligible) / len(eligible) if eligible else None,
            "macro_pass_at_k": {k: sum(v["pass_at_k"][k] for v in values) / len(values) for k in ks},
            "macro_expected_distinct_valid_modes_at_k": {k: sum(v["expected_distinct_valid_modes_at_k"][k] for v in values) / len(values) for k in ks}}


def common_eligibility(arms):
    require(set(arms) == {"base", "maxrl", "remax"}, "three explicit policies required")
    ids = list(arms["base"])
    require(all(set(m) == set(ids) for m in arms.values()), "paired task coverage differs")
    require(all(len({arms[a][t]["samples"] for a in arms}) == 1 for t in ids), "paired requested denominators differ")
    def subset(labels):
        common = [t for t in ids if all(arms[a][t]["pcmd_eligible"] for a in labels)]
        means = {a: sum(arms[a][t]["pcmd"] for t in common) / len(common) if common else None for a in labels}
        return {"task_ids": common, "eligible_tasks": len(common), "total_tasks": len(ids),
                "macro_pcmd": means,
                "remax_minus_maxrl": means["remax"] - means["maxrl"] if common else None}
    return {"arm_wise": {a: aggregate(m) for a, m in arms.items()},
            "common_pair": subset(("maxrl", "remax")), "common_all_three": subset(("base", "maxrl", "remax"))}


def validate_tokens(row, *, prompt, prompt_tokens, codec, eos_ids, upper, max_new_tokens=1024):
    tokens = row.get("token_ids")
    require(isinstance(tokens, list) and 1 <= len(tokens) <= max_new_tokens, "response token denominator invalid")
    require(all(type(t) is int and 0 <= t < upper for t in tokens), "masked or invalid response token")
    require(row.get("prompt_token_ids") == prompt_tokens == codec.encode(prompt, add_special_tokens=False), "prompt token identity mismatch")
    require(row.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest(), "prompt hash mismatch")
    require(row.get("token_count") == len(tokens), "token count mismatch")
    require(row.get("text") == codec.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False), "token/text decoding mismatch")
    require(row.get("text_sha256") == hashlib.sha256(row["text"].encode()).hexdigest(), "text hash mismatch")
    require(not any(t in eos_ids for t in tokens[:-1]), "response continues after EOS")
    if tokens[-1] in eos_ids:
        require(row.get("finish_reason") == "eos", "EOS termination mislabeled")
    else:
        require(row.get("finish_reason") == "length" and len(tokens) == max_new_tokens, "non-EOS response does not reach frozen cap")


def validate_sample_identity(row, *, arm, step, counts, global_ids=GLOBAL_IDS):
    task, index = row.get("task_id"), row.get("sample_index")
    require(task in counts and type(index) is int and 0 <= index < counts[task], "unexpected sample identity")
    task_index = global_ids.index(task)
    batch_start, batch_offset = (index // 4) * 4, index % 4
    require(row.get("global_task_index") == task_index, "shard-local index substituted for global index")
    require(row.get("batch_seed") == SAMPLING_SEED + task_index * 10000 + batch_start, "batch seed differs from fixed global seed")
    require(row.get("batch_offset") == batch_offset, "batch offset mismatch")
    require(row.get("arm") == arm and row.get("step") == step and row.get("checkpoint_step") == step, "row policy or checkpoint mismatch")
    require(row.get("batch_start_index") == batch_start and row.get("request_seed") == row["batch_seed"], "request batch seed identity mismatch")
    cohort = "train" if task in TRAIN_IDS else "development"
    phase = f"native_hf:{arm}:{cohort}"
    require(row.get("cohort") == cohort and row.get("phase") == phase, "row cohort or phase mismatch")
    require(row.get("request_id") == f"{phase}:{step}:{task}:{index}", "raw request identity mismatch")


def validate_training_request(request):
    cfg = request["config"]
    require(request.get("entrypoint") == training_audit.TRAINER_FILENAME and request.get("arm") in {"maxrl", "remax"}, "unknown training entrypoint or arm")
    for key, expected in {"train_ids": TRAIN_IDS, "eval_ids": DEV_IDS, "seed": TRAINING_SEED, "updates": 128,
                          "train_microbatch_size": 1, "checkpoint_every": 32, "group_size": 16,
                          "generation_batch_size": 4, "max_new_tokens": 1024, "learning_rate": 1e-5,
                          "lora_rank": 16, "lora_alpha": 32, "replay_alpha": .1, "initial_evaluation": False}.items():
        require(same(cfg.get(key), expected), "training request drift: " + key)
    require(cfg["adapter_config"].get("problem_ids") == GLOBAL_IDS and cfg["adapter_config"].get("allow_heldout") is False, "training data cohort drift")
    require(not RESERVED_IDS.intersection(cfg["adapter_config"]["problem_ids"]), "reserved test overlap")
    require(cfg["adapter_module"] == "build_constructive_code_hardened_20260921", "unhardened adapter")
    require(request.get("time_limit_minutes") == DECLARED_ALLOCATION["minutes"]
            and cfg.get("max_gpu_hours") == DECLARED_ALLOCATION["gpu_hours"],
            "training allocation differs from the declared allocation")
    return {"status": "pass", "arm": request["arm"], "train_tasks": 8, "development_tasks": 13,
            "primary_endpoint": 128, "reserved_test_overlap": []}


def audit_checkpoint(checkpoint, training_identity, *, arm, step, allowed_steps=(32, 64, 128)):
    checkpoint = Path(checkpoint)
    seal = read(checkpoint / "complete.json")
    require(step in allowed_steps, "unregistered evaluation checkpoint")
    require(seal.get("schema") == training_audit.TRAINER_SCHEMA and seal.get("arm") == arm and seal.get("completed_updates") == step,
            "checkpoint trainer schema/arm/step mismatch")
    require(seal.get("config_sha256") == training_identity["config_sha256"], "checkpoint training config mismatch")
    for name, field in (("bank.json", "bank_sha256"), ("training.pt", "training_state_sha256")):
        require(digest(checkpoint / name) == seal[field], "checkpoint data seal mismatch: " + name)
    files = {p.relative_to(checkpoint / "adapter").as_posix(): digest(p) for p in (checkpoint / "adapter").rglob("*") if p.is_file()}
    require(files == seal["adapter_files"], "checkpoint adapter seal mismatch")
    return {"checkpoint_seal_sha256": digest(checkpoint / "complete.json"), "step": step, "arm": arm, "adapter_files": files}



def audit_loaded_adapter(identity):
    config = identity["config"]
    if config["arm"] == "base":
        return
    from safetensors import safe_open
    path = Path(config["checkpoint_path"]) / "adapter" / "adapter_model.safetensors"
    hashes = {}
    count = 0
    with safe_open(path, framework="numpy") as reader:
        for key in reader.keys():
            require(".lora_A.weight" in key or ".lora_B.weight" in key, "unexpected checkpoint adapter tensor")
            tensor = reader.get_tensor(key)
            require(str(tensor.dtype) == "float32", "sealed adapter is not FP32")
            live_name = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
            require(live_name not in hashes, "duplicate loaded adapter tensor")
            hashes[live_name] = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
            count += tensor.size
    require(bool(hashes) and canonical(hashes) == identity["model"]["loaded_trainable_parameters_sha256"], "loaded native adapter hash differs from sealed FP32 tensors")
    require(count == identity["model"]["trainable_parameters"], "loaded adapter parameter denominator differs")



def combine_shards(audits):
    require(len(audits) == 13 and all(a.get("schema") == SCHEMA and a.get("status") == "pass" and a.get("kind") == "shard" for a in audits), "complete13 independently passing shards required")
    require(len({a["plan_sha256"] for a in audits}) == len({a["initial_trainable_parameters_sha256"] for a in audits}) == 1, "shard plan/initial policy drift")
    points = {(a, s): {} for a, s in [("base", 0)] + [(a, s) for a in ("maxrl", "remax") for s in (32, 64, 128)]}
    base_rows = {}
    for audit in audits:
        key = audit["arm"], audit["step"]
        require(key in points and not set(points[key]).intersection(audit["task_metrics"]), "duplicate shard task or unregistered checkpoint")
        require(digest(audit["receipt"]) == audit["receipt_sha256"], "audited receipt drift")
        points[key].update(audit["task_metrics"])
        if key == ("base", 0):
            descriptor = audit["raw_artifacts"]["attempts"]
            require(digest(descriptor["path"]) == descriptor["sha256"], "audited base raw artifact drift")
            for line in Path(descriptor["path"]).read_text().splitlines():
                row = json.loads(line)
                base_rows.setdefault(row["task_id"], []).append(row)
    require(all(set(tasks) == set(GLOBAL_IDS) for tasks in points.values()), "incomplete fixed task union")
    require(sum(v["samples"] for tasks in points.values() for v in tasks.values()) == 23040, "full study requested denominator mismatch")
    for (arm, step), tasks in points.items():
        for task, value in tasks.items():
            wanted = (512 if task in TRAIN_IDS else 128) if step in (0, 128) else (128 if task in TRAIN_IDS else 32)
            require(value["samples"] == wanted, "combined sample budget mismatch")
    trajectories = {}
    for step in (32, 64, 128):
        baseline = points[("base", 0)] if step == 128 else {t: metrics(sorted(base_rows[t], key=lambda r: r["sample_index"])[:128 if t in TRAIN_IDS else 32]) for t in GLOBAL_IDS}
        cohorts = {}
        for cohort, ids in (("train", TRAIN_IDS), ("development", DEV_IDS)):
            arms = {"base": {t: baseline[t] for t in ids}, **{a: {t: points[(a, step)][t] for t in ids} for a in ("maxrl", "remax")}}
            comparison = common_eligibility(arms)
            comparison["task_metrics"] = arms
            comparison["remax_minus_maxrl_all_tasks"] = {
                "macro_accuracy": comparison["arm_wise"]["remax"]["macro_accuracy"] - comparison["arm_wise"]["maxrl"]["macro_accuracy"],
                "macro_pass_at_k": {k: comparison["arm_wise"]["remax"]["macro_pass_at_k"][k] - comparison["arm_wise"]["maxrl"]["macro_pass_at_k"][k] for k in ("1", "8", "32")},
                "macro_expected_distinct_valid_modes_at_k": {k: comparison["arm_wise"]["remax"]["macro_expected_distinct_valid_modes_at_k"][k] - comparison["arm_wise"]["maxrl"]["macro_expected_distinct_valid_modes_at_k"][k] for k in ("1", "8", "32")}}
            cohorts[cohort] = comparison
        trajectories[str(step)] = {"primary": step == 128, "cohorts": cohorts}
    return {"schema": SCHEMA, "status": "pass", "kind": "complete_study", "requested_completions": 23040,
            "primary_checkpoint": 128, "trajectories": trajectories, "audited_shards": audits,
            "uncertainty": "One paired training seed. No seed confidence intervals; checkpoints32/64 are descriptive and final128 is fixed primary.",
            "analysis_sha256": digest(Path(__file__))}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bound(path, expected):
    require(digest(path) == expected, 'file checksum mismatch: ' + str(path))
    return read(path)


def parameter_hash(adapter):
    from safetensors import safe_open
    names = {}
    with safe_open(str(Path(adapter) / 'adapter_model.safetensors'), framework='numpy') as tensors:
        for name in tensors.keys():
            require(('.lora_A.' in name) != ('.lora_B.' in name), 'unexpected non-LoRA tensor')
            tensor = tensors.get_tensor(name)
            require(str(tensor.dtype) == 'float32', 'sealed adapter is not native FP32')
            target = name.replace('.lora_A.', '.lora_A.default.').replace('.lora_B.', '.lora_B.default.')
            require(target not in names, 'duplicate adapter parameter')
            names[target] = hashlib.sha256(tensor.tobytes(order='C')).hexdigest()
    require(bool(names), 'empty adapter tensor set')
    return canonical(names)


def source_audit(root, identity):
    import ast
    launch = read(root / 'identity.json')
    require(launch.get('schema') == 'real-domains-frozen-job-20260921-v1' and launch['request']['entrypoint'] == 'evaluate_real_domains_native_hf_20260922.py', 'unknown endpoint launcher')
    raw = bound(root / 'config.json', launch['config_sha256'])
    require(identity['input_config_sha256'] == launch['config_sha256'] and identity['config_sha256'] == canonical(identity['config']), 'configuration hash mismatch')
    config = identity['config']
    require({k:v for k,v in config.items() if k != 'training_config'} == {k:v for k,v in raw.items() if k != 'training_config'}, 'resolved endpoint config changed')
    trainer_path = root / 'bundle/ops' / training_audit.TRAINER_FILENAME
    assignments = [n for n in ast.parse(trainer_path.read_text()).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULTS' for t in n.targets)]
    require(len(assignments) == 1 and config['training_config'] == {**ast.literal_eval(assignments[0].value), **raw['training_config']}, 'training defaults differ')
    snapshots = {}
    for row in launch['files']:
        path = Path(row['snapshot']).resolve()
        require(path.is_relative_to(root / 'bundle') and path not in snapshots and digest(path) == row['sha256'], 'frozen source/data custody or checksum mismatch')
        snapshots[path] = row['sha256']
    for filename, field, expected in [('evaluate_real_domains_native_hf_20260922.py', 'runner_sha256', EVALUATOR_SHA256), (training_audit.TRAINER_FILENAME, 'trainer_sha256', training_audit.TRAINER_SHA256)]:
        require(identity[field] == snapshots.get(root / 'bundle/ops' / filename) == expected, 'unrecognized policy implementation: ' + filename)
    require(identity['metrics_helper_sha256'] == snapshots[root / 'bundle/ops/evaluate_real_domains_20260921.py'], 'metric helper source drift')
    require(identity['adapter_module_sha256'] == snapshots[root / 'bundle/ops/build_constructive_code_hardened_20260921.py'], 'adapter source drift')
    require(digest(Path(config['training_config']['model']) / 'config.json') == identity['model_config_sha256'], 'model configuration drift')
    return launch, snapshots


def native_model_audit(config, identity):
    training = config['training_config']; root = Path(training['model']).resolve()
    manifest = bound(config['model_manifest_path'], config['model_manifest_sha256'])
    require(manifest['schema'] == 'native-hf-model-files-20260922-v1' and Path(manifest['model_root']).resolve() == root and manifest['model_revision'] == training['model_revision'] == root.name, 'model snapshot identity mismatch')
    expected = {}
    for row in manifest['files']:
        relative = Path(row['relative_path']); path = root / relative
        require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() not in expected, 'unsafe model manifest')
        require(Path(row['path']) == path and path.stat().st_size == row['size_bytes'] and digest(path) == row['sha256'], 'model snapshot bytes differ')
        expected[relative.as_posix()] = row['sha256']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '.cache' not in p.relative_to(root).parts}
    require(actual == set(expected) and {'config.json', 'tokenizer_config.json'} <= actual and any(p.endswith('.safetensors') for p in actual), 'incomplete model snapshot manifest')
    require(identity['model_manifest'] == {'manifest_sha256':config['model_manifest_sha256'], 'files':len(expected), 'model_revision':training['model_revision']}, 'model manifest receipt mismatch')
    model = identity['model']
    audit_loaded_adapter(identity)
    require(model['adapter_dtypes'] == ['torch.float32'] and model['base_dtypes'] == ['torch.bfloat16'], 'native model precision drift')
    require(model['initial_trainable_parameters_sha256'] == config['expected_initial_trainable_parameters_sha256'], 'initial LoRA hash differs')
    loaded = parameter_hash(Path(config['checkpoint_path']) / 'adapter') if config['checkpoint_path'] else model['initial_trainable_parameters_sha256']
    require(model['loaded_trainable_parameters_sha256'] == loaded, 'loaded adapter parameters differ from sealed tensor bytes')
    return {'status':'pass', 'model_manifest_sha256':config['model_manifest_sha256'], 'initial_hash':model['initial_trainable_parameters_sha256'], 'loaded_hash':loaded}


def provenance_audit(config, identity, helper):
    declarations = config['training_provenance']; reported = identity['training_provenance']
    require(set(declarations['runs']) == set(reported['runs']) == {'maxrl','remax'}, 'paired training provenance missing')
    records = {}
    for arm, paths in declarations['runs'].items():
        record = bound(paths['identity_path'], paths['identity_sha256']); frozen = bound(paths['frozen_identity_path'], paths['frozen_identity_sha256'])
        require(record['schema'] == training_audit.TRAINER_SCHEMA and record['arm'] == arm and record['runner_sha256'] == training_audit.TRAINER_SHA256, 'training identity/source mismatch')
        require(record['config_sha256'] == canonical(record['config']) and record['input_config_sha256'] == frozen['config_sha256'], 'training configuration seal mismatch')
        require(record['initial_trainable_parameters_sha256'] == config['expected_initial_trainable_parameters_sha256'] and record['objective']['behavior_scoring'] == training_audit.BEHAVIOR_CONTRACT, 'initial policy or corrected scoring mismatch')
        left, right = dict(record['config']), dict(config['training_config']); mutable = {'build_root','runtime_root','launcher','scratch_root','slate_root'}
        left['adapter_config'] = {k:v for k,v in left['adapter_config'].items() if k not in mutable}; right['adapter_config'] = {k:v for k,v in right['adapter_config'].items() if k not in mutable}
        require(left == right, 'evaluation training contract differs from parent')
        require(record['adapter_module_sha256'] == identity['adapter_module_sha256'] and record['dataset_identity'] == identity['dataset_identity'], 'training/evaluation source or data mismatch')
        require(digest(Path(paths['original_run_root']) / 'identity.json') == paths['frozen_identity_sha256'], 'original parent frozen identity drift')
        training_audit.audit_corrected_frozen_sources(Path(paths['original_run_root']) / 'training', record, training=True, helper=helper)
        if not config.get('validation_only', False):
            validate_training_request(frozen['request'])
        result = None
        if paths.get('result_path'):
            result = bound(paths['result_path'], paths['result_sha256'])
            require(result['schema'] == training_audit.TRAINER_SCHEMA and result['status'] == 'complete' and result['arm'] == arm and result['completed_updates'] == declarations['terminal_updates'] == record['config']['updates'] and result['config_sha256'] == record['config_sha256'], 'parent training incomplete')
        else:
            require(config['arm'] == 'base', 'trained evaluation lacks completed parent')
        require(reported['runs'][arm] == {'identity':record,'result':result,'bindings':paths}, 'runtime provenance receipt mismatch')
        records[arm] = record
    if config['arm'] == 'base':
        require(config['checkpoint_path'] is None and reported['checkpoint'] is None, 'base unexpectedly loads checkpoint')
        return {'status':'pass','checkpoint':None}
    checkpoint = Path(config['checkpoint_path'])
    original_checkpoint = Path(declarations['runs'][config['arm']]['original_run_root']) / 'training' / f"checkpoint-{config['checkpoint_step']}"
    require(digest(original_checkpoint / 'complete.json') == digest(checkpoint / 'complete.json'), 'selected checkpoint differs from original training trajectory')
    checked = audit_checkpoint(checkpoint, records[config['arm']], arm=config['arm'], step=config['checkpoint_step'], allowed_steps=(8,) if config.get('validation_only',False) else (32,64,128))
    files = {'complete.json':checked['checkpoint_seal_sha256'], 'bank.json':digest(checkpoint/'bank.json'), 'training.pt':digest(checkpoint/'training.pt'), **{'adapter/'+k:v for k,v in checked['adapter_files'].items()}}
    require(reported['checkpoint'] == {'path':str(checkpoint),'seal':read(checkpoint/'complete.json'),'files_sha256':files}, 'checkpoint provenance receipt mismatch')
    return {'status':'pass','checkpoint':checked}


def sampler_audit(config, identity, codec):
    sampler = identity['sampler']
    fixed = {'backend':'transformers/PEFT','attention_implementation':'sdpa','generation_batch_size':4,'temperature':1.0,'top_k':0,'top_p':1.0,'do_sample':True,'num_beams':1,'repetition_penalty':1.0,'generation_use_cache':True,'model_config_use_cache':False,'max_new_tokens':1024,'prompt_truncation':False,'eval_seed':SAMPLING_SEED,'raw_generated_eos_retained':True,'post_eos_padding_removed':True,'seed_rule':'eval_seed + global_task_index * 10000 + batch_start_index','attention_mask':'ones over exact prompt+response; no persisted padding','loss_mask':'causally shifted; zero over prompt targets and one over every response token including EOS'}
    for key,value in fixed.items():
        require(same(sampler.get(key),value), 'native sampler drift: '+key)
    root = Path(config['training_config']['model']); eos = read(root/'generation_config.json').get('eos_token_id') or codec.eos_token_id; eos = sorted(eos if isinstance(eos,list) else [eos]); vocab = read(root/'config.json')['vocab_size']
    require(sampler['eos_token_ids'] == eos and sampler['pad_token_id'] == (codec.pad_token_id if codec.pad_token_id is not None else codec.eos_token_id) and sampler['bos_token_id'] == codec.bos_token_id, 'EOS/padding/BOS policy drift')
    require(sampler['tokenizer_length'] == len(codec) and sampler['model_vocab_size'] == vocab and sampler['tokenizer_vocab_upper'] == min(len(codec),vocab), 'invalid vocabulary mask')
    return sampler


def audit_run(path):
    path = Path(path).resolve(); root,path = (path,path/'evaluation.json') if path.is_dir() else (path.parent,path)
    receipt = read(path)
    require(receipt.get('schema') == EVALUATOR_SCHEMA and receipt.get('status') == 'complete', 'native evaluation is not complete')
    identity = receipt['identity']; config = identity['config']; launch,snapshots = source_audit(root,identity)
    helper = training_audit.load_pinned_helper(); provenance = provenance_audit(config,identity,helper); model = native_model_audit(config,identity)
    validation = config.get('validation_only',False)
    require(type(validation) is bool and receipt.get('validation_only') == validation, 'validation status mismatch')
    order,cohorts,selected,counts = config['global_task_order'],config['cohort_ids'],config['task_ids'],config['samples_per_task_by_id']
    require(order == cohorts['train']+cohorts['development'] and len(order)==len(set(order)) and not RESERVED_IDS.intersection(order), 'invalid global cohort')
    require(selected == [t for t in order if t in selected] and len(selected)==len(set(selected)) and set(counts)==set(selected), 'shard task selection drift')
    require(config['eval_seed']==SAMPLING_SEED and all(type(n)is int and n>0 and n%4==0 for n in counts.values()), 'fixed seed or batch denominator drift')
    arm,step = config['arm'],config['checkpoint_step']
    require(arm in {'base','maxrl','remax'} and ((arm=='base')==(step==0)), 'policy arm/step mismatch')
    if validation:
        require(config['training_config']['updates']==8 and step in (0,8) and sum(counts.values())<=32, 'validation exceeds tiny corrected8 scope')
    else:
        require(order==GLOBAL_IDS and cohorts=={'train':TRAIN_IDS,'development':DEV_IDS} and step in (0,32,64,128) and config['training_config']['updates']==128, 'production cohort/checkpoint drift')
        nt,nd = (128,32) if step in (32,64) else (512,128)
        require(counts=={t:nt if t in TRAIN_IDS else nd for t in selected}, 'predeclared sample denominator drift')
        plan = bound(config['plan_path'],config['plan_sha256'])
        require(plan['dataset']['train_ids']==TRAIN_IDS and plan['dataset']['untrained_development_ids']==DEV_IDS and plan['native_hf_evaluation']['global_task_order']==GLOBAL_IDS, 'bound plan cohort differs')
    training=config['training_config']; records=helper.code_records({**training,'task_ids':order},identity['dataset_identity'])
    require(set(records)==set(order), 'hardened source coverage incomplete')
    import evaluate_constructive_code_v3_coder_viability as renderer
    require(digest(Path(renderer.__file__))==snapshots[root/'bundle/ops/evaluate_constructive_code_v3_coder_viability.py'], 'prompt renderer source differs')
    codec=helper.local_codec(training); sampler=sampler_audit(config,identity,codec); prompts=unique(identity['tasks'],lambda r:r['task_id'])
    require(set(prompts)==set(order), 'prompt receipt cohort differs')
    rendered={t:renderer._prompt(records[t]['statement']) for t in order}
    for task in order:
        prompt=prompts[task]
        require(prompt['prompt_token_ids']==codec.encode(rendered[task],add_special_tokens=False) and prompt['prompt_sha256']==hashlib.sha256(rendered[task].encode()).hexdigest(), 'frozen source/prompt token mismatch')
        require(prompt['global_task_index']==order.index(task) and prompt['split']==records[task]['split'] and prompt['family']==records[task]['witness_family'], 'prompt metadata mismatch')
        require(len(prompt['prompt_token_ids'])+1024<=training['max_context_tokens'], 'truncated or overlong prompt')
    artifacts={}
    for name in ('responses','attempts'):
        descriptor=receipt['artifacts'][name]; filename=Path(descriptor['path'])
        require(filename.resolve().parent==root and digest(filename)==descriptor['sha256'], 'raw artifact checksum or custody mismatch')
        artifacts[name]=unique(read_rows(filename),lambda r:(r['task_id'],r['sample_index']))
    expected={(t,i) for t,n in counts.items() for i in range(n)}
    require(set(artifacts['responses'])==set(artifacts['attempts'])==expected, 'missing or duplicated requested responses/attempts')
    grouped={t:[] for t in selected}; requests=set()
    for key,raw in artifacts['responses'].items():
        task,index=key; validate_sample_identity(raw,arm=arm,step=step,counts=counts,global_ids=order)
        cohort='train' if task in cohorts['train'] else 'development'; phase=f'native_hf:{arm}:{cohort}'
        require(raw['phase']==phase and raw['request_id']==f'{phase}:{step}:{task}:{index}' and raw['request_id'] not in requests, 'request identifier binding mismatch'); requests.add(raw['request_id'])
        require(raw['cohort']==cohort and raw['checkpoint_step']==step and raw['request_seed']==raw['batch_seed'] and raw['batch_start_index']==(index//4)*4, 'raw policy/seed metadata drift')
        validate_tokens(raw,prompt=rendered[task],prompt_tokens=prompts[task]['prompt_token_ids'],codec=codec,eos_ids=set(sampler['eos_token_ids']),upper=sampler['tokenizer_vocab_upper'])
        require(raw['eos_token_ids']==sampler['eos_token_ids'] and raw['attention_mask']==[1]*(len(raw['prompt_token_ids'])+len(raw['token_ids'])) and raw['loss_mask']==[0]*(len(raw['prompt_token_ids'])-1)+[1]*len(raw['token_ids']), 'raw attention/loss/EOS mask mismatch')
        attempt=artifacts['attempts'][key]
        require(all(attempt.get(k)==v for k,v in raw.items()), 'raw/verified completion differs')
        require(all(attempt[k]==attempt['verdict'][k] for k in ('accepted','canonical_key','hard_violations','receipt')), 'nested/flat verifier mismatch')
        helper.inspect_verdict(raw['text'],attempt,task,code_contract=records); grouped[task].append(attempt)
    computed={t:metrics(sorted(rows,key=lambda r:r['sample_index'])) for t,rows in grouped.items()}; reported=unique(receipt['task_results'],lambda r:r['task_id'])
    require(set(reported)==set(computed), 'reported metric task coverage differs')
    for task,values in computed.items():
        require(all(same(reported[task].get(k),v) for k,v in values.items()), 'reported per-task metric differs: '+task)
    for label,values in [('summary',computed)]+[(c,{t:computed[t] for t in selected if t in cohorts[c]}) for c in cohorts]:
        report=receipt['summary'] if label=='summary' else receipt['cohort_summaries'][label]
        if values:
            calculated=aggregate(values)
            required={'tasks','samples','accepted','macro_accuracy','pcmd_eligible_tasks','pcmd_total_tasks','macro_pcmd_over_eligible_tasks','macro_pass_at_k','macro_expected_distinct_valid_modes_at_k'}
            require(required<=set(report) and all(same(report[k],calculated[k]) for k in required) and report.get('hard_violation_count')==0, 'reported cohort metric differs: '+label)
        else:
            require(report=={'tasks':0,'samples':0,'accepted':0,'pcmd_eligible_tasks':0,'macro_pcmd_over_eligible_tasks':None}, 'empty cohort report malformed')
    require(receipt['evaluation_bank_mutations']==receipt['evaluation_optimizer_steps']==0 and receipt['adapter_hash_unchanged'] is True, 'evaluation changed training state')
    return {'schema':SCHEMA,'status':'pass','validation_only':validation,'arm':arm,'checkpoint_step':step,'run':str(root),'evaluation_sha256':digest(path),'frozen_identity_sha256':digest(root/'identity.json'),'auditor_sha256':digest(Path(__file__)),'evaluator_sha256':EVALUATOR_SHA256,'plan_sha256':config.get('plan_sha256'),'parent_training_identity_sha256':{a:b['identity_sha256'] for a,b in config['training_provenance']['runs'].items()},'task_metrics':computed,'cohort_ids':cohorts,'sampler':sampler,'artifacts':receipt['artifacts'],'source_and_token_binding':'pass','model_binding':model,'training_binding':provenance,'correctness_evidence':'Full hardened-source/reference quality checks and sealed checker/stability receipts; generated programs are not reexecuted by this auditor.','scope':'Execution validation only' if validation else 'Single-seed fixed-cohort descriptive endpoint'}


def assemble(schedule_path):
    schedule_path=Path(schedule_path).resolve(); schedule=read(schedule_path)
    require(schedule['schema']=='native-hf-endpoint-schedule-20260922-v1' and len(schedule['shards'])==13 and schedule['total_completions']==23040, 'unknown or incomplete registered schedule')
    bound(schedule['plan_path'],schedule['plan_sha256']); bound(schedule['source_contract_path'],schedule['source_contract_sha256'])
    policies,shards,all_samples={},[],0
    for shard in schedule['shards']:
        result=audit_run(Path(schedule['run_parent'])/shard['name'])
        require(result['plan_sha256'] == schedule['plan_sha256'], 'schedule/shard plan mismatch')
        require(not result['validation_only'] and result['arm']==shard['arm'] and result['checkpoint_step']==shard['checkpoint_step'], 'schedule/shard policy mismatch')
        require({t:m['samples'] for t,m in result['task_metrics'].items()}==shard['samples_per_task_by_id'], 'schedule/shard denominator mismatch')
        key=(result['arm'],result['checkpoint_step']); policy=policies.setdefault(key,{})
        require(not set(policy).intersection(result['task_metrics']), 'duplicate task across shards')
        rows=read_rows(result['artifacts']['attempts']['path'])
        for task in result['task_metrics']:
            policy[task]=sorted([r for r in rows if r['task_id']==task],key=lambda r:r['sample_index'])
        all_samples+=sum(m['samples'] for m in result['task_metrics'].values()); shards.append(result)
    expected={('base',0)}|{(a,s) for a in ('maxrl','remax') for s in (32,64,128)}
    require(set(policies)==expected and all(set(p)==set(GLOBAL_IDS) for p in policies.values()) and all_samples==23040, 'incomplete fixed7-policy/21-task/23040-response campaign')
    require(all(s['parent_training_identity_sha256']==shards[0]['parent_training_identity_sha256'] for s in shards), 'cross-shard training trajectory drift')
    require(len({s['model_binding']['initial_hash'] for s in shards})==1 and all(s['sampler']==shards[0]['sampler'] for s in shards), 'cross-shard initial-policy/sampler drift')
    checkpoints={}
    for step in (32,64,128):
        groups={}
        for cohort,ids in (('train',TRAIN_IDS),('development',DEV_IDS)):
            n=(128 if cohort=='train' else 32) if step in (32,64) else (512 if cohort=='train' else 128)
            arms={arm:{t:metrics(policies[(arm,0 if arm=='base' else step)][t][:n]) for t in ids} for arm in ('base','maxrl','remax')}
            comparison=common_eligibility(arms); contrasts={}
            for treatment,control in (('remax','maxrl'),('remax','base'),('maxrl','base')):
                contrasts[treatment+'_minus_'+control]={'macro_accuracy':sum(arms[treatment][t]['accuracy']-arms[control][t]['accuracy'] for t in ids)/len(ids),'macro_pass_at_k':{k:sum(arms[treatment][t]['pass_at_k'][k]-arms[control][t]['pass_at_k'][k] for t in ids)/len(ids) for k in ('1','8','32')},'macro_expected_distinct_valid_modes_at_k':{k:sum(arms[treatment][t]['expected_distinct_valid_modes_at_k'][k]-arms[control][t]['expected_distinct_valid_modes_at_k'][k] for t in ids)/len(ids) for k in ('1','8','32')}}
            groups[cohort]={'task_ids':ids,'samples_per_task':n,'task_metrics':arms,**comparison,'contrasts':contrasts}
        checkpoints[str(step)]={'role':'primary_final' if step==128 else 'descriptive_trajectory','cohorts':groups}
    return {'schema':SCHEMA,'status':'pass','kind':'complete_fixed_native_code128_comparison','schedule_sha256':digest(schedule_path),'auditor_sha256':digest(Path(__file__)),'samples':all_samples,'shards':shards,'checkpoints':checkpoints,'primary_checkpoint':128,'paired_training_seeds':1,'uncertainty':'One paired seed establishes no training-seed confidence interval; checkpoint32/64 are descriptive only.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__); choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--run',type=Path); choice.add_argument('--schedule',type=Path); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--declared-allocation-minutes',type=int,required=True,
        help='training wall allocation the run actually carried; stated so it cannot drift silently')
    parser.add_argument('--declared-allocation-gpu-hours',type=float,required=True,
        help='training max_gpu_hours the run actually carried')
    args=parser.parse_args()
    DECLARED_ALLOCATION['minutes']=args.declared_allocation_minutes
    DECLARED_ALLOCATION['gpu_hours']=args.declared_allocation_gpu_hours
    require(not args.output.exists(),'audit output must be new')
    result=audit_run(args.run) if args.run else assemble(args.schedule)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'output':str(args.output)}))


if __name__=='__main__':
    main()
