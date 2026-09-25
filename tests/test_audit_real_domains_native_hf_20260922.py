import copy
import hashlib
import itertools
import json

import pytest

import audit_real_domains_native_hf_20260922 as audit


def verdict(key):
    return {"accepted": key is not None, "canonical_key": key, "hard_violations": []}


def test_discovery_matches_exhaustive_subsets():
    for n in range(1, 9):
        for m in range(n + 1):
            for k in range(1, n + 1):
                subsets = list(itertools.combinations(range(n), k))
                truth = sum(any(i < m for i in s) for s in subsets) / len(subsets)
                assert audit.discovery(n, m, k) == pytest.approx(truth, abs=1e-14)


def test_pcmd_threshold_denominator_and_unbiased_pairs():
    rows = [verdict("a")] * 15 + [verdict("b")] * 15 + [verdict(None)] * 2
    metric = audit.metrics(rows)
    assert metric["samples"] == 32 and metric["accepted"] == 30
    assert metric["accuracy"] == 30 / 32
    assert metric["pcmd"] == pytest.approx(450 / (30 * 29))
    assert metric["expected_distinct_valid_modes_at_k"]["32"] == 2
    assert audit.metrics(rows[1:])["pcmd"] is None


def test_all_wrong_never_fills_pcmd_with_zero():
    result = audit.metrics([verdict(None)] * 32)
    assert result["accuracy"] == 0 and result["pcmd"] is None
    assert set(result["expected_distinct_valid_modes_at_k"].values()) == {0.0}


def test_pairwise_and_all_three_eligibility_are_explicit():
    eligible = audit.metrics([verdict("a")] * 15 + [verdict("b")] * 15 + [verdict(None)] * 2)
    ineligible = audit.metrics([verdict(None)] * 32)
    arms = {"base": {"a": ineligible, "b": eligible}, "maxrl": {"a": eligible, "b": ineligible}, "remax": {"a": eligible, "b": eligible}}
    result = audit.common_eligibility(arms)
    assert result["common_pair"]["task_ids"] == ["a"]
    assert result["common_all_three"]["task_ids"] == []
    assert result["common_all_three"]["remax_minus_maxrl"] is None
    assert result["arm_wise"]["remax"]["pcmd_eligible_tasks"] == 2
    arms["remax"]["b"] = audit.metrics([verdict(None)] * 128)
    with pytest.raises(ValueError, match="denominators"):
        audit.common_eligibility(arms)


class Codec:
    def encode(self, text, **kwargs):
        return [1, 2]

    def decode(self, tokens, **kwargs):
        return "x"


@pytest.fixture
def raw():
    return {"task_id": "1513_A", "sample_index": 5, "global_task_index": 8,
            "batch_seed": audit.SAMPLING_SEED + 80000 + 4, "batch_offset": 1,
            "arm": "remax", "step": 128, "checkpoint_step": 128, "batch_start_index": 4,
            "request_seed": audit.SAMPLING_SEED + 80000 + 4, "cohort": "development",
            "phase": "native_hf:remax:development", "request_id": "native_hf:remax:development:128:1513_A:5",
            "prompt_token_ids": [1, 2],
            "prompt_sha256": hashlib.sha256(b"prompt").hexdigest(),
            "token_ids": [3, 9], "token_count": 2, "text": "x",
            "text_sha256": hashlib.sha256(b"x").hexdigest(), "finish_reason": "eos"}


def tokens(row):
    return audit.validate_tokens(row, prompt="prompt", prompt_tokens=[1, 2], codec=Codec(), eos_ids={8, 9}, upper=10, max_new_tokens=4)


def test_native_token_contract(raw):
    tokens(raw)
    audit.validate_sample_identity(raw, arm="remax", step=128, counts={"1513_A": 128})


@pytest.mark.parametrize("field,value", [("token_ids", [3, 10]), ("token_ids", [8, 9]), ("token_ids", []),
                                       ("token_ids", [3]), ("prompt_token_ids", [2, 1]),
                                       ("text_sha256", "wrong"), ("finish_reason", "length"), ("token_count", 3)])
def test_changed_token_contract_fails(raw, field, value):
    raw[field] = value
    with pytest.raises(ValueError):
        tokens(raw)


@pytest.mark.parametrize("field,value", [("global_task_index", 0), ("batch_seed", audit.SAMPLING_SEED + 4),
                                       ("batch_offset", 5), ("arm", "maxrl"), ("step", 96), ("sample_index", 128)])
def test_shard_index_seed_or_policy_drift_fails(raw, field, value):
    raw[field] = value
    with pytest.raises(ValueError):
        audit.validate_sample_identity(raw, arm="remax", step=128, counts={"1513_A": 128})


def test_fixed_training_requests_pass_and_task_deletion_fails():
    root = audit.Path(__file__).resolve().parents[1] / "var/artifacts/real_domains_pilot_20260921/code_corrected128_plan_unfrozen_unsubmitted"
    request = json.loads((root / "train_maxrl_request.json").read_text())
    assert audit.validate_training_request(request)["status"] == "pass"
    request["config"]["train_ids"].remove("1408_A")
    with pytest.raises(ValueError, match="train_ids"):
        audit.validate_training_request(request)


def test_checkpoint_requires_registered_step_and_full_seal(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "bank.json").write_text("{}")
    (tmp_path / "training.pt").write_bytes(b"optimizer")
    seal = {"schema": audit.training_audit.TRAINER_SCHEMA, "arm": "maxrl", "completed_updates": 32,
            "config_sha256": "fixed", "bank_sha256": audit.digest(tmp_path / "bank.json"),
            "training_state_sha256": audit.digest(tmp_path / "training.pt"),
            "adapter_files": {"adapter_model.safetensors": audit.digest(adapter / "adapter_model.safetensors")}}
    (tmp_path / "complete.json").write_text(json.dumps(seal))
    assert audit.audit_checkpoint(tmp_path, {"config_sha256": "fixed"}, arm="maxrl", step=32)["step"] == 32
    with pytest.raises(ValueError, match="unregistered"):
        audit.audit_checkpoint(tmp_path, {"config_sha256": "fixed"}, arm="maxrl", step=96)
    (tmp_path / "training.pt").write_bytes(b"changed optimizer")
    with pytest.raises(ValueError, match="data seal"):
        audit.audit_checkpoint(tmp_path, {"config_sha256": "fixed"}, arm="maxrl", step=32)

def test_loaded_adapter_hash_reconstructed_from_saved_fp32(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    weights = {"base_model.model.layer.lora_A.weight": np.array([[1, 2]], dtype=np.float32),
               "base_model.model.layer.lora_B.weight": np.array([[3], [4]], dtype=np.float32)}
    (tmp_path / "adapter").mkdir()
    save_file(weights, str(tmp_path / "adapter/adapter_model.safetensors"))
    expected = {k.replace(".weight", ".default.weight"): hashlib.sha256(v.tobytes()).hexdigest() for k, v in weights.items()}
    identity = {"config": {"arm": "maxrl", "checkpoint_path": str(tmp_path)},
                "model": {"loaded_trainable_parameters_sha256": audit.canonical(expected), "trainable_parameters": 4}}
    audit.audit_loaded_adapter(identity)
    identity["model"]["loaded_trainable_parameters_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sealed FP32"):
        audit.audit_loaded_adapter(identity)


@pytest.fixture
def complete_shards(tmp_path):
    result = []
    for arm, step in [("base", 0)] + [(a, s) for a in ("maxrl", "remax") for s in (32, 64, 128)]:
        groups = [audit.GLOBAL_IDS] if step in (32, 64) else [audit.TRAIN_IDS[:4], audit.TRAIN_IDS[4:], audit.DEV_IDS]
        for group_index, ids in enumerate(groups):
            point = tmp_path / f"{arm}_{step}_{group_index}"
            point.mkdir()
            task_metrics, all_rows = {}, []
            for task in ids:
                n = (512 if task in audit.TRAIN_IDS else 128) if step in (0, 128) else (128 if task in audit.TRAIN_IDS else 32)
                prefix = 128 if task in audit.TRAIN_IDS else 32
                rows = [{"task_id": task, "sample_index": i, **verdict(None if arm == "base" and i < prefix else "a" if i % 2 else "b")} for i in range(n)]
                task_metrics[task] = audit.metrics(rows)
                all_rows += rows
            attempts = point / "attempts.jsonl"
            attempts.write_text("".join(json.dumps(r) + "\n" for r in all_rows))
            receipt = point / "evaluation.json"
            receipt.write_text(json.dumps({"task_results": [{"task_id": t, **m} for t, m in task_metrics.items()]}))
            result.append({"schema": audit.SCHEMA, "status": "pass", "kind": "shard", "arm": arm, "step": step,
                           "plan_sha256": "fixed", "initial_trainable_parameters_sha256": "fixed",
                           "task_metrics": task_metrics, "receipt": str(receipt), "receipt_sha256": audit.digest(receipt),
                           "raw_artifacts": {"attempts": {"path": str(attempts), "sha256": audit.digest(attempts)}}})
    return result


def test_full_study_retains_fixed_denominators_and_base_prefix(complete_shards):
    result = audit.combine_shards(complete_shards)
    assert result["requested_completions"] == 23040 and result["primary_checkpoint"] == 128
    for cohort, count, final_count in (("train", 8, 512), ("development", 13, 128)):
        monitor = result["trajectories"]["32"]["cohorts"][cohort]
        final = result["trajectories"]["128"]["cohorts"][cohort]
        assert monitor["arm_wise"]["base"]["macro_accuracy"] == 0
        assert final["arm_wise"]["base"]["macro_accuracy"] == .75
        assert monitor["common_all_three"]["eligible_tasks"] == 0
        assert final["common_all_three"]["eligible_tasks"] == count
        assert final["arm_wise"]["base"]["samples"] == count * final_count
    assert "1408_A" in result["trajectories"]["128"]["cohorts"]["train"]["task_metrics"]["base"]


@pytest.mark.parametrize("change", ["missing_shard", "duplicate_shard", "delete_1408", "changed_plan", "changed_receipt"])
def test_incomplete_or_changed_shard_union_rejected(complete_shards, change):
    if change == "missing_shard":
        complete_shards.pop()
    elif change == "duplicate_shard":
        complete_shards[-1] = copy.deepcopy(complete_shards[-2])
    elif change == "delete_1408":
        complete_shards[0]["task_metrics"].pop("1408_A")
    elif change == "changed_plan":
        complete_shards[0]["plan_sha256"] = "different"
    elif change == "changed_receipt":
        audit.Path(complete_shards[0]["receipt"]).write_text("changed")
    with pytest.raises(ValueError):
        audit.combine_shards(complete_shards)


def test_validation_receipt_cannot_enter_production_shard_union(complete_shards):
    complete_shards[0]["kind"] = "validation"
    with pytest.raises(ValueError, match="independently passing"):
        audit.combine_shards(complete_shards)


def test_agreed_cli_is_reachable():
    import subprocess
    import sys
    completed = subprocess.run([sys.executable, str(audit.Path(audit.__file__)), '--help'], text=True, capture_output=True)
    assert completed.returncode == 0
    assert '--run RUN' in completed.stdout and '--schedule SCHEDULE' in completed.stdout
    assert '--shard' not in completed.stdout


def test_schedule_cli_assembly_uses_common_prefix_and_fixed_final(complete_shards, tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    sources = tmp_path / 'sources.json'
    sources.write_text('{}')
    adapted = {}
    shards = []
    for old in complete_shards:
        root = audit.Path(old['receipt']).parent
        adapted[root] = {'validation_only': False, 'arm': old['arm'], 'checkpoint_step': old['step'],
                         'plan_sha256': audit.digest(plan), 'parent_training_identity_sha256': {'maxrl':'one','remax':'two'},
                         'task_metrics': old['task_metrics'], 'artifacts': old['raw_artifacts'],
                         'model_binding': {'initial_hash':'common'}, 'sampler': {'policy':'same'}}
        shards.append({'name':root.name, 'arm':old['arm'], 'checkpoint_step':old['step'],
                       'samples_per_task_by_id':{t:m['samples'] for t,m in old['task_metrics'].items()}})
    schedule = {'schema':'native-hf-endpoint-schedule-20260922-v1','shards':shards,'total_completions':23040,
                'run_parent':str(tmp_path),'plan_path':str(plan),'plan_sha256':audit.digest(plan),
                'source_contract_path':str(sources),'source_contract_sha256':audit.digest(sources)}
    path = tmp_path / 'schedule.json'
    path.write_text(json.dumps(schedule))
    monkeypatch.setattr(audit, 'audit_run', lambda p: copy.deepcopy(adapted[p]))
    summary = audit.assemble(path)
    assert summary['samples'] == 23040 and summary['primary_checkpoint'] == 128
    for cohort, tasks in [('train',8),('development',13)]:
        monitor = summary['checkpoints']['32']['cohorts'][cohort]
        final = summary['checkpoints']['128']['cohorts'][cohort]
        assert monitor['arm_wise']['base']['macro_accuracy'] == 0
        assert final['arm_wise']['base']['macro_accuracy'] == .75
        assert monitor['common_all_three']['eligible_tasks'] == 0
        assert final['common_all_three']['eligible_tasks'] == tasks
        assert monitor['contrasts']['remax_minus_base']['macro_accuracy'] == 1
    adapted[next(iter(adapted))]['parent_training_identity_sha256']['maxrl'] = 'different'
    with pytest.raises(ValueError, match='training trajectory drift'):
        audit.assemble(path)
