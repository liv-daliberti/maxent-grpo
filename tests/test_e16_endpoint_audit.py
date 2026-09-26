from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ops.exp_scaling import audit_e16_canonical_endpoint as audit
from ops.exp_scaling.audit_e14_checkpoint import (
    enumerate_exact_policy as enumerate_e14_exact_policy,
)


def _tree(task: str, *, uniform: bool) -> dict[tuple[int, ...], torch.Tensor]:
    geometry = audit.policy_geometry(task)
    result = {}
    for ordinal, prefix in enumerate(geometry.prefixes):
        size = geometry.support_sizes[len(prefix)]
        if uniform:
            values = torch.zeros(size, dtype=torch.float64)
        else:
            values = torch.linspace(
                -0.17 * (ordinal + 1),
                0.11 * (ordinal + 2),
                size,
                dtype=torch.float64,
            )
        result[prefix] = torch.log_softmax(values, dim=0)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_endpoint_cell_identity_binds_protocol_execution_run_and_job(
    tmp_path, monkeypatch
):
    execution_root = tmp_path / "execution"
    auditor = execution_root / "ops/exp_scaling/audit_e16_canonical_endpoint.py"
    auditor.parent.mkdir(parents=True)
    auditor.write_text("# frozen auditor\n")
    source = execution_root / "src"
    source.mkdir()
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol\n")
    run_stamp = "gce16_canonical_maxent_joint_smoke_v3_maxent_s9006"
    run_dir = tmp_path / f"run_{run_stamp}"
    run_dir.mkdir()
    identity = {
        "schema": "e16_canonical_campaign_identity_v1",
        "protocol": "E16",
        "stage": "smoke",
        "task": "graph_coloring",
        "prefix": "gce16_canonical_maxent_joint_smoke_v3",
        "auto_resume": False,
        "watchdog_requeue": False,
        "protocol_path": str(protocol),
        "protocol_sha256": _sha256(protocol),
        "source_snapshot": str(source),
        "source_hash": "a" * 64,
        "source_snapshot_hash": "a" * 64,
        "execution_snapshot_root": str(execution_root),
        "execution_surface_hash": "b" * 64,
        "execution_identity": {
            "schema": "e16_execution_surface_identity_v1",
            "sha256": "b" * 64,
            "files": [
                {
                    "path": "ops/exp_scaling/audit_e16_canonical_endpoint.py",
                    "sha256": _sha256(auditor),
                }
            ],
        },
        "task_config": {
            "seeds": [9006],
            "stage": "smoke",
            "arms": {arm: {} for arm in audit.ARMS},
            "task": {"target_optimizer_updates": 32},
        },
    }
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity, sort_keys=True))
    monkeypatch.setattr(audit, "AUDIT_SOURCE_ROOT", source)
    monkeypatch.setattr(audit, "__file__", str(auditor))
    cell, binding = audit._load_smoke_cell_identity(
        identity_path=identity_path,
        expected_identity_sha256=_sha256(identity_path),
        task="graph_coloring",
        arm="maxent",
        seed=9006,
        run_stamp=run_stamp,
        job_id="12345",
        run_dir=run_dir,
    )
    assert cell == {
        "task": "graph_coloring",
        "arm": "maxent",
        "seed": 9006,
        "run_stamp": run_stamp,
        "job_id": "12345",
    }
    assert binding["sha256"] == _sha256(identity_path)
    assert binding["protocol_sha256"] == _sha256(protocol)
    assert binding["auditor"]["sha256"] == _sha256(auditor)

    with pytest.raises(ValueError, match="run stamp"):
        audit._load_smoke_cell_identity(
            identity_path=identity_path,
            expected_identity_sha256=_sha256(identity_path),
            task="graph_coloring",
            arm="maxent_dual",
            seed=9006,
            run_stamp=run_stamp,
            job_id="12345",
            run_dir=run_dir,
        )


def _teacher_forced(
    task: str, tree: dict[tuple[int, ...], torch.Tensor]
) -> torch.Tensor:
    geometry = audit.policy_geometry(task)
    return torch.stack(
        [
            torch.stack(
                [
                    tree[action[:depth]][action[depth]]
                    for depth in range(geometry.horizon)
                ]
            )
            for action in geometry.action_indices
        ]
    )


@pytest.mark.parametrize(
    ("task", "leaf_count"), (("graph_coloring", 27), ("countdown", 108))
)
def test_uniform_finite_tree_recovers_normalization_and_entropy_chain(
    task: str, leaf_count: int
):
    exact = audit.enumerate_exact_policy(_tree(task, uniform=True), task=task)
    assert exact.geometry.leaf_count == leaf_count
    assert exact.probability_sum == pytest.approx(1.0, abs=1e-12)
    assert exact.leaf_entropy == pytest.approx(math.log(leaf_count), abs=1e-12)
    assert exact.conditional_entropy == pytest.approx(
        exact.leaf_entropy, abs=1e-12
    )


def test_graph_generic_audit_is_numerically_identical_to_e14_math():
    tree = _tree("graph_coloring", uniform=False)
    generic = audit.enumerate_exact_policy(tree, task="graph_coloring")
    e14 = enumerate_e14_exact_policy(tree)
    assert torch.equal(generic.leaf_log_probabilities, e14.leaf_log_probabilities)
    assert torch.equal(generic.leaf_probabilities, e14.leaf_probabilities)
    assert generic.probability_sum == e14.probability_sum
    assert generic.leaf_entropy == e14.leaf_entropy
    assert generic.conditional_entropy == e14.conditional_entropy


def test_nonuniform_countdown_tree_matches_teacher_forcing():
    tree = _tree("countdown", uniform=False)
    exact = audit.enumerate_exact_policy(tree, task="countdown")
    comparison = audit.compare_tree_and_teacher_forced(
        tree, _teacher_forced("countdown", tree), task="countdown"
    )
    assert exact.probability_sum == pytest.approx(1.0, abs=1e-12)
    assert exact.leaf_entropy == pytest.approx(
        exact.conditional_entropy, abs=1e-12
    )
    assert comparison == {
        "per_token_max_abs_error": 0.0,
        "sequence_max_abs_error": 0.0,
    }


def test_valid_metrics_aggregate_semantic_aliases_before_entropy():
    exact = audit.enumerate_exact_policy(
        _tree("countdown", uniform=True), task="countdown"
    )
    rewards = [0] * 108
    rewards[0] = rewards[1] = 1
    keys = [f"key-{index}" for index in range(108)]
    keys[1] = keys[0]
    metrics = audit.valid_policy_metrics(exact, rewards, keys)
    assert metrics["valid_action_count"] == 2
    assert metrics["valid_semantic_key_count"] == 1
    assert metrics["p_valid"] == pytest.approx(2.0 / 108.0, abs=1e-12)
    assert metrics["h_valid"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["n_eff_valid"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["h_valid_action"] == pytest.approx(math.log(2.0), abs=1e-12)
    assert metrics["n_eff_valid_action"] == pytest.approx(2.0, abs=1e-12)

    conflicting = list(rewards)
    conflicting[1] = 0
    with pytest.raises(ValueError, match="conflicting grader rewards"):
        audit.valid_policy_metrics(exact, conflicting, keys)


class _ForbiddenLabel:
    def __int__(self):
        raise AssertionError("the decoder read a label")

    def __str__(self):
        raise AssertionError("the decoder read a label")


def test_countdown_codec_is_bijective_and_drops_target_before_decoding():
    reference = {
        "verifier": "countdown",
        "numbers": [2, 4, 8],
        "target": _ForbiddenLabel(),
        "solutions": _ForbiddenLabel(),
    }
    public = audit.public_decoder_reference("countdown", reference)
    assert public == {"verifier": "countdown", "numbers": [2, 4, 8]}
    records = audit.validate_code_decoder_bijection("countdown", reference)
    assert len(records) == 108
    assert len({record.code for record in records}) == 108
    assert len({record.decoded_response for record in records}) == 108
    assert len({record.semantic_key for record in records}) == 108
    assert records[0].code == "111"
    assert records[-1].code == "636"


def test_codec_bijection_check_fails_closed_on_decoder_alias(monkeypatch):
    monkeypatch.setattr(
        audit,
        "decode_canonical_action_response",
        lambda task, code, reference: "1 + 2 + 3",
    )
    with pytest.raises(ValueError, match="not bijective"):
        audit.validate_code_decoder_bijection(
            "countdown", {"verifier": "countdown", "numbers": [1, 2, 3]}
        )


def test_tree_and_metrics_fail_closed_on_malformed_inputs():
    tree = _tree("countdown", uniform=True)
    missing = dict(tree)
    missing.pop((0, 0))
    with pytest.raises(ValueError, match="prefix tree mismatch"):
        audit.enumerate_exact_policy(missing, task="countdown")

    wrong_width = dict(tree)
    wrong_width[(0,)] = torch.full((6,), -math.log(6.0), dtype=torch.float64)
    with pytest.raises(ValueError, match="must have 3"):
        audit.enumerate_exact_policy(wrong_width, task="countdown")

    unnormalized = dict(tree)
    unnormalized[()] = torch.zeros(6, dtype=torch.float64)
    with pytest.raises(ValueError, match="not normalized"):
        audit.enumerate_exact_policy(unnormalized, task="countdown")

    exact = audit.enumerate_exact_policy(tree, task="countdown")
    with pytest.raises(ValueError, match="at least one valid"):
        audit.valid_policy_metrics(exact, [0] * 108, [str(i) for i in range(108)])


class _PositionalToyModel(torch.nn.Module):
    """Causal table whose logits change with each observed prefix."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def forward(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
        del use_cache
        self.calls.append(
            (input_ids.detach().clone(), attention_mask.detach().clone(), logits_to_keep)
        )
        batch, sequence = input_ids.shape
        logits = torch.full((batch, sequence, 12), -100.0, device=input_ids.device)
        cumulative = input_ids.cumsum(dim=1).float()
        for token_id in range(1, 7):
            logits[:, :, token_id] = (
                0.013 * token_id * cumulative + 0.07 * token_id
            )
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])


@pytest.mark.parametrize(
    ("task", "supports"),
    (
        ("graph_coloring", ((1, 2, 3),) * 3),
        ("countdown", ((1, 2, 3, 4, 5, 6), (1, 2, 3), (1, 2, 3, 4, 5, 6))),
    ),
)
def test_cpu_model_prefix_tree_matches_full_leaf_teacher_forcing(task, supports):
    model = _PositionalToyModel()
    prompts = [[7], [8, 9]]
    trees = audit.infer_prefix_tree_log_probabilities(
        model,
        prompts,
        task=task,
        action_token_ids_by_position=supports,
        pad_token_id=0,
        device=torch.device("cpu"),
        batch_size=7,
    )
    prefix_calls = list(model.calls)
    model.calls.clear()
    teacher = audit.infer_teacher_forced_log_probabilities(
        model,
        prompts,
        task=task,
        action_token_ids_by_position=supports,
        pad_token_id=0,
        device=torch.device("cpu"),
        batch_size=7,
    )
    teacher_calls = list(model.calls)
    assert prefix_calls and teacher_calls
    for input_ids, attention_mask, logits_to_keep in prefix_calls + teacher_calls:
        assert input_ids.shape[0] == 7
        assert input_ids.shape[1] in {4, 5}
        assert torch.equal(attention_mask, torch.ones_like(attention_mask))
        assert logits_to_keep == 4
    for tree, observed in zip(trees, teacher, strict=True):
        exact = audit.enumerate_exact_policy(tree, task=task)
        comparison = audit.compare_tree_and_teacher_forced(
            tree, observed, task=task
        )
        assert exact.probability_sum == pytest.approx(1.0, abs=1e-6)
        assert comparison["per_token_max_abs_error"] == pytest.approx(0.0)
        assert comparison["sequence_max_abs_error"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("task", "data_root", "eval_count", "expected_hash"),
    (
        (
            "graph_coloring",
            Path("var/data/exact_answer_mode_probe"),
            96,
            audit.EXPECTED_GRAPH_CONTENT_HASH,
        ),
        (
            "countdown",
            Path("var/data/exact_countdown_easy3_probe"),
            128,
            audit.EXPECTED_COUNTDOWN_CONTENT_HASH,
        ),
    ),
)
def test_full_frozen_dataset_satisfies_codec_and_valid_support_contract(
    task: str, data_root: Path, eval_count: int, expected_hash: str
):
    rows, observed_hash = audit._load_frozen_rows(task, data_root)
    assert len(rows) == eval_count
    assert observed_hash == expected_hash
    for row in rows:
        codec = audit.validate_code_decoder_bijection(task, row["answer"])
        assert len(codec) == audit.policy_geometry(task).leaf_count
