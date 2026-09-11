from __future__ import annotations

import math
from pathlib import Path

from oat_drgrpo.interactive_episode_replay import (
    InteractiveDecisionRecord,
    InteractiveEpisodeRecord,
    VerifiedInteractiveReplayBank,
)
from train_point_maze_interactive_paired_smoke_v1 import (
    HORIZON,
    REPLAY_CAPACITY,
    _replay_slots,
)


ROOT = Path(__file__).resolve().parents[1]


def _episode(key: str, selected: int) -> InteractiveEpisodeRecord:
    decision = InteractiveDecisionRecord(
        prompt_token_ids=(1, 2, 3),
        allowed_token_ids=tuple(range(10, 19)),
        selected_token_id=selected,
        behavior_logprobs=tuple([-math.log(9)] * 9),
        transition_sha256="a" * 64,
    )
    return InteractiveEpisodeRecord(
        group_prompt_token_ids=(7, 8),
        outcome_key=key,
        task_reward=1.0,
        decisions=(decision,),
    )


def test_replay_pads_every_mode_to_the_full_decision_horizon():
    bank = VerifiedInteractiveReplayBank(capacity=REPLAY_CAPACITY)
    bank.observe_group([_episode("route-a", 10), _episode("route-b", 11)])
    group = bank.schedule_one_global_round_robin()
    action_ids = tuple(range(10, 19))

    control, control_diag = _replay_slots(
        group=group,
        padding_token_ids=(1, 2),
        action_token_ids=action_ids,
        compute_only=True,
    )
    treatment, treatment_diag = _replay_slots(
        group=group,
        padding_token_ids=(1, 2),
        action_token_ids=action_ids,
        compute_only=False,
    )

    assert len(control) == len(treatment) == REPLAY_CAPACITY * HORIZON
    assert control_diag["replay_decision_forward_slots"] == 1536
    assert control_diag["replay_raw_score_gradient_l2"] > 0
    assert control_diag["replay_applied_score_gradient_l2"] == 0
    assert treatment_diag["replay_applied_score_gradient_l2"] > 0
    assert sum(slot["active"] for slot in control) == 2


def test_audit_requires_full_simulator_and_prompt_replay():
    audit = (
        ROOT / "ops/audit_point_maze_interactive_paired_smoke_v1.py"
    ).read_text()
    assert "interactive_transition_sha256" in audit
    assert "render_point_policy_prompt_v3" in audit
    assert 'metric.get("replay_decision_forward_slots") != 1536.0' in audit
    assert 'metric.get("policy_microbatch_size") != (16 if version in {3, 4} else 4)' in audit
    assert "canonical key mismatch" in audit


def test_v2_uses_fresh_rows_seed_and_microbatch_four():
    launcher = (
        ROOT / "ops/exp_scaling/launch_point_maze_interactive_paired_smoke_v1.py"
    ).read_text()
    batch = (
        ROOT / "ops/slurm/train_point_maze_interactive_paired_smoke_v1.slurm"
    ).read_text()
    protocol = (
        ROOT / "paper/preregistration/point_maze_interactive_paired_smoke_v2_20260730.md"
    ).read_text()
    assert 'ROW_INDICES = (0, 2, 4, 6) if cohort in {"v1", "v4"} else (1, 3, 5, 7)' in launcher
    assert 'SEED = 75300 + version' in launcher
    assert '"policy_microbatch_size": 16 if COHORT in {"v3", "v4"} else 4' in launcher
    assert '[[ "$COHORT" == "v3" || "$COHORT" == "v4" ]] && MICROBATCH=16' in batch
    assert '`75302`' in protocol
    assert 'rows `1,3,5,7`' in protocol


def test_v3_binds_failed_v2_and_exact_batch16_diagnostic():
    launcher = (
        ROOT / "ops/exp_scaling/launch_point_maze_interactive_paired_smoke_v1.py"
    ).read_text()
    protocol = (
        ROOT / "paper/preregistration/point_maze_interactive_paired_smoke_v3_20260730.md"
    ).read_text()
    assert 'choices=("v1", "v2", "v3", "v4")' in launcher
    assert '"failed_predecessor_jobs": [30202913, 30202914, 30202915]' in launcher
    assert '"v2_batch_diagnostic_sha256": sha(V2_DIAGNOSTIC)' in launcher
    assert "fresh development seed 75303" in protocol
    assert "policy/replay microbatch 16" in protocol


def test_v4_is_frozen_geometry_shift_replacement():
    launcher = (
        ROOT / "ops/exp_scaling/launch_point_maze_interactive_paired_smoke_v1.py"
    ).read_text()
    train_batch = (
        ROOT / "ops/slurm/train_point_maze_interactive_paired_smoke_v1.slurm"
    ).read_text()
    audit_batch = (
        ROOT / "ops/slurm/audit_point_maze_interactive_paired_smoke_v1.slurm"
    ).read_text()
    protocol = (
        ROOT
        / "paper/preregistration/point_maze_geometry_shift_paired_smoke_v1_20260730.md"
    ).read_text()
    assert 'choices=("v1", "v2", "v3", "v4")' in launcher
    assert 'SEED = 75300 + version' in launcher
    assert 'COHORT in {"v3", "v4"}' in launcher
    assert 'point_maze_geometry_shift_v1' in train_batch
    assert 'point_maze_geometry_shift_v1' in audit_batch
    assert "fresh seed 75304" in protocol
    assert "evaluation rows are not loaded" in protocol
