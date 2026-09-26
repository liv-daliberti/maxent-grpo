from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from oat_drgrpo.args import validate_zero_math_args
from oat_drgrpo.semantic_shannon import SemanticShannonTracker


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e81_semantic_maxent_verified_replay_05b.py"
E78_LAUNCHER = ROOT / "ops/exp_scaling/launch_e78_verified_replay_only_05b.py"
PROTOCOL = ROOT / (
    "paper/preregistration/e81_semantic_maxent_on_verified_replay_05b_20260806.md"
)
EXPERIMENT_SH = ROOT / "ops/run_experiment.sh"

SEMANTIC_KEYS = {
    "OAT_ZERO_VARIANT",
    "OAT_ZERO_SEMANTIC_SHANNON_COEF",
    "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP",
    "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT",
    "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE",
    "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE",
    "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE",
    "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY",
}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shell_case_body(name: str) -> list[str]:
    """Return the exported lines of one run_experiment.sh variant branch."""

    text = EXPERIMENT_SH.read_text(encoding="utf-8")
    match = re.search(rf"^  {re.escape(name)}\)\n(.*?)^    ;;$", text, re.M | re.S)
    assert match is not None, f"variant {name} is absent from run_experiment.sh"
    return [
        line.strip()
        for line in match.group(1).splitlines()
        if line.strip().startswith("export ") or line.strip().startswith("VARIANT_TAG=")
    ]


# --------------------------------------------------------------------------
# cohort shape
# --------------------------------------------------------------------------


def test_e81_is_five_domains_one_new_arm_five_seeds_and_eight_passes():
    e81 = load(LAUNCHER, "e81_launcher")
    assert len(e81.DOMAINS) == 5
    assert e81.ARM == "semantic"
    assert e81.SEEDS == (43, 44, 45, 46, 47)
    assert e81.PASSES == 8
    assert e81.TRAIN_ROWS == 384
    assert e81.TARGET_STEPS == 3072
    assert e81.CHECKPOINT_INTERVAL == 192
    assert e81.SEMANTIC_COEF == 0.10
    assert e81.REPLAY_WEIGHT == 0.10


def test_e81_reuses_the_e78_domains_seeds_and_schedule_exactly():
    e78 = load(E78_LAUNCHER, "e78_launcher")
    e81 = load(LAUNCHER, "e81_launcher")
    assert tuple(e81.DOMAINS) == tuple(e78.DOMAINS)
    assert e81.SEEDS == e78.SEEDS
    assert e81.PASSES == e78.PASSES
    assert e81.TRAIN_ROWS == e78.TRAIN_ROWS
    assert e81.CHECKPOINT_INTERVAL == e78.CHECKPOINT_INTERVAL
    assert e81.SOURCE_MANIFEST == e78.SOURCE_MANIFEST
    assert e81.MODEL_TAG == e78.MODEL_TAG


# --------------------------------------------------------------------------
# the semantic term is the only applied difference from E78 replay
# --------------------------------------------------------------------------


def test_e81_objective_differs_from_e78_replay_only_in_the_semantic_term():
    e78 = load(E78_LAUNCHER, "e78_launcher")
    e81 = load(LAUNCHER, "e81_launcher")
    replay = e78.fixed_objective("replay")
    semantic = e81.fixed_objective()

    shared_replay = {k: v for k, v in replay.items() if k not in SEMANTIC_KEYS}
    shared_semantic = {k: v for k, v in semantic.items() if k not in SEMANTIC_KEYS}
    assert shared_replay == shared_semantic

    # replay itself is live and identically dosed in both arms
    assert semantic["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
    assert replay["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
    assert semantic["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "verified_likelihood_per_rollout"
    )
    assert semantic["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"

    # the treatment, fixed and un-adapted
    assert semantic["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.1"
    assert semantic["OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE"] == "1"
    assert (
        semantic["OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE"]
        == "1"
    )
    assert semantic["OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE"] == "0"


def test_e81_leaves_every_other_actuator_off():
    e81 = load(LAUNCHER, "e81_launcher")
    semantic = e81.fixed_objective()
    for key in (
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA",
        "OAT_ZERO_MAXENT_ALPHA",
        "OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO",
        "OAT_ZERO_MAXENT_DUAL_TARGET_RATIO",
        "OAT_ZERO_POLICY_ENTROPY_COEF",
        "OAT_ZERO_SEED_ENTROPY_ALPHA",
        "OAT_ZERO_BETA",
    ):
        assert float(semantic[key]) == 0.0
    for key in (
        "OAT_ZERO_MAXENT_INVERSE_ADAPTATION",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT",
    ):
        assert semantic[key] == "0"


def test_e81_shell_variant_is_the_e78_replay_branch_plus_semantic_exports_only():
    replay = shell_case_body("verified_first_replay_rehearsal_only")
    semantic = shell_case_body("verified_replay_semantic_maxent")

    def strip(lines: list[str]) -> list[str]:
        return [
            line
            for line in lines
            if "SEMANTIC_SHANNON" not in line and not line.startswith("VARIANT_TAG=")
        ]

    assert strip(replay) == strip(semantic)
    assert 'VARIANT_TAG="verified_replay_semantic_maxent"' in semantic
    assert (
        'export OAT_ZERO_SEMANTIC_SHANNON_COEF="$SEMANTIC_SHANNON_COEF"' in semantic
    )
    assert "export OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1" in semantic
    assert (
        "export OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1"
        in semantic
    )
    assert "export OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE=0" in semantic
    # the E78 branch is untouched
    assert "export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0" in replay


def test_e81_snapshot_patch_set_is_exactly_two_additive_files():
    e81 = load(LAUNCHER, "e81_launcher")
    assert sorted(e81.PATCHED_FILES) == [
        "ops/run_experiment.sh",
        "src/oat_drgrpo/args.py",
    ]


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------


def _validation_args(**overrides):
    helper = load(ROOT / "tests/test_args.py", "e81_args_helper")
    values = dict(
        xdr_tau=float("inf"),
        online_canonical_replay=True,
        online_canonical_replay_objective="verified_likelihood_per_rollout",
        online_canonical_replay_global_groups_per_step=1,
        test_split="multi_answer",
    )
    values.update(overrides)
    return helper._args(**values)


def test_semantic_maxent_composes_with_uniform_verified_likelihood_replay():
    args = _validation_args(
        semantic_shannon_coef=0.10,
        semantic_shannon_separate_advantage=True,
        semantic_shannon_success_conditioned_signed_advantage=True,
    )
    assert validate_zero_math_args(args) is args


def test_e78_replay_arm_still_validates_unchanged():
    args = _validation_args()
    assert validate_zero_math_args(args) is args


def test_semantic_maxent_still_excludes_the_unregistered_companions():
    with pytest.raises(ValueError, match="separate treatments"):
        validate_zero_math_args(
            _validation_args(
                semantic_shannon_coef=0.10,
                semantic_shannon_separate_advantage=True,
            )
        )
    with pytest.raises(ValueError, match="token entropy"):
        validate_zero_math_args(
            _validation_args(
                semantic_shannon_coef=0.10,
                semantic_shannon_separate_advantage=True,
                semantic_shannon_success_conditioned_signed_advantage=True,
                policy_entropy_coef=0.001,
            )
        )
    with pytest.raises(ValueError, match="SEED"):
        validate_zero_math_args(
            _validation_args(
                semantic_shannon_coef=0.10,
                semantic_shannon_separate_advantage=True,
                semantic_shannon_success_conditioned_signed_advantage=True,
                seed_entropy_alpha=0.001,
            )
        )


# --------------------------------------------------------------------------
# the semantic advantage itself
# --------------------------------------------------------------------------


def _tracker(coefficient: float = 0.10) -> SemanticShannonTracker:
    return SemanticShannonTracker(
        coefficient=coefficient,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_signed_advantage=True,
    )


def _score(tracker, keys, rewards, prompt=(11, 12, 13)):
    return tracker.score_success_conditioned_signed_advantages_and_update(
        prompt_token_ids=[list(prompt)] * len(keys),
        answer_keys=list(keys),
        task_rewards=list(rewards),
        active_mask=[1.0] * len(keys),
        num_samples=len(keys),
    )


def test_all_wrong_group_is_an_exact_no_op():
    tracker = _tracker()
    advantages, diagnostics = _score(
        tracker, ["a", "b", "c", "d"], [0.0, 0.0, 0.0, 0.0]
    )
    assert advantages == [0.0, 0.0, 0.0, 0.0]
    assert tracker.tracked_prompt_count == 0
    assert tracker.tracked_outcome_count == 0
    assert diagnostics.eligible_fraction == 0.0


def test_incorrect_and_unparseable_rows_receive_exact_zero_and_never_enter_support():
    tracker = _tracker()
    advantages, _ = _score(
        tracker,
        ["a", "a", "wrong_but_common", None],
        [1.0, 1.0, 0.0, 1.0],
    )
    assert advantages[2] == 0.0
    assert advantages[3] == 0.0
    # only the two verified "a" rows entered the predictor's support
    assert tracker.tracked_prompt_count == 1
    assert tracker.tracked_outcome_count == 1


def test_semantic_advantage_is_bounded_by_the_coefficient_and_is_not_clamped():
    tracker = _tracker()
    # Build a history dominated by one verified mode, then present a fresh one.
    for _ in range(4):
        _score(tracker, ["a"] * 8, [1.0] * 8)
    advantages, diagnostics = _score(
        tracker, ["a"] * 7 + ["novel"], [1.0] * 8
    )
    assert all(abs(value) <= 0.10 + 1e-12 for value in advantages)
    # the E43 +/- 0.05 outer clamp is not in play in the maintained fixed path
    assert diagnostics.advantage_cap == 0.0
    assert max(abs(value) for value in advantages) > 0.05
    # the rare verified outcome is pushed up, the common one down
    assert advantages[-1] > 0.0
    assert advantages[0] < 0.0


def test_semantic_advantage_is_invariant_to_canonical_key_renaming():
    plain, _ = _score(_tracker(), ["a", "a", "b", "c"], [1.0, 1.0, 1.0, 1.0])
    renamed, _ = _score(
        _tracker(), ["mode-7", "mode-7", "mode-1", "zzz"], [1.0, 1.0, 1.0, 1.0]
    )
    assert plain == pytest.approx(renamed)


def test_semantic_advantage_scales_linearly_in_the_fixed_coefficient():
    at_ten, _ = _score(_tracker(0.10), ["a", "a", "b", "c"], [1.0] * 4)
    at_five, _ = _score(_tracker(0.05), ["a", "a", "b", "c"], [1.0] * 4)
    assert at_five == pytest.approx([value / 2.0 for value in at_ten])


def test_semantic_predictor_state_survives_an_exact_resume():
    original = _tracker()
    _score(original, ["a", "a", "b", "c"], [1.0] * 4)
    resumed = _tracker()
    resumed.load_state_dict(original.state_dict())
    assert resumed.tracked_prompt_count == original.tracked_prompt_count
    assert resumed.tracked_outcome_count == original.tracked_outcome_count
    before, _ = _score(original, ["a", "b", "b", "d"], [1.0] * 4)
    after, _ = _score(resumed, ["a", "b", "b", "d"], [1.0] * 4)
    assert before == pytest.approx(after)


# --------------------------------------------------------------------------
# progress reporting
# --------------------------------------------------------------------------


def _status_snapshot(semantic_step: int) -> dict:
    return {
        "arms": ["semantic"],
        "checkpoint_interval": 192,
        "domains": ["graph_coloring"],
        "passes": 8,
        "replay_weight": 0.10,
        "semantic_coefficient": 0.10,
        "steps_per_pass": 384,
        "target": 3072,
        "rows": [
            {
                "arm": "semantic",
                "checkpoint": semantic_step,
                "domain": "graph_coloring",
                "job_id": 1,
                "seed": 43,
                "state": "RUNNING",
                "step": semantic_step,
            }
        ],
    }


def test_e78_status_reports_the_e81_arm_against_its_inherited_pair():
    status = load(ROOT / "ops/exp_scaling/status_e78.py", "e81_status_e78")
    pairs = {
        ("graph_coloring", 43, "replay"): 3072,
        ("graph_coloring", 43, "control"): 3072,
    }

    running = status.render_semantic(_status_snapshot(1536), pairs)
    assert "E81 semantic-MaxEnt extension (eta = 0.1" in running
    # the semantic arm is only half way, so the cell is not yet analysable
    assert "all at pass 8: 0/1" in running

    finished = status.render_semantic(_status_snapshot(3072), pairs)
    assert "all at pass 8: 1/1" in finished

    # a missing comparator must never count as analysable
    unpaired = status.render_semantic(
        _status_snapshot(3072), {("graph_coloring", 43, "replay"): 3072}
    )
    assert "all at pass 8: 0/1" in unpaired

    # the heading is per-cohort, so a Falcon section cannot read as a Qwen one
    falcon = status.render_semantic(
        _status_snapshot(3072), pairs, label="E82 semantic-MaxEnt on Falcon3-1B"
    )
    assert "E82 semantic-MaxEnt on Falcon3-1B (eta = 0.1" in falcon
    assert "E81" not in falcon


def test_status_e81_reuses_the_shared_pair_depth_reader():
    status = load(ROOT / "ops/exp_scaling/status_e81.py", "e81_status")
    assert status.LEDGER.name == (
        "e81_semantic_maxent_verified_replay_05b_jobs.json"
    )
    assert hasattr(status.base, "e78_pair_depth")
    assert hasattr(status.base, "load_semantic_snapshot")


def test_e80_monitor_pairs_each_semantic_arm_with_its_own_comparator():
    monitor = load(ROOT / "ops/exp_scaling/status_e80r1.py", "e81_status_e80r1")
    cohorts = {label: (ledger, pair) for ledger, pair, label in monitor.SEMANTIC_COHORTS}
    assert len(cohorts) == 3
    qwen = cohorts["E81 semantic-MaxEnt on Qwen2.5-0.5B"]
    falcon = cohorts["E82 semantic-MaxEnt on Falcon3-1B"]
    no_replay = cohorts["E83 semantic-MaxEnt without replay on Qwen2.5-0.5B"]
    # E81 and E83 pair against E78, E82 against E79; crossing them would compare
    # a Falcon arm to a Qwen comparator.
    assert qwen[0].name == "e81_semantic_maxent_verified_replay_05b_jobs.json"
    assert qwen[1].name == "e78_verified_replay_only_05b_jobs.json"
    assert falcon[0].name == "e82_falcon_semantic_maxent_verified_replay_jobs.json"
    assert falcon[1].name == "e79_falcon1b_aligned_verified_replay_jobs.json"
    assert no_replay[0].name == "e83_semantic_maxent_without_replay_05b_jobs.json"
    assert no_replay[1].name == "e78_verified_replay_only_05b_jobs.json"


def test_semantic_section_is_absent_until_its_ledger_exists(tmp_path):
    status = load(ROOT / "ops/exp_scaling/status_e78.py", "e81_status_absent")
    missing = tmp_path / "not_submitted.json"
    assert status.semantic_section(missing, missing, "E99 nothing") is None


def test_point_checkpoint_reader_accepts_every_cohorts_schema(tmp_path):
    import json as _json

    status = load(ROOT / "ops/exp_scaling/status_e78.py", "e81_status_point")

    def checkpoint(schema: str, update: int) -> int:
        directory = tmp_path / schema
        directory.mkdir()
        (directory / "COMPLETE.json").write_text(
            _json.dumps({"schema": schema, "update": update}), encoding="utf-8"
        )
        return status.point_checkpoint_step(directory)

    # Each PointMaze cohort stamps its own id; matching one literal reported
    # every other cohort as having saved nothing.
    assert checkpoint("e78pm-point-maze-rolling-checkpoint-v1", 3072) == 3072
    assert checkpoint("e79pm-point-maze-rolling-checkpoint-v1", 1728) == 1728
    assert checkpoint("e123pm-point-maze-rolling-checkpoint-v1", 192) == 192
    # an unrelated receipt must still be ignored rather than read as progress
    assert checkpoint("point-maze-waypoint-pilot-evaluation-v1", 999) == 0
    assert checkpoint("e79pm-point-maze-rolling-checkpoint-v2", 999) == 0
    assert status.point_checkpoint_step(tmp_path / "absent") == 0


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------


def test_e81_protocol_freezes_the_coefficient_pairing_and_reporting_rules():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "Frozen before submission",
        "does not re-run either E78 arm",
        "exactly eight passes",
        "3,072 optimizer updates",
        "every 192 updates",
        "passes 0, 0.5, 1.0, ..., 8.0",
        "5 domains x 1 arm x 5 seeds = 25 runs",
        "eta = 0.10",
        "does not identify an optimal coefficient",
        "There is no reward for discovering an outcome for the first",
        "`semantic - replay`",
        "Do not pool domains",
        "do not select a best checkpoint",
        "PointMaze is excluded",
    ):
        assert literal in text
