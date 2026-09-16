from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import e117_stage1_statistics as stats  # noqa: E402

EFFECTIVE_CONTRACT = ROOT / (
    "paper/preregistration/" "e117_successor_effective_contract_v11_20260825.md"
)


SEEDS = stats.STAGE1_TRAINING_SEEDS
DRAWS = stats.EVALUATION_DRAWS
CHECKPOINTS = stats.CHECKPOINTS


def complete_rows(
    *,
    proposal_distinct: float = 0.09,
    semantic_distinct: float = 0.08,
    proposal_pass: float = 0.0,
    semantic_pass: float = 0.0,
    training_seeds: tuple[int, ...] = SEEDS,
) -> list[dict[str, object]]:
    rows = []
    for sentinel in stats.SENTINELS:
        prompt_surface = hashlib.sha256(
            f"prompt:{sentinel}".encode("utf-8")
        ).hexdigest()
        for seed_index, seed in enumerate(training_seeds):
            seed_shift = (seed_index - 1) * 0.003
            for draw in DRAWS:
                request_surface = hashlib.sha256(
                    f"request:{sentinel}:{draw}".encode("utf-8")
                ).hexdigest()
                draw_shift = (draw - 7.5) * 0.0002
                for arm in stats.ARMS:
                    for checkpoint in CHECKPOINTS:
                        active = float(checkpoint > 0)
                        pass_delta = 0.0
                        distinct_delta = 0.0
                        if arm in ("p", "f"):
                            pass_delta += proposal_pass + seed_shift + draw_shift
                            distinct_delta += (
                                proposal_distinct + seed_shift + draw_shift
                            )
                        if arm == "f":
                            pass_delta += semantic_pass + seed_shift + draw_shift
                            distinct_delta += (
                                semantic_distinct + seed_shift + draw_shift
                            )
                        rows.append(
                            {
                                "sentinel": sentinel,
                                "training_seed": seed,
                                "arm": arm,
                                "checkpoint": checkpoint,
                                "evaluation_draw": draw,
                                "pass_at_8": 0.30 + active * pass_delta,
                                "raw_distinct_at_8": 0.70 + active * distinct_delta,
                                "prompt_surface_sha256": prompt_surface,
                                "request_surface_sha256": request_surface,
                            }
                        )
    return rows


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    return stats.analyze_registered_table(
        rows,
        training_seeds=SEEDS,
        evaluation_draws=DRAWS,
        checkpoints=CHECKPOINTS,
    )


def test_stage1_keeps_primitives_and_uncertainty_axes_separate():
    result = analyze(complete_rows())
    assert result["schema"] == "e117_stage1_paired_vector_statistics_v9"
    assert result["estimands"]["proposal_replay"]["contrast"] == "P-C"
    assert (
        result["estimands"]["proposal_replay"][
            "realized_post_treatment_support_held_fixed"
        ]
        is False
    )
    semantic_estimand = result["estimands"]["semantic_increment"]
    assert semantic_estimand["contrast"] == "F-P"
    assert semantic_estimand["controlled_direct_semantic_effect"] is False
    assert (
        semantic_estimand["semantic_without_proposal_arm_required_for_decomposition"]
        is True
    )
    assert result["endpoint_basis"] == ["pass_at_8", "raw_distinct_at_8"]
    assert result["derived_endpoint"]["independent_coordinate"] is False
    assert result["uncertainty"]["combined_interval"] is None
    assert result["advancement_basis"]["pass_safety_summaries"] == [
        "terminal",
        "normalized_auc",
    ]
    assert result["advancement_basis"]["pass_safety_contrasts"] == [
        "component numerator versus component denominator",
        "candidate numerator versus C",
    ]
    assert result["design"]["evaluation_identity"]["fields"] == [
        "prompt_surface_sha256",
        "request_surface_sha256",
    ]
    assert (
        result["design"]["evaluation_identity"][
            "request_surfaces_distinct_across_draws"
        ]
        is True
    )
    proposal = result["sentinels"][stats.SENTINELS[0]]["components"]["proposal_replay"]
    terminal = proposal["terminal"]
    assert terminal[stats.DERIVED]["estimate"] == pytest.approx(
        terminal[stats.PRIMITIVES[1]]["estimate"]
        - terminal[stats.PRIMITIVES[0]]["estimate"]
    )
    assert terminal[stats.DERIVED]["training_seed_se"] > 0.0
    assert terminal[stats.DERIVED]["evaluation_mc_se"] > 0.0
    assert (
        terminal[stats.DERIVED]["training_seed_se"]
        != terminal[stats.DERIVED]["evaluation_mc_se"]
    )
    assert result["advancement"]["proposal_replay"]["broad_candidate"] is True
    assert result["advancement"]["proposal_replay"]["cross_model_support"] is True
    assert result["advancement"]["semantic_increment"]["broad_candidate"] is True
    assert result["inferential_scope"]["prompt_population_inference"] is False
    assert result["inferential_scope"]["model_family_main_effect_identified"] is False
    assert result["uncertainty"]["prompt_population_se"] is None
    assert result["design"]["actual_start_orders"] == stats.STAGE1_START_ORDERS
    assert result["locked_confirmation"]["training_seeds"] == list(range(301, 307))


def test_effective_contract_matches_executable_v11_boundaries():
    contract = EFFECTIVE_CONTRACT.read_text(encoding="utf-8")
    assert "e117_stage1_paired_vector_statistics_v9" in contract
    assert "e117_confirmation_paired_vector_statistics_v1" in contract
    assert "201, 202, 203" in contract
    assert "301, ..., 306" in contract
    assert "draw labels `0, ..., 15`" in contract
    assert "raw distinct correct modes@8" in contract
    assert "cannot independently veto" in contract
    assert "not factorially crossed" in contract
    assert "not a controlled direct semantic effect" in contract
    assert "semantic-without-proposal" in contract
    assert "native JSON numbers" in contract
    assert "not exact FLOPs" in contract
    assert "descriptive diagnostic only" in contract


def test_accuracy_rescue_cannot_be_vetoed_by_the_derived_subtraction():
    result = analyze(
        complete_rows(
            proposal_distinct=0.30,
            proposal_pass=0.50,
            semantic_distinct=0.0,
            semantic_pass=0.0,
        )
    )
    terminal = result["sentinels"][stats.SENTINELS[0]]["components"]["proposal_replay"][
        "terminal"
    ]
    assert terminal["pass_at_8"]["estimate"] > 0.0
    assert terminal["raw_distinct_at_8"]["estimate"] > 0.0
    assert terminal[stats.DERIVED]["estimate"] < 0.0
    advancement = result["sentinels"][stats.SENTINELS[0]]["components"][
        "proposal_replay"
    ]["advancement"]
    assert advancement["advancement_endpoint"] == "raw_distinct_at_8"
    assert advancement["derived_adjusted_breadth_veto"] is False
    assert result["advancement"]["proposal_replay"]["scope"] == (
        "broad_development_candidate"
    )


def test_derived_only_gain_does_not_replace_the_primitive_advancement_endpoint():
    result = analyze(
        complete_rows(
            proposal_distinct=0.04,
            proposal_pass=-0.02,
            semantic_distinct=0.0,
            semantic_pass=0.0,
        )
    )
    proposal = result["sentinels"][stats.SENTINELS[0]]["components"]["proposal_replay"]
    assert proposal["terminal"][stats.DERIVED]["estimate"] > 0.05
    assert proposal["terminal"]["raw_distinct_at_8"]["estimate"] < 0.05
    assert proposal["advancement"]["actionable"] is False


def test_pass_safety_blocks_a_large_semantic_increment():
    rows = complete_rows(semantic_distinct=0.30)
    for row in rows:
        if (
            row["arm"] == "f"
            and row["training_seed"] == SEEDS[0]
            and row["checkpoint"] != 0
        ):
            row["pass_at_8"] = float(row["pass_at_8"]) - 0.12
    result = analyze(rows)
    semantic = result["sentinels"][stats.SENTINELS[0]]["components"][
        "semantic_increment"
    ]
    assert semantic["terminal"][stats.DERIVED]["estimate"] > 0.05
    assert (
        semantic["advancement"]["checks"]["terminal_every_seed_pass_safety_vs_c"]
        is False
    )
    assert semantic["advancement"]["actionable"] is False


def test_semantic_pass_loss_cannot_hide_behind_proposal_pass_gain():
    result = analyze(
        complete_rows(
            proposal_distinct=0.09,
            proposal_pass=0.08,
            semantic_distinct=0.30,
            semantic_pass=-0.06,
        )
    )
    semantic = result["sentinels"][stats.SENTINELS[0]]["components"][
        "semantic_increment"
    ]
    checks = semantic["advancement"]["checks"]
    assert checks["terminal_mean_pass_safety_vs_c"] is True
    assert checks["terminal_mean_pass_safety_vs_component_denominator"] is False
    assert semantic["advancement"]["actionable"] is False


def test_transient_pass_collapse_is_caught_by_auc_safety():
    rows = complete_rows(semantic_distinct=0.30)
    for row in rows:
        if row["arm"] == "f" and row["checkpoint"] not in (
            CHECKPOINTS[0],
            CHECKPOINTS[-1],
        ):
            row["pass_at_8"] = float(row["pass_at_8"]) - 0.12
    result = analyze(rows)
    semantic = result["sentinels"][stats.SENTINELS[0]]["components"][
        "semantic_increment"
    ]
    checks = semantic["advancement"]["checks"]
    assert checks["terminal_mean_pass_safety_vs_component_denominator"] is True
    assert checks["normalized_auc_mean_pass_safety_vs_component_denominator"] is False
    assert semantic["advancement"]["actionable"] is False


def test_incomplete_grid_and_step_zero_mismatch_fail_closed():
    rows = complete_rows()
    with pytest.raises(RuntimeError, match="incomplete Stage-1 grid"):
        analyze(rows[:-1])

    mismatched = deepcopy(rows)
    target = next(
        row for row in mismatched if row["arm"] == "f" and row["checkpoint"] == 0
    )
    target["raw_distinct_at_8"] = float(target["raw_distinct_at_8"]) + 0.01
    with pytest.raises(RuntimeError, match="step-zero endpoint mismatch"):
        analyze(mismatched)

    cross_seed_mismatch = deepcopy(rows)
    for row in cross_seed_mismatch:
        if row["training_seed"] == SEEDS[1] and row["checkpoint"] == CHECKPOINTS[0]:
            row["pass_at_8"] = float(row["pass_at_8"]) + 0.01
    with pytest.raises(RuntimeError, match="across paired seeds/C/P/F"):
        analyze(cross_seed_mismatch)


def test_prompt_and_common_request_surface_mismatch_fail_closed():
    rows = complete_rows()
    prompt_mismatch = deepcopy(rows)
    prompt_target = next(
        row
        for row in prompt_mismatch
        if row["arm"] == "f" and row["checkpoint"] == CHECKPOINTS[1]
    )
    prompt_target["prompt_surface_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="prompt surface mismatch"):
        analyze(prompt_mismatch)

    request_mismatch = deepcopy(rows)
    request_target = next(
        row
        for row in request_mismatch
        if row["arm"] == "p"
        and row["training_seed"] == SEEDS[1]
        and row["checkpoint"] == CHECKPOINTS[-1]
    )
    request_target["request_surface_sha256"] = "1" * 64
    with pytest.raises(RuntimeError, match="request surface mismatch"):
        analyze(request_mismatch)


def test_missing_or_malformed_identity_digest_fails_closed():
    rows = complete_rows()
    malformed = deepcopy(rows)
    malformed[0]["prompt_surface_sha256"] = "ABC"
    with pytest.raises(RuntimeError, match="lowercase SHA-256 digest"):
        analyze(malformed)

    missing = deepcopy(rows)
    del missing[0]["request_surface_sha256"]
    with pytest.raises(RuntimeError, match="invalid Stage-1 row 1"):
        analyze(missing)


def test_repeated_request_surface_cannot_masquerade_as_distinct_draws():
    rows = complete_rows()
    source_digest = next(
        row["request_surface_sha256"]
        for row in rows
        if row["sentinel"] == stats.SENTINELS[0] and row["evaluation_draw"] == DRAWS[0]
    )
    for row in rows:
        if row["sentinel"] == stats.SENTINELS[0] and row["evaluation_draw"] == DRAWS[1]:
            row["request_surface_sha256"] = source_digest
    with pytest.raises(RuntimeError, match="reuse a request surface"):
        analyze(rows)


def test_three_same_model_sentinels_cannot_be_called_broad():
    rows = complete_rows()
    for row in rows:
        if row["sentinel"] == "falcon1b/mathir" and row["checkpoint"] != 0:
            row["pass_at_8"] = 0.30
            row["raw_distinct_at_8"] = 0.70
    result = analyze(rows)
    for component in stats.COMPONENTS:
        decision = result["advancement"][component]
        assert decision["actionable_count"] == 3
        assert decision["actionable_model_families"] == ["qwen05b"]
        assert decision["cross_model_support"] is False
        assert decision["model_domain_factorially_crossed"] is False
        assert decision["model_family_main_effect_identified"] is False
        assert decision["broad_candidate"] is False
        assert decision["scope"] == "do_not_advance"


def test_frozen_sentinel_and_k_constants_fail_closed():
    rows = complete_rows()
    with pytest.raises(RuntimeError, match="exact four registered sentinels"):
        stats.analyze_registered_table(
            rows,
            training_seeds=SEEDS,
            evaluation_draws=DRAWS,
            checkpoints=CHECKPOINTS,
            sentinels=stats.SENTINELS[:-1],
        )
    with pytest.raises(RuntimeError, match="requires K=8"):
        stats.analyze_registered_table(
            rows,
            training_seeds=SEEDS,
            evaluation_draws=DRAWS,
            checkpoints=CHECKPOINTS,
            k=7,
        )


def test_exact_seed_draw_and_checkpoint_grid_is_not_caller_discretion():
    rows = complete_rows()
    with pytest.raises(RuntimeError, match="exact seeds 201,202,203"):
        stats.analyze_registered_table(
            rows,
            training_seeds=(201, 202, 204),
            evaluation_draws=DRAWS,
            checkpoints=CHECKPOINTS,
        )
    with pytest.raises(RuntimeError, match="exact draw labels 0..15"):
        stats.analyze_registered_table(
            rows,
            training_seeds=SEEDS,
            evaluation_draws=tuple(range(1, 17)),
            checkpoints=CHECKPOINTS,
        )
    with pytest.raises(RuntimeError, match="exact checkpoints 0:192:3072"):
        stats.analyze_registered_table(
            rows,
            training_seeds=SEEDS,
            evaluation_draws=DRAWS,
            checkpoints=CHECKPOINTS[:-1],
        )


def test_boolean_row_identifier_cannot_alias_an_integer():
    rows = complete_rows()
    corrupted = deepcopy(rows)
    target = next(row for row in corrupted if row["checkpoint"] == 0)
    target["checkpoint"] = False
    with pytest.raises(RuntimeError, match="invalid Stage-1 row"):
        analyze(corrupted)


@pytest.mark.parametrize("bad_value", [True, "0.30"])
def test_endpoint_values_must_be_native_json_numbers(bad_value):
    rows = complete_rows()
    corrupted = deepcopy(rows)
    corrupted[0]["pass_at_8"] = bad_value
    with pytest.raises(RuntimeError, match="invalid Stage-1 row 1"):
        analyze(corrupted)


def test_rejects_extra_seed_and_missing_draw():
    rows = complete_rows()
    with pytest.raises(RuntimeError, match="exact seeds 201,202,203"):
        stats.analyze_registered_table(
            rows,
            training_seeds=(*SEEDS, 204),
            evaluation_draws=DRAWS,
            checkpoints=CHECKPOINTS,
        )
    with pytest.raises(RuntimeError, match="at least 16"):
        stats.analyze_registered_table(
            rows,
            training_seeds=SEEDS,
            evaluation_draws=DRAWS[:15],
            checkpoints=CHECKPOINTS,
        )


def test_locked_confirmation_rule_is_executable_and_stricter():
    rows = complete_rows(training_seeds=stats.CONFIRMATION_TRAINING_SEEDS)
    result = stats.analyze_confirmation_table(
        rows,
        selected_scopes={"proposal_replay": "broad_development_candidate"},
    )
    assert result["schema"] == "e117_confirmation_paired_vector_statistics_v1"
    assert result["development_only"] is False
    assert result["design"]["training_seeds"] == list(range(301, 307))
    assert result["inferential_scope"]["prompt_target"].endswith("confirmation bank")
    proposal = result["sentinels"][stats.SENTINELS[0]]["components"]["proposal_replay"]
    checks = proposal["advancement"]["checks"]
    assert checks["terminal_raw_distinct_positive_in_five_of_six_seeds"] is True
    assert checks["terminal_raw_distinct_effect_gt_two_training_seed_se"] is True
    decision = result["confirmation_selection"]["decisions"]["proposal_replay"]
    assert decision["scope_reproduced"] is True
    assert decision["confirmation_scope"] == "broad_development_candidate"
    assert decision["scope_upgrade_permitted"] is False
    assert result["confirmation_selection"]["development_endpoint_values_used"] is False


def test_confirmation_requires_five_positive_seed_effects():
    rows = complete_rows(
        proposal_distinct=0.15,
        training_seeds=stats.CONFIRMATION_TRAINING_SEEDS,
    )
    for row in rows:
        if (
            row["training_seed"] in stats.CONFIRMATION_TRAINING_SEEDS[:2]
            and row["arm"] == "p"
            and row["checkpoint"] != CHECKPOINTS[0]
        ):
            row["raw_distinct_at_8"] = 0.69
    result = stats.analyze_confirmation_table(
        rows,
        selected_scopes={"proposal_replay": "broad_development_candidate"},
    )
    proposal = result["sentinels"][stats.SENTINELS[0]]["components"]["proposal_replay"]
    assert (
        proposal["advancement"]["checks"][
            "terminal_raw_distinct_positive_in_five_of_six_seeds"
        ]
        is False
    )
    assert proposal["advancement"]["actionable"] is False


def test_confirmation_cannot_upgrade_selected_scope_or_accept_unregistered_seed():
    rows = complete_rows(training_seeds=stats.CONFIRMATION_TRAINING_SEEDS)
    result = stats.analyze_confirmation_table(
        rows,
        selected_scopes={"proposal_replay": "countdown_domain_specific_candidate"},
    )
    decision = result["confirmation_selection"]["decisions"]["proposal_replay"]
    assert result["advancement"]["proposal_replay"]["broad_candidate"] is True
    assert decision["confirmation_scope"] == "countdown_domain_specific_candidate"
    assert decision["scope_upgrade_permitted"] is False

    with pytest.raises(RuntimeError, match="exact seeds 301..306"):
        stats.analyze_confirmation_table(
            rows,
            selected_scopes={"proposal_replay": "countdown_domain_specific_candidate"},
            training_seeds=(301, 302, 303, 304, 305, 307),
        )
    with pytest.raises(RuntimeError, match="selected component scopes"):
        stats.analyze_confirmation_table(rows, selected_scopes={})
