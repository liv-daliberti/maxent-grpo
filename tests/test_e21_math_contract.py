"""Prospective contract for E21's authentic free-form MATH experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper/preregistration/e21_math_token_policy_maxent.md"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e21_math_token_maxent.sh"
CHECKER = ROOT / "ops/exp_scaling/check_e21_math_token_smoke.py"
IMPORTER = ROOT / "ops/math500/import_oat_math.py"
ACTOR = ROOT / "src/oat_drgrpo/actor.py"


def test_protocol_is_free_form_and_length_neutral():
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "free-form conditional-token MaxEnt" in text
    assert "answer list, answer index, gold-derived support" in text
    assert "zero direct derivative with respect to the EOS logit" in text
    assert "one equal-weight mean regardless of its token count" in text
    assert "no prefix importance ratio" in text
    assert "No aggregation rescaling" in text
    assert "every 2,130 consumed prompts" in text


def test_launcher_refuses_legacy_sequence_entropy_and_length_dual():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean" in text
    assert "OAT_ZERO_MAXENT_LENGTH_TARGET=0" in text
    assert "OAT_ZERO_CANONICAL_ACTION_TASK=none" in text
    assert "OAT_ZERO_VERIFIER_VERSION=math_verify" in text
    assert "OAT_ZERO_EVAL_PROMPT_INTERVAL=\"$eval_interval\"" in text
    assert "OAT_ZERO_EVAL_MODE_COVERAGE_K" in text
    assert "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=1" in text
    assert "OAT_ZERO_TRAIN_GRES=gpu:a6000:1" in text
    assert "OAT_ZERO_TRAIN_PARTITION=cs" in text


def test_full_math_verifier_wait_covers_its_compound_inner_timeouts():
    text = ACTOR.read_text(encoding="utf-8")

    assert "FULL_VERIFIER_PROCESS_TIMEOUT_SECONDS = 5" in text
    assert "FullMathVerifierProcess(" in text


def test_smoke_checker_is_exact_four_arm_fail_closed_gate():
    spec = importlib.util.spec_from_file_location("check_e21", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.END_STEP == 64
    assert module.SEED == 9007
    assert set(module.ARMS) == {"grpo", "maxent", "maxent_control", "maxent_dual"}


def test_authentic_math_artifact_still_passes_import_audit():
    spec = importlib.util.spec_from_file_location("import_oat_math_e21", IMPORTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    audit = module.audit_materialized(module.DEFAULT_OUTPUT_ROOT)
    assert audit["train_rows"] == 8523
    assert audit["math500_rows"] == 500
    assert audit["exact_problem_overlap"] == 0
