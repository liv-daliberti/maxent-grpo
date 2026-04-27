from oat_drgrpo.correctness_schedule import (
    build_correctness_scheduled_settings,
    update_correctness_ema,
)


def test_update_correctness_ema_initializes_from_batch_mean():
    ema = update_correctness_ema(
        previous_ema=None,
        batch_any_correct_mean=0.25,
        ema_decay=0.98,
    )

    assert ema == 0.25


def test_correctness_schedule_is_exploratory_when_ema_is_low():
    settings = build_correctness_scheduled_settings(
        enabled=True,
        previous_ema=None,
        batch_any_correct_mean=0.05,
        ema_decay=0.98,
        correctness_low=0.10,
        correctness_high=0.45,
        static_budget_max=0.10,
        static_prompt_select_min_alpha_frac=0.50,
        static_mode_tau=0.05,
        static_intra_tau=0.01,
        budget_max_early=0.18,
        budget_max_late=0.05,
        prompt_select_min_alpha_frac_early=0.20,
        prompt_select_min_alpha_frac_late=0.55,
        mode_tau_early=0.08,
        mode_tau_late=0.03,
        intra_tau_early=0.03,
        intra_tau_late=0.005,
    )

    assert settings.any_correct_ema == 0.05
    assert settings.exploration_level == 1.0
    assert settings.consolidation_level == 0.0
    assert settings.budget_max == 0.18
    assert settings.prompt_select_min_alpha_frac == 0.20
    assert settings.mode_tau == 0.08
    assert settings.intra_tau == 0.03


def test_correctness_schedule_is_consolidating_when_ema_is_high():
    settings = build_correctness_scheduled_settings(
        enabled=True,
        previous_ema=None,
        batch_any_correct_mean=0.60,
        ema_decay=0.98,
        correctness_low=0.10,
        correctness_high=0.45,
        static_budget_max=0.10,
        static_prompt_select_min_alpha_frac=0.50,
        static_mode_tau=0.05,
        static_intra_tau=0.01,
        budget_max_early=0.18,
        budget_max_late=0.05,
        prompt_select_min_alpha_frac_early=0.20,
        prompt_select_min_alpha_frac_late=0.55,
        mode_tau_early=0.08,
        mode_tau_late=0.03,
        intra_tau_early=0.03,
        intra_tau_late=0.005,
    )

    assert settings.any_correct_ema == 0.60
    assert settings.exploration_level == 0.0
    assert settings.consolidation_level == 1.0
    assert settings.budget_max == 0.05
    assert settings.prompt_select_min_alpha_frac == 0.55
    assert settings.mode_tau == 0.03
    assert settings.intra_tau == 0.005


def test_correctness_schedule_disabled_uses_static_values():
    settings = build_correctness_scheduled_settings(
        enabled=False,
        previous_ema=0.20,
        batch_any_correct_mean=0.30,
        ema_decay=0.90,
        correctness_low=0.10,
        correctness_high=0.45,
        static_budget_max=0.10,
        static_prompt_select_min_alpha_frac=0.50,
        static_mode_tau=0.05,
        static_intra_tau=0.01,
        budget_max_early=0.18,
        budget_max_late=0.05,
        prompt_select_min_alpha_frac_early=0.20,
        prompt_select_min_alpha_frac_late=0.55,
        mode_tau_early=0.08,
        mode_tau_late=0.03,
        intra_tau_early=0.03,
        intra_tau_late=0.005,
    )

    assert abs(settings.any_correct_ema - 0.21) < 1e-9
    assert settings.budget_max == 0.10
    assert settings.prompt_select_min_alpha_frac == 0.50
    assert settings.mode_tau == 0.05
    assert settings.intra_tau == 0.01
