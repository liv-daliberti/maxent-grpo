"""Shared default values for optional config fields (e.g., InfoSeed)."""

INFO_SEED_DEFAULTS = {
    "info_seed_enabled": False,
    "info_seed_num_seeds": 0,
    "info_seed_lambda": 0.0,
    "info_seed_alpha_entropy": 0.0,
    "info_seed_prompt_template": "\n[seed={seed}]",
    "info_seed_loss_type": "infonce",
    "info_seed_pooling": "mean",
}

__all__ = ["INFO_SEED_DEFAULTS"]
