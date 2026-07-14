"""Candidate-level tempered aggregation weights for xDr.GRPO.

xDr.GRPO replaces Dr.GRPO's uniform 1/G aggregation over the G sampled
completions of a prompt with a tempered softmax over per-candidate utilities:
within each prompt group, w_i = softmax(U_i / tau), applied as a detached
per-row loss weight of G * w_i so that the group's loss becomes
sum_i w_i * U_i instead of (1/G) sum_i U_i.

The utility is the per-candidate Dr.GRPO surrogate evaluated at the rollout
policy, where the importance ratio is exactly 1 and the clipped surrogate
reduces to U_i = A_i * T_i / T_max (advantage times active completion tokens
over the constant length normalizer). Weights are computed once per rollout
batch and frozen for the update, like advantages.

tau = inf corresponds exactly to Dr.GRPO (uniform weights). Callers must skip
the reweighting entirely in that case rather than evaluating the softmax, so
the baseline code path stays bit-for-bit unchanged.
"""

from __future__ import annotations

import math

import torch


def compute_xdr_row_weights(
    advantages: torch.Tensor,
    response_token_counts: torch.Tensor,
    *,
    num_samples: int,
    tau: float,
    t_max: int,
    loss_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-row weights over each prompt group: softmax(U/tau) scaled to the
    number of participating rows.

    Args:
        advantages: [N] or [N, 1] per-sequence scalar advantages, prompt-major
            (G consecutive rows per prompt).
        response_token_counts: [N] number of active completion tokens per row.
        num_samples: group size G.
        tau: finite positive temperature.
        t_max: the constant Dr.GRPO length normalizer (generate_max_length).
        loss_masks: optional [N] 0/1 mask. Loss-masked rows contribute zero
            loss downstream regardless, but excluding them here keeps the
            softmax mass on the rows that actually train: weights are
            normalized over the valid rows of each group and scaled by the
            valid-row count, so they sum to n_valid exactly as the uniform
            Dr.GRPO aggregation effectively does. Masked rows get weight 0.

    Returns:
        Detached float tensor [N]; for fully valid groups the weights sum to
        G, so uniform groups (equal utilities) yield exactly 1.0 per row.
    """
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError(f"xdr tau must be finite and positive, got {tau}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    adv = advantages.detach().reshape(-1).float()
    counts = response_token_counts.detach().reshape(-1).float()
    if adv.numel() != counts.numel():
        raise ValueError(
            f"advantages ({adv.numel()}) and response_token_counts "
            f"({counts.numel()}) must have the same number of rows"
        )
    if adv.numel() % num_samples != 0:
        raise ValueError(
            f"batch of {adv.numel()} rows is not divisible by group size "
            f"{num_samples}; prompt groups must be intact"
        )
    utilities = (adv * counts / float(t_max)).view(-1, num_samples)
    logits = utilities / float(tau)
    if loss_masks is None:
        weights = torch.softmax(logits, dim=1)
        return (float(num_samples) * weights).reshape(-1)
    valid = (loss_masks.detach().reshape(-1).float() > 0).view(
        -1, num_samples
    )
    logits = torch.where(
        valid, logits, torch.full_like(logits, float("-inf"))
    )
    weights = torch.softmax(logits, dim=1)
    weights = torch.nan_to_num(weights, nan=0.0)
    n_valid = valid.float().sum(dim=1, keepdim=True)
    weights = weights * n_valid
    # All-masked groups (n_valid = 0) contribute no loss; leave them at 0.
    return weights.reshape(-1)


def aggregation_group_diagnostics(
    row_weights: torch.Tensor,
    final_rewards: torch.Tensor,
    *,
    num_samples: int,
) -> dict[str, torch.Tensor]:
    """Candidate-level exploration diagnostics from the aggregation weights.

    Defined identically for every quartet arm: pass the realized per-row
    aggregation weights — uniform ones (Dr.GRPO, Token-MaxEnt), the
    prompt-rescaled uniform SEED weights, or the tempered xDr.GRPO softmax
    weights — with loss-masked rows already zeroed. Each prompt group is
    normalized to a distribution over its valid rows; all-masked groups are
    dropped from the batch means.

    Returns the batch means of the effective number of active rollouts
    exp(H(w)) per prompt (n_valid for uniform weights) and the incorrect-mass
    share sum_i w_i * 1{r_i <= 0} per prompt.
    """
    with torch.no_grad():
        w = row_weights.detach().reshape(-1, num_samples).float()
        totals = w.sum(dim=1, keepdim=True)
        valid_groups = totals.reshape(-1) > 0
        if not bool(valid_groups.any()):
            zero = w.new_tensor(0.0)
            return {"agg_eff_rollouts": zero, "agg_incorrect_mass": zero}
        w = w[valid_groups] / totals[valid_groups]
        # w * log(clamped w) keeps the 0 log 0 = 0 convention for masked rows.
        entropy = -(w * w.clamp_min(1e-12).log()).sum(dim=1)
        eff_rollouts = entropy.exp().mean()
        incorrect = (
            final_rewards.detach().reshape(-1, num_samples)[valid_groups] <= 0
        ).float()
        incorrect_mass = (w * incorrect).sum(dim=1).mean()
    return {
        "agg_eff_rollouts": eff_rollouts,
        "agg_incorrect_mass": incorrect_mass,
    }
