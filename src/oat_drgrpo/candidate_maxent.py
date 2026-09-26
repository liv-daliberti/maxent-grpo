"""Historical finite-candidate projection primitives from the retired E6/E7 arms.

These functions remain unit-tested so the failed objective and its provenance
are reproducible. They are not reachable from the maintained trainer. Because
the candidates are sampled from the behavior policy, their weighted empirical
target includes that policy as a base measure and is not direct policy MaxEnt.
New experiments use :mod:`oat_drgrpo.on_policy_maxent` instead.
"""

from __future__ import annotations

import math

import torch


def _valid_mask(values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("candidate tensors must have shape [groups, candidates]")
    if valid_mask is None:
        return torch.ones_like(values, dtype=torch.bool)
    if valid_mask.shape != values.shape:
        raise ValueError("valid_mask must match the candidate tensor shape")
    return valid_mask.to(device=values.device, dtype=torch.bool)


def reward_target_distribution(
    utilities: torch.Tensor,
    *,
    temperature: float = 1.0,
    epsilon: float = 0.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Turn candidate utilities into the prompt-local reward target ``q``.

    ``epsilon`` is a probability floor on each valid candidate.  Invalid and
    all-masked groups carry zero mass.
    """

    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(epsilon)) or float(epsilon) < 0:
        raise ValueError("epsilon must be finite and non-negative")

    mask = _valid_mask(utilities, valid_mask)
    values = utilities.float()
    if not bool(torch.isfinite(values[mask]).all()):
        raise ValueError("utilities must be finite on valid candidates")
    logits = torch.where(
        mask,
        values / float(temperature),
        torch.full_like(values, float("-inf")),
    )
    probs = torch.softmax(logits, dim=1)
    probs = torch.nan_to_num(probs, nan=0.0)

    if epsilon > 0:
        valid_count = mask.sum(dim=1, keepdim=True).to(probs.dtype)
        if bool((float(epsilon) * valid_count[valid_count > 0] >= 1).any()):
            raise ValueError("epsilon times the number of valid candidates must be < 1")
        probs = torch.where(
            mask,
            probs * (1.0 - float(epsilon) * valid_count) + float(epsilon),
            torch.zeros_like(probs),
        )
    return probs


def solve_candidate_maxent_target(
    reward_target: torch.Tensor,
    *,
    tau: float,
    reference_log_probs: torch.Tensor | None = None,
    reference_kl_coef: float = 0.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Solve the candidate-level MaxEnt variational problem.

    For each prompt group this returns the maximizer of

    .. math::
       \langle w,\log q\rangle + \tau H(w)
       - \beta\,\mathrm{KL}(w\|\rho),

    where ``q`` is ``reward_target``, ``rho`` is the reference distribution,
    and ``beta`` is ``reference_kl_coef``.  With no reference term,
    ``w`` is proportional to ``q ** (1 / tau)``.  At ``tau=beta=0`` the
    function returns uniform mass over the exact argmax set of ``q``.
    """

    beta = float(reference_kl_coef)
    tau_value = float(tau)
    if not math.isfinite(tau_value) or tau_value < 0:
        raise ValueError("tau must be finite and non-negative")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("reference_kl_coef must be finite and non-negative")
    if beta > 0 and reference_log_probs is None:
        raise ValueError("a reference distribution is required when its KL coef is positive")

    mask = _valid_mask(reward_target, valid_mask)
    q = torch.where(mask, reward_target.float(), torch.zeros_like(reward_target.float()))
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("reward_target must be finite and non-negative")
    mass = q.sum(dim=1, keepdim=True)
    if bool(((mass <= 0) & mask.any(dim=1, keepdim=True)).any()):
        raise ValueError("each valid group must have positive reward-target mass")
    q = torch.where(mass > 0, q / mass.clamp_min(1e-12), torch.zeros_like(q))

    if tau_value == 0 and beta == 0:
        maxima = torch.where(mask, q, torch.full_like(q, float("-inf"))).max(
            dim=1, keepdim=True
        ).values
        hard = ((q == maxima) & mask).to(q.dtype)
        return hard / hard.sum(dim=1, keepdim=True).clamp_min(1.0)

    denom = tau_value + beta
    log_q = torch.where(q > 0, q.log(), torch.full_like(q, float("-inf")))
    logits = torch.where(mask, log_q / denom, torch.full_like(q, float("-inf")))
    if beta > 0:
        if reference_log_probs is None or reference_log_probs.shape != q.shape:
            raise ValueError("reference_log_probs must match reward_target")
        ref_values = reference_log_probs.to(device=q.device, dtype=q.dtype)
        valid_reference = ref_values[mask]
        if bool(torch.isnan(valid_reference).any()) or bool(
            torch.isposinf(valid_reference).any()
        ):
            raise ValueError("reference_log_probs must be finite or -inf")
        reference_support = (torch.isfinite(ref_values) & mask).any(
            dim=1, keepdim=True
        )
        if bool((mask.any(dim=1, keepdim=True) & ~reference_support).any()):
            raise ValueError("each valid group needs positive reference support")
        ref_logits = torch.where(
            mask,
            ref_values,
            torch.full_like(ref_values, float("-inf")),
        )
        log_reference = torch.log_softmax(ref_logits, dim=1)
        logits = logits + (beta / denom) * log_reference

    weights = torch.softmax(logits, dim=1)
    return torch.nan_to_num(weights, nan=0.0)


def candidate_projection_loss(
    sequence_log_probs: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    normalization_constant: float = 1.0,
    valid_mask: torch.Tensor | None = None,
    detach_target: bool = True,
) -> torch.Tensor:
    """Cross-entropy projection of policy sequences onto a MaxEnt target.

    This is the defining optimization step that the landed xDr experiments do
    *not* run.  xDr instead uses Gibbs weights on the signed Dr.GRPO surrogate.

    ``normalization_constant`` is one positive scalar shared by every
    candidate.  It only rescales the forward-KL objective.  A candidate-local
    ``1 / T_i`` factor is deliberately forbidden: on on-policy samples its
    expected uniform-target gradient is proportional to ``grad E[1 / T]`` and
    systematically rewards short/EOS completions rather than implementing a
    distribution projection.
    """

    if sequence_log_probs.shape != target_weights.shape:
        raise ValueError("sequence_log_probs and target_weights must match")
    mask = _valid_mask(sequence_log_probs, valid_mask)
    scale = float(normalization_constant)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("normalization_constant must be finite and positive")
    scores = sequence_log_probs / scale
    if not bool(torch.isfinite(scores[mask]).all()):
        raise ValueError("sequence_log_probs must be finite on valid candidates")

    weights = torch.where(mask, target_weights.to(scores), torch.zeros_like(scores))
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("target_weights must be finite and non-negative")
    mass = weights.sum(dim=1, keepdim=True)
    active_groups = mass.reshape(-1) > 0
    if not bool(active_groups.any()):
        return scores.sum() * 0.0
    weights = weights / mass.clamp_min(1e-12)
    if detach_target:
        weights = weights.detach()
    group_losses = -(weights * scores).sum(dim=1)
    return group_losses[active_groups].mean()


def candidate_projection_row_multipliers(
    target_weights: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flatten a grouped target into exact minibatch-decomposable multipliers.

    For a fully valid group of size ``G``, returning ``G * w_i`` makes a mean
    over rows exactly equal to the group-mean projection

    ``-mean_groups sum_i w_i log pi_i``.

    The maintained learner computes the target while prompt groups are intact,
    then shuffles rows into memory-bounded PPO minibatches.  These detached
    multipliers preserve Equation (projection) across that shuffle.  Invalid
    rows receive zero mass; each active target is normalized before scaling.
    """

    mask = _valid_mask(target_weights, valid_mask)
    weights = torch.where(
        mask,
        target_weights.float(),
        torch.zeros_like(target_weights.float()),
    )
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("target_weights must be finite and non-negative")
    mass = weights.sum(dim=1, keepdim=True)
    weights = torch.where(
        mass > 0,
        weights / mass.clamp_min(1e-12),
        torch.zeros_like(weights),
    )
    group_size = int(weights.shape[1])
    return (float(group_size) * weights).reshape(-1).detach()


def candidate_projection_minibatch_loss(
    token_log_probs: torch.Tensor,
    response_masks: torch.Tensor,
    row_multipliers: torch.Tensor,
    loss_masks: torch.Tensor,
    *,
    normalization_constant: float,
) -> torch.Tensor:
    """One shuffled-row minibatch contribution to the exact projection.

    The fixed constant preserves the forward-KL projection up to an overall
    scale and makes the expected gradient of an on-policy uniform target zero.
    Per-response length normalization does not have that property.
    """

    if token_log_probs.ndim != 2:
        raise ValueError("token_log_probs must have shape [rows, tokens]")
    if response_masks.shape != token_log_probs.shape:
        raise ValueError("response_masks must match token_log_probs")
    rows = int(token_log_probs.shape[0])
    if row_multipliers.reshape(-1).numel() != rows:
        raise ValueError("row_multipliers must contain one value per row")
    if loss_masks.reshape(-1).numel() != rows:
        raise ValueError("loss_masks must contain one value per row")
    scale = float(normalization_constant)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("normalization_constant must be finite and positive")
    mask = response_masks.to(token_log_probs)
    normalized_sequence_logps = (token_log_probs * mask).sum(dim=1) / scale
    multipliers = row_multipliers.reshape(-1).to(token_log_probs).detach()
    active = loss_masks.reshape(-1).to(token_log_probs)
    return -(normalized_sequence_logps * multipliers * active).mean()
