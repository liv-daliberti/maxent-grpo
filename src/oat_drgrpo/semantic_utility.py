"""Semantic utility helpers for Dr.X-GRPO."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

_SEMANTIC_UTILITY_CORRECTNESS_THRESHOLD = 0.999


@dataclass
class SemanticDrxUtilityDiagnostics:
    """Prompt-major diagnostics for Dr.X semantic utility shaping."""

    row_cluster_prob_grouped: torch.Tensor
    semantic_entropy_grouped: torch.Tensor
    semantic_surprisal_grouped: torch.Tensor
    semantic_valid_row_mask_grouped: torch.Tensor
    active_row_mask_grouped: torch.Tensor
    quality_grouped: torch.Tensor
    active_mass_grouped: torch.Tensor
    active_count_grouped: torch.Tensor
    active_cluster_count_grouped: torch.Tensor
    semantic_gate_grouped: torch.Tensor | None = None

    @property
    def utility_grouped(self) -> torch.Tensor:
        """Backward-compatible alias for older callers/tests."""

        return self.quality_grouped


def compute_semantic_cluster_surprisal_scores(
    *,
    quality_grouped: torch.Tensor,
    cluster_ids_grouped: torch.Tensor,
    valid_row_mask_grouped: torch.Tensor | None = None,
    mass_from_quality: bool = True,
) -> tuple[torch.Tensor, SemanticDrxUtilityDiagnostics]:
    """Return centered semantic rare-mode scores from semantic-valid mode mass."""

    if quality_grouped.shape != cluster_ids_grouped.shape:
        raise ValueError(
            "quality_grouped and cluster_ids_grouped must have matching shapes."
        )
    if valid_row_mask_grouped is None:
        valid_mask = torch.ones_like(quality_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != quality_grouped.shape:
            raise ValueError("valid_row_mask_grouped must match quality_grouped.")
        valid_mask = valid_row_mask_grouped.to(torch.bool)

    semantic_valid_mask = valid_mask & (cluster_ids_grouped >= 0)
    safe_quality = torch.where(
        semantic_valid_mask,
        quality_grouped.to(torch.float32).clamp(min=0.0, max=1.0),
        torch.zeros_like(quality_grouped, dtype=torch.float32),
    )
    if mass_from_quality:
        active_quality = safe_quality
        active_row_mask = semantic_valid_mask & (safe_quality > 0.0)
    else:
        active_quality = torch.where(
            semantic_valid_mask,
            torch.ones_like(safe_quality),
            torch.zeros_like(safe_quality),
        )
        active_row_mask = semantic_valid_mask

    row_cluster_prob = torch.zeros_like(quality_grouped, dtype=torch.float32)
    semantic_surprisal = torch.zeros_like(quality_grouped, dtype=torch.float32)
    semantic_entropy = torch.zeros(
        (quality_grouped.size(0),),
        device=quality_grouped.device,
        dtype=torch.float32,
    )
    active_mass = active_quality.sum(dim=1)
    active_count = active_row_mask.to(torch.float32).sum(dim=1)
    active_cluster_count = torch.zeros_like(active_count)

    for p in range(quality_grouped.size(0)):
        prompt_active_mask = active_row_mask[p]
        prompt_active_mass = float(active_mass[p].item())
        if prompt_active_mass <= 0.0:
            continue
        prompt_cluster_ids = cluster_ids_grouped[p, prompt_active_mask]
        unique_clusters = torch.unique(prompt_cluster_ids, sorted=True)
        active_cluster_count[p] = float(unique_clusters.numel())
        cluster_mass_map: dict[int, float] = {}
        cluster_probs = []
        for cluster_id in unique_clusters.tolist():
            member_mask = prompt_active_mask & (
                cluster_ids_grouped[p] == int(cluster_id)
            )
            cluster_mass = float(active_quality[p, member_mask].sum().item())
            cluster_mass_map[int(cluster_id)] = cluster_mass
            cluster_probs.append(cluster_mass / max(prompt_active_mass, 1e-12))
        probs = torch.tensor(
            cluster_probs,
            device=quality_grouped.device,
            dtype=torch.float32,
        )
        if unique_clusters.numel() > 1:
            semantic_entropy[p] = -(probs * probs.clamp(min=1e-12).log()).sum()
        else:
            semantic_entropy[p] = 0.0
        prompt_entropy = float(semantic_entropy[p].item())
        for row_idx in torch.where(prompt_active_mask)[0].tolist():
            cluster_id = int(cluster_ids_grouped[p, row_idx].item())
            cluster_prob = cluster_mass_map.get(cluster_id, 0.0) / max(
                prompt_active_mass,
                1e-12,
            )
            row_cluster_prob[p, row_idx] = float(cluster_prob)
            if cluster_prob > 0.0:
                centered_surprisal = -math.log(cluster_prob) - prompt_entropy
                semantic_surprisal[p, row_idx] = float(max(0.0, centered_surprisal))

    diagnostics = SemanticDrxUtilityDiagnostics(
        row_cluster_prob_grouped=row_cluster_prob,
        semantic_entropy_grouped=semantic_entropy,
        semantic_surprisal_grouped=semantic_surprisal,
        semantic_valid_row_mask_grouped=semantic_valid_mask,
        active_row_mask_grouped=active_row_mask,
        quality_grouped=safe_quality,
        active_mass_grouped=active_mass,
        active_count_grouped=active_count,
        active_cluster_count_grouped=active_cluster_count,
    )
    return semantic_surprisal, diagnostics


def compute_quality_centered_semantic_drx_utilities(
    *,
    reward_grouped: torch.Tensor,
    cluster_ids_grouped: torch.Tensor,
    semantic_entropy_lambda: float,
    candidate_correctness_grouped: torch.Tensor | None = None,
    valid_row_mask_grouped: torch.Tensor | None = None,
    semantic_correctness_target_frac: float = 0.5,
    semantic_correctness_sharpness: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, SemanticDrxUtilityDiagnostics]:
    """Return exact-listwise utilities with correctness-rate-gated semantics."""

    if reward_grouped.shape != cluster_ids_grouped.shape:
        raise ValueError(
            "reward_grouped and cluster_ids_grouped must have matching shapes."
        )
    if (
        candidate_correctness_grouped is not None
        and candidate_correctness_grouped.shape != reward_grouped.shape
    ):
        raise ValueError("candidate_correctness_grouped must match reward_grouped.")
    if valid_row_mask_grouped is None:
        valid_mask = torch.ones_like(reward_grouped, dtype=torch.bool)
    else:
        if valid_row_mask_grouped.shape != reward_grouped.shape:
            raise ValueError("valid_row_mask_grouped must match reward_grouped.")
        valid_mask = valid_row_mask_grouped.to(torch.bool)

    utility_dtype = (
        reward_grouped.dtype if reward_grouped.is_floating_point() else torch.float32
    )
    quality_grouped = torch.where(
        valid_mask,
        reward_grouped.to(torch.float32).clamp(min=0.0, max=1.0),
        torch.zeros_like(reward_grouped, dtype=torch.float32),
    )
    semantic_surprisal_grouped, semantic_diag = (
        compute_semantic_cluster_surprisal_scores(
            quality_grouped=quality_grouped,
            cluster_ids_grouped=cluster_ids_grouped,
            valid_row_mask_grouped=valid_mask,
            mass_from_quality=False,
        )
    )
    if candidate_correctness_grouped is None:
        correctness_grouped = quality_grouped
    else:
        correctness_grouped = candidate_correctness_grouped.to(
            device=quality_grouped.device,
            dtype=quality_grouped.dtype,
        ).clamp(min=0.0, max=1.0)
    correct_row_mask = correctness_grouped >= _SEMANTIC_UTILITY_CORRECTNESS_THRESHOLD
    valid_count = valid_mask.to(torch.float32).sum(dim=1).clamp(min=1.0)
    correct_count = (valid_mask & correct_row_mask).to(torch.float32).sum(dim=1)
    correctness_frac = correct_count / valid_count
    semantic_target = min(
        max(float(semantic_correctness_target_frac), 0.0),
        1.0,
    )
    semantic_sharpness = max(float(semantic_correctness_sharpness), 0.0)
    semantic_gate_grouped = -torch.tanh(
        semantic_sharpness * (correctness_frac - semantic_target)
    ).to(dtype=quality_grouped.dtype)
    semantic_surprisal_adjusted = semantic_surprisal_grouped.to(
        dtype=quality_grouped.dtype
    )
    utility_grouped = torch.where(
        valid_mask,
        quality_grouped
        + semantic_gate_grouped[:, None]
        * float(semantic_entropy_lambda)
        * semantic_surprisal_adjusted,
        torch.zeros_like(quality_grouped),
    )
    utility_grouped = utility_grouped.to(dtype=utility_dtype)
    adjusted_diag = SemanticDrxUtilityDiagnostics(
        row_cluster_prob_grouped=semantic_diag.row_cluster_prob_grouped,
        semantic_entropy_grouped=semantic_diag.semantic_entropy_grouped,
        semantic_surprisal_grouped=semantic_surprisal_adjusted,
        semantic_valid_row_mask_grouped=semantic_diag.semantic_valid_row_mask_grouped,
        active_row_mask_grouped=semantic_diag.active_row_mask_grouped,
        quality_grouped=semantic_diag.quality_grouped,
        active_mass_grouped=semantic_diag.active_mass_grouped,
        active_count_grouped=semantic_diag.active_count_grouped,
        active_cluster_count_grouped=semantic_diag.active_cluster_count_grouped,
        semantic_gate_grouped=semantic_gate_grouped,
    )
    return utility_grouped, quality_grouped.to(dtype=utility_dtype), adjusted_diag
