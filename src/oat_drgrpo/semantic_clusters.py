"""Runtime semantic-cluster routing for Dr.X-GRPO objectives."""

from __future__ import annotations

from typing import Any

import torch

from .listwise import (
    build_connected_component_semantic_cluster_bundle,
    build_semantic_cluster_bundle,
    build_spectral_semantic_cluster_bundle,
    normalize_semantic_cluster_method,
)


def build_runtime_semantic_cluster_bundle(
    *,
    args: Any,
    default_method: str,
    final_answer_keys_grouped: list[list[str | None]],
    valid_row_mask_grouped: torch.Tensor,
    reasoning_signature_keys_grouped: list[list[str | None]],
    reasoning_trace_embeddings_grouped: torch.Tensor | None,
    reasoning_trace_valid_row_mask_grouped: torch.Tensor | None,
    greedy_cluster_builder=None,
    connected_component_cluster_builder=None,
    spectral_cluster_builder=None,
):
    """Build semantic clusters with a runtime method override."""

    configured_method = normalize_semantic_cluster_method(
        getattr(args, "maxent_semantic_cluster_method", "default")
    )
    resolved_method = (
        default_method if configured_method == "default" else configured_method
    )
    common_kwargs = {
        "final_answer_keys_grouped": final_answer_keys_grouped,
        "valid_row_mask_grouped": valid_row_mask_grouped.detach(),
        "reasoning_signature_keys_grouped": reasoning_signature_keys_grouped,
        "reasoning_trace_embeddings_grouped": reasoning_trace_embeddings_grouped,
        "reasoning_trace_valid_row_mask_grouped": reasoning_trace_valid_row_mask_grouped,
    }
    if resolved_method == "greedy":
        cluster_builder = greedy_cluster_builder or build_semantic_cluster_bundle
        return cluster_builder(
            **common_kwargs,
            signature_jaccard_merge_threshold=float(
                args.maxent_semantic_similarity_threshold
            ),
            embedding_cosine_merge_threshold=float(
                args.maxent_semantic_embedding_similarity_threshold
            ),
        )
    if resolved_method == "connected_components":
        cluster_builder = (
            connected_component_cluster_builder
            or build_connected_component_semantic_cluster_bundle
        )
        return cluster_builder(
            **common_kwargs,
            signature_jaccard_merge_threshold=float(
                args.maxent_semantic_similarity_threshold
            ),
            embedding_cosine_merge_threshold=float(
                args.maxent_semantic_embedding_similarity_threshold
            ),
        )
    if resolved_method == "spectral":
        cluster_builder = (
            spectral_cluster_builder or build_spectral_semantic_cluster_bundle
        )
        return cluster_builder(
            **common_kwargs,
            spectral_max_num_clusters=int(
                getattr(args, "maxent_semantic_spectral_max_clusters", 0)
            ),
            spectral_eigengap_min=float(
                getattr(args, "maxent_semantic_spectral_eigengap_min", 0.05)
            ),
        )
    raise ValueError(f"Unsupported semantic cluster method: {resolved_method}")
