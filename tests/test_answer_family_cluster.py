from types import SimpleNamespace

import torch

from oat_drgrpo.listwise import normalize_semantic_cluster_method
from oat_drgrpo.semantic_clusters import build_runtime_semantic_cluster_bundle


def test_answer_family_cluster_method_aliases_exact_answers_only():
    assert normalize_semantic_cluster_method("answer") == "answer_family"
    assert normalize_semantic_cluster_method("exact_answer") == "answer_family"


def test_answer_family_runtime_clusters_by_final_answer_key_only():
    args = SimpleNamespace(maxent_semantic_cluster_method="answer_family")
    bundle = build_runtime_semantic_cluster_bundle(
        args=args,
        default_method="greedy",
        final_answer_keys_grouped=[["11", "11", "23", None]],
        valid_row_mask_grouped=torch.tensor([[True, True, True, True]]),
        reasoning_signature_keys_grouped=[["sig:a", "sig:b", "sig:c", "sig:d"]],
        reasoning_trace_embeddings_grouped=None,
        reasoning_trace_valid_row_mask_grouped=None,
    )

    assert bundle.cluster_ids_grouped.tolist() == [[0, 0, 1, -1]]
    assert bundle.num_clusters_per_group.tolist() == [2]
    assert bundle.semantic_valid_row_mask_grouped.tolist() == [
        [True, True, True, False]
    ]
