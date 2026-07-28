import pytest

from oat_drgrpo.replicated_group import (
    validate_replicated_group_layout,
)


def test_four_rank_group_uses_one_microbatch_per_rank():
    layout = validate_replicated_group_layout(
        num_samples=16,
        learner_world_size=4,
        train_batch_size=16,
        train_batch_size_per_device=4,
    )

    assert layout.local_candidate_count == 4
    assert layout.micro_batches_per_rank == 1


def test_one_rank_group_uses_memory_safe_gradient_accumulation():
    layout = validate_replicated_group_layout(
        num_samples=16,
        learner_world_size=1,
        train_batch_size=16,
        train_batch_size_per_device=4,
    )

    assert layout.local_candidate_count == 16
    assert layout.micro_batches_per_rank == 4


@pytest.mark.parametrize(
    ("train_batch_size", "microbatch", "message"),
    [
        (8, 4, "train_batch_size must equal num_samples"),
        (16, 3, "rank shard must divide"),
    ],
)
def test_invalid_logical_or_physical_layout_is_rejected(
    train_batch_size,
    microbatch,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_replicated_group_layout(
            num_samples=16,
            learner_world_size=1,
            train_batch_size=train_batch_size,
            train_batch_size_per_device=microbatch,
        )
