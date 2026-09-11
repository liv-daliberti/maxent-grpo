import pytest

from ops.resolve_eval_cadence import resolve_quarter_epoch_cadence


@pytest.mark.parametrize(
    ("rollout_batch_size", "expected_steps"),
    [(1, 96), (2, 48)],
)
def test_countdown_quarter_epoch_is_hardware_independent(
    rollout_batch_size,
    expected_steps,
):
    cadence = resolve_quarter_epoch_cadence(
        prompt_pool_size=384,
        max_train=384,
        rollout_batch_size=rollout_batch_size,
    )

    assert cadence.prompt_interval == 96
    assert cadence.eval_steps == expected_steps


def test_graph_quarter_epoch_uses_effective_training_pool():
    cadence = resolve_quarter_epoch_cadence(
        prompt_pool_size=1024,
        max_train=192,
        rollout_batch_size=2,
    )

    assert cadence.effective_pool_size == 192
    assert cadence.prompt_interval == 48
    assert cadence.eval_steps == 24


def test_looser_requested_interval_is_capped_to_quarter_epoch():
    cadence = resolve_quarter_epoch_cadence(
        prompt_pool_size=384,
        max_train=384,
        rollout_batch_size=2,
        requested_prompt_interval=256,
    )

    assert cadence.quarter_prompt_interval == 96
    assert cadence.prompt_interval == 96
    assert cadence.eval_steps == 48


def test_more_frequent_requested_interval_is_preserved():
    cadence = resolve_quarter_epoch_cadence(
        prompt_pool_size=384,
        max_train=384,
        rollout_batch_size=2,
        requested_prompt_interval=32,
    )

    assert cadence.prompt_interval == 32
    assert cadence.eval_steps == 16


def test_explicit_sparse_requested_interval_is_preserved():
    cadence = resolve_quarter_epoch_cadence(
        prompt_pool_size=384,
        max_train=384,
        rollout_batch_size=1,
        requested_prompt_interval=768,
        allow_sparse_requested_interval=True,
    )

    assert cadence.quarter_prompt_interval == 96
    assert cadence.prompt_interval == 768
    assert cadence.eval_steps == 768


@pytest.mark.parametrize("field", ["prompt_pool_size", "max_train", "rollout_batch_size"])
def test_nonpositive_inputs_are_rejected(field):
    kwargs = {
        "prompt_pool_size": 384,
        "max_train": 384,
        "rollout_batch_size": 1,
    }
    kwargs[field] = 0

    with pytest.raises(ValueError, match=field):
        resolve_quarter_epoch_cadence(**kwargs)
