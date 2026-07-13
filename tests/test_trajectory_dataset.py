from __future__ import annotations

from oat.types import TrajectoryData

from oat_drgrpo.trajectory_dataset import ZeroMathTrajectoryDataset


class _Tokenizer:
    pad_token_id = 0


class _Strategy:
    def is_rank_0(self):
        return False


def test_zero_math_trajectory_dataset_preserves_references():
    trajectory = TrajectoryData(
        prompt="p",
        prompt_ids=[1, 2],
        response="r",
        response_ids=[3],
        response_logprobs=[-0.1],
        rewards=[1.0],
        loss_mask=True,
        info={},
    )
    setattr(trajectory, "reference", '{"verifier":"graph_coloring"}')

    dataset = ZeroMathTrajectoryDataset([trajectory], _Tokenizer(), _Strategy())
    batch = dataset.collate_fn([dataset[0]])

    assert batch["references"] == ['{"verifier":"graph_coloring"}']
