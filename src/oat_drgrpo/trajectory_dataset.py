"""Trajectory dataset wrapper that preserves verifier references for Dr.X."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from oat.types import TrajectoryData
from torch.utils.data import Dataset
from tqdm import tqdm

if TYPE_CHECKING:
    from oat.utils.deepspeed import DeepspeedStrategy


def _zero_pad_sequences(
    sequences: list[torch.Tensor],
    *,
    side: str = "left",
    value: int | float = 0,
) -> torch.Tensor:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right.")
    max_len = max(int(seq.size(-1)) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - int(seq.size(-1))
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


class ZeroMathTrajectoryDataset(Dataset):
    """OAT trajectory dataset plus optional per-row verifier references."""

    def __init__(
        self,
        buffer: list[TrajectoryData],
        tokenizer: Callable,
        strategy: "DeepspeedStrategy",
        **_: Any,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.trajectories: list[dict[str, Any]] = []

        for index in tqdm(
            range(len(buffer)),
            disable=not strategy.is_rank_0(),
            desc="Constructing ppo dataset",
        ):
            item = buffer[index]
            trajectory_ids = list(item.prompt_ids) + list(item.response_ids)
            self.trajectories.append(
                {
                    "input_ids": torch.tensor(trajectory_ids),
                    "attention_mask": torch.ones(len(trajectory_ids)),
                    "action_ids": item.response_ids,
                    "rewards": item.rewards,
                    "loss_mask": item.loss_mask,
                    "prompt_ids_lens": len(item.prompt_ids),
                    "action_logprobs": item.response_logprobs,
                    "reference": getattr(item, "reference", None),
                }
            )

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.trajectories[index]

    def collate_fn(self, item_list: list[dict[str, Any]]) -> dict[str, Any]:
        batch_trajectories: dict[str, Any] = {
            "input_ids": [],
            "action_ids": [],
            "attention_mask": [],
            "rewards": [],
            "loss_masks": [],
            "prompt_ids_lens": [],
            "action_logprobs": [],
            "references": [],
        }
        for item in item_list:
            batch_trajectories["input_ids"].append(item["input_ids"])
            batch_trajectories["attention_mask"].append(item["attention_mask"])
            batch_trajectories["rewards"].append(item["rewards"])
            batch_trajectories["loss_masks"].append(item["loss_mask"])
            batch_trajectories["prompt_ids_lens"].append(item["prompt_ids_lens"])
            batch_trajectories["action_logprobs"].append(item["action_logprobs"])
            batch_trajectories["action_ids"].append(item["action_ids"])
            batch_trajectories["references"].append(item.get("reference"))

        padding_side = "right"
        batch_trajectories["input_ids"] = _zero_pad_sequences(
            batch_trajectories["input_ids"],
            side=padding_side,
            value=self.tokenizer.pad_token_id,
        )
        batch_trajectories["attention_mask"] = _zero_pad_sequences(
            batch_trajectories["attention_mask"],
            side=padding_side,
        )
        return batch_trajectories
