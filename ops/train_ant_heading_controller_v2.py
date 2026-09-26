#!/usr/bin/env python3
"""Train the prospective v2 command-conditioned Ant locomotion controller.

Version 1 passed all heading-motion gates but failed its frozen stationary
command gate. Version 2 is prospectively separated: it trains only on
open-plane Ant-v5, samples STOP on half of episodes, observes displacement
from the current primitive's start, and penalizes both speed and drift for
STOP. It never sees a maze map, route gate, language prompt, or treatment
outcome.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "var/maze_runtime/controllers/ant_heading_v2"
V1_RECEIPT = ROOT / "var/maze_runtime/controllers/ant_heading_v1.training.json"
COMMANDS = np.asarray(
    [
        (0.0, 1.0),
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        (1.0, 0.0),
        (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
        (0.0, -1.0),
        (-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
        (-1.0, 0.0),
        (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        (0.0, 0.0),
    ],
    dtype=np.float32,
)
COMMAND_PROBABILITIES = np.asarray([1.0 / 16.0] * 8 + [0.5], dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AntHeadingEnvV2(gym.Wrapper):
    """Open-plane Ant with an episode-fixed heading and local displacement."""

    def __init__(self, *, seed: int, episode_steps: int = 300) -> None:
        env = gym.make(
            "Ant-v5",
            max_episode_steps=episode_steps,
            exclude_current_positions_from_observation=True,
            reset_noise_scale=0.1,
        )
        super().__init__(env)
        self.command_rng = np.random.default_rng(seed)
        self.command = np.zeros(2, dtype=np.float32)
        self.primitive_start_xy = np.zeros(2, dtype=np.float32)
        base = env.observation_space
        assert isinstance(base, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [
                    base.low,
                    np.full(2, -1.0, dtype=np.float32),
                    np.full(2, -np.inf, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    base.high,
                    np.full(2, 1.0, dtype=np.float32),
                    np.full(2, np.inf, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        )

    def _displacement(self) -> np.ndarray:
        return (
            np.asarray(self.env.unwrapped.data.qpos[:2], dtype=np.float32)
            - self.primitive_start_xy
        )

    def _observation(self, observation) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(observation, dtype=np.float32),
                self.command,
                self._displacement(),
            ]
        ).astype(np.float32, copy=False)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        command_index = int(
            self.command_rng.choice(len(COMMANDS), p=COMMAND_PROBABILITIES)
        )
        self.command = COMMANDS[command_index].copy()
        self.primitive_start_xy = np.asarray(
            self.env.unwrapped.data.qpos[:2], dtype=np.float32
        ).copy()
        info["heading_command_index"] = command_index
        return self._observation(observation), info

    def step(self, action):
        observation, _native_reward, terminated, truncated, info = self.env.step(action)
        velocity = np.asarray(
            [float(info.get("x_velocity", 0.0)), float(info.get("y_velocity", 0.0))],
            dtype=np.float32,
        )
        speed = float(np.linalg.norm(velocity))
        command_norm = float(np.linalg.norm(self.command))
        if command_norm > 0.0:
            projected = float(np.dot(velocity, self.command))
            lateral = float(
                abs(-velocity[0] * self.command[1] + velocity[1] * self.command[0])
            )
            command_reward = 2.0 * projected - 0.25 * lateral
        else:
            drift = float(np.linalg.norm(self._displacement()))
            command_reward = -4.0 * speed - 1.5 * drift
        reward = (
            command_reward
            + float(info.get("reward_survive", 1.0))
            + 0.25 * float(info.get("reward_ctrl", 0.0))
            + 0.1 * float(info.get("reward_contact", 0.0))
        )
        info["heading_command_x"] = float(self.command[0])
        info["heading_command_y"] = float(self.command[1])
        info["heading_projected_velocity"] = (
            float(np.dot(velocity, self.command)) if command_norm > 0.0 else -speed
        )
        info["primitive_displacement"] = float(np.linalg.norm(self._displacement()))
        return self._observation(observation), reward, terminated, truncated, info


def _factory(seed: int, rank: int, episode_steps: int) -> Callable[[], gym.Env]:
    def make() -> gym.Env:
        env = AntHeadingEnvV2(seed=seed + 10_000 * rank, episode_steps=episode_steps)
        env.reset(seed=seed + rank)
        return env

    return make


def _evaluate(model: PPO, *, seed: int, episodes_per_command: int = 3) -> dict:
    rows = []
    for command_index, command in enumerate(COMMANDS):
        for replicate in range(episodes_per_command):
            env = gym.make(
                "Ant-v5",
                max_episode_steps=300,
                exclude_current_positions_from_observation=True,
                reset_noise_scale=0.1,
            )
            try:
                observation, _ = env.reset(seed=seed + 1000 * command_index + replicate)
                start_xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=float).copy()
                terminated = truncated = False
                steps = 0
                while not (terminated or truncated):
                    current_xy = np.asarray(
                        env.unwrapped.data.qpos[:2], dtype=np.float32
                    )
                    augmented = np.concatenate(
                        [
                            np.asarray(observation, dtype=np.float32),
                            command,
                            current_xy - start_xy.astype(np.float32),
                        ]
                    )
                    action, _ = model.predict(augmented, deterministic=True)
                    observation, _reward, terminated, truncated, _info = env.step(action)
                    steps += 1
                end_xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=float).copy()
                displacement = end_xy - start_xy
                norm = float(np.linalg.norm(command))
                projected = (
                    float(np.dot(displacement, command)) if norm > 0.0 else 0.0
                )
                rows.append(
                    {
                        "command_index": command_index,
                        "command": command.astype(float).tolist(),
                        "replicate": replicate,
                        "steps": steps,
                        "terminated_early": bool(terminated and steps < 300),
                        "displacement_xy": displacement.astype(float).tolist(),
                        "projected_displacement": projected,
                        "stationary_displacement": (
                            float(np.linalg.norm(displacement)) if norm == 0.0 else None
                        ),
                    }
                )
            finally:
                env.close()
    heading_rows = [row for row in rows if row["command_index"] < 8]
    stop_rows = [row for row in rows if row["command_index"] == 8]
    return {
        "episodes": rows,
        "summary": {
            "mean_heading_projected_displacement": float(
                np.mean([row["projected_displacement"] for row in heading_rows])
            ),
            "minimum_heading_mean_projected_displacement": float(
                min(
                    np.mean(
                        [
                            row["projected_displacement"]
                            for row in heading_rows
                            if row["command_index"] == command_index
                        ]
                    )
                    for command_index in range(8)
                )
            ),
            "early_termination_rate": float(
                np.mean([row["terminated_early"] for row in rows])
            ),
            "mean_stop_displacement": float(
                np.mean([row["stationary_displacement"] for row in stop_rows])
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73002)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps < 2048 or args.workers < 1:
        raise ValueError("timesteps must be >=2048 and workers must be positive")
    if not V1_RECEIPT.is_file():
        raise FileNotFoundError("v1 failure receipt is required for v2 provenance")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    factories = [
        _factory(args.seed, rank, args.episode_steps) for rank in range(args.workers)
    ]
    vec_env = (
        DummyVecEnv(factories)
        if args.workers == 1
        else SubprocVecEnv(factories, start_method="forkserver")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        n_steps = max(256, 2048 // args.workers)
        batch_size = min(512, n_steps * args.workers)
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [256, 256]},
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model.save(args.output)
    finally:
        vec_env.close()

    model_path = args.output.with_suffix(".zip")
    loaded = PPO.load(model_path, device="cpu")
    evaluation = _evaluate(loaded, seed=args.seed + 1_000_000)
    receipt = {
        "schema_version": "ant-heading-controller-training-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained_not_admitted",
        "information_firewall": (
            "Open-plane Ant-v5 only; no maze map, route gate, language prompt, "
            "MaxEnt outcome, or Dr.GRPO outcome was used."
        ),
        "prospective_change_from_v1": (
            "STOP sampled with probability 0.5; local primitive displacement "
            "added to observation; STOP penalizes speed and displacement."
        ),
        "seed": args.seed,
        "timesteps": args.timesteps,
        "workers": args.workers,
        "episode_steps": args.episode_steps,
        "commands": COMMANDS.astype(float).tolist(),
        "command_probabilities": COMMAND_PROBABILITIES.astype(float).tolist(),
        "versions": {
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
            "stable_baselines3": __import__("stable_baselines3").__version__,
            "torch": torch.__version__,
        },
        "hashes": {
            "model_sha256": _sha256(model_path),
            "training_source_sha256": _sha256(Path(__file__).resolve()),
            "v1_failure_receipt_sha256": _sha256(V1_RECEIPT),
        },
        "evaluation": evaluation,
    }
    receipt_path = args.output.with_suffix(".training.json")
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    print(
        "[ant-heading-controller-v2] "
        f"model={model_path} sha256={receipt['hashes']['model_sha256']} "
        f"mean_projected={evaluation['summary']['mean_heading_projected_displacement']:.3f} "
        f"mean_stop={evaluation['summary']['mean_stop_displacement']:.3f}"
    )


if __name__ == "__main__":
    main()
