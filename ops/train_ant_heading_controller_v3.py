#!/usr/bin/env python3
"""Train prospective v3 Ant controller on the eight navigation headings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import train_ant_heading_controller as v1  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var/maze_runtime/controllers/ant_heading_v3"
V1_RECEIPT = ROOT / "var/maze_runtime/controllers/ant_heading_v1.training.json"
V2_RECEIPT = ROOT / "var/maze_runtime/controllers/ant_heading_v2.training.json"
COMMANDS = v1.COMMANDS[:8].copy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AntHeadingEnvV3(v1.AntHeadingEnv):
    """The v1 locomotion task restricted to eight nonzero headings."""

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        command_index = int(self.command_rng.integers(0, len(COMMANDS)))
        self.command = COMMANDS[command_index].copy()
        info["heading_command_index"] = command_index
        return self._observation(observation), info


def _factory(seed: int, rank: int, episode_steps: int) -> Callable[[], gym.Env]:
    def make() -> gym.Env:
        env = AntHeadingEnvV3(seed=seed + 10_000 * rank, episode_steps=episode_steps)
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
                    augmented = np.concatenate(
                        [np.asarray(observation, dtype=np.float32), command]
                    )
                    action, _ = model.predict(augmented, deterministic=True)
                    observation, _reward, terminated, truncated, _info = env.step(action)
                    steps += 1
                end_xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=float).copy()
                displacement = end_xy - start_xy
                rows.append(
                    {
                        "command_index": command_index,
                        "command": command.astype(float).tolist(),
                        "replicate": replicate,
                        "steps": steps,
                        "terminated_early": bool(terminated and steps < 300),
                        "displacement_xy": displacement.astype(float).tolist(),
                        "projected_displacement": float(
                            np.dot(displacement, command)
                        ),
                    }
                )
            finally:
                env.close()
    return {
        "episodes": rows,
        "summary": {
            "mean_heading_projected_displacement": float(
                np.mean([row["projected_displacement"] for row in rows])
            ),
            "minimum_heading_mean_projected_displacement": float(
                min(
                    np.mean(
                        [
                            row["projected_displacement"]
                            for row in rows
                            if row["command_index"] == command_index
                        ]
                    )
                    for command_index in range(8)
                )
            ),
            "early_termination_rate": float(
                np.mean([row["terminated_early"] for row in rows])
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73003)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps < 2048 or args.workers < 1:
        raise ValueError("timesteps must be >=2048 and workers must be positive")
    if not V1_RECEIPT.is_file() or not V2_RECEIPT.is_file():
        raise FileNotFoundError("v1 and v2 failure receipts are required")
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
        "schema_version": "ant-heading-controller-training-v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained_not_admitted",
        "information_firewall": (
            "Open-plane Ant-v5 and eight navigation headings only; no maze map, "
            "route gate, language prompt, MaxEnt outcome, or Dr.GRPO outcome."
        ),
        "prospective_change": (
            "The non-navigation STOP command is removed from the controller and "
            "language action alphabet; v1 locomotion reward and PPO contract retained."
        ),
        "seed": args.seed,
        "timesteps": args.timesteps,
        "workers": args.workers,
        "episode_steps": args.episode_steps,
        "commands": COMMANDS.astype(float).tolist(),
        "versions": {
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
            "stable_baselines3": __import__("stable_baselines3").__version__,
            "torch": torch.__version__,
        },
        "hashes": {
            "model_sha256": _sha256(model_path),
            "training_source_sha256": _sha256(Path(__file__).resolve()),
            "v1_base_source_sha256": _sha256(Path(v1.__file__).resolve()),
            "v1_failure_receipt_sha256": _sha256(V1_RECEIPT),
            "v2_failure_receipt_sha256": _sha256(V2_RECEIPT),
        },
        "evaluation": evaluation,
    }
    receipt_path = args.output.with_suffix(".training.json")
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    print(
        "[ant-heading-controller-v3] "
        f"model={model_path} sha256={receipt['hashes']['model_sha256']} "
        f"mean_projected={evaluation['summary']['mean_heading_projected_displacement']:.3f}"
    )


if __name__ == "__main__":
    main()
