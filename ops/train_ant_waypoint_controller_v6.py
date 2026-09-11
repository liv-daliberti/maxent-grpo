#!/usr/bin/env python3
"""Train a maze-blind Ant controller to reach a relative 4-unit waypoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


ROOT = Path(
    os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
DEFAULT_OUTPUT = ROOT / "var/maze_runtime/controllers/ant_waypoint_v6"
INITIAL_MODEL = ROOT / "var/maze_runtime/controllers/ant_heading_v1.zip"
INITIAL_MODEL_SHA256 = (
    "526b669bb14cf8a07b44a1c725f94411632a3ac8a8e84966db09336cc96e4b0f"
)
HEADINGS = np.asarray(
    [
        (0.0, 1.0),
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        (1.0, 0.0),
        (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
        (0.0, -1.0),
        (-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
        (-1.0, 0.0),
        (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
    ],
    dtype=np.float32,
)
WAYPOINT_DISTANCE = 4.0
TRAINING_DISTANCES = np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float32)
SUCCESS_THRESHOLD = 0.45


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AntWaypointEnv(gym.Wrapper):
    """Open-plane Ant with a relative target and no maze information."""

    def __init__(self, *, seed: int, episode_steps: int = 400) -> None:
        env = gym.make(
            "Ant-v5",
            max_episode_steps=episode_steps,
            exclude_current_positions_from_observation=True,
            reset_noise_scale=0.1,
        )
        super().__init__(env)
        self.target_rng = np.random.default_rng(seed)
        self.target_xy = np.zeros(2, dtype=np.float32)
        self.previous_distance = WAYPOINT_DISTANCE
        base = env.observation_space
        assert isinstance(base, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([base.low, np.full(2, -1.0, dtype=np.float32)]),
            high=np.concatenate([base.high, np.full(2, 1.0, dtype=np.float32)]),
            dtype=np.float32,
        )

    def _current_xy(self) -> np.ndarray:
        return np.asarray(self.env.unwrapped.data.qpos[:2], dtype=np.float32)

    def _observation(self, observation) -> np.ndarray:
        relative = np.clip(
            (self.target_xy - self._current_xy()) / WAYPOINT_DISTANCE,
            -1.0,
            1.0,
        )
        return np.concatenate(
            [np.asarray(observation, dtype=np.float32), relative]
        ).astype(np.float32, copy=False)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        heading_index = int(self.target_rng.integers(0, len(HEADINGS)))
        target_distance = float(self.target_rng.choice(TRAINING_DISTANCES))
        self.target_xy = (
            self._current_xy() + target_distance * HEADINGS[heading_index]
        ).astype(np.float32)
        self.previous_distance = target_distance
        info["waypoint_heading_index"] = heading_index
        info["waypoint_target_distance"] = target_distance
        return self._observation(observation), info

    def step(self, action):
        observation, _reward, terminated, truncated, info = self.env.step(action)
        distance = float(np.linalg.norm(self.target_xy - self._current_xy()))
        progress = self.previous_distance - distance
        self.previous_distance = distance
        success = distance <= SUCCESS_THRESHOLD
        reward = (
            20.0 * progress
            - 0.02 * distance
            - 0.01
            + 0.10 * float(info.get("reward_survive", 1.0))
            + 0.10 * float(info.get("reward_ctrl", 0.0))
            + 0.05 * float(info.get("reward_contact", 0.0))
            + (50.0 if success else 0.0)
            - (25.0 if terminated and not success else 0.0)
        )
        info["waypoint_distance"] = distance
        info["waypoint_success"] = success
        return (
            self._observation(observation),
            reward,
            bool(terminated or success),
            truncated,
            info,
        )


def _factory(seed: int, rank: int, episode_steps: int) -> Callable[[], gym.Env]:
    def make() -> gym.Env:
        env = AntWaypointEnv(
            seed=seed + 10_000 * rank,
            episode_steps=episode_steps,
        )
        env.reset(seed=seed + rank)
        return env

    return make


def _evaluate(
    model: PPO,
    *,
    seed: int,
    episodes_per_heading: int,
    episode_steps: int,
) -> dict:
    episodes = []
    for heading_index, heading in enumerate(HEADINGS):
        for replicate in range(episodes_per_heading):
            env = gym.make(
                "Ant-v5",
                max_episode_steps=episode_steps,
                exclude_current_positions_from_observation=True,
                reset_noise_scale=0.1,
            )
            try:
                observation, _ = env.reset(
                    seed=seed + 1000 * heading_index + replicate
                )
                start = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float32)
                target = start + WAYPOINT_DISTANCE * heading
                success = False
                terminated = truncated = False
                steps = 0
                while not (terminated or truncated or success):
                    current = np.asarray(
                        env.unwrapped.data.qpos[:2], dtype=np.float32
                    )
                    augmented = np.concatenate(
                        [
                            np.asarray(observation, dtype=np.float32),
                            np.clip(
                                (target - current) / WAYPOINT_DISTANCE,
                                -1.0,
                                1.0,
                            ),
                        ]
                    )
                    action, _ = model.predict(augmented, deterministic=True)
                    observation, _reward, terminated, truncated, _info = env.step(
                        action
                    )
                    steps += 1
                    current = np.asarray(
                        env.unwrapped.data.qpos[:2], dtype=np.float32
                    )
                    distance = float(np.linalg.norm(target - current))
                    success = distance <= SUCCESS_THRESHOLD
                episodes.append(
                    {
                        "heading_index": heading_index,
                        "replicate": replicate,
                        "success": success,
                        "steps": steps,
                        "final_distance": distance,
                        "unhealthy_termination": bool(terminated and not success),
                    }
                )
            finally:
                env.close()
    heading_success_rates = {
        str(index): float(
            np.mean(
                [
                    row["success"]
                    for row in episodes
                    if row["heading_index"] == index
                ]
            )
        )
        for index in range(len(HEADINGS))
    }
    successful_steps = [row["steps"] for row in episodes if row["success"]]
    summary = {
        "episodes": len(episodes),
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "minimum_heading_success_rate": min(heading_success_rates.values()),
        "heading_success_rates": heading_success_rates,
        "unhealthy_termination_rate": float(
            np.mean([row["unhealthy_termination"] for row in episodes])
        ),
        "median_success_steps": (
            float(np.median(successful_steps)) if successful_steps else None
        ),
        "all_metrics_finite": all(
            math.isfinite(float(row["final_distance"])) for row in episodes
        ),
    }
    return {"episodes": episodes, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73006)
    parser.add_argument("--episode-steps", type=int, default=400)
    parser.add_argument("--evaluation-episodes-per-heading", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--development-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.output.with_suffix(".zip")
    receipt_path = args.output.with_suffix(".evaluation.json")
    if model_path.exists() or receipt_path.exists():
        raise FileExistsError("fresh Ant waypoint output is required")
    if args.timesteps < 2048 or args.workers < 1:
        raise ValueError("invalid Ant waypoint training size")
    if not INITIAL_MODEL.is_file() or _sha256(INITIAL_MODEL) != INITIAL_MODEL_SHA256:
        raise RuntimeError("maze-blind Ant heading initialization is unavailable")
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
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=n_steps,
            batch_size=min(512, n_steps * args.workers),
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
        initial = PPO.load(INITIAL_MODEL, device="cpu")
        model.policy.load_state_dict(initial.policy.state_dict())
        del initial
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model.save(args.output)
    finally:
        vec_env.close()

    loaded = PPO.load(model_path, device="cpu")
    evaluation = _evaluate(
        loaded,
        seed=args.seed + 1_000_000,
        episodes_per_heading=args.evaluation_episodes_per_heading,
        episode_steps=args.episode_steps,
    )
    summary = evaluation["summary"]
    checks = {
        "episode_count": summary["episodes"]
        == 8 * args.evaluation_episodes_per_heading,
        "success_rate_at_least_0p90": summary["success_rate"] >= 0.90,
        "minimum_heading_success_rate_at_least_0p75": (
            summary["minimum_heading_success_rate"] >= 0.75
        ),
        "unhealthy_termination_rate_at_most_0p10": (
            summary["unhealthy_termination_rate"] <= 0.10
        ),
        "median_success_steps_at_most_300": (
            summary["median_success_steps"] is not None
            and summary["median_success_steps"] <= 300
        ),
        "all_metrics_finite": summary["all_metrics_finite"],
    }
    status = "development_smoke" if args.development_smoke else (
        "pass" if all(checks.values()) else "fail"
    )
    receipt = {
        "schema_version": "ant-waypoint-controller-v6-evaluation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": (
            "development_smoke_only"
            if args.development_smoke
            else (
                "admitted_to_fresh_maze_route_gate"
                if status == "pass"
                else "ant_waypoint_v6_ineligible"
            )
        ),
        "information_firewall": (
            "Open-plane Ant-v5 only; no maze map, route gate, language prompt, "
            "MaxEnt outcome, Dr.GRPO outcome, or v5 fresh-map trajectory loaded."
        ),
        "initialization": (
            "Exact v1 open-plane heading-policy weights; the normalized relative "
            "waypoint occupies the former two-coordinate heading input."
        ),
        "seed": args.seed,
        "timesteps": args.timesteps,
        "workers": args.workers,
        "episode_steps": args.episode_steps,
        "learning_rate": args.learning_rate,
        "waypoint_distance": WAYPOINT_DISTANCE,
        "training_waypoint_distances": TRAINING_DISTANCES.astype(float).tolist(),
        "success_threshold": SUCCESS_THRESHOLD,
        "checks": checks,
        "evaluation": evaluation,
        "hashes": {
            "model_sha256": _sha256(model_path),
            "initial_model_sha256": _sha256(INITIAL_MODEL),
            "training_source_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    print(
        "[ant-waypoint-v6] "
        f"status={status} success={summary['success_rate']:.3f} "
        f"minimum_heading={summary['minimum_heading_success_rate']:.3f} "
        f"median_steps={summary['median_success_steps']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
