#!/usr/bin/env python3
"""Evaluate the prospective fixed-macro Ant heading controller v5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "var/maze_runtime/controllers/ant_heading_v1.zip"
DEFAULT_OUTPUT = ROOT / "var/maze_runtime/controllers/ant_heading_v5.evaluation.json"
EXPECTED_MODEL_SHA256 = (
    "526b669bb14cf8a07b44a1c725f94411632a3ac8a8e84966db09336cc96e4b0f"
)
COMMAND_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
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
    ],
    dtype=np.float32,
)
MACROS = ((0,), (1,), (1, 3), (3,), (4,), (5,), (6,), (7,))
HORIZON = 300


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-base", type=int, default=3_073_005)
    parser.add_argument("--episodes-per-heading", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed_base != 3_073_005 or args.episodes_per_heading != 12:
        raise ValueError("v5 seed schedule and episode count are frozen")
    model_sha256 = _sha256(args.model)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise ValueError("v5 model SHA-256 differs from preregistration")
    macro_identity = {
        "command_names": COMMAND_NAMES,
        "commands": COMMANDS.astype(float).tolist(),
        "macros": MACROS,
        "horizon": HORIZON,
    }
    model = PPO.load(args.model, device="cpu")
    rows = []
    for desired_index, desired_command in enumerate(COMMANDS):
        macro = MACROS[desired_index]
        segment_steps = HORIZON // len(macro)
        if segment_steps * len(macro) != HORIZON:
            raise ValueError("macro does not divide the frozen horizon")
        for replicate in range(args.episodes_per_heading):
            seed = args.seed_base + 1000 * desired_index + replicate
            env = gym.make(
                "Ant-v5",
                max_episode_steps=HORIZON,
                exclude_current_positions_from_observation=True,
                reset_noise_scale=0.1,
            )
            try:
                observation, _ = env.reset(seed=seed)
                start_xy = np.asarray(env.unwrapped.data.qpos[:2], dtype=float).copy()
                terminated = truncated = False
                steps = 0
                while not (terminated or truncated):
                    segment = min(steps // segment_steps, len(macro) - 1)
                    controller_command = COMMANDS[macro[segment]]
                    augmented = np.concatenate(
                        [np.asarray(observation, dtype=np.float32), controller_command]
                    )
                    action, _ = model.predict(augmented, deterministic=True)
                    observation, _reward, terminated, truncated, _info = env.step(
                        action
                    )
                    steps += 1
                displacement = (
                    np.asarray(env.unwrapped.data.qpos[:2], dtype=float) - start_xy
                )
                rows.append(
                    {
                        "desired_heading_index": desired_index,
                        "desired_heading": COMMAND_NAMES[desired_index],
                        "macro_indices": list(macro),
                        "macro_names": [COMMAND_NAMES[index] for index in macro],
                        "replicate": replicate,
                        "seed": seed,
                        "steps": steps,
                        "terminated_early": bool(
                            terminated and steps < HORIZON
                        ),
                        "displacement_xy": displacement.astype(float).tolist(),
                        "projected_displacement": float(
                            np.dot(displacement, desired_command)
                        ),
                    }
                )
            finally:
                env.close()
    heading_means = {
        str(index): float(
            np.mean(
                [
                    row["projected_displacement"]
                    for row in rows
                    if row["desired_heading_index"] == index
                ]
            )
        )
        for index in range(len(COMMANDS))
    }
    numeric_values = [
        value
        for row in rows
        for value in [*row["displacement_xy"], row["projected_displacement"]]
    ]
    summary = {
        "episodes": len(rows),
        "heading_mean_projected_displacement": heading_means,
        "minimum_heading_mean_projected_displacement": min(
            heading_means.values()
        ),
        "mean_heading_projected_displacement": float(
            np.mean([row["projected_displacement"] for row in rows])
        ),
        "early_termination_rate": float(
            np.mean([row["terminated_early"] for row in rows])
        ),
        "all_metrics_finite": all(
            math.isfinite(float(value)) for value in numeric_values
        ),
    }
    checks = {
        "episode_count_is_96": len(rows) == 96,
        "model_sha256_matches": model_sha256 == EXPECTED_MODEL_SHA256,
        "macro_identity_matches": (
            _canonical_sha256(macro_identity)
            == _canonical_sha256(
                {
                    "command_names": COMMAND_NAMES,
                    "commands": COMMANDS.astype(float).tolist(),
                    "macros": MACROS,
                    "horizon": 300,
                }
            )
        ),
        "all_metrics_finite": summary["all_metrics_finite"],
        "minimum_heading_mean_at_least_2": (
            summary["minimum_heading_mean_projected_displacement"] >= 2.0
        ),
        "overall_heading_mean_at_least_4": (
            summary["mean_heading_projected_displacement"] >= 4.0
        ),
        "early_termination_rate_at_most_0p10": (
            summary["early_termination_rate"] <= 0.10
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "schema_version": "ant-heading-controller-evaluation-v5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": (
            "admitted_to_ant_maze_route_gate"
            if status == "pass"
            else "ant_heading_v5_ineligible"
        ),
        "information_boundary": (
            "Open-plane Ant-v5 only; no maze, route, language completion, "
            "MaxEnt outcome, or Dr.GRPO outcome was loaded."
        ),
        "seed_base": args.seed_base,
        "episodes_per_heading": args.episodes_per_heading,
        "macro_identity": macro_identity,
        "macro_identity_sha256": _canonical_sha256(macro_identity),
        "checks": checks,
        "summary": summary,
        "episodes": rows,
        "hashes": {
            "model_sha256": model_sha256,
            "evaluation_source_sha256": _sha256(Path(__file__).resolve()),
            "preregistration_sha256": _sha256(
                ROOT
                / "paper/preregistration/ant_heading_controller_v5_20260729.md"
            ),
            "v4_development_receipt_sha256": _sha256(
                ROOT
                / "var/maze_runtime/controllers/ant_heading_v4.evaluation.json"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        "[ant-heading-controller-v5] "
        f"status={status} mean={summary['mean_heading_projected_displacement']:.3f} "
        f"minimum={summary['minimum_heading_mean_projected_displacement']:.3f} "
        f"early={summary['early_termination_rate']:.3f}"
    )
    raise SystemExit(0 if status == "pass" else 1)


if __name__ == "__main__":
    main()

