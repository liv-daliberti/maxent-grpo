"""Target-blind, disjoint n=8 RNG blocks for ModeBench prompt/draw pairs.

vLLM 0.8.4 V0 expands an n=8 request with seed s into eight requests
with seeds s through s+7. Consecutive draw labels must therefore never
be passed directly to SamplingParams. This policy hashes each prompt
and draw label into an aligned block; it is invariant to task sharding,
row ordering, model identity, answers, and dataset metadata.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

POLICY = "sha256_domain_problem_draw_aligned_n8_v2"
SAMPLES = 8
BLOCK_BITS = 60


def request_seed(domain: str, problem: str, draw_label: int) -> int:
    """Return a signed-64-bit-safe base seed, reserving the next seven."""
    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a nonempty string")
    if not isinstance(problem, str) or not problem:
        raise ValueError("problem must be a nonempty string")
    if type(draw_label) is not int or draw_label < 0:
        raise ValueError("draw label must be a nonnegative integer")
    payload = json.dumps([POLICY, domain, problem, draw_label],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    block = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 4
    return block * SAMPLES


def seed_schedule(domain: str, problems: Sequence[str],
                  draw_labels: Sequence[int]) -> list[list[int]]:
    """Build rows-by-draw request seeds, refusing duplicate inputs/collisions.

    Hash collisions are very unlikely but must be detected before sampling;
    selecting another seed after a collision is not part of this policy.
    """
    if not problems or not draw_labels:
        raise ValueError("nonempty prompts and draw labels required")
    if len(set(problems)) != len(problems):
        raise ValueError("duplicate prompt text is not an independent problem")
    if len(set(draw_labels)) != len(draw_labels):
        raise ValueError("duplicate draw label")
    schedule = [[request_seed(domain, problem, label) for label in draw_labels]
                for problem in problems]
    flat = [seed for row in schedule for seed in row]
    if any(seed % SAMPLES or seed < 0 or seed + SAMPLES - 1 >= 2**63
           for seed in flat):
        raise ValueError("unaligned or out-of-range RNG block")
    if len(flat) != len(set(flat)):
        raise ValueError("RNG block collision; refuse sampling")
    return schedule
