#!/usr/bin/env python3
"""Frozen constants shared by the official-verl E113-R4 DAPO tools."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

VERL_COMMIT = "4f80e465c2ec79ab9c3c30ec74b9745de61d0490"
VERL_ROOT = ROOT / "var/cache" / f"verl_dapo_{VERL_COMMIT}"
IMAGE = (
    ROOT
    / "var/cache"
    / "verl_dapo_ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0.sif"
)
IMAGE_OCI_MANIFEST_DIGEST = "sha256:335ed6cd1fe73090e458409cfa4394d6abf4cd0503ca44dbafdc28ff72e5ed20"
IMAGE_OCI_MANIFEST = (
    ROOT
    / "var/cache/apptainer-e113r4/cache/blob/blobs/sha256"
    / IMAGE_OCI_MANIFEST_DIGEST.removeprefix("sha256:")
)
IMAGE_BUILDER_SOURCE = ROOT / "var/cache/apptainer-1.5.3.tar.gz"
IMAGE_BUILDER_SOURCE_SHA256 = (
    "5a3bf360a5240086324aa7f7005ab7eeee91095e2091078b3f9783eaf6e7288a"
)
IMAGE_BUILDER_VERSION = "1.5.3"
VERIFIER_SITE = ROOT / "var/cache/e113r4_verifier_site"
VERIFIER_WHEELS = {
    ROOT / "var/cache/e113r4_verifier_wheels/latex2sympy2_extended-1.11.0-py3-none-any.whl": (
        "aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234"
    ),
    ROOT / "var/cache/e113r4_verifier_wheels/math_verify-0.9.0-py3-none-any.whl": (
        "3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194"
    ),
}

DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
DOMAIN_TAGS = {
    "graph_coloring": "graph",
    "countdown": "countdown",
    "python_factors": "python",
    "mathir": "mathir",
    "pantry_plan": "pantry",
}
FAMILIES = ("qwen05b", "falcon1b")
SEEDS = {
    "qwen05b": (43, 44, 45, 46, 47),
    "falcon1b": (55, 56, 57, 58, 59),
}
MODEL_NAMES = {
    "qwen05b": "Qwen2.5-0.5B-Instruct",
    "falcon1b": "Falcon3-1B-Instruct",
}
MODEL_ROOTS = {
    "qwen05b": ROOT
    / "var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "falcon1b": ROOT
    / "var/cache/huggingface/transformers/models--tiiuae--Falcon3-1B-Instruct/snapshots/28ba2251970a01dd1edc7ba7dad2eb71216ccfdf",
}

DATA_ROOTS = {
    "graph_coloring": ROOT / "var/data/graph_coloring_modebench_v2",
    "countdown": ROOT / "var/data/exact_countdown_easy3_probe",
    "python_factors": ROOT / "var/data/python_factor_modebench_v1",
    "mathir": ROOT / "var/data/mathir_action_menu_v1",
    "pantry_plan": ROOT / "var/data/pantry_plan_modebench_v2",
}
DATA_OUTPUT = ROOT / "var/data/e113r4_official_verl_dapo"
DATA_MANIFEST = ROOT / "var/artifacts/e113r4_official_verl_dapo_data.json"

PROMPT_TEMPLATES = {
    "qwen05b": {
        "graph_coloring": "qwen_boxed",
        "countdown": "qwen_boxed",
        "python_factors": "qwen_boxed",
        "mathir": "qwen_boxed",
        "pantry_plan": "qwen_pantry_support_mask",
    },
    "falcon1b": {
        "graph_coloring": "falcon_boxed",
        "countdown": "falcon_boxed",
        "python_factors": "falcon_boxed",
        "mathir": "falcon_boxed",
        "pantry_plan": "falcon_pantry_support_mask",
    },
}
PROMPT_LENGTHS = {
    "graph_coloring": 256,
    "countdown": 256,
    "python_factors": 256,
    "mathir": 256,
    "pantry_plan": 640,
}
RESPONSE_LENGTHS = {
    "qwen05b": {
        "graph_coloring": 192,
        "countdown": 192,
        "python_factors": 192,
        "mathir": 64,
        "pantry_plan": 8,
    },
    "falcon1b": {
        "graph_coloring": 192,
        "countdown": 192,
        "python_factors": 512,
        "mathir": 128,
        "pantry_plan": 8,
    },
}

# Published DAPO uses 512 train prompts, 1536 generation prompts, n=16, and
# 32-prompt PPO minibatches.  The 384-row ModeBench corpus permits a maximum
# no-duplicate generation batch of 384, so both top-level batches are divided
# by four while the 3:1 ratio and published minibatch remain exact.
TRAIN_PROMPT_BATCH = 128
GEN_PROMPT_BATCH = 384
RESPONSES_PER_PROMPT = 16
PPO_MINI_BATCH = 32
MAX_GENERATION_BATCHES = 10
ACCEPTED_PROMPT_GROUPS = 3_072
TOTAL_TRAINING_STEPS = ACCEPTED_PROMPT_GROUPS // TRAIN_PROMPT_BATCH
MAX_TOTAL_EPOCHS = TOTAL_TRAINING_STEPS * MAX_GENERATION_BATCHES
MAX_SAMPLED_RESPONSES = (
    TOTAL_TRAINING_STEPS
    * MAX_GENERATION_BATCHES
    * GEN_PROMPT_BATCH
    * RESPONSES_PER_PROMPT
)

LEARNING_RATE = 1e-6
CLIP_LOW = 0.20
CLIP_HIGH = 0.28
CLIP_C = 10.0
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 10
GRAD_CLIP = 1.0
OVERLONG_RATIO = 0.20

PROTOCOL = ROOT / "paper/preregistration/e113r4_official_verl_dapo_20260819.md"
LEDGER = ROOT / "var/artifacts/e113r4_official_verl_dapo_jobs.json"
RETIREMENT = ROOT / "var/artifacts/e113r3_retirement_for_official_dapo.json"
R3_LEDGER = ROOT / "var/artifacts/e113r3_dapo_full_relaunch_jobs.json"


def parquet_path(domain: str, split: str) -> Path:
    return DATA_OUTPUT / f"{DOMAIN_TAGS[domain]}_{split}.parquet"


def response_length(family: str, domain: str) -> int:
    return RESPONSE_LENGTHS[family][domain]


def overlong_buffer(family: str, domain: str) -> int:
    """Nearest positive integer to the published 20% response buffer."""

    return max(1, int(RESPONSE_LENGTHS[family][domain] * OVERLONG_RATIO + 0.5))


def run_stamp(family: str, domain: str, seed: int, *, smoke: bool = False) -> str:
    prefix = "e113r4s0" if smoke else "e113r4"
    return f"{prefix}_{family}_{DOMAIN_TAGS[domain]}_official_dapo_s{seed}"


def run_dir(family: str, domain: str, seed: int, *, smoke: bool = False) -> Path:
    return ROOT / "var/data" / run_stamp(family, domain, seed, smoke=smoke)
