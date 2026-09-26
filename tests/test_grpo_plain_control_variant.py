"""The plain-GRPO control must differ from Dr.GRPO in exactly the critic.

Every control in the paper is Dr.GRPO, so a reader can ask whether correct-mode
collapse is an artifact of Dr.GRPO's debiasing. `grpo_plain_control` answers
that. Its value depends entirely on differing from `grpo` in one key and no
others: a stray export would make it a second intervention reported as one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ops/run_experiment.sh"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e95_plain_grpo_control.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("e95_launcher_under_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def variant_block(variant: str) -> list[str]:
    """The export lines a variant sets, read from the runner itself.

    Sourcing the runner standalone is unreliable, and a skipped test guarantees
    nothing. Comparing the blocks textually always runs.
    """

    text = RUNNER.read_text(encoding="utf-8")
    block = text.split(f"\n  {variant})", 1)[1].split(";;", 1)[0]
    return sorted(
        line.strip()
        for line in block.splitlines()
        if line.strip().startswith("export ")
    )


def test_plain_grpo_is_reachable_and_selects_the_grpo_critic():
    text = RUNNER.read_text(encoding="utf-8")
    assert "grpo_plain_control)" in text
    assert "export OAT_ZERO_CRITIC_TYPE=grpo" in text


def test_train_script_no_longer_pins_the_critic():
    text = (ROOT / "ops/train.sh").read_text(encoding="utf-8")
    assert "--critic_type drgrpo" not in text, "the critic must not be hardcoded"
    assert '--critic_type "${OAT_ZERO_CRITIC_TYPE:-drgrpo}"' in text
    assert text.count("--critic_type") == 1


def test_default_is_still_drgrpo():
    """Every existing cohort must keep Dr.GRPO without setting anything."""

    text = (ROOT / "ops/train.sh").read_text(encoding="utf-8")
    assert ":-drgrpo}" in text


def test_guard_admits_exactly_two_critics():
    text = (ROOT / "src/oat_drgrpo/args.py").read_text(encoding="utf-8")
    assert 'if args.critic_type not in ("drgrpo", "grpo"):' in text
    # The bank and the semantic arms must still require Dr.GRPO.
    assert 'raise ValueError("online canonical bank requires critic_type=drgrpo")' in text


def test_plain_control_does_not_enable_the_replay_bank():
    """A control needs no bank, and the bank asserts drgrpo anyway."""

    block = RUNNER.read_text(encoding="utf-8").split("grpo_plain_control)", 1)[1]
    block = block.split(";;", 1)[0]
    assert "ONLINE_CANONICAL_REPLAY=1" not in block
    assert "ONLINE_CANONICAL_BANK_ALPHA" not in block


def test_plain_control_differs_from_grpo_in_exactly_the_critic():
    baseline = set(variant_block("grpo"))
    plain = set(variant_block("grpo_plain_control"))
    added = plain - baseline
    removed = baseline - plain
    assert added == {"export OAT_ZERO_CRITIC_TYPE=grpo"}, (
        f"plain GRPO must be one intervention, but also adds {sorted(added)}"
    )
    assert not removed, f"plain GRPO must not drop settings: {sorted(removed)}"


def test_e95_freezes_one_3b_seed_and_five_seeds_at_each_smaller_scale():
    launch = load_launcher()
    expected = {
        "Qwen2.5-3B": ((70,), 5),
        "Falcon3-1B": ((55, 56, 57, 58, 59), 25),
        "Qwen2.5-0.5B": ((43, 44, 45, 46, 47), 25),
    }
    assert len(launch.DOMAINS) == 5
    assert sum(cells for _seeds, cells in expected.values()) == 55
    for family, (seeds, cells) in expected.items():
        assert launch.FAMILY_SPECS[family]["seeds"] == seeds
        runs = [
            {"arm": "control", "domain": domain, "seed": seed}
            for domain in launch.DOMAINS
            for seed in seeds
        ]
        assert len(launch.selected_controls(family, {"runs": runs})) == cells


def test_e95_replaces_both_immutable_runtime_roots_and_keeps_jobs_held():
    launch = load_launcher()
    run = {"domain": "graph_coloring", "seed": 70}
    argv = [
        "sbatch",
        "--parsable",
        "--job-name=old",
        (
            "--export=ALL,OAT_ZERO_SEED=70,OAT_ZERO_VARIANT=old,"
            "OAT_ZERO_SOURCE_ROOT=/old/src,OAT_ZERO_OPS_SNAPSHOT_ROOT=/old/ops,"
            "SAVE_PATH=/old/save,RUN_STAMP=old"
        ),
        "--nice=100",
        "/repo/ops/slurm/train_node302.slurm",
    ]
    snapshot = Path("/immutable/e95")
    command = launch.swap(argv, "Qwen2.5-3B", run, snapshot, 0)
    exports = launch.export_pairs(command)
    assert "--hold" in command
    assert exports["OAT_ZERO_VARIANT"] == "grpo_plain_control"
    assert exports["OAT_ZERO_SOURCE_ROOT"] == "/immutable/e95/src"
    assert exports["OAT_ZERO_OPS_SNAPSHOT_ROOT"] == "/immutable/e95/ops"
    assert exports["SAVE_PATH"].endswith("e95_qwen3b_graph_coloring_grpo_s70")
    assert set(launch.PATCHED_FILES) == {
        "src/oat_drgrpo/args.py",
        "ops/run_experiment.sh",
        "ops/train.sh",
    }
