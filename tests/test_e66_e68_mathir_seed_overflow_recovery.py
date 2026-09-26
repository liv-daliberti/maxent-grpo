from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "ops/exp_scaling/recover_e66_e68_mathir_seed_overflow.py"
)
CONTRACT = (
    ROOT
    / "ops/exp_scaling/e66_e68_mathir_seed_overflow_contract.py"
)
E66_AUDITOR = (
    ROOT / "ops/exp_scaling/audit_e66_same_plumbing_actuator_ablation.py"
)
E68_AUDITOR = (
    ROOT / "ops/exp_scaling/audit_e68_separated_support_actuator_ablation.py"
)
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e66_e68_mathir_seed_overflow_recovery_amendment_20260728.md"
)


def _load():
    spec = importlib.util.spec_from_file_location("seedwrap_recovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "seedwrap_recovery_contract",
        CONTRACT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_covers_exact_paired_mathir_jobs():
    recovery = _load()

    assert recovery.CHECKPOINT_TAG == "step_04224"
    assert recovery.ARMS["e66"]["jobs"] == {
        43: 30128403,
        44: 30128404,
        45: 30128405,
    }
    assert recovery.ARMS["e68"]["jobs"] == {
        43: 30130478,
        44: 30130479,
        45: 30130480,
    }


def test_recovery_source_patch_is_exactly_two_files():
    recovery = _load()

    assert set(recovery.PATCHED_FILES) == {
        "oat_drgrpo/learner/grpo.py",
        "oat_drgrpo/replicated_group.py",
    }
    assert set(recovery.PATCHED_FILE_HASHES) == set(recovery.PATCHED_FILES)


def test_recovery_submission_only_mutates_execution_provenance():
    recovery = _load()
    job = {
        "argv": [
            "sbatch",
            "--parsable",
            (
                "--export=ALL,RUN_STAMP=frozen_s43,"
                "OAT_ZERO_SOURCE_ROOT=/old/src,OAT_ZERO_LOCAL_ROOT=/tmp/old,"
                "OAT_ZERO_VARIANT=arm,OAT_ZERO_SEED=43"
            ),
            "--hold",
            "/old/train.slurm",
        ],
        "export_items": [
            "ALL",
            "RUN_STAMP=frozen_s43",
            "OAT_ZERO_SOURCE_ROOT=/old/src",
            "OAT_ZERO_LOCAL_ROOT=/tmp/old",
            "OAT_ZERO_VARIANT=arm",
            "OAT_ZERO_SEED=43",
        ],
        "exports": {
            "RUN_STAMP": "frozen_s43",
            "OAT_ZERO_SOURCE_ROOT": "/old/src",
            "OAT_ZERO_LOCAL_ROOT": "/tmp/old",
            "OAT_ZERO_VARIANT": "arm",
            "OAT_ZERO_SEED": "43",
        },
        "patched_source": "/new/src",
        "checkpoint": {"checkpoint_root": "/old/checkpoints"},
    }

    argv = recovery._recovery_argv(job)
    exports = recovery._export_dict(recovery._split_submit_line(shlex_join(argv))[1])

    assert exports["RUN_STAMP"] == "frozen_s43"
    assert exports["OAT_ZERO_VARIANT"] == "arm"
    assert exports["OAT_ZERO_SEED"] == "43"
    assert exports["OAT_ZERO_SOURCE_ROOT"] == "/new/src"
    assert exports["OAT_ZERO_INITIAL_RESUME_DIR"] == "/old/checkpoints"
    assert exports["OAT_ZERO_INITIAL_RESUME_TAG"] == "step_04224"


def shlex_join(argv):
    import shlex

    return shlex.join(argv)


def test_amendment_forbids_outcome_dependent_changes():
    normalized = " ".join(AMENDMENT.read_text(encoding="utf-8").split())

    assert "change exactly" in normalized
    assert "preserve the original run stamp" in normalized
    assert "Any uncaught traceback in a recovery attempt remains fatal" in normalized
    assert "No observed E66/E68 outcome is used" in normalized


def test_registered_overflow_classifier_is_narrow():
    contract = _load_contract()
    traceback = "[rank0]: Traceback (most recent call last):"
    registered = (
        traceback
        + "\nValueError: Seed must be between 0 and 2**32 - 1\n"
    )
    unrelated = traceback + "\nRuntimeError: unrelated\n"

    assert contract.is_registered_seed_overflow(
        registered,
        registered.index(traceback) + len(traceback),
    )
    assert not contract.is_registered_seed_overflow(
        unrelated,
        unrelated.index(traceback) + len(traceback),
    )


def test_campaign_auditors_only_ignore_registered_original_overflow(tmp_path):
    e66 = _load_path("e66_recovery_audit", E66_AUDITOR)
    e68 = _load_path("e68_recovery_audit", E68_AUDITOR)
    log = tmp_path / "original.out"
    log.write_text(
        "[rank0]: Traceback (most recent call last):\n"
        "  File \"numpy/random/mtrand.pyx\", line 186, in "
        "numpy.random.mtrand.RandomState.__init__\n"
        "ValueError: Seed must be between 0 and 2**32 - 1\n",
        encoding="utf-8",
    )

    assert e66._scan_log(
        log,
        allow_registered_seed_overflow=True,
    ) == ([], 1)
    assert e68._scan_log(
        log,
        allow_registered_seed_overflow=True,
    ) == ([], 1)
    assert e66._scan_log(log)[0]
    assert e68._scan_log(log)[0]
