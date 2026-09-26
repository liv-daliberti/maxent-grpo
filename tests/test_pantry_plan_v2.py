from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pantry_v2_preregisters_three_disjoint_splits():
    protocol = (
        ROOT / "paper/preregistration/pantry_plan_three_way_split_v2_20260729.md"
    ).read_text()
    generator = (ROOT / "ops/make_pantry_plan_mode_data_v2.py").read_text()
    audit = (ROOT / "ops/audit_pantry_plan_mode_data_v2.py").read_text()

    assert "FROZEN BEFORE V2 DATA GENERATION OR MODEL SAMPLING" in protocol
    assert '"train": 384, "dev": 64, "eval": 128' in audit
    assert "train_fingerprints | dev_fingerprints" in generator
    assert "split fingerprints overlap" in generator
    assert "split_overlap_count" in audit
    assert "v1_rows_copied" in audit


def test_pantry_v2_job_has_no_model_sampling():
    slurm = (
        ROOT / "ops/slurm/materialize_pantry_plan_mode_data_v2.slurm"
    ).read_text()
    launcher = (
        ROOT / "ops/exp_scaling/launch_pantry_plan_mode_data_v2.sh"
    ).read_text()

    assert "#SBATCH --gres=gpu" not in slurm
    assert "make_pantry_plan_mode_data_v2.py" in slurm
    assert "audit_pantry_plan_mode_data_v2.py" in slurm
    assert "sbatch --parsable --hold" in launcher
    assert '"model_sampling": False' in launcher
