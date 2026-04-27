from oat_drgrpo.resume_state import (
    discover_local_wandb_resume_run,
    resolve_resume_progress_state,
)


def test_resume_progress_falls_back_to_checkpoint_step_when_client_state_missing():
    state = resolve_resume_progress_state(
        resume_states={},
        resume_step=240,
        update_interval=1,
        num_prompt_epochs=5,
        num_prompt_batches_per_epoch=93,
        rollout_batch_size=128,
    )

    assert state["checkpoint_step"] == 240
    assert state["next_step"] == 241
    assert state["prompt_batches_consumed_total"] == 240
    assert state["start_prompt_epoch"] == 2
    assert state["start_batch_offset"] == 54
    assert state["prompt_epoch"] == 2
    assert state["global_step"] == 240
    assert state["policy_sgd_step"] == 240.0
    assert state["query_step"] == 240 * 128
    assert state["prompt_consumed"] == 240 * 128


def test_resume_progress_prefers_explicit_prompt_cursor_state():
    state = resolve_resume_progress_state(
        resume_states={
            "steps": 240,
            "prompt_batches_consumed_total": 245,
            "prompt_epoch": 2,
            "global_step": 240,
            "policy_sgd_step": 248.0,
            "query_step": 31360,
            "prompt_consumed": 31360,
            "last_eval_query_step": 30720,
        },
        resume_step=240,
        update_interval=1,
        num_prompt_epochs=5,
        num_prompt_batches_per_epoch=93,
        rollout_batch_size=128,
    )

    assert state["checkpoint_step"] == 240
    assert state["next_step"] == 241
    assert state["prompt_batches_consumed_total"] == 245
    assert state["start_prompt_epoch"] == 2
    assert state["start_batch_offset"] == 59
    assert state["prompt_epoch"] == 2
    assert state["global_step"] == 240
    assert state["policy_sgd_step"] == 248.0
    assert state["query_step"] == 31360
    assert state["prompt_consumed"] == 31360
    assert state["last_eval_query_step"] == 30720


def test_resume_progress_clamps_when_checkpoint_completed_all_prompt_epochs():
    state = resolve_resume_progress_state(
        resume_states={"steps": 999, "prompt_batches_consumed_total": 999},
        resume_step=999,
        update_interval=1,
        num_prompt_epochs=5,
        num_prompt_batches_per_epoch=10,
        rollout_batch_size=128,
    )

    assert state["prompt_batches_consumed_total"] == 50
    assert state["start_prompt_epoch"] == 5
    assert state["start_batch_offset"] == 0
    assert state["prompt_epoch"] == 5


def test_discover_local_wandb_resume_run_prefers_checkpoint_save_logs(tmp_path):
    wandb_root = tmp_path / "var" / "wandb" / "runs" / "wandb"
    original_run = wandb_root / "run-20260412_031820-qn00h3xh" / "files"
    resumed_run = wandb_root / "run-20260414_134024-yhjbra2b" / "files"
    original_run.mkdir(parents=True)
    resumed_run.mkdir(parents=True)

    resume_dir = (
        tmp_path
        / "var"
        / "data"
        / "seed42"
        / "qwen-seed42_0412T03:16:46"
        / "checkpoints"
    )
    resume_dir_text = str(resume_dir)

    (original_run / "output.log").write_text(
        f"[Rank 0] Saving model checkpoint: {resume_dir_text}/step_00240/mp_rank_00_model_states.pt\n",
        encoding="utf-8",
    )
    (resumed_run / "output.log").write_text(
        f"[Torch] Loading checkpoint from {resume_dir_text}/step_00240/mp_rank_00_model_states.pt\n",
        encoding="utf-8",
    )

    run_id, run_name = discover_local_wandb_resume_run(
        wandb_run_roots=[wandb_root],
        resume_dir=resume_dir_text,
        resume_tag="step_00240",
        current_run_name="qwen-seed42-resume00240_0413T23:39:46",
    )

    assert run_id == "qn00h3xh"
    assert run_name == "qwen-seed42_0412T03:16:46"


def test_discover_local_wandb_resume_run_falls_back_to_run_name_match(tmp_path):
    wandb_root = tmp_path / "var" / "wandb" / "runs" / "wandb"
    current_run = wandb_root / "run-20260414_134024-yhjbra2b" / "files"
    current_run.mkdir(parents=True)
    (current_run / "output.log").write_text(
        "wb_run_name: qwen-seed42-resume00240_0413T23:39:46\n",
        encoding="utf-8",
    )

    run_id, run_name = discover_local_wandb_resume_run(
        wandb_run_roots=[wandb_root],
        resume_dir=str(tmp_path / "missing" / "checkpoints"),
        resume_tag="step_00240",
        current_run_name="qwen-seed42-resume00240_0413T23:39:46",
    )

    assert run_id == "yhjbra2b"
    assert run_name == "qwen-seed42-resume00240_0413T23:39:46"


def test_discover_local_wandb_resume_run_resolves_merged_checkpoint_symlinks(
    tmp_path,
):
    wandb_root = tmp_path / "var" / "wandb" / "runs" / "wandb"
    original_run = wandb_root / "run-20260412_160500-hi2sv2vb" / "files"
    original_run.mkdir(parents=True)

    original_ckpt_root = (
        tmp_path
        / "var"
        / "data"
        / "seed43"
        / "qwen-seed43_0412T16:03:41"
        / "checkpoints"
    )
    (original_ckpt_root / "latest").parent.mkdir(parents=True)
    (original_ckpt_root / "step_00352").mkdir(parents=True)
    (original_ckpt_root / "latest").symlink_to(original_ckpt_root / "step_00352")

    merged_ckpt_root = (
        tmp_path / "var" / "data" / "seed43" / "qwen-seed43_merged" / "checkpoints"
    )
    merged_ckpt_root.mkdir(parents=True)
    (merged_ckpt_root / "step_00352").symlink_to(original_ckpt_root / "step_00352")
    (merged_ckpt_root / "latest").symlink_to(original_ckpt_root / "latest")

    (original_run / "output.log").write_text(
        f"[Rank 0] Saving model checkpoint: {original_ckpt_root}/step_00352/mp_rank_00_model_states.pt\n",
        encoding="utf-8",
    )

    run_id, run_name = discover_local_wandb_resume_run(
        wandb_run_roots=[wandb_root],
        resume_dir=str(merged_ckpt_root),
        resume_tag="step_00352",
        current_run_name="qwen-seed43-resume00352_0413T23:39:48",
    )

    assert run_id == "hi2sv2vb"
    assert run_name == "qwen-seed43_0412T16:03:41"
