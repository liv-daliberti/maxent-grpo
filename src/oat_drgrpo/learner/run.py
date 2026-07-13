"""Learner run loop, data setup, evaluation, and logging helpers."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from oat.interface import lp
from oat.utils.data import PromptDataset, load_data_from_disk_or_hf
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ..logging_utils import (
    add_public_drx_training_metrics,
    filter_wandb_logs_for_public_comparison,
)
from ..resume_state import (
    discover_local_wandb_resume_run,
    resolve_resume_progress_state,
)
from ..templates import apply_prompt_template_to_example, collate_eval_prompt_items


def _parse_answer_mode_count(ref: str) -> int:
    """Extract the number of valid answer modes from a modebench reference JSON."""
    try:
        spec = json.loads(ref) if isinstance(ref, str) else ref
        return max(int(spec.get("num_completions", 1)), 1)
    except Exception:
        return 1


def _compute_mode_coverage_metrics(
    rewards: list[float],
    answer_keys: list,
    answer_mode_count: int,
) -> dict[str, float]:
    k = len(rewards)
    correct_keys = {
        str(key)
        for reward, key in zip(rewards, answer_keys)
        if float(reward) > 0.0 and key is not None
    }
    distinct = len(correct_keys)
    total = max(int(answer_mode_count), 1)
    return {
        "any_correct_at_k": float(any(float(r) > 0.0 for r in rewards)),
        "mean_at_k": float(sum(float(r) for r in rewards) / k),
        "distinct_correct_modes_at_k": float(distinct),
        "mode_coverage_at_k": float(distinct) / float(total),
    }


class ZeroMathRunMixin:
    """Data, training-loop, checkpoint, W&B, and eval orchestration."""

    def _ensure_training_progress_state(self) -> None:
        if hasattr(self, "_progress_metric_baselines"):
            return
        self._progress_metric_baselines: dict[str, float] = {}
        self._progress_metric_previous: dict[str, float] = {}
        self._progress_metric_bests: dict[str, float] = {}
        self._progress_metric_best_steps: dict[str, int] = {}
        self._actor_reward_ema: float | None = None
        self._actor_reward_ema_start: float | None = None

    @staticmethod
    def _coerce_log_float(value: Any) -> float | None:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "item"):
                value = value.item()
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        if scalar != scalar or scalar in {float("inf"), float("-inf")}:
            return None
        return scalar

    def _restore_training_progress_state(
        self,
        resume_states: dict[str, Any] | None,
    ) -> None:
        self._ensure_training_progress_state()
        if not isinstance(resume_states, dict):
            return
        for attr, key in (
            ("_progress_metric_baselines", "progress_metric_baselines"),
            ("_progress_metric_previous", "progress_metric_previous"),
            ("_progress_metric_bests", "progress_metric_bests"),
            ("_progress_metric_best_steps", "progress_metric_best_steps"),
        ):
            saved = resume_states.get(key)
            if isinstance(saved, dict):
                setattr(
                    self,
                    attr,
                    {
                        str(saved_key): (
                            int(saved_value)
                            if attr == "_progress_metric_best_steps"
                            else float(saved_value)
                        )
                        for saved_key, saved_value in saved.items()
                    },
                )
        self._actor_reward_ema = self._coerce_log_float(
            resume_states.get("actor_reward_ema")
        )
        self._actor_reward_ema_start = self._coerce_log_float(
            resume_states.get("actor_reward_ema_start")
        )

    def _record_progress_metric(
        self,
        logs_dict: dict[str, Any],
        *,
        source_key: str,
        output_prefix: str,
        higher_is_better: bool = True,
    ) -> None:
        self._ensure_training_progress_state()
        value = self._coerce_log_float(logs_dict.get(source_key))
        if value is None:
            return

        baselines = self._progress_metric_baselines
        previous = self._progress_metric_previous
        bests = self._progress_metric_bests
        best_steps = self._progress_metric_best_steps

        baseline = baselines.setdefault(source_key, value)
        prev = previous.get(source_key, value)
        best = bests.get(source_key, value)
        is_new_best = value > best if higher_is_better else value < best
        if source_key not in bests or is_new_best:
            best = value
            bests[source_key] = value
            best_steps[source_key] = int(self.steps)
        previous[source_key] = value

        logs_dict[f"{output_prefix}/value"] = value
        logs_dict[f"{output_prefix}/gain_from_start"] = value - baseline
        logs_dict[f"{output_prefix}/gain_from_prev"] = value - prev
        logs_dict[f"{output_prefix}/best"] = best
        logs_dict[f"{output_prefix}/best_gain_from_start"] = best - baseline
        logs_dict[f"{output_prefix}/best_step"] = int(best_steps[source_key])
        logs_dict[f"{output_prefix}/steps_since_best"] = int(
            self.steps - best_steps[source_key]
        )

    def _add_learning_progress_metrics(self, logs_dict: dict[str, Any]) -> None:
        self._record_progress_metric(
            logs_dict,
            source_key="eval/average/accuracy",
            output_prefix="drx/progress/eval_accuracy",
        )
        self._record_progress_metric(
            logs_dict,
            source_key="eval/average/score",
            output_prefix="drx/progress/eval_score",
        )
        self._record_progress_metric(
            logs_dict,
            source_key="actor/rewards",
            output_prefix="drx/progress/rollout_reward",
        )

        actor_reward = self._coerce_log_float(logs_dict.get("actor/rewards"))
        if actor_reward is None:
            return
        self._ensure_training_progress_state()
        if self._actor_reward_ema is None:
            self._actor_reward_ema = actor_reward
            self._actor_reward_ema_start = actor_reward
        else:
            self._actor_reward_ema = 0.9 * self._actor_reward_ema + 0.1 * actor_reward
        logs_dict["drx/progress/rollout_reward_ema"] = self._actor_reward_ema
        if self._actor_reward_ema_start is not None:
            logs_dict["drx/progress/rollout_reward_ema_gain_from_start"] = (
                self._actor_reward_ema - self._actor_reward_ema_start
            )

    def _format_compact_training_sample(self) -> str | None:
        """Return a compact rollout sample summary for console logs."""

        if not self.pi_buffer:
            return None
        sample = np.random.choice(self.pi_buffer)
        prompt = str(getattr(sample, "prompt", "") or "")
        response = str(getattr(sample, "response", "") or "")
        response_ids = getattr(sample, "response_ids", None)
        rewards = getattr(sample, "rewards", None)
        info = getattr(sample, "info", None)
        response_tokens = len(response_ids) if response_ids is not None else None
        reward_mean = None
        if rewards is not None:
            try:
                reward_values = list(rewards)
                if reward_values:
                    reward_mean = sum(float(value) for value in reward_values) / len(
                        reward_values
                    )
            except (TypeError, ValueError):
                reward_mean = None
        pieces = [
            f"prompt_chars={len(prompt)}",
            f"response_chars={len(response)}",
        ]
        if response_tokens is not None:
            pieces.append(f"response_tokens={response_tokens}")
        if reward_mean is not None:
            pieces.append(f"reward_mean={reward_mean:.4f}")
        if isinstance(info, dict):
            actor_reward = info.get("actor/rewards")
            if actor_reward is not None:
                try:
                    pieces.append(f"actor_reward={float(actor_reward):.4f}")
                except (TypeError, ValueError):
                    pass
        return "Training sample summary: " + " ".join(pieces)

    def learn(self, learning_round: int):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier()
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.strategy,
        )
        if learning_round == 1:
            # The upstream PPO learner prints the full first training example.
            # For long math trajectories this can dominate startup time and bury
            # the actual learner-step logs without changing the optimization.
            self.strategy.print("Training example omitted for compact startup logs")

        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=(True if self.args.critic_type == "ppo" else False),
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        step_bar = tqdm(
            range(len(dataloader)),
            desc="Train steps",
            disable=not self.strategy.is_rank_0(),
        )
        learn_batch_time = []

        self.model.train()
        if self.critic is not None:
            self.critic.train()
        st = time.time()

        logging.info(
            f"start learn() buffer_len={len(self.pi_buffer)} dl_len={len(dataloader)}"
        )
        for data in dataloader:
            if local_sgd_steps > self.args.max_sgd_steps:
                break
            infos = self.learning_step(data)
            self.policy_sgd_step += (
                len(dataset)
                * self.args.num_ppo_epochs
                / self.args.train_batch_size_per_device
                / self.strategy.grad_acc_step
            )
            learn_batch_time.append(time.time() - st)
            step_bar.update()

            self.global_step += 1
            if self.global_step % self.strategy.grad_acc_step == 0:
                self.gradient_update_elapse = time.time() - self.gradient_update_st
                st = time.time()
                self.gradient_update_st = time.time()

                local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            "total_time": time.time() - st,
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        # Keep distributed logging reductions aligned even when optional metrics
        # are populated by different local minibatch conditions.
        train_info = {key: train_info[key] for key in sorted(train_info)}
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        logging.info("finish learn()")
        return train_info

    def learning_step(self, trajectory):
        if self.objective == "maxent_listwise":
            return self._listwise_learning_step(trajectory)
        if self.objective == "grpo" or self._use_instrumented_grpo_learning_step():
            return self._grpo_learning_step_with_progress(trajectory)
        return super().learning_step(trajectory)

    def prepare_data(self, strategy, tokenizer):
        prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data)
        prompts_data = prompt_dataset[self.args.train_split].select(
            range(min(self.args.max_train, len(prompt_dataset[self.args.train_split])))
        )

        # Prepare the data: templated questions & gt final answers.
        prompts_data = prompts_data.map(
            lambda x: apply_prompt_template_to_example(
                x,
                input_key=self.args.input_key,
                prompt_template=self.args.prompt_template,
            )
        )

        self.prompts_dataset = PromptDataset(
            prompts_data,
            tokenizer,
            strategy,
            input_key=self.args.input_key,
            output_key=self.args.output_key,
            apply_chat_template=False,  # Because we have applied already.
            get_reference=True,
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            self.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
        )
        self.eval_prompts_dataset = self.eval_prompts_dataloader = (
            None  # We use our own `self.eval_dataset_dict`.
        )

    def _init_wandb(self, resume_states: dict[str, Any] | None = None) -> None:
        if not self._requested_use_wb or not self.strategy.is_rank_0():
            return
        if self._wandb is not None:
            return

        import wandb

        if not wandb.api.api_key and isinstance(self._requested_use_wb, str):
            wandb.login(key=self._requested_use_wb)

        env_run_id = os.environ.get("OAT_ZERO_WANDB_RUN_ID") or os.environ.get(
            "WANDB_RUN_ID"
        )
        saved_run_id = None
        saved_run_name = None
        if isinstance(resume_states, dict):
            raw_run_id = resume_states.get("wandb_run_id")
            if isinstance(raw_run_id, str) and raw_run_id:
                saved_run_id = raw_run_id
            raw_run_name = resume_states.get("wandb_run_name")
            if isinstance(raw_run_name, str) and raw_run_name:
                saved_run_name = raw_run_name

        discovered_run_id = None
        discovered_run_name = None
        if not env_run_id and not saved_run_id and self.args.resume_dir:
            wandb_run_roots = []
            env_wandb_dir = os.environ.get("WANDB_DIR")
            if env_wandb_dir:
                wandb_run_roots.append(Path(env_wandb_dir) / "runs" / "wandb")
            wandb_run_roots.extend(
                [
                    Path.cwd() / "var" / "wandb" / "runs" / "wandb",
                    Path.cwd() / "wandb" / "runs" / "wandb",
                ]
            )
            discovered_run_id, discovered_run_name = discover_local_wandb_resume_run(
                wandb_run_roots=wandb_run_roots,
                resume_dir=self.args.resume_dir,
                resume_tag=self.args.resume_tag,
                saved_run_name=saved_run_name,
                current_run_name=self._wandb_run_name,
            )
            if discovered_run_id:
                logging.info(
                    "Recovered W&B resume run id %s from local run logs for %s",
                    discovered_run_id,
                    self.args.resume_dir,
                )

        run_id = env_run_id or saved_run_id or discovered_run_id
        if env_run_id:
            if saved_run_id or discovered_run_id:
                logging.info(
                    "Using explicit W&B run id %s; skipping resumed/discovered lineage.",
                    env_run_id,
                )
            run_name = self._wandb_run_name
        else:
            run_name = saved_run_name or discovered_run_name or self._wandb_run_name
        init_kwargs: dict[str, Any] = {
            "entity": self.args.wb_org,
            "project": self.args.wb_project,
            "group": self.args.wb_group,
            "name": run_name,
            "config": self.args.__dict__,
            "reinit": True,
        }
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"

        self._wandb = wandb
        wandb.init(**init_kwargs)
        if wandb.run is not None:
            self._wandb_run_id = wandb.run.id
            self._wandb_run_name = wandb.run.name
        else:
            self._wandb_run_id = run_id
            self._wandb_run_name = run_name

        # Use the actual training step for chart alignment across resumes.
        wandb.define_metric("trainer/step")
        wandb.define_metric("*", step_metric="trainer/step")

    def _checkpoint_client_state(self) -> dict[str, Any]:
        self._ensure_training_progress_state()
        client_state: dict[str, Any] = {
            "global_step": int(self.global_step),
            "policy_sgd_step": float(self.policy_sgd_step),
            "query_step": int(self.query_step),
            "prompt_consumed": int(self.prompt_consumed),
            "prompt_epoch": int(self.prompt_epoch),
            "steps": int(self.steps),
            "prompt_batches_consumed_total": int(self._prompt_batches_consumed_total),
            "update_interval": int(self.update_interval),
            "progress_metric_baselines": dict(self._progress_metric_baselines),
            "progress_metric_previous": dict(self._progress_metric_previous),
            "progress_metric_bests": dict(self._progress_metric_bests),
            "progress_metric_best_steps": dict(self._progress_metric_best_steps),
        }
        if self._actor_reward_ema is not None:
            client_state["actor_reward_ema"] = float(self._actor_reward_ema)
        if self._actor_reward_ema_start is not None:
            client_state["actor_reward_ema_start"] = float(self._actor_reward_ema_start)
        if hasattr(self, "last_eval_query_step"):
            client_state["last_eval_query_step"] = int(self.last_eval_query_step)
        if hasattr(self, "_pending_eval"):
            client_state["_pending_eval"] = bool(self._pending_eval)
        if self._wandb_run_id:
            client_state["wandb_run_id"] = self._wandb_run_id
        if self._wandb_run_name:
            client_state["wandb_run_name"] = self._wandb_run_name
        return client_state

    def _restore_prompt_progress(
        self,
        resume_states: dict[str, Any] | None,
        resume_step: int,
    ) -> tuple[int, int, int]:
        saved_update_interval = None
        if isinstance(resume_states, dict) and "update_interval" in resume_states:
            try:
                saved_update_interval = int(resume_states["update_interval"])
            except (TypeError, ValueError):
                saved_update_interval = None
        if saved_update_interval is not None and saved_update_interval != int(
            self.update_interval
        ):
            logging.warning(
                "Checkpoint update_interval=%s does not match current update_interval=%s; "
                "continuing with the current setting.",
                saved_update_interval,
                int(self.update_interval),
            )

        progress_state = resolve_resume_progress_state(
            resume_states=resume_states,
            resume_step=resume_step,
            update_interval=int(self.update_interval),
            num_prompt_epochs=int(self.args.num_prompt_epoch),
            num_prompt_batches_per_epoch=int(len(self.prompts_dataloader)),
            rollout_batch_size=int(self.args.rollout_batch_size),
        )
        self.steps = int(progress_state["checkpoint_step"])
        self.global_step = int(progress_state["global_step"])
        self.policy_sgd_step = float(progress_state["policy_sgd_step"])
        self.query_step = int(progress_state["query_step"])
        self.prompt_consumed = int(progress_state["prompt_consumed"])
        self.prompt_epoch = int(progress_state["prompt_epoch"])
        self._prompt_batches_consumed_total = int(
            progress_state["prompt_batches_consumed_total"]
        )

        last_eval_query_step = int(progress_state["last_eval_query_step"])
        if last_eval_query_step > 0:
            self.last_eval_query_step = last_eval_query_step
        if isinstance(resume_states, dict) and "_pending_eval" in resume_states:
            self._pending_eval = bool(resume_states["_pending_eval"])
        self._restore_training_progress_state(resume_states)

        if self.steps % max(1, int(self.update_interval)) != 0:
            logging.warning(
                "Checkpoint step %s lands mid-update interval %s; optimizer step state "
                "resumes, but any in-memory rollout buffer from the unfinished interval "
                "cannot be reconstructed from checkpoints alone.",
                self.steps,
                int(self.update_interval),
            )

        used_saved_prompt_cursor = (
            isinstance(resume_states, dict)
            and "prompt_batches_consumed_total" in resume_states
        )
        log_fn = logging.info if used_saved_prompt_cursor else logging.warning
        log_fn(
            "%s prompt traversal state: checkpoint_step=%s completed_batches=%s "
            "start_epoch=%s start_batch_offset=%s query_step=%s prompt_consumed=%s",
            ("Restored" if used_saved_prompt_cursor else "Inferred fallback"),
            self.steps,
            self._prompt_batches_consumed_total,
            int(progress_state["start_prompt_epoch"]),
            int(progress_state["start_batch_offset"]),
            self.query_step,
            self.prompt_consumed,
        )

        return (
            int(progress_state["next_step"]),
            int(progress_state["start_prompt_epoch"]),
            int(progress_state["start_batch_offset"]),
        )

    def run(self):
        self._init(self.args, self.actors)

        resume_step = 0
        next_step = 1
        start_prompt_epoch = 0
        start_batch_offset = 0
        resume_states: dict[str, Any] | None = None
        if self.args.resume_dir:
            _, resume_states = self.strategy.load_ckpt(
                self.model.model,
                self.args.resume_dir,
                self.args.resume_tag,
            )
            resume_step = self._infer_resume_step(resume_states)
            next_step, start_prompt_epoch, start_batch_offset = (
                self._restore_prompt_progress(resume_states, resume_step)
            )
        else:
            self.steps = 0
            self._prompt_batches_consumed_total = 0

        early_stop = False
        self.start_time = time.time()

        self.actor_info = {}
        train_info: dict[str, Any] = {}
        self._init_wandb(resume_states)

        if not self.strategy.args.debug:
            self.eval_and_log({}, eval=True, save=False)

        self.steps = next_step
        self.gradient_update_st = time.time()
        for p_ep in range(start_prompt_epoch, self.args.num_prompt_epoch):
            batch_offset = start_batch_offset if p_ep == start_prompt_epoch else 0
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(p_ep)
                self.strategy.print(f"Set DistributedSampler at epoch {p_ep}")
            if batch_offset > 0:
                self.strategy.print(
                    "Skipping "
                    f"{batch_offset} already-consumed prompt batches in epoch {p_ep}"
                )
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
                initial=batch_offset,
            )

            for batch_idx, (processed_prompts, raw_prompts, refs) in enumerate(
                self.prompts_dataloader
            ):
                if batch_idx < batch_offset:
                    continue
                if early_stop:
                    break
                self._prompt_batches_consumed_total += 1

                feedback_data, self.actor_info = self.collector.collect_feedback(
                    raw_prompts,
                    processed_prompts,
                    refs,
                    self._same_actor_group,
                )
                dist.barrier()

                if feedback_data is None:
                    continue
                self.prompt_consumed += len(feedback_data)

                self.process_feedback_data(feedback_data)

                if (
                    self.args.dump_replay_every > 0
                    and self.steps % self.args.dump_replay_every == 0
                ):
                    if not self.strategy.is_rank_0():
                        dist.gather_object(self.pi_buffer)
                    else:
                        gather_all_buffer = [None] * self.strategy.world_size
                        dist.gather_object(self.pi_buffer, gather_all_buffer)
                        pd.to_pickle(
                            (processed_prompts, refs, gather_all_buffer),
                            os.path.join(
                                self.save_path,
                                f"buffer_step{self.steps:05}.pkl",
                            ),
                        )

                if self.steps % self.update_interval == 0:
                    logging.info("pre-learning start step=%s", self.steps)
                    self._pre_learning()
                    logging.info("learn start step=%s", self.steps)
                    train_info = self.learn(self.steps // self.update_interval)
                    logging.info("post-learning start step=%s", self.steps)
                    self._post_learning()
                    logging.info("post-learning done step=%s", self.steps)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.sync_params_every == 0:
                        logging.info("sync params to actors start step=%s", self.steps)
                        self.sync_params_to_actors()
                        logging.info("sync params to actors done step=%s", self.steps)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                    logging.info("eval/log start step=%s", self.steps)
                    self.eval_and_log(train_info)
                    logging.info("eval/log done step=%s", self.steps)

                progress_bar.update()
                self.steps += 1

                if self.get_current_query() > self.args.max_queries:
                    early_stop = True

            self.prompt_epoch = p_ep + 1
            if early_stop:
                break

        self.eval_and_log(train_info, eval=True, save=True)

        if self.args.dump_all_buffer:  # For debug purpose.
            if not self.strategy.is_rank_0():
                dist.gather_object(self.all_buffer)
            else:
                gather_all_buffer = [None] * self.strategy.world_size
                dist.gather_object(self.all_buffer, gather_all_buffer)
                pd.to_pickle(
                    gather_all_buffer,
                    os.path.join(self.save_path, "all_buffer.pkl"),
                )

        if self.strategy.is_rank_0():
            self._wandb.finish() if self._wandb else None
            lp.stop()

    def eval_and_log(self, train_info, eval=False, save=False):
        eval_info = {}
        should_eval = (self.args.eval_steps > 0 and eval) or self._should_do(
            self.args.eval_steps
        )
        should_save = (self.args.save_steps > 0 and save) or (
            self.steps > 0
            and self._should_do(self.args.save_steps)
            and self.steps >= self.args.save_from
        )

        if should_save and should_eval and self.strategy.is_rank_0():
            logging.info(
                "Checkpoint boundary at step %s: saving checkpoint before evaluation.",
                self.steps,
            )

        if should_save:
            self.strategy.save_model(
                self.model,
                self.tokenizer,
                os.path.join(self.save_path, "saved_models"),
                tag="step_{:05d}".format(self.steps),
                max_num=self.args.max_save_num,
                max_mem=self.args.max_save_mem,
            )
            if self.args.save_ckpt:
                self.strategy.save_ckpt(
                    self.model.model,
                    os.path.join(self.save_path, "checkpoints"),
                    tag="step_{:05d}".format(self.steps),
                    max_num=self.args.max_save_num,
                    max_mem=self.args.max_save_mem,
                    client_state=self._checkpoint_client_state(),
                )

        if should_eval:
            eval_info = self.evaluate(self.eval_prompts_dataloader, self.steps)

        if eval_info or self.steps % self.args.logging_steps == 0:
            misc_info = self.get_misc_info()
            misc_info["lr"] = self.scheduler.get_last_lr()[0]

            misc_info = {
                "misc/%s" % k: v
                for k, v in {
                    **misc_info,
                }.items()
            }
            logs_dict = {**train_info, **eval_info, **self.actor_info, **misc_info}
            logs_dict = self.strategy.all_reduce(logs_dict)
            logs_dict.update(
                self.strategy.all_reduce(
                    {
                        "misc/query_step": self.query_step,
                        "misc/prompt_consumed": self.prompt_consumed,
                    },
                    op="sum",
                )
            )
            logs_dict["trainer/step"] = int(self.steps)
            logs_dict["trainer/global_step"] = int(self.global_step)
            logs_dict["trainer/policy_sgd_step"] = float(self.policy_sgd_step)
            logs_dict = add_public_drx_training_metrics(logs_dict)
            self._add_learning_progress_metrics(logs_dict)

            if self.strategy.is_rank_0():
                sample_summary = self._format_compact_training_sample()
                if sample_summary:
                    self.strategy.print(sample_summary)
                self.strategy.pprint(logs_dict)
                if self._wandb is not None:
                    self._wandb.log(
                        filter_wandb_logs_for_public_comparison(logs_dict),
                        step=int(self.steps),
                    )

    def eval_dataloader_collate_fn(self, item_list):
        return collate_eval_prompt_items(
            item_list,
            prompt_template=self.args.prompt_template,
        )

    def evaluate(self, dataloader, steps):
        # Discard the default eval dataloader, and run eval on multiple benchmarks.
        del dataloader
        all_metrics = {}
        accuracies = []
        scores = []
        lens = []
        total_benchmarks = len(self.eval_dataset_dict)
        for benchmark_idx, (benchmark_name, dataset) in enumerate(
            self.eval_dataset_dict.items(), start=1
        ):
            eval_prompts_dataloader = DataLoader(
                dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.eval_dataloader_collate_fn,
            )
            if self.strategy.is_rank_0():
                logging.info(
                    "Starting eval benchmark %s/%s: %s (%s prompts, %s batches) at step %s",
                    benchmark_idx,
                    total_benchmarks,
                    benchmark_name,
                    len(dataset),
                    len(eval_prompts_dataloader),
                    steps,
                )
            metrics = super().evaluate(
                eval_prompts_dataloader, f"{steps}_{benchmark_name}"
            )
            if self.strategy.is_rank_0():
                logging.info(
                    "Finished eval benchmark %s/%s: %s accuracy=%.4f score=%.4f avg_len=%.2f at step %s",
                    benchmark_idx,
                    total_benchmarks,
                    benchmark_name,
                    float(metrics["eval/accuracy"]),
                    float(metrics["eval/score"]),
                    float(metrics["eval/response_tok_len"]),
                    steps,
                )
            all_metrics.update(
                {
                    k.replace("eval/", f"eval/{benchmark_name}/"): v
                    for k, v in metrics.items()
                }
            )
            accuracies.append(metrics["eval/accuracy"])
            scores.append(metrics["eval/score"])
            lens.append(metrics["eval/response_tok_len"])
        all_metrics.update(
            {
                "eval/average/accuracy": np.mean(accuracies),
                "eval/average/score": np.mean(scores),
                "eval/average/response_tok_len": np.mean(lens),
            }
        )

        if self.args.eval_mode_coverage_k > 0:
            mc_coverages = []
            for benchmark_name, dataset in self.eval_dataset_dict.items():
                mc = self.evaluate_mode_coverage(
                    dataset,
                    benchmark_name,
                    steps,
                    k=self.args.eval_mode_coverage_k,
                    temperature=self.args.eval_mode_coverage_temperature,
                )
                all_metrics.update(mc)
                cov_key = f"eval/{benchmark_name}/sampled_mode_coverage_at_{self.args.eval_mode_coverage_k}"
                if cov_key in mc:
                    mc_coverages.append(mc[cov_key])
            if mc_coverages:
                all_metrics[f"eval/average/sampled_mode_coverage_at_{self.args.eval_mode_coverage_k}"] = float(
                    np.mean(mc_coverages)
                )

        return all_metrics

    def evaluate_mode_coverage(
        self,
        dataset,
        benchmark_name: str,
        steps: int,
        *,
        k: int,
        temperature: float,
    ) -> dict[str, float]:
        """Sampled mode-coverage eval: K completions per prompt at T=temperature.

        Runs on the live vLLM engine without saving a checkpoint. All distributed
        ranks participate in the barriers; only rank 0 drives actor inference.
        Results are broadcast from rank 0 so that the subsequent all_reduce in
        eval_and_log produces correct per-run values.
        """
        self._pre_evaluate()
        try:
            metrics = self._run_sampled_mode_coverage(
                dataset, benchmark_name, steps, k=k, temperature=temperature
            )
        finally:
            self._post_evaluate()
        # All ranks must call broadcast the same number of times (once per key).
        # Non-rank-0 processes return {} from _run_sampled_mode_coverage, so
        # pre-populate the canonical keys with 0.0 on every rank before broadcast.
        canonical_keys = [
            f"eval/{benchmark_name}/sampled_mode_coverage_at_{k}",
            f"eval/{benchmark_name}/sampled_any_correct_at_{k}",
            f"eval/{benchmark_name}/sampled_mean_at_{k}",
            f"eval/{benchmark_name}/sampled_distinct_correct_at_{k}",
        ]
        for key in canonical_keys:
            metrics.setdefault(key, 0.0)
        metrics = self.strategy.broadcast(metrics)
        return metrics

    def _run_sampled_mode_coverage(
        self,
        dataset,
        benchmark_name: str,
        steps: int,
        *,
        k: int,
        temperature: float,
    ) -> dict[str, float]:
        if not self.strategy.is_rank_0():
            return {}

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.eval_dataloader_collate_fn,
        )

        logging.info(
            "Starting sampled mode-coverage eval %s/%s: %s (%s prompts, K=%s, T=%s) at step %s",
            benchmark_name,
            len(self.eval_dataset_dict),
            benchmark_name,
            len(dataset),
            k,
            temperature,
            steps,
        )

        per_prompt_metrics: list[dict[str, float]] = []
        futs: list = []
        pending_refs: list[list[str]] = []

        for i, (batch_formatted, _batch_raw, batch_refs) in enumerate(dataloader):
            actor = self.actors[i % len(self.actors)]
            fut = actor.futures.generate_for_mode_coverage(
                list(batch_formatted), list(batch_refs), k, temperature
            )
            futs.append(fut)
            pending_refs.append(list(batch_refs))
            if len(futs) == len(self.actors) or i == len(dataloader) - 1:
                for fut, refs_batch in zip(futs, pending_refs):
                    result = fut.result()
                    for rewards, answer_keys, ref in zip(
                        result["rewards"], result["answer_keys"], refs_batch
                    ):
                        mode_count = _parse_answer_mode_count(ref)
                        per_prompt_metrics.append(
                            _compute_mode_coverage_metrics(rewards, answer_keys, mode_count)
                        )
                futs.clear()
                pending_refs.clear()

        if not per_prompt_metrics:
            return {}

        mean: dict[str, float] = {
            key: float(sum(m[key] for m in per_prompt_metrics) / len(per_prompt_metrics))
            for key in per_prompt_metrics[0]
        }

        logging.info(
            "Finished sampled mode-coverage eval %s: "
            "mode_cov@%s=%.4f any_correct@%s=%.4f mean@%s=%.4f distinct@%s=%.4f at step %s",
            benchmark_name,
            k,
            mean["mode_coverage_at_k"],
            k,
            mean["any_correct_at_k"],
            k,
            mean["mean_at_k"],
            k,
            mean["distinct_correct_modes_at_k"],
            steps,
        )

        return {
            f"eval/{benchmark_name}/sampled_mode_coverage_at_{k}": mean["mode_coverage_at_k"],
            f"eval/{benchmark_name}/sampled_any_correct_at_{k}": mean["any_correct_at_k"],
            f"eval/{benchmark_name}/sampled_mean_at_{k}": mean["mean_at_k"],
            f"eval/{benchmark_name}/sampled_distinct_correct_at_{k}": mean[
                "distinct_correct_modes_at_k"
            ],
        }
