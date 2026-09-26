from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest
import torch
from oat.utils.ops import masked_sum

from oat_drgrpo.learner.grpo import ZeroMathGrpoMixin
from oat_drgrpo.online_canonical_bank import VerifiedCanonicalReplayGroup
from oat_drgrpo.semantic_shannon import (
    SemanticShannonTracker,
    add_semantic_shannon_separate_advantage,
)


class _ThreeModePolicy(torch.nn.Module):
    """One shared categorical policy for common, rare, and failed modes."""

    def __init__(self) -> None:
        super().__init__()
        self.mode_logits = torch.nn.Parameter(torch.zeros(3))

    def forward(self, input_ids, *, attention_mask):
        del attention_mask
        return {
            "logits": self.mode_logits.view(1, 1, 3).expand(
                input_ids.size(0),
                input_ids.size(1),
                3,
            )
        }


class _CpuStrategy:
    def __init__(self, *, grad_acc_step: int = 16) -> None:
        self.grad_acc_step = int(grad_acc_step)
        self.gradient_before_step: torch.Tensor | None = None
        self.optimizer_calls = 0

    def backward(self, loss, model, optimizer):
        del model, optimizer
        # Match DeepSpeed's division across the registered 16 one-row
        # microbatches.
        (loss / self.grad_acc_step).backward()

    def get_gradient_norm(self, model):
        gradient = model.mode_logits.grad
        assert gradient is not None
        self.gradient_before_step = gradient.detach().clone()
        return float(torch.linalg.vector_norm(gradient))

    def optimizer_step(self, optimizer, model, scheduler):
        del model, scheduler
        self.optimizer_calls += 1
        if self.optimizer_calls % self.grad_acc_step == 0:
            optimizer.step()

    def is_rank_0(self):
        return False


class _CpuProductionLearner(ZeroMathGrpoMixin):
    def __init__(self, *, train_batch_size_per_device: int = 1) -> None:
        if 16 % int(train_batch_size_per_device) != 0:
            raise ValueError("test microbatch size must divide the 16-row group")
        self.args = SimpleNamespace(
            beta=0.0,
            canonical_action_task="none",
            canonical_graph_actions=False,
            canonical_graph_learner_sampling=False,
            cliprange=0.2,
            critic_type="drgrpo",
            generate_max_length=1,
            maxent_alpha=0.0,
            maxent_objective="sequence",
            num_ppo_epochs=1,
            num_samples=16,
            online_canonical_replay_alpha=0.1,
            online_canonical_replay_bank_normalized=False,
            online_canonical_replay_compute_only=False,
            online_canonical_replay_mass_alpha=0.1,
            online_canonical_replay_objective="verified_likelihood_per_rollout",
            online_canonical_replay_retention_safe_balance=False,
            policy_entropy_coef=0.0,
            reinforce_update=False,
            replicated_freeform_sampling=False,
            seed=43,
            temperature=1.0,
            train_batch_size=5,
            train_batch_size_per_device=int(train_batch_size_per_device),
        )
        self.model = _ThreeModePolicy()
        self.strategy = _CpuStrategy(
            grad_acc_step=16 // int(train_batch_size_per_device)
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.5)
        self.scheduler = None
        self.tokenizer = SimpleNamespace(pad_token_id=2, eos_token_id=2)
        self.masked_aggregator = functools.partial(
            masked_sum,
            constant_normalizer=self.args.generate_max_length,
        )
        self._baseline_grad_norm_logging_disabled_warned = False

    def _resolve_scoring_vocab_upper_bound(self, model):
        del model
        return 3

    def _mask_invalid_scoring_logit_columns(self, logits, **kwargs):
        del kwargs
        return logits

    def _sanitize_scoring_token_ids(self, input_ids, **kwargs):
        del kwargs
        return input_ids

    def _policy_logps_and_optional_entropy(
        self,
        logits,
        labels,
        response_masks,
        *,
        need_entropy,
    ):
        token_logits = logits[:, :-1, :]
        token_labels = labels[:, 1:]
        log_distribution = torch.log_softmax(token_logits, dim=-1)
        selected = log_distribution.gather(
            dim=-1,
            index=token_labels.unsqueeze(-1),
        ).squeeze(-1)
        selected = selected * response_masks
        entropy = None
        if need_entropy:
            probabilities = log_distribution.exp()
            entropy = -(probabilities * log_distribution).sum(dim=-1)
        return selected, entropy


def test_v6_advantage_drives_the_exact_production_loss_in_theory_direction():
    """Rare verified likelihood rises; common falls; failure has zero gradient."""

    tracker = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_group_centered_advantage=True,
    )
    semantic_advantages, diagnostics = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[[17, 23]] * 16,
            answer_keys=["common"] * 14 + ["rare", None],
            task_rewards=[1.0] * 15 + [0.0],
            active_mask=[1.0] * 16,
            num_samples=16,
        )
    )
    assert semantic_advantages[:14] == pytest.approx(
        [semantic_advantages[0]] * 14
    )
    assert semantic_advantages[0] < 0.0
    assert semantic_advantages[14] > 0.0
    assert semantic_advantages[15] == 0.0
    assert sum(semantic_advantages) == pytest.approx(0.0, abs=1e-14)
    assert diagnostics.eligible_fraction == pytest.approx(15 / 16)

    learner = _CpuProductionLearner()
    # The next-token ids are the three policy modes above. The failed row is
    # present in the optimizer batch, but its detached v6 advantage is zero.
    input_ids = torch.tensor(
        [[2, 0]] * 14 + [[2, 1], [2, 2]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    response_masks = torch.ones((16, 1), dtype=torch.float32)
    with torch.no_grad():
        initial_logits = learner.model.mode_logits.detach().clone()
        initial_output = learner.model(input_ids, attention_mask=attention_mask)
        old_logps, _ = learner._policy_logps_and_optional_entropy(
            initial_output["logits"],
            input_ids,
            response_masks,
            need_entropy=False,
        )
    combined_advantages = add_semantic_shannon_separate_advantage(
        torch.zeros((16, 1)),
        torch.tensor(semantic_advantages).reshape(16, 1),
    )

    learner._baseline_update_with_precomputed_advantages(
        input_ids=input_ids,
        att_mask=attention_mask,
        prompt_id_lens=[1] * 16,
        loss_masks=torch.ones(16),
        response_masks=response_masks,
        logps=old_logps,
        ref_logps=None,
        advantages=combined_advantages,
        final_rewards=torch.tensor([[1.0]] * 15 + [[0.0]]),
        policy_vocab_upper_bound=3,
    )

    gradient = learner.strategy.gradient_before_step
    assert gradient is not None
    # Gradient descent moves opposite these signs.
    assert gradient[0] > 0.0
    assert gradient[1] < 0.0
    assert gradient[2].item() == pytest.approx(0.0, abs=1e-9)
    assert gradient.sum().item() == pytest.approx(0.0, abs=1e-9)

    updated_logits = learner.model.mode_logits.detach()
    assert updated_logits[0] < initial_logits[0]
    assert updated_logits[1] > initial_logits[1]
    assert updated_logits[2].item() == pytest.approx(
        initial_logits[2].item(),
        abs=1e-9,
    )


def test_replay_and_v6_jointly_retain_verified_mass_and_favor_the_rare_mode():
    """Replay raises verified mass while v6 redistributes it toward rarity."""

    tracker = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_group_centered_advantage=True,
    )
    semantic_values, _ = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[[17, 23]] * 16,
            answer_keys=["common"] * 14 + ["rare", None],
            task_rewards=[1.0] * 15 + [0.0],
            active_mask=[1.0] * 16,
            num_samples=16,
        )
    )
    input_ids = torch.tensor(
        [[2, 0]] * 14 + [[2, 1], [2, 2]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    response_masks = torch.ones((16, 1), dtype=torch.float32)
    advantages = add_semantic_shannon_separate_advantage(
        torch.zeros((16, 1)),
        torch.tensor(semantic_values).reshape(16, 1),
    )

    def update(*, replay: bool):
        learner = _CpuProductionLearner()
        with torch.no_grad():
            output = learner.model(input_ids, attention_mask=attention_mask)
            old_logps, _ = learner._policy_logps_and_optional_entropy(
                output["logits"],
                input_ids,
                response_masks,
                need_entropy=False,
            )
        replay_groups = None
        if replay:
            replay_groups = [
                VerifiedCanonicalReplayGroup(
                    prompt_token_ids=(2,),
                    outcome_keys=("common", "rare"),
                    response_token_ids=((0,), (1,)),
                    mass_weights=(1.0, 1.0),
                )
            ]
        infos = learner._baseline_update_with_precomputed_advantages(
            input_ids=input_ids,
            att_mask=attention_mask,
            prompt_id_lens=[1] * 16,
            loss_masks=torch.ones(16),
            response_masks=response_masks,
            logps=old_logps,
            ref_logps=None,
            advantages=advantages,
            final_rewards=torch.tensor([[1.0]] * 15 + [[0.0]]),
            policy_vocab_upper_bound=3,
            canonical_replay_groups=replay_groups,
        )
        gradient = learner.strategy.gradient_before_step
        assert gradient is not None
        return learner.model.mode_logits.detach(), gradient, infos

    semantic_logits, semantic_gradient, _ = update(replay=False)
    combined_logits, combined_gradient, infos = update(replay=True)
    replay_gradient = combined_gradient - semantic_gradient

    # Uniform verified-likelihood replay raises both discovered modes equally
    # against the failed mode; it does not alter their relative semantic tilt.
    assert replay_gradient[0] < 0.0
    assert replay_gradient[1] < 0.0
    assert replay_gradient[0].item() == pytest.approx(
        replay_gradient[1].item(),
        abs=1e-8,
    )
    assert replay_gradient[2] > 0.0
    assert replay_gradient.sum().item() == pytest.approx(0.0, abs=1e-8)

    semantic_gap = semantic_logits[1] - semantic_logits[0]
    combined_gap = combined_logits[1] - combined_logits[0]
    assert combined_gap.item() == pytest.approx(semantic_gap.item(), abs=1e-8)
    assert combined_gap > 0.0

    combined_probabilities = torch.softmax(combined_logits, dim=0)
    assert combined_probabilities[:2].sum() > (2 / 3)
    assert combined_probabilities[1] > combined_probabilities[0]
    assert combined_probabilities[2] < (1 / 3)
    assert infos["canonical_replay_verified_likelihood_active"].item() == 1.0
    assert infos["canonical_replay_applied_score_gradient_l2"].item() > 0.0


def test_joint_update_is_invariant_to_e105_microbatch_geometry():
    """One-row and four-row E105 layouts apply the same combined gradient."""

    tracker = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_group_centered_advantage=True,
    )
    semantic_values, _ = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[[17, 23]] * 16,
            answer_keys=["common"] * 14 + ["rare", None],
            task_rewards=[1.0] * 15 + [0.0],
            active_mask=[1.0] * 16,
            num_samples=16,
        )
    )
    input_ids = torch.tensor(
        [[2, 0]] * 14 + [[2, 1], [2, 2]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    response_masks = torch.ones((16, 1), dtype=torch.float32)
    advantages = add_semantic_shannon_separate_advantage(
        torch.zeros((16, 1)),
        torch.tensor(semantic_values).reshape(16, 1),
    )
    replay_groups = [
        VerifiedCanonicalReplayGroup(
            prompt_token_ids=(2,),
            outcome_keys=("common", "rare"),
            response_token_ids=((0,), (1,)),
            mass_weights=(1.0, 1.0),
        )
    ]

    def update(microbatch_size: int):
        learner = _CpuProductionLearner(
            train_batch_size_per_device=microbatch_size
        )
        with torch.no_grad():
            output = learner.model(input_ids, attention_mask=attention_mask)
            old_logps, _ = learner._policy_logps_and_optional_entropy(
                output["logits"],
                input_ids,
                response_masks,
                need_entropy=False,
            )
        infos = learner._baseline_update_with_precomputed_advantages(
            input_ids=input_ids,
            att_mask=attention_mask,
            prompt_id_lens=[1] * 16,
            loss_masks=torch.ones(16),
            response_masks=response_masks,
            logps=old_logps,
            ref_logps=None,
            advantages=advantages,
            final_rewards=torch.tensor([[1.0]] * 15 + [[0.0]]),
            policy_vocab_upper_bound=3,
            canonical_replay_groups=replay_groups,
        )
        gradient = learner.strategy.gradient_before_step
        assert gradient is not None
        return learner.model.mode_logits.detach(), gradient, infos

    one_row_logits, one_row_gradient, one_row_infos = update(1)
    four_row_logits, four_row_gradient, four_row_infos = update(4)

    assert four_row_gradient.tolist() == pytest.approx(
        one_row_gradient.tolist(), abs=1e-8
    )
    assert four_row_logits.tolist() == pytest.approx(
        one_row_logits.tolist(), abs=1e-8
    )
    assert one_row_infos["canonical_replay_backward_scale"].item() == 16.0
    assert four_row_infos["canonical_replay_backward_scale"].item() == 4.0
    assert one_row_infos["canonical_replay_chunk_size"].item() == 1.0
    assert four_row_infos["canonical_replay_chunk_size"].item() == 4.0


def test_v7_sparse_singleton_and_replay_jointly_raise_verified_mass():
    """A rare singleton stays live and ReplayDr protects both banked modes."""

    tracker = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_verified_support_advantage=True,
    )
    tracker.score_success_conditioned_signed_advantages_and_update(
        prompt_token_ids=[[17, 23]] * 16,
        answer_keys=["common"] * 12 + ["rare"] * 4,
        task_rewards=[1.0] * 16,
        active_mask=[1.0] * 16,
        num_samples=16,
    )
    semantic_values, diagnostics = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[[17, 23]] * 16,
            answer_keys=["rare"] + [None] * 15,
            task_rewards=[1.0] + [0.0] * 15,
            active_mask=[1.0] * 16,
            num_samples=16,
        )
    )
    assert semantic_values[0] > 0.0
    assert semantic_values[1:] == [0.0] * 15
    assert diagnostics.effective_advantage_rms > 0.0

    learner = _CpuProductionLearner()
    input_ids = torch.tensor(
        [[2, 1]] + [[2, 2]] * 15,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    response_masks = torch.ones((16, 1), dtype=torch.float32)
    with torch.no_grad():
        initial_logits = learner.model.mode_logits.detach().clone()
        output = learner.model(input_ids, attention_mask=attention_mask)
        old_logps, _ = learner._policy_logps_and_optional_entropy(
            output["logits"],
            input_ids,
            response_masks,
            need_entropy=False,
        )
    advantages = add_semantic_shannon_separate_advantage(
        torch.zeros((16, 1)),
        torch.tensor(semantic_values).reshape(16, 1),
    )
    replay_groups = [
        VerifiedCanonicalReplayGroup(
            prompt_token_ids=(2,),
            outcome_keys=("common", "rare"),
            response_token_ids=((0,), (1,)),
            mass_weights=(1.0, 1.0),
        )
    ]

    infos = learner._baseline_update_with_precomputed_advantages(
        input_ids=input_ids,
        att_mask=attention_mask,
        prompt_id_lens=[1] * 16,
        loss_masks=torch.ones(16),
        response_masks=response_masks,
        logps=old_logps,
        ref_logps=None,
        advantages=advantages,
        final_rewards=torch.tensor([[1.0]] + [[0.0]] * 15),
        policy_vocab_upper_bound=3,
        canonical_replay_groups=replay_groups,
    )

    updated_logits = learner.model.mode_logits.detach()
    assert updated_logits[0] > initial_logits[0]
    assert updated_logits[1] > updated_logits[0]
    assert updated_logits[2] < initial_logits[2]
    probabilities = torch.softmax(updated_logits, dim=0)
    assert probabilities[:2].sum() > (2 / 3)
    assert probabilities[1] > probabilities[0]
    assert infos["canonical_replay_applied_score_gradient_l2"].item() > 0.0



def test_v7_discovered_unsampled_mode_is_lifted_by_uniform_replaydr():
    """Support membership actuates v7; ReplayDr supplies the rare-mode row."""

    tracker = SemanticShannonTracker(
        coefficient=0.1,
        surprisal_clip=5.0,
        pseudocount=1.0,
        success_conditioned_verified_support_advantage=True,
    )
    tracker.score_success_conditioned_signed_advantages_and_update(
        prompt_token_ids=[[17, 23]] * 16,
        answer_keys=["common"] * 16,
        task_rewards=[1.0] * 16,
        active_mask=[1.0] * 16,
        num_samples=16,
    )
    semantic_values, diagnostics = (
        tracker.score_success_conditioned_signed_advantages_and_update(
            prompt_token_ids=[[17, 23]] * 16,
            answer_keys=["common"] + [None] * 15,
            task_rewards=[1.0] + [0.0] * 15,
            active_mask=[1.0] * 16,
            num_samples=16,
            verified_support_keys_by_group=[("common", "proposal_rare")],
        )
    )
    assert semantic_values[0] < 0.0
    assert semantic_values[1:] == [0.0] * 15
    assert diagnostics.verified_support_size_mean == pytest.approx(2.0)
    counts = next(iter(tracker.state_dict()["counts"].values()))
    assert counts == {"common": 17}
    assert "proposal_rare" not in counts

    learner = _CpuProductionLearner()
    input_ids = torch.tensor(
        [[2, 0]] + [[2, 2]] * 15,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    response_masks = torch.ones((16, 1), dtype=torch.float32)
    with torch.no_grad():
        initial_logits = learner.model.mode_logits.detach().clone()
        output = learner.model(input_ids, attention_mask=attention_mask)
        old_logps, _ = learner._policy_logps_and_optional_entropy(
            output["logits"],
            input_ids,
            response_masks,
            need_entropy=False,
        )
    advantages = add_semantic_shannon_separate_advantage(
        torch.zeros((16, 1)),
        torch.tensor(semantic_values).reshape(16, 1),
    )
    replay_groups = [
        VerifiedCanonicalReplayGroup(
            prompt_token_ids=(2,),
            outcome_keys=("common", "proposal_rare"),
            response_token_ids=((0,), (1,)),
            mass_weights=(1.0, 1.0),
        )
    ]

    infos = learner._baseline_update_with_precomputed_advantages(
        input_ids=input_ids,
        att_mask=attention_mask,
        prompt_id_lens=[1] * 16,
        loss_masks=torch.ones(16),
        response_masks=response_masks,
        logps=old_logps,
        ref_logps=None,
        advantages=advantages,
        final_rewards=torch.tensor([[1.0]] + [[0.0]] * 15),
        policy_vocab_upper_bound=3,
        canonical_replay_groups=replay_groups,
    )

    updated_logits = learner.model.mode_logits.detach()
    assert updated_logits[0] > initial_logits[0]
    assert updated_logits[1] > updated_logits[0]
    assert updated_logits[2] < initial_logits[2]
    assert infos["canonical_replay_mass_weight_min"].item() == pytest.approx(1.0)
    assert infos["canonical_replay_mass_weight_max"].item() == pytest.approx(1.0)
    assert infos["canonical_replay_applied_score_gradient_l2"].item() > 0.0
