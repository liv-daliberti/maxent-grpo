from __future__ import annotations

from types import SimpleNamespace

import pytest

from oat_drgrpo.learner import run as run_module
from oat_drgrpo.learner.run import ZeroMathRunMixin
from oat_drgrpo.math_grader import VerifiedExplorationIdentity
from oat_drgrpo.online_canonical_bank import OnlineCanonicalBank
from oat_drgrpo.verified_route_library import VerifiedRouteLibrary


def _observe(
    library: VerifiedRouteLibrary,
    *,
    prompt: int,
    route: str | None,
    endpoint: str = "endpoint",
    response: int = 20,
    logprob: float = -0.5,
) -> None:
    library.observe_neutral(
        prompt_token_ids=[[prompt]],
        verifier_ids=["mathir_action_menu_v1"],
        endpoint_keys=[endpoint],
        route_signatures=[route],
        response_token_ids=[[response]],
        model_mean_logprobs=[logprob],
        task_verified=[True],
        active_mask=[True],
    )


def test_route_library_requires_cross_prompt_recurrence_and_excludes_current():
    library = VerifiedRouteLibrary(replay_groups_per_step=1)
    _observe(library, prompt=1, route="route-a", response=11)
    assert library.scheduled_cross_prompt_replay_groups([[9]]) == []

    _observe(library, prompt=2, route="route-a", response=22)
    diagnostics = library.diagnostics()
    assert diagnostics.distinct_routes == 1
    assert diagnostics.recurring_routes == 1
    assert diagnostics.cross_prompt_neutral_reproductions == 1
    first_prompt = library.prompt_exemplars([1])
    assert len(first_prompt) == 1
    assert first_prompt[0].route_signature == "route-a"
    assert first_prompt[0].neutral_reproduced
    assert not first_prompt[0].proposal_observed

    group = library.scheduled_cross_prompt_replay_groups([[1]])[0]
    assert group.prompt_token_ids == (2,)
    assert group.outcome_keys == ("route-a",)
    assert group.response_token_ids == ((22,),)
    assert group.prompt_token_ids != (1,)


def test_route_library_proposal_is_support_only_until_neutral_graduation():
    library = VerifiedRouteLibrary(
        replay_groups_per_step=1,
        proposal_max_mean_logprob_drop=1.0,
    )
    _observe(library, prompt=1, route="route-a", response=11)
    rejected = library.admit_proposal(
        prompt_token_ids=[2],
        verifier="mathir_action_menu_v1",
        endpoint_key="endpoint",
        route_signature="route-a",
        response_token_ids=[22],
        proposal_mean_logprob=-2.01,
        anchor_mean_logprob=-1.0,
    )
    assert not rejected.admitted
    assert rejected.rejected_trust

    admitted = library.admit_proposal(
        prompt_token_ids=[2],
        verifier="mathir_action_menu_v1",
        endpoint_key="endpoint",
        route_signature="route-a",
        response_token_ids=[22],
        proposal_mean_logprob=-1.9,
        anchor_mean_logprob=-1.0,
    )
    assert admitted.admitted
    assert library.diagnostics().recurring_routes == 0
    assert library.scheduled_cross_prompt_replay_groups([[9]]) == []

    _observe(
        library,
        prompt=2,
        route="route-a",
        response=23,
        logprob=-0.8,
    )
    diagnostics = library.diagnostics()
    assert diagnostics.proposal_graduations == 1
    assert diagnostics.recurring_routes == 1
    assert diagnostics.proposal_rows_admitted == 1
    assert diagnostics.proposal_rows_rejected_trust == 1


def test_route_library_ignores_wrong_inactive_and_routeless_neutral_rows():
    library = VerifiedRouteLibrary(replay_groups_per_step=1)
    library.observe_neutral(
        prompt_token_ids=[[1], [2], [3]],
        verifier_ids=["v", "v", "v"],
        endpoint_keys=["e", "e", "e"],
        route_signatures=["wrong", "inactive", None],
        response_token_ids=[[4], [5], [6]],
        model_mean_logprobs=[-1.0, -1.0, -1.0],
        task_verified=[False, True, True],
        active_mask=[True, False, True],
    )
    diagnostics = library.diagnostics()
    assert diagnostics.neutral_rows_observed == 3
    assert diagnostics.neutral_routes_observed == 0
    assert diagnostics.distinct_routes == 0


def test_route_library_checkpoint_resume_is_exact_and_configuration_bound():
    library = VerifiedRouteLibrary(
        replay_groups_per_step=1,
        replay_capacity_per_route=3,
        proposal_max_mean_logprob_drop=0.75,
    )
    _observe(library, prompt=1, route="route-a", response=11)
    _observe(library, prompt=2, route="route-a", response=22)
    library.scheduled_cross_prompt_replay_groups([[1]])
    state = library.state_dict()

    restored = VerifiedRouteLibrary(
        replay_groups_per_step=1,
        replay_capacity_per_route=3,
        proposal_max_mean_logprob_drop=0.75,
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.scheduled_cross_prompt_replay_groups([[2]])[0].prompt_token_ids == (
        1,
    )

    mismatch = VerifiedRouteLibrary(
        replay_groups_per_step=2,
        replay_capacity_per_route=3,
        proposal_max_mean_logprob_drop=0.75,
    )
    with pytest.raises(ValueError, match="replay_groups_per_step"):
        mismatch.load_state_dict(state)


def test_route_library_rejects_nonfinite_or_positive_model_logprob():
    library = VerifiedRouteLibrary(replay_groups_per_step=1)
    with pytest.raises(ValueError, match="non-positive"):
        _observe(library, prompt=1, route="route-a", logprob=0.1)
    with pytest.raises(ValueError, match="finite"):
        library.admit_proposal(
            prompt_token_ids=[1],
            verifier="v",
            endpoint_key="e",
            route_signature="r",
            response_token_ids=[2],
            proposal_mean_logprob=float("nan"),
            anchor_mean_logprob=-1.0,
        )


def test_neutral_evidence_evicts_proposal_only_row_at_route_capacity():
    library = VerifiedRouteLibrary(
        replay_groups_per_step=1,
        replay_capacity_per_route=2,
    )
    _observe(library, prompt=1, route="route-a", response=11)
    admitted = library.admit_proposal(
        prompt_token_ids=[2],
        verifier="mathir_action_menu_v1",
        endpoint_key="endpoint",
        route_signature="route-a",
        response_token_ids=[22],
        proposal_mean_logprob=-0.6,
        anchor_mean_logprob=-0.5,
    )
    assert admitted.admitted

    _observe(library, prompt=3, route="route-a", response=33)

    assert library.prompt_exemplars([2]) == ()
    assert library.prompt_exemplars([3])[0].neutral_reproduced
    assert library.diagnostics().recurring_routes == 1


class _RouteActor:
    def __init__(self) -> None:
        self.calls: list[tuple[float, int]] = []

    def step(
        self,
        raw_prompts,
        processed_prompts,
        refs,
        *,
        sampling_temperature,
        sampling_seed,
    ):
        self.calls.append((sampling_temperature, sampling_seed))
        return "route-proposals"


class _RouteIpc:
    def __init__(self, rows) -> None:
        self.rows = rows

    def deserialize_ipc(self, handle):
        assert handle == "route-proposals"
        return self.rows


def _route_row(
    response: str,
    token_id: int,
    *,
    reward: float,
    mean_logprob: float,
):
    return SimpleNamespace(
        prompt_ids=[7, 8],
        response=response,
        response_ids=[token_id],
        response_logprobs=[mean_logprob],
        rewards=[reward],
        loss_mask=True,
    )


def _route_proposal_learner(proposal_rows):
    return SimpleNamespace(
        args=SimpleNamespace(
            seed=43,
            num_samples=2,
            online_evaluation=True,
            online_canonical_key_mode="verified_route",
            online_canonical_counterfactual_anchor_max_tokens=64,
            online_canonical_counterfactual_max_attempts=1,
            online_canonical_counterfactual_sampling_temperature=1.0,
            verified_route_proposal_max_mean_logprob_drop=1.0,
            verifier_version="fast",
        ),
        _online_canonical_bank=OnlineCanonicalBank(
            entropy_alpha=0.0,
            novelty_beta=0.0,
            retain_exemplars=True,
            separate_proposal_objective_support=True,
        ),
        _verified_route_library=VerifiedRouteLibrary(
            replay_groups_per_step=1,
            proposal_max_mean_logprob_drop=1.0,
        ),
        _prompt_batches_consumed_total=1,
        collector=SimpleNamespace(
            ipc_client=_RouteIpc(proposal_rows),
        ),
    )


def test_route_proposal_payload_is_executable_trusted_and_support_only(
    monkeypatch,
):
    identities = {
        "anchor": VerifiedExplorationIdentity("v", "endpoint-a", "route-a"),
        "novel": VerifiedExplorationIdentity("v", "endpoint-b", "route-b"),
        "untrusted": VerifiedExplorationIdentity("v", "endpoint-c", "route-c"),
    }
    monkeypatch.setattr(
        run_module,
        "validated_exploration_identity",
        lambda response, problem, reference, **kwargs: identities.get(response),
    )
    proposals = [
        _route_row("novel", 21, reward=1.0, mean_logprob=-1.0),
        _route_row("untrusted", 22, reward=1.0, mean_logprob=-2.0),
    ]
    learner = _route_proposal_learner(proposals)
    actor = _RouteActor()
    neutral = [
        _route_row("anchor", 11, reward=1.0, mean_logprob=-0.5),
        _route_row("invalid", 12, reward=0.0, mean_logprob=-0.8),
    ]

    payload, metrics = ZeroMathRunMixin._generate_verified_counterfactual_proposals(
        learner,
        actor=actor,
        raw_prompts=["problem"],
        processed_prompts=["formatted"],
        refs=["reference"],
        neutral_feedback=neutral,
    )

    assert payload == {
        "prompt_token_ids": [[7, 8]],
        "outcome_keys": ["route-b"],
        "response_token_ids": [[21]],
        "endpoint_keys": ["endpoint-b"],
        "route_signatures": ["route-b"],
        "verifier_ids": ["v"],
        "proposal_mean_logprobs": [-1.0],
        "anchor_mean_logprobs": [-0.5],
    }
    assert metrics["actor/counterfactual_proposal_trust_checked_rows"] == 2
    assert metrics["actor/counterfactual_proposal_trust_rejected_rows"] == 1
    assert metrics["actor/counterfactual_proposal_temperature_step"] == 0
    assert metrics["actor/counterfactual_proposal_max_admissions_per_update"] == 1
    assert len(actor.calls) == 1
    assert actor.calls[0][0] == 1.0
    assert actor.calls[0][1] >= 0


def test_route_explorer_does_not_run_for_non_singleton_verified_support(
    monkeypatch,
):
    identities = {
        "route-a": VerifiedExplorationIdentity("v", "endpoint", "route-a"),
        "route-b": VerifiedExplorationIdentity("v", "endpoint", "route-b"),
    }
    monkeypatch.setattr(
        run_module,
        "validated_exploration_identity",
        lambda response, problem, reference, **kwargs: identities.get(response),
    )
    learner = _route_proposal_learner(
        [
            _route_row("route-c", 21, reward=1.0, mean_logprob=-1.0),
            _route_row("invalid", 22, reward=0.0, mean_logprob=-1.0),
        ]
    )
    actor = _RouteActor()
    neutral = [
        _route_row("route-a", 11, reward=1.0, mean_logprob=-0.5),
        _route_row("route-b", 12, reward=1.0, mean_logprob=-0.6),
    ]

    payload, _metrics = ZeroMathRunMixin._generate_verified_counterfactual_proposals(
        learner,
        actor=actor,
        raw_prompts=["problem"],
        processed_prompts=["formatted"],
        refs=["reference"],
        neutral_feedback=neutral,
    )

    assert payload["outcome_keys"] == []
    assert actor.calls == []


def test_replicated_route_admission_never_changes_neutral_objective_support(
    monkeypatch,
):
    neutral_rows = [
        _route_row("anchor", 11, reward=1.0, mean_logprob=-0.5),
        _route_row("invalid", 12, reward=0.0, mean_logprob=-0.8),
    ]

    class Actor:
        def step(self, *args, **kwargs):
            assert "sampling_seed" in kwargs
            return "neutral"

    class Ipc:
        def deserialize_ipc(self, handle):
            assert handle == "neutral"
            return neutral_rows

    class Collector:
        ipc_client = Ipc()

        @staticmethod
        def get_metrics(elapsed, feedback):
            assert elapsed >= 0
            assert feedback is neutral_rows
            return {}

    monkeypatch.setattr(run_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(run_module.dist, "get_rank", lambda: 0)

    def gather(output, value):
        output[0] = value

    monkeypatch.setattr(run_module.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        run_module.dist,
        "broadcast_object_list",
        lambda payload, src: None,
    )

    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        novelty_beta=0.0,
        retain_exemplars=True,
        separate_proposal_objective_support=True,
    )
    route_library = VerifiedRouteLibrary(replay_groups_per_step=1)
    proposal_payload = {
        "prompt_token_ids": [[7, 8]],
        "outcome_keys": ["route-b"],
        "response_token_ids": [[21]],
        "endpoint_keys": ["endpoint-b"],
        "route_signatures": ["route-b"],
        "verifier_ids": ["v"],
        "proposal_mean_logprobs": [-1.0],
        "anchor_mean_logprobs": [-0.5],
    }
    learner = SimpleNamespace(
        args=SimpleNamespace(
            replicated_freeform_sampling=True,
            canonical_action_task="none",
            canonical_graph_actions=False,
            rollout_batch_size=1,
            num_samples=2,
            train_batch_size=2,
            train_batch_size_per_device=2,
            online_evaluation=True,
            online_canonical_counterfactual_proposals=True,
            online_canonical_key_mode="verified_route",
            seed=43,
        ),
        update_interval=1,
        strategy=SimpleNamespace(grad_acc_step=1),
        actors=[Actor()],
        collector=Collector(),
        _prompt_batches_consumed_total=1,
        _online_canonical_bank=bank,
        _verified_route_library=route_library,
        _generate_verified_counterfactual_proposals=(
            lambda **kwargs: (proposal_payload, {})
        ),
    )

    feedback, metrics = ZeroMathRunMixin._sample_replicated_freeform_feedback(
        learner,
        ["problem"],
        ["formatted"],
        ["reference"],
    )

    assert feedback is neutral_rows
    assert bank.tracked_outcome_count == 0
    assert route_library.diagnostics().proposal_rows_admitted == 1
    exemplar = route_library.prompt_exemplars([7, 8])[0]
    assert exemplar.proposal_observed
    assert not exemplar.neutral_reproduced
    assert metrics["actor/counterfactual_proposal_objective_outcome_delta"] == 0
    assert metrics["actor/counterfactual_proposal_route_admitted"] == 1
    assert metrics["actor/counterfactual_proposal_conditioned_rows_sent_to_ppo"] == 0
