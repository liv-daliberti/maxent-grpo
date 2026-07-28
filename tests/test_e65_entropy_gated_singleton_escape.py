from types import SimpleNamespace

from oat_drgrpo.learner import run as run_module
from oat_drgrpo.learner.run import ZeroMathRunMixin
from oat_drgrpo.online_canonical_bank import OnlineCanonicalBank
from oat_drgrpo.semantic_shannon import SemanticShannonTracker


class _Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode("utf-8"))

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return bytes(token_ids).decode("utf-8")


def _row(response, *, prompt_ids=(1, 2), reward=1.0):
    return SimpleNamespace(
        response=response,
        response_ids=list(response.encode("utf-8")),
        rewards=[reward],
        loss_mask=True,
        prompt_ids=list(prompt_ids),
    )


def _learner(tracker):
    return SimpleNamespace(
        args=SimpleNamespace(
            seed=43,
            num_samples=2,
            online_evaluation=True,
            online_canonical_counterfactual_singleton_entropy_gate=True,
            online_canonical_counterfactual_max_attempts=3,
            online_canonical_counterfactual_sampling_temperature=1.0,
            generate_max_length=256,
        ),
        _online_canonical_bank=OnlineCanonicalBank(
            entropy_alpha=0.0,
            novelty_beta=0.0,
            retain_exemplars=True,
            replay_capacity=16,
        ),
        _semantic_shannon_tracker=tracker,
        _prompt_batches_consumed_total=2,
        tokenizer=_Tokenizer(),
        collector=SimpleNamespace(),
    )


def _tracker(*, collapsed):
    tracker = SemanticShannonTracker(
        coefficient=0.1,
        success_conditioned_signed_advantage=True,
        open_set_inverse_adaptation=True,
        open_set_warmup_steps=1,
    )
    controller = tracker._open_set_controller
    assert controller is not None
    controller.observe(0.8)
    if collapsed:
        controller.observe(0.4)
    return tracker


def test_entropy_gate_stays_closed_before_self_relative_collapse(monkeypatch):
    monkeypatch.setattr(
        run_module,
        "validated_modebench_outcome_key",
        lambda response, _ref: {"anchor": "a"}.get(response),
    )
    learner = _learner(_tracker(collapsed=False))
    payload, metrics = (
        ZeroMathRunMixin._generate_verified_counterfactual_proposals(
            learner,
            actor=SimpleNamespace(),
            raw_prompts=["raw"],
            processed_prompts=["processed"],
            refs=["ref"],
            neutral_feedback=[_row("anchor"), _row("invalid", reward=0.0)],
        )
    )
    assert payload == {
        "prompt_token_ids": [],
        "outcome_keys": [],
        "response_token_ids": [],
    }
    assert (
        metrics[
            "actor/counterfactual_proposal_singleton_entropy_gate_active"
        ]
        == 0.0
    )
    assert metrics["actor/counterfactual_proposal_rows_generated"] == 0.0


def test_entropy_gate_admits_only_one_mode_then_becomes_ineligible(monkeypatch):
    keys = {"anchor": "a", "alternate-1": "b", "alternate-2": "c"}
    monkeypatch.setattr(
        run_module,
        "validated_modebench_outcome_key",
        lambda response, _ref: keys.get(response),
    )
    monkeypatch.setattr(
        run_module,
        "derive_validator_preserving_counterfactuals",
        lambda _surface, _ref: ("alternate-2", "alternate-1"),
    )
    learner = _learner(_tracker(collapsed=True))
    neutral = [_row("anchor"), _row("invalid", reward=0.0)]
    payload, metrics = (
        ZeroMathRunMixin._generate_verified_counterfactual_proposals(
            learner,
            actor=SimpleNamespace(),
            raw_prompts=["raw"],
            processed_prompts=["processed"],
            refs=["ref"],
            neutral_feedback=neutral,
        )
    )
    assert payload["outcome_keys"] == ["b"]
    assert len(payload["response_token_ids"]) == 1
    assert (
        metrics[
            "actor/counterfactual_proposal_singleton_entropy_gate_active"
        ]
        == 1.0
    )
    learner._online_canonical_bank.admit_verified_proposals(**payload)

    second_payload, second_metrics = (
        ZeroMathRunMixin._generate_verified_counterfactual_proposals(
            learner,
            actor=SimpleNamespace(),
            raw_prompts=["raw"],
            processed_prompts=["processed"],
            refs=["ref"],
            neutral_feedback=neutral,
        )
    )
    assert second_payload["outcome_keys"] == []
    assert (
        second_metrics[
            "actor/counterfactual_proposal_singleton_entropy_gate_known_support"
        ]
        == 2.0
    )
    assert second_metrics["actor/counterfactual_proposal_rows_generated"] == 0.0
