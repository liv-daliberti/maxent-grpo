from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import train_real_domains_pilot_20260921 as pilot
from oat_drgrpo.online_canonical_bank import VerifiedCanonicalReplayGroup


def config(**updates):
    return {**pilot.DEFAULTS, "group_size": 4, "train_microbatch_size": 1,
            "max_new_tokens": 4, "max_grad_norm": 1000., **updates}


def sample(tokens, accepted, key=None, request="r"):
    return {"token_ids": tokens, "text": str(tokens), "request_id": request,
            "verdict": {"accepted": accepted, "canonical_key": key,
                        "hard_violations": [], "receipt": {}}}


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([
            [.2, -.1, .3], [-.2, .4, .1], [.1, .2, -.3]], dtype=torch.float64))
        self.config = SimpleNamespace(vocab_size=3)

    def forward(self, input_ids, attention_mask):
        return {"logits": self.weight[input_ids]}


def score(model, prompt, response):
    tokens = torch.tensor([list(prompt) + list(response)])
    logits = model(tokens, torch.ones_like(tokens))["logits"]
    selected = logits[:, :-1].log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
    return selected[0, len(prompt)-1:]


@pytest.mark.parametrize("arm", ["maxrl", "remax"])
@pytest.mark.parametrize("micro", [1, 2, 4])
def test_production_gradient_matches_literal_maxrl_and_uniform_replay(arm, micro):
    model = TinyPolicy()
    reference = deepcopy(model)
    prompt = [0]
    rows = [sample([1, 2], True, "a"), sample([2], False),
            sample([1, 1, 2], True, "b"), sample([0, 2], False)]
    group = VerifiedCanonicalReplayGroup(tuple(prompt), ("a", "b"), ((1, 2), (1, 1, 2)), (100, 1))
    c = config(train_microbatch_size=micro)
    optimizer = torch.optim.SGD(model.parameters(), lr=.2)
    learner = pilot.build_learner(model, SimpleNamespace(pad_token_id=2, eos_token_id=2, vocab_size=3), c, arm, optimizer)
    # Literal centered MaxRL, token-level behavior importance, fixed Tmax
    # normalization; replay is uniform despite 100:1 fresh observation counts.
    losses = []
    for advantage, row in zip([1., -1., 1., -1.], rows):
        logps = score(reference, prompt, row["token_ids"])
        ratio = (logps - logps.detach()).exp()
        losses.append((-advantage * ratio).sum() / c["max_new_tokens"])
    loss = torch.stack(losses).mean()
    if arm == "remax":
        replay = -torch.stack([score(reference, prompt, tokens).mean() for tokens in group.response_token_ids]).mean()
        loss = loss + .1 * 3 / 16 * replay
    loss.backward()
    expected = reference.weight.detach() - .2 * reference.weight.grad
    info = pilot.policy_update(learner, prompt, rows, [group])
    assert learner.strategy.updates == 1
    torch.testing.assert_close(model.weight, expected, atol=2e-8, rtol=2e-7)
    assert (info["canonical_replay_applied_score_gradient_l2"] > 0) == (arm == "remax")
    assert info["fresh_tokens"] == 8


@pytest.mark.parametrize("reward", [False, True])
def test_degenerate_fresh_groups_leave_control_unchanged_but_replay_learns(reward):
    rows = [sample([1, 2], reward, "a" if reward else None) for _ in range(4)]
    group = VerifiedCanonicalReplayGroup((0,), ("known",), ((1, 2),), (1,))
    original = TinyPolicy().weight.detach().clone()
    for arm in ("maxrl", "remax"):
        model = TinyPolicy()
        learner = pilot.build_learner(model, SimpleNamespace(pad_token_id=2, eos_token_id=2, vocab_size=3), config(), arm,
                                      torch.optim.SGD(model.parameters(), lr=.2))
        info = pilot.policy_update(learner, [0], rows, [group])
        assert info["mixed_group"] == 0
        assert torch.equal(model.weight, original) == (arm == "maxrl")


def test_bank_stores_first_raw_policy_exemplar_and_uniform_keys():
    bank = pilot.ReplayBank(capacity=2)
    assert bank.entries == {}
    assert bank.observe("p", [0], [sample([1, 2], True, "a", "first"), sample([1, 1, 2], True, "a", "later"), sample([0, 2], False)]) == 1
    assert bank.observe("p", [0], [sample([2], True, "b"), sample([1], True, "c")]) == 1
    _, groups = bank.next_group()
    assert groups[0].outcome_keys == ("a", "b")
    assert groups[0].response_token_ids == ((1, 2), (2,))
    assert groups[0].fresh_observation_counts == (2, 1)
    assert groups[0].mass_weights == ()
    with pytest.raises(ValueError, match="prompt changed"):
        bank.observe("p", [1], [sample([1], True, "a")])


def test_masks_keep_eos_and_exclude_padding_and_prompt_tokens():
    assert pilot.trim_response([7, 9, 9, 9], {9}) == (7, 9)
    ids, attention, masks = pilot.tensor_batch([1, 2], [(7, 9), (9,)], 9, "cpu")
    assert ids.tolist() == [[1, 2, 7, 9], [1, 2, 9, 9]]
    assert attention.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert masks.tolist() == [[0., 1., 1.], [0., 1., 0.]]


def test_reward_without_verified_key_and_hard_violation_fail_closed():
    with pytest.raises(ValueError):
        pilot.validate_verdict({"accepted": True, "canonical_key": None, "hard_violations": []})
    with pytest.raises(RuntimeError):
        pilot.validate_verdict({"accepted": False, "canonical_key": None, "hard_violations": ["checker disagreement"]})


def test_all_prompt_metrics_include_failed_samples():
    rows = [sample([1], True, "a") for _ in range(16)] + [sample([2], True, "b") for _ in range(16)] + [sample([0], False) for _ in range(32)]
    result = pilot.summarize_samples(rows)
    assert result["pass_at_1"] == .5
    assert result["distinct_valid_at_32"] <= 2
    assert result["pcmd"] == pytest.approx(1 - 480 / 992)


def test_config_rejects_split_overlap_and_nonmatching_model_pin():
    raw = {"model": "/cache/" + "a" * 40, "model_revision": "a" * 40,
           "adapter_module": "x", "adapter_config": {}, "train_ids": ["a"], "eval_ids": ["b"]}
    assert pilot.resolve_config(raw)["replay_alpha"] == .1
    with pytest.raises(ValueError, match="disjoint"):
        pilot.resolve_config({**raw, "eval_ids": ["a"]})
    with pytest.raises(ValueError, match="snapshot"):
        pilot.resolve_config({**raw, "model_revision": "b" * 40})


def test_checkpoint_preserves_bank_cursor_and_rejects_wrong_arm_or_corruption(tmp_path):
    class SerializablePolicy(TinyPolicy):
        def save_pretrained(self, path, safe_serialization):
            path.mkdir()
            torch.save(self.state_dict(), path / "adapter.pt")
    model = SerializablePolicy()
    learner = pilot.build_learner(model, SimpleNamespace(pad_token_id=2, eos_token_id=2, vocab_size=3), config(), "remax")
    bank = pilot.ReplayBank()
    bank.observe("train", [0], [sample([1, 2], True, "a")])
    bank.next_group()
    path = tmp_path / "checkpoint-0"
    pilot.save_checkpoint(path, model, learner, bank, 0, "frozen", "remax")
    state, restored = pilot.read_checkpoint(path, "frozen", "remax")
    assert state["completed_updates"] == 0
    assert restored.state() == bank.state()
    with pytest.raises(ValueError, match="arm"):
        pilot.read_checkpoint(path, "frozen", "maxrl")
    (path / "adapter" / "adapter.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="adapter"):
        pilot.read_checkpoint(path, "frozen", "remax")
