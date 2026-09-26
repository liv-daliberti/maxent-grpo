from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from datasets import load_from_disk

from oat_drgrpo.modebench_guided import countdown_legal_regex, legal_regex
from oat_drgrpo.templates import TEMPLATE_FACTORY


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e119_level2_qwen05b_factorial.py"
PROTOCOL = ROOT / "paper/preregistration/e119_level2_qwen05b_factorial_20260901.md"


def load_launcher():
    spec = importlib.util.spec_from_file_location("e119_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_e119_is_complete_five_domain_four_arm_five_seed_factorial():
    e119 = load_launcher()
    assert len(e119.DOMAINS) == 5
    assert e119.ARMS == ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
    assert e119.SEEDS == (43, 44, 45, 46, 47)
    assert e119.PASSES == 8 and e119.TRAIN_ROWS == 384
    assert e119.EVAL_ROWS == 128 and e119.TARGET_STEPS == 3072
    assert e119.CHECKPOINT_INTERVAL == 192


def test_e119_objectives_are_exact_two_by_two_factorial():
    e119 = load_launcher()
    cells = {arm: e119.objective(arm) for arm in e119.ARMS}
    for arm, objective in cells.items():
        assert objective["OAT_ZERO_MAXRL_TASK_OBJECTIVE"] == ("1" if "maxrl" in arm else "0")
        assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == ("0" if arm.startswith("replay_") else "1")
        assert objective["OAT_ZERO_CRITIC_TYPE"] == "drgrpo"
        assert objective["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
        assert objective["OAT_ZERO_MAXENT_ALPHA"] == "0.0"
        assert objective["OAT_ZERO_DAPO_ENABLED"] == "0"
        assert objective["OAT_ZERO_RLEP_REPLAY_COUNT"] == "0"
    ignored = {"OAT_ZERO_VARIANT", "OAT_ZERO_MAXRL_TASK_OBJECTIVE", "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"}
    assert len({tuple(sorted((k, v) for k, v in row.items() if k not in ignored)) for row in cells.values()}) == 1


def test_e119_uses_only_admitted_r5_train_and_eval_splits():
    e119 = load_launcher(); data = ROOT / e119.DATA_ROOT
    assert e119.digest(data / "identity.json") == e119.IDENTITY_SHA256
    for report in (ROOT / e119.BASELINE_REPORT, ROOT / e119.REPEAT_REPORT):
        payload = json.loads(report.read_text())
        assert payload["status"] == "pass"
        assert set(payload["domain_decisions"].values()) == {"admit"}
    for domain in e119.DOMAINS:
        directory = data / e119.DOMAIN_DIR[domain]
        assert len(load_from_disk(str(directory / "train"))["train"]) == 384
        assert len(load_from_disk(str(directory / "eval"))["multi_answer"]) == 128


def test_e119_prompt_and_target_blind_syntax_match_admission():
    e119 = load_launcher()
    for domain, template in e119.PROMPTS.items():
        rendered = TEMPLATE_FACTORY[template]("ROW_SENTINEL")
        assert "ROW_SENTINEL" in rendered and "\\boxed{}" in rendered
        assert "\b" not in rendered
        assert e119.SYNTAX[domain] == ("none" if domain == "graph_coloring" else "countdown_legal_v3" if domain == "countdown" else "domain_legal_v1")
    first = json.dumps({"numbers": [1, 2, 3, 4], "target": 24})
    other = json.dumps({"numbers": [1, 2, 3, 4], "target": 999999})
    assert countdown_legal_regex(first) == countdown_legal_regex(other)
    evaluator_spec = importlib.util.spec_from_file_location(
        "level2_viability", ROOT / "ops/evaluate_modebench_level2_viability.py"
    )
    assert evaluator_spec is not None and evaluator_spec.loader is not None
    evaluator = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator)
    row = {"answer": first}
    grammar = re.compile(countdown_legal_regex(first))
    choices = evaluator.countdown_legal_choices(row)
    assert len(choices) == 7680 and all(grammar.fullmatch(choice) for choice in choices)
    assert countdown_legal_regex(first) == evaluator.countdown_legal_regex(row)
    assert legal_regex("domain_legal_v1", "mathir", "{}") == r"\\boxed\{[A-F](?:;[A-F]){0,3}\}"


def test_e119_protocol_freezes_all_100_new_cells_before_training():
    text = PROTOCOL.read_text()
    for literal in ("before any E119 training submission", "100 new runs", "3,072 optimizer updates", "0, 0.5, 1.0, ..., 8.0", "do not select checkpoints"):
        assert literal in text
