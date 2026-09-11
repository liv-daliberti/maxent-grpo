"""Falcon chat-surface twins of the frozen Qwen prompt contracts.

The Falcon3 cohort exists to show the ModeBench result is not a Qwen artifact.
That claim only holds if the two families differ in the chat surface and in
nothing else, so these tests pin three things: the Qwen prompts are byte-frozen,
each Falcon twin is a pure surface swap of its Qwen counterpart, and the Falcon
surface reproduces the base model's own published chat template.
"""

from __future__ import annotations

import pathlib

import pytest

from oat_drgrpo.answer_options import format_answer_option_prompt
from oat_drgrpo.canonical_actions import resolve_canonical_action_space
from oat_drgrpo.templates import (
    CANONICAL_TASK_PROMPT_TEMPLATES,
    CHAT_SURFACES,
    PROMPT_TEMPLATE_FAMILY_TWINS,
    TEMPLATE_FACTORY,
    prompt_template_role,
    prompt_template_surface,
    validate_canonical_prompt_materialization,
)

FALCON_SNAPSHOT = pathlib.Path(
    "var/cache/huggingface/transformers/models--tiiuae--Falcon3-1B-Instruct"
    "/snapshots/28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"
)

# Real frozen rows: the three canonical templates reject any other prompt body.
GRAPH_QUESTION = (
    "Color the graph with vertices 1,2,3 and edges (1,2),(2,3) using colors "
    "1,2,3 so that adjacent vertices differ. Give the coloring "
    "inside \\boxed{}."
)
COUNTDOWN_QUESTION = (
    "Using 4, 5, and 6 exactly once with + - * /, make 26. Give "
    "the expression inside \\boxed{}."
)
PANTRY_QUESTION = (
    "Pantry rows: 1 flour; 2 sugar; 3 butter; 4 eggs; 5 milk; 6 salt. "
    "Return only ingredient_id=grams pairs separated by semicolons inside "
    "\\boxed{}. Do not add a recipe name or preparation prose."
)

QUESTION_FOR_TEMPLATE = {
    "qwen_boxed": "What is 2+2?",
    "qwen_math": "What is 2+2?",
    "qwen_math_route": "What is 2+2?",
    "qwen_graph_digits": GRAPH_QUESTION,
    "qwen_countdown_digits": COUNTDOWN_QUESTION,
    "qwen_pantry_support_mask": PANTRY_QUESTION,
}


def _swap_falcon_surface_to_qwen(prompt: str) -> str:
    """Rewrite Falcon role markers back onto the Qwen surface."""

    falcon = CHAT_SURFACES["falcon"]
    qwen = CHAT_SURFACES["qwen"]
    for falcon_marker, qwen_marker in zip(falcon, qwen):
        prompt = prompt.replace(falcon_marker, qwen_marker)
    return prompt


def test_every_qwen_template_has_exactly_one_falcon_twin():
    qwen_templates = {
        name for name in TEMPLATE_FACTORY if prompt_template_surface(name) == "qwen"
    }
    assert set(PROMPT_TEMPLATE_FAMILY_TWINS) == qwen_templates
    for qwen_name, falcon_name in PROMPT_TEMPLATE_FAMILY_TWINS.items():
        assert falcon_name in TEMPLATE_FACTORY
        assert prompt_template_surface(falcon_name) == "falcon"
        # The twin encodes the same training contract, not merely a similar one.
        assert prompt_template_role(falcon_name) == prompt_template_role(qwen_name)


@pytest.mark.parametrize("qwen_name", sorted(QUESTION_FOR_TEMPLATE))
def test_falcon_twin_is_a_pure_surface_swap(qwen_name):
    question = QUESTION_FOR_TEMPLATE[qwen_name]
    falcon_name = PROMPT_TEMPLATE_FAMILY_TWINS[qwen_name]
    qwen_prompt = TEMPLATE_FACTORY[qwen_name](question)
    falcon_prompt = TEMPLATE_FACTORY[falcon_name](question)

    assert falcon_prompt != qwen_prompt
    assert "<|im_start|>" not in falcon_prompt
    assert "<|im_end|>" not in falcon_prompt
    # Every byte outside the role markers is shared, so the instruction text,
    # the canonical answer rewrite, and the user body cannot silently diverge.
    assert _swap_falcon_surface_to_qwen(falcon_prompt) == qwen_prompt


@pytest.mark.parametrize(
    "template_name",
    ["falcon_graph_digits", "falcon_countdown_digits", "falcon_pantry_support_mask"],
)
def test_falcon_canonical_templates_reject_foreign_prompt_bodies(template_name):
    with pytest.raises(ValueError, match=template_name):
        TEMPLATE_FACTORY[template_name]("an unrelated question")


def test_canonical_task_binding_covers_both_surfaces():
    for task, allowed in CANONICAL_TASK_PROMPT_TEMPLATES.items():
        surfaces = {prompt_template_surface(name) for name in allowed}
        assert surfaces == {"qwen", "falcon"}
        roles = {prompt_template_role(name) for name in allowed}
        assert len(roles) == 1, f"{task} must bind exactly one contract role"


def test_materialization_validator_rejects_cross_surface_rows():
    questions = [GRAPH_QUESTION]
    qwen_rows = [TEMPLATE_FACTORY["qwen_graph_digits"](GRAPH_QUESTION)]
    falcon_rows = [TEMPLATE_FACTORY["falcon_graph_digits"](GRAPH_QUESTION)]

    validate_canonical_prompt_materialization(
        "qwen_graph_digits", questions, qwen_rows
    )
    validate_canonical_prompt_materialization(
        "falcon_graph_digits", questions, falcon_rows
    )
    # A Falcon run handed Qwen-rendered rows must fail closed rather than train
    # the policy against role markers its base model never saw.
    with pytest.raises(RuntimeError, match="falcon_graph_digits"):
        validate_canonical_prompt_materialization(
            "falcon_graph_digits", questions, qwen_rows
        )
    with pytest.raises(RuntimeError, match="qwen_graph_digits"):
        validate_canonical_prompt_materialization(
            "qwen_graph_digits", questions, falcon_rows
        )


def test_answer_option_latent_lands_inside_the_falcon_system_turn():
    falcon_prompt = TEMPLATE_FACTORY["falcon_boxed"]("What is 2+2?")
    conditioned = format_answer_option_prompt(falcon_prompt, z=1, num_options=4)

    system_marker, user_marker, _ = CHAT_SURFACES["falcon"]
    latent_at = conditioned.find("Answer-option latent:")
    assert latent_at > 0
    # The latent must sit inside the system turn, not in the dead zone between
    # roles where the model would not read it as an instruction.
    assert latent_at > conditioned.find(system_marker)
    assert latent_at < conditioned.find(user_marker)


def test_run_experiment_dispatches_the_falcon_base_model():
    """A Falcon cohort must be launchable without falling back to `custom`.

    The experiment entrypoint validates OAT_ZERO_MODEL against a closed list and
    exits when it does not match, so an unregistered family name fails every job
    in a cohort within seconds of it starting.
    """

    entrypoint = (
        pathlib.Path(__file__).resolve().parents[1] / "ops/run_experiment.sh"
    ).read_text(encoding="utf-8")

    assert "falcon3-1b-instruct|falcon3-1b|falcon-1b)" in entrypoint
    assert 'DEFAULT_PRETRAIN="tiiuae/Falcon3-1B-Instruct"' in entrypoint
    # The default template for this family must be a Falcon surface, never the
    # Qwen ChatML markers the base model was not tuned on.
    assert 'DEFAULT_PROMPT_TEMPLATE="falcon_math"' in entrypoint
    assert 'MODEL_TAG="falcon3_1b_instruct"' in entrypoint
    assert "falcon3-1b-instruct," in entrypoint


@pytest.mark.skipif(
    not (FALCON_SNAPSHOT / "tokenizer_config.json").exists(),
    reason="Falcon3-1B-Instruct snapshot is not present in the local cache",
)
def test_falcon_surface_matches_the_published_chat_template():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(FALCON_SNAPSHOT))
    system = "Return only the final answer inside \\boxed{}. Do not explain."
    question = "What is 2+2?"

    official = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert TEMPLATE_FACTORY["falcon_boxed"](question) == official


@pytest.mark.skipif(
    not (FALCON_SNAPSHOT / "tokenizer_config.json").exists(),
    reason="Falcon3-1B-Instruct snapshot is not present in the local cache",
)
@pytest.mark.parametrize(
    "task", ["graph_coloring", "countdown", "pantry_support_mask"]
)
def test_canonical_action_space_is_identical_under_the_falcon_tokenizer(task):
    from transformers import AutoTokenizer

    qwen_snapshot = pathlib.Path(
        "var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct"
        "/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    )
    if not (qwen_snapshot / "tokenizer_config.json").exists():
        pytest.skip("Qwen2.5-0.5B-Instruct snapshot is not present")

    falcon_space = resolve_canonical_action_space(
        AutoTokenizer.from_pretrained(str(FALCON_SNAPSHOT)), task
    )
    qwen_space = resolve_canonical_action_space(
        AutoTokenizer.from_pretrained(str(qwen_snapshot)), task
    )
    # Token ids necessarily differ across vocabularies; the action geometry the
    # objective is defined over must not.
    assert falcon_space.horizon == qwen_space.horizon
    assert falcon_space.sequence_count == qwen_space.sequence_count
    assert falcon_space.max_sequence_entropy == pytest.approx(
        qwen_space.max_sequence_entropy
    )
