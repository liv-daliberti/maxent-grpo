#!/usr/bin/env python3
"""Render paired, validator-checked ModeBench examples for Section 3.1."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from oat_drgrpo.pantry_plan import validate_pantry_plan  # noqa: E402
OUT = ROOT / "paper/figures/modebench_examples"
OUT_PANTRY = ROOT / "paper/figures/modebench_pantry_example"
DATA_ROOTS = {
    "graph": ROOT / "var/data/exact_answer_mode_probe/eval",
    "countdown": ROOT / "var/data/exact_countdown_easy3_probe/eval",
    "python": ROOT / "var/data/python_factor_modebench_v1/eval",
    "mathir": ROOT / "var/data/mathir_action_menu_v1/eval",
    "pantry": ROOT / "var/data/pantry_plan_modebench_v2/eval",
}

INK = "#19324A"
MUTED = "#607487"
GRID = "#D8E2EA"
PANEL = "#F6F9FB"
WHITE = "#FFFFFF"
PURPLE = "#6C5CE7"
ORANGE = "#D95F45"
BLUE = "#3A7CA5"
GREEN = "#2A9D8F"
NODE_COLORS = {1: "#F2C14E", 2: "#2A9D8F", 3: "#7B6FD0"}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


def box(ax, x, y, w, h, *, face=WHITE, edge=GRID, radius=0.012, lw=0.9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, **kwargs):
    defaults = {
        "transform": ax.transAxes,
        "fontsize": 7.5,
        "color": INK,
        "ha": "center",
        "va": "center",
    }
    defaults.update(kwargs)
    return ax.text(x, y, value, **defaults)


def load_and_validate() -> dict[str, dict]:
    """Freeze two distinct verified answers for each depicted released row."""
    for root in DATA_ROOTS.values():
        assert root.is_dir(), root

    graph = {
        "instance_id": "eval_multi_answer-10000-0",
        "n": 6,
        "edges": [[1, 2], [1, 3], [3, 5], [3, 6], [4, 6], [5, 6]],
        "partial_colors": [2, None, None, 1, None, 2],
        "num_completions": 4,
    }
    graph_modes = ((2, 1, 1, 1, 3, 2), (2, 1, 3, 1, 1, 2))
    valid_graph_modes = [
        colors
        for colors in itertools.product((1, 2, 3), repeat=graph["n"])
        if all(
            fixed is None or colors[index] == fixed
            for index, fixed in enumerate(graph["partial_colors"])
        )
        and all(colors[u - 1] != colors[v - 1] for u, v in graph["edges"])
    ]
    assert len(valid_graph_modes) == graph["num_completions"] == 4
    assert all(mode in valid_graph_modes for mode in graph_modes)
    graph_answers = tuple("".join(str(mode[i]) for i in (1, 2, 4)) for mode in graph_modes)
    assert graph_answers == ("113", "131")

    countdown = {
        "instance_id": "eval_multi_answer-10000-0",
        "numbers": [3, 6, 9],
        "target": 18,
        "num_completions": 8,
    }
    countdown_answers = (
        {"answer": "(6 × 9) / 3", "key": "div(mul(6,9),3)"},
        {"answer": "6 + 3 + 9", "key": "add(9,add(3,6))"},
    )
    assert 6 * 9 / 3 == countdown["target"]
    assert 6 + 3 + 9 == countdown["target"]
    assert countdown_answers[0]["key"] != countdown_answers[1]["key"]

    python_spec = {
        "instance_id": "eval-15100-93",
        "cases": [18, 82, 91, 93],
        "num_modes": 32,
    }
    python_answers = (
        {
            "lines": ("lambda n: 2 if n%2==0 else", "(7 if n%7==0 else 3)"),
            "outputs": (2, 2, 7, 3),
        },
        {
            "lines": ("lambda n: 3 if n%3==0 else", "(13 if n%13==0 else 41)"),
            "outputs": (3, 41, 13, 3),
        },
    )
    for answer in python_answers:
        assert all(
            1 < divisor < value and value % divisor == 0
            for value, divisor in zip(python_spec["cases"], answer["outputs"])
        )
    assert python_answers[0]["outputs"] != python_answers[1]["outputs"]

    mathir = {
        "instance_id": "multi_answer-15900-61",
        "bindings": {"a": 2, "b": -9, "c": -1},
        "valid_mode_count": 5,
    }
    mathir_answers = (
        {"answer": "C;F", "trace": ("x/2 = 8", "x = 16")},
        {"answer": "F;E", "trace": ("x − 18 = −2", "x = 16")},
    )
    assert mathir_answers[0]["trace"] != mathir_answers[1]["trace"]
    assert all(answer["trace"][-1] == "x = 16" for answer in mathir_answers)

    def pantry_ingredient(ingredient_id, available_g, energy, protein, fiber, sodium, tags=()):
        return {
            "id": ingredient_id,
            "available_g": available_g,
            "step_g": 25,
            "min_if_used_g": 50,
            "attributes_per_100g": {
                "energy_kcal": energy,
                "protein_g": protein,
                "fiber_g": fiber,
                "sodium_mg": sodium,
            },
            "tags": list(tags),
        }

    pantry_spec = {
        "verifier": "pantry_plan",
        "pantry_version": "pantry-v1",
        "instance_id": "eval-94002-high_fiber_snack-19",
        "family": "high_fiber_snack",
        "ingredients": [
            pantry_ingredient("almonds", 100, "584", "21.5", "10.8", "0.0", ("tree_nut",)),
            pantry_ingredient("banana", 125, "97.0", "0.74", "4.62", "0.0"),
            pantry_ingredient("carrots", 150, "45.0", "0.941", "3.1", "86.6"),
            pantry_ingredient("grape_tomatoes", 100, "27.0", "0.83", "2.1", "6.0"),
            pantry_ingredient("navel_orange", 150, "47.0", "0.91", "2.0", "9.0"),
            pantry_ingredient("sunflower_seeds", 125, "571", "18.9", "7.22", "0.0"),
        ],
        "targets": {
            "mass_g": {"min": "125", "max": "200"},
            "energy_kcal": {"min": "239.4", "max": "432.25"},
            "protein_g": {"min": "6.423"},
            "fiber_g": {"min": "3.478"},
            "sodium_mg": {"max": "37.6"},
        },
        "min_ingredients": 2,
        "max_ingredients": 4,
        "forbidden_tags": [],
        "certified_mode_count": 14,
    }
    pantry_answers = (
        "navel_orange=75;sunflower_seeds=50",
        "grape_tomatoes=75;sunflower_seeds=50",
    )
    pantry_validations = tuple(
        validate_pantry_plan(answer, pantry_spec) for answer in pantry_answers
    )
    assert all(validation is not None for validation in pantry_validations)
    pantry_keys = tuple(
        validation.canonical_key.removeprefix("pantry_plan:pantry-v1:")
        for validation in pantry_validations
        if validation is not None
    )
    assert pantry_keys == (
        "navel_orange+sunflower_seeds",
        "grape_tomatoes+sunflower_seeds",
    )
    return {
        "graph": {"spec": graph, "modes": graph_modes, "answers": graph_answers},
        "countdown": {"spec": countdown, "answers": countdown_answers},
        "python": {"spec": python_spec, "answers": python_answers},
        "mathir": {"spec": mathir, "answers": mathir_answers},
        "pantry": {
            "spec": pantry_spec,
            "answers": pantry_answers,
            "keys": pantry_keys,
        },
    }


def card(ax, x, y, letter, title, accent, prompt):
    box(ax, x, y, 0.476, 0.450, face=PANEL, edge=GRID, radius=0.018, lw=1.0)
    text(ax, x + 0.020, y + 0.407, letter, fontsize=10.5, fontweight="bold", color=accent, ha="left")
    text(ax, x + 0.054, y + 0.407, title, fontsize=9.4, fontweight="bold", ha="left")
    box(ax, x + 0.020, y + 0.306, 0.436, 0.067, face=WHITE, edge=GRID, radius=0.009, lw=0.8)
    text(ax, x + 0.036, y + 0.340, "PROMPT", fontsize=6.1, fontweight="bold", color=accent, ha="left")
    text(ax, x + 0.103, y + 0.340, prompt, fontsize=6.8, color=MUTED, ha="left")


def answer_box(ax, x, y, *, number, accent, answer, detail, key, code=False):
    box(ax, x, y, 0.207, 0.218, face=WHITE, edge=accent, radius=0.012, lw=1.0)
    text(ax, x + 0.012, y + 0.190, f"ANSWER {number}", fontsize=5.9, fontweight="bold", color=accent, ha="left")
    text(
        ax,
        x + 0.1035,
        y + 0.139,
        answer,
        fontsize=6.4 if code else 8.4,
        fontweight="bold",
        fontfamily="DejaVu Sans Mono" if code else "DejaVu Sans",
    )
    text(ax, x + 0.1035, y + 0.086, detail, fontsize=6.4, color=MUTED)
    box(ax, x + 0.012, y + 0.015, 0.183, 0.039, face=PANEL, edge=GRID, radius=0.007, lw=0.6)
    text(ax, x + 0.1035, y + 0.0345, key, fontsize=6.2, fontweight="bold", color=accent)


def footer(ax, x, y, accent):
    text(ax, x + 0.238, y + 0.025, "✓ both correct     key 1 ≠ key 2", fontsize=6.5, fontweight="bold", color=accent)


def draw_graph(ax, x, y, example):
    card(ax, x, y, "A", "Graph coloring", PURPLE, "partial colors 2??1?2 · fill vertices 2, 3, 5")
    for idx, (answer, mode) in enumerate(zip(example["answers"], example["modes"]), start=1):
        xx = x + (0.020 if idx == 1 else 0.249)
        answer_box(
            ax,
            xx,
            y + 0.066,
            number=idx,
            accent=PURPLE,
            answer=answer,
            detail="complete valid coloring",
            key="key  " + "".join(str(value) for value in mode),
        )
    footer(ax, x, y, PURPLE)


def draw_countdown(ax, x, y, example):
    card(ax, x, y, "B", "Countdown", ORANGE, "tiles {3, 6, 9} · target 18 · use each tile once")
    for idx, answer in enumerate(example["answers"], start=1):
        xx = x + (0.020 if idx == 1 else 0.249)
        answer_box(
            ax,
            xx,
            y + 0.066,
            number=idx,
            accent=ORANGE,
            answer=answer["answer"],
            detail="executes to 18  ✓",
            key="AST  " + answer["key"],
        )
    footer(ax, x, y, ORANGE)


def draw_python(ax, x, y, example):
    card(ax, x, y, "C", "Python factors", BLUE, "lambda n: EXPR · proper factor for 18, 82, 91, 93")
    for idx, answer in enumerate(example["answers"], start=1):
        xx = x + (0.020 if idx == 1 else 0.249)
        answer_box(
            ax,
            xx,
            y + 0.066,
            number=idx,
            accent=BLUE,
            answer="\n".join(answer["lines"]),
            detail="external worker accepts  ✓",
            key="key  [" + ", ".join(str(value) for value in answer["outputs"]) + "]",
            code=True,
        )
    footer(ax, x, y, BLUE)


def draw_mathir(ax, x, y, example):
    card(ax, x, y, "D", "MathIR", GREEN, "x/2 − 9 = −1 · C: subtract −9 · F: ×2 · E: subtract −18")
    for idx, answer in enumerate(example["answers"], start=1):
        xx = x + (0.020 if idx == 1 else 0.249)
        answer_box(
            ax,
            xx,
            y + 0.066,
            number=idx,
            accent=GREEN,
            answer=answer["answer"],
            detail="executes to x = 16  ✓",
            key="key  " + "  →  ".join(answer["trace"]),
        )
    footer(ax, x, y, GREEN)



def draw_pantry(ax, example):
    box(ax, 0.012, 0.055, 0.976, 0.890, face=PANEL, edge=GRID, radius=0.025, lw=1.0)
    text(ax, 0.032, 0.845, "E", fontsize=11.0, fontweight="bold", color=PURPLE, ha="left")
    text(ax, 0.071, 0.845, "PantryPlan", fontsize=10.2, fontweight="bold", ha="left")
    text(
        ax, 0.195, 0.845, "14 exact feasible ingredient supports",
        fontsize=7.0, color=MUTED, ha="left",
    )
    box(ax, 0.032, 0.650, 0.936, 0.125, face=WHITE, edge=GRID, radius=0.012, lw=0.8)
    text(ax, 0.049, 0.712, "PROMPT", fontsize=6.2, fontweight="bold", color=PURPLE, ha="left")
    text(
        ax, 0.126, 0.712,
        "choose 2–4 ingredients · 125–200 g · meet exact energy, protein, fiber, sodium bounds",
        fontsize=7.0, color=MUTED, ha="left",
    )
    for index, (answer, key) in enumerate(
        zip(example["answers"], example["keys"]), start=1
    ):
        x = 0.032 if index == 1 else 0.510
        box(ax, x, 0.190, 0.458, 0.380, face=WHITE, edge=PURPLE, radius=0.014, lw=1.0)
        text(
            ax, x + 0.016, 0.518, f"ANSWER {index}", fontsize=6.2,
            fontweight="bold", color=PURPLE, ha="left",
        )
        text(
            ax, x + 0.229, 0.407, answer.replace(";", ";\n"),
            fontsize=7.4, fontweight="bold", fontfamily="DejaVu Sans Mono",
        )
        box(ax, x + 0.016, 0.228, 0.426, 0.085, face=PANEL, edge=GRID, radius=0.008, lw=0.6)
        text(
            ax, x + 0.229, 0.270,
            "key  support{" + key.replace("+", ", ") + "}",
            fontsize=6.6, fontweight="bold", color=PURPLE,
        )
    text(
        ax, 0.500, 0.108,
        "✓ both allocations satisfy every bound     ingredient support 1 ≠ support 2",
        fontsize=7.0, fontweight="bold", color=PURPLE,
    )
def render() -> None:
    examples = load_and_validate()
    fig = plt.figure(figsize=(7.35, 3.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    draw_graph(ax, 0.012, 0.522, examples["graph"])
    draw_countdown(ax, 0.512, 0.522, examples["countdown"])
    draw_python(ax, 0.012, 0.035, examples["python"])
    draw_mathir(ax, 0.512, 0.035, examples["mathir"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.png')}")


    pantry_fig = plt.figure(figsize=(7.35, 1.48))
    pantry_ax = pantry_fig.add_axes([0, 0, 1, 1])
    pantry_ax.set_axis_off()
    draw_pantry(pantry_ax, examples["pantry"])
    pantry_fig.savefig(
        OUT_PANTRY.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035
    )
    pantry_fig.savefig(
        OUT_PANTRY.with_suffix(".png"), dpi=240,
        bbox_inches="tight", pad_inches=0.035,
    )
    plt.close(pantry_fig)
    print(f"Wrote {OUT_PANTRY.with_suffix('.pdf')}")
    print(f"Wrote {OUT_PANTRY.with_suffix('.png')}")

if __name__ == "__main__":
    render()
