#!/usr/bin/env python3
"""Render paired, validator-checked ModeBench examples for Section 3.1.

One figure, one row per domain, one shared reading order: the response the
policy emits, the execution that accepts it, and the canonical key that
execution produces. The two answers of a row are the same two verified modes
that Figure 1 tracks, so they carry Figure 1's plasma mode colours.
"""

from __future__ import annotations

import itertools
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from oat_drgrpo.pantry_plan import validate_pantry_plan  # noqa: E402
OUT = ROOT / "paper/figures/modebench_examples"
DATA_ROOTS = {
    "graph": ROOT / "var/data/exact_answer_mode_probe/eval",
    "countdown": ROOT / "var/data/exact_countdown_easy3_probe/eval",
    "python": ROOT / "var/data/python_factor_modebench_v1/eval",
    "mathir": ROOT / "var/data/mathir_action_menu_v1/eval",
    "pantry": ROOT / "var/data/pantry_plan_modebench_v2/eval",
}

# One type size for the whole figure, exactly as in the collapse story, so no
# label reads as a second-class annotation once the page scales it down.
FONT = 17.0
MONO = "DejaVu Sans Mono"

INK = "#1B2733"
MUTED = "#5B6B7B"
GRID = "#F0DEC6"
FRAME = "#C7AE8E"
PANEL = "#FEF4E7"
WHITE = "#FFFFFF"

# The two mode colours are quoted verbatim from `plot_paper_collapse_toy.py`
# (`plt.cm.plasma` at 0.10 and 0.45). There they separate two verified modes of
# one Graph Coloring prompt; here they separate two verified modes of one prompt
# in every domain, so the reader meets the same encoding twice.
MODE_ONE = "#41049D"
MODE_TWO = "#BF3984"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
        "font.size": FONT,
        "mathtext.fontset": "dejavusans",
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)

# Every coordinate below is in inches on a figure whose single axes spans the
# canvas one-to-one, so the layout can be read as the printed page geometry.
WIDTH = 13.2
MARGIN = 0.18
X_BADGE = 0.40
X_TITLE = 1.00
X_PROMPT = 3.30
X_SWATCH = 1.00
X_LABEL = 1.30
X_RESPONSE = 3.00
X_CHECK = 7.10
X_KEY_RIGHT = 12.55
X_NEQ = 12.80
LINE = 0.28
ROW_PAD = 0.08
HEADER_ZONE = 0.42
BLOCK_PAD = 0.05
BLOCK_GAP = 0.11
HEADLINE = 0.36
CAPTION = 0.30


def box(ax, x, y, w, h, *, face=WHITE, edge=GRID, radius=0.06, lw=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, **kwargs):
    defaults = {
        "fontsize": FONT,
        "color": INK,
        "ha": "left",
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
    # Disjoint supports, so the two rows share no ingredient at all and the
    # key difference cannot be read as a quantity difference.
    pantry_answers = (
        "navel_orange=75;sunflower_seeds=50",
        "grape_tomatoes=75;almonds=50",
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
        "almonds+grape_tomatoes",
    )
    assert not set(pantry_keys[0].split("+")) & set(pantry_keys[1].split("+"))
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


def build_blocks(examples: dict[str, dict]) -> list[dict]:
    """Turn the validated examples into one uniform row-per-domain layout.

    Every row carries the same three fields the reader is asked to compare:
    the emitted response, the check that accepted it, and the canonical key
    that same check produced.
    """

    graph = examples["graph"]
    countdown = examples["countdown"]
    python = examples["python"]
    mathir = examples["mathir"]
    pantry = examples["pantry"]

    blocks = [
        {
            "letter": "A",
            "title": "Graph coloring",
            "prompt": "partial colors 2??1?2 · fill vertices 2, 3, 5",
            "rows": [
                {
                    "response": [answer],
                    "check": "✓ valid coloring",
                    "key": ["".join(str(value) for value in mode)],
                }
                for answer, mode in zip(graph["answers"], graph["modes"])
            ],
        },
        {
            "letter": "B",
            "title": "Countdown",
            "prompt": "tiles {3, 6, 9} · target 18 · use each tile once",
            "rows": [
                {
                    "response": [answer["answer"]],
                    "check": "✓ = 18",
                    "key": [answer["key"]],
                }
                for answer in countdown["answers"]
            ],
        },
        {
            "letter": "C",
            "title": "Python factors",
            "prompt": "lambda n: EXPR · called once per n in 18, 82, 91, 93 · return a proper factor d",
            "rows": [
                {
                    "response": list(answer["lines"]),
                    "check": "✓ divides each n",
                    "key": ["[" + ", ".join(str(v) for v in answer["outputs"]) + "]"],
                }
                for answer in python["answers"]
            ],
        },
        {
            "letter": "D",
            "title": "MathIR",
            "prompt": "x/2 − 9 = −1 · C: subtract −9 · F: ×2 · E: subtract −18",
            "rows": [
                {
                    "response": [answer["answer"]],
                    "check": "✓ x = 16",
                    "key": [" → ".join(answer["trace"])],
                }
                for answer in mathir["answers"]
            ],
        },
        {
            "letter": "E",
            "title": "PantryPlan",
            "prompt": "2–4 ingredients · 125–200 g · four exact nutrition bounds",
            "rows": [
                {
                    "response": [line + ";" for line in answer.split(";")[:-1]]
                    + [answer.rsplit(";", 1)[-1]],
                    "check": "✓ feasible",
                    "key": [key.split("+")[0] + " +", key.split("+")[1]],
                }
                for answer, key in zip(pantry["answers"], pantry["keys"])
            ],
        },
    ]
    for block in blocks:
        assert len(block["rows"]) == 2
        for index, row in enumerate(block["rows"]):
            row["accent"] = (MODE_ONE, MODE_TWO)[index]
            row["label"] = f"ANSWER {index + 1}"
            row["lines"] = max(len(row["response"]), len(row["key"]))
        assert block["rows"][0]["key"] != block["rows"][1]["key"]
    return blocks


def measure(fig, artist) -> tuple[float, float]:
    """Width and height of a drawn artist in inches, i.e. in layout units."""

    extent = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    return extent.width / fig.dpi, extent.height / fig.dpi


def row_height(lines: int) -> float:
    return 2 * ROW_PAD + lines * LINE


def block_height(block: dict) -> float:
    rows = sum(row_height(row["lines"]) for row in block["rows"])
    return HEADER_ZONE + rows + BLOCK_PAD


def draw_key(fig, ax, right: float, y: float, lines: list[str], accent: str) -> float:
    """Right-aligned key chip that grows to fit its own text."""

    artists = [
        text(
            ax,
            right - 0.13,
            y - (index - (len(lines) - 1) / 2) * LINE,
            line,
            ha="right",
            fontfamily=MONO,
            fontweight="bold",
            zorder=3,
        )
        for index, line in enumerate(lines)
    ]
    width = max(measure(fig, artist)[0] for artist in artists) + 0.26
    # Kept under the row pitch so consecutive chips never touch.
    height = len(lines) * LINE + 0.08
    box(
        ax,
        right - width,
        y - height / 2,
        width,
        height,
        face=WHITE,
        edge=accent,
        radius=0.05,
        lw=1.4,
    )
    return right - width


def draw_block(fig, ax, block: dict, top: float) -> None:
    height = block_height(block)
    bottom = top - height
    box(
        ax,
        MARGIN,
        bottom,
        WIDTH - 2 * MARGIN,
        height,
        face=PANEL,
        edge=GRID,
        radius=0.09,
        lw=1.1,
    )

    header = top - HEADER_ZONE / 2 - 0.02
    box(ax, X_BADGE, header - 0.15, 0.32, 0.30, face=INK, edge=INK, radius=0.05)
    text(ax, X_BADGE + 0.16, header, block["letter"], color=WHITE, fontweight="bold", ha="center")
    text(ax, X_TITLE, header, block["title"], fontweight="bold")
    prompt = text(ax, X_PROMPT, header, block["prompt"], color=MUTED)
    assert X_PROMPT + measure(fig, prompt)[0] < WIDTH - MARGIN - 0.15, block["title"]

    centers = []
    cursor = top - HEADER_ZONE
    for row in block["rows"]:
        span = row_height(row["lines"])
        center = cursor - span / 2
        centers.append(center)
        accent = row["accent"]

        box(ax, X_SWATCH, center - 0.09, 0.18, 0.18, face=accent, edge=accent, radius=0.04)
        text(ax, X_LABEL, center, row["label"], color=accent, fontweight="bold")
        for index, line in enumerate(row["response"]):
            response = text(
                ax,
                X_RESPONSE,
                center - (index - (len(row["response"]) - 1) / 2) * LINE,
                line,
                fontfamily=MONO,
                fontweight="bold",
            )
            assert X_RESPONSE + measure(fig, response)[0] < X_CHECK - 0.15, block["title"]
        check = text(ax, X_CHECK, center, row["check"], color=MUTED)
        assert X_CHECK + measure(fig, check)[0] < X_KEY_RIGHT, block["title"]
        left = draw_key(fig, ax, X_KEY_RIGHT, center, row["key"], accent)
        assert left > X_CHECK + measure(fig, check)[0] + 0.10, block["title"]
        cursor -= span

    # The whole figure exists to say that both keys are accepted and that they
    # are not the same key, so the comparison itself gets a mark.
    ax.plot(
        [X_NEQ - 0.06, X_NEQ - 0.06],
        [centers[0], centers[1]],
        color=FRAME,
        linewidth=1.0,
        solid_capstyle="butt",
        zorder=1,
    )
    text(
        ax,
        X_NEQ - 0.06,
        (centers[0] + centers[1]) / 2,
        "≠",
        ha="center",
        fontweight="bold",
        zorder=3,
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.5},
    )


def render() -> None:
    examples = load_and_validate()
    blocks = build_blocks(examples)

    stack = sum(block_height(block) for block in blocks)
    stack += BLOCK_GAP * (len(blocks) - 1)
    height = 0.10 + HEADLINE + CAPTION + stack + 0.12

    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    fig.canvas.draw()

    top = height - 0.10
    text(ax, MARGIN + 0.22, top - HEADLINE / 2, "One prompt, two verified answers, two different keys", fontweight="bold")
    caption = top - HEADLINE - CAPTION / 2
    text(ax, X_RESPONSE, caption, "response", color=MUTED)
    text(ax, X_CHECK, caption, "execute + verify", color=MUTED)
    text(ax, X_KEY_RIGHT - 0.13, caption, "canonical key", color=MUTED, ha="right")

    cursor = top - HEADLINE - CAPTION
    for block in blocks:
        draw_block(fig, ax, block, cursor)
        cursor -= block_height(block) + BLOCK_GAP

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    render()
