#!/usr/bin/env python3
"""Render paired, validator-checked ModeBench examples for Section 3.1.

One figure, one row per domain, one shared reading order: the response the
policy emits, the execution that accepts it, and the canonical key that
execution produces. The two answers of a row are the same two verified modes
that Figure 1 tracks, so they carry Figure 1's plasma mode colours.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from oat_drgrpo.pantry_plan import validate_pantry_plan  # noqa: E402

if str(ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

OUT = ROOT / "paper/figures/modebench_examples"
DATA_ROOTS = {
    "graph": ROOT / "var/data/exact_answer_mode_probe/eval",
    "countdown": ROOT / "var/data/exact_countdown_easy3_probe/eval",
    "python": ROOT / "var/data/python_factor_modebench_v1/eval",
    "mathir": ROOT / "var/data/mathir_action_menu_v1/eval",
    "pantry": ROOT / "var/data/pantry_plan_modebench_v2/eval",
}

# Inches on a figure whose single axes spans the canvas one-to-one, so the
# layout below can be read as the printed page geometry.
CANVAS_WIDTH = 13.2

# One type size for the whole figure, exactly as in the collapse story, so no
# label reads as a second-class annotation once the page scales it down. The
# shared helper picks the size that matches every other figure on paper.
FONT = style.font_for_canvas(CANVAS_WIDTH)
MONO = "DejaVu Sans Mono"

INK = style.INK
MUTED = style.MUTED
GRID = style.GRID
FRAME = style.MUTED
PANEL = style.PANEL  # pale orange; caps the plasma range short of its yellow end
WHITE = style.WHITE

# The two mode colours are the first two slots of the shared ``MODE_RAMP``, as
# in `plot_paper_collapse_toy.py`. There they separate two verified modes of
# one Graph Coloring prompt; here they separate two verified modes of one prompt
# in every domain, so the reader meets the same encoding twice.
# Sampled from plt.cm.plasma. Plasma is lightness-monotonic, so any two stops
# differ in luminance as well as hue: the encoding survives colour-vision
# deficiency, grayscale printing, and photocopying. Stops stop at 0.60 because
# beyond it plasma turns yellow and falls under 3:1 on the panel wash.
MODE_ONE = "#41049d"  # plasma 0.10
MODE_TWO = "#dd5e66"  # plasma 0.58

# The puzzle's three paints, copied from the collapse story: a paint is part of
# the question, never one of the measured modes, so it stays off the ramp.
# The puzzle's three paints: a small ordered set, which is what a sequential
# ramp is for, so they take plasma light-to-dark.
NODE_PAINTS = {1: "#e16462", 2: "#9e199d", 3: "#2f0596"}  # plasma .60/.34/.06
# Turned a quarter-turn from the obvious chain layout so the 3-5-6 triangle
# opens to the right: no node then falls on an edge it is not part of.
GRAPH_POSITIONS = {
    1: (0.06, 0.86),
    2: (0.06, 0.14),
    3: (0.40, 0.86),
    4: (0.96, 0.14),
    5: (0.46, 0.14),
    6: (0.78, 0.58),
}
# Naturalistic pictograms, kept desaturated so they read as illustration
# rather than as another data colour.
INGREDIENT_COLORS = {
    "navel_orange": ("#E3893F", "#C06D28"),
    "sunflower_seeds": ("#54483A", "#3A3128"),
    "grape_tomatoes": ("#C4453A", "#9C3229"),
    "almonds": ("#D6A97B", "#A97C51"),
}
LEAF = "#6F8F5A"
# Executable operators are the one thing in a key that is not data, so they
# take the ramp's warm slot against ink-coloured operands.
OPERATOR = "#bc3587"  # plasma 0.44
# One hue per operation, so the reader can see at a glance that the two
# Countdown keys differ in *which* operations they execute and not merely in
# their operands. Symbol and executed-name forms of an operation share a hue.
# Four operations, four plasma stops. Adjacent pairs separate by 1.55:1 in
# luminance, thin for colour alone --- but every operation also prints its own
# name or symbol, so colour reinforces here and never carries the distinction.
OPERATOR_COLORS = {
    "mul": "#20068f",
    "×": "#20068f",
    "div": "#7a02a8",
    "/": "#7a02a8",
    "add": "#bc3587",
    "+": "#bc3587",
    "sub": "#e16462",
    "−": "#e16462",
}
OPERATOR_TOKENS = frozenset(OPERATOR_COLORS)
NODE_TEXT = {1: INK, 2: WHITE, 3: WHITE}

style.apply_rcparams(font_size=FONT)
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
        "mathtext.fontset": "dejavusans",
    }
)

WIDTH = CANVAS_WIDTH
MARGIN = 0.18
COLUMNS = 3
GUTTER = 0.22
CARD_PAD = 0.16
LINE = 0.28
KEY_GAP = 0.06
ROW_GAP = 0.12
HEADER = 0.34
PROMPT = 0.28
ROW_SPACING = 0.16
HEADLINE = 0.42
GRAPH_WIDTH = 1.46
GRAPH_HEIGHT = 1.24
PAINT_SIZE = 0.24
PAINT_STEP = 0.28


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
    # The prompt asks for `lambda n: EXPR`, so the panel shows the EXPR the
    # policy fills in. Each displayed expression is executed here on the real
    # cases, so the picture cannot drift from the vector it claims to return.
    python_answers = (
        {
            "lines": ("2 if n%2==0 else", "7 if n%7==0 else 3"),
            "outputs": (2, 2, 7, 3),
        },
        {
            "lines": ("3 if n%3==0 else", "13 if n%13==0 else 41"),
            "outputs": (3, 41, 13, 3),
        },
    )
    for answer in python_answers:
        shown = eval("lambda n: " + " ".join(answer["lines"]))  # noqa: S307
        assert tuple(shown(value) for value in python_spec["cases"]) == answer["outputs"]
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
    # The menu letters are opaque on their own, so each action carries the plain
    # operation it performs and the panel prints that rather than the letter.
    mathir_menu = {"C": "add 9", "F": "×2", "E": "add 18"}
    mathir_answers = (
        {"answer": "C;F", "trace": ("x/2 = 8", "x = 16")},
        {"answer": "F;E", "trace": ("x − 18 = −2", "x = 16")},
    )
    for answer in mathir_answers:
        answer["named"] = ", then ".join(
            mathir_menu[step] for step in answer["answer"].split(";")
        )
    assert mathir_answers[0]["trace"] != mathir_answers[1]["trace"]
    assert mathir_answers[0]["named"] != mathir_answers[1]["named"]
    assert all(answer["trace"][-1] == "x = 16" for answer in mathir_answers)
    # Execute the depicted menus so the picture cannot drift from the algebra:
    # both orderings of the same two equation-preserving moves reach x = 16.
    assert (-1 + 9) * 2 == 16, "C then F"
    assert (-1 * 2) + 18 == 16, "F then E"

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
    """Turn the validated examples into a three-column grid of domain cards.

    A card is narrower than a key, so each answer sets its canonical key on the
    line beneath the response rather than beside it. PantryPlan spans the two
    free cells of the second row, so the grid closes with content.
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
            "prompt": [
                {"label": "partial colors", "paints": tuple(graph["spec"]["partial_colors"])},
                "fill vertices 2, 3, 5",
            ],
            "check": "✓ valid",
            "span": 1,
            "glyph": "graph",
            "key_kind": "paints",
            "answers": [
                {
                    # The answer names colours, so the card shows colours: the
                    # three vertices the policy fills, then the whole vector
                    # the validator keyed on.
                    "response": [[(digit, vertex) for digit, vertex in zip(answer, (2, 3, 5))]],
                    "response_kind": "paints",
                    "key": [mode],
                    "mode": mode,
                }
                for answer, mode in zip(graph["answers"], graph["modes"])
            ],
        },
        {
            "letter": "B",
            "title": "Countdown",
            "prompt": ["tiles {3, 6, 9} · target 18", "use each tile once"],
            "check": "✓ = 18",
            "span": 1,
            "answers": [
                {
                    "response": [operators(answer["answer"])],
                    "key": [operators(answer["key"])],
                }
                for answer in countdown["answers"]
            ],
        },
        {
            "letter": "C",
            "title": "Python factors",
            "prompt": ["lambda n: EXPR", "one call per n in 18, 82, 91, 93"],
            "check": "✓ divides",
            "span": 1,
            "answers": [
                {
                    "response": list(answer["lines"]),
                    "key": ["[" + ", ".join(str(v) for v in answer["outputs"]) + "]"],
                }
                for answer in python["answers"]
            ],
        },
        {
            "letter": "D",
            "title": "MathIR",
            "prompt": ["solve  x/2 − 9 = −1", "C: add 9 · F: ×2 · E: add 18"],
            "check": "✓ x = 16",
            "span": 1,
            "answers": [
                {
                    "response": [answer["answer"]],
                    # Read like PantryPlan's key: the named components the
                    # execution ran, then the value the validator accepted. The
                    # naming sits in the box for both answers rather than as a
                    # side note on the first, so the two modes are comparable.
                    "key": [answer["named"], " → ".join(answer["trace"])],
                }
                for answer in mathir["answers"]
            ],
        },
        {
            "letter": "E",
            "title": "PantryPlan",
            "prompt": [
                # Split for the single-column card; both facts are kept.
                "2–4 ingredients · 125–200 g",
                "four exact nutrition bounds",
            ],
            "check": "✓ feasible",
            # One unit like the others, which stacks its two answers and lets
            # the cards pack by height instead of by row.
            "span": 1,
            "glyph": "ingredient",
            "key_kind": "icons",
            "answers": [
                {
                    "response": [line + ";" for line in answer.split(";")[:-1]]
                    + [answer.rsplit(";", 1)[-1]],
                    "key": [tuple(key.split("+"))],
                }
                for answer, key in zip(pantry["answers"], pantry["keys"])
            ],
        },
    ]
    for block in blocks:
        assert len(block["answers"]) == 2
        for index, answer in enumerate(block["answers"]):
            answer["accent"] = (MODE_ONE, MODE_TWO)[index]
            answer["number"] = str(index + 1)
        assert block["answers"][0]["key"] != block["answers"][1]["key"]
        block["prompt_lines"] = len(block["prompt"])
        for answer in block["answers"]:
            key_rows = 1 if block.get("key_kind") in {"paints", "icons"} else len(answer["key"])
            stack = len(answer["response"]) * LINE + KEY_GAP + key_rows * LINE + 0.08
            # A card that draws thumbnails is as tall as the thumbnail.
            answer["height"] = max(stack, GRAPH_HEIGHT + 0.22) if block.get("glyph") == "graph" else stack
        block["answer_height"] = max(answer["height"] for answer in block["answers"])
    return blocks


def operators(value: str) -> list[tuple[str, str]]:
    """Split an executed expression into operator and operand segments."""

    parts = [part for part in re.split(r"(div|mul|add|sub|×|/|\+|−)", value) if part]
    return [(part, OPERATOR_COLORS.get(part, INK)) for part in parts]


def draw_line(fig, ax, x: float, y: float, line, *, mono: bool = True) -> float:
    """Draw one line, which may be plain text or coloured segments."""

    segments = [(line, INK)] if isinstance(line, str) else line
    cursor = x
    for value, color in segments:
        artist = text(
            ax,
            cursor,
            y,
            value,
            color=color,
            fontfamily=MONO if mono else "sans-serif",
            fontweight="bold",
            zorder=3,
        )
        cursor += measure(fig, artist)
    return cursor - x


def measure(fig, artist) -> float:
    """Width of a drawn artist in inches, i.e. in layout units."""

    return artist.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi


def card_width(span: int) -> float:
    unit = (WIDTH - 2 * MARGIN - (COLUMNS - 1) * GUTTER) / COLUMNS
    return unit * span + GUTTER * (span - 1)


def card_height(block: dict) -> float:
    rows = 1 if block["span"] == 2 else 2
    body = rows * block["answer_height"] + (rows - 1) * ROW_GAP
    return CARD_PAD + HEADER + block["prompt_lines"] * PROMPT + body + CARD_PAD


def draw_mini_graph(ax, left: float, center: float, width: float, height: float, mode) -> None:
    """The whole coloring the answer commits to, at thumbnail size.

    The three digits in the response fill only the hidden vertices; this shows
    the completion the validator actually executed, numbered and in the
    puzzle's paints.
    """

    def place(node):
        x, y = GRAPH_POSITIONS[node]
        return left + x * width, center - height / 2 + y * height

    for first, second in ((1, 2), (1, 3), (3, 5), (3, 6), (4, 6), (5, 6)):
        start, end = place(first), place(second)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#A8B4BE",
            linewidth=1.2,
            zorder=2,
            solid_capstyle="round",
        )
    for node, color in enumerate(mode, start=1):
        x, y = place(node)
        ax.add_patch(
            Circle(
                (x, y),
                0.205,
                facecolor=NODE_PAINTS[color],
                edgecolor=WHITE,
                linewidth=1.1,
                zorder=3,
                clip_on=False,
            )
        )
        text(ax, x, y, str(node), color=NODE_TEXT[color], fontweight="bold", ha="center", zorder=4)


def draw_ingredient(ax, name: str, x: float, y: float) -> None:
    """A pictogram of the ingredient the allocation names."""

    fill, edge = INGREDIENT_COLORS[name]
    common = {"linewidth": 0.7, "zorder": 3, "clip_on": False}
    if name == "navel_orange":
        ax.add_patch(Ellipse((x, y - 0.062), 0.086, 0.052, angle=-25,
                             facecolor=LEAF, edgecolor=LEAF, **common))
        ax.add_patch(Circle((x, y), 0.086, facecolor=fill, edgecolor=edge, **common))
        return
    if name == "grape_tomatoes":
        for offset in (-0.052, 0.052):
            ax.add_patch(Circle((x + offset, y - 0.014), 0.060,
                                facecolor=fill, edgecolor=edge, **common))
        ax.add_patch(Ellipse((x, y + 0.060), 0.080, 0.034,
                             facecolor=LEAF, edgecolor=LEAF, **common))
        return
    if name == "sunflower_seeds":
        for offset, angle in ((-0.048, 28), (0.048, -28)):
            ax.add_patch(Ellipse((x + offset, y), 0.066, 0.120, angle=angle,
                                 facecolor=fill, edgecolor=edge, **common))
        return
    ax.add_patch(Ellipse((x - 0.034, y), 0.080, 0.132, angle=32,
                         facecolor=fill, edgecolor=edge, **common))
    ax.add_patch(Ellipse((x + 0.044, y - 0.012), 0.080, 0.132, angle=-20,
                         facecolor=fill, edgecolor=edge, **common))


def draw_paint_row(ax, left: float, y: float, paints, labels=None) -> float:
    """A row of paint chips, one per vertex, in vertex order."""

    step = PAINT_STEP
    for index, paint in enumerate(paints):
        centre = left + PAINT_SIZE / 2 + index * step
        known = paint is not None
        ax.add_patch(
            Circle(
                (centre, y),
                PAINT_SIZE / 2,
                facecolor=NODE_PAINTS[int(paint)] if known else WHITE,
                edgecolor=WHITE if known else FRAME,
                linewidth=1.0,
                zorder=3,
                clip_on=False,
            )
        )
        label = labels[index] if labels is not None else (None if known else "?")
        if label is not None:
            text(
                ax, centre, y, str(label),
                color=NODE_TEXT[int(paint)] if known else MUTED,
                fontweight="bold", ha="center", zorder=4,
            )
    return len(paints) * step - (step - PAINT_SIZE)


def draw_icon_row(fig, ax, left: float, y: float, names) -> float:
    """The support as pictograms alone, joined by the key's own plus sign."""

    cursor = left
    for index, name in enumerate(names):
        if index:
            plus = text(ax, cursor, y, "+", color=MUTED, fontweight="bold", zorder=3)
            cursor += measure(fig, plus) + 0.10
        draw_ingredient(ax, name, cursor + 0.11, y)
        cursor += 0.32
    return cursor - left


def draw_key(
    fig, ax, left: float, top: float, answer: dict, accent: str, limit: float,
    *, kind: str = "text",
) -> None:
    """Key chip on its own line, sized to its content, under the response."""

    lines = answer["key"]
    baseline = top - 0.04 - 0.5 * LINE
    if kind == "paints":
        width = draw_paint_row(ax, left + 0.12, baseline, lines[0]) + 0.24
    elif kind == "icons":
        width = draw_icon_row(fig, ax, left + 0.12, baseline, lines[0]) + 0.20
    else:
        widths = [
            draw_line(fig, ax, left + 0.12, top - 0.04 - (index + 0.5) * LINE, line)
            for index, line in enumerate(lines)
        ]
        width = max(widths) + 0.24
    assert width <= limit, f"key {lines} needs {width:.2f}in of {limit:.2f}in"
    rows = 1 if kind in {"paints", "icons"} else len(lines)
    box(
        ax,
        left,
        top - 0.08 - rows * LINE,
        width,
        rows * LINE + 0.08,
        face=WHITE,
        edge=accent,
        radius=0.05,
        lw=1.4,
    )


def draw_answer(
    fig, ax, answer: dict, left: float, top: float, limit: float, where: str,
    *, glyph: str | None = None, key_kind: str = "text",
) -> None:
    accent = answer["accent"]
    # A thumbnail owns the right of the card, so the text and the key chip are
    # held to what is left of it rather than to the full card width.
    card_limit = limit
    limit = limit - (GRAPH_WIDTH + 0.12) if glyph == "graph" else limit
    row = top - LINE / 2
    box(ax, left, row - 0.12, 0.24, 0.24, face=accent, edge=accent, radius=0.05)
    text(ax, left + 0.12, row, answer["number"], color=WHITE, fontweight="bold", ha="center")
    # Ingredient rows carry a pictogram of what they allocate, so the text
    # shifts right to leave it room.
    indent = 0.34 + (0.26 if glyph == "ingredient" else 0.0)
    for index, line in enumerate(answer["response"]):
        baseline = row - index * LINE
        if answer.get("response_kind") == "paints":
            # Each chip is one hidden vertex, labelled with the vertex it fills.
            width = draw_paint_row(
                ax, left + indent, baseline,
                [paint for paint, _ in line], [vertex for _, vertex in line],
            )
        else:
            if glyph == "ingredient":
                draw_ingredient(ax, line.split("=", 1)[0], left + 0.45, baseline)
            width = draw_line(fig, ax, left + indent, baseline, line)
        assert width <= limit - indent, f"{where} answer {answer['number']}"
        if index == 0 and answer.get("note"):
            # A route through an action menu is opaque on its own, so the
            # arithmetic it stands for is spelled out beside it.
            note = text(ax, left + indent + width + 0.18, baseline, answer["note"], color=MUTED)
            assert (
                indent + width + 0.18 + measure(fig, note) <= limit
            ), f"{where} note {answer['number']}"
    draw_key(
        fig,
        ax,
        left + indent,
        top - len(answer["response"]) * LINE - KEY_GAP,
        answer,
        accent,
        limit - indent,
        kind=key_kind,
    )
    if glyph == "graph":
        draw_mini_graph(
            ax,
            left + card_limit - GRAPH_WIDTH,
            top - answer["height"] / 2,
            GRAPH_WIDTH,
            GRAPH_HEIGHT,
            answer["mode"],
        )


def draw_card(fig, ax, block: dict, left: float, top: float, height: float) -> None:
    # Both cards in a row are drawn to the row's height, so the grid keeps a
    # flat baseline even when one domain needs two-line answers.
    width = card_width(block["span"])
    box(ax, left, top - height, width, height, face=PANEL, edge=GRID, radius=0.09, lw=1.1)

    inner_left = left + CARD_PAD
    inner_right = left + width - CARD_PAD
    header = top - CARD_PAD - HEADER / 2
    box(ax, inner_left, header - 0.15, 0.32, 0.30, face=INK, edge=INK, radius=0.05)
    text(ax, inner_left + 0.16, header, block["letter"], color=WHITE, fontweight="bold", ha="center")
    title = text(ax, inner_left + 0.42, header, block["title"], fontweight="bold")
    check = text(ax, inner_right, header, block["check"], color=MUTED, fontweight="bold", ha="right")
    assert (
        inner_left + 0.42 + measure(fig, title)
        < inner_right - measure(fig, check) - 0.15
    ), block["title"]

    for index, line in enumerate(block["prompt"]):
        baseline = top - CARD_PAD - HEADER - (index + 0.5) * PROMPT
        if isinstance(line, dict):
            # The prompt's partial coloring is set in the same chips the answer
            # and the key use, so the reader compares like with like.
            label = text(ax, inner_left, baseline, line["label"], color=MUTED)
            chips_left = inner_left + measure(fig, label) + 0.14
            width = draw_paint_row(ax, chips_left + 0.10, baseline, line["paints"]) + 0.20
            box(ax, chips_left, baseline - PROMPT / 2 - 0.02, width, PROMPT + 0.04,
                face=WHITE, edge=GRID, radius=0.05, lw=1.1)
            assert chips_left + width <= inner_right, block["title"]
            continue
        prompt = text(ax, inner_left, baseline, line, color=MUTED)
        assert inner_left + measure(fig, prompt) <= inner_right, block["title"]

    rows = 1 if block["span"] == 2 else 2
    body = rows * block["answer_height"] + (rows - 1) * ROW_GAP
    # Headers and prompts align across a row; the answers are centred in what
    # the tallest card of that row leaves, so no card looks bottom-heavy.
    spare = (height - CARD_PAD - HEADER - block["prompt_lines"] * PROMPT - CARD_PAD) - body
    body_top = top - CARD_PAD - HEADER - block["prompt_lines"] * PROMPT - spare / 2
    if block["span"] == 2:
        # The wide card has room to set both answers side by side.
        cell = (inner_right - inner_left - GUTTER) / 2
        for index, answer in enumerate(block["answers"]):
            draw_answer(
                fig, ax, answer, inner_left + index * (cell + GUTTER),
                body_top, cell, block["title"], glyph=block.get("glyph"),
                key_kind=block.get("key_kind", "text"),
            )
        return
    for index, answer in enumerate(block["answers"]):
        draw_answer(
            fig, ax, answer, inner_left,
            body_top - index * (block["answer_height"] + ROW_GAP),
            inner_right - inner_left, block["title"], glyph=block.get("glyph"),
            key_kind=block.get("key_kind", "text"),
        )


def render() -> None:
    examples = load_and_validate()
    blocks = build_blocks(examples)
    # Column packing: each card goes to whichever column currently has the most
    # room left. Row-based placement padded every card to its row's tallest and
    # left a band of empty wash under the short ones.
    heights = [card_height(block) for block in blocks]
    order = list(range(len(blocks)))
    packed: list[list[int]] = [[] for _ in range(COLUMNS)]
    used = [0.0] * COLUMNS
    for index in order:
        column = min(range(COLUMNS), key=lambda c: used[c])
        packed[column].append(index)
        used[column] += heights[index] + ROW_SPACING

    height = 0.10 + HEADLINE + max(used) - ROW_SPACING + 0.12

    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    fig.canvas.draw()

    top = height - 0.10
    text(
        ax,
        MARGIN + 0.20,
        top - HEADLINE / 2,
        "One prompt, two verified answers, two different canonical keys",
        fontweight="bold",
    )

    unit = card_width(1)
    for column, indices in enumerate(packed):
        left = MARGIN + column * (unit + GUTTER)
        cursor = top - HEADLINE
        for index in indices:
            block = blocks[index]
            draw_card(fig, ax, block, left, cursor, heights[index])
            cursor -= heights[index] + ROW_SPACING

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Wrote {OUT.with_suffix('.pdf')}  ({WIDTH:.2f} x {height:.2f} in)")
    print(f"Wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    render()
