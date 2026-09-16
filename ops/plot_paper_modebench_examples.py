#!/usr/bin/env python3
"""Render paired, validator-checked ModeBench examples for Section 3.1.

One figure, one card per domain in a single row, one shared reading order: the
response the policy emits, the execution that accepts it, and the canonical key
that execution produces. The two answers of a card are the same two verified
modes that Figure 1 tracks, so they carry Figure 1's plasma mode colours.

The row is content-sized rather than laid on a column grid: every card is
measured against the text it actually holds and is given exactly that width, so
the canvas ends where the five domains end. A grid of equal columns cannot do
this --- its column has to hold the widest line in the figure, which leaves the
cards with shorter lines part empty --- and a second row compounds it, because
then every card in a row is also drawn down to the height of whichever card in
it carries a picture. One row of measured cards has no such slack to leave.
"""

from __future__ import annotations

import itertools
import json
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
# layout below can be read as the printed page geometry. The canvas the figure
# actually ships at is the sum of the measured cards, computed in ``render``;
# this is the type scale every hand-placed inch below is tuned against, and it
# is what the printed size is reported relative to.
TYPE_CANVAS = 13.2

# One type size for the whole figure, exactly as in the collapse story, so no
# label reads as a second-class annotation once the page scales it down. The
# shared helper picks the size that matches every other figure on paper.
FONT = style.font_for_canvas(TYPE_CANVAS)
MONO = "DejaVu Sans Mono"

INK = style.INK
MUTED = style.MUTED
GRID = style.GRID
FRAME = style.MUTED
PANEL = style.PANEL  # the shared pale-blue wash; kept as Graph Coloring's tint below
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

# One tint per card, so the five domains separate at a glance instead of
# sitting in five identical pale-blue boxes. Every hue is drawn from the
# green-to-blue arc (95 deg-225 deg) and stays well clear of the violet (about
# 264 deg) and coral-red (about 356 deg) hues that MODE_ONE, MODE_TWO,
# OPERATOR_COLORS, NODE_PAINTS, and the ingredient pictograms already own: a
# wash the same family as a foreground identity colour would make that colour
# look like it belongs to the card rather than to the answer. Lightness and
# saturation are shared across all five (HLS 0.945 / 0.70) so the set reads as
# one wash tinted five ways, not five unrelated backgrounds; Graph Coloring
# keeps the paper's own PANEL blue exactly, so this figure still reads as part
# of the same visual family as every other figure's panel wash. Each tint
# clears >= 3:1 against MODE_TWO, the hardest mark it has to hold (the shipped
# PANEL blue itself clears MODE_TWO at 3.13:1; these hold 3.03-3.34:1).
DOMAIN_PANEL = {
    "A": PANEL,
    "B": "#E7FBF6",
    "C": "#EFFBE7",
    "D": "#E7FBEE",
    "E": "#E7ECFB",
}

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

# Five cards in one row buy their width from one another, so the gaps between
# them are set as tight as they can be and still read as gaps: a card edge
# already separates neighbours, and every tenth of an inch spent here is a
# tenth taken off all five cards' text.
MARGIN = 0.12
GUTTER = 0.18
CARD_PAD = 0.14
LINE = 0.28
KEY_GAP = 0.06
ROW_GAP = 0.12
HEADER = 0.34
PROMPT = 0.28
# A card that draws a picture sets its two answers side by side instead of one
# above the other: two stacked thumbnails would make that card twice the height
# of its neighbours, and the row is drawn down to its tallest card.
PAIR_GAP = 0.12
THUMB_GAP = 0.10
# The box the six vertices are *positioned* in. A vertex is drawn as a disc
# centred on its position, so the thumbnail's ink runs a radius past this box on
# every side; ``GRAPH_INK_*`` is what the layout has to reserve. Keeping the two
# apart is what stops the pair of colourings from touching: reserve the box and
# the outermost discs of the two thumbnails collide in the middle of the card.
GRAPH_WIDTH = 1.40
GRAPH_HEIGHT = 1.15
NODE_RADIUS = 0.205
GRAPH_INK_WIDTH = GRAPH_WIDTH + 2 * NODE_RADIUS
GRAPH_INK_HEIGHT = GRAPH_HEIGHT + 2 * NODE_RADIUS
PAINT_SIZE = 0.24
PAINT_STEP = 0.26
# The key's chips carry no vertex numbers --- the response above them already
# numbered the three it fills --- so they are set smaller and tighter than the
# labelled rows, and the six-vertex key then fits inside the width the two
# thumbnails already ask for.
KEY_PAINT_SIZE = 0.18
KEY_PAINT_STEP = 0.20


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
    # The released row offers six actions, A: div(a), B: add(b), C: sub(b),
    # D: sub(c), E: sub(mul(a,b)), F: mul(a). The panel prints the tail C-F,
    # which holds every action the two answers select and one, D, that neither
    # does: the menu is a choice among legal moves, not a list of the moves the
    # policy happened to make. A leading ellipsis stands for the two elided
    # entries; F closes the released menu, so nothing follows it.
    mathir_elided = ("A", "B")
    # The menu letters are opaque on their own, so each action carries the plain
    # operation it performs and the panel prints that rather than the letter.
    mathir_menu = {
        "C": {"action": "sub(b)", "name": "add 9"},
        "D": {"action": "sub(c)", "name": "add 1"},
        "E": {"action": "sub(mul(a,b))", "name": "add 18"},
        "F": {"action": "mul(a)", "name": "×2"},
    }
    assert mathir_elided + tuple(mathir_menu) == ("A", "B", "C", "D", "E", "F")
    # Execute each printed name against the action it stands for, under this
    # row's own bindings, so the menu cannot drift from the one the verifier
    # applies: an action is applied to both sides, so subtracting a negative
    # binding is set as the addition it performs.
    operands = dict(mathir["bindings"], mul=lambda x, y: x * y)
    for letter, entry in mathir_menu.items():
        action = entry["action"]
        if action.startswith("sub("):
            operand = eval(action[4:-1], {"__builtins__": {}}, operands)  # noqa: S307
            assert entry["name"] == f"add {-operand}", letter
        else:
            assert action == "mul(a)" and entry["name"] == f"×{operands['a']}", letter
    # Set over two lines rather than one: on a card sized to its own content, a
    # forty-character menu would buy the widest card in the row for a line that
    # is read once, and every other line on the card would sit in its shadow.
    entries = [f"{letter}: {entry['name']}" for letter, entry in mathir_menu.items()]
    mathir["menu"] = ("… · " + " · ".join(entries[:2]), " · ".join(entries[2:]))
    mathir_answers = (
        {"answer": "C;F", "trace": ("x/2 = 8", "x = 16")},
        {"answer": "F;E", "trace": ("x − 18 = −2", "x = 16")},
    )
    for answer in mathir_answers:
        answer["named"] = ", then ".join(
            mathir_menu[step]["name"] for step in answer["answer"].split(";")
        )
    selected = {step for answer in mathir_answers for step in answer["answer"].split(";")}
    assert sorted(set(mathir_menu) - selected) == ["D"], "D must stay the unused action"
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
    """Turn the validated examples into one row of domain cards.

    Each answer sets its canonical key on the line beneath the response. No
    card is given a width here: ``measure_row`` reads what each one holds and
    sizes it to that. What a card does choose is how it spends the width it
    asks for --- a prompt broken over three short lines rather than two long
    ones asks for a narrower card and a taller one, which is the right trade
    while a picture card is setting the height of the row anyway.
    """

    graph = examples["graph"]
    countdown = examples["countdown"]
    python = examples["python"]
    mathir = examples["mathir"]
    pantry = examples["pantry"]

    blocks = [
        {
            "letter": "A",
            "title": "Graph",
            "prompt": [
                {"label": "partial colors", "paints": tuple(graph["spec"]["partial_colors"])},
                "fill vertices 2, 3, 5",
            ],
            "check": "✓ valid",
            "glyph": "graph",
            # The one card that draws a picture, so the one card that sets its
            # two answers side by side: stacked thumbnails would make it the
            # tallest card by a clear head and leave slack in all four others.
            "pair": True,
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
            "prompt": ["tiles {3, 6, 9}", "target 18", "use each tile once"],
            "check": "✓ = 18",
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
            "title": "Python",
            "prompt": ["lambda n: EXPR", "one call per n in", "18, 82, 91, 93"],
            "check": "✓ divides",
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
            "prompt": ["solve  x/2 − 9 = −1", *mathir["spec"]["menu"]],
            "check": "✓ x = 16",
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
                "2–4 ingredients",
                "125–200 g total",
                "four exact nutrition bounds",
            ],
            "check": "✓ feasible",
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
            thumbnail_height = {
                "graph": GRAPH_INK_HEIGHT,
            }.get(block.get("glyph"))
            # The text stack always keeps its own height; a picture is added
            # underneath it rather than set alongside, so nothing is padded up
            # to a thumbnail it does not fill.
            answer["stack"] = stack
            answer["height"] = (
                stack + THUMB_GAP + thumbnail_height
                if thumbnail_height is not None
                else stack
            )
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


def probe(fig, ax, value, *, mono: bool = False, bold: bool = False) -> float:
    """Width of a string without leaving it on the canvas.

    The measuring pass and the drawing pass have to agree to the inch, so both
    ask the same renderer the same question; this draws the artist, measures it
    the way ``measure`` does, and removes it again.
    """

    artist = text(
        ax, 0.0, 0.0, value,
        fontfamily=MONO if mono else "sans-serif",
        fontweight="bold" if bold else "normal",
    )
    width = measure(fig, artist)
    artist.remove()
    return width


def segments_width(fig, ax, line) -> float:
    """Width of one response or key line, whether plain text or coloured runs."""

    if isinstance(line, str):
        return probe(fig, ax, line, mono=True, bold=True)
    return sum(probe(fig, ax, value, mono=True, bold=True) for value, _ in line)


def answer_indent(block: dict) -> float:
    """Left offset the response text starts at, past the numbered badge."""

    return 0.34 + (0.26 if block.get("glyph") == "ingredient" else 0.0)


def cell_requirement(fig, ax, block: dict) -> float:
    """Width one answer needs: its widest response line, key chip, or picture."""

    indent = answer_indent(block)
    key_left = 0.0 if block.get("glyph") == "graph" else indent
    needs = [GRAPH_INK_WIDTH] if block.get("glyph") == "graph" else []
    for answer in block["answers"]:
        for index, line in enumerate(answer["response"]):
            if answer.get("response_kind") == "paints":
                needs.append(indent + paint_row_width([paint for paint, _ in line]))
                continue
            width = indent + segments_width(fig, ax, line)
            if index == 0 and answer.get("note"):
                width += 0.18 + probe(fig, ax, answer["note"])
            needs.append(width)
        kind = block.get("key_kind", "text")
        if kind == "paints":
            ink = paint_row_width(
                answer["key"][0], size=KEY_PAINT_SIZE, step=KEY_PAINT_STEP
            )
            needs.append(key_left + 0.12 + ink + 0.24)
        elif kind == "icons":
            needs.append(key_left + 0.12 + icon_row_width(fig, ax, answer["key"][0]) + 0.20)
        else:
            ink = max(segments_width(fig, ax, line) for line in answer["key"])
            needs.append(key_left + 0.12 + ink + 0.24)
    return max(needs)


def card_requirement(fig, ax, block: dict) -> float:
    """Outer width of a card: the widest thing it holds, plus its own padding.

    Measuring rather than assigning is the whole point of the single row. Every
    card asks for exactly what its text needs, the canvas is the sum of those
    asks, and no card is left holding width that belongs to another card's
    longest line.
    """

    needs = [
        0.42 + probe(fig, ax, block["title"], bold=True)
        + 0.16 + probe(fig, ax, block["check"], bold=True)
    ]
    for line in block["prompt"]:
        if isinstance(line, dict):
            needs.append(
                probe(fig, ax, line["label"]) + 0.14 + paint_row_width(line["paints"])
            )
        else:
            needs.append(probe(fig, ax, line))
    cell = cell_requirement(fig, ax, block)
    needs.append(2 * cell + PAIR_GAP if block.get("pair") else cell)
    return max(needs) + 2 * CARD_PAD


def answer_rows(block: dict) -> int:
    """Two answers stacked, unless the card sets them side by side."""

    return 1 if block.get("pair") else 2


def card_height(block: dict) -> float:
    rows = answer_rows(block)
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
                NODE_RADIUS,
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


def paint_row_width(paints, *, size: float = PAINT_SIZE, step: float = PAINT_STEP) -> float:
    """Ink width of a chip row, which is what both passes lay out against."""

    return len(paints) * step - (step - size)


def draw_paint_row(
    ax, left: float, y: float, paints, labels=None,
    *, size: float = PAINT_SIZE, step: float = PAINT_STEP,
) -> float:
    """A row of paint chips, one per vertex, in vertex order."""

    for index, paint in enumerate(paints):
        centre = left + size / 2 + index * step
        known = paint is not None
        ax.add_patch(
            Circle(
                (centre, y),
                size / 2,
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
    return paint_row_width(paints, size=size, step=step)


def icon_row_width(fig, ax, names) -> float:
    """Ink width of a pictogram row, measured the way ``draw_icon_row`` lays it."""

    plus = probe(fig, ax, "+", bold=True)
    return len(names) * 0.32 + (len(names) - 1) * (plus + 0.10)


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
        width = draw_paint_row(
            ax, left + 0.12, baseline, lines[0],
            size=KEY_PAINT_SIZE, step=KEY_PAINT_STEP,
        ) + 0.24
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
    key_height = rows * LINE + 0.08
    # A one-line key rounds all the way to a pill, so it reads as a tag the
    # execution produced rather than another form field on the card. A
    # two-line key keeps a gentler curve: full-pill at that height reads as
    # a lozenge and the two lines start to look cropped by their own edge.
    box(
        ax,
        left,
        top - 0.08 - rows * LINE,
        width,
        key_height,
        face=WHITE,
        edge=accent,
        radius=key_height / 2 if rows == 1 else 0.14,
        lw=1.4,
    )


def draw_answer(
    fig, ax, answer: dict, left: float, top: float, limit: float, where: str,
    *, glyph: str | None = None, key_kind: str = "text",
) -> None:
    accent = answer["accent"]
    row = top - LINE / 2
    # A circle, not a rounded square: it reads as a numbered marker rather
    # than a third size of box competing with the card and the key chip.
    ax.add_patch(
        Circle((left + 0.12, row), 0.12, facecolor=accent, edgecolor=accent, zorder=3, clip_on=False)
    )
    text(ax, left + 0.12, row, answer["number"], color=WHITE, fontweight="bold", ha="center", zorder=4)
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
    # The badge hangs to the left of the response, and the key chip returns to
    # the badge's own margin: in a half-card there is no room to indent it and
    # still fit a six-vertex key.
    key_left = left if glyph == "graph" else left + indent
    draw_key(
        fig,
        ax,
        key_left,
        top - len(answer["response"]) * LINE - KEY_GAP,
        answer,
        accent,
        limit - (key_left - left),
        kind=key_kind,
    )
    # The picture sits under the text it belongs to, centred on the half-card,
    # so the two modes of the domain stand side by side and compare directly.
    thumbnail_top = top - answer["stack"] - THUMB_GAP
    if glyph == "graph":
        draw_mini_graph(
            ax,
            left + (limit - GRAPH_WIDTH) / 2,
            thumbnail_top - NODE_RADIUS - GRAPH_HEIGHT / 2,
            GRAPH_WIDTH,
            GRAPH_HEIGHT,
            answer["mode"],
        )


def draw_card(fig, ax, block: dict, left: float, top: float, height: float) -> None:
    # Every card is drawn to the row's height, so the row keeps a flat baseline
    # even when one domain needs two-line answers.
    width = block["width"]
    # Softer than a form field, short of a pill: a card is the biggest box on
    # the page, so it is the one place a hard corner reads as "boxy" rather
    # than "framed".
    box(
        ax, left, top - height, width, height,
        face=DOMAIN_PANEL[block["letter"]], edge=GRID, radius=0.16, lw=1.1,
    )

    inner_left = left + CARD_PAD
    inner_right = left + width - CARD_PAD
    header = top - CARD_PAD - HEADER / 2
    # A circular letter marker, matching the numbered badges below it, rather
    # than a third rounded-rectangle shape in the same card.
    ax.add_patch(
        Circle((inner_left + 0.16, header), 0.16, facecolor=INK, edgecolor=INK, zorder=3, clip_on=False)
    )
    text(ax, inner_left + 0.16, header, block["letter"], color=WHITE, fontweight="bold", ha="center", zorder=4)
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
            # and the key use, so the reader compares like with like. The chips
            # sit straight on the card wash, with no chip-around-a-box-around-a-
            # card nesting: the card edge is already the only frame this needs.
            label = text(ax, inner_left, baseline, line["label"], color=MUTED)
            chips_left = inner_left + measure(fig, label) + 0.14
            width = draw_paint_row(ax, chips_left, baseline, line["paints"])
            assert chips_left + width <= inner_right, block["title"]
            continue
        prompt = text(ax, inner_left, baseline, line, color=MUTED)
        assert inner_left + measure(fig, prompt) <= inner_right, block["title"]

    rows = answer_rows(block)
    body = rows * block["answer_height"] + (rows - 1) * ROW_GAP
    # Headers and prompts align across a row; the answers are centred in what
    # the tallest card of that row leaves, so no card looks bottom-heavy.
    spare = (height - CARD_PAD - HEADER - block["prompt_lines"] * PROMPT - CARD_PAD) - body
    body_top = top - CARD_PAD - HEADER - block["prompt_lines"] * PROMPT - spare / 2
    if block.get("pair"):
        # Both answers on one row, each thumbnail under its own text: a picture
        # card that stacked them would stand twice as tall as its neighbours,
        # and the row is drawn down to whichever card is tallest.
        cell = (inner_right - inner_left - PAIR_GAP) / 2
        for index, answer in enumerate(block["answers"]):
            draw_answer(
                fig, ax, answer, inner_left + index * (cell + PAIR_GAP),
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


def measure_row(blocks: list[dict]) -> float:
    """Give every card the width its own content needs; return the canvas width.

    Measuring needs a renderer, and the renderer needs a figure, so the pass
    runs on a scratch canvas that is thrown away: type is set in points, so what
    it measures in inches is what the real canvas will draw.
    """

    scratch = plt.figure(figsize=(TYPE_CANVAS, 1.0))
    ax = scratch.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    scratch.canvas.draw()
    for block in blocks:
        block["width"] = card_requirement(scratch, ax, block)
    plt.close(scratch)
    return (
        2 * MARGIN
        + sum(block["width"] for block in blocks)
        + GUTTER * (len(blocks) - 1)
    )


def render() -> None:
    examples = load_and_validate()
    blocks = build_blocks(examples)
    # One row, read in letter order, each card as wide as its own text and no
    # wider. The canvas is whatever those five widths come to, so the figure
    # carries no margin it did not earn.
    width = measure_row(blocks)
    row_height = max(card_height(block) for block in blocks)
    height = 0.10 + row_height + 0.12

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    fig.canvas.draw()

    # No headline over the row: the caption already says that one prompt has
    # two verified answers with different keys, and a title that repeats its
    # own caption costs a band of the page to tell the reader nothing twice.
    left = MARGIN
    for block in blocks:
        draw_card(fig, ax, block, left, height - 0.10, row_height)
        left += block["width"] + GUTTER
    assert abs(left - GUTTER - (width - MARGIN)) < 1e-9, "row must close on the margin"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    # The figure is included at \linewidth, so its printed type is the canvas
    # type scaled by the ratio of the text block to the canvas. Reported here
    # because a content-sized canvas moves it whenever the content moves.
    printed = FONT * style.WIDTH / width
    print(f"Wrote {OUT.with_suffix('.pdf')}  ({width:.2f} x {height:.2f} in)")
    print(f"  cards: " + ", ".join(f"{b['letter']}={b['width']:.2f}" for b in blocks))
    print(f"  type on the page at \\linewidth: {printed:.2f}pt of the {style.FONT:.2f}pt standard")
    print(f"Wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    render()
