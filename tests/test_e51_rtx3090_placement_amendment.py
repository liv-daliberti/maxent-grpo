from pathlib import Path


ROOT = Path(__file__).parents[1]
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e51_rtx3090_placement_amendment_20260724.md"
)


def test_e51_rtx3090_amendment_is_frozen_and_placement_only():
    text = AMENDMENT.read_text(encoding="utf-8")

    for required in (
        "**Status: FROZEN BEFORE SCHEDULER AMENDMENT",
        "`30074931--30074936`",
        "`30074937--30074942`",
        "zero runtime",
        "zero restarts",
        "account `allcs`",
        "partition `lowprio`",
        "one `gpu:rtx_3090` per job",
        "`node020,node022,node023,node024,node026`",
        "changes only scheduler placement",
        "Graph-coloring jobs are not amended",
        "Countdown `30074931--30074936`: node020",
        "optimizer step 6 for Countdown",
        "step 2 for Python",
    ):
        assert required in text
