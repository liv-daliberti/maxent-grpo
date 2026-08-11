"""One registry of every released cohort, and what each is for.

Three times a cohort was launched and then failed to appear where it belonged:
a semantic arm missing from Figure 4, another missing from the campaign table, a
third missing from the 3B row. Each time the runs existed and only the wiring was
absent, because the plotter and the status table each carried their own
hand-maintained list.

This module is the single source of truth. Adding a cohort here is what makes it
visible everywhere; a released ledger that is *not* here fails
`test_cohort_registry.py`, so forgetting is a test failure rather than a silently
missing line. ``plotted`` is a deliberate field: a cohort may be excluded from
the figure, but the reason has to be written down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"

# Row labels used by the figure; a cohort's family must be one of these or None.
FAMILIES = ("Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B")


@dataclass(frozen=True)
class Cohort:
    tag: str
    label: str
    ledger: str
    kind: str                       # paired | semantic | replay_dose |
                                    # point_maze | repair
    family: str | None = None
    arm: str | None = None          # figure arm key, for kind == "semantic"
    comparator: str | None = None   # arm its paired difference is read against
    plotted: bool = False
    not_plotted_because: str = ""
    supersedes_domains_of: str | None = None

    def path(self) -> Path:
        return ARTIFACTS / self.ledger

    def exists(self) -> bool:
        return self.path().is_file()


REGISTRY: tuple[Cohort, ...] = (
    # --- the paired control/replay cohorts, one per family -----------------
    Cohort("e78", "E78  Qwen-0.5B   replay vs control",
           "e78_verified_replay_only_05b_jobs.json", "paired", "Qwen2.5-0.5B"),
    Cohort("e79", "E79  Falcon-1B   replay vs control",
           "e79_falcon1b_aligned_verified_replay_jobs.json", "paired", "Falcon3-1B"),
    Cohort("e80r1", "E80r1 Qwen-3B    replay vs control",
           "e80r1_qwen3b_aligned_verified_replay_jobs.json", "paired", "Qwen2.5-3B"),

    # --- fixed-dose semantic MaxEnt arms ----------------------------------
    # Comparator differs by arm and is load-bearing: an arm added on top of
    # `replay` must be read against `replay`, and one added on top of `control`
    # against `control`. Crossing them reports two interventions as one.
    Cohort("e81", "E81  Qwen-0.5B   replay + MaxEnt",
           "e81_semantic_maxent_verified_replay_05b_jobs.json", "semantic",
           "Qwen2.5-0.5B", "semantic", "replay", plotted=True),
    Cohort("e82", "E82  Falcon-1B   replay + MaxEnt",
           "e82_falcon_semantic_maxent_verified_replay_jobs.json", "semantic",
           "Falcon3-1B", "semantic", "replay", plotted=True),
    Cohort("e83", "E83  Qwen-0.5B   MaxEnt only",
           "e83_semantic_maxent_without_replay_05b_jobs.json", "semantic",
           "Qwen2.5-0.5B", "semantic_only", "control", plotted=True),
    Cohort("e86", "E86  Falcon-1B   MaxEnt only",
           "e86_falcon_semantic_maxent_without_replay_jobs.json", "semantic",
           "Falcon3-1B", "semantic_only", "control", plotted=True),
    Cohort("e87", "E87  Qwen-3B     replay + MaxEnt",
           "e87_qwen3b_semantic_maxent_seed70_jobs.json", "semantic",
           "Qwen2.5-3B", "semantic", "replay", plotted=True),

    # --- repair cohort ----------------------------------------------------
    Cohort("e85", "E85  both        Pantry repair",
           "e85_pantry_semantic_repair_jobs.json", "repair",
           supersedes_domains_of="e81,e82,e83"),

    # --- adaptive-coefficient arms ----------------------------------------
    Cohort("e88", "E88  Qwen-0.5B   replay + adaptive MaxEnt",
           "e88_adaptive_semantic_maxent_05b_jobs.json", "semantic",
           "Qwen2.5-0.5B", "semantic", "replay", plotted=False,
           not_plotted_because=(
               "closed early on a registered mechanism-gate failure; its cells "
               "are evidence about the controller, not about breadth, and "
               "plotting them beside the fixed arm would invite reading a "
               "saturated coefficient as a treatment effect")),
    Cohort("e89", "E89  Qwen-0.5B   replay + adaptive rho=.015",
           "e89_adaptive_semantic_maxent_reachable_05b_jobs.json", "semantic",
           "Qwen2.5-0.5B", "adaptive_semantic", "replay", plotted=True),

    # The same adaptive arm on the other two families. One registered rho
    # across all three is what makes "uniform semantic pressure" a single
    # treatment rather than three tuned ones; reachability was checked per
    # family before submission (see each preregistration).
    Cohort("e91", "E91  Falcon-1B   replay + adaptive MaxEnt",
           "e91_falcon_adaptive_semantic_maxent_jobs.json", "semantic",
           "Falcon3-1B", "adaptive_semantic", "replay", plotted=True),
    Cohort("e92", "E92  Qwen-3B     replay + adaptive MaxEnt",
           "e92_qwen3b_adaptive_semantic_maxent_jobs.json", "semantic",
           "Qwen2.5-3B", "adaptive_semantic", "replay", plotted=True),

    # --- replay-dose arms -------------------------------------------------
    # Not semantic: this varies the replay dose rule, not the objective. It is
    # drawn like the other overlay arms because it is read the same way -- a
    # paired difference against the arm it modifies.
    Cohort("e90", "E90  Qwen-0.5B   bank-normalized replay",
           "e90_bank_normalized_replay_05b_jobs.json", "replay_dose",
           "Qwen2.5-0.5B", "bank_normalized_replay", "replay", plotted=True),

    # --- PointMaze extensions --------------------------------------------
    Cohort("e78pm", "E78-PM Qwen-0.5B PointMaze",
           "e78pm_point_maze_verified_replay_only_05b_jobs.json",
           "point_maze", "Qwen2.5-0.5B"),
    Cohort("e79pm", "E79-PM Falcon-1B PointMaze",
           "e79pm_falcon_point_maze_verified_replay_jobs.json",
           "point_maze", "Falcon3-1B"),

    # --- PointMaze Tour, the redesigned sixth domain ----------------------
    # v1 (e78pm/e79pm) is inert: its control moves distinct@8 by +0.016 over
    # eight passes against a within-run checkpoint SD of 0.048, so it measures
    # nothing about retention. These cells are the admission ladder for the
    # replacement, not evidence for it.
    Cohort("tourgate", "TOUR  Qwen-0.5B  Tour admission gate",
           "point_maze_tour_gate_jobs.json", "point_maze", "Qwen2.5-0.5B",
           plotted=False,
           not_plotted_because=(
               "stage-2 collapse gate and stage-3 smoke on development maps; "
               "they decide whether the redesigned sixth domain earns a "
               "cohort, and plotting a domain-admission check beside measured "
               "treatment effects would read as an effect estimate")),
)

# Released ledgers that are deliberately outside the registry, with the reason.
EXCLUDED: dict[str, str] = {
    "e80_qwen3b_aligned_verified_replay_jobs.json":
        "cancelled after a cosine-horizon error; superseded by e80r1 and "
        "retained as audit-only evidence",
}


def by_tag(tag: str) -> Cohort:
    for cohort in REGISTRY:
        if cohort.tag == tag:
            return cohort
    raise KeyError(f"no cohort registered under {tag!r}")


# Arm kinds that are drawn as an overlay curve against a comparator. Semantic
# arms change the objective; replay-dose arms change how the replay dose is
# set. Both are read as a paired difference against the arm they modify, so
# both are attached and drawn by the same code path.
OVERLAY_KINDS = ("semantic", "replay_dose")


def semantic_arms(family: str, *, plotted_only: bool = True) -> list[Cohort]:
    """Overlay arms belonging to one figure row, in registry order."""

    return [
        c for c in REGISTRY
        if c.kind in OVERLAY_KINDS and c.family == family
        and (c.plotted or not plotted_only)
    ]


def repair_for(tag: str) -> Cohort | None:
    for cohort in REGISTRY:
        if cohort.kind == "repair" and tag in (
            cohort.supersedes_domains_of or ""
        ).split(","):
            return cohort
    return None


def unregistered_released_ledgers() -> list[str]:
    """Released job ledgers on disk that no registry entry accounts for."""

    known = {c.ledger for c in REGISTRY} | set(EXCLUDED)
    missing = []
    for path in sorted(ARTIFACTS.glob("*_jobs.json")):
        if path.name in known:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("released") and payload.get("runs"):
            missing.append(path.name)
    return missing
