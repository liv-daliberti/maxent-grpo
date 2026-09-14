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
    # `kind` says what a cohort *is* for the figure. It used to also decide how
    # its ledger is *read*, and those are not the same question: a semantic arm
    # on an interactive domain has a semantic role but records progress in an
    # append-only metrics file rather than an oat run directory. Conflating them
    # made a running cohort report zero steps. Leave `reader` unset to derive it
    # from `kind`; set it when the two differ.
    reader: str | None = None
    # Which figure/table column this ledger populates, for cohorts read with the
    # point reader. Two consumers each kept their own hand-written map of these,
    # and a cohort wired into one but not the other raised a KeyError mid-refresh.
    domain_key: str | None = None
    # Some repair ledgers retain superseded cells for auditability. Consumers
    # monitor only the effective domains while the immutable ledger remains
    # untouched.
    excluded_domains: tuple[str, ...] = field(default_factory=tuple)

    def resolved_reader(self) -> str:
        """"point" reads an append-only metrics file; "static" reads a run dir."""

        if self.reader is not None:
            return self.reader
        return "point" if self.kind == "point_maze" else "static"

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

    # --- replay-side open-bank MaxEnt bundle ------------------------------
    # One new arm paired by domain/seed with E78's completed replay arm. It is
    # intentionally not folded into the fixed-dose semantic series: balancing
    # acts on length-normalized exemplar scores (not exact mode probability),
    # and the arm also includes target-free support expansion plus fresh-mode
    # replay priority.
    Cohort("e102", "E102 Qwen-0.5B   length-norm replay + score balance + explorer",
           "e102_full_open_bank_maxent_replay_05b_jobs.json", "semantic",
           "Qwen2.5-0.5B", "full_open_bank", "replay", plotted=False,
           not_plotted_because=(
               "a bundled mechanism arm against E78 replay; keep it visible in "
               "campaign_stats while its preregistered five-domain run resolves "
               "before deciding whether it belongs in the primary figure")),
    Cohort("e103", "E103 Qwen-0.5B   length-norm replay + score balance + fallback explorer",
           "e103_starvation_fallback_maxent_replay_05b_jobs.json", "semantic",
           "Qwen2.5-0.5B", "starvation_fallback", "replay", plotted=False,
           not_plotted_because=(
               "a discovery-reliability intervention whose primary paired "
               "comparator is E102; keep it in campaign_stats while the full "
               "five-domain cohort runs and audit it separately against E102")),
    Cohort("e108", "E108 Qwen-0.5B   admission-to-retention mechanism gate",
           "e108_admission_retention_mechanism_gate_jobs.json", "repair",
           reader="smoke"),

    # --- repaired conditional-entropy mechanism gate ----------------------
    # E104's three Python cells were superseded as one domain by E106 after
    # the boxed LaTeX-lambda admission bug was identified. Keep the immutable
    # E104 ledger intact, but report the effective 12 non-Python cells here so
    # campaign_stats does not double-count superseded Python work.
    Cohort("e104", "E104 all scales    v6 MaxEnt gate (non-Python)",
           "e104_group_centered_semantic_repair_three_scale_jobs.json",
           "repair", reader="smoke", excluded_domains=("python_factors",)),
    Cohort("e106", "E106 all scales    v6 MaxEnt Python repair",
           "e106_python_lambda_normalization_three_scale_jobs.json",
           "repair", reader="smoke"),
    # E106 Falcon/Python stopped before the pre-existing seed-55 discovery
    # horizon (historical first admission: step 179). E110 is the separately
    # registered 192-step effective replacement; the failed E106 cell remains
    # visible rather than being erased from campaign accounting.
    Cohort("e110", "E110 Falcon-1B    Python admission-horizon replacement",
           "e110_falcon_python_admission_horizon_jobs.json",
           "repair", reader="smoke"),
    # The v7 mechanism gate couples target-free verified support discovery to
    # uniform ReplayDr and the persistent verified-support entropy estimator.
    # It is registered prospectively so its held and live states are visible.
    Cohort("e111", "E111 all scales    verified-support discovery gate",
           "e111_verified_support_discovery_mechanism_gate_jobs.json",
           "repair", reader="smoke"),
    Cohort("e117r1", "E117-R1 same-plumbing C/P/F mechanism preflight",
           "e117r1_same_plumbing_component_preflight_jobs.json",
           "repair", reader="smoke"),
    Cohort("e117r2", "E117-R2 repaired same-plumbing C/P/F mechanism preflight",
           "e117r2_same_plumbing_component_preflight_jobs.json",
           "repair", reader="smoke"),
    Cohort("e117s1", "E117 Stage 1 development C/P/F efficacy screen",
           "e117_stage1_development_jobs.json",
           "repair", reader="smoke"),
    Cohort("e118", "E118 Qwen-0.5B/Falcon-1B/Qwen-3B MaxRL / Re:MaxRL factorial",
           "e118_all_scales_maxrl_verified_replay_jobs.json",
           "repair", reader="smoke"),
    Cohort("e119", "E119 Qwen-0.5B Level-2 Dr.GRPO / Re:Dr.GRPO / MaxRL / Re:MaxRL factorial",
           "e119_level2_qwen05b_factorial_jobs.json",
           "repair", reader="smoke"),
    Cohort("e120r1", "E120-R1 fresh-frequency replay ablation",
           "e120r1_frequency_weighted_replay_jobs.json",
           "repair", reader="smoke"),
    Cohort("e121", "E121 Qwen-0.5B Graph fixed-bank survival telemetry",
           "e121_fixed_bank_survival_telemetry_jobs.json",
           "repair", reader="smoke"),
    # Prospective Level-3 factorial; its planned cells remain visible while
    # admission and model selection are pending, before scheduler jobs exist.
    Cohort("e122", "E122 Qwen-0.5B Level-3 Dr.GRPO / Re:Dr.GRPO / MaxRL / Re:MaxRL factorial",
           "e122_level3_factorial_jobs.json",
           "repair", reader="smoke"),
    # Separate Qwen3B Level-3 factorial; its systems benchmark does not count
    # as science progress, and its held/released allocations remain distinct.
    Cohort("e123", "E123 Qwen-3B Level-3 Dr.GRPO / Re:Dr.GRPO / MaxRL / Re:MaxRL factorial",
           "e123_level3_factorial_jobs.json",
           "repair", reader="smoke"),
    # Conditional full efficacy successor to E105; preregistered before E111
    # terminalization and visible here before its release ledger exists.
    Cohort("e112", "E112 all scales    failed sampler-contract launch",
           "e112_verified_support_discovery_full_three_scale_jobs.json",
           "repair", plotted=False,
           not_plotted_because=(
               "retired before efficacy analysis after the non-Pantry sampler "
               "contract failed before step one")),
    Cohort("e112r1", "E112-R1 all scales corrected verified-support MaxEnt relaunch",
           "e112r1_verified_support_discovery_full_three_scale_jobs.json",
           "repair"),
    # The parser repair is shared by semantic MaxEnt and ReplayDr admission.
    # These fifteen Python-only ReplayDr cells remove that change from E105's
    # paired treatment effect instead of comparing repaired treatment to an
    # old-parser baseline.
    Cohort("e109", "E109 all scales    repaired Python ReplayDr comparators",
           "e109_repaired_python_replay_comparators_jobs.json", "repair"),
    # Dormant until its combined E104+E106 mechanism gate passes. Registering
    # before release prevents the full evaluation from silently disappearing
    # from campaign_stats when its ledger is created.
    Cohort("e105", "E105 all scales    v6 MaxEnt full evaluation",
           "e105_group_centered_semantic_repair_full_three_scale_jobs.json",
           "repair"),

    # --- plain GRPO baseline ----------------------------------------------
    # Single-arm: it is a *second control*, not a treatment. Every other control
    # in the campaign is Dr.GRPO, so this answers whether correct-mode collapse
    # depends on Dr.GRPO's debiasing or on group-relative binary reward as such.
    # Not plotted: Figure 4 draws paired treatment-minus-control differences and
    # this arm has no partner, so a curve would read as an effect estimate.
    Cohort("e95_3b", "E95  Qwen-3B     plain GRPO control",
           "e95_plain_grpo_Qwen25-3B_jobs.json", "paired", "Qwen2.5-3B",
           plotted=False,
           not_plotted_because=(
               "a second control with no paired treatment arm; Figure 4 shows "
               "paired differences, so plotting an unpaired baseline beside "
               "them would read as a treatment effect")),
    Cohort("e95_1b", "E95  Falcon-1B   plain GRPO control",
           "e95_plain_grpo_Falcon3-1B_jobs.json", "paired", "Falcon3-1B",
           plotted=False,
           not_plotted_because=(
               "a second control with no paired treatment arm; Figure 4 shows "
               "paired differences, so plotting an unpaired baseline beside "
               "them would read as a treatment effect")),
    Cohort("e95_05b", "E95  Qwen-0.5B   plain GRPO control",
           "e95_plain_grpo_Qwen25-05B_jobs.json", "paired", "Qwen2.5-0.5B",
           plotted=False,
           not_plotted_because=(
               "a second control with no paired treatment arm; Figure 4 shows "
               "paired differences, so plotting an unpaired baseline beside "
               "them would read as a treatment effect")),
    Cohort("e114", "E114 Qwen-3B     plain GRPO seed extension",
           "e114_plain_grpo_qwen3b_extension_jobs.json", "paired", "Qwen2.5-3B",
           plotted=False,
           not_plotted_because=(
               "the remaining seeds of the E95 second-control arm; direct "
               "paired endpoint figures combine E95 and E114 by cell")),

    # --- external comparative baselines ----------------------------------
    # Single-arm extensions on a three-domain subset. They remain visible in
    # campaign_stats, but are not figure overlays: each is read directly
    # against the completed E78 control rather than adding a proposed-method
    # curve to the five-domain primary panel.
    Cohort("e97", "E97  Qwen-0.5B   UCPO baseline",
           "e97_ucpo_05b_jobs.json", "paired", "Qwen2.5-0.5B"),
    Cohort("e98r1", "E98-R1 Qwen-0.5B sparse RLEP-Dr",
           "e98r1_sparse_rlep_dr_05b_jobs.json", "paired", "Qwen2.5-0.5B"),
    Cohort("e99", "E99  Falcon-1B   UCPO baseline",
           "e99_ucpo_falcon1b_jobs.json", "paired", "Falcon3-1B"),
    Cohort("e100", "E100 Falcon-1B   sparse RLEP-Dr",
           "e100_sparse_rlep_dr_falcon1b_jobs.json", "paired", "Falcon3-1B"),
    Cohort("e115_05b", "E115 Qwen-0.5B   UCPO domain extension",
           "e115_ucpo_qwen05b_domain_extension_jobs.json", "paired", "Qwen2.5-0.5B"),
    Cohort("e115_3b", "E115 Qwen-3B     UCPO baseline",
           "e115_ucpo_qwen3b_jobs.json", "paired", "Qwen2.5-3B"),
    Cohort("e116_05b", "E116 Qwen-0.5B   sparse RLEP-Dr extension",
           "e116_sparse_rlep_qwen05b_domain_extension_jobs.json", "paired", "Qwen2.5-0.5B"),
    Cohort("e116_3b", "E116 Qwen-3B     sparse RLEP-Dr",
           "e116_sparse_rlep_qwen3b_jobs.json", "paired", "Qwen2.5-3B"),
    # One direct DAPO arm spans both completed E78/E79 control families. Its
    # two smoke gates live in the same ledger but are operational checks, not
    # scientific cells; campaign_stats reports them separately from the 50
    # registered domain/seed cells.
    Cohort("e113", "E113 Qwen-0.5B/Falcon-1B DAPO baseline",
           "e113_dapo_direct_baseline_jobs.json", "paired", plotted=False,
           not_plotted_because=(
               "a prospective direct comparator paired against completed E78 "
               "and E79 controls; add it to result figures only after terminal "
               "audit")),
    # E113-R1 contains only two non-scientific Graph recovery smokes.  Its
    # empty runs list keeps it out of scientific totals, while the optional
    # smoke-gate reader makes pass/fail state explicit.
    Cohort("e113r1", "E113-R1 DAPO recovery smokes (non-scientific)",
           "e113r1_dapo_recovery_smoke_jobs.json", "paired", plotted=False,
           not_plotted_because=(
               "an operational recovery gate with zero scientific cells")),
    Cohort("e113r1m1", "E113-R1-M1 Qwen DAPO memory recovery (non-scientific)",
           "e113r1m1_qwen_memory_recovery_jobs.json", "paired", plotted=False,
           not_plotted_because=(
               "a one-job A6000 capacity repair with zero scientific cells")),
    # R2 is a preserved but closed prospective protocol: its required original
    # R1 gate failed, so it can never be released. R3 uses the separately
    # frozen effective Falcon+M1 gate and restores all 50 scientific cells.
    Cohort("e113r3", "E113-R3 Qwen-0.5B/Falcon-1B DAPO relaunch",
           "e113r3_dapo_full_relaunch_jobs.json", "paired", plotted=False,
           not_plotted_because=(
               "a full 50-cell direct-comparator relaunch gated on Falcon "
               "post-receipt completion plus exit-zero Qwen M1; plot only "
               "terminal paired cells")),
    # R4 permanently replaces R3 for every named-DAPO scientific comparison.
    Cohort("e113r4", "E113-R4 Qwen-0.5B/Falcon-1B official-verl DAPO",
           "e113r4_official_verl_dapo_jobs.json", "paired", plotted=False,
           not_plotted_because=(
               "the pinned unmodified upstream verl DAPO recipe with two "
               "operational smokes and 50 dependency-gated scientific cells; "
               "plot only terminal paired cells")),


    # --- PointMaze extensions --------------------------------------------
    Cohort("e78pm", "E78-PM Qwen-0.5B PointMaze",
           "e78pm_point_maze_verified_replay_only_05b_jobs.json",
           "point_maze", "Qwen2.5-0.5B",
           domain_key="point_maze"),
    Cohort("e79pm", "E79-PM Falcon-1B PointMaze",
           "e79pm_falcon_point_maze_verified_replay_jobs.json",
           "point_maze", "Falcon3-1B",
           domain_key="point_maze"),

    # --- PointMaze Tour, the redesigned sixth domain ----------------------
    # v1 (e78pm/e79pm) is inert: its control moves distinct@8 by +0.016 over
    # eight passes against a within-run checkpoint SD of 0.048, so it measures
    # nothing about retention. These cells are the admission ladder for the
    # replacement, not evidence for it.
    # The replacement sixth domain. Admitted on the phenomenon rather than the
    # registered .50 collapse threshold; the deviation and its reasoning are in
    # the preregistration, and the gate's own endpoint is reported regardless.
    # E93-PT, not E91-PT: in this registry a "-PM"/"-PT" suffix means the
    # PointMaze extension *of that experiment*, and this cohort does not derive
    # from E91 (Falcon adaptive MaxEnt). It takes the next free number.
    Cohort("e93pt", "E93-PT Qwen-0.5B PointMaze Tour",
           "e93pt_point_maze_tour_verified_replay_05b_jobs.json",
           "point_maze", "Qwen2.5-0.5B",
           domain_key="point_maze_tour"),

    # Second family on the identical release, so the two Tour cohorts differ in
    # model and prompt surface only. Its stage-1 distinct@8 of 2.125 sits below
    # the 2.5 admission floor; that deviation is registered, not discovered.
    # The arm PointMaze was excluded from. Read against `replay`, since it is
    # added on top of replay, not on top of the control.
    # A semantic arm by role, but its progress lives in a metrics file, so it
    # needs the point reader. This is the case the `reader` field exists for.
    Cohort("e96pt", "E96-PT Qwen-0.5B Tour replay + MaxEnt",
           "e96pt_point_maze_tour_semantic_maxent_jobs.json", "semantic",
           "Qwen2.5-0.5B", "semantic", "replay", plotted=False, reader="point",
           not_plotted_because=(
               "the Tour panel already carries control and replay; a third "
               "curve is added once the cohort completes and its registered "
               "rank-statistic prediction is resolved"),
           domain_key="point_maze_tour"),

    Cohort("e94pt", "E94-PT Falcon-1B PointMaze Tour",
           "e94pt_falcon_point_maze_tour_jobs.json",
           "point_maze", "Falcon3-1B",
           domain_key="point_maze_tour"),

    Cohort("tourgate", "TOUR  Qwen-0.5B  Tour admission gate",
           "point_maze_tour_gate_jobs.json", "point_maze", "Qwen2.5-0.5B",
           plotted=False,
           not_plotted_because=(
               "stage-2 collapse gate and stage-3 smoke on development maps; "
               "they decide whether the redesigned sixth domain earns a "
               "cohort, and plotting a domain-admission check beside measured "
               "treatment effects would read as an effect estimate"),
           domain_key=""),
)

# Released ledgers that are deliberately outside the registry, with the reason.
EXCLUDED: dict[str, str] = {
    "e118r2_maxrl_verified_replay_factorial_jobs.json":
        "source ledger represented by the registered 100-cell E118 aggregate",
    "e118q5_maxrl_verified_replay_extension_jobs.json":
        "source ledger represented by the registered 100-cell E118 aggregate",
    "e118q3_maxrl_verified_replay_extension_jobs.json":
        "source ledger represented by the registered 100-cell E118 aggregate",
    "e118f1_maxrl_verified_replay_extension_jobs.json":
        "source ledger represented by the registered 100-cell E118 aggregate",
    "e118r1_maxrl_verified_replay_factorial_jobs.json":
        "zero-step repair used a semantic wrapper that overrode neutral flags; "
        "superseded by explicit-variant E118-R2",
    "e118_maxrl_verified_replay_factorial_jobs.json":
        "zero-step launch failure from unregistered descriptive variant labels; "
        "superseded by the same-science E118-R1 variant-label repair ledger",
    "e80_qwen3b_aligned_verified_replay_jobs.json":
        "cancelled after a cosine-horizon error; superseded by e80r1 and "
        "retained as audit-only evidence",
    "e98_rlep_dr_05b_jobs.json":
        "failed its preregistered all-prompts replay-pool feasibility gate; "
        "superseded for live monitoring by the separately preregistered E98-R1 "
        "sparse repair and retained as audit-only failure evidence",
    "e88_adaptive_semantic_maxent_05b_jobs.json":
        "closed early on its registered mechanism-gate failure; superseded for "
        "live monitoring by E89 and retained as audit-only controller evidence",
    "e101_open_bank_countdown_pilot_jobs.json":
        "three-arm 128-update Countdown mechanism pilot, not a full campaign; "
        "retained as audit-only evidence for the E102 design",
    "e117_same_plumbing_component_preflight_jobs.json":
        "retired at zero runtime after its frozen node requirements were "
        "incompatible with the submit-routed partition; superseded by the "
        "same-science E117-R1 replacement ledger",
    "e117_stage1s3_superseded_storage_unsafe_jobs.json":
        "all 36 jobs canceled at zero runtime after measured checkpoint sizes "
        "showed unsafe aggregate storage demand; superseded by the S3 "
        "completion-serialized one-checkpoint replacement",
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


def point_family_ledgers(family: str) -> dict[str, Path]:
    """Figure column -> ledger, for every point-reader cohort of one family.

    Both the plotter and the interim-table builder need this. Each used to keep
    its own literal map, so a cohort added to one and not the other produced a
    KeyError only once the missing family had data.
    """

    out: dict[str, Path] = {}
    for cohort in REGISTRY:
        if cohort.family != family or cohort.resolved_reader() != "point":
            continue
        if cohort.domain_key is None:
            raise ValueError(f"{cohort.tag}: point cohort without a domain_key")
        if not cohort.domain_key:
            # Empty means "populates no figure column", which is how the
            # development-only admission gate is registered.
            continue
        # Several cohorts can share a column (arms of one domain); the first
        # registered one owns the paired control/replay curves.
        out.setdefault(cohort.domain_key, cohort.path())
    return out
