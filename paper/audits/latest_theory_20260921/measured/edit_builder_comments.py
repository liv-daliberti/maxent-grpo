from pathlib import Path
p=Path('/n/fs/similarity/maxent-grpo/ops/build_reference_kl_comparison.py');s=p.read_text();a=s.index('"""');b=s.index('"""',a+3)+3
s=s[:a]+'''"""Summarize reference-KL measurements and descriptive replay occupancy.

Terminal PCMD is a promptwise pair-collision complement computed from
verified responses. The control and replay arms use the existing 32-stream
resample; KL arms use their own disjoint evaluation draws.

Initial-reference PCMD describes the conditional target of the idealized
categorical stationary solution. It is not a bound on neural or transient
PCMD. Replay occupancy is the mean number of replayed keys per logged
update, averaged over runs. Its transform 1 - 1 / mean occupancy is a
descriptive summary, not a measured mean fixed-bank limit or a bound on
held-out neural diversity. Legacy JSON field names containing 'ceiling'
are retained for data compatibility; the paper labels these as summaries.
"""'''+s[b:]
s=s.replace('''# Re:Max carries its own bank, so it gets its own occupancy rather than
# borrowing Re:Dr's: MaxRL's larger coefficient near P = 0
# (Corollary "What a stronger fresh objective does buy") is a discovery
# advantage, and discovery is what fills a bank.''','''# Each replay objective has its own measured occupancy. The categorical
# small-P coefficient motivates a discovery hypothesis, not an identified
# cause of the empirical occupancy differences.''')
s=s.replace('''# An anchor cannot carry more breadth than its reference has, so what
# references actually have decides what anchoring can deliver. The hosted
# cohort is the measured answer for deployed models.''','''# Hosted diversity supplies additional descriptions of candidate references.
# It is not a ceiling on subsequent neural training with reference KL.''')
s=s.replace('''# Five seeds is the design. A coefficient still gathering them is marked
# where it stands rather than in a caveat several pages away, so no cell
# can be read at face value while it rests on one run.''','''# Superscripts distinguish coefficient cells with fewer than five seeds.''')
s=s.replace('''            # A frozen bound resting on a handful of prompts is not a bound.
            # PythonFactors' frozen policy solves too little to define one at
            # all, and a stray seed with two defined prompts would otherwise
            # report it as exactly zero.''','''            # Initial PCMD requires at least MIN_DEFINED eligible prompts.
            # A rare defined prompt does not make the domain summary reportable.''')
s=s.replace('''        # the three bounds, each beside the family it governs''','''        # Initial PCMD and two descriptive occupancy transforms.''')
s=s.replace('''    # Everything the prose says about this cohort is derived here rather than
    # typed, because these cells are still landing and a literal in the body
    # goes stale on the next seed.''','''    # Derive point-estimate comparison summaries from the same cells as the table.''')
s=s.replace('''    # A coefficient that outreaches the memory arms on breadth has done so
    # either for free or by spending correctness, and the two cases carry
    # opposite readings. Separating them is what keeps the withdrawal in
    # App. Q.6 from overstating the anchor's case in either direction.''','''    # Classify coefficients by whether they exceed the larger replay PCMD
    # estimate and also match that arm's pass@8 estimate. These comparisons
    # do not estimate significance or equalize total computation.''')
s=s.replace('''    # The anchor's best outright win: furthest breadth past both memory arms
    # at no cost in correctness. It is the strongest cell the anchor holds,
    # and it is reported with the seeds standing behind it.''','''    # Largest KL PCMD among cells exceeding the larger replay PCMD estimate
    # without a lower pass@8 point estimate for that comparator.''')
s=s.replace('''    # The narrowest reference in the cohort, which is where the flow's
    # correctness ceiling is furthest from what eight passes realize.''','''    # Smallest positive mean initial single-draw correctness. This differs
    # from pass@8 and is used only in a descriptive stationary plug-in.''')
s=s.replace('    # What an anchor pointed at a deployed model could inherit, per cell.','    # Reportable hosted reference descriptions, with their own eligibility sets.')
p.write_text(s)
