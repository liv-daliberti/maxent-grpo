from pathlib import Path
root=Path('/n/fs/similarity/maxent-grpo')
p=root/'ops/plot_paper_reference_kl_knee.py'
s=p.read_text()
a=s.index('"""');b=s.index('"""',a+3)+3
s=s[:a]+'''"""Reference-KL coefficient sweeps and categorical reference summaries.

Each domain panel shows seed-mean terminal PCMD and pass@8 on the same
[0, 1] scale. Horizontal rules show initial PCMD, when reportable, and Re:Dr
PCMD. The vertical rule substitutes mean initial single-draw correctness
into the categorical beta-star formula; it is neither a fitted knee nor a
prediction for pass@8. Green and red identify the two sides of this plug-in
threshold. Reference PCMD is not an upper bound on neural checkpoints.
"""'''+s[b:]
s=s.replace('''# Below beta* the flow's correctness identity still puts stationary
# correctness above one half; above it, the coefficient is spending
# correctness. The two washes say which side of that a coefficient falls on,
# so the verdict is read off the panel rather than off the caption.''','''# Shading encodes only the categorical plug-in stationary calculation.
# Domain-average initial correctness is not a promptwise prediction, and
# pass@8 is a different estimand from stationary single-draw correctness.''')
s=s.replace('CEILING = style.MUTED         # the frozen-breadth rule','CEILING = style.MUTED         # initial PCMD, not a neural upper bound')
s=s.replace('KNEE = style.ADD_ON           # the predicted knee','KNEE = style.ADD_ON           # categorical plug-in threshold')
s=s.replace("# the anchor's own bound, and the memory arm it has to overtake",'# initial-policy PCMD and the Re:Dr comparison')
s=s.replace('''        # Drawn wherever it falls inside the axis, including well past the
        # swept coefficients: a rule to the right of the last point is the
        # prediction that this domain's correctness has not turned yet, and
        # is the reason the axis runs past the data.''','''        # The threshold may fall beyond the measured coefficients. The axis
        # displays the plug-in calculation without extrapolating the data.''')
s=s.replace('label="PCMD, the anchor"','label="reference-KL PCMD"')
s=s.replace('label="pass@8, the anchor"','label="reference-KL pass@8"')
s=s.replace('label="frozen PCMD (the anchor\'s bound)"','label="initial-policy PCMD"')
s=s.replace('label=r"predicted $\\beta^\\ast$"','label=r"categorical $\\beta^\\star$"')
p.write_text(s)
p=root/'ops/plot_paper_reference_kl_plane.py'
s=p.read_text();a=s.index('"""');b=s.index('"""',a+3)+3
s=s[:a]+'''"""Terminal correctness and diversity for reference KL and replay.

The panel equally averages the domains with reportable initial PCMD and
uses coefficients available in every included domain. Replay and MaxRL
points use the same domain set. Lines connect measured means; the colored
regions extend a piecewise-linear guide and are not confidence regions,
attainable frontiers, or causal comparisons. Initial-policy rules describe
a reference, not upper bounds on neural performance. Only one replay dose
is represented, so this figure does not compare dose sensitivity.
"""'''+s[b:]
s=s.replace('from matplotlib.lines import Line2D  # noqa: E402\nfrom matplotlib.lines import Line2D  # noqa: E402','from matplotlib.lines import Line2D  # noqa: E402')
s=s.replace('''# The anchor's curve is a frontier, and the two washes say which side of it a
# point falls on. Same pair as the knee plate, so one reading carries across.''','''# Shading marks the two sides of an interpolated visual guide. It does not
# describe statistical confidence or unmeasured attainable policies.''')
s=s.replace('''    # A base-model bound has to be measurable to be drawn. The resample's own
    # bar is thirty defined prompts, and a frozen policy that rarely succeeds
    # clears it nowhere: Countdown and PythonFactors are excluded here for that
    # reason, not for their results. Their curves are in the per-domain plate.''','''    # Initial PCMD is displayed only with at least thirty defined prompts.
    # This selects Graph, MathIR, and Pantry; the other two domains remain in
    # the domain-level table and coefficient figure.''')
s=s.replace('    # every coefficient the whole set carries, so the curve grows as cells land','    # Use the intersection of measured coefficients across the included domains.')
s=s.replace('''    # The compute-matched control is this family at beta = 0: same objective,
    # same bank scaffolding with a zero derivative, one knob unturned. It opens
    # the curve rather than standing apart from it.''','''    # The beta = 0 point is the control with a zero replay derivative.
    # Optimizer updates and fresh rollouts match, but total compute is not
    # equalized or reallocated to effective baseline training.''')
s=s.replace('''    # PCMD rises monotonically with beta, so the curve is a function of
    # breadth: at each breadth level it gives the pass@8 the anchor manages
    # there. Green is the side that beats it at the same breadth, red the side
    # that does not. Past the swept range the endpoints are simply held, which
    # is why the wash extends to the frame without the curve doing so.''','''    # The current common-domain means increase in PCMD with beta. Shading
    # follows their linear interpolation and holds the endpoints outside the
    # measured PCMD range. It is a visual guide, not an attainable frontier.
    # This aggregate ordering need not hold within individual domains.''')
s=s.replace('label="base model accuracy"','label="base model pass@8"')
p.write_text(s)
