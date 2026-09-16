# Verified replay: theoretical extensions

This standalone supplement contains complete conditional proofs, literature attributions, and limitations for optimizer-aware retention, adaptive admission, exact entropy, per-mode certificates, and exact categorical natural gradient.

Build from this directory with a standard TeX installation:

    pdflatex -interaction=nonstopmode -halt-on-error theory_extensions.tex
    pdflatex -interaction=nonstopmode -halt-on-error theory_extensions.tex

The master contains its bibliography; BibTeX is not needed. The five input fragments are included. survival_body.tex is the body extracted from the separately compilable survival.tex, with its section headings lowered one level. optimizer.bib is supplied as reusable verified metadata. The derivation notes contain additional detail and code-to-assumption checks.

The three Python verification scripts use NumPy and SciPy. Their saved JSON results are numerical corroboration, not substitutes for proofs and not training measurements.

These results do not certify the current AdamW/PPO learner. See the assumptions and application limitations in each section.
