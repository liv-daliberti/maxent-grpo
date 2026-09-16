# Anonymous workshop footer correction — September 6, 2026

The official local style defines `\@trackname` for `dblblindworkshop` at
`neurips_2026.sty:90–92`, but only uses it in final mode at lines 396–399. The
anonymous notice at lines 401–403 ignores the workshop option and hardcodes the
generic NeurIPS conference submission notice. Therefore, the existing public
`\workshoptitle` setting alone cannot fix this particular anonymous-mode defect.
The official template example uses the same options and title setting. No public
package option selects an anonymous workshop-specific notice in this version.

## Exact candidate

`footer-preamble.tex` is the current workshop preamble plus the following block,
inserted immediately after the existing `\workshoptitle` declaration:

```tex
% The official style omits its workshop title from the anonymous notice.
% Keep the upstream style, anonymity and line numbering; correct only this text.
\makeatletter
\if@neuripsfinal\else
  \renewcommand{\@noticestring}{%
    Submitted to \@workshoptitle\ (NeurIPS \@neuripsyear). Do not distribute.%
  }
\fi
\makeatother
```

The anonymous footer reads:

> Submitted to The 6th Workshop on Mathematical Reasoning and AI (NeurIPS 2026). Do not distribute.

This is a narrow preamble override of the internal notice macro, not a separate
public style option. It leaves the official `.sty` byte-identical, retains
`dblblindworkshop`, author anonymization, review line numbering, fonts, margins,
and the notice-box layout. Only the preamble's manifest binding and rebuilt
receipt need updating. The existing official-style hash guard remains valid.

## Isolated verification

Compiled both modes using the actual preamble and copied local styles in
`/tmp/mathai-footer-probe-n_nyxl_h/`; no active manuscript or build was changed.
Both compilations succeeded with no overfull diagnostic.

- Anonymous mode reports `ANONYMOUS YES` and `LINE-NUMBERS ON`. Extracted output
  contains `Anonymous Author(s)` and the exact new workshop notice, and lacks
  the old generic `Submitted to 40th Conference` notice.
- Camera-ready mode reports `ANONYMOUS NO` and `LINENO ABSENT`, shows the probe's
  named author, and retains the official track notice: `40th Conference on
  Neural Information Processing Systems (NeurIPS 2026). Workshop: The 6th Workshop
  on Mathematical Reasoning and AI.` It does not acquire `Submitted to` wording.

Retained probe logs and extracted PDF text are `footer-anonymous-build.log`,
`footer-camera-ready-build.log`, `footer-anonymous.txt`, and
`footer-camera-ready.txt` beside this audit.

## Minimal submission check

After the existing anonymous-author check in `check_submission.py`, normalize
first-page whitespace and require the corrected notice while rejecting the old
one. For example:

```python
first_page = ' '.join(pages[0].split())
require('Submitted to The 6th Workshop on Mathematical Reasoning and AI '
        '(NeurIPS 2026). Do not distribute.' in first_page,
        'missing MATH-AI workshop submission notice')
require('Submitted to 40th Conference' not in first_page,
        'generic NeurIPS conference notice remains')
```

This guards the compiled artifact, and the existing anonymous-author assertion
prevents substituting final mode merely to obtain a workshop footer. The full
paper build should still enforce four content pages and Figure 1 on page 1 after
integration, since even a footer wording change can alter available page space.

Official style SHA256 (unchanged): `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`.

Verified at 2026-09-06T18:49:49.835825+00:00.
