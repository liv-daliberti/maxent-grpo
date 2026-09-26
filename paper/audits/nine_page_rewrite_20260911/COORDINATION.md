# Concurrent rewrite coordination

This session is preserving the newer shared `paper/main.tex` assembled in
`paper/audits/main_story_20260911/`. It will not overwrite that draft or run
concurrent TeX builds in `paper/`. It is independently reviewing the final
scientific framing, source preservation, and rendered main-page boundary.

The active `ops/check_paper_main_length.py` now supersedes this session's
`check_paper_main_layout.py`; the latter and its tests are archived here
to avoid two competing validators. The original proposed prose and verified
numerical relocation fragment remain in this directory.

Independent review observed duplicate numerical relocation blocks; the newer
shared source now removes them and tightens the Figure 1 and partial-scale
wording. The root PDF was corrupted by the earlier overlapping TeX builds
(`pdftotext` reports invalid page objects), so a fresh isolated build is needed.
This session is checking a frozen copy under `/tmp/nine-page-review-*` and
will promote only validated artifacts that still match the shared source.
The superseded layout helper's Makefile dependency has also been removed.
