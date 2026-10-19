# Can your AI find another way?

A blog post about mode collapse, with five HTML-only puzzles. Native details/summary elements provide choices, hints, feedback, and complete answer keys without JavaScript. The discovery graph is a static figure with an expandable table of exact values.

View the post at https://can-your-ai-find-another-way.onrender.com/ after deploying this commit. The article includes the requested closing sentence about high rewards masking repetitive solutions.

## Deploy on Render

[Deploy to Render](https://render.com/deploy?repo=https://github.com/liv-daliberti/maxent-grpo/tree/mode-collapse-blog)

The Blueprint deploys a static site from `public/`. There are no runtime dependencies or API keys. The build command checks the pages, local assets, and paper PDF. Automatic deployment is disabled; deploy the latest commit explicitly when updating the post.

Manual setup: select repository `liv-daliberti/maxent-grpo`, branch `mode-collapse-blog`, service type **Static Site**, build command `node scripts/check-site.mjs`, publish directory `public`.

## Preview

Run `python3 -m http.server 8000 --directory public`, then open http://localhost:8000.

The article at `public/index.html` works with JavaScript disabled. Its CSS is optional presentation; the puzzle reveals also work without CSS. `public/paper.pdf` is the accompanying paper, and figures are in `public/figures/`.

The older JavaScript companion pages remain at their existing URLs for prior links. The current article does not load or link to them. The old `tests/check_activity.py`, `tests/check_games.py`, and `tests/check_totals.py` describe those earlier controls; the deployment check in `scripts/check-site.mjs` verifies the new article contract and all local references.
