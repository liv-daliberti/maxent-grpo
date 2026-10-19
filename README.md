# Can your AI find another way?

The article now includes five playable HTML + CSS puzzles with no runtime JavaScript: graph coloring, arithmetic, factor rules, algebra routes and pantry planning. A horizontal row of native radio tabs switches between games and keeps their progress. Native controls update the diagrams and feedback; readers can save distinct valid solutions to a collection and reset each puzzle. Figures 2, 3 and 4 use the original artwork, including the feedback loop’s return arrow. The exact-average comparison dropdown has been removed; the sampling figure remains. The article closes with practical examples of why alternative solutions matter.

## Preview and Drupal

Run `python3 -m http.server 8000 --directory public`, then open http://localhost:8000. JavaScript can be disabled. `public/drupal-theme.css` is required for puzzle behavior, including validation and discovery counts. `public/reading.html` contains the illustrated reading alternative with answer keys. See `public/drupal-editor-requirements.txt` for the exact Drupal tag, attribute, editor and theme requirements. The target Drupal installation has not yet been tested.

## Render

Existing service: https://can-your-ai-find-another-way.onrender.com/

Repository: `liv-daliberti/maxent-grpo`, branch: `mode-collapse-blog`. Build command: `node scripts/check-site.mjs`. Publish directory: `public`. Automatic deployment is disabled in the existing Blueprint; manually deploy the latest commit to update the running site.

The older JavaScript companion pages remain at their existing URLs for prior links. The article does not load or link to them. All article interactions use HTML controls and CSS selectors; no runtime dependencies or API keys are required.
