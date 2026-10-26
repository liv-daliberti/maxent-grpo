# Can your AI find another way?

The homepage is the fully static Drupal article: the author’s revised TL;DR, one
worked graph-coloring puzzle with six solutions, and five PNG figures. It uses
ordinary HTML and has no CSS, scripts, controls, or expandable answers. The full
Python and pantry solution lists and other puzzle walkthroughs have been removed.
The study’s reported numbers and explanatory graphics are retained.

## Preview and Drupal

Run `python3 -m http.server 8000 --directory public`, then open http://localhost:8000.
The homepage and `reading.html` show the same unstyled article. Browser defaults
supply typography; Drupal will supply its own typography and image layout.
Paste `public/basic-html.html` into Drupal's source editor, upload the PNGs, and
replace image and paper URLs with Drupal's upload URLs. See
`public/drupal-editor-requirements.txt` for instructions. Complete and figures-only
ZIPs are available at `/drupal_static_package.zip` and `/drupal_static_figures.zip`.

## Render

Existing service: https://can-your-ai-find-another-way.onrender.com/

Repository: `liv-daliberti/maxent-grpo`, branch: `mode-collapse-blog`.
Build command: `node scripts/check-site.mjs`. Publish directory: `public`.
Automatic deployment is disabled in the existing Blueprint; manually deploy the
latest commit to update the running site.

Older companion pages and their assets remain at their existing URLs for prior
links. The static article does not load or link to them.
