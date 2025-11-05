"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
from datetime import datetime

# Ensure project package is importable (for autodoc),
# independent of the current working directory used to invoke Sphinx.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'src')))

project = 'Open R1'
author = 'Hugging Face + Liv d\'Aliberti'
copyright = f"{datetime.now().year}, {author}"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    # Quality-of-life extensions
    'sphinx.ext.autosectionlabel',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
]

autosummary_generate = True
autosummary_generate_overwrite = True
# Avoid evaluating typing annotations (safer with mocked deps)
autodoc_typehints = 'none'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# MyST / Markdown configuration
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
myst_enable_extensions = [
    'colon_fence',
    'linkify',
]
myst_heading_anchors = 3

# Allow section labels across files without collisions
autosectionlabel_prefix_document = True

# Cross-project links to popular libraries
intersphinx_mapping = {
    # For Sphinx >= 8, the second item must be a string or None
    'python': ('https://docs.python.org/3', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
# Avoid network lookups during local/pre-commit builds; enable online on RTD or when explicitly requested
_ONLINE_DOCS = (os.environ.get('READTHEDOCS', '').lower() in {'true', '1'}) or (
    os.environ.get('SPHINX_ONLINE', '').lower() in {'true', '1'}
)
if not _ONLINE_DOCS:
    intersphinx_mapping = {}

# Mock heavy deps so RTD builds without GPU stacks
autodoc_mock_imports = [
    'torch', 'transformers', 'trl', 'accelerate', 'datasets', 'peft', 'deepspeed',
    'bitsandbytes', 'vllm', 'wandb', 'numpy', 'requests', 'distilabel', 'huggingface_hub',
]

templates_path = ['_templates']
exclude_patterns = []


def _choose_theme():
    """Prefer a modern theme with graceful fallback.

    Order: Furo → PyData → RTD → Alabaster.
    """
    try:
        import furo  # noqa: F401
        return 'furo', {
            'light_css_variables': {
                'color-brand-primary': '#7c4dff',
                'color-brand-content': '#7c4dff',
                'font-stack': "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Ubuntu, Cantarell, 'Noto Sans', 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji'",
                'font-stack--monospace': "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
            },
            'dark_css_variables': {
                'color-brand-primary': '#b388ff',
                'color-brand-content': '#b388ff',
            },
        }
    except Exception:
        try:
            import pydata_sphinx_theme  # noqa: F401
            return 'pydata_sphinx_theme', {
                'logo': {
                    'text': project,
                },
                'navbar_center': ['navbar-nav'],
                'header_links_before_dropdown': 6,
                'use_edit_page_button': False,
            }
        except Exception:
            try:
                import sphinx_rtd_theme  # noqa: F401
                return 'sphinx_rtd_theme', {
                    'style_nav_header_background': '#7c4dff',
                    'collapse_navigation': False,
                }
            except Exception:
                return 'alabaster', {
                    'description': 'Clean baseline docs',
                    'page_width': '980px',
                    'fixed_sidebar': True,
                }


html_theme, html_theme_options = _choose_theme()

html_title = f"{project} · Developer Docs"
html_static_path = ['_static']
html_css_files = ['custom.css']
pygments_style = 'friendly'
pygments_dark_style = 'monokai'

# Silence autosummary import noise for optional modules
suppress_warnings = [
    'autodoc.import_object',
]

# Make TODOs visible in the rendered docs (fun callouts)
todo_include_todos = True

# Default to dark mode when available (for Furo)
html_context = {
    'default_mode': 'dark',
}

# Keep the landing page snappy on RTD by limiting autosummary depth in nav
html_theme_options.setdefault('navigation_with_keys', True)
