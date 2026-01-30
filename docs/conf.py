"""
Sphinx configuration for the project documentation.

This configuration keeps imports light and mocks heavy optional dependencies so
docs build without GPU stacks. It prefers modern themes (Furo/PyData) with
graceful fallback, enables autosummary/autodoc with Google/NumPy style parsing,
and adds a few quality‑of‑life tweaks for rendering and navigation.

License
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the specific language governing permissions and
limitations under the License.
"""

import os
import sys
from datetime import datetime
from types import ModuleType

# Ensure project package is importable (for autodoc),
# independent of the current working directory used to invoke Sphinx.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

_AUTOSUMMARY_DIR = os.path.join(_HERE, "_autosummary")
os.makedirs(_AUTOSUMMARY_DIR, exist_ok=True)


def _ensure_stub(module_name: str) -> ModuleType:
    """Create a lightweight stub module when optional deps are missing."""
    module = sys.modules.get(module_name)
    if module is None:
        module = ModuleType(module_name)
        module.__file__ = __file__
        sys.modules[module_name] = module
    return module


def _make_stub_class(module_name: str, name: str, attrs: dict | None = None) -> type:
    attrs = attrs or {}
    cls = type(name, (), attrs)
    cls.__module__ = module_name
    return cls


# Provide simple stand-ins for optional dependencies so Sphinx can resolve
# inheritance references and type annotations even when the real packages are
# not installed in the doc build environment.
trl_mod = _ensure_stub("trl")
trl_mod.ScriptArguments = getattr(
    trl_mod, "ScriptArguments", _make_stub_class("trl", "ScriptArguments")
)
trl_mod.GRPOConfig = getattr(
    trl_mod, "GRPOConfig", _make_stub_class("trl", "GRPOConfig")
)
trl_mod.ModelConfig = getattr(
    trl_mod, "ModelConfig", _make_stub_class("trl", "ModelConfig")
)
trl_mod.GRPOTrainer = getattr(trl_mod, "GRPOTrainer", type("GRPOTrainer", (), {}))


def _trl_dummy(*args, **kwargs):
    return None


trl_mod.get_peft_config = getattr(trl_mod, "get_peft_config", _trl_dummy)
trl_mod.get_kbit_device_map = getattr(trl_mod, "get_kbit_device_map", _trl_dummy)
trl_mod.get_quantization_config = getattr(
    trl_mod, "get_quantization_config", _trl_dummy
)
trl_mod.TrlParser = getattr(
    trl_mod,
    "TrlParser",
    type("TrlParser", (), {"__call__": lambda self, *a, **k: None}),
)

accelerate_mod = _ensure_stub("accelerate")
accelerate_mod.Accelerator = getattr(
    accelerate_mod, "Accelerator", type("Accelerator", (), {})
)
acc_utils = _ensure_stub("accelerate.utils")
acc_utils.is_peft_model = getattr(
    acc_utils, "is_peft_model", lambda *_args, **_kwargs: False
)
# Ensure submodule imports like ``accelerate.accelerator`` resolve to a stub too.
accelerator_submod = _ensure_stub("accelerate.accelerator")
if not hasattr(accelerator_submod, "Accelerator"):
    accelerator_submod.Accelerator = accelerate_mod.Accelerator

transformers_mod = _ensure_stub("transformers")


class _BaseStub:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def save_pretrained(self, *args, **kwargs):
        return None


transformers_mod.PreTrainedModel = getattr(
    transformers_mod, "PreTrainedModel", type("PreTrainedModel", (_BaseStub,), {})
)
try:
    transformers_mod.PreTrainedModel.__module__ = "transformers"
except Exception:
    pass
transformers_mod.PreTrainedTokenizerBase = getattr(
    transformers_mod,
    "PreTrainedTokenizerBase",
    type(
        "PreTrainedTokenizerBase",
        (_BaseStub,),
        {"pad_token_id": None, "eos_token_id": None},
    ),
)
try:
    transformers_mod.PreTrainedTokenizerBase.__module__ = "transformers"
except Exception:
    pass
transformers_mod.PreTrainedTokenizer = getattr(
    transformers_mod,
    "PreTrainedTokenizer",
    type(
        "PreTrainedTokenizer",
        (_BaseStub,),
        {"pad_token_id": None, "eos_token_id": None},
    ),
)
try:
    transformers_mod.PreTrainedTokenizer.__module__ = "transformers"
except Exception:
    pass
transformers_mod.AutoConfig = getattr(
    transformers_mod, "AutoConfig", type("AutoConfig", (_BaseStub,), {})
)
try:
    transformers_mod.AutoConfig.__module__ = "transformers"
except Exception:
    pass
transformers_mod.AutoTokenizer = getattr(
    transformers_mod, "AutoTokenizer", type("AutoTokenizer", (_BaseStub,), {})
)
try:
    transformers_mod.AutoTokenizer.__module__ = "transformers"
except Exception:
    pass
transformers_mod.AutoModelForCausalLM = getattr(
    transformers_mod,
    "AutoModelForCausalLM",
    type("AutoModelForCausalLM", (_BaseStub,), {}),
)
try:
    transformers_mod.AutoModelForCausalLM.__module__ = "transformers"
except Exception:
    pass
trainer_utils_mod = _ensure_stub("transformers.trainer_utils")
trainer_utils_mod.get_last_checkpoint = getattr(
    trainer_utils_mod, "get_last_checkpoint", lambda *a, **k: None
)
transformers_mod.trainer_utils = trainer_utils_mod  # type: ignore[attr-defined]
tf_utils_mod = _ensure_stub("transformers.utils")
tf_utils_mod.logging = getattr(
    tf_utils_mod,
    "logging",
    type(
        "logging",
        (),
        {
            "set_verbosity": staticmethod(lambda *a, **k: None),
            "enable_default_handler": staticmethod(lambda *a, **k: None),
            "enable_explicit_format": staticmethod(lambda *a, **k: None),
        },
    ),
)
transformers_mod.utils = tf_utils_mod  # type: ignore[attr-defined]
transformers_mod.set_seed = getattr(transformers_mod, "set_seed", lambda *a, **k: None)

distilabel_mod = _ensure_stub("distilabel")
pipeline_mod = _ensure_stub("distilabel.pipeline")
pipeline_mod.Pipeline = getattr(pipeline_mod, "Pipeline", type("Pipeline", (), {}))
distilabel_mod.pipeline = pipeline_mod  # type: ignore[attr-defined]
steps_mod = _ensure_stub("distilabel.steps")
tasks_mod = _ensure_stub("distilabel.steps.tasks")
tasks_mod.TextGeneration = getattr(
    tasks_mod, "TextGeneration", type("TextGeneration", (), {})
)
steps_mod.tasks = tasks_mod  # type: ignore[attr-defined]
distilabel_mod.steps = steps_mod  # type: ignore[attr-defined]
llms_mod = _ensure_stub("distilabel.llms")
llms_mod.OpenAILLM = getattr(llms_mod, "OpenAILLM", type("OpenAILLM", (), {}))
distilabel_mod.llms = llms_mod  # type: ignore[attr-defined]

sys.modules.setdefault("distilabel", distilabel_mod)
sys.modules.setdefault("distilabel.pipeline", pipeline_mod)

project = "Open R1"
author = "Hugging Face + Liv d'Aliberti"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    # Core Sphinx features we rely on
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
]

source_suffix = {
    ".rst": "restructuredtext",
}

# Optional, nicer extensions. Import-guard them so a lightweight pre-commit
# environment without optional packages doesn't hard-fail the docs build.
_optional_exts = [
    "sphinx_copybutton",
    "sphinx_design",
]
for _ext in _optional_exts:
    try:
        import importlib

        importlib.import_module(_ext)
        extensions.append(_ext)
    except Exception:
        # Missing optional dependency; skip enabling the extension.
        pass

autosummary_generate = True
autosummary_generate_overwrite = True
# Avoid evaluating typing annotations (safer with mocked deps)
# Render type hints in the description (cleaner signatures on RTD)
autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "AutoModelForCausalLM,AutoTokenizer,PreTrainedTokenizer",
}
autodoc_type_aliases = {
    "CompletionType": "maxent_grpo.rewards.basic.CompletionType",
    "RewardFunction": "maxent_grpo.rewards.basic.RewardFunction",
    "DataLoader": "maxent_grpo.training.types.DataLoader",
    "GenerationFn": "maxent_grpo.training.types.GenerationFn",
    "GenerationBatch": "maxent_grpo.training.types.GenerationBatch",
    "RewardComputation": "maxent_grpo.training.types.RewardComputation",
    "RewardSpec": "maxent_grpo.training.types.RewardSpec",
    "SeedAugmentationConfig": "maxent_grpo.training.runtime.SeedAugmentationConfig",
    "DatasetMixtureConfig": "maxent_grpo.config.dataset.DatasetMixtureConfig",
    "GRPOConfig": "maxent_grpo.config.grpo.GRPOConfig",
    "GRPOScriptArguments": "maxent_grpo.config.grpo.GRPOScriptArguments",
    "RewardConfig": "maxent_grpo.rewards.basic.RewardConfig",
    "GenerationServiceError": "maxent_grpo.generation.errors.GenerationServiceError",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

myst_enable_extensions = [
    # Keep lightweight: only enable extensions we ship and don't require
    # optional linkify dependencies in the pre-commit environment.
    "colon_fence",
]
myst_heading_anchors = 3

# Enable MyST linkify only when its runtime deps are available (linkify-it-py + mdurl).
try:
    import importlib

    importlib.import_module("linkify_it")
    importlib.import_module("mdurl")
    myst_enable_extensions.append("linkify")
except Exception:
    # Leave linkify disabled in minimal environments.
    pass

# Allow section labels across files without collisions
autosectionlabel_prefix_document = True

# Cross-project links to popular libraries
intersphinx_mapping = {
    # For Sphinx >= 8, the second item must be a string or None
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
# Avoid network lookups during local/pre-commit builds; enable online on RTD or when explicitly requested
_ONLINE_DOCS = (os.environ.get("READTHEDOCS", "").lower() in {"true", "1"}) or (
    os.environ.get("SPHINX_ONLINE", "").lower() in {"true", "1"}
)
if not _ONLINE_DOCS:
    intersphinx_mapping = {}

nitpick_ignore = [
    ("py:class", "torch.device"),
    ("py:class", "types.ModuleType"),
]

# Mock heavy deps so RTD builds without GPU stacks
autodoc_mock_imports = [
    "torch",
    "datasets",
    "peft",
    "deepspeed",
    "bitsandbytes",
    "vllm",
    "wandb",
    "numpy",
    "requests",
    "huggingface_hub",
]

templates_path = ["_templates"]
exclude_patterns = []


def _choose_theme():
    """Prefer a modern theme with graceful fallback.

    Order: Furo → PyData → RTD → Alabaster.
    """
    try:
        import furo  # noqa: F401

        return "furo", {
            "light_css_variables": {
                "color-brand-primary": "#7c4dff",
                "color-brand-content": "#7c4dff",
                "font-stack": "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Ubuntu, Cantarell, 'Noto Sans', 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji'",
                "font-stack--monospace": "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
            },
            "dark_css_variables": {
                "color-brand-primary": "#b388ff",
                "color-brand-content": "#b388ff",
            },
        }
    except Exception:
        try:
            import pydata_sphinx_theme  # noqa: F401

            return "pydata_sphinx_theme", {
                "logo": {
                    "text": project,
                },
                "navbar_center": ["navbar-nav"],
                "header_links_before_dropdown": 6,
                "use_edit_page_button": False,
            }
        except Exception:
            try:
                import sphinx_rtd_theme  # noqa: F401

                return "sphinx_rtd_theme", {
                    "style_nav_header_background": "#7c4dff",
                    "collapse_navigation": False,
                }
            except Exception:
                return "alabaster", {
                    "description": "Clean baseline docs",
                    "page_width": "980px",
                    "fixed_sidebar": True,
                }


html_theme, html_theme_options = _choose_theme()

html_title = f"{project} · Developer Docs"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
pygments_style = "friendly"
pygments_dark_style = "monokai"

# Silence autosummary import noise for optional modules
suppress_warnings = [
    "autodoc.import_object",
]

# Make TODOs visible in the rendered docs (fun callouts)
todo_include_todos = True

# Default to dark mode when available (for Furo)
html_context = {
    "default_mode": "dark",
}

# Keep the landing page snappy on RTD by limiting autosummary depth in nav
html_theme_options.setdefault("navigation_with_keys", True)


_DUPLICATE_EXPORTS = {
    # Skip re-exported names in package-level modules to avoid ambiguous refs.
    "maxent_grpo.config": {
        "DatasetConfig",
        "DatasetMixtureConfig",
        "GRPOConfig",
        "GRPOScriptArguments",
        "ScriptArguments",
    },
    "maxent_grpo.rewards": {
        "RewardConfig",
    },
    "maxent_grpo.generation.vllm": {
        "GenerationServiceError",
        "VLLMServiceError",
    },
}


def _skip_external_members(app, what, name, obj, skip, options):  # noqa: D401
    """Skip documenting external or duplicate members with noisy docstrings."""

    module_name = getattr(obj, "__module__", "") or ""
    if module_name.startswith("accelerate."):
        return True
    current_module = app.env.temp_data.get("autodoc:module") if app.env else None
    if current_module in _DUPLICATE_EXPORTS and name in _DUPLICATE_EXPORTS[current_module]:
        return True
    return skip


def setup(app):  # type: ignore[override]
    app.connect("autodoc-skip-member", _skip_external_members)
