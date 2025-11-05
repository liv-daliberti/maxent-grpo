import os
import sys
from datetime import datetime

# Ensure project package is importable (for autodoc)
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))

project = 'Open R1'
author = 'Hugging Face + Liv d\'Aliberti'
copyright = f"{datetime.now().year}, {author}"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autosummary_generate = True
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock heavy deps so RTD builds without GPU stacks
autodoc_mock_imports = [
    'torch', 'transformers', 'trl', 'accelerate', 'datasets', 'peft', 'deepspeed',
    'bitsandbytes', 'vllm', 'wandb', 'numpy', 'requests', 'distilabel', 'huggingface_hub',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

