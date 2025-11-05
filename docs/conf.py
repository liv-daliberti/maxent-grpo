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
# Avoid evaluating typing annotations (safer with mocked deps)
autodoc_typehints = 'none'
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

# Silence autosummary import noise for optional modules
suppress_warnings = [
    'autodoc.import_object',
]
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
