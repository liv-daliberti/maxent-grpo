.PHONY: help install-dev install-local ensure-path venv install-venv conda-local lint format test precommit docs docs-clean clean-local validate-logs

VAR_DIR := $(CURDIR)/var
export RUFF_CACHE_DIR := $(VAR_DIR)/cache/ruff

help:
	@echo "Targets:"
	@echo "  install-dev  - pip install -e .[dev]"
	@echo "  install-local- user base under $(PWD)/.local"
	@echo "  ensure-path  - append $(PWD)/.local/bin to your shell rc"
	@echo "  conda-local  - create local conda env at ./var/openr1 via configs/environment.yml"
	@echo "  venv         - create .venv in repo"
	@echo "  install-venv - install project into .venv"
	@echo "  lint         - ruff check . && pylint src"
	@echo "  format       - ruff check --fix . && isort ."
	@echo "  test         - pytest -q -c configs/pytest.ini"
	@echo "  precommit    - pre-commit run -a"
	@echo "  docs         - build Sphinx HTML to var/docs/_build/html"
	@echo "  docs-clean   - remove var/docs/_build"
	@echo "  clean-local  - remove local envs/caches in this repo"

install-dev:
	pip install -e .[dev]

install-local:
	export PYTHONUSERBASE=$(PWD)/.local; \
	  python -m pip install --upgrade pip; \
	  pip install --user -e .[dev]
	@echo "Add to PATH for local scripts: export PATH=\"$(PWD)/.local/bin:$$PATH\""

ensure-path:
	bash ops/tools/ensure_local_path.sh --apply

venv:
	python -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install-venv: venv
	. .venv/bin/activate && pip install -e .[dev]

conda-local:
	@set -euo pipefail; \
	# Avoid alias conflict between CONDA_ENVS_DIRS and CONDA_ENVS_PATH (cluster sets PATH);
	# conda 24+ errors if both are present. Prefer CONDA_ENVS_DIRS.
	unset CONDA_ENVS_PATH; \
		ROOT_DIR="$(CURDIR)"; \
		VAR_DIR="$(VAR_DIR)"; \
	echo "Creating local conda env under $$VAR_DIR/openr1"; \
	CONDARC="$$ROOT_DIR/.condarc"; \
	CONDA_PKGS_DIRS="$$VAR_DIR/conda/pkgs"; \
	CONDA_ENVS_DIRS="$$VAR_DIR/conda/envs"; \
	PIP_CACHE_DIR="$$VAR_DIR/cache/pip"; \
	HF_HOME="$$VAR_DIR/cache/huggingface"; \
	PIP_CONFIG_FILE="$$VAR_DIR/pip/pip.conf"; \
	TMPDIR="$$VAR_DIR/tmp"; \
	ACTIVATE_HOOK="$$VAR_DIR/openr1/etc/conda/activate.d/00-local-paths.sh"; \
	DEACTIVATE_HOOK="$$VAR_DIR/openr1/etc/conda/deactivate.d/00-local-paths.sh"; \
	mkdir -p "$$VAR_DIR" "$$CONDA_PKGS_DIRS" "$$CONDA_ENVS_DIRS" "$$PIP_CACHE_DIR" "$$HF_HOME" "$$TMPDIR" "$$VAR_DIR/pip"; \
	if [ -f /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh ]; then \
	  . /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh; \
	elif command -v conda >/dev/null 2>&1; then \
	  eval "$(conda shell.bash hook)"; \
	else \
	  echo "Conda not found on PATH" >&2; exit 1; \
	fi; \
	if [ -x "$$VAR_DIR/openr1/bin/python" ]; then \
	  echo "Env already exists at $$VAR_DIR/openr1 (skipping create)."; \
	  echo "To refresh, run: CONDA_NO_PLUGINS=true conda env update -p $$VAR_DIR/openr1 -f configs/environment.yml --prune"; \
	else \
	  cd "$$ROOT_DIR/configs" && \
	    CONDARC="$$CONDARC" CONDA_PKGS_DIRS="$$CONDA_PKGS_DIRS" CONDA_ENVS_DIRS="$$CONDA_ENVS_DIRS" \
	    CONDA_NOTICES_PATH="$$VAR_DIR/conda/notices" CONDA_NO_PLUGINS=true \
	    PIP_CACHE_DIR="$$PIP_CACHE_DIR" TMPDIR="$$TMPDIR" \
	    conda env create -p "$$VAR_DIR/openr1" -f environment.yml; \
	  echo "✅ Env created at: $$VAR_DIR/openr1"; \
	fi; \
	mkdir -p "$$(dirname "$$ACTIVATE_HOOK")" "$$(dirname "$$DEACTIVATE_HOOK")"; \
	cat > "$$ACTIVATE_HOOK" <<-EOF
	export PIP_CACHE_DIR="$$PIP_CACHE_DIR"
	export TMPDIR="$$TMPDIR"
	export HF_HOME="$$HF_HOME"
	EOF
	cat > "$$DEACTIVATE_HOOK" <<-'EOF'
	unset PIP_CACHE_DIR
	unset TMPDIR
	unset HF_HOME
	EOF
	echo "Activate with: conda activate $$VAR_DIR/openr1"

lint:
	mkdir -p $(RUFF_CACHE_DIR)
	ruff check .
	pylint --rcfile=.pylintrc src

format:
	mkdir -p $(RUFF_CACHE_DIR)
	black .
	ruff check --fix .
	isort .

test:
	pytest -q -c configs/pytest.ini

precommit:
	pre-commit run -a

docs:
	mkdir -p $(VAR_DIR)/docs/_build
	python -m sphinx -b html docs $(VAR_DIR)/docs/_build/html

docs-clean:
	rm -rf $(VAR_DIR)/docs/_build

clean-local:
	rm -rf var .venv .local

validate-logs:
	python tools/validate_logs.py
