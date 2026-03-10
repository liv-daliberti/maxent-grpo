.PHONY: help install-dev install-local venv install-venv conda-local lint format test precommit docs docs-clean clean-local clean-local-deep smoke

VAR_DIR := $(CURDIR)/var
export RUFF_CACHE_DIR := $(VAR_DIR)/cache/ruff
export PYLINTHOME ?= $(VAR_DIR)/cache/pylint

help:
	@echo "Targets:"
	@echo "  install-dev  - pip install -e .[dev]"
	@echo "  install-local- user base under $(PWD)/.local"
	@echo "  conda-local  - create local conda env at ./var/openr1 via configs/environment.yml"
	@echo "  venv         - create .venv in repo"
	@echo "  install-venv - install project into .venv"
	@echo "  lint         - ruff check . && pylint src"
	@echo "  format       - ruff check --fix . && isort ."
	@echo "  test         - pytest -q"
	@echo "  precommit    - pre-commit run -a"
	@echo "  docs         - build Sphinx HTML to var/docs/_build/html"
	@echo "  docs-clean   - remove var/docs/_build"
	@echo "  clean-local  - remove local envs/runtime caches in this repo"
	@echo "  clean-local-deep - remove additional local cache/chaff directories"

install-dev:
	pip install -e .[dev]

install-local:
	export PYTHONUSERBASE=$(PWD)/.local; \
	  python -m pip install --upgrade pip; \
	  pip install --user -e .[dev]
	@echo "Add to PATH for local scripts: export PATH=\"$(PWD)/.local/bin:$$PATH\""

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
	mkdir -p $(RUFF_CACHE_DIR) $(PYLINTHOME)
	ruff check .
	pylint --rcfile=.pylintrc src

format:
	mkdir -p $(RUFF_CACHE_DIR)
	black .
	ruff check --fix .
	isort .

test:
	pytest -q

precommit:
	pre-commit run -a

docs:
	mkdir -p $(VAR_DIR)/docs/_build
	mkdir -p docs/_autosummary
	python -m sphinx -E -b html docs $(VAR_DIR)/docs/_build/html

docs-clean:
	rm -rf $(VAR_DIR)/docs/_build

clean-local:
	rm -rf var .venv .local .cache

clean-local-deep: clean-local
	rm -rf .conda_pkgs .pip_cache .tmp

smoke:
	HF_HOME=$(VAR_DIR)/cache/huggingface HF_DATASETS_CACHE=$(VAR_DIR)/cache/huggingface/datasets TRANSFORMERS_CACHE=$(VAR_DIR)/cache/huggingface/transformers PIP_CACHE_DIR=$(VAR_DIR)/cache/pip WANDB_DIR=$(VAR_DIR)/logs TMPDIR=$(VAR_DIR)/tmp WANDB_MODE=offline \
	  bash ops/run_paired_cpu_tiny.sh
