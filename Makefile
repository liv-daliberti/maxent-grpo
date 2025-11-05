.PHONY: help install-dev install-local venv install-venv lint format test precommit docs docs-clean

help:
	@echo "Targets:"
	@echo "  install-dev  - pip install -e .[dev]"
	@echo "  install-local- user base under $(PWD)/.local"
	@echo "  venv         - create .venv in repo"
	@echo "  install-venv - install project into .venv"
	@echo "  lint         - ruff check . && pylint src/open_r1"
	@echo "  format       - ruff check --fix . && isort ."
	@echo "  test         - pytest -q"
	@echo "  precommit    - pre-commit run -a"
	@echo "  docs         - build Sphinx HTML to _build/html"
	@echo "  docs-clean   - remove _build"

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

lint:
	ruff check .
	pylint --rcfile=.pylintrc src/open_r1

format:
	ruff check --fix .
	isort .

test:
	pytest -q

precommit:
	pre-commit run -a

docs:
	sphinx-build -b html docs _build/html

docs-clean:
	rm -rf _build
