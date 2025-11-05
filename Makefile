.PHONY: help install-dev lint format test precommit docs docs-clean

help:
	@echo "Targets:"
	@echo "  install-dev  - pip install -e .[dev]"
	@echo "  lint         - ruff check . && pylint src/open_r1"
	@echo "  format       - ruff check --fix . && isort ."
	@echo "  test         - pytest -q"
	@echo "  precommit    - pre-commit run -a"
	@echo "  docs         - build Sphinx HTML to _build/html"
	@echo "  docs-clean   - remove _build"

install-dev:
	pip install -e .[dev]

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

