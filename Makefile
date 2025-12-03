PYTHON = /opt/venv/bin/python
UV = /opt/uv/uv

.PHONY: deps build clean install install-edit coverage check-all check-diff docs

deps:
	if [ -f requirements-test.txt ]; then \
		$(PYTHON) -m pip install .[test]; \
	elif [ -f requirements.txt ]; then \
		$(PYTHON) -m pip install .; \
	elif grep -q "^\[dependency-groups\]" pyproject.toml && grep -q "^dev = \[" pyproject.toml; then \
		echo "Dependency group 'dev' found. Running 'uv pip install --group dev'."; \
		$(UV) pip install -r pyproject.toml --group dev; \
	else \
		echo "Dependency group 'dev' not found. Running 'uv pip install'."; \
		$(UV) pip install -r pyproject.toml; \
	fi

build:
	$(PYTHON) -m build

clean:
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ build/ dist/ docs/_build/ .coverage
	find . \( -name '__pycache__' -o -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf {} +
	find . \( -name '*.egg' -o -name '*.egg-info' \) -exec rm -rf {} +

install:
	$(UV) pip install .

install-edit:
	$(UV) pip install -e .

coverage:
	pytest --cov --cov-config=pyproject.toml -m unit

check-all:
	pre-commit run -a

check-diff:
	pre-commit run

docs:
	sphinx-apidoc --templatedir docs/_templates -M -f -o docs/ src/
	cd docs && make html
