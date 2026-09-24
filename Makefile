.PHONY: install test lint fmt

install:
	pip install -e ".[dev]"

test:
	pytest -x

lint:
	ruff check src tests
	mypy src

fmt:
	ruff format src tests
	ruff check --fix src tests
