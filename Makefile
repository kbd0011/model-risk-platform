.PHONY: setup test lint typecheck check run
# One-command workflows. Seeds are fixed in code; configs drive the runs (reproducible).

setup:          ## install dependencies
	pip install -r requirements.txt

test:           ## run the test suite
	pytest -q

lint:           ## ruff lint
	ruff check src tests

typecheck:      ## mypy type check
	mypy src --ignore-missing-imports

check: lint typecheck test   ## lint + typecheck + test (what CI runs)

run:            ## reproduce the real German-Credit validation numbers + figure
	PYTHONPATH=. python scripts/run_german_credit.py
