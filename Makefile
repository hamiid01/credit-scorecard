.PHONY: help install test demo reject clean

PY ?= python

help:
	@echo "install  editable install with dev extras"
	@echo "test     run the test suite"
	@echo "demo     develop a scorecard on the Taiwan default panel"
	@echo "reject   run the reject-inference experiment"
	@echo "clean    remove build and cache artefacts"

install:
	$(PY) -m pip install -e ".[dev,plots]"

test:
	$(PY) -m pytest -q

demo:
	cd experiments && $(PY) -m credit_scorecard -o scorecard.csv

reject:
	cd experiments && $(PY) reject_inference_experiment.py

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
