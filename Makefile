PYTHON ?= python3
VENV ?= venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup index run test health evaluate clean clean-data

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

index:
	$(PY) scripts/prepare_documents.py
	$(PY) scripts/embed_documents.py
	$(PY) scripts/create_final_data.py

run:
	$(PY) -m src.app

test:
	$(PY) -m unittest discover -s tests -v

health:
	$(PYTHON) -c "from src.health import get_health_report; import json; print(json.dumps(get_health_report().to_dict(), indent=2))"

CASES ?= eval/rag_cases.json
evaluate:
	$(PY) scripts/evaluate_rag.py --cases $(CASES)

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage

clean-data:
	find data -mindepth 1 ! -name '.gitkeep' -delete
