.PHONY: pipeline ingest preprocess train export serve dashboard test lint precommit install

install:
	uv pip install -e ".[dev]"

pipeline:
	python scripts/run_pipeline.py

ingest:
	python -m src.ingestion.grid_client
	python -m src.ingestion.weather_client

preprocess:
	python -m src.processing.cleaner
	python -m src.processing.features

train:
	python -m src.training.trainer

export:
	python -m src.training.export

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest -v

lint:
	ruff check src tests
	ruff format --check src tests

precommit:
	ruff check src tests && ruff format --check src tests && gitleaks detect --no-git --source .
