.PHONY: up down logs build test

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build --no-cache

test-ingest:
	PYTHONPATH=packages/sdk-py:services/ingest python -m pytest services/ingest/tests/ -v

test-detector:
	PYTHONPATH=packages/sdk-py:services/detector python -m pytest services/detector/tests/ -v

test-explainer:
	PYTHONPATH=packages/sdk-py:services/explainer python -m pytest services/explainer/tests/ -v

test-alerts:
	PYTHONPATH=packages/sdk-py:services/explainer:services/alerts python -m pytest services/alerts/tests/ -v

test-api:
	PYTHONPATH=packages/sdk-py:services/explainer:services/api python -m pytest services/api/tests/ -v

test:
	$(MAKE) test-ingest test-detector test-explainer test-alerts test-api
