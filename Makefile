.PHONY: up down logs build test test-sdk-ts

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build --no-cache

test-ingest:
	PYTHONPATH=packages/sdk-py:packages/schemas-py:services/ingest python -m pytest services/ingest/tests/ -v

test-schemas:
	PYTHONPATH=packages/sdk-py:packages/schemas-py python -m pytest packages/schemas-py/tests/ -v

test-detector:
	PYTHONPATH=packages/sdk-py:services/detector python -m pytest services/detector/tests/ -v

test-explainer:
	PYTHONPATH=packages/sdk-py:services/explainer python -m pytest services/explainer/tests/ -v

test-alerts:
	PYTHONPATH=packages/sdk-py:services/explainer:services/alerts python -m pytest services/alerts/tests/ -v

test-api:
	PYTHONPATH=packages/sdk-py:services/explainer:services/api python -m pytest services/api/tests/ -v

test-mcp:
	python -m pytest packages/mcp-server/tests/ -v

test-sdk-ts:
	cd packages/sdk-ts && npm test

test:
	$(MAKE) test-schemas test-ingest test-detector test-explainer test-alerts test-api test-mcp test-sdk-ts
