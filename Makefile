.PHONY: up down logs build test test-sdk-py test-sdk-ts test-schema-parity

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
	PYTHONPATH=packages/schemas-py:packages/sdk-py:services/detector python -m pytest services/detector/tests/ -v

test-explainer:
	PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer python -m pytest services/explainer/tests/ -v

test-alerts:
	PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer:services/alerts python -m pytest services/alerts/tests/ -v

test-api:
	PYTHONPATH=packages/schemas-py:packages/sdk-py:services/explainer:services/api python -m pytest services/api/tests/ -v

# pytest, not `unittest discover`: discovery finds ~110 fewer tests here (the
# whole tests/test_integrations/ tree among them), and pre-commit and CI both
# use pytest — so a green `unittest` run could and did hide a real failure.
test-sdk-py:
	cd packages/sdk-py && python -m pytest tests/ -q

test-mcp:
	python -m pytest packages/mcp-server/tests/ -v

test-semantic:
	PYTHONPATH=packages/schemas-py:services/explainer:services/semantic python -m pytest services/semantic/tests/ -v

test-integrations:
	PYTHONPATH=packages/schemas-py:services/integrations python -m pytest services/integrations/tests/ -v

test-sdk-ts:
	cd packages/sdk-ts && npm test

# Cross-service: asserts the tables declared by more than one service agree.
# No PYTHONPATH — it parses the sources rather than importing them.
test-schema-parity:
	python -m pytest tests/ -v

test:
	$(MAKE) test-sdk-py test-schemas test-ingest test-detector test-explainer test-alerts test-api test-mcp test-semantic test-integrations test-schema-parity test-sdk-ts
