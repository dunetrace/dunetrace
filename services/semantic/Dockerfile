FROM python:3.12-slim
WORKDIR /app
COPY services/semantic/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# audit Finding 17 (HIGH): explainer_svc imports `dunetrace` (the SDK), so the
# sdk-py package must be present + on PYTHONPATH — it was missing here, which
# crash-looped the worker. Matches services/alerts/Dockerfile. (Authorized during
# audit to enable Phase 3 semantic testing.)
COPY packages/sdk-py/ packages/sdk-py/
COPY services/explainer/ services/explainer/
COPY services/semantic/ services/semantic/
ENV PYTHONPATH=/app/packages/sdk-py:/app/services/explainer:/app/services/semantic
CMD ["python", "-m", "semantic_svc.worker"]
