# explainer_svc is a library, not a runtime service.
# This Dockerfile is for running its tests in isolation only.
# In production it is COPY'd into the alerts and api service images.
FROM python:3.12-slim
WORKDIR /app
COPY services/explainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/sdk-py/ packages/sdk-py/
COPY services/explainer/ services/explainer/
ENV PYTHONPATH=/app/packages/sdk-py:/app/services/explainer
CMD ["python", "-m", "pytest", "services/explainer/tests/", "-v"]
