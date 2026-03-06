FROM python:3.12-slim
WORKDIR /app
COPY services/alerts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/sdk-py/ packages/sdk-py/
COPY services/explainer/ services/explainer/
COPY services/alerts/ services/alerts/
ENV PYTHONPATH=/app/packages/sdk-py:/app/services/explainer:/app/services/alerts
CMD ["python", "-m", "alerts_svc.worker"]
