FROM python:3.12-slim
WORKDIR /app
COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/sdk-py/ packages/sdk-py/
COPY services/explainer/ services/explainer/
COPY services/api/ services/api/
ENV PYTHONPATH=/app/packages/sdk-py:/app/services/explainer:/app/services/api
CMD ["uvicorn", "api_svc.main:app", "--host", "0.0.0.0", "--port", "8002"]
