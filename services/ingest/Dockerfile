FROM python:3.12-slim
WORKDIR /app
COPY services/ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/schemas-py/ packages/schemas-py/
COPY services/ingest/ services/ingest/
ENV PYTHONPATH=/app/packages/schemas-py:/app/services/ingest
CMD ["uvicorn", "ingest_svc.main:app", "--host", "0.0.0.0", "--port", "8001"]
