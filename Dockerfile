FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/sdk-py/ packages/sdk-py/
COPY services/ services/

ENV PYTHONPATH=/app/packages/sdk-py:/app/services/ingest
EXPOSE 8001
CMD ["uvicorn", "ingest_svc.main:app", "--host", "0.0.0.0", "--port", "8001"]
