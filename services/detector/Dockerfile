FROM python:3.12-slim
WORKDIR /app
COPY services/detector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/sdk-py/ packages/sdk-py/
COPY services/detector/ services/detector/
ENV PYTHONPATH=/app/packages/sdk-py:/app/services/detector
CMD ["python", "-m", "detector_svc.worker"]
