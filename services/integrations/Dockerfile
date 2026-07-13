FROM python:3.12-slim
WORKDIR /app
COPY services/integrations/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/integrations/ services/integrations/
ENV PYTHONPATH=/app/services/integrations
CMD ["python", "-m", "integrations_svc.worker"]
