#!/usr/bin/env bash
# deploy.sh — pull latest and restart services
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Pulling latest images..."
docker compose -f infra/docker-compose.yml -f infra/docker-compose.prod.yml pull

echo "Restarting services..."
docker compose -f infra/docker-compose.yml -f infra/docker-compose.prod.yml up -d --remove-orphans

echo "Running migrations..."
docker compose -f infra/docker-compose.yml exec api python -m alembic upgrade head 2>/dev/null || true

echo "Done."
docker compose -f infra/docker-compose.yml ps
