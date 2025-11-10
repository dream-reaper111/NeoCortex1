#!/usr/bin/env bash
set -euo pipefail

SERVICES=(api celery-worker celery-beat redis postgres)

for service in "${SERVICES[@]}"; do
  if ! docker compose ps --status=running "$service" >/dev/null 2>&1; then
    echo "[auto-restart] Restarting $service"
    docker compose restart "$service"
  fi
done
