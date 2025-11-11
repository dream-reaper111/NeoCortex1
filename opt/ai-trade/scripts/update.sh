#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "[-] Docker is not installed. Run scripts/install.sh first." >&2
  exit 1
fi

echo "[+] Pulling latest container images..."
docker compose -f "$BASE_DIR/docker-compose.yml" pull

echo "[+] Recreating services with updated images..."
docker compose -f "$BASE_DIR/docker-compose.yml" up -d

echo "[+] Removing unused volumes to reclaim space..."
docker volume prune -f

echo "[+] Update complete."
