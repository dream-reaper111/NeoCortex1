#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "[+] Installing Docker Engine..."
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg lsb-release
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") $(lsb_release -cs) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo systemctl enable --now docker
else
  echo "[+] Docker already installed."
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "[+] Docker Compose plugin not detected. Verify installation."
  exit 1
fi

echo "[+] Pulling container images..."
docker compose -f "$BASE_DIR/docker-compose.yml" pull --ignore-pull-failures

echo "[+] Starting AI-Trading stack..."
docker compose -f "$BASE_DIR/docker-compose.yml" up -d

echo "[+] Ensuring Caddy service is enabled at boot..."
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl enable caddy || true
fi

echo "[+] Deployment complete — visit your ngrok domain to access the APIs."
