# AI-Trading Software Deployment Guide

This guide walks through deploying the AI-Trading stack using Docker Compose. The configuration targets a three-GPU host and assumes only software-level access (no BIOS, driver, or hardware changes).

## 1. Prerequisites
- Linux host with Docker-compatible kernel.
- NVIDIA drivers and CUDA toolkit already installed and operational (outside the scope of this guide).
- Administrative shell access with sudo privileges.
- DNS records (optional) pointing to the host if you plan to use custom domains.

## 2. Directory Layout
All deployment assets live under `/opt/ai-trade/`:
```
/opt/ai-trade/
├── docker-compose.yml
├── Caddyfile
├── ngrok.yml
├── .env                # create from .env.example
└── scripts/
    ├── install.sh
    └── update.sh
```

## 3. Configure Environment Variables
1. Copy the sample environment file:
   ```bash
   cd /opt/ai-trade
   cp .env.example .env
   ```
2. Edit `.env` and set strong values for:
   - `REDIS_URL`
   - `DB_PASS`
   - `JWT_SECRET`
   - `AUTHELIA_JWT_SECRET`
   - `NGROK_AUTH_TOKEN`
   - Any optional overrides (Grafana admin password, Authelia session secret, public domain).

## 4. Initial Deployment
Run the installation script to install Docker (if necessary) and launch the stack:
```bash
cd /opt/ai-trade/scripts
sudo ./install.sh
```
The script installs Docker Engine, enables the Docker service, pulls required container images, starts the stack, and ensures the Caddy container is enabled for autostart.

## 5. Accessing Services
- **APIs (FastAPI on GPU 0/1)**: Access through the public ngrok hostname(s). All `/api/*` requests are protected by Authelia multi-factor authentication. After authenticating, Authelia injects JWT headers for downstream authorization.
- **Grafana Dashboard**: Available internally on `http://localhost:3000` or via the Caddy reverse proxy at `https://<your-domain>/grafana`. Default credentials depend on the values you set in `.env` (`GF_SECURITY_ADMIN_PASSWORD`).
- **PostgreSQL / Redis**: Reachable only inside the `ai_net` Docker network. Use service names `db` (port 5432) and `redis` (port 6379) when connecting from other containers.
- **Caddy Logs**: Stored under the `caddy_logs` Docker volume. View with:
  ```bash
  docker compose -f /opt/ai-trade/docker-compose.yml logs -f caddy
  ```
- **ngrok**: The `ngrok.yml` file defines three persistent tunnels (`ai0`, `ai1`, `ai-train`) that forward HTTPS traffic to Caddy. After deployment, verify tunnel status at https://dashboard.ngrok.com.

## 6. Monitoring GPUs and Services
- Check GPU allocation per container:
  ```bash
  docker compose -f /opt/ai-trade/docker-compose.yml ps
  ```
- Inspect GPU utilization (example command):
  ```bash
  docker compose -f /opt/ai-trade/docker-compose.yml exec api_gpu0 nvidia-smi
  ```
  Repeat with `api_gpu1` and `trainer_gpu2` to confirm logical isolation via `CUDA_VISIBLE_DEVICES`.
- View service metrics in Grafana. Prometheus scraping can be added by extending the Compose file with exporters if required.

## 7. Routine Maintenance
- Update containers and clean unused volumes:
  ```bash
  cd /opt/ai-trade/scripts
  sudo ./update.sh
  ```
- Restart the stack without changes:
  ```bash
  docker compose -f /opt/ai-trade/docker-compose.yml restart
  ```

## 8. Backup Strategy
1. Stop services briefly (optional but recommended):
   ```bash
   docker compose -f /opt/ai-trade/docker-compose.yml stop api_gpu0 api_gpu1 trainer_gpu2
   ```
2. Archive critical volumes:
   ```bash
   docker run --rm \
     -v ai-trade_db_data:/data/db \
     -v $(pwd):/backup \
     alpine:3 tar czf /backup/ai-trade-db_$(date +%F).tgz -C /data/db .
   ```
   Repeat for `grafana_data`, `caddy_data`, and other volumes as needed.
3. Restart services:
   ```bash
   docker compose -f /opt/ai-trade/docker-compose.yml start
   ```

## 9. Scaling Guidance
- **Additional API GPUs**: Duplicate the `api_gpuX` service section in `docker-compose.yml`, increment the `container_name`, `CUDA_VISIBLE_DEVICES`, and add the service to the `reverse_proxy` `to` list in `Caddyfile`.
- **Horizontal Trainer Workers**: Clone the `trainer_gpu2` service with new names (`trainer_gpu3`, etc.) and assign unused GPU IDs. Adjust queues/topics in environment variables as necessary.
- **Non-GPU Scaling**: To add CPU-only background workers, create new services without GPU reservations and expose them via Redis job queues.
- After making scaling changes, run `docker compose up -d` to apply updates and confirm via `docker compose ps`.

## 10. Troubleshooting
- Use `docker compose logs <service>` for diagnostics.
- Ensure Authelia configuration files exist under `/opt/ai-trade/authelia/` and that secrets in `.env` match Authelia expectations.
- Verify ngrok tunnels with `docker compose logs ngrok` if external access fails.

---
Deployment is now fully software-defined. No physical configuration changes are required.
