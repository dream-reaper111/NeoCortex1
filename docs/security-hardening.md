# Deployment Hardening Checklist

This checklist captures the immediate, near-term, and long-term actions that
should be applied when exposing the NeoAI API over the internet (for example
through an ngrok tunnel). Follow the tasks in priority order.

## Immediate (minutes)

1. **Require authentication for the tunnel**
   - Edit `~/.ngrok2/ngrok.yml` on the host that runs ngrok:
     ```yaml
     authtoken: <your-ngrok-token>
     http_auth:
       username: "<strong-username>"
       password: "<strong-random-password>"
     tunnels:
       web:
         proto: http
         addr: 127.0.0.1:8000
         hostname: <your-subdomain>.ngrok-free.dev
     ```
   - Restart ngrok: `pkill ngrok && ngrok start --all`.
   - Validate: `curl -u "<strong-username>:<strong-random-password>" https://<your-subdomain>.ngrok-free.dev/`.

2. **Lock down the server firewall (UFW)**
   ```bash
   sudo apt update
   sudo apt install -y ufw
   sudo ufw default deny incoming
   sudo ufw default allow outgoing
   sudo ufw allow OpenSSH
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   sudo ufw status verbose
   ```
   - Confirm listening sockets: `sudo ss -tulpn`.

3. **Patch and harden Nginx (or your reverse proxy)**
   ```bash
   sudo apt upgrade nginx
   sudo tee /etc/nginx/sites-available/neocortex <<'NGINX'
   server {
       listen 80 default_server;
       listen [::]:80 default_server;
       return 301 https://$host$request_uri;
   }

   server {
       listen 443 ssl http2;
       listen [::]:443 ssl http2;
       server_name <your-subdomain>.ngrok-free.dev;

       ssl_certificate /etc/letsencrypt/live/<your-subdomain>.ngrok-free.dev/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/<your-subdomain>.ngrok-free.dev/privkey.pem;
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers 'TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256';
       add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-Proto https;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }
   }
   NGINX
   sudo ln -sf /etc/nginx/sites-available/neocortex /etc/nginx/sites-enabled/neocortex
   sudo nginx -t && sudo systemctl reload nginx
   ```
   - Probe TLS configuration: `nmap --script ssl-enum-ciphers -p 443 <host>`.

4. **Disable unintended DNS exposure**
   ```bash
   sudo systemctl disable --now bind9 named
   sudo ss -tulpn | grep :53 || echo "Port 53 closed"
   ```

## Near Term (hours)

1. **Install Fail2Ban for SSH and Nginx auth**
   ```bash
   sudo apt install fail2ban
   sudo tee /etc/fail2ban/jail.local <<'JAIL'
   [sshd]
   enabled = true
   port = ssh
   filter = sshd
   logpath = /var/log/auth.log
   maxretry = 5

   [nginx-http-auth]
   enabled = true
   filter = nginx-http-auth
   port = http,https
   logpath = /var/log/nginx/error.log
   maxretry = 5
   bantime = 3600
   JAIL
   sudo systemctl restart fail2ban
   sudo fail2ban-client status nginx-http-auth
   ```

2. **Restrict DNS resolver usage (if required)**
   ```bash
   sudo ufw allow from 127.0.0.1 to any port 53 proto tcp
   sudo ufw allow from 127.0.0.1 to any port 53 proto udp
   sudo ufw deny 53/tcp
   sudo ufw deny 53/udp
   sudo systemctl restart systemd-resolved
   ```
   - Verify remotely: `dig @<public-ip> example.com` should fail.

3. **Harden Let's Encrypt renewals**
   ```bash
   sudo apt install unattended-upgrades
   sudo certbot renew --dry-run
   ```
   - Ensure a renewal timer or cron entry exists.

## Medium Term (days)

1. **Application security testing**
   - Run OWASP ZAP or Nikto locally against the public URL.
   - Address cross-site scripting, cookie, and header findings.

2. **Centralised logging and monitoring**
   - Ship Nginx access/error logs to a SIEM or logging stack.
   - Alert on 4xx/5xx spikes and authentication failures.

3. **Configuration management**
   - Store ngrok, Nginx, and firewall configuration in a private repo.
   - Document deployment and rollback procedures.

## Long Term (weeks)

1. **Move off ad-hoc tunnels**
   - Migrate to a dedicated reverse proxy or zero-trust access solution (WireGuard, Cloudflare Access).

2. **Schedule regular penetration tests**
   - Perform at least quarterly, alternating between DNS, network, and web application scopes.

3. **Security policy and backups**
   - Maintain an incident response plan, MFA-enforced access control, and encrypted, tested backups.

