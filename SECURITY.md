# Security Policy

## Supported Versions

The `main` branch receives security updates on a continuous basis. Releases and
tagged commits derived from `main` inherit the most recent fixes.

| Version | Supported |
| ------- | --------- |
| main    | ✅        |

## Reporting a Vulnerability

Please report suspected vulnerabilities to **security@neocortex.ai**. Include a
description of the issue, the affected endpoints or components, and any proof
of concept material you can share safely. We acknowledge reports within two
business days and aim to provide a triage update within five business days.
Coordinated disclosure is encouraged; we will work with you on reasonable
release timelines so mitigations are in place before public disclosure.

For sensitive payloads, request our PGP key via the email address above.

## Deployment Hardening Checklist

* **Network controls.** Restrict ingress traffic with UFW so that only SSH and
  the FastAPI application are reachable:

  ```bash
  sudo ufw default deny incoming
  sudo ufw default allow outgoing
  sudo ufw allow 22/tcp comment "SSH"
  sudo ufw allow 8000/tcp comment "Neo Cortex API"
  sudo ufw enable
  ```

* **Intrusion detection.** Install and enable `fail2ban` (to block repeated
  authentication failures) and `rkhunter` (to monitor for rootkits). Both tools
  should be configured to alert operators via email or your central monitoring
  stack.

* **Dependency security.** The CI pipeline runs `safety check` to flag
  vulnerable Python dependencies before builds are published. Treat any failing
  job as a release blocker and address the affected packages.

* **CORS and origins.** Set the `ALLOWED_ORIGINS` environment variable to a
  comma-separated list of trusted origins. Requests from unlisted origins are
  rejected automatically.

* **Tunnel protection.** When exposing the service through ngrok, set
  `NGROK_BASIC_AUTH` (or the paired user/password environment variables) so the
  tunnel is always protected with HTTP Basic authentication in addition to
  HTTPS.
