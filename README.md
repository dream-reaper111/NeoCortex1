# NeoAI Training + End‑User Dashboard with GPU and ngrok Support

This repository contains the NeoAI training API, example front‑end, and
integration helpers for running on Windows with Visual Studio, optional GPU
support, and an [ngrok](https://ngrok.com/) tunnel for external access. It
also includes a simple end‑user dashboard that interacts with the API to
display account positions and unrealized P&L from Alpaca Markets.

## Contents

- `server.py` – FastAPI backend for training models, fetching Alpaca positions
  and P&L, and serving metrics. Includes a new `/alpaca/webhook` endpoint for
  receiving trade update callbacks.
- `model.py` – GPU‑aware PyTorch model with clear import error handling. If
  PyTorch cannot be imported (e.g. mismatched CUDA DLLs on Windows), an
  explanatory `ImportError` is raised.
- `run_with_ngrok.py` – Helper script that starts an ngrok tunnel to your
  local API port and runs the FastAPI application via Uvicorn.
- `strategies.py` – Liquidity sweep/footprint analytics shared by the API and
  UI.
- `requirements.txt` – Dependency list. Includes `pyngrok` for ngrok
  integration. **Important:** To use the GPU, install a PyTorch wheel that
  matches your CUDA version. See below.
- `public/enduserapp/` – Self‑contained End User Console with Alpaca webhook
  tooling (copyable URL, test payload generator, and artifact viewer).
- `public/liquidity/` – Mobile and desktop friendly Liquidity Sweep Radar UI
  that consumes the `/strategy/liquidity-sweeps` API.

## Installation

1. **Clone or extract** the repository and open it in Visual Studio or your
   preferred IDE. Ensure you have Python 3.12 or later installed.
2. **Create a virtual environment** (recommended):

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

3. **Install dependencies**:

   ```powershell
   # CPU‑only (works on any system)
   python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

   # OR, GPU build (replace ``cu118`` with your CUDA version)
   python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
   ```

   The training code will automatically use the GPU if available:

   ```python
   import torch
   print(torch.cuda.is_available())  # True if GPU is detected
   ```

   If you see an import error referencing `caffe2_nvrtc.dll`, you installed a
   CUDA‑enabled PyTorch wheel without the matching NVIDIA driver. Either
   install a CPU wheel (see above) or update your driver.

## Running the API

The default API host and port are ``0.0.0.0`` and ``8000``. You can change
these via environment variables (`API_HOST`, `API_PORT`).

### Without ngrok

To start the API locally (no tunnel):

```powershell
python server.py
```

Access `http://localhost:8000` in your browser. The `/gpu-info` endpoint
reports whether CUDA is available. The `/positions` and `/pnl` endpoints call
Alpaca (paper or live) to fetch account data. Set your Alpaca credentials via
environment variables (see below).

### With ngrok (external access)

To expose your local API to the internet, provide your ngrok token and set a
strong username/password pair for HTTP basic authentication before running the
helper script:

```powershell
$env:NGROK_AUTH_TOKEN = "<your-ngrok-auth-token>"
$env:NGROK_BASIC_AUTH_USER = "<strong-username>"
$env:NGROK_BASIC_AUTH_PASS = "<strong-random-password>"
# optional: request a reserved domain if your plan supports it
$env:NGROK_DOMAIN = "trader.example.ngrok-free.dev"
python run_with_ngrok.py
```

The script prints a public URL (e.g. `https://1234.ngrok.io`). Use this URL to
access your API externally. For Alpaca webhooks, append `/alpaca/webhook` to
that URL and configure it in your Alpaca dashboard.

If you want helper endpoints (such as `/ngrok/cloud-endpoint`) to advertise a
fixed hostname, set `DEFAULT_PUBLIC_BASE_URL` to your external URL before
starting the server.

`run_with_ngrok.py` enforces HTTP basic authentication. Provide credentials via
either `NGROK_BASIC_AUTH` (format `user:pass`) or the
`NGROK_BASIC_AUTH_USER`/`NGROK_BASIC_AUTH_PASS` pair shown above. You can
optionally restrict access to specific client networks via
`NGROK_ALLOWED_CIDRS="198.51.100.0/24,203.0.113.5/32"`. If you have reserved a
domain inside the ngrok dashboard, set `NGROK_DOMAIN` accordingly; otherwise
leave it unset to allow ngrok to allocate a random hostname.

Additional hardening guidance—including firewall, Fail2Ban, TLS, and DNS
recommendations—is available in
[`docs/security-hardening.md`](docs/security-hardening.md).
## Configuring Alpaca

The API uses Alpaca Markets for positions and P&L. You must set the
appropriate credentials as environment variables before running the server.

- **Paper trading** (simulated):

  ```powershell
  $env:ALPACA_KEY_PAPER    = "YOUR_PAPER_KEY"
  $env:ALPACA_SECRET_PAPER = "YOUR_PAPER_SECRET"
  # optional – defaults to https://paper-api.alpaca.markets
  $env:ALPACA_BASE_URL_PAPER = "https://paper-api.alpaca.markets"
  ```

- **Live trading** (funded):

  ```powershell
  $env:ALPACA_KEY_FUND    = "YOUR_FUNDED_ACCOUNT_KEY"
  $env:ALPACA_SECRET_FUND = "YOUR_FUNDED_ACCOUNT_SECRET"
  $env:ALPACA_BASE_URL_FUND = "https://api.alpaca.markets"  # optional when using the default
  ```

Set only the environment variables you need. If a key/secret pair is missing
for a given account type, the corresponding endpoint will return an error.

### Per-user credential storage and login

The API now ships with a lightweight SQLite database (`auth.db` by default)
that stores user accounts and optional Alpaca credentials. Passwords are
hashed with a salted 2048-bit PBKDF2-SHA512 digest before being persisted.

1. Register a user with `POST /register` and a JSON body of
   `{ "username": "alice", "password": "<strong-password>" }`.
2. Log in with `POST /login` using the same payload to receive a bearer
   token. Supply this token to other endpoints via
   `Authorization: Bearer <token>`.


   A browser-friendly login flow now lives at [`/login`](http://localhost:8000/login).
   It issues the same bearer token and stores it alongside an HTTP-only session
   cookie so the `/dashboard` UI can enforce authentication before loading.

3. To save Alpaca API keys that are unique to the authenticated user, call
   `POST /alpaca/credentials` with:

   ```json
   {
     "account_type": "paper",  // or "funded"
     "api_key": "YOUR_KEY",
     "api_secret": "YOUR_SECRET",
     "base_url": "https://paper-api.alpaca.markets"  // optional
   }
   ```

   Calling `GET /alpaca/credentials` returns the stored metadata (API keys are
   returned, secrets remain server-side). When the bearer token is supplied to
   `/positions` or `/pnl`, the API automatically prefers the stored credentials
   over environment defaults.

   > **Encryption required:** set `CREDENTIALS_ENCRYPTION_KEY` to a
   > base64-encoded 32-byte Fernet key (generate one with
   > `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).
   > User-supplied API keys are stored encrypted at rest and cannot be persisted
   > without this key.

You can customize how the browser session cookie is issued by setting:

- `SESSION_COOKIE_NAME` – defaults to `session_token`.
- `SESSION_COOKIE_MAX_AGE` – lifetime in seconds (default 7 days).
- `SESSION_COOKIE_SAMESITE` – defaults to `lax`.
- `AUTH_COOKIE_SECURE` – set to `1` to require HTTPS when sending the cookie.

### Whop membership onboarding

If you sell access through [Whop](https://whop.com), the login screen now
supports a "Continue with Whop" flow that verifies a member's license before
allowing them to create local credentials and register their funding keys.

1. Set the following environment variables for the API server:

   - `WHOP_API_KEY` – a private token from the Whop dashboard.
   - `WHOP_PORTAL_URL` – the URL Whop should send the user to when they need to
     authenticate (for example your product portal). Include a `{callback}`
     placeholder in this URL if you want the API to inject the callback URL
     automatically, e.g. `https://whop.com/portal/your-product?redirect={callback}`.
   - Optional: `WHOP_API_BASE` (defaults to `https://api.whop.com`) and
     `WHOP_SESSION_TTL` (in seconds, default 900) to control how long the
     onboarding link is valid.

2. Configure Whop to redirect members back to
   `https://<your-host>/auth/whop/callback` with a `license_key` query parameter
   once they are authenticated. The server validates this license with Whop,
   creates an onboarding session, and redirects the browser to the regular login
   page with a short-lived `whop_token`.

3. When the login page detects `whop_token`, it prompts the user to create a
   username/password and provide their Alpaca funding keys in one step. The
   backend exposes helper endpoints to support this flow:

   - `GET /auth/whop/start` – sends the browser to Whop, filling in the
     callback URL and optional `next` destination.
   - `GET /auth/whop/session?token=...` – validates the onboarding session and
     returns basic metadata (email and license).
   - `POST /register/whop` – completes registration, saving encrypted API keys
     and issuing a session cookie/bearer token.

Session cookies continue to work through an ngrok tunnel so multiple clients can
use the hosted console simultaneously.

### Webhook Verification

If you supply `ALPACA_WEBHOOK_SECRET`, incoming webhook requests must include
a matching `X-Webhook-Signature` header. A mismatch yields a 400 error.

### Webhook test helper

Use `POST /alpaca/webhook/test` to generate a signed sample payload. The
endpoint writes the payload to `public/alpaca_webhook_tests/` and, when a
secret is configured, returns a base64-encoded signature that mirrors what
Alpaca would send.

## Liquidity Sweep Radar (mobile + desktop)

The `/strategy/liquidity-sweeps` endpoint analyses the New York morning session
for a given ticker and interval, returning:

- Detected liquidity sweeps with volume/range ratios.
- Manipulation clusters (alternating sweeps within a 20-minute window).
- Footprint totals (estimated buy/sell imbalance).
- A volume heatmap stored under `public/liquidity/assets/`.

You can interact with these analytics via the responsive UI in
`public/liquidity/`:

```powershell
python -m http.server --directory public 8001
# open http://localhost:8001/liquidity/ or use http://localhost:8000/liquidity when the API is running
```

## End‑User Console + Alpaca Webhook Tools

The new **End User Console** lives under `public/enduserapp/` and is served
directly by the FastAPI backend at `http://localhost:8000/ui/enduserapp/` (or
your ngrok tunnel + `/ui/enduserapp/`). It packages a few handy utilities for
teams who only need webhook management:

1. **Copy the webhook URL.** The console derives the correct
   `https://<your-host>/alpaca/webhook` endpoint based on the current origin so
   you can paste it into the Alpaca dashboard.
2. **Send test payloads.** A form wraps the `POST /alpaca/webhook/test` helper
   and shows the JSON response returned by the API. Optional overrides let you
   paste a custom payload for regression testing.
3. **Browse saved artifacts.** Every test call is written to
   `public/alpaca_webhook_tests/`. The console lists the files via the new
   `GET /alpaca/webhook/tests` endpoint and provides download links for quick
   auditing or sharing with teammates.

No build step or extra server is required—just run `python server.py` (or
`python run_with_ngrok.py`) and open the UI in a browser.

## Notes

- This project is for educational and testing purposes. The trading API
  endpoints and webhook handler do **not** place or manage orders. To
  implement actual trading logic, integrate with Alpaca's order endpoints and
  comply with all relevant regulations and platform policies.
- The included model is a simple multi‑layer perceptron (MLP) and should not
  be used for real trading decisions without extensive validation.
- Running the training process with GPU support requires an appropriate CUDA
  wheel and compatible NVIDIA drivers.

Feel free to customize and extend this codebase for your own research and
development.
