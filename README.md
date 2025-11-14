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

## Platform Highlights

- **py-Trading API Enhancements** – Mirror trades between multiple linked
  accounts with adjustable size factors so risk can be tuned per follower.
- **Strategy Leaderboards** – Rank strategies by configurable metrics such as
  Sharpe ratio or profit factor to spotlight top performers for users.
- **Multi-User Dashboard** – Segregate data per user with OAuth2-protected
  access controls across the entire analytics suite.

## Developer Utilities

- **Strategy Backtester API** – Accepts JSON definitions, transforms them into
  Pandas data structures, and returns structured results for rapid iteration.
- **Indicator Sandbox** – Compile Pine scripts into Python with `ta-lib`
  bindings, enabling custom indicator experimentation inside the platform.
- **Live Code Reloader** – Hot-swap strategies at runtime without restarting
  the FastAPI server, drastically reducing deployment downtime.
- **API Versioning & Docs** – Leverages FastAPI's interactive `/docs` explorer
  so developers can quickly validate request/response schemas.

## Suggested Next Builds

1. **AI Portfolio Dashboard** – Deliver full analytics, attribution, and visual
   insights across portfolios.
2. **Broker Aggregator** – Manage and rebalance multiple brokerage accounts
   from a unified control panel.
3. **Model Lab** – Train, test, and export machine learning models directly
   within NeoCortex.
4. **Security Core v2** – Expand vault storage, auditing, and self-healing
   watchdog capabilities.
5. **Data Reactor** – Integrate sentiment and macroeconomic data feeds to
   enrich strategy inputs and signals.

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

## Deployment Tooling

- `docker-compose.yml` orchestrates the FastAPI API, Redis, PostgreSQL, Celery workers,
  an embedded n8n automation node, and the Prometheus/Grafana monitoring stack. Start
  everything locally with `docker compose up --build`.
- `deploy/k8s/` contains Kubernetes manifests for clustered deployments, including health probes,
  config maps, secrets, and monitoring components.
- `deploy/fly/fly.toml` provides a ready-to-use Fly.io application definition for quickly pushing
  the API to the edge.

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

### Workflow automation with n8n

The `services/model_orchestration/n8n_workflow.py` module provides a
standard-library-friendly REST client for triggering n8n workflows. Supply the
base URL of your n8n instance and an API key:

```python
from services.model_orchestration.n8n_workflow import N8nConfig, N8nWorkflowClient

config = N8nConfig(base_url="https://n8n.example.com", api_key="<token>")
client = N8nWorkflowClient(config)
result = client.trigger("42", payload={"symbol": "AAPL"})
```

By default the client polls the execution endpoint until the workflow finishes;
set `wait=False` to fire-and-forget.  This allows Celery tasks or FastAPI
endpoints to orchestrate downstream automations without shelling out to the n8n
CLI.

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

#### Admin private key configuration

Admin registration is gated by a shared secret so only trusted operators can
create privileged accounts. The backend requires this secret at startup via the
`ADMIN_PRIVATE_KEY` environment variable—requests to `POST /register` are
rejected until the variable is set. Export the value in your shell (PowerShell
example shown below) or place it in a `.env` file that `server.py` will load
automatically if [`python-dotenv`](https://pypi.org/project/python-dotenv/)
is installed.

```powershell
$env:ADMIN_PRIVATE_KEY = "change-me-with-a-strong-secret"
python server.py
```

All admin registration or login attempts must present the matching private key
in their payload. The key is not required for end-user logins or Whop-based
member onboarding.

1. Register an admin user with `POST /register` and a JSON body like
   `{ "username": "alice", "password": "<strong-password>", "admin_key": "<your-admin-key>" }`.
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
- `AUTH_COOKIE_SECURE` – defaults to `1` when TLS is configured (set to `0`
  explicitly if you are terminating HTTPS upstream).

#### Serving the dashboard over HTTPS

The FastAPI server can terminate TLS directly so credentials never traverse the
network in plain text. Provide file paths to a certificate and private key via
environment variables before launching `server.py`:

```bash
export SSL_CERTFILE="/etc/ssl/certs/fullchain.pem"
export SSL_KEYFILE="/etc/ssl/private/privkey.pem"
python server.py
```

When both variables are set, Uvicorn boots in HTTPS mode, the app issues
HTTP-only cookies with the `Secure` attribute, and an automatic redirect ensures
HTTP requests are upgraded to HTTPS. You can optionally supply a password for
an encrypted key (`SSL_KEYFILE_PASSWORD`) or a custom CA bundle
(`SSL_CA_CERTS`).

Strict Transport Security headers (`Strict-Transport-Security`) are enabled by
default for HTTPS deployments. Tweak the behaviour with:

- `FORCE_HTTPS_REDIRECT` – disable (`0`) if HTTPS termination happens upstream.
- `ENABLE_HSTS` – disable (`0`) to skip writing HSTS headers.
- `HSTS_MAX_AGE`, `HSTS_INCLUDE_SUBDOMAINS`, `HSTS_PRELOAD` – customize the HSTS
  directives if your domain policy differs from the defaults.

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
   - `POST /auth/whop/session` – validates the onboarding session and
     returns basic metadata (email and license).
   - `POST /register/whop` – completes registration, saving encrypted API keys
     and issuing a session cookie/bearer token.

Session cookies continue to work through an ngrok tunnel so multiple clients can
use the hosted console simultaneously.

### Webhook Verification

If you supply `ALPACA_WEBHOOK_SECRET`, incoming webhook requests must include
both `X-Webhook-Signature` and `X-Webhook-Timestamp` headers. The signature is
calculated as `base64(hmac_sha256(secret, f"{timestamp}.{body}"))`. Payloads
missing either header, whose signatures fail comparison, or whose timestamps
fall outside `ALPACA_WEBHOOK_TOLERANCE_SECONDS` (default 300 seconds) are
rejected. Replay attempts reuse the same signature and will be blocked with a
409 response. Set `ALPACA_ALLOW_UNAUTHENTICATED_WEBHOOKS=1` only for local
testing if you need to bypass signature enforcement.

### Webhook test helper

Use `POST /alpaca/webhook/test` to generate a signed sample payload. The
endpoint writes the payload to `public/alpaca_webhook_tests/` and, when a
secret is configured, returns a base64-encoded signature that mirrors what
Alpaca would send. Configure either `ADMIN_PORTAL_GATE_TOKEN` or
`ADMIN_PORTAL_BASIC_USER`/`ADMIN_PORTAL_BASIC_PASS` and leave
`ALPACA_WEBHOOK_TEST_REQUIRE_AUTH` at its default `true` value to restrict this
helper to trusted operators.

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

## Neo Cortex AI Trainer — Unified Technical Architecture & Mathematical Foundations

The Neo Cortex AI Trainer is a fully integrated FastAPI application that merges data ingestion, trading simulation, machine-learning training orchestration, and live brokerage connectivity into one deployable framework. It was written with modular independence: every layer, from HTTPS routing to neural feature generation, can run in isolation or together. What follows is a continuous technical narrative explaining both how the code works and the mathematics that underlie it.

### 1. Initialization and Environment Control

Execution begins by importing `os`, `json`, `time`, `math`, and similar core modules, then immediately setting two environment variables:

```
PYTORCH_ENABLE_COMPILATION = 0
TORCHDYNAMO_DISABLE = 1
```

These disable just-in-time graph compilation within PyTorch, ensuring deterministic execution. Certain CUDA builds introduce nondeterminism when tracing computation graphs, so disabling compilation means each forward pass uses explicit matrix multiplications rather than fused kernels, guaranteeing repeatable weights and identical gradient results.

Next, all optional dependencies are imported with fallbacks. Libraries such as `pandas`, `matplotlib`, and `dotenv` are wrapped in try/except guards. If unavailable, the code provides minimal stub classes so that API endpoints remain callable. For example, `_PandasStub` raises a friendly installation error when accessed. This keeps the system operational in minimal container environments or headless Linux servers.

Mathematically this design follows the idea of dependency projection: the execution space `E` is partitioned into required subset `R` and optional subset `O`, such that the program `f : E → S` (states) remains defined for all `e ∈ R` even if `e ∉ O`.

### 2. Directory Bootstrap and Persistent Paths

Upon startup, several directories are ensured to exist:

```
artifacts/
public/
public/liquidity/
public/liquidity/assets/
public/alpaca_webhook_tests/
public/enduserapp/
```

Each folder is created with `mkdir(parents=True, exist_ok=True)` to avoid race conditions. This guarantees that all later file writes (JSON logs, PNG charts, or trained models) will succeed even if the user runs the server for the first time. From a systems-engineering view, this creates a deterministic state manifold for the application’s file I/O—the mapping from runtime events to files is total and idempotent.

### 3. Alpaca API Configuration

Two parallel key sets define credentials:

- Paper trading: `ALPACA_KEY_PAPER`, `ALPACA_SECRET_PAPER`
- Funded trading: `ALPACA_KEY_FUND`, `ALPACA_SECRET_FUND`

If the live account keys are missing, the system defaults to paper mode. The base URLs differ (`paper-api.alpaca.markets` vs. `api.alpaca.markets`).

Mathematically the key resolution function is

```
C(account_type) = {
    {K_fund, S_fund, U_fund}  if account_type == "funded"
    {K_paper, S_paper, U_paper} otherwise
}
```

where `K = API key`, `S = secret`, and `U = base URL`. Later, `_resolve_alpaca_credentials()` merges user-specific encrypted credentials with these defaults to produce an effective credential vector per request.

### 4. In-Memory Data Structures

Several global dictionaries serve as live data stores:

- `Buffers` — mapping `"SYMBOL|INTERVAL" → pandas.DataFrame` of OHLCV data
- `Exog` — exogenous features (fundamentals, indicators) keyed by symbol
- `IngestStats` — integer counters of how many items were received per source
- `IdleTasks` — asynchronous tasks executing periodic training

The pair `{Buffers, Exog}` represents the model’s dynamic market state. Each update applies `pd.concat + groupby(level=0).last()`, a mathematical equivalent of a merge operation that keeps the most recent observation for any timestamp key.

### 5. Paper-Trading Simulation System

This subsystem reproduces market behavior for testing strategies without sending real orders.

#### 5.1 State Persistence

`PaperTradeState` is a JSON file stored at `public/papertrade_state.json`. It keeps an array of orders, each with fields such as symbol, side, quantity, price, and noise seed. A global asyncio lock ensures atomic read-write access.

#### 5.2 Price Oscillation Model

The simulator defines each instrument’s price evolution as a deterministic pseudo-sinusoidal function:

```
P(t) = P0 * (1 + A * sin(t/60 + φ))
```

where `P0 = entry price`, `A = 0.05` (amplitude of ±5 %), and `φ = (seed mod 360)` random phase per order, derived from the SHA1 hash of `symbol + timestamp`. This represents a simple harmonic oscillator with bounded amplitude. The use of sine avoids drift (`mean = P0`) and yields a predictable variance `σ² = (A · P0)² / 2`.

#### 5.3 Profit and Loss

For each order:

```
PnL = (P_current - P_entry) * Q * D * M
```

where `Q = quantity`, `D = direction (1 for long, –1 for short)`, and `M = multiplier (100 for options, 1 for equities)`. Total PnL across all open orders is then split into long PnL and short PnL, with `Total = Σ PnL_long + Σ PnL_short`. Expected mean PnL over time is zero since oscillation is symmetric; however instantaneous variance drives a visually realistic “breathing” equity curve on the dashboard.

#### 5.4 AI Trainer Decision Stub

Each new paper-trade order passes through `_run_ai_trainer()`, which produces a pseudo-confidence:

```
u = SHA1(symbol + instrument + side) / 16_777_215
confidence = 0.45 + 0.6 * (u - 0.5) + bias
```

`bias = +0.08` for long, `–0.08` for short, plus `+0.06` for options. If `confidence ≥ 0.48`, the action executes; otherwise the order is rejected. This defines a deterministic but symbol-unique function in `[0, 1]` that acts as a placeholder for the neural model’s probability of success.

### 6. Authentication, Authorization, and Encryption

The authentication layer is built on SQLite with tables `users`, `sessions`, `alpaca_credentials`, and `whop_sessions`.

#### 6.1 Password Hashing

Each password is salted with 32 random bytes and hashed using PBKDF2-HMAC-SHA512:

```
hash = PBKDF2(password, salt, iterations=200000, dkLen=256)
```

This produces a 2048-bit derived key. Computation cost ensures brute-force resistance (~100 ms per verification on a standard CPU).

#### 6.2 Credential Encryption

Alpaca API keys are stored encrypted with Fernet:

```
token = Fernet(key).encrypt(value)
```

Internally, Fernet uses AES-128 CBC with HMAC-SHA256. Encryption ensures database compromise cannot reveal secrets without the master key.

#### 6.3 Sessions

A session token is 32 bytes of URL-safe randomness. Tokens live in both DB and client cookie and expire based on `SESSION_COOKIE_MAX_AGE` (default 7 days). Security headers use `SameSite = Lax`, preventing cross-site request replay.

### 7. Whop Licensing Integration

Whop is treated as a licensing authority. A request to `/auth/whop/start` redirects the browser to the Whop Portal with a callback parameter. When Whop returns a license key, the server verifies it via REST API using bearer authentication:

```
Authorization: Bearer WHOP_API_KEY
GET /api/v2/licenses/<license_key>
```

The result’s `status` field must be in `{ active, trialing, paid }`. Upon validation, the system creates a temporary session lasting 15 minutes, allowing the user to register and attach Alpaca credentials. Mathematically, licensing validity is a Boolean predicate `L(license) = (status ∈ {active, trialing, paid})`. Access is granted iff `L = True`.

### 8. Alpaca Credentials Management

Endpoints `/alpaca/credentials` (GET/POST) let users manage encrypted API keys. The encryption–decryption path is:

```
cipher = Fernet(master_key)
enc_key = "enc:" + cipher.encrypt(api_key)
dec_key = cipher.decrypt(enc_key[4:])
```

When resolving credentials for any trading endpoint, the system chooses the user’s stored version if present; otherwise it falls back to environment variables. This gives each user isolation of broker access.

### 9. Account Data and P&L Mathematics

The `/positions` endpoint calls Alpaca’s `/v2/positions` API. Returned JSON is a list of open positions; each contains fields `qty`, `avg_entry_price`, `market_value`, `cost_basis`, and `unrealized_pl`.

If `unrealized_pl` is missing, it is recomputed as:

```
UPL = (MarketValue - CostBasis)
```

or equivalently `UPL = (Price_mark - Price_entry) * Quantity`. Cumulative portfolio P&L is `Σ UPL_i`. If we define `r_i = UPL_i / CostBasis_i`, total return `R = Σ r_i w_i` where `w_i = CostBasis_i / Σ CostBasis`.

Expected variance of total P&L across independent positions is `Var(R) = Σ w_i² σ_i²`. When correlation is considered, `Var(R) = wᵀ Σ w`, where `Σ` is the covariance matrix of position returns. The server does not compute correlation yet, but the math lays the foundation for later integration into risk models.

### 10. Liquidity Sweep Analysis

`/strategy/liquidity-sweeps` delegates to `analyze_liquidity_session()` inside `strategies.py`. While code is external, the mathematics typically include computation of bid–ask imbalances per time window, liquidity `L(t) = Σ(|buy_volume − sell_volume|)`, and Z-score normalization `z = (L − mean_L) / std_L`. Heatmap cell intensity is proportional to `z²` to highlight abnormal sweeps. The endpoint then exposes these assets through `/public/liquidity/assets/<file>` so that a front-end dashboard can visualize market structure.

### 11. Alpaca Webhook Security and Math

Incoming webhook messages are authenticated using an HMAC:

```
expected = base64(b64encode(HMAC_SHA256(secret, body)))
if not constant_time_compare(expected, header):
    reject 400
```

Mathematically this computes the digest `H = h_k(m) = SHA256(k ⊕ opad || SHA256(k ⊕ ipad || m))`, where `m = body` bytes and `k = secret`. The `compare_digest` call ensures `O(n)` constant time to prevent timing attacks. The webhook’s function is informational—it logs the JSON and returns `{"ok": true}`. In future expansions, one could use this feed to update local order states, making the paper-trade dashboard reflect real brokerage fills.

### 12. Alpaca Webhook Testing

`/alpaca/webhook/test` synthesizes payloads like:

```json
{
  "event": "trade_update",
  "order": {
    "symbol": "SPY",
    "qty": 1,
    "side": "buy",
    "status": "filled"
  }
}
```

and signs them if a secret exists. It then writes JSON to `public/alpaca_webhook_tests/test-SPY-<timestamp>.json`. From a math standpoint, signature generation is identical to the webhook verification equation; thus `signature = base64(HMAC_SHA256(secret, body))`. This self-test allows end-to-end signature round-trip verification.

### 13. TradingView and External Data Ingestion

#### 13.1 TradingView Webhook

TradingView alerts send OHLCV bars and optional signal features. The code converts them into a single-row DataFrame and updates `Buffers` via:

```
Buffers[k] = concat(Buffers[k], bar).groupby(index).last().tail(20000)
```

Mathematically this is a temporal union over timestamp set `T = {t₁,…,tₙ}`, selecting the latest observation per `t`. Signals (`long`, `short`, `flat`) are one-hot encoded: `tv_long = 1 if signal == "long" else 0`, `tv_short = 1 if signal == "short" else 0`, `tv_flat = 1 if signal == "flat" else 0`. These binary features later augment model inputs.

#### 13.2 Generic Candles, Features, and Fundamentals

Generic ingestion normalizes OHLCV, converts timestamps to UTC, and upserts data via concatenation plus `last()` to maintain continuity. Each feature row appended to `Exog` for the symbol allows the training function to compute correlation between price movement and exogenous signals—a key principle in multivariate time-series modeling.

### 14. Model Training Subsystem

#### 14.1 Input Normalization

When a `/train` request arrives, the server constructs an environment vector of parameters (ticker, period, interval, aggressiveness, etc.). It downloads historical bars from `yfinance`. Each series is standardized using the helper `standardize_ohlcv(df)`:

- Lowercase columns → `["open", "high", "low", "close", "volume"]`
- Convert index to UTC
- Sort by time
- Drop NaNs

In mathematical terms, this converts any data matrix `X_raw ∈ ℝ^(T×C)` into `X = sort_time(clean(X_raw))`. Ensuring monotonically increasing time is crucial for computing differences `ΔX_t = X_t − X_{t−1}` used in returns.

#### 14.2 Feature Engineering

The function `build_features()` computes additional columns such as:

- Rolling mean: `μ_t = mean(P_{t−L:t})`
- Rolling std: `σ_t = std(P_{t−L:t})`
- Z-score: `z_t = (P_t − μ_t) / σ_t`
- Volume z-score, volatility K-scaling, RSI, MACD, and more

If exogenous fundamentals are present, they are merged on timestamp alignment. The resulting feature matrix `X ∈ ℝ^(T×N)` represents the full state vector per time `t`.

#### 14.3 Aggressiveness Mapping

The model uses a preset mapping from aggressiveness to entry threshold:

| Aggressiveness | Entry Threshold |
| -------------- | ---------------- |
| chill          | 1.15             |
| normal         | 0.95             |
| spicy          | 0.75             |
| insane         | 0.55             |

Mathematically this is a scalar multiplier controlling the entry probability threshold. It scales the z-score trigger: a more “spicy” mode accepts trades at smaller `z` values, increasing variance but potentially higher reward.

#### 14.4 Model Training and Neural Math

`train_and_save()` trains the model and returns accuracy and sample count. Forward pass and backpropagation follow standard neural network equations:

```
ŷ = f(W₂ · f(W₁ · X + b₁) + b₂)
Loss = mean((ŷ − Y_true)²)
dL/dW = (2/N) * (ŷ − Y_true) * Xᵀ
W_new = W − η * dL/dW
```

with learning rate `η`. GPU acceleration exploits parallel matrix multiplication. Each CUDA core executes a subset of the dot-product operations concurrently.

Training accuracy `A = 1 − Loss / L_max`. If `n` is the number of samples, standard error of accuracy is approximately `sqrt(A * (1 − A) / n)`, allowing confidence intervals for true accuracy to be computed on the fly.

#### 14.5 Error Handling

If PyTorch is mis-installed, the server intercepts exceptions referencing `torch._dynamo` and responds with a fix hint. This keeps API uptime high even under misconfigured environments.

#### 14.6 Visualization

After training, `_save_price_preview()` generates a PNG chart of closing prices using Matplotlib (or a stub). This provides quick visual confirmation that the data feed was parsed correctly.

#### 14.7 Batch Training

The `/train/multi` endpoint runs parallel training over multiple tickers using a `ThreadPoolExecutor`. Mathematically, tasks execute concurrently; total wall time is approximately `max_i (t_i)` rather than `Σ t_i`, bounded by available CPU threads.

### 15. Idle Training Loops

An asynchronous function `_idle_loop(name, spec)` repeatedly invokes `train_multi()` every `spec.every_sec` seconds. This implements an internal cron scheduler. From a control-theory perspective, it is a discrete-time periodic process with period `T = every_sec`. Its stability criterion is that training duration `≤ T` to avoid overlap. The dictionary `IdleTasks` keeps handles so users can start or cancel tasks via `/idle/start` and `/idle/stop`.

### 16. Runtime and Metrics System

After training loops are defined, the framework moves into live-monitoring mode. Every training run produces metrics, logs, and neural-network state files inside its unique `artifacts/run-<ticker>-<timestamp>` directory. The runtime layer exposes these through HTTP so dashboards can stream results in real time.

#### 16.1 Metrics Lifecycle

Each run stores:

- `metrics.json`
- `train.log.jsonl`
- `nn_state.jsonl`
- `nn_graph.json`
- `preview.png`

When a client queries `/metrics/latest`, the server reads the most recent directory by modification time and loads `metrics.json`. The response merges three data sources:

- Static metrics – loss, accuracy, epoch count
- Runtime stats – CPU usage, RAM, GPU status via `psutil` and `torch.cuda`
- Temporal metadata – run ID, timestamp, and last-modified delta

Mathematically the metric vector `M(t)` is `[accuracy(t), loss(t), gpu_mem(t), cpu_load(t), Δtime]`. A client can then estimate the derivative `dM/dt` to monitor convergence rate. For instance, if `loss(t) = L₀ exp(−k t)`, then `k ≈ −ln(L_t / L_{t−1})`, providing a numerical measure of how fast the model learns per epoch.

#### 16.2 Artifact Streaming

Endpoints such as `/artifacts/file/<name>` return any artifact file from the latest run. File selection algorithm:

```
latest_run = max(glob("artifacts/run-*"), key=mtime)
file_path  = join(latest_run, name)
return FileResponse(file_path)
```

This mapping defines a surjective relation between artifacts and runs so each run remains self-contained.

### 17. Server-Sent Event Streams

Real-time dashboards depend on continuous data. The system uses two asynchronous Server-Sent Event (SSE) endpoints:

- `/stream/training` – tails `train.log.jsonl` and sends incremental JSON lines labeled `"ping"`
- `/stream/nn` – tails `nn_state.jsonl` and sends `"nn_ping"` events containing neural-network weight statistics

The underlying `_tail(file_path)` coroutine opens the file and yields new lines as soon as they are appended. From a signals perspective this implements a discrete-time sampling of the training process: each JSON line represents a state vector `s_t = (loss, acc, epoch)`. Clients can approximate gradients `Δs = s_t − s_{t−1}` to animate convergence curves in real time. Throughput is approximately `O(1 KB s⁻¹)`—small enough for browser consumption yet responsive enough for GPU training updates every second.

### 18. Neural-Network State and Mathematics

#### 18.1 Graph Representation

`/nn/graph` returns JSON describing layer topology. A minimal example:

```json
{
  "layers": [
    {"type": "Input", "size": 32},
    {"type": "Dense", "size": 64, "activation": "relu"},
    {"type": "Dense", "size": 32, "activation": "relu"},
    {"type": "Dense", "size": 1,  "activation": "sigmoid"}
  ]
}
```

Forward propagation follows `h_i = f_i(W_i · h_{i−1} + b_i)` where `f` is the activation (ReLU = `max(0, x)`, Sigmoid = `1/(1+e^(−x))`). Training adjusts each `W, b` to minimize loss `L(ŷ, y)` via gradient descent:

```
W_i ← W_i − η ∂L/∂W_i
b_i ← b_i − η ∂L/∂b_i
```

For binary classification with sigmoid output, loss is cross-entropy:

```
L = −[y · log(ŷ) + (1 − y) · log(1 − ŷ)]
```

The derivative with respect to ŷ is `∂L/∂ŷ = (ŷ − y) / (ŷ (1 − ŷ))`, which propagates backward through the chain rule.

#### 18.2 Batch Normalization and Stability

If BatchNorm layers are added, each feature is normalized:

```
x̂ = (x − μ_B) / √(σ_B² + ε)
y = γ x̂ + β
```

This ensures variance ≈ 1 and stabilizes gradient magnitudes. Without it, exploding or vanishing gradients can occur when multiplying many Jacobians whose eigenvalues ≠ 1.

#### 18.3 Regularization and Expected Loss

L2 regularization adds `λ ||W||²`, yielding total loss:

```
L_total = L_data + λ Σ ||W_i||²
```

Expected gradient magnitude decreases by `2λW`, so weights decay toward zero, reducing overfitting. Empirically this behaves like adding Gaussian noise with variance ≈ `1/(2λ)`.

### 19. Risk and Drawdown Modeling

#### 19.1 Return Definitions

For any position with prices `P_t`:

```
r_t = ln(P_t / P_{t−1})
```

Portfolio cumulative return `R_T = Σ_t r_t`. Average return `μ = E[r_t]`, variance `σ² = Var[r_t]`. Expected profit per trade ≈ `μQ`, risk ≈ `σQ`.

#### 19.2 Drawdown Computation

Drawdown `D_t = max_{i≤t}(R_i) − R_t`. Maximum drawdown = `max_t D_t`. If returns are normal `N(μ, σ²)`, expected max drawdown ≈ `σ √(2 ln n)`. The simulator could add this formula to chart account risk over time.

#### 19.3 Monte Carlo Projection

For forward simulation the model can use geometric Brownian motion:

```
dS_t = μ S_t dt + σ S_t dW_t
S_{t+Δt} = S_t * exp((μ − ½σ²)Δt + σ √Δt · Z)
```

where `Z ~ N(0, 1)`. This allows estimation of future price distribution given current volatility `σ`. Expected mean `E[S_T] = S_0 e^{μT}`. Variance `Var[S_T] = S_0² e^{2μT}(e^{σ²T} − 1)`. Such stochastic modeling justifies the AI’s decision thresholds: if predicted `μ / σ > threshold`, the trade executes.

### 20. Confidence Scoring and Signal Math

Within the pseudo-AI module the confidence metric mimics a logistic-regression probability:

```
conf = 1 / (1 + exp(−(w · x + b)))
```

In the current stub the feature vector `x` is replaced by hashed randomness, but in the full model each `x` would contain `[z_price, z_volume, RSI, MACD, sentiment, feature_n …]`. Weights `w` are trained to maximize likelihood `P(y | x)`. During inference the sign of `conf − 0.5` determines side; magnitude `|conf − 0.5|` indicates conviction. Mathematically this aligns with binary hypothesis testing where `H₁ = profit trade` and `H₀ = loss trade`. The decision rule executes when `P(H₁ | x) ≥ τ`, minimizing expected loss `= c₁ · P(false positive) + c₂ · P(false negative)`.

### 21. Performance Metrics and Confidence Intervals

Suppose training yields accuracy `Â` on `n` samples. Standard error `SE = sqrt(Â (1 − Â) / n)`. A 95% confidence interval ≈ `Â ± 1.96 SE`. If `Â = 0.98` on `n = 892`, `SE ≈ 0.0047`, so CI ≈ `[0.971, 0.989]`. This quantifies statistical reliability of each model run shown in `metrics.json`.

For regression outputs (e.g., predicted return `r̂_t`) performance uses RMSE `= √(Σ(r̂ − r)² / n)`. Lower RMSE implies tighter predictive variance `σ_pred²`.

### 22. GPU Parallelization Math

Each forward pass executes many matrix multiplies `C = A · B`. For `A` of `m×k` and `B` of `k×n`, total FLOPs = `2mkn`. Modern GPU streams distribute computation across `p` cores, so time ≈ `2mkn/p`. Training throughput (samples/s) ∝ `1 / (batch_size · layers · param_count / p)`. Memory footprint ≈ `4 bytes × (total params + activations)`. Monitoring `torch.cuda.memory_allocated()` lets the server report utilization in `/metrics/latest`.

### 23. System Diagnostics and Preflight

Before serving requests, `run_preflight()` performs checks: verify Python packages, inspect free disk and RAM, detect CUDA devices, and log UTC time. If GPU present, `torch.cuda.get_device_properties()` returns name, vRAM, and SM count. Reported as JSON:

```json
{
  "packages": {"ok": true},
  "hardware": {"ram_avail_gb": 31.4, "disk_free_gb": 212.3},
  "gpu": {"cuda_available": true, "devices": ["RTX4090"]},
  "time_utc": "2025-11-09 21:05:00"
}
```

Mathematically these diagnostics provide baseline vector `S₀ = (RAM, Disk, GPU)` used later to compare runtime drift `S_t − S₀`.

### 24. Public Interface and Static Routes

The FastAPI router exposes HTML pages: `/login`, `/admin/login`, `/dashboard`, `/liquidity`. Each serves static files from `public/`. Headers `Cache-Control: no-store` ensure live reloads after builds. Functionally this layer closes the loop between AI server and user interface. All dashboard graphs read JSON from `/metrics/latest` and SSE streams, rendering portfolio and neural metrics dynamically.

### 25. End-to-End Mathematical Summary

Bringing every subsystem together:

1. Data ingestion builds feature matrix `X_t` from price, volume, and fundamentals.
2. Neural model estimates `p_t = P(trade success | X_t)` via `σ(W · X_t + b)`.
3. Decision layer triggers trade if `p_t ≥ τ` (aggressiveness-dependent).
4. Execution simulator evolves price `P(t) = P₀ (1 + 0.05 sin(ωt + φ))`.
5. `PnL(t) = (P(t) − P₀) Q D M`.
6. Cumulative return `R_T = Σ ln(P_t / P_{t−1})`; drawdown `D_t = max_{i≤t}(R_i) − R_t`.
7. Training minimizes expected loss `E[(ŷ − y)²]`; weights update `W ← W − η ∇L`.
8. Metrics stream over SSE; confidence intervals quantify statistical reliability.
9. GPU parallelism reduces compute time ∝ `1/p`; Fernet encryption secures credentials; PBKDF2 protects user passwords.

Together these form a reproducible computational graph: `raw data → standardized features → neural inference → simulated execution → streamed metrics`. Mathematical rigor ensures determinism and transparency; software engineering discipline ensures deployability. This dual structure—the engineering foundation (FastAPI, SQLite, PyTorch) fused with quantitative theory (stochastic modeling, gradient learning, statistical confidence)—defines the Neo Cortex AI Trainer as both a practical tool and a research-grade system, as well as a killer synthetic cognitive auto trading AI.

### 26. Architectural Perspective

Conceptually the system can be viewed as a four-layer stack:

1. **Transport Layer:** FastAPI + ngrok for secure HTTPS access.
2. **Data Layer:** SQLite for authentication, `Buffers`/`Exog` for time-series.
3. **Computation Layer:** PyTorch models, NumPy/Pandas analytics, CUDA acceleration.
4. **Presentation Layer:** Static HTML dashboard + SSE streams.

Each layer obeys clear mathematical boundaries—transformations are functional and mostly stateless. Because everything is deterministic given seed values, every run is reproducible: same data + same parameters ⇒ same artifacts.

### 27. Future Mathematical Extensions

The existing infrastructure can readily integrate advanced methods:

- Kalman filters for real-time signal estimation: `x̂_t = x̂_{t−1} + K_t(z_t − H x̂_{t−1})`.
- Bayesian weight updates: `posterior ∝ likelihood × prior`.
- Reinforcement learning policies: maximize `E[Σ γ^t r_t]` using Q-learning.
- VaR (Value at Risk) and CVaR computation: `VaR_α = inf{l : P(L > l) ≤ α}`.
- Correlation matrix visualization for liquidity clusters.

All can attach seamlessly to existing endpoints due to the modular FastAPI core.

### 28. Closing Summary

The Neo Cortex AI Trainer unifies secure authentication, quantitative simulation, and neural inference into a single runtime that is both human-readable and mathematically explicit. Every component—from Fernet-encrypted credentials to the sinusoidal Alpaca paper-market simulator and GPU-accelerated neural network—forms part of a reproducible computational graph:

```
raw data → standardized features → neural inference → simulated execution → streamed metrics
```

Mathematical rigor ensures determinism and transparency; software engineering discipline ensures deployability. This fusion of engineering foundations with quantitative theory defines the Neo Cortex AI Trainer as both a practical tool and a research-grade system.

