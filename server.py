# -*- coding: utf-8 -*-
from __future__ import annotations

# ---- Torch compile guards (before any torch/model import) ----
import os as _os
_os.environ.setdefault("PYTORCH_ENABLE_COMPILATION", "0")
_os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# ---- std imports ----
import os, json, time, math, shutil, asyncio, importlib, hashlib, sqlite3, secrets, logging
from urllib.parse import parse_qs, quote, urlencode
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Literal

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency in tests
    class _PandasStub:
        """Lazy stub that raises a helpful error when pandas is unavailable."""

        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                "pandas is required for data-processing endpoints. Install it with 'pip install pandas'."
            )

    pd = _PandasStub()  # type: ignore

from fastapi import FastAPI, HTTPException, Request, Header, Cookie
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse

try:
    from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
except ModuleNotFoundError:  # pragma: no cover - fallback for trimmed fastapi distro
    class HTTPSRedirectMiddleware:
        """Minimal HTTPS redirect middleware compatible with FastAPI's interface."""

        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope.get("type") != "http":
                await self.app(scope, receive, send)
                return

            scheme = scope.get("scheme", "http")
            if scheme == "https":
                await self.app(scope, receive, send)
                return

            headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
            host = headers.get("host")
            if not host:
                server = scope.get("server")
                if server:
                    host = f"{server[0]}:{server[1]}" if server[1] else server[0]
                else:
                    host = ""

            path = scope.get("raw_path") or scope.get("path", "")
            if isinstance(path, bytes):
                path = path.decode("latin-1")

            query = scope.get("query_string", b"")
            if query:
                path = f"{path}?{query.decode('latin-1')}"

            target_url = f"https://{host}{path}" if host else "https://" + path.lstrip("/")
            response = RedirectResponse(url=target_url, status_code=307)
            await response(scope, receive, send)

from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False


def _env_flag(name: str, default: bool = False) -> bool:
    """Interpret an environment variable as a boolean flag."""

    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _optional_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return str(Path(value).expanduser())


def _load_optional_module(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    try:
        return importlib.import_module(name)
    except Exception:
        return None


psutil = _load_optional_module("psutil")

logger = logging.getLogger("neocortex.server")

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    matplotlib = None  # type: ignore

    class _MatplotlibStub:
        def figure(self, *args, **kwargs):
            return self

        def plot(self, *args, **kwargs):
            return None

        def title(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

        def tight_layout(self, *args, **kwargs):
            return None

        def savefig(self, *args, **kwargs):
            path = kwargs.get("fname") or (args[0] if args else None)
            if path:
                Path(path).write_bytes(b"")
            return None

        def close(self, *args, **kwargs):
            return None

    plt = _MatplotlibStub()  # type: ignore

# added imports for Alpaca integration
import requests
from cryptography.fernet import Fernet, InvalidToken

# ---- your model utils (import AFTER env guards) ----
try:
    from model import build_features, train_and_save, latest_run_path
except Exception as exc:  # pragma: no cover - optional heavy dependency fallback
    def _model_unavailable(*_args, **_kwargs):
        raise RuntimeError(f"Model dependencies unavailable: {exc}")

    def build_features(*args, **kwargs):  # type: ignore
        return _model_unavailable(*args, **kwargs)

    def train_and_save(*args, **kwargs):  # type: ignore
        return _model_unavailable(*args, **kwargs)

    def latest_run_path() -> Optional[str]:  # type: ignore
        return None

try:
    from strategies import analyze_liquidity_session, StrategyError
except Exception as exc:  # pragma: no cover - optional dependency fallback
    class StrategyError(RuntimeError):
        pass

    def analyze_liquidity_session(*_args, **_kwargs):
        raise StrategyError(f"Strategy module unavailable: {exc}")

load_dotenv(override=False)

SSL_CERTFILE = _optional_path(os.getenv("SSL_CERTFILE"))
SSL_KEYFILE = _optional_path(os.getenv("SSL_KEYFILE"))
SSL_KEYFILE_PASSWORD = os.getenv("SSL_KEYFILE_PASSWORD") or None
SSL_CA_CERTS = _optional_path(os.getenv("SSL_CA_CERTS"))
SSL_ENABLED = bool(SSL_CERTFILE and SSL_KEYFILE)
FORCE_HTTPS_REDIRECT = _env_flag("FORCE_HTTPS_REDIRECT", default=SSL_ENABLED)
ENABLE_HSTS = _env_flag("ENABLE_HSTS", default=SSL_ENABLED)
HSTS_MAX_AGE = int(os.getenv("HSTS_MAX_AGE", "31536000"))
HSTS_INCLUDE_SUBDOMAINS = _env_flag("HSTS_INCLUDE_SUBDOMAINS", default=True)
HSTS_PRELOAD = _env_flag("HSTS_PRELOAD", default=False)
if ENABLE_HSTS:
    _hsts_directives = [f"max-age={HSTS_MAX_AGE}"]
    if HSTS_INCLUDE_SUBDOMAINS:
        _hsts_directives.append("includeSubDomains")
    if HSTS_PRELOAD:
        _hsts_directives.append("preload")
    HSTS_HEADER_VALUE = "; ".join(_hsts_directives)
else:
    HSTS_HEADER_VALUE = None

API_HOST   = os.getenv("API_HOST","0.0.0.0")
API_PORT   = int(os.getenv("API_PORT","8000"))
RUNS_ROOT  = Path(os.getenv("RUNS_ROOT","artifacts")).resolve()
STATIC_DIR = Path(os.getenv("STATIC_DIR","static")).resolve(); STATIC_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR","public")).resolve(); PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
LIQUIDITY_DIR = PUBLIC_DIR / "liquidity"
LIQUIDITY_DIR.mkdir(parents=True, exist_ok=True)
LIQUIDITY_ASSETS = LIQUIDITY_DIR / "assets"
LIQUIDITY_ASSETS.mkdir(parents=True, exist_ok=True)
ALPACA_TEST_DIR = PUBLIC_DIR / "alpaca_webhook_tests"
ALPACA_TEST_DIR.mkdir(parents=True, exist_ok=True)

LOGIN_PAGE = PUBLIC_DIR / "login.html"
ADMIN_LOGIN_PAGE = PUBLIC_DIR / "admin-login.html"

ENDUSERAPP_DIR = PUBLIC_DIR / "enduserapp"
ENDUSERAPP_DIR.mkdir(parents=True, exist_ok=True)

NGROK_ENDPOINT_TEMPLATE_PATH = PUBLIC_DIR / "ngrok-cloud-endpoint.html"
try:
    NGROK_ENDPOINT_TEMPLATE = NGROK_ENDPOINT_TEMPLATE_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    NGROK_ENDPOINT_TEMPLATE = (
        "<!doctype html><html><body><h1>ngrok endpoint</h1>"
        "<p>Webhook URL: {{WEBHOOK_URL}}</p></body></html>"
    )
# ---------

NGROK_ENDPOINT_TEMPLATE_PATH = PUBLIC_DIR / "ngrok-cloud-endpoint.html"
try:
    NGROK_ENDPOINT_TEMPLATE = NGROK_ENDPOINT_TEMPLATE_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    NGROK_ENDPOINT_TEMPLATE = (
        "<!doctype html><html><body><h1>ngrok endpoint</h1><p>Webhook URL: {{WEBHOOK_URL}}</p></body></html>"
    )

ENDUSERAPP_DIR = PUBLIC_DIR / "enduserapp"
ENDUSERAPP_DIR.mkdir(parents=True, exist_ok=True)

ENDUSERAPP_DIR = PUBLIC_DIR / "enduserapp"
ENDUSERAPP_DIR.mkdir(parents=True, exist_ok=True)

NGROK_ENDPOINT_TEMPLATE_PATH = PUBLIC_DIR / "ngrok-cloud-endpoint.html"


def _load_ngrok_template(path: Path) -> str:
    """Best-effort loader for the ngrok cloud endpoint HTML template."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Fall back to a tiny inline page so the endpoint still renders
        # something useful even if the repository asset is missing.
        return (
            "<!doctype html><html><body><h1>ngrok endpoint</h1><p>Webhook URL: "
            "{{WEBHOOK_URL}}</p></body></html>"
        )


NGROK_ENDPOINT_TEMPLATE = _load_ngrok_template(NGROK_ENDPOINT_TEMPLATE_PATH)

ENDUSERAPP_DIR = PUBLIC_DIR / "enduserapp"
ENDUSERAPP_DIR.mkdir(parents=True, exist_ok=True)

NGROK_ENDPOINT_TEMPLATE_PATH = PUBLIC_DIR / "ngrok-cloud-endpoint.html"


def _load_ngrok_template(path: Path) -> str:
    """Best-effort loader for the ngrok cloud endpoint HTML template."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Fall back to a tiny inline page so the endpoint still renders
        # something useful even if the repository asset is missing.
        return (
            "<!doctype html><html><body><h1>ngrok endpoint</h1><p>Webhook URL: "
            "{{WEBHOOK_URL}}</p></body></html>"
        )


NGROK_ENDPOINT_TEMPLATE = _load_ngrok_template(NGROK_ENDPOINT_TEMPLATE_PATH)

ENDUSERAPP_DIR = PUBLIC_DIR / "enduserapp"
ENDUSERAPP_DIR.mkdir(parents=True, exist_ok=True)

NGROK_ENDPOINT_TEMPLATE_PATH = PUBLIC_DIR / "ngrok-cloud-endpoint.html"


def _load_ngrok_template(path: Path) -> str:
    """Best-effort loader for the ngrok cloud endpoint HTML template."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Fall back to a tiny inline page so the endpoint still renders
        # something useful even if the repository asset is missing.
        return (
            "<!doctype html><html><body><h1>ngrok endpoint</h1><p>Webhook URL: "
            "{{WEBHOOK_URL}}</p></body></html>"
        )


NGROK_ENDPOINT_TEMPLATE = _load_ngrok_template(NGROK_ENDPOINT_TEMPLATE_PATH)

DEFAULT_PUBLIC_BASE_URL = os.getenv("DEFAULT_PUBLIC_BASE_URL", "").rstrip("/")


DEFAULT_PUBLIC_BASE_URL = os.getenv(
    "DEFAULT_PUBLIC_BASE_URL",
    "https://tamara-unleavened-nonpromiscuously.ngrok-free.dev",
).rstrip("/")


# -----------------------------------------------------------------------------
# Alpaca configuration
#
# This server integrates with the Alpaca Markets API for retrieving account
# positions and computing unrealized profit and loss (P&L). To support both
# paper (simulated) and funded trading accounts, two sets of credentials
# (API key, secret and base URL) can be provided via environment variables.
#
# For paper trading:
#   ALPACA_KEY_PAPER        – your Alpaca paper account API key
#   ALPACA_SECRET_PAPER     – your Alpaca paper account secret key
#   ALPACA_BASE_URL_PAPER   – optional; defaults to https://paper-api.alpaca.markets
#
# For funded (live) trading:
#   ALPACA_KEY_FUND         – your Alpaca live account API key
#   ALPACA_SECRET_FUND      – your Alpaca live account secret key
#   ALPACA_BASE_URL_FUND    – optional; defaults to https://api.alpaca.markets
#
# If either the key or secret is missing for a given account type, the
# corresponding endpoint will return an error.

# Load Alpaca credentials from environment. Paper trading endpoints are used by
# default when no account type is specified.
ALPACA_KEY_PAPER = os.getenv("ALPACA_KEY_PAPER")
ALPACA_SECRET_PAPER = os.getenv("ALPACA_SECRET_PAPER")
ALPACA_BASE_URL_PAPER = os.getenv("ALPACA_BASE_URL_PAPER", "https://paper-api.alpaca.markets")

ALPACA_KEY_FUND = os.getenv("ALPACA_KEY_FUND")
ALPACA_SECRET_FUND = os.getenv("ALPACA_SECRET_FUND")
ALPACA_BASE_URL_FUND = os.getenv("ALPACA_BASE_URL_FUND", "https://api.alpaca.markets")

Buffers: Dict[str, pd.DataFrame] = {}
Exog: Dict[str, pd.DataFrame] = {}          # features/fundamentals/signals aligned by timestamp
IngestStats: Dict[str, Any] = {"tradingview":0, "robinhood":0, "webull":0, "features":0, "candles":0}
IdleTasks: Dict[str, asyncio.Task] = {}

PAPERTRADE_STATE_PATH = PUBLIC_DIR / "papertrade_state.json"
PAPERTRADE_STATE_LOCK = asyncio.Lock()
PaperTradeState: Dict[str, Any] = {"orders": []}
PaperTradeLoaded = False


def _load_papertrade_state() -> None:
    global PaperTradeState, PaperTradeLoaded
    if PaperTradeLoaded:
        return
    try:
        if PAPERTRADE_STATE_PATH.exists():
            data = json.loads(PAPERTRADE_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("orders"), list):
                PaperTradeState["orders"] = data["orders"]
    except Exception:
        # fall back to empty state on read error
        PaperTradeState["orders"] = []
    PaperTradeLoaded = True


def _save_papertrade_state() -> None:
    try:
        PAPERTRADE_STATE_PATH.write_text(
            json.dumps({"orders": PaperTradeState.get("orders", [])}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("[papertrade] unable to persist state", exc_info=True)


def _papertrade_noise_seed(order: Dict[str, Any]) -> int:
    seed_value = order.get("noise_seed")
    if isinstance(seed_value, str):
        try:
            return int(seed_value, 16)
        except ValueError:
            pass
    base = f"{order.get('symbol','')}-{order.get('timestamp','')}"
    digest = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def _papertrade_price(order: Dict[str, Any], now: Optional[float] = None) -> float:
    entry_price = float(order.get("entry_price") or order.get("price") or 1.0)
    entry_price = max(entry_price, 0.01)
    ts = (now or time.time()) / 60.0
    seed = _papertrade_noise_seed(order)
    oscillation = math.sin(ts + (seed % 360)) * 0.05
    price = entry_price * (1.0 + oscillation)
    return round(max(price, 0.01), 2)


def _papertrade_multiplier(order: Dict[str, Any]) -> float:
    return 100.0 if (order.get("instrument") == "option") else 1.0


def _papertrade_pnl(order: Dict[str, Any], current_price: float) -> float:
    direction = 1.0 if order.get("side") == "long" else -1.0
    qty = float(order.get("quantity") or 0.0)
    entry = float(order.get("entry_price") or order.get("price") or current_price)
    multiplier = _papertrade_multiplier(order)
    pnl = (current_price - entry) * qty * direction * multiplier
    return round(pnl, 2)


def _papertrade_dashboard(now: Optional[float] = None) -> Dict[str, Any]:
    orders = PaperTradeState.get("orders", [])
    long_orders: List[Dict[str, Any]] = []
    short_orders: List[Dict[str, Any]] = []
    long_pnl = 0.0
    short_pnl = 0.0
    snapshot_time = now or time.time()

    for raw in orders:
        order = dict(raw)
        order.pop("noise_seed", None)
        status = order.get("status") or "executed"
        if status == "executed":
            current_price = _papertrade_price(raw, snapshot_time)
            pnl = _papertrade_pnl(raw, current_price)
        else:
            current_price = float(order.get("entry_price") or order.get("price") or 0.0)
            pnl = 0.0
        order["current_price"] = round(current_price, 2)
        order["pnl"] = round(pnl, 2)

        if (order.get("side") or "long") == "long":
            if status == "executed":
                long_pnl += pnl
            long_orders.append(order)
        else:
            if status == "executed":
                short_pnl += pnl
            short_orders.append(order)

    total_pnl = round(long_pnl + short_pnl, 2)
    return {
        "pnl": {
            "total": round(total_pnl, 2),
            "long": round(long_pnl, 2),
            "short": round(short_pnl, 2),
        },
        "orders": {
            "long": long_orders,
            "short": short_orders,
        },
        "generated_at": datetime.fromtimestamp(snapshot_time, tz=timezone.utc).isoformat(),
    }


def _run_ai_trainer(order: Dict[str, Any]) -> Dict[str, Any]:
    symbol = (order.get("symbol") or "").upper()
    digest = hashlib.sha1(f"{symbol}-{order.get('instrument')}-{order.get('side')}".encode("utf-8", errors="ignore")).hexdigest()
    signal = int(digest[:6], 16) / float(0xFFFFFF)
    bias = 0.08 if order.get("side") == "long" else -0.08
    bias += 0.06 if order.get("instrument") == "option" else 0.04
    confidence = max(0.0, min(1.0, 0.45 + (signal - 0.5) * 0.6 + bias))
    action = "execute" if confidence >= 0.48 else "reject"
    message = (
        "Neo Cortex AI executed the paper order."
        if action == "execute"
        else "Neo Cortex AI parked the order for review."
    )
    return {
        "trainer": "Neo Cortex AI Trainer",
        "action": action,
        "confidence": round(confidence, 3),
        "message": message,
    }


try:
    _load_papertrade_state()
except Exception:
    logger.warning("[papertrade] failed to load initial state", exc_info=True)

# --- authentication and credential storage -------------------------------------------------------
AUTH_DB_PATH = Path(os.getenv("AUTH_DB_PATH", "auth.db")).resolve()
AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "session_token")
SESSION_COOKIE_SECURE = _env_flag("AUTH_COOKIE_SECURE", default=SSL_ENABLED)
SESSION_COOKIE_MAX_AGE = int(os.getenv("SESSION_COOKIE_MAX_AGE", str(7 * 24 * 3600)))
SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "lax") or "lax"

CREDENTIALS_ENCRYPTION_KEY = (os.getenv("CREDENTIALS_ENCRYPTION_KEY") or "").strip() or None
WHOP_API_KEY = (os.getenv("WHOP_API_KEY") or "").strip() or None
WHOP_API_BASE = (os.getenv("WHOP_API_BASE") or "https://api.whop.com").rstrip("/")
WHOP_PORTAL_URL = (os.getenv("WHOP_PORTAL_URL") or "").strip() or None
WHOP_SESSION_TTL = int(os.getenv("WHOP_SESSION_TTL", "900"))
ADMIN_PRIVATE_KEY = (os.getenv("ADMIN_PRIVATE_KEY") or "the3istheD3T").strip()


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_auth_db() -> None:
    with _db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        info = conn.execute("PRAGMA table_info(users)").fetchall()
        column_names = {row["name"] for row in info}
        if "is_admin" not in column_names:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alpaca_credentials (
                user_id INTEGER NOT NULL,
                account_type TEXT NOT NULL,
                api_key TEXT NOT NULL,
                api_secret TEXT NOT NULL,
                base_url TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, account_type),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS whop_sessions (
                token TEXT PRIMARY KEY,
                license_key TEXT NOT NULL,
                email TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                consumed_at TEXT
            )
            """
        )


_init_auth_db()


class CredentialEncryptionError(RuntimeError):
    pass


_CREDENTIALS_CIPHER: Optional[Fernet] = None


def _credentials_cipher() -> Fernet:
    global _CREDENTIALS_CIPHER
    if _CREDENTIALS_CIPHER is not None:
        return _CREDENTIALS_CIPHER
    if not CREDENTIALS_ENCRYPTION_KEY:
        raise CredentialEncryptionError(
            "CREDENTIALS_ENCRYPTION_KEY must be set to store API credentials securely"
        )
    key = CREDENTIALS_ENCRYPTION_KEY.encode("utf-8")
    try:
        _CREDENTIALS_CIPHER = Fernet(key)
    except Exception as exc:  # pragma: no cover - defensive
        raise CredentialEncryptionError(f"Invalid CREDENTIALS_ENCRYPTION_KEY: {exc}") from exc
    return _CREDENTIALS_CIPHER


def _encrypt_secret(value: str) -> str:
    if not value:
        return value
    cipher = _credentials_cipher()
    token = cipher.encrypt(value.encode("utf-8")).decode("utf-8")
    return f"enc:{token}"


def _decrypt_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if value.startswith("enc:"):
        token = value[4:]
        cipher = _credentials_cipher()
        try:
            decrypted = cipher.decrypt(token.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:  # pragma: no cover - data corruption safeguard
            raise CredentialEncryptionError("Unable to decrypt stored credential") from exc
        return decrypted
    return value


def _create_whop_session(license_key: str, email: Optional[str], metadata: Optional[Dict[str, Any]]) -> str:
    token = secrets.token_urlsafe(32)
    payload = json.dumps(metadata or {}, separators=(",", ":")) if metadata else None
    with _db_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO whop_sessions (token, license_key, email, metadata, created_at, consumed_at)
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (
                token,
                license_key,
                email,
                payload,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return token


def _consume_whop_session(token: str) -> None:
    if not token:
        return
    with _db_conn() as conn:
        conn.execute(
            "UPDATE whop_sessions SET consumed_at = ? WHERE token = ?",
            (datetime.now(timezone.utc).isoformat(), token),
        )


def _get_whop_session(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT license_key, email, metadata, created_at, consumed_at FROM whop_sessions WHERE token = ?",
            (token,),
        ).fetchone()
    if row is None:
        return None
    if row["consumed_at"]:
        return None
    try:
        created_at = datetime.fromisoformat(row["created_at"])
    except Exception:  # pragma: no cover - defensive parsing
        return None
    age = datetime.now(timezone.utc) - created_at
    if age.total_seconds() > WHOP_SESSION_TTL:
        return None
    metadata: Dict[str, Any] = {}
    if row["metadata"]:
        try:
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError:  # pragma: no cover - defensive
            metadata = {}
    return {
        "license_key": row["license_key"],
        "email": row["email"],
        "metadata": metadata,
        "created_at": row["created_at"],
    }


def _verify_whop_license(license_key: str) -> Dict[str, Any]:
    if not WHOP_API_KEY:
        raise HTTPException(status_code=503, detail="Whop integration is not configured")
    url = f"{WHOP_API_BASE}/api/v2/licenses/{license_key}"
    headers = {
        "Authorization": f"Bearer {WHOP_API_KEY}",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException as exc:
        logger.error("Whop license verification failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to reach Whop API") from exc
    if resp.status_code == 404:
        raise HTTPException(status_code=401, detail="Invalid Whop license")
    if resp.status_code >= 400:
        logger.error("Whop API responded with %s: %s", resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail="Whop API rejected the request")
    try:
        data = resp.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Malformed response from Whop API") from exc
    status = str(data.get("status") or data.get("state") or "").lower()
    if status not in {"active", "trialing", "paid"}:
        raise HTTPException(status_code=403, detail="Whop license is not active")
    email = (
        data.get("email")
        or data.get("user", {}).get("email")
        or data.get("customer", {}).get("email")
    )
    return {"license_key": license_key, "email": email, "raw": data}


def _hash_password_sha2048(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Derive a 2048-bit PBKDF2-SHA512 digest and return ``(hash_hex, salt_hex)``."""
    if salt is None:
        salt = secrets.token_hex(32)
    salt_bytes = bytes.fromhex(salt)
    derived = hashlib.pbkdf2_hmac(
        "sha512",
        password.encode("utf-8"),
        salt_bytes,
        200_000,
        dklen=256,
    )
    return derived.hex(), salt


def _create_session_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _db_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, datetime.now(timezone.utc).isoformat()),
        )
    return token


def _delete_session_token(token: str) -> None:
    if not token:
        return
    with _db_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def _authorization_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    header = authorization.strip()
    if not header:
        return None
    parts = header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return header


def _user_from_token(token: str) -> Optional[sqlite3.Row]:
    if not token:
        return None
    with _db_conn() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.username
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
    return row


def _require_user(authorization: Optional[str], session_token: Optional[str] = None) -> sqlite3.Row:
    token = _authorization_token(authorization)
    if not token and session_token:
        token = session_token
    if not token:
        raise HTTPException(status_code=401, detail="authorization required")
    user = _user_from_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return user


def _fetch_user_credentials(user_id: int, account_type: str) -> Optional[sqlite3.Row]:
    with _db_conn() as conn:
        return conn.execute(
            """
            SELECT api_key, api_secret, base_url
            FROM alpaca_credentials
            WHERE user_id = ? AND account_type = ?
            """,
            (user_id, account_type),
        ).fetchone()


def _save_user_credentials(
    user_id: int,
    account_type: str,
    api_key: str,
    api_secret: str,
    base_url: Optional[str],
) -> None:
    stored_key = _encrypt_secret(api_key)
    stored_secret = _encrypt_secret(api_secret)
    with _db_conn() as conn:
        conn.execute(
            """
            INSERT INTO alpaca_credentials (user_id, account_type, api_key, api_secret, base_url, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, account_type) DO UPDATE SET
                api_key = excluded.api_key,
                api_secret = excluded.api_secret,
                base_url = excluded.base_url,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                account_type,
                stored_key,
                stored_secret,
                base_url,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def _hash_password_sha2048(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Derive a 2048-bit PBKDF2-SHA512 digest and return ``(hash_hex, salt_hex)``."""
    if salt is None:
        salt = secrets.token_hex(32)
    salt_bytes = bytes.fromhex(salt)
    derived = hashlib.pbkdf2_hmac(
        "sha512",
        password.encode("utf-8"),
        salt_bytes,
        200_000,
        dklen=256,
    )
    return derived.hex(), salt


def _create_session_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _db_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, datetime.now(timezone.utc).isoformat()),
        )
    return token


def _authorization_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    header = authorization.strip()
    if not header:
        return None
    parts = header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return header


def _user_from_token(token: str) -> Optional[sqlite3.Row]:
    if not token:
        return None
    with _db_conn() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.username
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
    return row


def _require_user(authorization: Optional[str]) -> sqlite3.Row:
    token = _authorization_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="authorization required")
    user = _user_from_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return user


def _fetch_user_credentials(user_id: int, account_type: str) -> Optional[sqlite3.Row]:
    with _db_conn() as conn:
        return conn.execute(
            """
            SELECT api_key, api_secret, base_url
            FROM alpaca_credentials
            WHERE user_id = ? AND account_type = ?
            """,
            (user_id, account_type),
        ).fetchone()


def _save_user_credentials(
    user_id: int,
    account_type: str,
    api_key: str,
    api_secret: str,
    base_url: Optional[str],
) -> None:
    stored_key = _encrypt_secret(api_key)
    stored_secret = _encrypt_secret(api_secret)
    with _db_conn() as conn:
        conn.execute(
            """
            INSERT INTO alpaca_credentials (user_id, account_type, api_key, api_secret, base_url, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, account_type) DO UPDATE SET
                api_key = excluded.api_key,
                api_secret = excluded.api_secret,
                base_url = excluded.base_url,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                account_type,
                stored_key,
                stored_secret,
                base_url,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def _resolve_alpaca_credentials(account: str, user_id: Optional[int]) -> Dict[str, Optional[str]]:
    account_type = (account or "paper").strip().lower()
    if user_id is not None:
        row = _fetch_user_credentials(user_id, account_type)
        if row is not None:
            base_url = row["base_url"] or (
                ALPACA_BASE_URL_FUND if account_type == "funded" else ALPACA_BASE_URL_PAPER
            )
            try:
                key = _decrypt_secret(row["api_key"])
                secret = _decrypt_secret(row["api_secret"])
            except CredentialEncryptionError as exc:
                logger.error("Failed to decrypt stored credentials for user %s", user_id)
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return {
                "key": key,
                "secret": secret,
                "base_url": base_url,
            }
    if account_type == "funded":
        return {
            "key": ALPACA_KEY_FUND,
            "secret": ALPACA_SECRET_FUND,
            "base_url": ALPACA_BASE_URL_FUND,
        }
    return {
        "key": ALPACA_KEY_PAPER,
        "secret": ALPACA_SECRET_PAPER,
        "base_url": ALPACA_BASE_URL_PAPER,
    }

def _nocache() -> Dict[str,str]:
    return {"Cache-Control":"no-store, no-cache, must-revalidate, max-age=0", "Pragma":"no-cache", "Expires":"0"}

def _json(obj: Any, code: int = 200) -> JSONResponse:
    return JSONResponse(obj, status_code=code, headers=_nocache())

def _latest_run_dir() -> Optional[str]:
    d = latest_run_path()
    if d: return d
    if RUNS_ROOT.exists():
        runs = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
        if runs:
            runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return runs[0].as_posix()
    return None

def key(sym: str, tf: str) -> str:
    return f"{sym.upper()}|{tf}"

# ---------- system / gpu ----------
def _gpu_info() -> Dict[str, Any]:
    info = {"framework":"none","cuda_available":False,"devices":[]}
    try:
        import torch
        info["framework"]="torch"
        info["cuda_available"]=torch.cuda.is_available()
        if info["cuda_available"]:
            info["devices"]=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return info

def run_preflight():
    miss=[]; vers={}
    for name in ["pandas","yfinance","fastapi","psutil","matplotlib","sklearn"]:
        try:
            m=importlib.import_module(name); vers[name]=getattr(m,"__version__","unknown")
        except Exception:
            miss.append(name)
    total, used, free = shutil.disk_usage(os.getcwd())
    ram_avail_gb=None
    if psutil is not None:
        try:
            ram_avail_gb=round(psutil.virtual_memory().available/1024**3,2)
        except Exception:
            ram_avail_gb=None
    if ram_avail_gb is None:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_avail_gb = round(pages * page_size / 1024**3, 2)
        except (AttributeError, ValueError, OSError):
            ram_avail_gb = None
    return {"packages":{"ok":not miss,"missing":miss,"versions":vers},
            "hardware":{"disk_free_gb":round(free/1024**3,2),"ram_avail_gb":ram_avail_gb},
            "gpu":_gpu_info(),
            "time_utc":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(json.dumps(run_preflight(), indent=2), flush=True)
    yield

app = FastAPI(title="Neo Cortex AI Trainer", version="4.4", lifespan=lifespan)
if FORCE_HTTPS_REDIRECT:
    app.add_middleware(HTTPSRedirectMiddleware)


if ENABLE_HSTS and HSTS_HEADER_VALUE:

    @app.middleware("http")
    async def _add_security_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.scheme == "https":
            response.headers.setdefault("Strict-Transport-Security", HSTS_HEADER_VALUE)
        return response

app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
app.mount("/ui/liquidity", StaticFiles(directory=str(LIQUIDITY_DIR), html=True), name="liquidity-ui")
app.mount("/ui/enduserapp", StaticFiles(directory=str(ENDUSERAPP_DIR), html=True), name="enduserapp-ui")


# ---------- normalizers ----------
def standardize_ohlcv(raw: pd.DataFrame, ticker: Optional[str]=None) -> pd.DataFrame:
    if raw is None or raw.empty: raise ValueError("No data")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if ticker: df = df.xs(ticker, axis=1, level=-1, drop_level=True)
            else: df = df.droplevel(-1, axis=1)
        except Exception:
            df = df.droplevel(-1, axis=1)
    m = {c.lower().replace("_",""): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in m: return m[c]
        return None
    cols = {"Open":pick("open"), "High":pick("high"), "Low":pick("low"),
            "Close":pick("close","adjclose","adjustedclose"), "Volume":pick("volume")}
    if any(v is None for v in cols.values()): raise KeyError("Missing OHLCV columns")
    out = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k,v in cols.items()})
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    else:
        try: out.index = out.index.tz_convert("UTC")
        except Exception: out.index = out.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    out = out.sort_index().dropna(how="any")
    if out.empty: raise ValueError("Empty after cleaning")
    return out

def _save_price_preview(df: pd.DataFrame, tkr: str, run_dir: Path):
    try:
        plt.figure(figsize=(8,3)); plt.plot(df.index, df["Close"].values, lw=1.1)
        plt.title(f"{tkr} - Close"); plt.grid(True, alpha=.25); plt.tight_layout()
        plt.savefig(run_dir/"yfinance_price.png", dpi=130); plt.close()
    except Exception:
        pass

# ---------- pydantic schemas ----------
class TrainReq(BaseModel):
    ticker: str
    period: Optional[str] = "6mo"
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1h"
    max_iter: int = 300
    mode: str = "HFT"         # HFT/LFT
    pine_mode: str = "OFF"    # OFF/SCALPER/ULTRA/SWING (fed to env for model tweaks)
    aggressiveness: str = "normal"
    sides: str = "BOTH"
    entry_thr: Optional[float] = None
    z_thr: Optional[float] = None
    vol_k: Optional[float] = None
    source: str = "both"      # yfinance/ingest/both
    merge_sources: bool = True
    use_fundamentals: bool = True

class TrainMultiReq(BaseModel):
    tickers: List[str]
    period: Optional[str] = "6mo"
    interval: str = "1h"
    max_iter: int = 250
    mode: str = "HFT"
    pine_mode: str = "OFF"
    aggressiveness: str = "normal"
    sides: str = "BOTH"
    source: str = "both"
    merge_sources: bool = True
    use_fundamentals: bool = True

class IdleReq(BaseModel):
    name: str
    tickers: List[str]
    every_sec: int = 3600
    period: Optional[str] = "6mo"
    interval: str = "1h"
    max_iter: int = 250
    mode: str = "HFT"
    pine_mode: str = "OFF"
    aggressiveness: str = "normal"
    sides: str = "BOTH"
    source: str = "both"
    merge_sources: bool = True
    use_fundamentals: bool = True

# ---------- exogenous store ----------
def upsert_bars(sym: str, tf: str, bars: pd.DataFrame) -> pd.DataFrame:
    k = key(sym, tf)
    cur = Buffers.get(k, pd.DataFrame())
    df = pd.concat([cur, bars]).sort_index().groupby(level=0).last() if len(cur) else bars
    Buffers[k] = df.tail(20000)
    IngestStats["candles"] += len(bars)
    return Buffers[k]

def upsert_exog(sym: str, rows: pd.DataFrame) -> pd.DataFrame:
    sU = sym.upper()
    cur = Exog.get(sU, pd.DataFrame())
    df = pd.concat([cur, rows]).sort_index().groupby(level=0).last() if len(cur) else rows
    Exog[sU] = df.tail(20000)
    IngestStats["features"] += len(rows)
    return Exog[sU]

def _get_exog(sym: str, idx: pd.DatetimeIndex, include_fa: bool) -> pd.DataFrame:
    sU = sym.upper()
    ex = Exog.get(sU, pd.DataFrame())
    ex = ex.reindex(idx)
    ex = ex.fillna(0.0) if ex is not None else pd.DataFrame(index=idx)
    if include_fa:
        for k in ("fa_pe","fa_pb","fa_eps","fa_div"):
            if k in ex.columns: ex[k] = pd.to_numeric(ex[k], errors="coerce").fillna(method="ffill").fillna(0.0)
    return ex.fillna(0.0)

# ---------- routes ----------
@app.get("/")
def root():
    return {"ok": True, "msg": "Neo Cortex AI API ready"}


@app.get("/ngrok/cloud-endpoint", response_class=HTMLResponse)
async def ngrok_cloud_endpoint(request: Request) -> HTMLResponse:
    """Render a friendly landing page for ngrok Cloud Endpoints."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if not host:
        if not DEFAULT_PUBLIC_BASE_URL:
            raise HTTPException(
                status_code=500,
                detail="DEFAULT_PUBLIC_BASE_URL must be set when the application is served without Host headers.",
            )
        base_url = DEFAULT_PUBLIC_BASE_URL
    else:
        base_url = f"{proto}://{host}".rstrip("/")
    webhook_url = f"{base_url}/alpaca/webhook"
    html = (
        NGROK_ENDPOINT_TEMPLATE.replace("{{WEBHOOK_URL}}", webhook_url)
        .replace("{{BASE_URL}}", base_url)
    )
    return HTMLResponse(content=html, status_code=200)

@app.get("/preflight")
def preflight():
    return run_preflight()

@app.get("/gpu-info")
def gpu_info():
    return _gpu_info()

# --- auth: user registration and login ---
class AuthReq(BaseModel):
    username: str
    password: str
    admin_key: Optional[str] = None
    require_admin: bool = False


async def _auth_req_from_request(request: Request) -> AuthReq:
    """Parse an AuthReq from JSON, form-encoded, or query parameters."""

    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    data: Dict[str, Any] = {}

    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("failed to parse multipart auth payload: %s", exc)
        else:
            data = {k: v for k, v in form.items() if v is not None}
    else:
        body_bytes = await request.body()
        parsed: Optional[Dict[str, Any]] = None
        if body_bytes:
            try:
                candidate = json.loads(body_bytes)
            except json.JSONDecodeError:
                candidate = None
            if isinstance(candidate, dict):
                parsed = candidate
            else:
                parsed_qs = parse_qs(body_bytes.decode("utf-8", errors="ignore"))
                if parsed_qs:
                    parsed = {k: v[-1] for k, v in parsed_qs.items() if v}
        if parsed:
            data = parsed

    if not data and request.query_params:
        data = dict(request.query_params)

    if not data:
        raise HTTPException(status_code=400, detail="username and password required")

    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            value = value[-1]
        if value is None:
            continue
        normalized[key] = value if isinstance(value, str) else str(value)

    try:
        return AuthReq(**normalized)
    except ValidationError:
        raise HTTPException(status_code=400, detail="username and password required")


class CredentialReq(BaseModel):
    account_type: str = "paper"
    api_key: str
    api_secret: str
    base_url: Optional[str] = None


class WhopRegistrationReq(BaseModel):
    token: str
    username: str
    password: str
    api_key: str
    api_secret: str
    account_type: Literal["paper", "funded"] = "funded"
    base_url: Optional[str] = None


class AlpacaWebhookTest(BaseModel):
    symbol: str = "SPY"
    quantity: float = 1.0
    price: float = 0.0
    side: str = "buy"
    status: str = "filled"
    event: str = "trade_update"
    timestamp: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class PaperTradeOrderRequest(BaseModel):
    instrument: Literal["option", "future"] = "option"
    symbol: str
    side: Literal["long", "short"] = "long"
    quantity: float = Field(default=1.0, gt=0)
    price: float = Field(default=1.0, ge=0)
    option_type: Optional[Literal["call", "put"]] = None
    expiry: Optional[str] = None
    strike: Optional[float] = Field(default=None, ge=0)
    future_month: Optional[str] = None
    future_year: Optional[int] = Field(default=None, ge=2000, le=2100)


@app.get("/auth/whop/start")
async def whop_start(request: Request, next: Optional[str] = None):
    if not WHOP_PORTAL_URL:
        raise HTTPException(status_code=404, detail="Whop integration is not configured")
    callback_url = str(request.url_for("whop_callback"))
    params: Dict[str, str] = {}
    if next:
        next = next.strip()
        if next.startswith("/"):
            params["next"] = next
    if params:
        callback_url = f"{callback_url}?{urlencode(params)}"
    encoded_callback = quote(callback_url, safe="")
    destination = WHOP_PORTAL_URL
    if "{{callback}}" in destination:
        destination = destination.replace("{{callback}}", encoded_callback)
    if "{callback}" in destination:
        destination = destination.replace("{callback}", encoded_callback)
    if destination == WHOP_PORTAL_URL:
        sep = "&" if "?" in destination else "?"
        destination = f"{destination}{sep}callback={encoded_callback}"
    return RedirectResponse(destination)


@app.get("/auth/whop/callback")
async def whop_callback(
    request: Request,
    license_key: Optional[str] = None,
    state: Optional[str] = None,
    next: Optional[str] = None,
):
    license_key = (license_key or "").strip()
    if not license_key:
        raise HTTPException(status_code=400, detail="license_key is required")
    verification = _verify_whop_license(license_key)
    metadata = {
        "state": state,
        "verification": verification.get("raw"),
    }
    token = _create_whop_session(license_key, verification.get("email"), metadata)
    query: Dict[str, str] = {"whop_token": token}
    next_param = (next or "").strip()
    if next_param and next_param.startswith("/"):
        query["next"] = next_param
    redirect_url = str(request.url_for("login_page"))
    redirect_url = f"{redirect_url}?{urlencode(query)}"
    return RedirectResponse(redirect_url)


@app.get("/auth/whop/session")
async def whop_session(token: str):
    session = _get_whop_session(token.strip())
    if not session:
        return _json({"ok": False, "detail": "Invalid or expired Whop session"}, 404)
    return _json(
        {
            "ok": True,
            "license_key": session["license_key"],
            "email": session.get("email"),
            "created_at": session["created_at"],
        }
    )


@app.post("/register/whop")
async def register_whop(req: WhopRegistrationReq):
    token = (req.token or "").strip()
    session = _get_whop_session(token)
    if not session:
        return _json({"ok": False, "detail": "Whop session expired or invalid"}, 400)
    uname = req.username.strip().lower()
    if not uname or not req.password:
        return _json({"ok": False, "detail": "username and password required"}, 400)
    api_key = req.api_key.strip()
    api_secret = req.api_secret.strip()
    if not api_key or not api_secret:
        return _json({"ok": False, "detail": "api_key and api_secret are required"}, 400)
    acct_type = (req.account_type or "funded").strip().lower()
    if acct_type not in {"paper", "funded"}:
        return _json({"ok": False, "detail": "account_type must be 'paper' or 'funded'"}, 400)
    base_url = (req.base_url or "").strip()
    if not base_url:
        base_url = ALPACA_BASE_URL_FUND if acct_type == "funded" else ALPACA_BASE_URL_PAPER
    pw_hash, salt = _hash_password_sha2048(req.password)
    try:
        with _db_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, salt, is_admin, created_at) VALUES (?, ?, ?, ?, ?)",
                (uname, pw_hash, salt, 0, datetime.now(timezone.utc).isoformat()),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return _json({"ok": False, "detail": "username already exists"}, 400)
    try:
        _save_user_credentials(user_id, acct_type, api_key, api_secret, base_url)
    except CredentialEncryptionError as exc:
        logger.error("Failed to store credentials for new Whop user %s", user_id)
        with _db_conn() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return _json({"ok": False, "detail": str(exc)}, 500)
    _consume_whop_session(token)
    session_token = _create_session_token(user_id)
    resp = _json(
        {
            "ok": True,
            "token": session_token,
            "username": uname,
            "session_cookie": SESSION_COOKIE_NAME,
            "account_type": acct_type,
            "base_url": base_url,
        }
    )
    resp.set_cookie(
        SESSION_COOKIE_NAME,
        session_token,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        max_age=SESSION_COOKIE_MAX_AGE,
        samesite=SESSION_COOKIE_SAMESITE,
        path="/",
    )
    return resp


@app.post("/register")
async def register(request: Request):
    """
    Create a new user account backed by the SQLite credential store. Passwords are
    hashed with a 2048-bit PBKDF2-SHA512 digest and salted prior to being persisted.
    """
    req = await _auth_req_from_request(request)
    uname = req.username.strip().lower()
    if not uname or not req.password:
        return _json({"ok": False, "detail": "username and password required"}, 400)
    if ADMIN_PRIVATE_KEY and (req.admin_key or "").strip() != ADMIN_PRIVATE_KEY:
        return _json({"ok": False, "detail": "invalid admin key"}, 403)
    pw_hash, salt = _hash_password_sha2048(req.password)
    try:
        with _db_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, is_admin, created_at) VALUES (?, ?, ?, ?, ?)",
                (uname, pw_hash, salt, 1, datetime.now(timezone.utc).isoformat()),
            )
    except sqlite3.IntegrityError:
        return _json({"ok": False, "detail": "username already exists"}, 400)
    return _json({"ok": True, "created": uname})

@app.post("/login")
async def login(request: Request):
    """
    Authenticate a user and return a bearer token tied to the SQLite credential store.
    The token may be supplied via the ``Authorization: Bearer`` header for endpoints that
    manage user-specific Alpaca credentials.
    """
    req = await _auth_req_from_request(request)
    uname = req.username.strip().lower()
    if not uname or not req.password:
        return _json({"ok": False, "detail": "username and password required"}, 400)
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT id, password_hash, salt, is_admin FROM users WHERE username = ?",
            (uname,),
        ).fetchone()
    if row is None:
        return _json({"ok": False, "detail": "invalid credentials"}, 401)
    expected_hash, _ = _hash_password_sha2048(req.password, row["salt"])
    if expected_hash != row["password_hash"]:
        return _json({"ok": False, "detail": "invalid credentials"}, 401)
    if req.require_admin and not row["is_admin"]:
        return _json({"ok": False, "detail": "admin access required"}, 403)
    token = _create_session_token(row["id"])
    resp = _json(
        {
            "ok": True,
            "token": token,
            "username": uname,
            "session_cookie": SESSION_COOKIE_NAME,
            "is_admin": bool(row["is_admin"]),
        }
    )
    resp.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        max_age=SESSION_COOKIE_MAX_AGE,
        samesite=SESSION_COOKIE_SAMESITE,
        path="/",
    )
    return resp


@app.post("/logout")
async def logout(
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    token = _authorization_token(authorization) or session_token
    if token:
        _delete_session_token(token)
    resp = _json({"ok": True})
    resp.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        samesite=SESSION_COOKIE_SAMESITE,
        secure=SESSION_COOKIE_SECURE,
    )
    return resp


@app.get("/alpaca/credentials")
async def list_alpaca_credentials(
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    """Return the Alpaca credential entries associated with the authenticated user."""
    user = _require_user(authorization, session_token)
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT account_type, api_key, base_url, updated_at
            FROM alpaca_credentials
            WHERE user_id = ?
            ORDER BY account_type
            """,
            (user["id"],),
        ).fetchall()
    credentials = []
    for row in rows:
        try:
            api_key = _decrypt_secret(row["api_key"])
        except CredentialEncryptionError as exc:
            logger.error("Failed to decrypt stored credentials for user %s", user["id"])
            return _json({"ok": False, "detail": str(exc)}, 500)
        credentials.append(
            {
                "account_type": row["account_type"],
                "api_key": api_key,
                "base_url": row["base_url"],
                "updated_at": row["updated_at"],
            }
        )
    return _json({"ok": True, "credentials": credentials})


@app.post("/alpaca/credentials")
async def set_alpaca_credentials(
    req: CredentialReq,
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    """Create or update Alpaca API credentials for the authenticated user."""
    user = _require_user(authorization, session_token)
    acct_type = (req.account_type or "paper").strip().lower()
    if acct_type not in {"paper", "funded"}:
        return _json({"ok": False, "detail": "account_type must be 'paper' or 'funded'"}, 400)
    api_key = req.api_key.strip()
    api_secret = req.api_secret.strip()
    if not api_key or not api_secret:
        return _json({"ok": False, "detail": "api_key and api_secret are required"}, 400)
    base_url = (req.base_url or "").strip()
    if not base_url:
        base_url = ALPACA_BASE_URL_FUND if acct_type == "funded" else ALPACA_BASE_URL_PAPER
    try:
        _save_user_credentials(user["id"], acct_type, api_key, api_secret, base_url)
    except CredentialEncryptionError as exc:
        logger.error("Failed to persist credentials for user %s", user["id"])
        return _json({"ok": False, "detail": str(exc)}, 500)
    return _json({"ok": True, "account_type": acct_type, "base_url": base_url})

# --- account data: positions and P&L ---
@app.get("/positions")
async def get_positions(
    account: str = "paper",
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    """Fetch the current positions for the specified Alpaca account."""

    acct_type = (account or "paper").lower()
    token = _authorization_token(authorization) or session_token
    user = None
    if token:
        user = _user_from_token(token)
        if user is None:
            return _json({"ok": False, "detail": "invalid or expired token"}, 401)

    creds = _resolve_alpaca_credentials(acct_type, user["id"] if user else None)
    key = creds.get("key")
    secret = creds.get("secret")
    base_url = creds.get("base_url")
    if not key or not secret:
        return _json({"ok": False, "detail": "Alpaca credentials not configured"}, 500)

    url = f"{base_url}/v2/positions"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    try:
        resp = requests.get(url, headers=headers)
        return _json({"ok": True, "positions": resp.json()})
    except Exception as exc:
        return _json({"ok": False, "detail": f"Failed to fetch positions: {exc}"}, 500)


@app.post("/positions/{symbol}/close")
async def close_position(
    symbol: str,
    account: str = "paper",
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    """Close a single Alpaca position for the authenticated user."""

    if not symbol:
        return _json({"ok": False, "detail": "symbol is required"}, 400)

    token = _authorization_token(authorization) or session_token
    if not token:
        return _json({"ok": False, "detail": "authorization required"}, 401)

    user = _user_from_token(token)
    if user is None:
        return _json({"ok": False, "detail": "invalid or expired token"}, 401)

    creds = _resolve_alpaca_credentials(account, user["id"])
    key = creds.get("key")
    secret = creds.get("secret")
    base_url = creds.get("base_url")
    if not key or not secret:
        return _json({"ok": False, "detail": "Alpaca credentials not configured"}, 500)

    url = f"{base_url}/v2/positions/{symbol}"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    try:
        resp = requests.delete(url, headers=headers)
    except Exception as exc:
        return _json({"ok": False, "detail": f"Failed to close position: {exc}"}, 500)

    alpaca_payload: Optional[Any] = None
    try:
        if resp.content:
            alpaca_payload = resp.json()
    except ValueError:
        alpaca_payload = None

    if 200 <= resp.status_code < 300:
        detail = (
            (alpaca_payload or {}).get("message")
            if isinstance(alpaca_payload, dict)
            else None
        ) or f"Closed position for {symbol.upper()}"
        body: Dict[str, Any] = {"ok": True, "detail": detail}
        if alpaca_payload is not None:
            body["alpaca"] = alpaca_payload
        return _json(body, resp.status_code)

    error_detail: Optional[str] = None
    if isinstance(alpaca_payload, dict):
        error_detail = (
            alpaca_payload.get("message")
            or alpaca_payload.get("error")
            or alpaca_payload.get("detail")
        )
    if not error_detail:
        error_detail = resp.text.strip() or "Failed to close position"
    error_body: Dict[str, Any] = {"ok": False, "detail": error_detail}
    if alpaca_payload is not None:
        error_body["alpaca"] = alpaca_payload
    status_code = resp.status_code or 502
    return _json(error_body, status_code)

@app.get("/pnl")
async def get_pnl(
    account: str = "paper",
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    """Compute unrealized P&L for the authenticated user's Alpaca account."""

    acct_type = (account or "paper").lower()
    token = _authorization_token(authorization) or session_token
    user = None
    if token:
        user = _user_from_token(token)
        if user is None:
            return _json({"ok": False, "detail": "invalid or expired token"}, 401)

    creds = _resolve_alpaca_credentials(acct_type, user["id"] if user else None)
    key = creds.get("key")
    secret = creds.get("secret")
    base_url = creds.get("base_url")
    if not key or not secret:
        return _json({"ok": False, "detail": "Alpaca credentials not configured"}, 500)

    pos_url = f"{base_url}/v2/positions"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    try:
        pos_resp = requests.get(pos_url, headers=headers)
        pos_list = pos_resp.json() or []
    except Exception as exc:
        return _json({"ok": False, "detail": f"Failed to compute P&L: {exc}"}, 500)

    if not pos_list:
        return _json({"ok": True, "total_pnl": 0.0, "positions": []})

    results = []
    total_pnl = 0.0
    for position in pos_list:
        qty = float(position.get("qty") or position.get("quantity") or 0)
        cost_basis = float(position.get("cost_basis") or 0)
        market_value = float(position.get("market_value") or 0)
        if position.get("unrealized_pl") is not None:
            pnl = float(position.get("unrealized_pl"))
        else:
            pnl = market_value - cost_basis
        total_pnl += pnl
        results.append(
            {
                "symbol": position.get("symbol"),
                "quantity": qty,
                "avg_entry_price": float(position.get("avg_entry_price") or 0),
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pl": pnl,
            }
        )

    return _json({"ok": True, "total_pnl": total_pnl, "positions": results})


@app.get("/strategy/liquidity-sweeps")
def strategy_liquidity_sweeps(ticker: str, session_date: Optional[str] = None, interval: str = "1m"):
    ticker = (ticker or "").strip()
    if not ticker:
        return _json({"ok": False, "detail": "ticker query parameter is required"}, 400)
    try:
        analysis = analyze_liquidity_session(
            ticker,
            session_date=session_date,
            interval=interval,
            asset_dir=LIQUIDITY_ASSETS,
        )
    except StrategyError as exc:
        return _json({"ok": False, "detail": str(exc)}, 400)
    except Exception as exc:
        return _json({"ok": False, "detail": f"{type(exc).__name__}: {exc}"}, 500)

    heatmap_info = analysis.get("heatmap")
    if heatmap_info:
        filename = heatmap_info.get("relative_url")
        if filename:
            heatmap_info["public_url"] = f"/public/liquidity/assets/{filename}"
    analysis["ok"] = True
    return _json(analysis)

# -----------------------------------------------------------------------------
# Alpaca webhook handler
#
# Alpaca can be configured to send trade updates and other events to a webhook
# endpoint. This handler accepts those webhook callbacks and logs the payload.
# To test paper trading events, point your Alpaca paper account's webhook URL at
# `https://<your-ngrok-url>/alpaca/webhook`. For example, when using the
# run_with_ngrok.py script included in this repository, ngrok will provide a
# public URL that tunnels traffic to your local server running on port 8000.
# Copy that URL into your Alpaca console under "Paper Trading" → "API Config"
# → "Webhook URL".
#
# NOTE: This handler does not perform any trading actions. It simply returns
# ``{"ok": true}`` and logs the received JSON. You can extend it to update
# your own database or notify your front‑end. In this hosted context, any
# external HTTP requests (e.g. to Alpaca) are not executed; the code is
# illustrative only.
@app.post("/alpaca/webhook")
async def alpaca_webhook(req: Request):
    """
    Receive webhook callbacks from Alpaca. This endpoint accepts any JSON
    payload and logs it. It returns ``{"ok": True}`` on success. If you set
    ``ALPACA_WEBHOOK_SECRET`` in your environment, the request's header
    ``X‑Webhook‑Signature`` will be verified against this secret using HMAC
    SHA‑256. Mismatched signatures return a 400.
    """
    body_bytes = await req.body()
    try:
        payload = await req.json()
    except Exception:
        payload = None
    # optional signature verification
    secret = os.getenv("ALPACA_WEBHOOK_SECRET")
    if secret:
        sig_header = req.headers.get("X-Webhook-Signature")
        if not sig_header:
            return _json({"ok": False, "detail": "missing signature"}, 400)

        import hmac, hashlib, base64

    sig_header = req.headers.get("X-Webhook-Signature")
    if secret:
        import hmac, hashlib, base64

        if not sig_header:
            return _json({"ok": False, "detail": "missing signature"}, 400)

        if not sig_header:
            return _json({"ok": False, "detail": "missing signature"}, 400)

        import hmac, hashlib, base64
      
        digest = hmac.new(secret.encode(), body_bytes, hashlib.sha256).digest()
        expected = base64.b64encode(digest).decode()
        if not hmac.compare_digest(expected, sig_header):
            return _json({"ok": False, "detail": "invalid signature"}, 400)

    # In a production system you might persist the payload to a database or
    # notify other services. Here we simply print it to stdout.
    logger.info("[Alpaca webhook] %s", json.dumps(payload, indent=2))
    return _json({"ok": True})


@app.post("/alpaca/webhook/test")
def alpaca_webhook_test(req: AlpacaWebhookTest):
    payload = req.raw.copy() if isinstance(req.raw, dict) else {
        "event": req.event,
        "order": {
            "symbol": req.symbol,
            "qty": req.quantity,
            "filled_avg_price": req.price,
            "side": req.side,
            "status": req.status,
        },
    }
    payload.setdefault("event", req.event)
    payload.setdefault("timestamp", req.timestamp or datetime.now(timezone.utc).isoformat())

    body_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    signature = None
    secret = os.getenv("ALPACA_WEBHOOK_SECRET")
    if secret:
        import hmac, hashlib, base64

        digest = hmac.new(secret.encode(), body_bytes, hashlib.sha256).digest()
        signature = base64.b64encode(digest).decode()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fname = f"test-{req.symbol.upper()}-{stamp}.json"
    path = ALPACA_TEST_DIR / fname
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return _json(
        {
            "ok": True,
            "payload": payload,
            "suggested_signature": signature,
            "artifact": f"/public/{path.relative_to(PUBLIC_DIR)}",
            "default_webhook_url": f"{DEFAULT_PUBLIC_BASE_URL}/alpaca/webhook" if DEFAULT_PUBLIC_BASE_URL else None,
        }
    )


@app.get("/alpaca/webhook/tests")
def alpaca_webhook_tests():
    """Return recently generated Alpaca webhook test artifacts."""
    tests = []
    for file_path in sorted(ALPACA_TEST_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        symbol = (
            (payload.get("order") or {}).get("symbol")
            or payload.get("symbol")
            or payload.get("ticker")
            or ""
        )
        stat = file_path.stat()
        tests.append(
            {
                "name": file_path.name,
                "symbol": symbol,
                "created": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "size": stat.st_size,
                "url": f"/public/{file_path.relative_to(PUBLIC_DIR)}",
            }
        )
    return _json({"ok": True, "tests": tests})


@app.get("/papertrade/status")
async def papertrade_status():
    async with PAPERTRADE_STATE_LOCK:
        _load_papertrade_state()
        dashboard = _papertrade_dashboard()
    return _json({"ok": True, "dashboard": dashboard})


@app.post("/papertrade/orders")
async def papertrade_orders(req: PaperTradeOrderRequest):
    symbol = (req.symbol or "").strip().upper()
    if not symbol:
        return _json({"ok": False, "detail": "symbol is required"}, 400)

    instrument = req.instrument
    side = req.side
    quantity = float(req.quantity)
    price = float(req.price)
    if price <= 0:
        price = 1.0

    order_payload: Dict[str, Any] = {
        "symbol": symbol,
        "instrument": instrument,
        "side": side,
        "quantity": round(quantity, 4),
        "price": round(price, 4),
    }

    if instrument == "option":
        option_type = (req.option_type or "call").lower()
        expiry = (req.expiry or "").strip()
        strike = float(req.strike) if req.strike is not None else None
        order_payload.update(
            {
                "option_type": option_type,
                "expiry": expiry or None,
                "strike": round(strike, 4) if strike is not None else None,
            }
        )
    else:
        month = (req.future_month or "").strip().upper()
        year = int(req.future_year) if req.future_year is not None else None
        order_payload.update(
            {
                "future_month": month or None,
                "future_year": year,
            }
        )

    ai_decision = _run_ai_trainer(order_payload)
    status = "executed" if ai_decision.get("action") == "execute" else "review"

    order_record = {
        "id": secrets.token_hex(6),
        **order_payload,
        "status": status,
        "ai": ai_decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry_price": round(price, 4),
        "noise_seed": secrets.token_hex(6),
    }

    async with PAPERTRADE_STATE_LOCK:
        _load_papertrade_state()
        orders = PaperTradeState.setdefault("orders", [])
        orders.append(order_record)
        # keep history manageable
        if len(orders) > 500:
            del orders[:-500]
        _save_papertrade_state()
        dashboard = _papertrade_dashboard()

    response_order = dict(order_record)
    response_order.pop("noise_seed", None)

    return _json({"ok": True, "order": response_order, "dashboard": dashboard})

# --- ingestion: TradingView (signals + bars optional) ---
@app.post("/webhook/tradingview")
async def webhook_tv(req: Request):
    try:
        j = await req.json()
    except Exception:
        return _json({"ok": False, "detail": "invalid json"}, 400)

    tkr = (j.get("ticker") or "").upper()
    tf  = j.get("interval") or "1h"
    ts  = pd.to_datetime(j.get("time"), utc=True, errors="coerce")

    if all(k in j for k in ("open","high","low","close","volume")) and ts is not pd.NaT:
        bar = pd.DataFrame([{
            "Open": float(j["open"]), "High": float(j["high"]), "Low": float(j["low"]),
            "Close": float(j["close"]), "Volume": float(j["volume"])
        }], index=[ts])
        upsert_bars(tkr, tf, bar)

    feats = j.get("features") or {}
    sig = (j.get("signal") or "").lower()
    if sig in ("long","short","flat"):
        feats.setdefault("tv_long",  1.0 if sig=="long"  else 0.0)
        feats.setdefault("tv_short", 1.0 if sig=="short" else 0.0)
        feats.setdefault("tv_flat",  1.0 if sig=="flat"  else 0.0)
    if feats and ts is not pd.NaT:
        ex = pd.DataFrame([feats], index=[ts])
        upsert_exog(tkr, ex)

    IngestStats["tradingview"] += 1
    return _json({"ok": True})

# --- webhook: Alpaca broker (disabled) ---
@app.post("/webhook/alpaca")
async def webhook_alpaca(req: Request):
    """
    Placeholder for a future Alpaca trade execution webhook.

    Executing high‑stakes financial transactions (such as buying or selling securities)
    is disabled in this environment. Any POSTed payload will be ignored and an error
    response returned. To integrate live trading functionality, you would need to
    use the Alpaca Orders API with appropriate safeguards, and run the code outside
    of this assistant.
    """
    return _json({"ok": False, "detail": "Trade execution via Alpaca is disabled in this environment"}, 403)

# --- ingestion: generic candles (Robinhood/Webull) ---
@app.post("/ingest/candles")
async def ingest_candles(req: Request):
    j = await req.json()
    tkr = (j.get("ticker") or "").upper()
    tf  = j.get("interval") or "1h"
    prov = (j.get("provider") or "").lower()
    rows = j.get("rows") or []
    if not tkr or not rows:
        return _json({"ok": False, "detail":"missing rows/ticker"},400)
    df = pd.DataFrame([{
        "Open": float(r["open"]), "High": float(r["high"]), "Low": float(r["low"]),
        "Close": float(r["close"]), "Volume": float(r["volume"]),
        "time": r.get("time")
    } for r in rows])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    upsert_bars(tkr, tf, df)
    if "robinhood" in prov: IngestStats["robinhood"] += len(rows)
    if "webull" in prov:    IngestStats["webull"]    += len(rows)
    return _json({"ok": True, "n": len(df)})

# --- ingestion: arbitrary features/fundamentals ---
@app.post("/ingest/features")
async def ingest_features(req: Request):
    j = await req.json()
    tkr = (j.get("ticker") or "").upper()
    rows = j.get("rows") or []
    if not tkr or not rows:
        return _json({"ok": False, "detail":"missing"},400)
    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        return _json({"ok": False, "detail":"rows missing 'time'"},400)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    upsert_exog(tkr, df)
    return _json({"ok": True, "n": len(df)})

@app.post("/ingest/fundamentals")
async def ingest_fa(req: Request):
    j = await req.json()
    tkr = (j.get("ticker") or "").upper()
    ts = pd.to_datetime(j.get("time"), utc=True, errors="coerce")
    if not tkr or ts is pd.NaT:
        return _json({"ok": False, "detail":"missing"},400)
    d = {k: float(j[k]) for k in ("fa_pe","fa_pb","fa_eps","fa_div") if k in j}
    df = pd.DataFrame([d], index=[ts])
    upsert_exog(tkr, df)
    return _json({"ok": True})

# ---------- training ----------
@app.post("/train")
def train(req: TrainReq):
    import yfinance as yf

    os.environ["PINE_TICKER"]= req.ticker.upper()
    os.environ["PINE_TF"]    = (req.mode or "HFT").upper()
    os.environ["PINE_MODE"]  = (req.pine_mode or "OFF").upper()

    # yfinance (safe combos)
    yf_df = None
    try:
        per = req.period or "6mo"
        if req.interval in ("1m","2m") and per not in ("1d","5d","1mo"): per = "1mo"
        yfd = yf.download(req.ticker, period=per, interval=req.interval,
                          progress=False, group_by="column", threads=False, auto_adjust=True)
        if yfd is not None and not yfd.empty:
            yf_df = standardize_ohlcv(yfd, ticker=req.ticker)
    except Exception:
        yf_df = None

    # external
    ext_df = None
    k = key(req.ticker, req.interval)
    if k in Buffers and not Buffers[k].empty:
        ext_df = Buffers[k][["Open","High","Low","Close","Volume"]].copy()

    # merge / prefer
    if req.source == "yfinance": base = yf_df
    elif req.source == "ingest": base = ext_df
    else:
        if req.merge_sources and (yf_df is not None) and (ext_df is not None):
            base = pd.concat([yf_df, ext_df]).sort_index().groupby(level=0).last()
        else:
            base = ext_df if ext_df is not None else yf_df

    if base is None or base.empty:
        return _json({"ok": False, "detail":"No data from sources"}, 400)

    exog = _get_exog(req.ticker, base.index, include_fa=req.use_fundamentals)

    presets = {"chill":1.15,"normal":0.95,"spicy":0.75,"insane":0.55}
    entry_thr = req.entry_thr if req.entry_thr is not None else presets.get((req.aggressiveness or "normal"), 0.95)

    feat = build_features(
        base,
        hftMode=(req.mode.upper()=="HFT"),
        zThrIn=req.z_thr,
        volKIn=req.vol_k,
        entryThrIn=entry_thr,
        exog=exog
    )

    # ---- guard around torch._dynamo error so API doesn't 500 ----
    try:
        acc, n = train_and_save(feat, max_iter=int(req.max_iter))
    except ModuleNotFoundError as e:
        if "torch._dynamo" in str(e):
            return _json({
                "ok": False,
                "detail": "PyTorch install is inconsistent (missing torch._dynamo). "
                          "Install PyTorch 2.x GPU/CPU wheels or remove any third-party torch-compile shim."
            }, 500)
        raise
    except Exception as e:
        return _json({"ok": False, "detail": f"{type(e).__name__}: {e}"}, 500)

    d = _latest_run_dir()
    if d: _save_price_preview(base, req.ticker, Path(d))
    return _json({"ok": True, "train_acc": round(acc,3), "n": int(n), "run_dir": d})

@app.post("/train/multi")
def train_multi(req: TrainMultiReq):
    import concurrent.futures as cf
    results, errors = {}, {}
    def _one(tkr: str):
        sub = TrainReq(ticker=tkr, period=req.period, interval=req.interval, max_iter=req.max_iter,
                       mode=req.mode, pine_mode=req.pine_mode, aggressiveness=req.aggressiveness,
                       sides=req.sides, source=req.source, merge_sources=req.merge_sources,
                       use_fundamentals=req.use_fundamentals)
        try:
            resp = train(sub)
            return tkr, json.loads(resp.body.decode("utf-8"))
        except Exception as e:
            return tkr, {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    with cf.ThreadPoolExecutor(max_workers=min(8, max(1, len(req.tickers)))) as exe:
        futs = [exe.submit(_one, t) for t in req.tickers]
        for f in cf.as_completed(futs):
            tkr, payload = f.result()
            (results if payload.get("ok") else errors)[tkr] = payload
    return _json({"ok": True, "results": results, "errors": errors})

# ---------- idle loop (async, AnyIO-safe) ----------
async def _idle_loop(name: str, spec: IdleReq):
    while True:
        try:
            train_multi(TrainMultiReq(
                tickers=spec.tickers, period=spec.period, interval=spec.interval, max_iter=spec.max_iter,
                mode=spec.mode, pine_mode=spec.pine_mode, aggressiveness=spec.aggressiveness, sides=spec.sides,
                source=spec.source, merge_sources=spec.merge_sources, use_fundamentals=spec.use_fundamentals
            ))
        except Exception as e:
            print(f"[idle:{name}] {type(e).__name__}: {e}", flush=True)
        await asyncio.sleep(max(30, int(spec.every_sec)))

@app.post("/idle/start")
async def idle_start(req: IdleReq):
    if req.name in IdleTasks and not IdleTasks[req.name].done():
        return _json({"ok": False, "detail":"task already running"}, 400)
    loop = asyncio.get_running_loop()
    IdleTasks[req.name] = loop.create_task(_idle_loop(req.name, req))
    return _json({"ok": True, "started": req.name})

@app.post("/idle/stop")
async def idle_stop(name: str):
    t = IdleTasks.get(name)
    if not t: raise HTTPException(404, "no such idle task")
    t.cancel()
    return _json({"ok": True, "stopped": name})

@app.get("/idle/status")
def idle_status():
    return _json({"ok": True, "running": {k: (not v.done()) for k,v in IdleTasks.items()}})

# ---------- metrics/artifacts ----------
@app.get("/metrics/latest")
def metrics_latest():
    d = _latest_run_dir()
    if not d: return _json({"ok": False, "reason":"no runs yet"})
    p = Path(d)/"metrics.json"
    if not p.exists(): return _json({"ok": False, "reason":"metrics.json not found", "run_dir": d})
    txt = p.read_text(encoding="utf-8", errors="ignore").replace("NaN","null").replace("Infinity","null").replace("-Infinity","null")
    try: obj = json.loads(txt)
    except Exception: obj = {}
    obj["ok"]=True; obj["run_dir"]=d; obj["ingest_stats"]=IngestStats; obj.setdefault("time_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    return _json(obj)

@app.get("/artifacts/file/{name}")
def artifacts_file(name: str):
    d = _latest_run_dir()
    if not d: raise HTTPException(404, "No runs found")
    path = Path(d)/name
    if not path.exists(): raise HTTPException(404, f"{name} not found")
    mt = "image/png" if name.lower().endswith(".png") else "application/json"
    return FileResponse(path, media_type=mt, headers=_nocache())

# ---------- SSE ----------
async def _tail(path: Path, event_name: str):
    last = time.monotonic()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                now = time.monotonic()
                if now-last >= 5.0:
                    yield f"event: {event_name}\ndata: alive\n\n"; last = now
                await asyncio.sleep(0.25); continue
            yield f"data: {line.strip()}\n\n"

async def _wait_for(relname: str) -> Path:
    while True:
        d = _latest_run_dir()
        if d:
            p = Path(d)/relname
            if p.exists(): return p
        await asyncio.sleep(0.4)

@app.get("/stream/training")
async def stream_training():
    p = await _wait_for("train.log.jsonl")
    return StreamingResponse(_tail(p,"ping"), media_type="text/event-stream", headers=_nocache())

@app.get("/stream/nn")
async def stream_nn():
    p = await _wait_for("nn_state.jsonl")
    return StreamingResponse(_tail(p,"nn_ping"), media_type="text/event-stream", headers=_nocache())

# ---------- NN graph ----------
@app.get("/nn/graph")
def nn_graph():
    d = _latest_run_dir()
    if not d: return _json({"ok": False, "detail":"no runs"})
    p = Path(d)/"nn_graph.json"
    if not p.exists():
        return _json({"ok": True, "graph":{"title":"-","model":"-","layers":[{"size":16},{"size":16},{"size":2}]}})
    return _json({"ok": True, "graph": json.loads(p.read_text(encoding="utf-8", errors="ignore"))})

# ---------- login + dashboard UI ----------
@app.get("/")
def root():
    return RedirectResponse(url="/login")


@app.get("/login")
def login_page():
    if not LOGIN_PAGE.exists():
        return HTMLResponse("<h1>public/login.html missing</h1>", status_code=404, headers=_nocache())
    return FileResponse(LOGIN_PAGE, media_type="text/html", headers=_nocache())


@app.get("/admin/login")
def admin_login_page():
    if not ADMIN_LOGIN_PAGE.exists():
        return HTMLResponse("<h1>public/admin-login.html missing</h1>", status_code=404, headers=_nocache())
    return FileResponse(ADMIN_LOGIN_PAGE, media_type="text/html", headers=_nocache())


@app.get("/dashboard")
def dashboard():
    html = PUBLIC_DIR / "dashboard.html"
    if not html.exists():
        return HTMLResponse("<h1>public/dashboard.html missing</h1>", status_code=404, headers=_nocache())
    return FileResponse(html, media_type="text/html", headers=_nocache())


@app.get("/liquidity")
def liquidity_ui():
    html = LIQUIDITY_DIR / "index.html"
    if not html.exists():
        return HTMLResponse("<h1>public/liquidity/index.html missing</h1>", status_code=404, headers=_nocache())
    return FileResponse(html, media_type="text/html", headers=_nocache())

if __name__ == "__main__":
    print(json.dumps(run_preflight(), indent=2))
    import uvicorn

    uvicorn_kwargs = {"host": API_HOST, "port": API_PORT, "reload": False}
    if SSL_ENABLED:
        uvicorn_kwargs.update(
            {
                "ssl_certfile": SSL_CERTFILE,
                "ssl_keyfile": SSL_KEYFILE,
            }
        )
        if SSL_KEYFILE_PASSWORD:
            uvicorn_kwargs["ssl_keyfile_password"] = SSL_KEYFILE_PASSWORD
        if SSL_CA_CERTS:
            uvicorn_kwargs["ssl_ca_certs"] = SSL_CA_CERTS
    uvicorn.run(app, **uvicorn_kwargs)
