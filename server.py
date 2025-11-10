# -*- coding: utf-8 -*-
from __future__ import annotations

import os

try:  # pragma: no cover - optional dependency
    import aikido_zen  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    class _AikidoZenStub:
        """Fallback shim when ``aikido_zen`` is unavailable."""

        available = False

        def protect(self) -> bool:
            return False

        def __getattr__(self, name: str):
            raise AttributeError(name)

    aikido_zen = _AikidoZenStub()  # type: ignore[assignment]
    _ZEN_LIBRARY_AVAILABLE = False
else:  # pragma: no cover - passthrough when dependency is present
    _ZEN_LIBRARY_AVAILABLE = True

    # Bridge legacy environment variables that the upstream library expects so
    # that projects configured with ``ZEN_ACCESS_TOKEN`` still start the
    # firewall/background services when ``aikido_zen.protect`` is invoked at
    # import time.  The Zen CLI/tooling historically looked for
    # ``AIKIDO_TOKEN`` only, so we copy whichever value the user supplied to the
    # canonical name before ``protect`` runs.
    if not os.environ.get("AIKIDO_TOKEN"):
        _fallback_token = (
            os.environ.get("ZEN_ACCESS_TOKEN")
            or os.environ.get("ZEN_TOKEN")
            or os.environ.get("ZEN_API_TOKEN")
        )
        if _fallback_token:
            os.environ["AIKIDO_TOKEN"] = _fallback_token

aikido_zen.protect()

# ---- Torch compile guards (before any torch/model import) ----
import os as _os
_os.environ.setdefault("PYTORCH_ENABLE_COMPILATION", "0")
_os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# ---- std imports ----
import os, json, time, math, shutil, asyncio, importlib, hashlib, sqlite3, secrets, logging, hmac, re, base64, threading, random, ipaddress
from urllib.parse import parse_qs, quote, urlencode
import importlib.util
from pathlib import Path
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict, List, Optional, Tuple, Literal, Set, Iterable
from types import SimpleNamespace
from collections import defaultdict, deque

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency in tests
    class _PandasStub:
        '''Lazy stub that raises a helpful error when pandas is unavailable.'''

        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                "pandas is required for data-processing endpoints. Install it with 'pip install pandas'."
            )

    pd = _PandasStub()  # type: ignore
    _PANDAS_AVAILABLE = False
else:  # pragma: no cover - exercised when pandas is installed
    _PANDAS_AVAILABLE = True

from fastapi import FastAPI, HTTPException, Request, Header, Cookie, Depends, Form
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
try:
    from fastapi.middleware.cors import CORSMiddleware
except ModuleNotFoundError:  # pragma: no cover - fallback when optional dependency missing
    class CORSMiddleware:  # type: ignore[override]
        """Minimal ASGI middleware stub used when FastAPI's CORS middleware is unavailable."""

        def __init__(self, app, **_: object) -> None:
            self.app = app

        async def __call__(self, scope, receive, send):  # pragma: no cover - pass-through behaviour
            await self.app(scope, receive, send)
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
try:
    from fastapi.templating import Jinja2Templates
except ModuleNotFoundError:  # pragma: no cover - fallback when optional extras missing
    try:  # pragma: no cover - secondary fallback if FastAPI re-export unavailable
        from starlette.templating import Jinja2Templates  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - minimal templating shim
        class Jinja2Templates:
            """Lightweight template renderer used when Starlette's helper is unavailable."""

            _TOKEN_RE = re.compile(r"(\{\{.*?\}\}|\{%.*?%\})", re.S)

            def __init__(self, directory: str) -> None:
                self.directory = Path(directory)
                self._cache: Dict[str, List[Tuple[str, str]]] = {}

            def _wrap(self, value: Any) -> Any:
                if isinstance(value, dict):
                    return SimpleNamespace(**{k: self._wrap(v) for k, v in value.items()})
                if isinstance(value, list):
                    return [self._wrap(v) for v in value]
                if isinstance(value, set):
                    return [self._wrap(v) for v in sorted(value)]
                return value

            def _prepare_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
                prepared: Dict[str, Any] = {}
                for key, value in context.items():
                    if key == "request":
                        prepared[key] = value
                    else:
                        prepared[key] = self._wrap(value)
                return prepared

            def _load_template(self, name: str) -> str:
                path = (self.directory / name).resolve()
                if not str(path).startswith(str(self.directory.resolve())):
                    raise ValueError(f"Invalid template path: {name}")
                try:
                    return path.read_text(encoding="utf-8")
                except FileNotFoundError as exc:  # pragma: no cover - configuration issue
                    raise RuntimeError(f"Template '{name}' not found") from exc

            def _tokenize(self, name: str, text: str) -> List[Tuple[str, str]]:
                cached = self._cache.get(name)
                if cached is not None:
                    return cached
                tokens: List[Tuple[str, str]] = []
                pos = 0
                for match in self._TOKEN_RE.finditer(text):
                    if match.start() > pos:
                        tokens.append(("text", text[pos:match.start()]))
                    raw = match.group(0)
                    if raw.startswith("{{"):
                        tokens.append(("expr", raw[2:-2].strip()))
                    else:
                        tokens.append(("stmt", raw[2:-2].strip()))
                    pos = match.end()
                if pos < len(text):
                    tokens.append(("text", text[pos:]))
                self._cache[name] = tokens
                return tokens

            def _eval(self, expr: str, context: Dict[str, Any]) -> Any:
                locals_ctx = self._prepare_context(context)
                try:
                    return eval(expr, {"__builtins__": {}}, locals_ctx)  # noqa: S307 - controlled input
                except Exception:
                    return ""

            def _render_tokens(
                self,
                tokens: List[Tuple[str, str]],
                context: Dict[str, Any],
                start: int = 0,
                stop_tokens: Optional[Set[str]] = None,
            ) -> Tuple[str, int]:
                pieces: List[str] = []
                idx = start
                while idx < len(tokens):
                    kind, value = tokens[idx]
                    if kind == "text":
                        pieces.append(value)
                    elif kind == "expr":
                        rendered = self._eval(value, context)
                        pieces.append("" if rendered is None else str(rendered))
                    elif kind == "stmt":
                        stmt = value.strip()
                        if stop_tokens and stmt in stop_tokens:
                            return "".join(pieces), idx
                        if stmt.startswith("include"):
                            include_path = self._parse_include(stmt)
                            included_html = self._render_template(include_path, context)
                            pieces.append(included_html)
                        elif stmt.startswith("if "):
                            cond_expr = stmt[3:].strip()
                            true_block, next_idx = self._render_tokens(tokens, context, idx + 1, {"else", "endif"})
                            idx = next_idx
                            false_block = ""
                            if idx < len(tokens):
                                kind2, value2 = tokens[idx]
                                if kind2 == "stmt" and value2.strip() == "else":
                                    false_block, next_idx = self._render_tokens(tokens, context, idx + 1, {"endif"})
                                    idx = next_idx
                            if idx < len(tokens):
                                kind3, value3 = tokens[idx]
                                if not (kind3 == "stmt" and value3.strip() == "endif"):
                                    raise RuntimeError("Unmatched {% if %} block")
                            if self._truthy(cond_expr, context):
                                pieces.append(true_block)
                            else:
                                pieces.append(false_block)
                        elif stmt == "else" or stmt == "endif":
                            if stop_tokens:
                                return "".join(pieces), idx
                        else:
                            # Unsupported directive is ignored for compatibility.
                            pass
                    idx += 1
                return "".join(pieces), idx

            def _truthy(self, expr: str, context: Dict[str, Any]) -> bool:
                result = self._eval(expr, context)
                return bool(result)

            def _parse_include(self, stmt: str) -> str:
                match = re.search(r"include\s+['\"]([^'\"]+)['\"]", stmt)
                if not match:
                    raise RuntimeError(f"Invalid include statement: {stmt}")
                return match.group(1)

            def _render_template(self, name: str, context: Dict[str, Any]) -> str:
                source = self._load_template(name)
                tokens = self._tokenize(name, source)
                rendered, _ = self._render_tokens(tokens, context)
                return rendered

            def TemplateResponse(self, name: str, context: Dict[str, Any], status_code: int = 200) -> HTMLResponse:
                if "request" not in context:
                    raise RuntimeError("Template context must include 'request'")
                html = self._render_template(name, context)
                return HTMLResponse(html, status_code=status_code)
from auth import create_access_token, get_current_user
from fastapi.templating import Jinja2Templates
from auth import create_access_token, get_current_user
from fastapi import FastAPI, HTTPException, Request, Header, Cookie, Form
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse


class _FallbackHTTPSRedirectMiddleware:
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

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
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


def _resolve_https_redirect_middleware():
    try:  # pragma: no cover - import is environment-dependent
        module = importlib.import_module("fastapi.middleware.httpsredirect")
        middleware = getattr(module, "HTTPSRedirectMiddleware", None)
        if middleware is not None:
            return middleware
    except Exception:  # pragma: no cover - fallback for trimmed/partial installs
        return _FallbackHTTPSRedirectMiddleware
    return _FallbackHTTPSRedirectMiddleware


HTTPSRedirectMiddleware = _resolve_https_redirect_middleware()

from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

try:
    import jwt
except ModuleNotFoundError as exc:  # pragma: no cover - dependency should be installed
    raise ModuleNotFoundError(
        "PyJWT is required to run the server. Install it with 'pip install PyJWT'."
    ) from exc

try:
    from jwt import InvalidTokenError, ExpiredSignatureError
except (ImportError, AttributeError) as exc:  # pragma: no cover - guard against wrong package
    raise ImportError(
        "The imported 'jwt' package is not PyJWT. Please install PyJWT and remove conflicting 'jwt' packages."
    ) from exc
try:
    import bcrypt
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback for test environments
    import hashlib
    import os

    class _BcryptStub:
        def gensalt(self, rounds: int = 12) -> bytes:
            return os.urandom(16)

        def hashpw(self, password: bytes, salt: bytes) -> bytes:
            digest = hashlib.pbkdf2_hmac("sha256", password, salt, 390000)
            return salt + digest

        def checkpw(self, password: bytes, hashed: bytes) -> bool:
            salt, digest = hashed[:16], hashed[16:]
            expected = hashlib.pbkdf2_hmac("sha256", password, salt, 390000)
            return expected == digest

    bcrypt = _BcryptStub()  # type: ignore[assignment]
try:
    import pyotp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pyotp = None  # type: ignore[assignment]

PYOTP_MISSING_MESSAGE = (
    "pyotp is required for multi-factor authentication features. Install it with 'pip install pyotp'."
)
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

try:
    from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
except ModuleNotFoundError:  # pragma: no cover - provide lightweight serializer for tests
    import base64
    import hashlib
    import hmac as _hmac
    import json as _json

    class BadSignature(Exception):
        pass

    class SignatureExpired(BadSignature):
        pass

    class URLSafeTimedSerializer:
        def __init__(self, secret_key: str, salt: str = "", serializer: Any | None = None) -> None:
            self._secret = (secret_key + salt).encode("utf-8")

        def dumps(self, obj: Any) -> str:
            payload = _json.dumps(obj, separators=(",", ":")).encode("utf-8")
            signature = _hmac.new(self._secret, payload, hashlib.sha256).digest()
            return base64.urlsafe_b64encode(payload + signature).decode("ascii")

        def loads(self, token: str, max_age: int | None = None) -> Any:
            raw = base64.urlsafe_b64decode(token.encode("ascii"))
            payload, signature = raw[:-32], raw[-32:]
            expected = _hmac.new(self._secret, payload, hashlib.sha256).digest()
            if not _hmac.compare_digest(signature, expected):
                raise BadSignature("Signature mismatch")
            return _json.loads(payload.decode("utf-8"))


def _env_flag(name: str, default: bool = False) -> bool:
    '''Interpret an environment variable as a boolean flag.'''

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


def _comma_separated_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_optional_module(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    try:
        return importlib.import_module(name)
    except Exception:
        return None


psutil = _load_optional_module("psutil")

_SENSITIVE_PATTERNS = [
    re.compile(r"(authorization\\s*[:=]\\s*)([^\s]+)", re.IGNORECASE),
    re.compile(r"(bearer\\s+)[A-Za-z0-9._\-+=/]+", re.IGNORECASE),
    re.compile(r"(apikey\\s+)[A-Za-z0-9._\-+=/]+", re.IGNORECASE),
    re.compile(r"(token=)([^&\s]+)", re.IGNORECASE),
    re.compile(r"(password=)([^&\s]+)", re.IGNORECASE),
    re.compile(r"(secret=)([^&\s]+)", re.IGNORECASE),
]
_SENSITIVE_KEYS = {
    "password",
    "passcode",
    "secret",
    "token",
    "api_key",
    "api_secret",
    "authorization",
    "access_token",
    "refresh_token",
}


def _sanitize_log_text(value: str) -> str:
    cleaned = value
    for pattern in _SENSITIVE_PATTERNS:
        cleaned = pattern.sub(lambda m: m.group(1) + "[REDACTED]", cleaned)
    return cleaned


def _sanitize_log_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_log_text(value)
    if isinstance(value, dict):
        sanitized: Dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = _sanitize_log_value(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        iterable = [_sanitize_log_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(iterable)
        if isinstance(value, set):
            return set(iterable)
        return iterable
    return value


class _SensitiveDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - logging integration
        if record.args:
            record.args = tuple(_sanitize_log_value(arg) for arg in record.args)
        record.msg = _sanitize_log_value(record.msg)
        if not isinstance(record.msg, str):
            record.msg = _sanitize_log_text(str(record.msg))
        return True


logger = logging.getLogger("neocortex.server")
logger.addFilter(_SensitiveDataFilter())
AUDIT_LOGGER = logging.getLogger("neocortex.audit")
AUDIT_LOGGER.setLevel(logging.INFO)
AUDIT_LOGGER.addFilter(_SensitiveDataFilter())

ALLOWED_HTTP_METHODS: Set[str] = {"GET", "POST"}
LOGIN_RATE_LIMIT_MAX_ATTEMPTS = int(os.getenv("LOGIN_RATE_LIMIT_MAX_ATTEMPTS", "5"))
LOGIN_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("LOGIN_RATE_LIMIT_WINDOW_SECONDS", "60"))
_LOGIN_ATTEMPTS: Dict[str, deque] = defaultdict(deque)
_LOGIN_RATE_LOCK = threading.Lock()

AUTH_FAILURE_ALERT_THRESHOLD = int(os.getenv("AUTH_FAILURE_ALERT_THRESHOLD", "5"))
AUTH_FAILURE_ALERT_WINDOW_SECONDS = int(os.getenv("AUTH_FAILURE_ALERT_WINDOW_SECONDS", "300"))
_AUTH_FAILURES: Dict[str, deque] = defaultdict(deque)
_AUTH_FAILURE_LOCK = threading.Lock()

_IP_ALLOWLIST_RAW = _comma_separated_list(os.getenv("IP_ALLOWLIST"))
IP_ALLOWLIST: List[ipaddress._BaseNetwork] = []
for entry in _IP_ALLOWLIST_RAW:
    try:
        IP_ALLOWLIST.append(ipaddress.ip_network(entry, strict=False))
    except ValueError:
        logger.warning("invalid IP allowlist entry ignored: %s", entry)

ALLOWED_ORIGINS = _comma_separated_list(os.getenv("ALLOWED_ORIGINS") or os.getenv("CORS_ALLOWED_ORIGINS"))
CSRF_COOKIE_NAME = os.getenv("CSRF_COOKIE_NAME", "csrftoken")
CSRF_HEADER_NAME = os.getenv("CSRF_HEADER_NAME", "X-CSRF-Token")
CSRF_TOKEN_TTL_SECONDS = int(os.getenv("CSRF_TOKEN_TTL_SECONDS", "3600"))
BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))
AUDIT_METADATA_MAX_LENGTH = int(os.getenv("AUDIT_METADATA_MAX_LENGTH", "2048"))


def _ip_allowed(ip: str) -> bool:
    if not IP_ALLOWLIST:
        return True
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in network for network in IP_ALLOWLIST)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    client = getattr(request, "client", None)
    if client:
        host = None
        if isinstance(client, tuple):
            host = client[0]
        else:
            host = getattr(client, "host", None)
        if host:
            return str(host)
    return "0.0.0.0"


def _client_user_agent(request: Request) -> str:
    return request.headers.get("user-agent", "unknown")[:512]


def _client_fingerprint(request: Request) -> str:
    identifier = f"{_client_ip(request)}|{_client_user_agent(request)}"
    secret = JWT_SECRET_KEY.encode("utf-8") if isinstance(JWT_SECRET_KEY, str) else bytes(JWT_SECRET_KEY)
    return hmac.new(secret, identifier.encode("utf-8"), hashlib.sha256).hexdigest()


def _enforce_login_rate_limit(identifier: str) -> None:
    key = identifier or "unknown"
    now = time.monotonic()
    with _LOGIN_RATE_LOCK:
        attempts = _LOGIN_ATTEMPTS[key]
        while attempts and now - attempts[0] > LOGIN_RATE_LIMIT_WINDOW_SECONDS:
            attempts.popleft()
        if len(attempts) >= LOGIN_RATE_LIMIT_MAX_ATTEMPTS:
            raise HTTPException(
                status_code=429,
                detail="too many login attempts from this source; please retry later",
            )
        attempts.append(now)


def _reset_login_attempts(identifier: str) -> None:
    key = identifier or "unknown"
    with _LOGIN_RATE_LOCK:
        _LOGIN_ATTEMPTS.pop(key, None)


def _track_auth_failure(ip: str) -> None:
    key = ip or "unknown"
    now = time.monotonic()
    with _AUTH_FAILURE_LOCK:
        entries = _AUTH_FAILURES[key]
        while entries and now - entries[0] > AUTH_FAILURE_ALERT_WINDOW_SECONDS:
            entries.popleft()
        entries.append(now)
        if len(entries) >= AUTH_FAILURE_ALERT_THRESHOLD:
            logger.warning("multiple authentication failures detected for %s", key)
            entries.clear()


def _clear_auth_failures(ip: str) -> None:
    key = ip or "unknown"
    with _AUTH_FAILURE_LOCK:
        _AUTH_FAILURES.pop(key, None)


def _register_auth_failure(request: Optional[Request], ip: str) -> None:
    _track_auth_failure(ip)
    if request is not None:
        try:
            request.state.auth_failure_logged = True
        except AttributeError:
            pass


def _generate_csrf_token(request: Request) -> str:
    payload = {"fp": _client_fingerprint(request), "ts": int(time.time())}
    return _CSRF_SERIALIZER.dumps(payload)


def _validate_csrf_token(request: Request, token: Optional[str]) -> bool:
    token = (token or "").strip()
    if not token:
        return False
    try:
        data = _CSRF_SERIALIZER.loads(token, max_age=CSRF_TOKEN_TTL_SECONDS)
    except (BadSignature, SignatureExpired):
        return False
    expected_fp = _client_fingerprint(request)
    provided_fp = str(data.get("fp") or "")
    return bool(provided_fp) and secrets.compare_digest(expected_fp, provided_fp)


def _should_enforce_csrf(request: Request) -> bool:
    if request.method.upper() != "POST":
        return False
    content_type = (request.headers.get("content-type") or "").lower()
    if any(
        marker in content_type
        for marker in ("application/x-www-form-urlencoded", "multipart/form-data")
    ):
        return True
    if SESSION_COOKIE_NAME in request.cookies or REFRESH_COOKIE_NAME in request.cookies:
        return True
    return False

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

with suppress(ModuleNotFoundError):
    from cryptography.fernet import Fernet, InvalidToken  # type: ignore

if "Fernet" not in globals():  # pragma: no cover - optional dependency fallback
    class InvalidToken(Exception):
        '''Raised when decrypting stored credentials with an invalid token.'''

    class _MissingCryptographyFernet:
        def __init__(self, *_args, **_kwargs):
            raise ModuleNotFoundError(
                "cryptography is required for credential encryption. Install it with 'pip install cryptography'."
            )

    Fernet = _MissingCryptographyFernet  # type: ignore

# ---- your model utils (import AFTER env guards) ----
MODEL_IMPORT_ERROR: Optional[Exception] = None

try:
    from model import build_features, train_and_save, latest_run_path
except Exception as exc:  # pragma: no cover - optional heavy dependency fallback
    MODEL_IMPORT_ERROR = exc

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
ENABLE_SECURITY_HEADERS = _env_flag("ENABLE_SECURITY_HEADERS", default=True)
DISABLE_SERVER_HEADER = _env_flag("DISABLE_SERVER_HEADER", default=True)
CSRF_COOKIE_SECURE = _env_flag("CSRF_COOKIE_SECURE", default=SSL_ENABLED)
CSRF_COOKIE_SAMESITE = os.getenv("CSRF_COOKIE_SAMESITE", "lax") or "lax"
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

DEFAULT_SECURITY_HEADERS: Dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), geolocation=(), microphone=()",
}


ZEN_FIREWALL_ENABLED = _env_flag("ZEN_FIREWALL_ENABLED", default=True)
ZEN_FIREWALL_PROFILE = os.getenv("ZEN_FIREWALL_PROFILE", "balanced")
ZEN_FIREWALL_RULES = os.getenv("ZEN_FIREWALL_RULES")
ZEN_FIREWALL_DEFAULT_POLICY = os.getenv("ZEN_FIREWALL_DEFAULT_POLICY", "allow")
ZEN_ACCESS_TOKEN = (
    os.getenv("ZEN_ACCESS_TOKEN")
    or os.getenv("ZEN_TOKEN")
    or os.getenv("ZEN_API_TOKEN")
    or os.getenv("AIKIDO_TOKEN")
)
ZEN_TOR_ENABLED = _env_flag("ZEN_TOR_ENABLED", default=False)
ZEN_TOR_EXIT_NODES = os.getenv("ZEN_TOR_EXIT_NODES")
ZEN_TOR_STRICT_NODES = _env_flag("ZEN_TOR_STRICT_NODES", default=False)
ZEN_TOR_PERFORMANCE_MODE = os.getenv("ZEN_TOR_PERFORMANCE_MODE", "fast")
ZEN_TOR_MAX_LATENCY_MS = os.getenv("ZEN_TOR_MAX_LATENCY_MS", "150")
ZEN_TOR_NEW_CIRCUIT_SECONDS = os.getenv("ZEN_TOR_NEW_CIRCUIT_SECONDS", "300")


def _configure_aikido_services() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "library_available": _ZEN_LIBRARY_AVAILABLE,
        "firewall": {
            "requested": ZEN_FIREWALL_ENABLED,
            "profile": ZEN_FIREWALL_PROFILE,
            "default_policy": ZEN_FIREWALL_DEFAULT_POLICY,
            "applied": False,
        },
        "token": {
            "provided": bool(ZEN_ACCESS_TOKEN),
            "applied": False,
        },
        "tor": {
            "requested": ZEN_TOR_ENABLED,
            "performance_mode": ZEN_TOR_PERFORMANCE_MODE,
            "strict_nodes": ZEN_TOR_STRICT_NODES,
            "exit_nodes": _comma_separated_list(ZEN_TOR_EXIT_NODES),
            "max_latency_ms": None,
            "new_circuit_seconds": None,
            "applied": False,
        },
    }

    if not _ZEN_LIBRARY_AVAILABLE:
        status["detail"] = "aikido_zen module not installed"
        return status

    # Configure the Zen firewall if requested.
    if ZEN_FIREWALL_ENABLED:
        firewall_kwargs: Dict[str, Any] = {
            "profile": ZEN_FIREWALL_PROFILE,
            "default_policy": ZEN_FIREWALL_DEFAULT_POLICY,
        }

        if ZEN_FIREWALL_RULES:
            try:
                firewall_kwargs["rules"] = json.loads(ZEN_FIREWALL_RULES)
            except json.JSONDecodeError as exc:
                logger.warning("ZEN_FIREWALL_RULES is not valid JSON: %s", exc)
                status["firewall"]["error"] = f"invalid rules: {exc}"
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Unexpected error parsing ZEN_FIREWALL_RULES")
                status["firewall"]["error"] = str(exc)

        if "error" not in status["firewall"]:
            firewall_handler = getattr(aikido_zen, "enable_firewall", None) or getattr(
                aikido_zen, "configure_firewall", None
            )
            if callable(firewall_handler):
                try:
                    result = firewall_handler(**firewall_kwargs)
                    status["firewall"]["applied"] = True
                    if result is not None:
                        status["firewall"]["result"] = result
                except Exception as exc:  # pragma: no cover - runtime safety
                    logger.exception("Failed to configure Zen firewall")
                    status["firewall"]["error"] = str(exc)
            else:
                status["firewall"]["error"] = "firewall controls unavailable"

    # Register secure token with aikido_zen if provided.
    if ZEN_ACCESS_TOKEN:
        token_handler = (
            getattr(aikido_zen, "register_token", None)
            or getattr(aikido_zen, "set_token", None)
            or getattr(aikido_zen, "add_token", None)
        )
        if callable(token_handler):
            try:
                token_handler(ZEN_ACCESS_TOKEN)
                status["token"]["applied"] = True
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception("Failed to register Zen access token")
                status["token"]["error"] = str(exc)
        else:
            status["token"]["error"] = "token registration unavailable"

    # Enable Tor services with performance-focused defaults if requested.
    if ZEN_TOR_ENABLED:
        try:
            max_latency = max(10, int(str(ZEN_TOR_MAX_LATENCY_MS)))
        except (TypeError, ValueError):
            logger.warning(
                "ZEN_TOR_MAX_LATENCY_MS must be an integer; falling back to 150"
            )
            max_latency = 150
        try:
            new_circuit_seconds = max(30, int(str(ZEN_TOR_NEW_CIRCUIT_SECONDS)))
        except (TypeError, ValueError):
            logger.warning(
                "ZEN_TOR_NEW_CIRCUIT_SECONDS must be an integer; falling back to 300"
            )
            new_circuit_seconds = 300

        status["tor"]["max_latency_ms"] = max_latency
        status["tor"]["new_circuit_seconds"] = new_circuit_seconds

        tor_handler = (
            getattr(aikido_zen, "enable_tor", None)
            or getattr(aikido_zen, "configure_tor", None)
            or getattr(aikido_zen, "start_tor_service", None)
        )
        if callable(tor_handler):
            tor_kwargs: Dict[str, Any] = {
                "performance": ZEN_TOR_PERFORMANCE_MODE,
                "strict_nodes": ZEN_TOR_STRICT_NODES,
                "exit_nodes": status["tor"]["exit_nodes"],
                "max_latency_ms": max_latency,
                "new_circuit_seconds": new_circuit_seconds,
            }
            if not tor_kwargs["exit_nodes"]:
                tor_kwargs.pop("exit_nodes")
            try:
                result = tor_handler(**tor_kwargs)
                status["tor"]["applied"] = True
                if result is not None:
                    status["tor"]["result"] = result
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception("Failed to enable Zen Tor services")
                status["tor"]["error"] = str(exc)
        else:
            status["tor"]["error"] = "tor integration unavailable"

    return status


ZEN_SECURITY_STATUS = _configure_aikido_services()


def _is_subpath(base: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        return False


SAFE_FILENAME_COMPONENT = re.compile(r"[^A-Z0-9_-]")


def _sanitize_filename_component(value: str) -> str:
    cleaned = SAFE_FILENAME_COMPONENT.sub("_", value.upper())
    return cleaned or "UNKNOWN"

API_HOST   = os.getenv("API_HOST","0.0.0.0")
API_PORT   = int(os.getenv("API_PORT","8000"))
RUNS_ROOT  = Path(os.getenv("RUNS_ROOT","artifacts")).resolve()
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(os.getenv("STATIC_DIR","static")).resolve(); STATIC_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR = Path(os.getenv("PUBLIC_DIR","public")).resolve(); PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = Path(os.getenv("TEMPLATES_DIR", "templates")).resolve(); TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

LIQUIDITY_DIR = PUBLIC_DIR / "liquidity"
LIQUIDITY_DIR.mkdir(parents=True, exist_ok=True)
LIQUIDITY_ASSETS = LIQUIDITY_DIR / "assets"
LIQUIDITY_ASSETS.mkdir(parents=True, exist_ok=True)
ALPACA_TEST_DIR = PUBLIC_DIR / "alpaca_webhook_tests"
ALPACA_TEST_DIR.mkdir(parents=True, exist_ok=True)
ALPACA_TEST_DIR = ALPACA_TEST_DIR.resolve()

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
    '''Best-effort loader for the ngrok cloud endpoint HTML template.'''

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
    '''Best-effort loader for the ngrok cloud endpoint HTML template.'''

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
    '''Best-effort loader for the ngrok cloud endpoint HTML template.'''

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
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "access_token")
SESSION_COOKIE_SECURE = _env_flag("AUTH_COOKIE_SECURE", default=SSL_ENABLED)
SESSION_COOKIE_MAX_AGE = int(os.getenv("SESSION_COOKIE_MAX_AGE", str(15 * 60)))
SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "lax") or "lax"
REFRESH_COOKIE_NAME = os.getenv("REFRESH_TOKEN_COOKIE_NAME", "refresh_token")
REFRESH_COOKIE_SECURE = _env_flag("REFRESH_COOKIE_SECURE", default=SESSION_COOKIE_SECURE)
REFRESH_COOKIE_MAX_AGE = int(os.getenv("REFRESH_COOKIE_MAX_AGE", str(30 * 24 * 3600)))
REFRESH_COOKIE_SAMESITE = os.getenv("REFRESH_COOKIE_SAMESITE", "lax") or "lax"

_EPHEMERAL_JWT_SECRET = False
_configured_jwt_secret = (os.getenv("JWT_SECRET_KEY") or "").strip()
if not _configured_jwt_secret:
    logger.warning("JWT_SECRET_KEY not set; generating ephemeral signing key. Tokens will be invalidated on restart.")
    _configured_jwt_secret = secrets.token_urlsafe(64)
    _EPHEMERAL_JWT_SECRET = True
JWT_SECRET_KEY = _configured_jwt_secret
JWT_REFRESH_SECRET_KEY = (os.getenv("JWT_REFRESH_SECRET_KEY") or JWT_SECRET_KEY).strip()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS512")
JWT_ACCESS_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
JWT_REFRESH_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))
JWT_ISSUER = os.getenv("JWT_ISSUER", "neo-cortex")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "neo-cortex-clients")
JWT_LEEWAY_SECONDS = int(os.getenv("JWT_LEEWAY_SECONDS", "30"))
TOKEN_REFRESH_LEEWAY_SECONDS = int(os.getenv("TOKEN_REFRESH_LEEWAY_SECONDS", "60"))
CSRF_SECRET_KEY = os.getenv("CSRF_SECRET_KEY") or JWT_SECRET_KEY
_CSRF_SERIALIZER = URLSafeTimedSerializer(str(CSRF_SECRET_KEY), salt="neo-cortex-csrf")

ROLE_ADMIN = "admin"
ROLE_TRADER = "trader"
ROLE_READ_ONLY = "read-only"
ROLE_LICENSE = "license_user"
DEFAULT_ROLE = os.getenv("DEFAULT_USER_ROLE", ROLE_READ_ONLY)
ROLE_SCOPES: Dict[str, Set[str]] = {
    ROLE_ADMIN: {"admin", "trade", "read", "positions", "credentials", "api-keys"},
    ROLE_TRADER: {"trade", "read", "positions"},
    ROLE_LICENSE: {"license", "read"},
    ROLE_READ_ONLY: {"read"},
}
MFA_REQUIRED_ROLES: Set[str] = {role.strip() for role in os.getenv("MFA_REQUIRED_ROLES", f"{ROLE_ADMIN},{ROLE_TRADER}").split(",") if role.strip()}
API_TOKEN_PREFIX = os.getenv("API_TOKEN_PREFIX", "ApiKey")
API_KEY_SCOPE_SEPARATOR = os.getenv("API_KEY_SCOPE_SEPARATOR", " ")
ALLOWED_API_SCOPES: Set[str] = set().union(*ROLE_SCOPES.values())
TOTP_VALID_WINDOW = int(os.getenv("TOTP_VALIDATION_WINDOW", "1"))
MFA_RECOVERY_CODE_COUNT = int(os.getenv("MFA_RECOVERY_CODE_COUNT", "5"))
MFA_RECOVERY_CODE_LENGTH = int(os.getenv("MFA_RECOVERY_CODE_LENGTH", "10"))

CREDENTIALS_LEGACY_KEY = (os.getenv("CREDENTIALS_ENCRYPTION_KEY") or "").strip() or None
CREDENTIALS_KEYSET_RAW = (os.getenv("CREDENTIALS_ENCRYPTION_KEYS") or "").strip()
CREDENTIALS_ACTIVE_KEY_VERSION = (os.getenv("CREDENTIALS_ACTIVE_KEY_VERSION") or "active").strip()
CREDENTIALS_ROTATE_ON_START = _env_flag("CREDENTIALS_ROTATE_ON_START", default=False)

AUTH_DB_ENCRYPTION_PASSPHRASE = (os.getenv("AUTH_DB_ENCRYPTION_PASSPHRASE") or "").strip() or None

PINNED_CERT_FINGERPRINT = (os.getenv("PINNED_CERT_FINGERPRINT") or "").replace(":", "").strip().lower()
PINNED_CERT_HASH_ALGO = (os.getenv("PINNED_CERT_HASH_ALGO") or "sha256").strip().lower()
ENABLE_CERT_PINNING = bool(PINNED_CERT_FINGERPRINT)

SECURE_HEADERS_TEMPLATE: Dict[str, str] = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; object-src 'none'",
}
CUSTOM_SECURE_HEADERS = os.getenv("SECURE_HEADERS_EXTRA", "")
_STRICT_TRANSPORT_TEMPLATE: Optional[str] = None
if CUSTOM_SECURE_HEADERS:
    updated_headers = dict(SECURE_HEADERS_TEMPLATE)
    for header_pair in CUSTOM_SECURE_HEADERS.split(";;"):
        if not header_pair.strip():
            continue
        if "=" not in header_pair:
            continue
        name, value = header_pair.split("=", 1)
        header_name = name.strip()
        header_value = value.strip()
        if header_name.lower() == "strict-transport-security":
            _STRICT_TRANSPORT_TEMPLATE = header_value
            continue
        updated_headers[header_name] = header_value
    SECURE_HEADERS_TEMPLATE = updated_headers
else:
    _STRICT_TRANSPORT_TEMPLATE = None

DEFAULT_SECURITY_HEADERS.update(SECURE_HEADERS_TEMPLATE)

WHOP_API_KEY = (os.getenv("WHOP_API_KEY") or "").strip() or None
WHOP_API_BASE = (os.getenv("WHOP_API_BASE") or "https://api.whop.com").rstrip("/")
WHOP_PORTAL_URL = (os.getenv("WHOP_PORTAL_URL") or "").strip() or None
WHOP_SESSION_TTL = int(os.getenv("WHOP_SESSION_TTL", "900"))
ADMIN_PRIVATE_KEY = (os.getenv("ADMIN_PRIVATE_KEY") or "").strip()
ADMIN_PORTAL_BASIC_USER = (os.getenv("ADMIN_PORTAL_BASIC_USER") or "").strip()
ADMIN_PORTAL_BASIC_PASS = (os.getenv("ADMIN_PORTAL_BASIC_PASS") or "").strip()
ADMIN_PORTAL_BASIC_REALM = (
    os.getenv("ADMIN_PORTAL_BASIC_REALM", "Neo Cortex Admin") or "Neo Cortex Admin"
).strip()
ADMIN_PORTAL_GATE_TOKEN = (os.getenv("ADMIN_PORTAL_GATE_TOKEN") or "").strip()
ADMIN_PORTAL_GATE_HEADER = (os.getenv("ADMIN_PORTAL_GATE_HEADER") or "x-admin-portal-key").strip()
if not ADMIN_PORTAL_GATE_HEADER:
    ADMIN_PORTAL_GATE_HEADER = "x-admin-portal-key"
DISABLE_ACCESS_LOGS = _env_flag("DISABLE_ACCESS_LOGS", default=True)
ADMIN_PORTAL_HTTP_BASIC = HTTPBasic(auto_error=False)
if DISABLE_ACCESS_LOGS:
    logging.getLogger("uvicorn.access").disabled = True

ALPACA_WEBHOOK_SECRET = (os.getenv("ALPACA_WEBHOOK_SECRET") or "").strip()
ALPACA_ALLOW_UNAUTHENTICATED_WEBHOOKS = _env_flag(
    "ALPACA_ALLOW_UNAUTHENTICATED_WEBHOOKS", default=False
)
ALPACA_WEBHOOK_SIGNATURE_HEADER = (
    os.getenv("ALPACA_WEBHOOK_SIGNATURE_HEADER", "X-Webhook-Signature") or "X-Webhook-Signature"
).strip()
ALPACA_WEBHOOK_TIMESTAMP_HEADER = (
    os.getenv("ALPACA_WEBHOOK_TIMESTAMP_HEADER", "X-Webhook-Timestamp") or "X-Webhook-Timestamp"
).strip()
ALPACA_WEBHOOK_TOLERANCE_SECONDS = max(
    0,
    int(os.getenv("ALPACA_WEBHOOK_TOLERANCE_SECONDS", "300")),
)
ALPACA_WEBHOOK_REPLAY_CAPACITY = max(
    1,
    int(os.getenv("ALPACA_WEBHOOK_REPLAY_CAPACITY", "2048")),
)
ALPACA_WEBHOOK_TEST_REQUIRE_AUTH = _env_flag(
    "ALPACA_WEBHOOK_TEST_REQUIRE_AUTH", default=True
)

if not ADMIN_PRIVATE_KEY:
    logger.warning(
        "ADMIN_PRIVATE_KEY is not configured; the /register admin bootstrap endpoint will refuse requests until it is set."
    )

if not ALPACA_WEBHOOK_SECRET and not ALPACA_ALLOW_UNAUTHENTICATED_WEBHOOKS:
    logger.warning(
        "ALPACA_WEBHOOK_SECRET is not configured. Incoming /alpaca/webhook requests will be rejected until it is provided."
    )


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    conn.row_factory = sqlite3.Row
    if AUTH_DB_ENCRYPTION_PASSPHRASE:
        try:
            conn.execute("PRAGMA key = ?", (AUTH_DB_ENCRYPTION_PASSPHRASE,))
            conn.execute("PRAGMA cipher_memory_security = ON")
        except sqlite3.DatabaseError as exc:
            logger.error("Failed to apply SQLCipher key: %s", exc)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _record_audit_event(
    event: str,
    *,
    user_id: Optional[int] = None,
    username: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    serialized_metadata: Optional[str] = None
    if metadata is not None:
        try:
            cleaned = _sanitize_log_value(metadata)
            serialized_metadata = json.dumps(cleaned, default=str)
        except TypeError:
            serialized_metadata = json.dumps({"detail": str(metadata)})
        if len(serialized_metadata) > AUDIT_METADATA_MAX_LENGTH:
            serialized_metadata = serialized_metadata[:AUDIT_METADATA_MAX_LENGTH]
    with _db_conn() as conn:
        conn.execute(
            '''
            INSERT INTO audit_events (event, user_id, username, ip_address, user_agent, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                event,
                user_id,
                username,
                ip_address,
                user_agent,
                serialized_metadata,
                timestamp,
            ),
        )
    AUDIT_LOGGER.info(
        {
            "event": event,
            "user_id": user_id,
            "username": username,
            "ip": ip_address,
            "user_agent": user_agent,
            "timestamp": timestamp,
            "metadata": _sanitize_log_value(metadata) if metadata is not None else None,
        }
    )


def _init_auth_db() -> None:
    with _db_conn() as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                password_algo TEXT NOT NULL DEFAULT 'pbkdf2',
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            '''
        )
        info = conn.execute("PRAGMA table_info(users)").fetchall()
        column_names = {row["name"] for row in info}
        if "is_admin" not in column_names:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        if "role" not in column_names:
            default_role_literal = DEFAULT_ROLE.replace("'", "''")
            conn.execute(
                f"ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT '{default_role_literal}'"
            )
        if "password_algo" not in column_names:
            conn.execute(
                "ALTER TABLE users ADD COLUMN password_algo TEXT NOT NULL DEFAULT 'pbkdf2'"
            )
        if "totp_secret" not in column_names:
            conn.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT")
        if "mfa_enabled" not in column_names:
            conn.execute("ALTER TABLE users ADD COLUMN mfa_enabled INTEGER NOT NULL DEFAULT 0")
        if "mfa_delivery" not in column_names:
            conn.execute("ALTER TABLE users ADD COLUMN mfa_delivery TEXT DEFAULT 'totp'")
        if "mfa_recovery_codes" not in column_names:
            conn.execute("ALTER TABLE users ADD COLUMN mfa_recovery_codes TEXT")
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                token_hash TEXT,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                last_used_at TEXT,
                ip_address TEXT,
                user_agent TEXT,
                fingerprint TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            '''
        )
        session_info = conn.execute("PRAGMA table_info(sessions)").fetchall()
        session_columns = {row["name"] for row in session_info}
        if "token_hash" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN token_hash TEXT")
        if "expires_at" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN expires_at TEXT")
        if "last_used_at" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN last_used_at TEXT")
        if "ip_address" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN ip_address TEXT")
        if "user_agent" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN user_agent TEXT")
        if "fingerprint" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN fingerprint TEXT")
        conn.execute(
            '''
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
            '''
        )
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS api_tokens (
                token_id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                scopes TEXT NOT NULL,
                label TEXT,
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                revoked INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            '''
        )
        api_info = conn.execute("PRAGMA table_info(api_tokens)").fetchall()
        api_columns = {row["name"] for row in api_info}
        if "label" not in api_columns:
            conn.execute("ALTER TABLE api_tokens ADD COLUMN label TEXT")
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS credential_key_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_version TEXT NOT NULL,
                rotated_at TEXT NOT NULL,
                total_records INTEGER NOT NULL DEFAULT 0
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT NOT NULL,
                user_id INTEGER,
                username TEXT,
                ip_address TEXT,
                user_agent TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS whop_sessions (
                token TEXT PRIMARY KEY,
                license_key TEXT NOT NULL,
                email TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                consumed_at TEXT
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS whop_accounts (
                license_key TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            '''
        )


_init_auth_db()


class CredentialEncryptionError(RuntimeError):
    pass


_CREDENTIAL_CIPHERS: Dict[str, Fernet] = {}
_CREDENTIAL_KEYSET_CACHE: Dict[str, str] = {}
_ACTIVE_CREDENTIAL_KEY_VERSION: Optional[str] = None
_EPHEMERAL_CREDENTIAL_KEY = False


def _reload_credential_keys() -> None:
    global _CREDENTIAL_KEYSET_CACHE, _ACTIVE_CREDENTIAL_KEY_VERSION, _EPHEMERAL_CREDENTIAL_KEY
    keyset: Dict[str, str] = {}
    if CREDENTIALS_KEYSET_RAW:
        for item in CREDENTIALS_KEYSET_RAW.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            version, key = item.split(":", 1)
            version = version.strip()
            key = key.strip()
            if not version or not key:
                continue
            keyset[version] = key
    if CREDENTIALS_LEGACY_KEY and "legacy" not in keyset:
        keyset["legacy"] = CREDENTIALS_LEGACY_KEY
    if not keyset:
        logger.warning(
            "No CREDENTIALS_ENCRYPTION_KEYS configured; generating ephemeral key. Stored credentials will not persist across restarts."
        )
        generated = Fernet.generate_key().decode("utf-8")
        keyset["ephemeral"] = generated
        _EPHEMERAL_CREDENTIAL_KEY = True
    active_version = CREDENTIALS_ACTIVE_KEY_VERSION or "active"
    if active_version == "active" or active_version not in keyset:
        active_version = next(iter(keyset))
    _CREDENTIAL_KEYSET_CACHE = keyset
    _ACTIVE_CREDENTIAL_KEY_VERSION = active_version


def _credentials_cipher(version: Optional[str] = None) -> Tuple[str, Fernet]:
    if not _CREDENTIAL_KEYSET_CACHE:
        _reload_credential_keys()
    if not _CREDENTIAL_KEYSET_CACHE:
        raise CredentialEncryptionError(
            "No credential encryption keys are configured and an ephemeral key could not be generated."
        )
    key_version = version or _ACTIVE_CREDENTIAL_KEY_VERSION
    if key_version not in _CREDENTIAL_KEYSET_CACHE:
        raise CredentialEncryptionError(f"Unknown credential key version '{key_version}'")
    cipher = _CREDENTIAL_CIPHERS.get(key_version)
    if cipher is None:
        key_value = _CREDENTIAL_KEYSET_CACHE[key_version]
        try:
            cipher = Fernet(key_value.encode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            raise CredentialEncryptionError(f"Invalid credential key for version '{key_version}': {exc}") from exc
        _CREDENTIAL_CIPHERS[key_version] = cipher
    return key_version, cipher


def _encrypt_secret(value: str) -> str:
    if not value:
        return value
    version, cipher = _credentials_cipher()
    token = cipher.encrypt(value.encode("utf-8")).decode("utf-8")
    return f"enc:{version}:{token}"


def _decrypt_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if value.startswith("enc:"):
        payload = value[4:]
        version = None
        token = payload
        if ":" in payload:
            version, token = payload.split(":", 1)
        key_version, cipher = _credentials_cipher(version)
        try:
            decrypted = cipher.decrypt(token.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:  # pragma: no cover - data corruption safeguard
            raise CredentialEncryptionError(
                f"Unable to decrypt stored credential with key '{key_version}'"
            ) from exc
        return decrypted
    return value


def _rotate_encrypted_credentials(target_version: Optional[str] = None) -> int:
    if not _CREDENTIAL_KEYSET_CACHE:
        _reload_credential_keys()
    target_version = target_version or _ACTIVE_CREDENTIAL_KEY_VERSION
    if not target_version:
        raise CredentialEncryptionError("No active credential encryption key configured")
    target_version, target_cipher = _credentials_cipher(target_version)
    migrated = 0
    with _db_conn() as conn:
        rows = conn.execute(
            "SELECT user_id, account_type, api_key, api_secret FROM alpaca_credentials"
        ).fetchall()
        for row in rows:
            api_key = row["api_key"]
            api_secret = row["api_secret"]
            try:
                decrypted_key = _decrypt_secret(api_key)
                decrypted_secret = _decrypt_secret(api_secret)
            except CredentialEncryptionError:
                continue
            if not decrypted_key and not decrypted_secret:
                continue
            current_version = None
            if isinstance(api_key, str) and api_key.startswith("enc:"):
                payload = api_key[4:]
                if ":" in payload:
                    current_version = payload.split(":", 1)[0]
            if current_version == target_version:
                continue
            enc_key = f"enc:{target_version}:{target_cipher.encrypt(decrypted_key.encode('utf-8')).decode('utf-8')}"
            enc_secret = f"enc:{target_version}:{target_cipher.encrypt(decrypted_secret.encode('utf-8')).decode('utf-8')}"
            conn.execute(
                '''
                UPDATE alpaca_credentials
                SET api_key = ?, api_secret = ?, updated_at = ?
                WHERE user_id = ? AND account_type = ?
                ''',
                (
                    enc_key,
                    enc_secret,
                    datetime.now(timezone.utc).isoformat(),
                    row["user_id"],
                    row["account_type"],
                ),
            )
            migrated += 1
        if migrated:
            conn.execute(
                "INSERT INTO credential_key_history (key_version, rotated_at, total_records) VALUES (?, ?, ?)",
                (target_version, datetime.now(timezone.utc).isoformat(), migrated),
            )
    return migrated


_reload_credential_keys()
if CREDENTIALS_ROTATE_ON_START:
    try:
        _rotate_encrypted_credentials()
    except CredentialEncryptionError as exc:
        logger.error("Credential key rotation failed: %s", exc)


def _create_whop_session(license_key: str, email: Optional[str], metadata: Optional[Dict[str, Any]]) -> str:
    token = secrets.token_urlsafe(32)
    payload = json.dumps(metadata or {}, separators=(",", ":")) if metadata else None
    with _db_conn() as conn:
        conn.execute(
            '''
            INSERT OR REPLACE INTO whop_sessions (token, license_key, email, metadata, created_at, consumed_at)
            VALUES (?, ?, ?, ?, ?, NULL)
            ''',
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


def _lookup_whop_account(license_key: str) -> Optional[Dict[str, Any]]:
    if not license_key:
        return None
    with _db_conn() as conn:
        row = conn.execute(
            '''
            SELECT wa.user_id, u.username
            FROM whop_accounts wa
            JOIN users u ON u.id = wa.user_id
            WHERE wa.license_key = ?
            ''',
            (license_key,),
        ).fetchone()
    if row is None:
        return None
    return {"user_id": row["user_id"], "username": row["username"]}


def _link_whop_account(license_key: str, user_id: int) -> None:
    '''Link a Whop account to a user in the local credential store.'''
    pass
def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse_authorization_header(authorization: Optional[str]) -> Tuple[str, Optional[str]]:
    if not authorization:
        return "", None
    header = authorization.strip()
    if not header:
        return "", None
    parts = header.split(" ", 1)
    if len(parts) == 2:
        scheme, value = parts[0].lower(), parts[1].strip()
        return scheme, value or None
    return "bearer", header


def _load_recovery_codes(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [str(item) for item in data if str(item).strip()]
    return []


def _roles_from_string(raw_roles: Optional[str], is_admin: bool) -> List[str]:
    roles: Set[str] = set()
    if raw_roles:
        roles = {role.strip() for role in raw_roles.split(",") if role.strip()}
    if is_admin:
        roles.add(ROLE_ADMIN)
    if not roles:
        roles.add(DEFAULT_ROLE)
    normalized = []
    for role in roles:
        normalized.append(role if role in ROLE_SCOPES else DEFAULT_ROLE)
    return sorted(set(normalized))


def _scopes_for_roles(roles: List[str]) -> Set[str]:
    scopes: Set[str] = set()
    for role in roles:
        scopes.update(ROLE_SCOPES.get(role, set()))
    return scopes


def _load_user_record(user_id: int) -> Dict[str, Any]:
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT id, username, is_admin, role, mfa_enabled, mfa_delivery, totp_secret, mfa_recovery_codes FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="unknown user")
    roles = _roles_from_string(row["role"], bool(row["is_admin"]))
    scopes = _scopes_for_roles(roles)
    return {
        "id": row["id"],
        "username": row["username"],
        "roles": roles,
        "scopes": scopes,
        "mfa_enabled": bool(row["mfa_enabled"]),
        "mfa_delivery": row["mfa_delivery"] or "totp",
        "totp_secret": row["totp_secret"],
        "recovery_codes": _load_recovery_codes(row["mfa_recovery_codes"]),
    }


def _create_access_token(
    *,
    user: Dict[str, Any],
    refresh_id: Optional[str],
    fingerprint: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    payload = {
        "iss": JWT_ISSUER,
        "sub": str(user["id"]),
        "aud": JWT_AUDIENCE,
        "type": "access",
        "sid": refresh_id,
        "username": user["username"],
        "roles": user["roles"],
        "scopes": sorted(user["scopes"]),
        "fp": fingerprint,
    }
    return create_access_token(payload, expires_delta=expires_delta)


def _create_refresh_token(
    user_id: int,
    *,
    fingerprint: str,
    ip_address: str,
    user_agent: str,
) -> Tuple[str, str, datetime]:
    token_id = secrets.token_hex(16)
    secret = secrets.token_urlsafe(48)
    token = f"{token_id}.{secret}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=JWT_REFRESH_EXPIRE_DAYS)
    token_hash = _hash_token(token)
    with _db_conn() as conn:
        conn.execute(
            '''
            INSERT OR REPLACE INTO sessions (token, token_hash, user_id, created_at, expires_at, last_used_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (
                token_id,
                token_hash,
                user_id,
                datetime.now(timezone.utc).isoformat(),
                expires_at.isoformat(),
                None,
            ),
        )
        conn.execute(
            '''
            UPDATE sessions
            SET ip_address = ?, user_agent = ?, fingerprint = ?
            WHERE token = ?
            ''',
            (ip_address, user_agent, fingerprint, token_id),
        )
    return token, token_id, expires_at


def _decode_access_token(token: str, *, verify_exp: bool = True) -> Dict[str, Any]:
    options = {"verify_exp": verify_exp}
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            leeway=JWT_LEEWAY_SECONDS,
            options=options,
        )
    except ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="access token expired") from exc
    except InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="invalid access token") from exc
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="invalid access token type")
    return payload


def _verify_refresh_token(
    token: str,
    *,
    update_last_used: bool = True,
    request: Optional[Request] = None,
) -> Dict[str, Any]:
    token = (token or "").strip()
    if not token or "." not in token:
        raise HTTPException(status_code=401, detail="invalid refresh token")
    token_id, _secret = token.split(".", 1)
    token_hash = _hash_token(token)
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT token, token_hash, user_id, expires_at FROM sessions WHERE token = ?",
            (token_id,),
        ).fetchone()
        if row is None or not row["token_hash"]:
            raise HTTPException(status_code=401, detail="refresh token revoked")
        if row["token_hash"] != token_hash:
            raise HTTPException(status_code=401, detail="refresh token mismatch")
        if row["expires_at"]:
            try:
                expires_at = datetime.fromisoformat(row["expires_at"])
            except ValueError:
                raise HTTPException(status_code=401, detail="refresh token invalid expiry")
            if expires_at < datetime.now(timezone.utc):
                raise HTTPException(status_code=401, detail="refresh token expired")
        current_ip = _client_ip(request) if request else None
        current_ua = _client_user_agent(request) if request else None
        current_fp = _client_fingerprint(request) if request else None
        stored_fp = row["fingerprint"] or ""
        if request:
            if stored_fp and current_fp and not secrets.compare_digest(stored_fp, current_fp):
                _revoke_refresh_token(token_id=token_id)
                raise HTTPException(status_code=401, detail="refresh token context mismatch")
            stored_ip = (row["ip_address"] or "").strip()
            if stored_ip and current_ip and stored_ip != current_ip:
                _revoke_refresh_token(token_id=token_id)
                raise HTTPException(status_code=401, detail="refresh token context mismatch")
            stored_agent = row["user_agent"] or ""
            if stored_agent and current_ua and not secrets.compare_digest(stored_agent, current_ua):
                _revoke_refresh_token(token_id=token_id)
                raise HTTPException(status_code=401, detail="refresh token context mismatch")
        if update_last_used:
            conn.execute(
                """
                UPDATE sessions
                SET last_used_at = ?, ip_address = COALESCE(?, ip_address), user_agent = COALESCE(?, user_agent), fingerprint = COALESCE(?, fingerprint)
                WHERE token = ?
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    current_ip,
                    current_ua,
                    current_fp,
                    token_id,
                ),
            )
    return {
        "token_id": token_id,
        "user_id": row["user_id"],
        "expires_at": row["expires_at"],
    }


def _issue_token_pair(user: Dict[str, Any], *, request: Request) -> Dict[str, Any]:
    fingerprint = getattr(request.state, "client_fingerprint", _client_fingerprint(request))
    client_ip = getattr(request.state, "client_ip", _client_ip(request))
    user_agent = getattr(request.state, "client_user_agent", _client_user_agent(request))
    refresh_token, refresh_id, refresh_exp = _create_refresh_token(
        user["id"],
        fingerprint=fingerprint,
        ip_address=client_ip,
        user_agent=user_agent,
    )
    access_token = _create_access_token(
        user=user,
        refresh_id=refresh_id,
        fingerprint=fingerprint,
    )
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "refresh_expires_at": refresh_exp,
        "refresh_id": refresh_id,
    }


def _set_auth_cookies(response: JSONResponse, tokens: Dict[str, Any]) -> None:
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]
    response.set_cookie(
        SESSION_COOKIE_NAME,
        access_token,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        max_age=SESSION_COOKIE_MAX_AGE,
        samesite=SESSION_COOKIE_SAMESITE,
        path="/",
    )
    response.set_cookie(
        REFRESH_COOKIE_NAME,
        refresh_token,
        httponly=True,
        secure=REFRESH_COOKIE_SECURE,
        max_age=REFRESH_COOKIE_MAX_AGE,
        samesite=REFRESH_COOKIE_SAMESITE,
        path="/auth",
    )


def _clear_auth_cookies(response: JSONResponse) -> None:
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        samesite=SESSION_COOKIE_SAMESITE,
        secure=SESSION_COOKIE_SECURE,
    )
    response.delete_cookie(
        REFRESH_COOKIE_NAME,
        path="/auth",
        samesite=REFRESH_COOKIE_SAMESITE,
        secure=REFRESH_COOKIE_SECURE,
    )


def _revoke_refresh_token(refresh_token: Optional[str] = None, *, token_id: Optional[str] = None) -> None:
    candidate_id = token_id
    if refresh_token:
        refresh_token = refresh_token.strip()
        if "." in refresh_token:
            candidate_id = refresh_token.split(".", 1)[0]
    if not candidate_id:
        return
    with _db_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (candidate_id,))


def _revoke_tokens(access_token: Optional[str], refresh_token: Optional[str]) -> None:
    sid = None
    if access_token:
        try:
            payload = _decode_access_token(access_token, verify_exp=False)
            sid = payload.get("sid")
        except HTTPException:
            sid = None
    _revoke_refresh_token(refresh_token, token_id=sid)


class _PinnedHTTPSAdapter(requests.adapters.HTTPAdapter):
    def cert_verify(self, conn, url, verify, cert):
        super().cert_verify(conn, url, verify, cert)
        if not ENABLE_CERT_PINNING:
            return
        try:
            der_cert = conn.sock.getpeercert(binary_form=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise requests.exceptions.SSLError("Unable to read TLS certificate") from exc
        fingerprint = hashlib.new(PINNED_CERT_HASH_ALGO, der_cert).hexdigest()
        if fingerprint.lower() != PINNED_CERT_FINGERPRINT:
            raise requests.exceptions.SSLError("Certificate pinning validation failed")


_HTTP_SESSION: Optional[requests.Session] = None


def _http_session() -> requests.Session:
    global _HTTP_SESSION
    if _HTTP_SESSION is not None:
        return _HTTP_SESSION
    session = requests.Session()
    if ENABLE_CERT_PINNING:
        session.mount("https://", _PinnedHTTPSAdapter())
    _HTTP_SESSION = session
    return session


def _generate_recovery_codes(count: int = MFA_RECOVERY_CODE_COUNT) -> List[str]:
    codes: List[str] = []
    for _ in range(max(1, count)):
        raw = secrets.token_hex(max(4, MFA_RECOVERY_CODE_LENGTH // 2))
        codes.append(raw[:MFA_RECOVERY_CODE_LENGTH])
    return codes


def _store_recovery_codes(codes: List[str]) -> str:
    return json.dumps(codes)


def _ensure_pyotp() -> Any:
    if pyotp is None:
        raise ModuleNotFoundError(PYOTP_MISSING_MESSAGE)
    return pyotp


def _verify_totp_code(secret: Optional[str], code: Optional[str]) -> bool:
    if not secret or not code:
        return False
    if pyotp is None:
        logger.warning("pyotp is not installed; cannot verify TOTP codes")
        return False
    try:
        totp = pyotp.TOTP(secret)
        return bool(totp.verify(str(code).strip(), valid_window=TOTP_VALID_WINDOW))
    except Exception:
        return False


def _totp_provisioning_uri(username: str, secret: str) -> str:
    module = _ensure_pyotp()
    totp = module.TOTP(secret)
    return totp.provisioning_uri(name=username, issuer_name=JWT_ISSUER)


def _update_mfa_settings(
    user_id: int,
    *,
    secret: Optional[str],
    enabled: bool,
    recovery_codes: Optional[List[str]] = None,
) -> None:
    codes_json = _store_recovery_codes(recovery_codes or []) if recovery_codes else (None if not enabled else _store_recovery_codes([]))
    with _db_conn() as conn:
        conn.execute(
            "UPDATE users SET totp_secret = ?, mfa_enabled = ?, mfa_recovery_codes = ? WHERE id = ?",
            (secret, int(enabled), codes_json, user_id),
        )


def _consume_recovery_code(user_id: int, code: str) -> bool:
    if not code:
        return False
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT mfa_recovery_codes FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return False
        codes = _load_recovery_codes(row["mfa_recovery_codes"])
        normalized = [c for c in codes if c]
        if code not in normalized:
            return False
        normalized.remove(code)
        conn.execute(
            "UPDATE users SET mfa_recovery_codes = ? WHERE id = ?",
            (_store_recovery_codes(normalized), user_id),
        )
    return True


def _normalize_scopes(scopes: Iterable[str]) -> Set[str]:
    normalized = {scope.strip() for scope in scopes if scope and scope.strip() in ALLOWED_API_SCOPES}
    return normalized


def _create_api_token(user_id: int, scopes: Set[str], label: Optional[str] = None) -> Dict[str, Any]:
    token_id = secrets.token_hex(16)
    token_secret = secrets.token_urlsafe(32)
    raw_token = f"{token_id}.{token_secret}"
    token_hash = _hash_token(raw_token)
    scope_str = API_KEY_SCOPE_SEPARATOR.join(sorted(scopes))
    now = datetime.now(timezone.utc).isoformat()
    with _db_conn() as conn:
        conn.execute(
            '''
            INSERT INTO whop_accounts (license_key, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(license_key) DO UPDATE SET
                user_id = excluded.user_id,
                updated_at = excluded.updated_at
            ''',
            (license_key, user_id, now, now),
        )
        conn.execute(
            '''
            INSERT INTO api_tokens (token_id, token_hash, user_id, scopes, label, created_at, revoked)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            ''',
            (token_id, token_hash, user_id, scope_str, label, now),
        )
    return {
        "token": raw_token,
        "token_id": token_id,
        "scopes": sorted(scopes),
        "label": label,
        "created_at": now,
    }


def _list_api_tokens(user_id: int) -> List[Dict[str, Any]]:
    with _db_conn() as conn:
        rows = conn.execute(
            '''
            SELECT token_id, scopes, label, created_at, last_used_at, revoked
            FROM api_tokens
            WHERE user_id = ?
            ORDER BY created_at DESC
            ''',
            (user_id,),
        ).fetchall()
    tokens: List[Dict[str, Any]] = []
    for row in rows:
        tokens.append(
            {
                "token_id": row["token_id"],
                "scopes": [scope for scope in (row["scopes"] or "").split(API_KEY_SCOPE_SEPARATOR) if scope],
                "label": row["label"],
                "created_at": row["created_at"],
                "last_used_at": row["last_used_at"],
                "revoked": bool(row["revoked"]),
            }
        )
    return tokens


def _revoke_api_token(user_id: int, token_id: str) -> bool:
    with _db_conn() as conn:
        result = conn.execute(
            "UPDATE api_tokens SET revoked = 1 WHERE user_id = ? AND token_id = ?",
            (user_id, token_id),
        )
    return result.rowcount > 0


def _ensure_scopes(user_scopes: Set[str], required_scopes: Optional[Set[str]]) -> None:
    if not required_scopes:
        return
    if not required_scopes.issubset(user_scopes):
        raise HTTPException(status_code=403, detail="insufficient scope")


def _ensure_roles(user_roles: List[str], required_roles: Optional[Set[str]]) -> None:
    if not required_roles:
        return
    if not required_roles.intersection(set(user_roles)):
        raise HTTPException(status_code=403, detail="insufficient role")


def _get_user_from_access_token(
    token: str,
    *,
    required_scopes: Optional[Set[str]] = None,
    required_roles: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    payload = _decode_access_token(token)
    user = _load_user_record(int(payload["sub"]))
    _ensure_scopes(set(payload.get("scopes", [])), required_scopes)
    _ensure_roles(payload.get("roles", []), required_roles)
    return user


def _get_user_from_api_key(token: str, *, required_scopes: Optional[Set[str]] = None) -> Dict[str, Any]:
    token_hash = _hash_token(token)
    with _db_conn() as conn:
        row = conn.execute(
            '''
            SELECT token_id, token_hash, user_id, scopes, revoked
            FROM api_tokens
            WHERE token_hash = ?
            ''',
            (token_hash,),
        ).fetchone()
        if row is None or row["revoked"]:
            raise HTTPException(status_code=401, detail="api token revoked")
        conn.execute(
            "UPDATE api_tokens SET last_used_at = ? WHERE token_id = ?",
            (datetime.now(timezone.utc).isoformat(), row["token_id"]),
        )
    scopes = set((row["scopes"] or "").split(API_KEY_SCOPE_SEPARATOR)) if row["scopes"] else set()
    user = _load_user_record(int(row["user_id"]))
    _ensure_scopes(scopes, required_scopes)
    return user


def _require_user(
    authorization: Optional[str],
    session_token: Optional[str] = None,
    *,
    required_scopes: Optional[Set[str]] = None,
    required_roles: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    scheme, token = _parse_authorization_header(authorization)
    if not token and session_token:
        token = session_token
        scheme = "bearer"
    if not token:
        raise HTTPException(status_code=401, detail="authorization required")
    if scheme == "bearer":
        return _get_user_from_access_token(
            token,
            required_scopes=required_scopes,
            required_roles=required_roles,
        )
    if scheme == API_TOKEN_PREFIX.lower():
        return _get_user_from_api_key(token, required_scopes=required_scopes)
    raise HTTPException(status_code=401, detail="unsupported authorization scheme")


def _verify_whop_license(license_key: str) -> Dict[str, Any]:
    if not WHOP_API_KEY:
        raise HTTPException(status_code=503, detail="Whop integration is not configured")
    url = f"{WHOP_API_BASE}/api/v2/licenses/{license_key}"
    headers = {
        "Authorization": f"Bearer {WHOP_API_KEY}",
        "Accept": "application/json",
    }
    session = _http_session()
    try:
        resp = session.get(url, headers=headers, timeout=10)
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
    '''Derive a 2048-bit PBKDF2-SHA512 digest and return (hash_hex, salt_hex).'''
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


def _hash_password_secure(password: str) -> Tuple[str, str, str]:
    salt = secrets.token_hex(16)
    hashed = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=max(4, BCRYPT_ROUNDS)),
    )
    return hashed.decode("utf-8"), salt, "bcrypt"


def _verify_password(password: str, stored_hash: str, salt: str, algo: Optional[str]) -> bool:
    algorithm = (algo or "").lower() or ("bcrypt" if stored_hash.startswith("$2") else "pbkdf2")
    if algorithm == "bcrypt":
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except ValueError:
            return False
    expected_hash, _ = _hash_password_sha2048(password, salt)
    return secrets.compare_digest(expected_hash, stored_hash)


def _fetch_user_credentials(user_id: int, account_type: str) -> Optional[sqlite3.Row]:
    with _db_conn() as conn:
        return conn.execute(
            '''
            SELECT api_key, api_secret, base_url
            FROM alpaca_credentials
            WHERE user_id = ? AND account_type = ?
            ''',
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
            '''
            INSERT INTO alpaca_credentials (user_id, account_type, api_key, api_secret, base_url, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, account_type) DO UPDATE SET
                api_key = excluded.api_key,
                api_secret = excluded.api_secret,
                base_url = excluded.base_url,
                updated_at = excluded.updated_at
            ''',
            (
                user_id,
                account_type,
                stored_key,
                stored_secret,
                base_url,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


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
            '''
            INSERT INTO alpaca_credentials (user_id, account_type, api_key, api_secret, base_url, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, account_type) DO UPDATE SET
                api_key = excluded.api_key,
                api_secret = excluded.api_secret,
                base_url = excluded.base_url,
                updated_at = excluded.updated_at
            ''',
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


def _render_template(
    name: str,
    request: Request,
    context: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
):
    payload: Dict[str, Any] = {"request": request}
    if context:
        payload.update(context)
    response = templates.TemplateResponse(name, payload, status_code=status_code)
    for key, value in _nocache().items():
        response.headers.setdefault(key, value)
    return response


class _WebhookReplayProtector:
    """Track recently processed webhook signatures to block replay attempts."""

    def __init__(self, ttl_seconds: int, capacity: int) -> None:
        self._ttl = max(ttl_seconds, 0)
        self._capacity = max(capacity, 1)
        self._seen: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        if not self._seen or self._ttl <= 0:
            if self._ttl <= 0:
                self._seen.clear()
            return
        cutoff = now - self._ttl
        expired = [key for key, ts in self._seen.items() if ts < cutoff]
        for key in expired:
            self._seen.pop(key, None)

    def register(self, signature: str) -> bool:
        """Return ``True`` when the signature is new, ``False`` when replayed."""

        now = time.time()
        with self._lock:
            self._purge(now)
            if signature in self._seen:
                return False
            if len(self._seen) >= self._capacity:
                # Remove the oldest entry to make space for the newest signature.
                oldest_key = min(self._seen.items(), key=lambda item: item[1])[0]
                self._seen.pop(oldest_key, None)
            self._seen[signature] = now
            return True


_alpaca_replay_guard = _WebhookReplayProtector(
    ALPACA_WEBHOOK_TOLERANCE_SECONDS,
    ALPACA_WEBHOOK_REPLAY_CAPACITY,
)


def _compute_webhook_signature(secret: str, timestamp: str, body: bytes) -> str:
    message = timestamp.encode("utf-8") + b"." + body
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _parse_webhook_timestamp(raw: str) -> float:
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("empty timestamp")
    try:
        return float(cleaned)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError as exc:
            raise ValueError("invalid timestamp") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()


def _enforce_admin_portal_gate(
    request: Request, credentials: Optional[HTTPBasicCredentials]
) -> None:
    """Apply optional HTTP basic auth or header token gating for the admin portal."""

    if ADMIN_PORTAL_GATE_TOKEN:
        provided = request.headers.get(ADMIN_PORTAL_GATE_HEADER)
        if not provided or not secrets.compare_digest(provided, ADMIN_PORTAL_GATE_TOKEN):
            raise HTTPException(status_code=401, detail="admin portal token required")

    if ADMIN_PORTAL_BASIC_USER and ADMIN_PORTAL_BASIC_PASS:
        if not credentials or not (
            secrets.compare_digest(credentials.username, ADMIN_PORTAL_BASIC_USER)
            and secrets.compare_digest(credentials.password, ADMIN_PORTAL_BASIC_PASS)
        ):
            raise HTTPException(
                status_code=401,
                detail="admin portal authentication required",
                headers={"WWW-Authenticate": f'Basic realm="{ADMIN_PORTAL_BASIC_REALM}"'},
            )

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
    return {
        "packages": {"ok": not miss, "missing": miss, "versions": vers},
        "hardware": {"disk_free_gb": round(free / 1024**3, 2), "ram_avail_gb": ram_avail_gb},
        "gpu": _gpu_info(),
        "zen_security": ZEN_SECURITY_STATUS,
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(json.dumps(run_preflight(), indent=2), flush=True)
    yield

app = FastAPI(title="Neo Cortex AI Trainer", version="4.4", lifespan=lifespan)
if FORCE_HTTPS_REDIRECT:
    app.add_middleware(HTTPSRedirectMiddleware)


if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=sorted(ALLOWED_HTTP_METHODS),
        allow_headers=["Authorization", "Content-Type", CSRF_HEADER_NAME, "X-Requested-With"],
    )


@app.middleware("http")
async def _request_security_guard(request: Request, call_next):
    request.state.client_ip = _client_ip(request)
    request.state.client_user_agent = _client_user_agent(request)
    request.state.client_fingerprint = _client_fingerprint(request)

    method = request.method.upper()
    if method not in ALLOWED_HTTP_METHODS:
        return _json({"ok": False, "detail": "method not allowed"}, 405)

    if not _ip_allowed(request.state.client_ip):
        logger.warning("blocked request from disallowed IP %s", request.state.client_ip)
        _register_auth_failure(request, request.state.client_ip)
        return _json({"ok": False, "detail": "forbidden"}, 403)

    if method == "POST" and _should_enforce_csrf(request):
        header_token = request.headers.get(CSRF_HEADER_NAME)
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        if not header_token or not cookie_token:
            _register_auth_failure(request, request.state.client_ip)
            return _json({"ok": False, "detail": "csrf token missing"}, 403)
        if header_token != cookie_token or not _validate_csrf_token(request, header_token):
            _register_auth_failure(request, request.state.client_ip)
            return _json({"ok": False, "detail": "csrf validation failed"}, 403)

    response = await call_next(request)

    if method == "GET" and "text/html" in (request.headers.get("accept", "").lower()):
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        if not cookie_token or not _validate_csrf_token(request, cookie_token):
            cookie_token = _generate_csrf_token(request)
        response.set_cookie(
            CSRF_COOKIE_NAME,
            cookie_token,
            secure=CSRF_COOKIE_SECURE,
            httponly=False,
            samesite=CSRF_COOKIE_SAMESITE,
            max_age=CSRF_TOKEN_TTL_SECONDS,
            path="/",
        )

    return response


if ENABLE_SECURITY_HEADERS or ENABLE_HSTS or DISABLE_SERVER_HEADER:

    @app.middleware("http")
    async def _apply_security_headers(request: Request, call_next):
        response = await call_next(request)

        if ENABLE_SECURITY_HEADERS:
            for header_name, header_value in DEFAULT_SECURITY_HEADERS.items():
                response.headers.setdefault(header_name, header_value)

        if ENABLE_HSTS and request.url.scheme == "https":
            sts_value = _STRICT_TRANSPORT_TEMPLATE or HSTS_HEADER_VALUE
            if sts_value:
                response.headers["Strict-Transport-Security"] = sts_value
        else:
            response.headers.pop("Strict-Transport-Security", None)

        if DISABLE_SERVER_HEADER and "server" in response.headers:
            del response.headers["server"]

        return response


@app.middleware("http")
async def _enforce_token_expiry(request: Request, call_next):
    authorization = request.headers.get("authorization")
    token_payload: Optional[Dict[str, Any]] = None
    scheme = None
    token = None
    if authorization:
        scheme, token = _parse_authorization_header(authorization)
        if scheme == "bearer" and token:
            try:
                token_payload = _decode_access_token(token)
            except HTTPException as exc:
                return _json({"ok": False, "detail": exc.detail}, exc.status_code)
            fingerprint = token_payload.get("fp")
            request_fp = getattr(request.state, "client_fingerprint", _client_fingerprint(request))
            if fingerprint and not secrets.compare_digest(str(fingerprint), request_fp):
                _register_auth_failure(request, getattr(request.state, "client_ip", _client_ip(request)))
                return _json({"ok": False, "detail": "access token context mismatch"}, 401)
            request.state.access_token_payload = token_payload
    response = await call_next(request)
    if token_payload and scheme == "bearer":
        expires_at = token_payload.get("exp")
        if isinstance(expires_at, int):
            remaining = expires_at - int(time.time())
            response.headers.setdefault("X-Access-Token-Expires-In", str(max(0, remaining)))
            if remaining <= TOKEN_REFRESH_LEEWAY_SECONDS:
                response.headers.setdefault("X-Access-Token-Refresh", "required")
    return response


@app.middleware("http")
async def _auth_failure_observer(request: Request, call_next):
    response = await call_next(request)
    if response.status_code in {401, 403} and not getattr(request.state, "auth_failure_logged", False):
        ip = getattr(request.state, "client_ip", None) or _client_ip(request)
        _register_auth_failure(None, ip)
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


_PLACEHOLDER_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)


def _write_placeholder_png(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_PLACEHOLDER_PNG_BYTES)
    except Exception:
        with path.open("wb") as handle:
            handle.write(_PLACEHOLDER_PNG_BYTES)


def _simulate_training_stream(run_dir: Path, ticker: str, *, steps: int = 20) -> None:
    log_path = run_dir / "train.log.jsonl"
    nn_path = run_dir / "nn_state.jsonl"

    def _worker() -> None:
        try:
            time.sleep(0.5)
            equity = 10000.0 + random.uniform(-50, 50)
            with log_path.open("a", encoding="utf-8") as log, nn_path.open("a", encoding="utf-8") as nn:
                trade_open = False
                for idx in range(steps):
                    now = datetime.now(timezone.utc).isoformat()
                    equity += random.uniform(-25, 45)
                    row = {
                        "phase": "epoch",
                        "type": "equity",
                        "t": now,
                        "equity": round(equity, 2),
                        "entry_col": "PredLong",
                        "ticker": ticker,
                    }
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if idx in {3, 9, 15}:
                        if not trade_open:
                            trade = {
                                "type": "trade_open",
                                "t": now,
                                "side": "long",
                                "price": round(100 + random.uniform(-3, 3), 2),
                                "pnl": 0.0,
                                "reason": "synthetic-entry",
                                "entry_col": "PredLong",
                            }
                            trade_open = True
                        else:
                            trade = {
                                "type": "trade_close",
                                "t": now,
                                "side": "long",
                                "price": round(100 + random.uniform(-3, 3), 2),
                                "pnl": round(random.uniform(-12, 18), 2),
                                "reason": "synthetic-exit",
                                "entry_col": "PredLong",
                            }
                            trade_open = False
                        log.write(json.dumps(trade) + "\n")
                        log.flush()
                    stats = {
                        "type": "nn_stats",
                        "epoch": idx + 1,
                        "loss_long": round(max(0.02, 1.4 / (idx + 2)), 4),
                        "loss_short": round(max(0.02, 1.2 / (idx + 2)), 4),
                    }
                    nn.write(json.dumps(stats) + "\n")
                    nn.flush()
                    time.sleep(0.45)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("synthetic training stream failed: %s", exc)

    threading.Thread(target=_worker, name="synthetic-training-stream", daemon=True).start()


def _create_synthetic_training_run(req: TrainReq, reason: Optional[str]) -> tuple[float, int, str]:
    ticker = (req.ticker or "SYN").upper()
    safe_ticker = _sanitize_filename_component(ticker)
    run_dir = RUNS_ROOT / f"run-{safe_ticker}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "feature_importance_long.png",
        "feature_importance_short.png",
        "cm_long.png",
        "cm_short.png",
        "roc_long.png",
        "roc_short.png",
        "trades_label_long.png",
        "trades_pred_long.png",
        "yfinance_price.png",
    ):
        _write_placeholder_png(run_dir / name)

    metrics_samples = random.randint(80, 180)
    base_accuracy = round(random.uniform(0.68, 0.92), 3)
    metrics: Dict[str, Any] = {
        "ok": True,
        "model": "synthetic-demo",
        "timeframe": req.interval,
        "n_samples": metrics_samples,
        "accuracy": base_accuracy,
        "accuracy_long": min(0.99, round(base_accuracy + 0.015, 3)),
        "accuracy_short": max(0.5, round(base_accuracy - 0.02, 3)),
        "roc_auc": round(base_accuracy - 0.03, 3),
        "roc_auc_long": round(base_accuracy - 0.01, 3),
        "roc_auc_short": round(base_accuracy - 0.05, 3),
        "class_balance_long": {"NO(0)": metrics_samples - 24, "YES(1)": 24},
        "class_balance_short": {"NO(0)": metrics_samples - 18, "YES(1)": 18},
        "features": ["fast", "slow", "rsi", "volspike", "tv_long"],
        "backtest_pred": {"return_pct": 4.2, "win_rate_pct": 61.5},
    }
    if reason:
        metrics["notes"] = f"synthetic dataset generated because: {reason}"

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "train.log.jsonl").write_text("")
    (run_dir / "nn_state.jsonl").write_text("")

    graph = {
        "title": ticker,
        "model": "SyntheticNet",
        "layers": [
            {"size": 6, "type": "input"},
            {"size": 8, "type": "hidden"},
            {"size": 4, "type": "hidden"},
            {"size": 2, "type": "output"},
        ],
    }
    (run_dir / "nn_graph.json").write_text(json.dumps(graph, indent=2))

    _simulate_training_stream(run_dir, ticker)

    return base_accuracy, metrics_samples, run_dir.as_posix()


def _synthetic_train_response(req: TrainReq, reason: str) -> JSONResponse:
    logger.warning(
        "Synthetic training run generated for %s (%s)",
        req.ticker,
        reason,
    )
    acc, samples, run_dir = _create_synthetic_training_run(req, reason)
    payload = {
        "ok": True,
        "train_acc": acc,
        "n": samples,
        "run_dir": run_dir,
        "synthetic": True,
        "detail": reason,
    }
    return _json(payload)


def _should_use_synthetic_training() -> Optional[str]:
    if not _PANDAS_AVAILABLE:
        return "pandas dependency missing"
    if MODEL_IMPORT_ERROR is not None:
        return f"model import error: {MODEL_IMPORT_ERROR}"
    return None

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
    '''Render a friendly landing page for ngrok Cloud Endpoints.'''
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
    otp_code: Optional[str] = None


_SENSITIVE_QUERY_KEYS: Set[str] = {
    "username",
    "password",
    "admin_key",
    "adminkey",
    "otp_code",
    "require_admin",
}


async def _auth_req_from_request(request: Request) -> AuthReq:
    '''Parse an AuthReq from JSON or form-encoded payloads.'''

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
        provided = {
            key.lower()
            for key in request.query_params.keys()
            if key and key.lower() in _SENSITIVE_QUERY_KEYS
        }
        if provided:
            logger.warning(
                "rejected authentication attempt with credentials in query string (%s)",
                ", ".join(sorted(provided)),
            )
            raise HTTPException(
                status_code=400,
                detail="Credentials must be sent in the request body, not the URL query string.",
            )

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


class WhopLoginReq(BaseModel):
    token: str


class WhopSessionRequest(BaseModel):
    token: str


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


class AutoTradeOrderRequest(BaseModel):
    symbol: str
    side: Literal["long", "short"] = "long"
    quantity: float = Field(default=1.0, gt=0)
    price: Optional[float] = Field(default=None, ge=0)
    account: Literal["paper", "funded"] = "paper"
    instrument: Literal["equity", "option", "future"] = "equity"
    option_type: Optional[Literal["call", "put"]] = None
    expiry: Optional[str] = None
    strike: Optional[float] = Field(default=None, ge=0)
    future_month: Optional[str] = None
    future_year: Optional[int] = Field(default=None, ge=2000, le=2100)


class TokenRefreshRequest(BaseModel):
    refresh_token: Optional[str] = None


class APITokenCreateRequest(BaseModel):
    scopes: List[str] = Field(default_factory=list)
    label: Optional[str] = None


class APITokenRevokeRequest(BaseModel):
    token_id: str


class MFASetupResponse(BaseModel):
    secret: str
    provisioning_uri: str
    recovery_codes: List[str]


class MFAEnableRequest(BaseModel):
    secret: str
    otp_code: str
    recovery_codes: Optional[List[str]] = None


class MFADisableRequest(BaseModel):
    otp_code: Optional[str] = None
    recovery_code: Optional[str] = None


@app.get("/auth/whop/start")
async def whop_start(request: Request, next: Optional[str] = None, mode: Optional[str] = None):
    if (mode or "").strip().lower() == "status":
        return _json({"ok": True, "enabled": bool(WHOP_PORTAL_URL)})
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


@app.post("/auth/whop/session")
async def whop_session(req: WhopSessionRequest):
    token = (req.token or "").strip()
    session = _get_whop_session(token)
    if not session:
        return _json({"ok": False, "detail": "Invalid or expired Whop session"}, 404)
    account = _lookup_whop_account(session["license_key"])
    response_payload: Dict[str, Any] = {
        "ok": True,
        "license_key": session["license_key"],
        "email": session.get("email"),
        "created_at": session["created_at"],
        "registered": account is not None,
    }
    if account:
        response_payload["username"] = account["username"]
    return _json(response_payload)


@app.post("/register/whop")
async def register_whop(req: WhopRegistrationReq, request: Request):
    token = (req.token or "").strip()
    session = _get_whop_session(token)
    if not session:
        return _json({"ok": False, "detail": "Whop session expired or invalid"}, 400)
    license_key = session["license_key"]
    existing = _lookup_whop_account(license_key)
    if existing is not None:
        return _json(
            {
                "ok": False,
                "detail": "This Whop license is already linked to an account. Sign in through Whop.",
            },
            409,
        )
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
    pw_hash, salt, algo = _hash_password_secure(req.password)
    try:
        with _db_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, password_algo, is_admin, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uname,
                    pw_hash,
                    salt,
                    algo,
                    0,
                    ROLE_LICENSE,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return _json({"ok": False, "detail": "username already exists"}, 400)
    try:
        _save_user_credentials(user_id, acct_type, api_key, api_secret, base_url)
        _link_whop_account(license_key, user_id)
    except (CredentialEncryptionError, sqlite3.IntegrityError) as exc:
        logger.error("Failed to store credentials for new Whop user %s", user_id)
        with _db_conn() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return _json({"ok": False, "detail": str(exc)}, 500)
    _consume_whop_session(token)
    user = _load_user_record(user_id)
    tokens = _issue_token_pair(user, request=request)
    resp = _json(
        {
            "ok": True,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "username": uname,
            "roles": user["roles"],
            "scopes": sorted(user["scopes"]),
            "account_type": acct_type,
            "base_url": base_url,
        }
    )
    _set_auth_cookies(resp, tokens)
    return resp


@app.post("/auth/whop/login")
async def whop_login(req: WhopLoginReq):
    token = (req.token or "").strip()
    session = _get_whop_session(token)
    if not session:
        return _json({"ok": False, "detail": "Whop session expired or invalid"}, 400)
    account = _lookup_whop_account(session["license_key"])
    if not account:
        return _json({"ok": False, "detail": "Whop membership is not linked to an account"}, 404)
    _consume_whop_session(token)
    session_token = _create_session_token(account["user_id"])
    resp = _json(
        {
            "ok": True,
            "token": session_token,
            "username": account["username"],
            "session_cookie": SESSION_COOKIE_NAME,
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
    # Create a new user account backed by the SQLite credential store. Passwords are
    # hashed with a 2048-bit PBKDF2-SHA512 digest and salted prior to being persisted.
    '''
    Create a new user account backed by the SQLite credential store. Passwords are
    hashed with a 2048-bit PBKDF2-SHA512 digest and salted prior to being persisted.
    '''
    req = await _auth_req_from_request(request)
    uname = req.username.strip().lower()
    if not uname or not req.password:
        return _json({"ok": False, "detail": "username and password required"}, 400)
    if not ADMIN_PRIVATE_KEY:
        return _json({"ok": False, "detail": "admin registration disabled"}, 503)
    if (req.admin_key or "").strip() != ADMIN_PRIVATE_KEY:
        return _json({"ok": False, "detail": "invalid admin key"}, 403)
    pw_hash, salt, algo = _hash_password_secure(req.password)
    try:
        with _db_conn() as conn:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, password_algo, is_admin, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uname,
                    pw_hash,
                    salt,
                    algo,
                    1,
                    ROLE_ADMIN,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except sqlite3.IntegrityError:
        return _json({"ok": False, "detail": "username already exists"}, 400)
    client_ip = getattr(request.state, "client_ip", _client_ip(request))
    user_agent = getattr(request.state, "client_user_agent", _client_user_agent(request))
    _record_audit_event(
        "admin.user.create",
        username=uname,
        ip_address=client_ip,
        user_agent=user_agent,
        metadata={"roles": [ROLE_ADMIN]},
    )
    return _json({"ok": True, "created": uname})

def _handle_login(req: AuthReq, request: Request, *, enforce_admin: bool = False) -> JSONResponse:
    uname = req.username.strip().lower()
    client_ip = getattr(request.state, "client_ip", _client_ip(request))
    user_agent = getattr(request.state, "client_user_agent", _client_user_agent(request))
    try:
        _enforce_login_rate_limit(client_ip)
    except HTTPException as exc:
        _register_auth_failure(request, client_ip)
        _record_audit_event(
            "login.rate_limited",
            username=uname or None,
            ip_address=client_ip,
            user_agent=user_agent,
            metadata={"detail": exc.detail},
        )
        raise
    if not uname or not req.password:
        _register_auth_failure(request, client_ip)
        _record_audit_event(
            "login.failure",
            username=uname or None,
            ip_address=client_ip,
            user_agent=user_agent,
            metadata={"reason": "missing-credentials"},
        )
        return _json({"ok": False, "detail": "username and password required"}, 400)
    with _db_conn() as conn:
        row = conn.execute(
            """
            SELECT id, password_hash, salt, password_algo, is_admin, role, mfa_enabled, totp_secret
            FROM users
            WHERE username = ?
            """,
            (uname,),
        ).fetchone()
    if row is None:
        _register_auth_failure(request, client_ip)
        _record_audit_event(
            "login.failure",
            username=uname,
            ip_address=client_ip,
            user_agent=user_agent,
            metadata={"reason": "unknown-user"},
        )
        return _json({"ok": False, "detail": "invalid credentials"}, 401)
    if not _verify_password(req.password, row["password_hash"], row["salt"], row["password_algo"]):
        _register_auth_failure(request, client_ip)
        _record_audit_event(
            "login.failure",
            username=uname,
            ip_address=client_ip,
            user_agent=user_agent,
            metadata={"reason": "bad-password"},
        )
        return _json({"ok": False, "detail": "invalid credentials"}, 401)
    if req.require_admin and not row["is_admin"]:
        _register_auth_failure(request, client_ip)
        _record_audit_event(
            "login.failure",
            username=uname,
            ip_address=client_ip,
            user_agent=user_agent,
            metadata={"reason": "admin-required"},
        )
    require_admin = enforce_admin or req.require_admin
    if require_admin and not row["is_admin"]:
        return _json({"ok": False, "detail": "admin access required"}, 403)
    user = _load_user_record(row["id"])
    roles = set(user["roles"])
    require_mfa = user["mfa_enabled"] or bool(roles.intersection(MFA_REQUIRED_ROLES))
    if require_mfa:
        if row["totp_secret"]:
            if pyotp is None:
                return _json({"ok": False, "detail": PYOTP_MISSING_MESSAGE}, 500)
            otp_ok = _verify_totp_code(row["totp_secret"], req.otp_code)
            recovery_ok = False
            if not otp_ok and req.otp_code:
                recovery_ok = _consume_recovery_code(user["id"], req.otp_code)
                if recovery_ok:
                    user = _load_user_record(user["id"])
            if not otp_ok and not recovery_ok:
                _register_auth_failure(request, client_ip)
                _record_audit_event(
                    "login.failure",
                    username=uname,
                    user_id=user["id"],
                    ip_address=client_ip,
                    user_agent=user_agent,
                    metadata={"reason": "mfa-invalid"},
                )
                return _json({"ok": False, "detail": "otp verification failed"}, 401)
        else:
            _register_auth_failure(request, client_ip)
            _record_audit_event(
                "login.failure",
                username=uname,
                user_id=user["id"],
                ip_address=client_ip,
                user_agent=user_agent,
                metadata={"reason": "mfa-required"},
            )
            return _json({"ok": False, "detail": "mfa enrollment required"}, 403)
    _reset_login_attempts(client_ip)
    _clear_auth_failures(client_ip)
    _record_audit_event(
        "login.success",
        user_id=user["id"],
        username=uname,
        ip_address=client_ip,
        user_agent=user_agent,
        metadata={"roles": user["roles"]},
    )
    tokens = _issue_token_pair(user, request=request)
    primary_role = user["roles"][0] if user["roles"] else DEFAULT_ROLE
    resp = _json(
        {
            "ok": True,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token": tokens["access_token"],
            "username": uname,
            "roles": user["roles"],
            "scopes": sorted(user["scopes"]),
            "mfa_required": require_mfa,
            "session_cookie": SESSION_COOKIE_NAME,
            "refresh_cookie": REFRESH_COOKIE_NAME,
            "refresh_expires_at": tokens.get("refresh_expires_at"),
            "token_type": "bearer",
            "role": primary_role,
        }
    )
    _set_auth_cookies(resp, tokens)
    return resp


def _require_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return user


def _require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if ROLE_ADMIN not in set(user.get("roles", [])):
        raise HTTPException(status_code=403, detail="admin access required")
    return user


@app.post("/login")
async def login(request: Request):
    """Authenticate a user and issue a bearer token."""

    req = await _auth_req_from_request(request)
    return _handle_login(req, request)


@app.post("/admin/login")
async def admin_login(request: Request):
    """Admin-specific login endpoint that enforces elevated privileges."""

    req = await _auth_req_from_request(request)
    return _handle_login(req, request, enforce_admin=True)


@app.post("/logout")
async def logout(
    request: Request,
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
    refresh_cookie: Optional[str] = Cookie(None, alias=REFRESH_COOKIE_NAME),
):
    scheme, header_token = _parse_authorization_header(authorization)
    access_token = header_token if scheme == "bearer" else session_token
    refresh_token = refresh_cookie
    client_ip = getattr(request.state, "client_ip", _client_ip(request))
    user_agent = getattr(request.state, "client_user_agent", _client_user_agent(request))
    user_id: Optional[int] = None
    username: Optional[str] = None
    if access_token:
        try:
            payload = _decode_access_token(access_token, verify_exp=False)
            user_id = int(payload.get("sub")) if payload.get("sub") is not None else None
            username = payload.get("username")
        except HTTPException:
            pass
    _revoke_tokens(access_token, refresh_token)
    _record_audit_event(
        "logout",
        user_id=user_id,
        username=username,
        ip_address=client_ip,
        user_agent=user_agent,
    )
    _clear_auth_failures(client_ip)
    resp = _json({"ok": True})
    _clear_auth_cookies(resp)
    return resp


@app.post("/auth/refresh")
async def refresh_tokens(
    payload: TokenRefreshRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    refresh_cookie: Optional[str] = Cookie(None, alias=REFRESH_COOKIE_NAME),
):
    refresh_token = payload.refresh_token or refresh_cookie
    if not refresh_token and authorization:
        scheme, token = _parse_authorization_header(authorization)
        if scheme == "bearer":
            refresh_token = token
    if not refresh_token:
        raise HTTPException(status_code=401, detail="refresh token required")
    session = _verify_refresh_token(refresh_token, request=request)
    _revoke_refresh_token(token_id=session["token_id"])
    user = _load_user_record(int(session["user_id"]))
    tokens = _issue_token_pair(user, request=request)
    resp = _json(
        {
            "ok": True,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "roles": user["roles"],
            "scopes": sorted(user["scopes"]),
        }
    )
    _set_auth_cookies(resp, tokens)
    return resp


@app.post("/auth/api-keys/list")
async def list_api_keys(
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    user = _require_user(authorization, session_token, required_scopes={"api-keys"})
    tokens = _list_api_tokens(user["id"])
    return _json({"ok": True, "api_keys": tokens, "scheme": API_TOKEN_PREFIX})


@app.post("/auth/api-keys")
async def create_api_key(
    req: APITokenCreateRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    user = _require_user(authorization, session_token, required_scopes={"api-keys"})
    requested_scopes = _normalize_scopes(req.scopes)
    if not requested_scopes:
        raise HTTPException(status_code=400, detail="at least one valid scope is required")
    token_data = _create_api_token(user["id"], requested_scopes, label=req.label)
    client_ip = getattr(request.state, "client_ip", _client_ip(request))
    user_agent = getattr(request.state, "client_user_agent", _client_user_agent(request))
    _record_audit_event(
        "api-key.create",
        user_id=user["id"],
        username=user.get("username"),
        ip_address=client_ip,
        user_agent=user_agent,
        metadata={"token_id": token_data["token_id"], "scopes": token_data["scopes"], "label": token_data.get("label")},
    )
    return _json(
        {
            "ok": True,
            "token": token_data["token"],
            "token_id": token_data["token_id"],
            "scopes": token_data["scopes"],
            "label": token_data["label"],
            "created_at": token_data["created_at"],
            "scheme": API_TOKEN_PREFIX,
        },
        201,
    )


@app.post("/auth/api-keys/revoke")
async def revoke_api_key(
    req: APITokenRevokeRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    user = _require_user(authorization, session_token, required_scopes={"api-keys"})
    token_id = req.token_id.strip()
    if not token_id:
        raise HTTPException(status_code=400, detail="token_id is required")
    if not _revoke_api_token(user["id"], token_id):
        raise HTTPException(status_code=404, detail="api key not found")
    client_ip = getattr(request.state, "client_ip", _client_ip(request))
    user_agent = getattr(request.state, "client_user_agent", _client_user_agent(request))
    _record_audit_event(
        "api-key.revoke",
        user_id=user["id"],
        username=user.get("username"),
        ip_address=client_ip,
        user_agent=user_agent,
        metadata={"token_id": token_id},
    )
    return _json({"ok": True, "token_id": token_id})


@app.post("/auth/mfa/status")
async def mfa_status(
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    user = _require_user(authorization, session_token)
    return _json(
        {
            "ok": True,
            "mfa_enabled": user["mfa_enabled"],
            "delivery": user["mfa_delivery"],
            "recovery_codes_remaining": len(user.get("recovery_codes", [])),
        }
    )


@app.post("/auth/mfa/setup")
async def mfa_setup(
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    user = _require_user(authorization, session_token)
    try:
        module = _ensure_pyotp()
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    secret = module.random_base32()
    recovery_codes = _generate_recovery_codes()
    response = MFASetupResponse(
        secret=secret,
        provisioning_uri=_totp_provisioning_uri(user["username"], secret),
        recovery_codes=recovery_codes,
    )
    return _json({"ok": True, **response.dict()})


@app.post("/auth/mfa/enable")
async def mfa_enable(
    req: MFAEnableRequest,
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    user = _require_user(authorization, session_token)
    if not req.secret or not req.otp_code:
        raise HTTPException(status_code=400, detail="secret and otp_code are required")
    if pyotp is None:
        raise HTTPException(status_code=500, detail=PYOTP_MISSING_MESSAGE)
    if not _verify_totp_code(req.secret, req.otp_code):
        raise HTTPException(status_code=401, detail="otp verification failed")
    recovery_codes = req.recovery_codes or _generate_recovery_codes()
    _update_mfa_settings(user["id"], secret=req.secret, enabled=True, recovery_codes=recovery_codes)
    updated = _load_user_record(user["id"])
    return _json(
        {
            "ok": True,
            "mfa_enabled": True,
            "recovery_codes": updated.get("recovery_codes", []),
        }
    )


@app.post("/auth/mfa/disable")
async def mfa_disable(
    req: MFADisableRequest,
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    user = _require_user(authorization, session_token)
    if not user.get("mfa_enabled"):
        return _json({"ok": True, "mfa_enabled": False})
    verified = False
    if req.recovery_code and _consume_recovery_code(user["id"], req.recovery_code):
        verified = True
    elif pyotp is None:
        raise HTTPException(status_code=500, detail=PYOTP_MISSING_MESSAGE)
    elif _verify_totp_code(user.get("totp_secret"), req.otp_code):
        verified = True
    if not verified:
        raise HTTPException(status_code=401, detail="otp or recovery code required")
    _update_mfa_settings(user["id"], secret=None, enabled=False, recovery_codes=None)
    return _json({"ok": True, "mfa_enabled": False})


@app.post("/alpaca/credentials/read")
async def list_alpaca_credentials(
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    # Return the Alpaca credential entries associated with the authenticated user.

    '''Return the Alpaca credential entries associated with the authenticated user.'''
    user = _require_user(authorization, session_token, required_scopes={"credentials"})
    query = (
        "SELECT account_type, api_key, base_url, updated_at\n"
        "FROM alpaca_credentials\n"
        "WHERE user_id = ?\n"
        "ORDER BY account_type"
    )
    with _db_conn() as conn:
        rows = conn.execute(query, (user["id"],)).fetchall()

        rows = conn.execute(
            '''
            SELECT account_type, api_key, base_url, updated_at
            FROM alpaca_credentials
            WHERE user_id = ?
            ORDER BY account_type
            ''',
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
    '''Create or update Alpaca API credentials for the authenticated user.'''
    user = _require_user(authorization, session_token, required_scopes={"credentials"})
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
@app.post("/positions")
async def get_positions(
    account: str = "paper",
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    '''Fetch the current positions for the specified Alpaca account.'''

    acct_type = (account or "paper").lower()
    user = None
    if authorization or session_token:
        user = _require_user(authorization, session_token, required_scopes={"positions"})

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
    session = _http_session()
    try:
        resp = session.get(url, headers=headers, timeout=15)
        positions = resp.json() if resp.content else []
    except Exception as exc:
        return _json({"ok": False, "detail": f"Failed to fetch positions: {exc}"}, 500)
    return _json({"ok": True, "positions": positions, "account_type": acct_type})


@app.post("/positions/{symbol}/close")
async def close_position(
    symbol: str,
    account: str = "paper",
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    '''Close a single Alpaca position for the authenticated user.'''

    if not symbol:
        return _json({"ok": False, "detail": "symbol is required"}, 400)

    user = _require_user(authorization, session_token, required_scopes={"trade"})

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
    session = _http_session()
    try:
        resp = session.delete(url, headers=headers, timeout=15)
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

@app.post("/pnl")
async def get_pnl(
    account: str = "paper",
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    '''Compute unrealized P&L for the authenticated user\'s Alpaca account.'''

    acct_type = (account or "paper").lower()
    user = None
    if authorization or session_token:
        user = _require_user(authorization, session_token, required_scopes={"positions"})

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
    session = _http_session()
    try:
        pos_resp = session.get(pos_url, headers=headers, timeout=15)
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


@app.post("/strategy/liquidity-sweeps")
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
# your own database or notify your front-end. In this hosted context, any
# external HTTP requests (e.g. to Alpaca) are not executed; the code is
# illustrative only.
@app.post("/alpaca/webhook")
async def alpaca_webhook(req: Request):
    '''
    Receive webhook callbacks from Alpaca. All requests must include a valid
    ``ALPACA_WEBHOOK_SECRET`` signature unless
    ``ALPACA_ALLOW_UNAUTHENTICATED_WEBHOOKS`` is explicitly enabled. The
    signature is calculated as ``base64(hmac_sha256(secret, f"{timestamp}.{body}"))``
    and supplied via ``X-Webhook-Signature`` alongside an ``X-Webhook-Timestamp``
    header to prevent replay attacks.
    '''

    body_bytes = await req.body()
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except Exception:
        payload = None

    if not ALPACA_WEBHOOK_SECRET and not ALPACA_ALLOW_UNAUTHENTICATED_WEBHOOKS:
        return _json({"ok": False, "detail": "webhook secret not configured"}, 503)

    if ALPACA_WEBHOOK_SECRET:
        signature = req.headers.get(ALPACA_WEBHOOK_SIGNATURE_HEADER)
        if not signature:
            return _json({"ok": False, "detail": "missing signature"}, 400)

        timestamp_header = req.headers.get(ALPACA_WEBHOOK_TIMESTAMP_HEADER)
        if not timestamp_header:
            return _json({"ok": False, "detail": "missing timestamp"}, 400)

        timestamp_header = timestamp_header.strip()
        if not timestamp_header:
            return _json({"ok": False, "detail": "missing timestamp"}, 400)

        try:
            timestamp_value = _parse_webhook_timestamp(timestamp_header)
        except ValueError:
            return _json({"ok": False, "detail": "invalid timestamp"}, 400)

        now = time.time()
        tolerance = ALPACA_WEBHOOK_TOLERANCE_SECONDS
        if tolerance and now - timestamp_value > tolerance:
            return _json({"ok": False, "detail": "stale webhook"}, 400)
        if tolerance and timestamp_value - now > tolerance:
            return _json({"ok": False, "detail": "timestamp too far in future"}, 400)

        expected_signature = _compute_webhook_signature(
            ALPACA_WEBHOOK_SECRET,
            timestamp_header,
            body_bytes,
        )
        if not hmac.compare_digest(expected_signature, signature):
            return _json({"ok": False, "detail": "invalid signature"}, 400)

        if tolerance and not _alpaca_replay_guard.register(expected_signature):
            return _json({"ok": False, "detail": "webhook replay detected"}, 409)

    # In a production system you might persist the payload to a database or
    # notify other services. Here we simply print it to stdout.
    logger.info("[Alpaca webhook] %s", json.dumps(payload, indent=2))
    return _json({"ok": True})


@app.post("/alpaca/webhook/test")
async def alpaca_webhook_test(
    req: AlpacaWebhookTest,
    request: Request,
    credentials: HTTPBasicCredentials = Depends(ADMIN_PORTAL_HTTP_BASIC),
):
    gating_configured = bool(
        ADMIN_PORTAL_GATE_TOKEN or (ADMIN_PORTAL_BASIC_USER and ADMIN_PORTAL_BASIC_PASS)
    )
    if ALPACA_WEBHOOK_TEST_REQUIRE_AUTH:
        if not gating_configured:
            raise HTTPException(
                status_code=503,
                detail="admin authentication must be configured to use the webhook test endpoint",
            )
        _enforce_admin_portal_gate(request, credentials)
    elif gating_configured:
        _enforce_admin_portal_gate(request, credentials)

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
    timestamp_value = payload.get("timestamp") or req.timestamp
    if not timestamp_value:
        timestamp_value = datetime.now(timezone.utc).isoformat()
    payload["timestamp"] = timestamp_value

    body_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    signature = None
    if ALPACA_WEBHOOK_SECRET:
        signature = _compute_webhook_signature(ALPACA_WEBHOOK_SECRET, timestamp_value, body_bytes)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    symbol_component = _sanitize_filename_component(req.symbol or "")
    fname = f"test-{symbol_component}-{stamp}.json"
    path = (ALPACA_TEST_DIR / fname).resolve()
    if not _is_subpath(ALPACA_TEST_DIR, path):
        raise HTTPException(status_code=400, detail="invalid symbol for artifact path")
    path.parent.mkdir(parents=True, exist_ok=True)
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


@app.post("/alpaca/webhook/tests")
def alpaca_webhook_tests():
    '''Return recently generated Alpaca webhook test artifacts.'''
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


@app.post("/papertrade/status")
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


@app.post("/trade/auto")
async def trade_auto(
    req: AutoTradeOrderRequest,
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    """Execute an order through Alpaca when approved by the AI trainer."""

    user = _require_user(authorization, session_token, required_scopes={"trade"})

    symbol = (req.symbol or "").strip().upper()
    if not symbol:
        return _json({"ok": False, "detail": "symbol is required"}, 400)

    account_type = "funded" if req.account == "funded" else "paper"
    quantity = float(req.quantity)
    price = float(req.price) if req.price is not None else None
    if price is not None and price <= 0:
        price = None

    order_summary = {
        "symbol": symbol,
        "quantity": round(quantity, 6),
        "side": req.side,
        "price": round(price, 4) if price is not None else None,
        "instrument": req.instrument,
    }

    ai_payload = {
        "instrument": req.instrument,
        "symbol": symbol,
        "side": req.side,
        "quantity": quantity,
        "price": price or 0.0,
        "option_type": req.option_type,
        "expiry": req.expiry,
        "strike": req.strike,
        "future_month": req.future_month,
        "future_year": req.future_year,
    }
    ai_decision = _run_ai_trainer(ai_payload)
    if ai_decision.get("action") != "execute":
        detail = ai_decision.get("message") or "Neo Cortex AI parked the order for review."
        return _json(
            {
                "ok": True,
                "executed": False,
                "ai": ai_decision,
                "detail": detail,
                "account": account_type,
                "order": order_summary,
            }
        )

    creds = _resolve_alpaca_credentials(account_type, user["id"])
    key = creds.get("key")
    secret = creds.get("secret")
    base_url = creds.get("base_url")
    if not key or not secret or not base_url:
        return _json({"ok": False, "detail": "Alpaca credentials not configured"}, 500)

    def _format_qty(value: float) -> str:
        if not math.isfinite(value) or value <= 0:
            return "1"
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:.6f}".rstrip("0").rstrip(".")

    qty_str = _format_qty(quantity)
    order_payload: Dict[str, Any] = {
        "symbol": symbol,
        "qty": qty_str,
        "side": "buy" if req.side == "long" else "sell",
        "type": "market",
        "time_in_force": "day",
        "client_order_id": f"nc-{secrets.token_hex(10)}",
    }
    if price is not None:
        order_payload["type"] = "limit"
        order_payload["limit_price"] = round(price, 4)

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }

    session = _http_session()
    try:
        resp = session.post(
            f"{base_url}/v2/orders",
            headers=headers,
            json=order_payload,
            timeout=20,
        )
    except Exception as exc:
        return _json(
            {
                "ok": False,
                "executed": False,
                "ai": ai_decision,
                "detail": f"Failed to submit order: {exc}",
                "account": account_type,
                "order": order_summary,
                "alpaca_order": order_payload,
            },
            502,
        )

    alpaca_payload: Optional[Any]
    try:
        alpaca_payload = resp.json() if resp.content else None
    except ValueError:
        alpaca_payload = resp.text.strip() or None

    if 200 <= resp.status_code < 300:
        detail: Optional[str] = None
        if isinstance(alpaca_payload, dict):
            detail = (
                alpaca_payload.get("status_message")
                or alpaca_payload.get("message")
                or alpaca_payload.get("status")
            )
        if not detail:
            detail = "Alpaca accepted the order."
        body: Dict[str, Any] = {
            "ok": True,
            "executed": True,
            "ai": ai_decision,
            "detail": detail,
            "account": account_type,
            "order": order_summary,
            "alpaca": alpaca_payload,
            "alpaca_order": order_payload,
        }
        return _json(body, resp.status_code or 200)

    error_detail: Optional[str] = None
    if isinstance(alpaca_payload, dict):
        error_detail = (
            alpaca_payload.get("message")
            or alpaca_payload.get("error")
            or alpaca_payload.get("detail")
        )
    if not error_detail:
        error_detail = resp.text.strip() or f"Alpaca order rejected with HTTP {resp.status_code}"

    error_body: Dict[str, Any] = {
        "ok": False,
        "executed": False,
        "ai": ai_decision,
        "detail": error_detail,
        "account": account_type,
        "order": order_summary,
        "alpaca_order": order_payload,
    }
    if alpaca_payload is not None:
        error_body["alpaca"] = alpaca_payload

    return _json(error_body, resp.status_code or 502)

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
    '''
    Placeholder for a future Alpaca trade execution webhook.

    Executing high‑stakes financial transactions (such as buying or selling securities)
    is disabled in this environment. Any POSTed payload will be ignored and an error
    response returned. To integrate live trading functionality, you would need to
    use the Alpaca Orders API with appropriate safeguards, and run the code outside
    of this assistant.
    '''
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
    synthetic_reason = _should_use_synthetic_training()
    if synthetic_reason:
        return _synthetic_train_response(req, synthetic_reason)

    try:
        import yfinance as yf
    except ModuleNotFoundError:
        return _synthetic_train_response(req, "yfinance dependency missing")

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

    if base is None or getattr(base, "empty", True):
        return _synthetic_train_response(req, "no market data available")

    exog = _get_exog(req.ticker, base.index, include_fa=req.use_fundamentals)

    presets = {"chill":1.15,"normal":0.95,"spicy":0.75,"insane":0.55}
    entry_thr = req.entry_thr if req.entry_thr is not None else presets.get((req.aggressiveness or "normal"), 0.95)

    try:
        feat = build_features(
            base,
            hftMode=(req.mode.upper()=="HFT"),
            zThrIn=req.z_thr,
            volKIn=req.vol_k,
            entryThrIn=entry_thr,
            exog=exog
        )
    except Exception as exc:
        logger.warning("build_features failed for %s: %s", req.ticker, exc)
        return _synthetic_train_response(req, f"feature engineering failed: {exc}")

    # ---- guard around torch._dynamo error so API doesn't 500 ----
    try:
        acc, n = train_and_save(feat, max_iter=int(req.max_iter))
    except ModuleNotFoundError as e:
        message = str(e)
        if "torch._dynamo" in message:
            reason = (
                "PyTorch install is inconsistent (missing torch._dynamo). "
                "Install PyTorch 2.x GPU/CPU wheels or remove any third-party torch-compile shim."
            )
        else:
            reason = message or "required ML dependency missing"
        return _synthetic_train_response(req, reason)
    except Exception as e:
        logger.warning("train_and_save failed for %s: %s", req.ticker, e)
        return _synthetic_train_response(req, f"training error: {type(e).__name__}: {e}")

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

@app.post("/idle/status")
def idle_status():
    return _json({"ok": True, "running": {k: (not v.done()) for k,v in IdleTasks.items()}})

# ---------- metrics/artifacts ----------
@app.post("/metrics/latest")
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


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return _render_template("login.html", request)


@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return _render_template("admin_login.html", request)


@app.get("/admin", response_class=HTMLResponse)
def admin_portal(request: Request, user: Dict[str, Any] = Depends(_require_admin)):
    return _render_template("admin.html", request, {"user": user})


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, user: Dict[str, Any] = Depends(_require_admin)):
    return _render_template("dashboard.html", request, {"user": user})


@app.get("/enduserapp", response_class=HTMLResponse)
def enduserapp(request: Request, user: Dict[str, Any] = Depends(_require_user)):
    return _render_template("enduserapp.html", request, {"user": user})


@app.get("/radar", response_class=HTMLResponse)
def radar(request: Request, user: Dict[str, Any] = Depends(_require_user)):
    return _render_template("radar.html", request, {"user": user})


@app.get("/liquidity")
def liquidity_ui():
    return RedirectResponse(url="/radar")

if __name__ == "__main__":
    print(json.dumps(run_preflight(), indent=2))
    import uvicorn

    uvicorn_kwargs = {"host": API_HOST, "port": API_PORT, "reload": False}
    if DISABLE_ACCESS_LOGS:
        uvicorn_kwargs["access_log"] = False
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
