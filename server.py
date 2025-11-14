# -*- coding: utf-8 -*-
fromdy
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

app = FastAPI(
    title="Neo Cortex AI Trainer",
    version="4.4",
    lifespan=lifespan,
    dependencies=[Depends(csrf_manager)],
)


@app.get("/csrf-token")
async def get_csrf_token(request: Request, response: Response) -> Dict[str, str]:
    """Return a fresh CSRF token bound to the caller's session."""

    token = issue_csrf_token(request, response)
    response.headers.setdefault("cache-control", "no-store")
    response.headers.setdefault("pragma", "no-cache")
    return {"csrf_token": token}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"[NeoCortex API] {request.method} {request.url}")
    response = await call_next(request)
    return response


@app.middleware("http")
async def admin_redirect_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException as exc:
        if (
            exc.status_code == HTTP_403_FORBIDDEN
            and request.method.upper() == "GET"
            and "text/html" in (request.headers.get("accept", "").lower())
        ):
            return RedirectResponse("/login?error=not_admin")
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        if exc.status_code == HTTP_403_FORBIDDEN and str(exc.detail) in {"not_admin", "Admin access only"}:
            return RedirectResponse(url="/login?error=not_admin", status_code=302)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
    print("[NeoCortex Error]", traceback.format_exc())
    # Honour FastAPI-style ``HTTPException`` responses so that callers receive
    # the exact status code and error payload that the endpoint intended to
    # send.  The previous implementation wrapped *all* exceptions in a generic
    # 500 response which caused security-sensitive endpoints (for example the

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


@app.middleware("http")
async def csp_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self';"
        "img-src 'self' data: blob:;"
        "style-src 'self' 'unsafe-inline';"
        "script-src 'self' 'unsafe-inline' 'unsafe-eval';"
        "connect-src 'self' ws: wss: https:;"
        "font-src 'self' data:;"
        "frame-ancestors 'none';"
    )
    return response


app.mount("/static", StaticFiles(directory="static"), name="static")
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
    next: Optional[str] = None


_SENSITIVE_QUERY_KEYS: Set[str] = {
    "username",
    "password",
    "admin_key",
    "adminkey",
    "otp_code",
    "require_admin",
}


_AUTH_FIELD_ALIASES: Dict[str, str] = {
    "adminkey": "admin_key",
    "admin-key": "admin_key",
    "admin key": "admin_key",
    "requireadmin": "require_admin",
    "otp": "otp_code",
}


async def _auth_req_from_request(
    request: Request,
    data_override: Optional[Mapping[str, Any]] = None,
) -> AuthReq:
    '''Parse an AuthReq from JSON or form-encoded payloads.'''

    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    data: Dict[str, Any] = {}

    async def _parse_json(*, warn: bool = False) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover - defensive
            if warn:
                logger.warning("failed to parse json auth payload: %s", exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    async def _parse_form(*, warn: bool = False) -> Dict[str, Any]:
        try:
            form = await request.form()
        except Exception as exc:  # pragma: no cover - defensive
            if warn:
                logger.warning("failed to parse form auth payload: %s", exc)
            return {}
        return {k: v for k, v in form.items() if v is not None}

    form_markers = ("multipart/form-data", "application/x-www-form-urlencoded")
    json_markers = ("application/json", "text/json")

    if data_override is not None:
        items = data_override.items() if hasattr(data_override, "items") else data_override
        if items is not None:
            for key, value in items:
                if value is not None:
                    data[str(key)] = value
    else:
        if any(marker in content_type for marker in form_markers):
            data = await _parse_form(warn=True)
        elif content_type.endswith("+json") or any(marker in content_type for marker in json_markers):
            data = await _parse_json(warn=True)
        else:
            data = await _parse_json()
            if not data:
                data = await _parse_form()

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
        normalized_key = key.strip()
        alias = _AUTH_FIELD_ALIASES.get(normalized_key.lower())
        if alias:
            normalized_key = alias
        normalized[normalized_key] = value if isinstance(value, str) else str(value)

    try:
        return AuthReq(**normalized)
    except ValidationError:
        raise HTTPException(status_code=400, detail="username and password required")


class CredentialReq(BaseModel):
    account_type: str = "paper"
    api_key: str
    api_secret: str
    base_url: Optional[str] = None


class ClientCredentialReq(BaseModel):
    label: str
    account_type: Literal["paper", "funded"] = "paper"
    api_key: str
    api_secret: str
    base_url: Optional[str] = None
    client_id: Optional[int] = None


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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatCompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    history: List[ChatMessage] = Field(default_factory=list)
    model: Optional[str] = Field(default=None, max_length=128)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(default=None, ge=1, le=4096)


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
    client_id: Optional[int] = None
    client_label: Optional[str] = None


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
    qr_code: Optional[str] = None


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

def _wants_html_response(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    if not accept:
        return False
    if "text/html" in accept or "application/xhtml+xml" in accept:
        return True
    return False


def _resolve_login_redirect(candidate: Optional[str], user: Dict[str, Any]) -> str:
    fallback = "/dashboard" if ROLE_ADMIN in set(user.get("roles", [])) else "/enduserapp"
    if candidate:
        value = candidate.strip()
        if value:
            parsed = urlparse(value)
            if not parsed.scheme and not parsed.netloc:
                path = parsed.path or "/"
                if path.startswith("/"):
                    destination = path
                    if parsed.query:
                        destination = f"{destination}?{parsed.query}"
                    if parsed.fragment:
                        destination = f"{destination}#{parsed.fragment}"
                    return destination
    return fallback


def _handle_login(
    req: AuthReq,
    request: Request,
    *,
    enforce_admin: bool = False,
    prefer_redirect: bool = False,
    next_hint: Optional[str] = None,
) -> Response:
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
    redirect_path = _resolve_login_redirect(next_hint or req.next, user)
    if prefer_redirect:
        response: Response = RedirectResponse(url=redirect_path, status_code=303)
        _set_auth_cookies(response, tokens)
        return response
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
            "redirect_to": redirect_path,
        }
    )
    _set_auth_cookies(resp, tokens)
    return resp


def _require_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return user


def require_admin(
    request: Request, user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    user_obj: Any = (
        getattr(request.state, "user", None)
        or getattr(request, "user", None)
        or {}
    )

    if not user_obj and user:
        user_obj = user

    if isinstance(user_obj, Mapping):
        roles_source: Iterable[Any] = user_obj.get("roles") or []
    else:
        roles_source = getattr(user_obj, "roles", None) or []

    normalized_roles = {
        str(role).strip().lower() for role in roles_source if role is not None
    }
    if "admin" not in normalized_roles:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Admin access only",
        )

    if isinstance(user_obj, Mapping):
        result: Dict[str, Any] = dict(user_obj)
    elif isinstance(user_obj, dict):
        result = user_obj
    else:
        result = {}
        if hasattr(user_obj, "__dict__") and isinstance(user_obj.__dict__, Mapping):
            result.update(user_obj.__dict__)
        result.setdefault("roles", list(normalized_roles))

    request.state.user = result
    return result


def _require_admin(user: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    return user


@app.post("/login")
async def login(request: Request):
    """Authenticate a user and issue a bearer token."""

    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    form_data: Optional[Mapping[str, Any]] = None
    if content_type in {"application/x-www-form-urlencoded", "multipart/form-data"}:
        with suppress(Exception):
            form = await request.form()
            form_data = dict(form.multi_items()) if hasattr(form, "multi_items") else dict(form.items())

    req = await _auth_req_from_request(request, data_override=form_data)
    next_hint = req.next or request.query_params.get("next")
    prefer_redirect = bool(form_data) or _wants_html_response(request)
    return _handle_login(
        req,
        request,
        prefer_redirect=prefer_redirect,
        next_hint=next_hint,
    )


@app.post("/admin/login")
async def admin_login(request: Request):
    """Admin-specific login endpoint that enforces elevated privileges."""

    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    form_data: Optional[Mapping[str, Any]] = None
    if content_type in {"application/x-www-form-urlencoded", "multipart/form-data"}:
        with suppress(Exception):
            form = await request.form()
            form_data = dict(form.multi_items()) if hasattr(form, "multi_items") else dict(form.items())

    req = await _auth_req_from_request(request, data_override=form_data)
    next_hint = req.next or request.query_params.get("next") or ("/admin/dashboard" if form_data else None)
    prefer_redirect = bool(form_data) or _wants_html_response(request)
    return _handle_login(
        req,
        request,
        enforce_admin=True,
        prefer_redirect=prefer_redirect,
        next_hint=next_hint,
    )


@app.post("/whop/login")
async def whop_login(payload: WhopTokenRequest) -> JSONResponse:
    """Authenticate an end-user session using a Whop access token."""

    whop_user = await _fetch_whop_user(payload.token)
    return JSONResponse({"ok": True, "user": whop_user})


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
    clear_csrf(request, resp)
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
    provisioning_uri = _totp_provisioning_uri(user["username"], secret)
    qr_code = _generate_totp_qr_code(provisioning_uri)
    response = MFASetupResponse(
        secret=secret,
        provisioning_uri=provisioning_uri,
        recovery_codes=recovery_codes,
        qr_code=qr_code,
    )
    return _json({"ok": True, **response.dict(exclude_none=True)})


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
    try:
        client_credentials = _list_client_credentials(user["id"])
    except HTTPException:
        raise
    return _json({"ok": True, "credentials": credentials, "clients": client_credentials})


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


@app.post("/alpaca/clients")
async def save_alpaca_client_credentials(
    req: ClientCredentialReq,
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    user = _require_user(authorization, session_token, required_scopes={"credentials"})
    label = (req.label or "").strip()
    if not label:
        return _json({"ok": False, "detail": "label is required"}, 400)
    acct_type = (req.account_type or "paper").strip().lower()
    if acct_type not in {"paper", "funded"}:
        return _json({"ok": False, "detail": "account_type must be 'paper' or 'funded'"}, 400)
    api_key = (req.api_key or "").strip()
    api_secret = (req.api_secret or "").strip()
    if not api_key or not api_secret:
        return _json({"ok": False, "detail": "api_key and api_secret are required"}, 400)
    base_url = (req.base_url or "").strip()
    if not base_url:
        base_url = ALPACA_BASE_URL_FUND if acct_type == "funded" else ALPACA_BASE_URL_PAPER
    try:
        client_id = _save_client_credentials(
            user["id"],
            label,
            acct_type,
            api_key,
            api_secret,
            base_url,
            req.client_id,
        )
    except CredentialEncryptionError as exc:
        logger.error("Failed to persist client credentials for user %s", user["id"])
        return _json({"ok": False, "detail": str(exc)}, 500)
    return _json(
        {
            "ok": True,
            "client_id": client_id,
            "label": label,
            "account_type": acct_type,
            "base_url": base_url,
        }
    )


@app.delete("/alpaca/clients/{client_id}")
async def delete_alpaca_client_credentials(
    client_id: int,
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    user = _require_user(authorization, session_token, required_scopes={"credentials"})
    removed = _delete_client_credential(user["id"], int(client_id))
    if not removed:
        raise HTTPException(status_code=404, detail="client_not_found")
    return _json({"ok": True, "client_id": int(client_id)})

# --- account data: positions and P&L ---
@app.post("/positions")
async def get_positions(
    account: str = "paper",
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    '''Fetch the current positions for the specified Alpaca account.'''

    try:
        acct_type = (account or "paper").lower()
        user = None
        if authorization or session_token:
            user = _require_user(
                authorization,
                session_token,
                required_scopes={"positions"},
            )

        creds = _resolve_alpaca_credentials(acct_type, user["id"] if user else None)
        key = creds.get("key")
        secret = creds.get("secret")
        base_url = creds.get("base_url")
        if not key or not secret:
            return _json(
                {"ok": False, "detail": "Alpaca credentials not configured"},
                500,
            )

        url = f"{base_url}/v2/positions"
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
        session = _http_session()
        resp = await asyncio.to_thread(
            session.get,
            url,
            headers=headers,
            timeout=15,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error retrieving positions", exc_info=exc)
        return _json(
            {
                "ok": False,
                "detail": "Unexpected error retrieving positions. Please try again.",
            },
            502,
        )

    payload = _json_or_none(resp, context="Alpaca /v2/positions")
    if resp.status_code >= 400:
        detail = "Failed to fetch positions from Alpaca"
        if isinstance(payload, dict):
            detail = (
                payload.get("message")
                or payload.get("error")
                or payload.get("detail")
                or detail
            )
        body: Dict[str, Any] = {"ok": False, "detail": detail}
        if payload is not None:
            body["alpaca"] = payload
        return _json(body, resp.status_code or 502)

    raw_positions: List[Mapping[str, Any]]
    if isinstance(payload, list):
        raw_positions = payload
    elif isinstance(payload, dict):
        raw_positions = list(payload.get("positions") or [])
    else:
        raw_positions = []

    serialised: List[Dict[str, Any]] = []
    for record in raw_positions:
        if not isinstance(record, Mapping):
            continue
        serialised.append(
            {
                "symbol": record.get("symbol"),
                "quantity": _safe_float(record.get("qty") or record.get("quantity")),
                "avg_entry_price": _safe_float(record.get("avg_entry_price")),
                "market_value": _safe_float(record.get("market_value")),
                "cost_basis": _safe_float(record.get("cost_basis")),
                "unrealized_pl": _safe_float(record.get("unrealized_pl")),
            }
        )

    return _json({"ok": True, "positions": serialised, "account_type": acct_type})


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

    try:
        acct_type = (account or "paper").lower()
        user = None
        if authorization or session_token:
            user = _require_user(
                authorization,
                session_token,
                required_scopes={"positions"},
            )

        creds = _resolve_alpaca_credentials(acct_type, user["id"] if user else None)
        key = creds.get("key")
        secret = creds.get("secret")
        base_url = creds.get("base_url")
        if not key or not secret:
            return _json(
                {"ok": False, "detail": "Alpaca credentials not configured"},
                500,
            )

        pos_url = f"{base_url}/v2/positions"
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
        session = _http_session()
        pos_resp = await asyncio.to_thread(
            session.get,
            pos_url,
            headers=headers,
            timeout=15,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected error computing P&L", exc_info=exc)
        return _json(
            {
                "ok": False,
                "detail": "Unexpected error computing P&L. Please try again.",
            },
            502,
        )

    payload = _json_or_none(pos_resp, context="Alpaca /v2/positions")
    if pos_resp.status_code >= 400:
        detail = "Failed to fetch positions from Alpaca"
        if isinstance(payload, dict):
            detail = (
                payload.get("message")
                or payload.get("error")
                or payload.get("detail")
                or detail
            )
        body: Dict[str, Any] = {"ok": False, "detail": detail}
        if payload is not None:
            body["alpaca"] = payload
        return _json(body, pos_resp.status_code or 502)

    raw_positions: List[Mapping[str, Any]]
    if isinstance(payload, list):
        raw_positions = payload
    elif isinstance(payload, dict):
        raw_positions = list(payload.get("positions") or [])
    else:
        raw_positions = []

    if not raw_positions:
        return _json({"ok": True, "total_pnl": 0.0, "positions": []})

    results: List[Dict[str, Any]] = []
    total_pnl = 0.0
    for position in raw_positions:
        if not isinstance(position, Mapping):
            continue
        qty = _safe_float(position.get("qty") or position.get("quantity"))
        cost_basis = _safe_float(position.get("cost_basis"))
        market_value = _safe_float(position.get("market_value"))
        if position.get("unrealized_pl") is not None:
            pnl = _safe_float(position.get("unrealized_pl"))
        else:
            pnl = market_value - cost_basis
        total_pnl += pnl
        results.append(
            {
                "symbol": position.get("symbol"),
                "quantity": qty,
                "avg_entry_price": _safe_float(position.get("avg_entry_price")),
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pl": pnl,
            }
        )

    return _json(
        {
            "ok": True,
            "total_pnl": round(total_pnl, 2),
            "positions": results,
        }
    )


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

    user = _require_user(authorization, session_token, required_scopes={"trade"})

    symbol = (req.symbol or "").strip().upper()
    if not symbol:
        return _json({"ok": False, "detail": "symbol is required"}, 400)

    client_id = req.client_id
    client_label = (req.client_label or "").strip() or None
    selected_client: Optional[Dict[str, Any]] = None
    account_type = "funded" if req.account == "funded" else "paper"
    if client_id is not None or client_label:
        selected_client = _fetch_client_credential(
            user["id"],
            client_id=client_id if client_id is not None else None,
            label=None if client_id is not None else client_label,
        )
        if selected_client is None:
            raise HTTPException(status_code=404, detail="client_not_found")
        account_type = selected_client["account_type"]
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
        "account": account_type,
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
        response_payload: Dict[str, Any] = {
            "ok": True,
            "executed": False,
            "ai": ai_decision,
            "detail": detail,
            "account": account_type,
            "order": order_summary,
        }
        if selected_client:
            response_payload["client_id"] = selected_client["id"]
            response_payload["client_label"] = selected_client["label"]
        return _json(response_payload)

    if selected_client:
        key = selected_client["key"]
        secret = selected_client["secret"]
        base_url = selected_client["base_url"]
        order_summary["client_id"] = selected_client["id"]
        order_summary["client_label"] = selected_client["label"]
    else:
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
        failure_payload: Dict[str, Any] = {

            "account": account_type,
            "order": order_summary,
            "alpaca": alpaca_payload,
            "alpaca_order": order_payload,
        }
        if selected_client:
            body["client_id"] = selected_client["id"]
            body["client_label"] = selected_client["label"]
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
    if selected_client:
        error_body["client_id"] = selected_client["id"]
        error_body["client_label"] = selected_client["label"]
    if alpaca_payload is not None:
        error_body["alpaca"] = alpaca_payload

    return _json(error_body, resp.status_code or 502)


@app.post("/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
):
    """Proxy chat completion requests to the configured language model."""

    if not ENDUSER_CHAT_ENABLED:
        raise HTTPException(status_code=404, detail="assistant is not enabled")

    user = _require_user(authorization, session_token, required_scopes={"ml-chat"})
    result = await _run_chat_completion(req, user)
    body = {
        "ok": True,
        "reply": result["message"],
        "model": result["model"],
        "usage": result.get("usage", {}),
    }
    return _json(body)

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
    return _json({"ok": True, "stopped": name})
@app.post("/idle/status")
def idle_status():
    return _json({"ok": True, "running": {k: (not v.done()) for k,v in IdleTasks.items()
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



@app.get("/dashboard", response_class=HTMLResponse)

        if SSL_KEYFILE_PASSWORD:
            uvicorn_kwargs["ssl_keyfile_password"] = SSL_KEYFILE_PASSWORD
        if SSL_CA_CERTS:
            uvicorn_kwargs["ssl_ca_certs"] = SSL_CA_CERTS
    uvicorn.run(app, **uvicorn_kwargs)
