# -*- coding: utf-8 -*-
from __future__ import annotations

# ---- Torch compile guards (before any torch/model import) ----
import os as _os
_os.environ.setdefault("PYTORCH_ENABLE_COMPILATION", "0")
_os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# ---- std imports ----
import os, json, time, shutil, asyncio, importlib
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv


def _load_optional_module(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    try:
        return importlib.import_module(name)
    except Exception:
        return None


psutil = _load_optional_module("psutil")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# added imports for Alpaca integration
import requests

# ---- your model utils (import AFTER env guards) ----
from model import build_features, train_and_save, latest_run_path
from strategies import analyze_liquidity_session, StrategyError

load_dotenv(override=False)

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
NGROK_ENDPOINT_TEMPLATE_PATH = PUBLIC_DIR / "ngrok-cloud-endpoint.html"
try:
    NGROK_ENDPOINT_TEMPLATE = NGROK_ENDPOINT_TEMPLATE_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    NGROK_ENDPOINT_TEMPLATE = """<!doctype html><html><body><h1>ngrok endpoint</h1><p>Webhook URL: {{WEBHOOK_URL}}</p></body></html>"""

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

# --- simple user store for login/registration ---
# WARNING: this is an in-memory placeholder implementation. For production, use a secure database and
# proper password hashing (e.g. bcrypt). Do not use plain or weak hashing in production.
import hashlib, uuid

Users: Dict[str, str] = {}  # username -> sha256 hashed password

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode('utf-8')).hexdigest()

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
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
app.mount("/ui/liquidity", StaticFiles(directory=str(LIQUIDITY_DIR), html=True), name="liquidity-ui")

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
        host = f"localhost:{API_PORT}"
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


class AlpacaWebhookTest(BaseModel):
    symbol: str = "SPY"
    quantity: float = 1.0
    price: float = 0.0
    side: str = "buy"
    status: str = "filled"
    event: str = "trade_update"
    timestamp: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

@app.post("/register")
async def register(req: AuthReq):
    """
    Create a new user account. Stores the username and a hashed password in memory.
    This is a very basic implementation and should not be used in production without
    proper security measures. For a real application, store users in a persistent database
    and use a strong password hashing algorithm (e.g. bcrypt).
    """
    uname = req.username.strip().lower()
    if not uname or not req.password:
        return _json({"ok": False, "detail": "username and password required"}, 400)
    if uname in Users:
        return _json({"ok": False, "detail": "username already exists"}, 400)
    Users[uname] = _hash_pw(req.password)
    return _json({"ok": True, "created": uname})

@app.post("/login")
async def login(req: AuthReq):
    """
    Authenticate a user. On success, returns a simple session token. This token is not
    persisted and is intended solely for demonstration. A real implementation should
    issue a JWT or session cookie and validate it on subsequent requests.
    """
    uname = req.username.strip().lower()
    if not uname or not req.password:
        return _json({"ok": False, "detail": "username and password required"}, 400)
    if Users.get(uname) != _hash_pw(req.password):
        return _json({"ok": False, "detail": "invalid credentials"}, 401)
    token = str(uuid.uuid4())
    # in this demo, we do not maintain session state beyond returning the token
    return _json({"ok": True, "token": token, "username": uname})

# --- account data: positions and P&L ---
@app.get("/positions")
async def get_positions(account: str = "paper"):
    """
    Fetch the current positions for the specified Alpaca account. The
    ``account`` parameter selects either the paper or funded account. Environment
    variables ALPACA_KEY_PAPER/ALPACA_SECRET_PAPER and ALPACA_KEY_FUND/ALPACA_SECRET_FUND
    must be set accordingly. Returns the JSON response from the Alpaca API.

    In this hosted assistant context, any network requests are not executed; the
    code is provided for illustrative purposes only.
    """
    acct_type = (account or "paper").lower()
    if acct_type == "funded":
        key = ALPACA_KEY_FUND
        secret = ALPACA_SECRET_FUND
        base_url = ALPACA_BASE_URL_FUND
    else:
        key = ALPACA_KEY_PAPER
        secret = ALPACA_SECRET_PAPER
        base_url = ALPACA_BASE_URL_PAPER
    if not key or not secret:
        return _json({"ok": False, "detail": "Alpaca credentials not configured"}, 500)
    url = f"{base_url}/v2/positions"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    try:
        resp = requests.get(url, headers=headers)
        # Alpaca returns a list of position objects for this endpoint
        return _json({"ok": True, "positions": resp.json()})
    except Exception as e:
        return _json({"ok": False, "detail": f"Failed to fetch positions: {e}"}, 500)

@app.get("/pnl")
async def get_pnl(account: str = "paper"):
    """
    Compute unrealized P&L for the positions held in an Alpaca account. This
    endpoint fetches the open positions from Alpaca and sums the ``unrealized_pl``
    for each position (or computes it from ``market_value`` minus ``cost_basis`` if
    not provided). It returns both a total P&L and per‑position details. In this
    hosted assistant context, external HTTP requests are not executed; the code is
    provided for illustrative purposes only.
    """
    acct_type = (account or "paper").lower()
    if acct_type == "funded":
        key = ALPACA_KEY_FUND
        secret = ALPACA_SECRET_FUND
        base_url = ALPACA_BASE_URL_FUND
    else:
        key = ALPACA_KEY_PAPER
        secret = ALPACA_SECRET_PAPER
        base_url = ALPACA_BASE_URL_PAPER
    if not key or not secret:
        return _json({"ok": False, "detail": "Alpaca credentials not configured"}, 500)

    # fetch positions
    pos_url = f"{base_url}/v2/positions"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    try:
        pos_resp = requests.get(pos_url, headers=headers)
        pos_list = pos_resp.json() or []
        if not pos_list:
            return _json({"ok": True, "total_pnl": 0.0, "positions": []})
        results = []
        total_pnl = 0.0
        for p in pos_list:
            # Alpaca returns quantity as a string; convert to float
            qty = float(p.get("qty") or p.get("quantity") or 0)
            cost_basis = float(p.get("cost_basis") or 0)
            market_value = float(p.get("market_value") or 0)
            # compute P&L: prefer unrealized_pl if provided
            if p.get("unrealized_pl") is not None:
                pnl = float(p.get("unrealized_pl"))
            else:
                pnl = market_value - cost_basis
            total_pnl += pnl
            results.append({
                "symbol": p.get("symbol"),
                "quantity": qty,
                "avg_entry_price": float(p.get("avg_entry_price") or 0),
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pl": pnl
            })
        return _json({"ok": True, "total_pnl": total_pnl, "positions": results})
    except Exception as e:
        return _json({"ok": False, "detail": f"Failed to compute P&L: {e}"}, 500)


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
    sig_header = req.headers.get("X-Webhook-Signature")
    if secret and sig_header:
        import hmac, hashlib, base64
        digest = hmac.new(secret.encode(), body_bytes, hashlib.sha256).digest()
        expected = base64.b64encode(digest).decode()
        if not hmac.compare_digest(expected, sig_header):
            return _json({"ok": False, "detail": "invalid signature"}, 400)
    # In a production system you might persist the payload to a database or
    # notify other services. Here we simply print it to stdout.
    print("[Alpaca webhook]", json.dumps(payload, indent=2), flush=True)
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
        }
    )

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

# ---------- dashboard (no more NameError) ----------
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
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False)
