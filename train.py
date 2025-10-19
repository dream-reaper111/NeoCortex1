# -*- coding: utf-8 -*-
"""
Trainer with optional models (logreg | torch | lgbm), live NN emit, artifacts/metrics save,
PLUS TradingView/webhook features, fundamentals, multi-ticker idle mode, and exogenous features.

This version hardens feature engineering so we never end with X of shape (0, d).
- Rolling ops use min_periods.
- Windows auto-scale for short datasets.
- NaN-safe cleanup targets only needed columns.
- Fallback "basic" feature pack if advanced pack < 2 rows.
- Trainers guard for n<2 (duplicate if n==1, clean error if n==0).
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse, math, time, json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Try CuPy; fallback to NumPy
try:
    import cupy as cp
    xp = cp
    GPU = True
except Exception:
    import numpy as np  # ensure np symbol exists here too
    xp = np
    GPU = False

# Optional extras
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    import lightgbm as lgb
    LGBM_OK = True
except Exception:
    LGBM_OK = False

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(os.getenv("PINE_ROOT", ".")).resolve()
RUNS_ROOT = (ROOT / "artifacts"); RUNS_ROOT.mkdir(parents=True, exist_ok=True)
DATA_DIR = (ROOT / "data").resolve()

# -----------------------------
# Indicator primitives
# -----------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(span=length, adjust=False, min_periods=1).mean()

def rma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(alpha=1.0/float(length), adjust=False, min_periods=1).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, length)

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ru = rma(up, length)
    rd = rma(down, length).replace(0.0, pd.NA)
    rs = ru / rd
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    fast = max(2, int(fast)); slow = max(fast+1, int(slow)); signal = max(2, int(signal))
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def vwap_from_ohlcv(df: pd.DataFrame, price_col="Close", vol_col="Volume", window=20):
    price = df[price_col]
    vol = df[vol_col]
    pv = price * vol
    num = pv.rolling(window, min_periods=1).sum()
    den = vol.rolling(window, min_periods=1).sum().replace(0.0, pd.NA)
    return (num / den).ffill()

# -----------------------------
# Data fetch
# -----------------------------
def fetch_data(ticker: str, start=None, end=None, period=None, interval="1h") -> pd.DataFrame:
    os.environ.setdefault("PYTHON_YAHOO_USER_AGENT", "Mozilla/5.0 NeoTrainer/1.0")
    last_err = None
    for attempt in range(3):
        try:
            if period:
                df = yf.download(ticker, period=period, interval=interval,
                                 progress=False, threads=False, group_by="column")
            else:
                df = yf.download(ticker, start=start, end=end, interval=interval,
                                 progress=False, threads=False, group_by="column")
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_err = e
        time.sleep(0.8 * (attempt + 1))
    else:
        if last_err:
            raise last_err
        df = pd.DataFrame()

    if df is None or df.empty:
        raise SystemExit("No data from yfinance. Check ticker / time range / interval.")

    # flatten if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.droplevel(-1, axis=1)
        except Exception:
            pass

    cols_keep = {"open","high","low","close","adjclose","volume"}
    df = df[[c for c in df.columns if str(c).strip().lower().replace(" ", "") in cols_keep]].copy()

    if "Adj Close" in df.columns and "Close" not in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)

    needed = ["Open","High","Low","Close","Volume"]
    for col in needed:
        if col not in df.columns:
            raise SystemExit(f"Missing column: {col}")
    df = df[needed].copy()

    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index().dropna(how="any")
    if df.empty:
        raise SystemExit("After cleaning, OHLCV frame is empty.")
    return df

# -----------------------------
# TradingView/webhooks → features
# -----------------------------
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                out.append({"payload": {"message": line}})
    return out

def load_tradingview_webhooks(ticker: str) -> pd.DataFrame:
    f = DATA_DIR / "webhooks" / "tradingview" / f"{ticker.upper()}.jsonl"
    rows = _read_jsonl(f)
    recs: List[Dict[str, Any]] = []
    for r in rows:
        p = r.get("payload", {})
        order = p.get("order", {}) if isinstance(p, dict) else {}
        sym = order.get("symbol") or p.get("symbol") or p.get("ticker") or ticker
        side = (order.get("side") or "").lower()
        reduce_only = bool(order.get("reduceOnly", False))
        client_id = str(p.get("clientOrderId") or p.get("client_id") or p.get("clientOrderID") or "")

        evt = "entry"
        if reduce_only or side == "sell":
            evt = "exit"
            if client_id.endswith("-TP1"): evt = "tp1"
            elif client_id.endswith("-TP2"): evt = "tp2"
            elif client_id.endswith("-TP3"): evt = "tp3"
            elif client_id.endswith("-SL"):  evt = "sl"

        t_bar_ms = r.get("t_bar")
        if t_bar_ms:
            ts = pd.to_datetime(int(t_bar_ms), unit="ms", utc=True)
        else:
            ts = pd.to_datetime(r.get("t_server"), utc=True, errors="coerce")

        recs.append({
            "ts": ts,
            "ticker": str(sym).upper(),
            "evt": evt,
            "side": side or ("buy" if evt == "entry" else "sell")
        })
    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return pd.DataFrame(columns=["ts","ticker","evt","side"]).astype({"ts":"datetime64[ns, UTC]"})
    return df.sort_values("ts").reset_index(drop=True)

def augment_with_tradingview(df_bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df_bars.copy()
    out["tv_entry_recent"] = 0
    out["tv_exit_recent"]  = 0
    tv = load_tradingview_webhooks(ticker)
    if tv.empty:
        return out

    dt = (df_bars.index[2] - df_bars.index[1]) if len(df_bars) >= 3 else pd.Timedelta(minutes=60)
    entries = tv[tv["evt"] == "entry"][["ts"]].rename(columns={"ts":"ts_tv"}).copy()
    exits   = tv[tv["evt"].isin(["tp1","tp2","tp3","sl","exit"])][["ts"]].rename(columns={"ts":"ts_tv"}).copy()

    def _flag_recent(events: pd.DataFrame) -> pd.Series:
        if events.empty:
            return pd.Series(0, index=df_bars.index, dtype=int)
        merged = pd.merge_asof(
            left=pd.DataFrame({"ts_bar": df_bars.index}).sort_values("ts_bar"),
            right=events.sort_values("ts_tv"),
            left_on="ts_bar", right_on="ts_tv",
            direction="backward", tolerance=dt
        )
        return merged["ts_tv"].notna().astype(int).set_axis(df_bars.index)

    out["tv_entry_recent"] = _flag_recent(entries)
    out["tv_exit_recent"]  = _flag_recent(exits)
    return out

# -----------------------------
# Optional: Robinhood/Webull recent-bar flags
# -----------------------------
def _load_ext_ohlcv(source: str, ticker: str) -> pd.DataFrame:
    f = DATA_DIR / source.lower() / "ohlcv" / f"{ticker.upper()}.jsonl"
    rows = _read_jsonl(f)
    recs = []
    for r in rows:
        bar = r.get("bar", {})
        t = bar.get("t") or bar.get("time") or r.get("t_server")
        if t is None:
            continue
        if isinstance(t, (int, float)) and t > 1e11:
            ts = pd.to_datetime(int(t), unit="ms", utc=True)
        else:
            ts = pd.to_datetime(t, utc=True, errors="coerce")
        recs.append({"ts": ts})
    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return pd.DataFrame(columns=["ts"]).astype({"ts":"datetime64[ns, UTC]"})
    return df.sort_values("ts").reset_index(drop=True)

def augment_with_ext_sources(df_bars: pd.DataFrame, ticker: str, sources: List[str]) -> pd.DataFrame:
    out = df_bars.copy()
    dt = (df_bars.index[2] - df_bars.index[1]) if len(df_bars) >= 3 else pd.Timedelta(minutes=60)
    for src in sources:
        df = _load_ext_ohlcv(src, ticker)
        col = f"{src.lower()}_bar_recent"
        out[col] = 0
        if df.empty:
            continue
        merged = pd.merge_asof(
            left=pd.DataFrame({"ts_bar": df_bars.index}).sort_values("ts_bar"),
            right=df.rename(columns={"ts":"ts_ext"}).sort_values("ts_ext"),
            left_on="ts_bar", right_on="ts_ext",
            direction="backward", tolerance=dt
        )
        out[col] = merged["ts_ext"].notna().astype(int).set_axis(df_bars.index)
    return out

# -----------------------------
# Fundamentals (yfinance snapshot)
# -----------------------------
FA_KEYS = [
    ("trailingPE",        "fa_pe"),
    ("forwardPE",         "fa_forward_pe"),
    ("priceToBook",       "fa_pb"),
    ("priceToSalesTrailing12Months","fa_ps"),
    ("profitMargins",     "fa_profit_margin"),
    ("operatingMargins",  "fa_oper_margin"),
    ("returnOnEquity",    "fa_roe"),
    ("grossMargins",      "fa_gross_margin"),
    ("dividendYield",     "fa_div_yield"),
    ("beta",              "fa_beta"),
    ("marketCap",         "fa_market_cap")
]

def fetch_fundamentals_yf(ticker: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        tkr = yf.Ticker(ticker)
        info = {}
        try:
            info = tkr.info or {}
        except Exception:
            info = {}
        fast = getattr(tkr, "fast_info", None)
        for key, col in FA_KEYS:
            val = info.get(key, None)
            if val is None and fast is not None:
                val = getattr(fast, key, None) if hasattr(fast, key) else None
            try:
                out[col] = float(val) if val is not None and np.isfinite(val) else np.nan
            except Exception:
                out[col] = np.nan
    except Exception:
        for _, col in FA_KEYS:
            out[col] = np.nan
    return out

def attach_fundamentals(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    fa = fetch_fundamentals_yf(ticker)
    out = df.copy()
    for col in [c for _, c in FA_KEYS]:
        out[col] = fa.get(col, np.nan)
    for col in [c for _, c in FA_KEYS]:
        med = out[col].median(skipna=True)
        out[col] = out[col].fillna(0.0 if np.isnan(med) else med)
    return out

# -----------------------------
# Features & Labels (Long + Short) — hardened
# -----------------------------
def _scale(n: int, base: int, lo: int = 3) -> int:
    # shrink windows when data is short; baseline ~260 bars
    s = min(1.0, max(0.2, n / 260.0))
    return max(lo, int(round(base * s)))

def _nan_safe_finalize(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    # forward/back fill numerics; keep labels deterministic
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(float)
    # only enforce required cols
    out[required_cols] = out[required_cols].fillna(method="ffill").fillna(method="bfill")
    out = out.dropna(subset=required_cols, how="any")
    return out

def _basic_pack(df: pd.DataFrame) -> pd.DataFrame:
    # ultra-compact pack for tiny datasets (still gives 2+ rows)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    fast = ema(c, 3); slow = ema(c, 7)
    mline, msig, _ = macd(c, fast=4, slow=8, signal=3)
    rsi7 = rsi_wilder(c, 7)
    upVol   = v.where(c > o, 0.0); downVol = v.where(c < o, 0.0)
    delta   = upVol - downVol
    bullEngulf = (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))
    bearEngulf = (c < o) & (c.shift(1) > o.shift(1)) & (c < o.shift(1)) & (o > c.shift(1))
    scoreL = ((fast>slow).astype(float) + (mline>msig).astype(float) + (rsi7.between(35,75)).astype(float)) / 3.0
    scoreS = ((fast<slow).astype(float) + (mline<msig).astype(float) + (rsi7.between(25,65)).astype(float)) / 3.0
    out = df.copy()
    out["fast"]=fast; out["slow"]=slow
    out["macdLine"]=mline; out["sigLine"]=msig
    out["rsi"]=rsi7; out["deltaZ"]=0.0
    out["volSpike"]= (v > v.rolling(3, min_periods=1).mean()).astype(int)
    out["bullEngulf"]=bullEngulf.astype(int); out["bearEngulf"]=bearEngulf.astype(int)
    out["LabelLong"]=(scoreL>=0.5).astype(int); out["LabelShort"]=(scoreS>=0.5).astype(int)
    req = ["fast","slow","macdLine","sigLine","rsi","deltaZ","volSpike","bullEngulf","bearEngulf","LabelLong","LabelShort"]
    out = _nan_safe_finalize(out, req)
    if len(out) == 1:
        out = pd.concat([out, out.tail(1)], axis=0)  # duplicate to reach 2
    return out

def build_features_and_labels(
    df: pd.DataFrame,
    *,
    mode: str = "hft",
    # thresholds left as-is; robustness handled internally
    htfTf: str = "60",
    entryThrBase: float = 1.2,
    zThreshBase: float = -2.4,
    cooldownBase: int = 0,
    minVotesBase: int = 3,
    relaxAfterBarsBase: int = 300,
    relaxFactorBase: float = 0.85,
    useHTFTrend: bool = False,
    useHTFVWAP: bool = False,
    useVoteGate: bool = False,
    useTimeWindow: bool = False,
    sessionTimes: str = "0930-1600",
    debugMode: bool = False,
    debugEveryNBars: int = 0,
    onePerHTF: bool = False,
) -> pd.DataFrame:
    n = len(df)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # Adaptive windows (shorter when n is small)
    if mode.lower() == "lft":
        atrLen       = _scale(n, 12)
        obLen        = _scale(n, 8, lo=3)
        rsi_low_L, rsi_high_L = 38, 78
        rsi_low_S, rsi_high_S = 22, 62
        volSpikeK    = 0.95
        entryThreshold = max(0.9, entryThrBase * 0.85)
        cooldownBars   = max(0, int(cooldownBase * 0.5))
        minVotes       = max(1, int(minVotesBase - 1))
        relaxAfterBars = max(100, int(relaxAfterBarsBase * 0.6))
        relaxFactor    = max(0.70, relaxFactorBase * 0.9)
        fastLen, slowLen = _scale(n, 8), max(_scale(n, 21), _scale(n, 8)+1)
        m_fast, m_slow, m_sig = _scale(n, 12), max(_scale(n, 26), _scale(n, 12)+1), _scale(n, 9)
    else:
        atrLen       = _scale(n, 14)
        obLen        = _scale(n, 10, lo=3)
        rsi_low_L, rsi_high_L = 40, 75
        rsi_low_S, rsi_high_S = 25, 60
        volSpikeK    = 1.00
        entryThreshold = entryThrBase
        cooldownBars   = cooldownBase
        minVotes       = minVotesBase
        relaxAfterBars = relaxAfterBarsBase
        relaxFactor    = relaxFactorBase
        fastLen, slowLen = _scale(n, 8), max(_scale(n, 21), _scale(n, 8)+1)
        m_fast, m_slow, m_sig = _scale(n, 12), max(_scale(n, 26), _scale(n, 12)+1), _scale(n, 9)

    zThresh = float(zThreshBase); zAbs = abs(zThresh)

    # Core indicators with min_periods and adaptive windows
    atrSeries = atr(h, l, c, atrLen)
    fastMA = ema(c, fastLen)
    slowMA = ema(c, slowLen)
    macdLine, sigLine, _ = macd(c, fast=m_fast, slow=m_slow, signal=m_sig)
    rsi = rsi_wilder(c, max(7, _scale(n, 14)))

    # Delta / OB proxy (min_periods set)
    upVol   = v.where(c > o, 0.0)
    downVol = v.where(c < o, 0.0)
    delta   = upVol - downVol

    bullEngulf = (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))
    bearEngulf = (c < o) & (c.shift(1) > o.shift(1)) & (c < o.shift(1)) & (o > c.shift(1))

    deltaMA  = delta.rolling(obLen, min_periods=1).mean()
    deltaStd = delta.rolling(obLen, min_periods=2).std(ddof=0)
    deltaStd = deltaStd.replace(0.0, pd.NA)
    deltaZ   = ((delta - deltaMA) / deltaStd).fillna(0.0)

    volAvg   = v.rolling(obLen, min_periods=1).mean()
    volSpike = (v > (volAvg * volSpikeK))

    obLong  = (deltaZ >  zThresh) & volSpike
    obShort = (deltaZ < -zAbs)    & volSpike

    maLong   = (fastMA > slowMA) & (fastMA.shift(1) <= slowMA.shift(1))
    maShort  = (fastMA < slowMA) & (fastMA.shift(1) >= slowMA.shift(1))
    macdLong = (macdLine > sigLine) & (macdLine.shift(1) <= sigLine.shift(1))
    macdShort= (macdLine < sigLine) & (macdLine.shift(1) >= sigLine.shift(1))

    rsiLong  = (rsi > rsi_low_L) & (rsi < rsi_high_L)
    rsiShort = (rsi > rsi_low_S) & (rsi < rsi_high_S)

    scoreL = (
        (bullEngulf & (delta > 0)).astype(float) * 0.7 +
        obLong.astype(float) * 0.7 +
        maLong.astype(float) * 0.6 +
        macdLong.astype(float) * 0.6 +
        rsiLong.astype(float) * 0.4
    )
    scoreS = (
        (bearEngulf & (delta < 0)).astype(float) * 0.7 +
        obShort.astype(float) * 0.7 +
        maShort.astype(float) * 0.6 +
        macdShort.astype(float) * 0.6 +
        rsiShort.astype(float) * 0.4
    )

    doEnterLong  = (scoreL >= (entryThreshold))
    doEnterShort = (scoreS >= (entryThreshold))

    out = df.copy()
    out["fast"] = fastMA; out["slow"] = slowMA
    out["macdLine"] = macdLine; out["sigLine"] = sigLine
    out["rsi"] = rsi; out["deltaZ"] = deltaZ
    out["volSpike"] = volSpike.astype(int)
    out["bullEngulf"] = bullEngulf.astype(int)
    out["bearEngulf"] = bearEngulf.astype(int)
    out["scoreL"] = scoreL; out["scoreS"] = scoreS
    out["LabelLong"]  = doEnterLong.astype(int)
    out["LabelShort"] = doEnterShort.astype(int)

    # Only require these to be finite; don't blanket-drop
    required = ["fast","slow","macdLine","sigLine","rsi","deltaZ","volSpike","bullEngulf","bearEngulf","LabelLong","LabelShort"]
    out = _nan_safe_finalize(out, required)

    if len(out) < 2:
        # fallback tiny pack
        out = _basic_pack(df)

    print(f"Bars={len(out)}  L= {int(out['LabelLong'].sum())} ({out['LabelLong'].mean():.2%})  "
          f"S= {int(out['LabelShort'].sum())} ({out['LabelShort'].mean():.2%})  mode={mode.upper()}")
    return out

# -----------------------------
# UI compatibility: build_features wrapper with exog
# -----------------------------
def _harmonize_exog(df_base: pd.DataFrame, exog: Any) -> pd.DataFrame:
    if exog is None:
        return pd.DataFrame(index=df_base.index)
    if isinstance(exog, dict):
        exog = pd.DataFrame(exog)
    if not isinstance(exog, pd.DataFrame):
        raise TypeError("exog must be a pandas DataFrame or dict of arrays/Series")
    ex = exog.copy()
    ex.index = pd.to_datetime(ex.index, utc=True, errors="coerce")
    ex = ex.sort_index().reindex(df_base.index, method="ffill")
    cols = {}
    for c in ex.columns:
        col = f"ex_{c}" if not str(c).startswith("ex_") else str(c)
        cols[c] = col
    ex = ex.rename(columns=cols)
    for c in ex.columns:
        ex[c] = pd.to_numeric(ex[c], errors="coerce")
    ex = ex.fillna(method="ffill").fillna(method="bfill")
    return ex

def build_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    exog = kwargs.pop("exog", None)
    out = build_features_and_labels(df, **kwargs)
    if exog is not None and len(out):
        ex = _harmonize_exog(out, exog)
        out = out.join(ex, how="left")
    return out

# -----------------------------
# Models (logreg | torch | lgbm)
# -----------------------------
BASE_X_COLUMNS = ["fast","slow","macdLine","sigLine","rsi","deltaZ","volSpike","bullEngulf","bearEngulf"]
TV_X_COLUMNS   = ["tv_entry_recent","tv_exit_recent"]
FA_X_COLUMNS   = [c for _, c in FA_KEYS]
EXT_X_COLUMNS  = ["robinhood_bar_recent","webull_bar_recent"]

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _open_jsonl(path: Path):
    _ensure_dir(path.parent)
    return path.open("a", encoding="utf-8")

def _emit_jsonl(fh, obj):
    fh.write(json.dumps(obj, separators=(",",":")) + "\n"); fh.flush()

def _write_nn_graph(run_dir: Path, layers: List[int], model_name: str):
    obj = {"title": "Neural Network", "model": f"{model_name} " + "+".join(map(str,layers)), "layers":[{"size":s} for s in layers]}
    (run_dir / "nn_graph.json").write_text(json.dumps(obj, indent=2))

def _make_nn_emitter(run_dir: Path):
    _ensure_dir(run_dir)
    fh = _open_jsonl(run_dir / "nn_state.jsonl")
    def emit(**kw):
        row = {"type":"nn_stats","t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
        _emit_jsonl(fh, row)
    return emit

def _make_train_logger(run_dir: Path):
    _ensure_dir(run_dir)
    fh = _open_jsonl(run_dir / "train.log.jsonl")
    def emit(row):
        _emit_jsonl(fh, row)
    return emit

def _select_xcols(df_feat: pd.DataFrame, use_tv: bool, use_fa: bool, use_ext: bool) -> List[str]:
    cols = list(BASE_X_COLUMNS)
    if use_tv:  cols += [c for c in TV_X_COLUMNS if c in df_feat.columns]
    if use_fa:  cols += [c for c in FA_X_COLUMNS if c in df_feat.columns]
    if use_ext: cols += [c for c in EXT_X_COLUMNS if c in df_feat.columns]
    cols += [c for c in df_feat.columns if str(c).startswith("ex_")]
    seen = set(); out = []
    for c in cols:
        if c in df_feat.columns and c not in seen:
            out.append(c); seen.add(c)
    return out

def sigmoid(x):
    return 1 / (1 + xp.exp(-x))

def _guard_min_samples(X: np.ndarray, y: np.ndarray, need: int = 2) -> (np.ndarray, np.ndarray):
    n = X.shape[0]
    if n >= need:
        return X, y
    if n == 1:
        X = np.vstack([X, X])
        y = np.append(y, y[-1])
        return X, y
    raise SystemExit("Not enough samples after feature engineering; try a longer period or a lower interval.")

def gpu_logreg_fit(X_np: np.ndarray, y_np: np.ndarray, *, lr=0.05, epochs=1500, l2=0.0, verbose=True, nn_emit=None):
    n, d = X_np.shape
    X_np, y_np = _guard_min_samples(X_np, y_np, need=2)
    n, d = X_np.shape
    X = xp.asarray(X_np); y = xp.asarray(y_np).reshape(-1, 1)
    w = xp.zeros((d, 1), dtype=X.dtype); b = xp.zeros((1,), dtype=X.dtype)
    for e in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        grad_w = (X.T @ (p - y)) / n + l2 * w
        grad_b = xp.mean(p - y, axis=0)
        w -= lr * grad_w; b -= lr * grad_b
        if verbose and (e+1) % max(1, epochs//5) == 0:
            eps = 1e-9
            loss = -xp.mean(y * xp.log(p+eps) + (1-y) * xp.log(1-p+eps)) + (l2/2)*xp.sum(w*w)
            val = float(loss.get() if GPU else loss)
            if nn_emit:
                wnp = cp.asnumpy(w) if GPU else w
                nn_emit(epoch=e+1, loss=float(val), layers=[int(d), 1],
                        weight_mean=float(wnp.mean()), weight_std=float(wnp.std()))
    w_np = cp.asnumpy(w) if GPU else w
    b_np = cp.asnumpy(b) if GPU else b
    p_train = sigmoid(xp.asarray(X_np) @ (xp.asarray(w_np)) + xp.asarray(b_np))
    probs_np = cp.asnumpy(p_train).ravel() if GPU else p_train.ravel()
    preds = (p_train >= 0.5).astype(int)
    preds_np = cp.asnumpy(preds).ravel() if GPU else preds.ravel()
    return w_np, b_np, preds_np, probs_np

class MLP(nn.Module):
    def __init__(self, d_in, d_hid=32):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_hid)
        self.fc3 = nn.Linear(d_hid, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_logreg(df_feat: pd.DataFrame, side: str, *, epochs=1500, lr=0.05, l2=0.0, verbose=True, run_dir: Path=None, xcols: Optional[List[str]]=None):
    nn_emit = _make_nn_emitter(run_dir) if run_dir else None
    Xcols = xcols or BASE_X_COLUMNS
    X = df_feat[Xcols].astype(float).values
    y = (df_feat["LabelLong"] if side=="long" else df_feat["LabelShort"]).astype(int).values
    X, y = _guard_min_samples(X, y, need=2)
    w, b, preds, probs = gpu_logreg_fit(X, y, lr=lr, epochs=epochs, l2=l2, verbose=verbose, nn_emit=nn_emit)
    out = df_feat.copy()
    col = "PredLong" if side=="long" else "PredShort"
    out[col] = preds
    _write_nn_graph(run_dir, [len(Xcols), 1], "logreg")
    acc = (out[col] == (df_feat["LabelLong"] if side=="long" else df_feat["LabelShort"])).mean()
    return out, {"acc": float(acc), "probs": probs.tolist(), "xcols": Xcols}

def train_torch(df_feat: pd.DataFrame, side: str, *, epochs=200, lr=1e-3, run_dir: Path=None, device=None, xcols: Optional[List[str]]=None):
    if not TORCH_OK:
        raise SystemExit("PyTorch not installed, cannot use --model torch")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xcols = xcols or BASE_X_COLUMNS
    X = df_feat[Xcols].astype(float).values
    y = (df_feat["LabelLong"] if side=="long" else df_feat["LabelShort"]).astype(float).values.reshape(-1,1)
    X, y = _guard_min_samples(X, y.ravel(), need=2); y = y.reshape(-1,1)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    model = MLP(X.shape[1], d_hid=32).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    nn_emit = _make_nn_emitter(run_dir) if run_dir else (lambda **kw: None)
    _write_nn_graph(run_dir, [X.shape[1], 32, 32, 1], "torch")
    model.train()
    for e in range(1, epochs+1):
        opt.zero_grad()
        logits = model(X); loss = lossf(logits, y)
        loss.backward(); opt.step()
        if e % max(1, epochs//5) == 0:
            val = float(loss.detach().cpu().item())
            with torch.no_grad():
                wmean = float(torch.cat([p.view(-1) for p in model.parameters()]).mean().cpu())
                wstd  = float(torch.cat([p.view(-1) for p in model.parameters()]).std().cpu())
            nn_emit(epoch=e, loss=val, device=device, weight_mean=wmean, weight_std=wstd)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).cpu().numpy().ravel()
    preds = (probs >= 0.5).astype(int)
    out = df_feat.copy()
    col = "PredLong" if side=="long" else "PredShort"
    out[col] = preds
    acc = (out[col] == (df_feat["LabelLong"] if side=="long" else df_feat["LabelShort"])).mean()
    return out, {"acc": float(acc), "probs": probs.tolist(), "xcols": Xcols}

def train_lgbm(df_feat: pd.DataFrame, side: str, *, run_dir: Path=None, xcols: Optional[List[str]]=None):
    if not LGBM_OK:
        raise SystemExit("LightGBM not installed, cannot use --model lgbm")
    Xcols = xcols or BASE_X_COLUMNS
    X = df_feat[Xcols].astype(float).values
    y = (df_feat["LabelLong"] if side=="long" else df_feat["LabelShort"]).astype(int).values
    X, y = _guard_min_samples(X, y, need=2)
    dtrain = lgb.Dataset(X, label=y)
    params = dict(objective="binary", metric="auc", verbose=-1, learning_rate=0.05, num_leaves=15,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1)
    model = lgb.train(params, dtrain, num_boost_round=200)
    probs = model.predict(X)
    preds = (probs >= 0.5).astype(int)
    out = df_feat.copy()
    col = "PredLong" if side=="long" else "PredShort"
    out[col] = preds
    try:
        imp = model.feature_importance(); names = Xcols
        idx = np.argsort(imp)[::-1]
        plt.figure(figsize=(max(6, len(names)*0.4), 4))
        plt.bar(range(len(names)), np.array(imp)[idx])
        plt.xticks(range(len(names)), [names[i] for i in idx], rotation=45, ha="right")
        plt.title(f"Feature Importance ({side})")
        plt.tight_layout()
        plt.savefig(run_dir / f"feature_importance_{side}.png", dpi=140)
        plt.close()
    except Exception:
        pass
    acc = (out[col] == (df_feat["LabelLong"] if side=="long" else df_feat["LabelShort"])).mean()
    return out, {"acc": float(acc), "probs": probs.tolist(), "xcols": Xcols}

# -----------------------------
# Plots & artifact helpers
# -----------------------------
def plot_results_to(run_dir: Path, df_feat: pd.DataFrame, ticker: str, use_pred: bool, side: str):
    close = df_feat["Close"]
    if side in ("long", "both"):
        lblL = df_feat.index[df_feat["LabelLong"]==1]
        prdL = df_feat.index[df_feat.get("PredLong", pd.Series(index=df_feat.index, dtype=int))==1] if use_pred else lblL
    else:
        lblL = prdL = pd.Index([])
    if side in ("short", "both"):
        lblS = df_feat.index[df_feat["LabelShort"]==1]
        prdS = df_feat.index[df_feat.get("PredShort", pd.Series(index=df_feat.index, dtype=int))==1] if use_pred else lblS
    else:
        lblS = prdS = pd.Index([])
    plt.figure(figsize=(13,6))
    plt.plot(df_feat.index, close, label="Close", linewidth=1.0)
    if len(lblL):
        ymin = close.loc[lblL] * 0.995; ymax = close.loc[lblL] * 1.005
        plt.vlines(lblL, ymin=ymin, ymax=ymax, colors="green", alpha=0.25, label="Label Long")
    if len(lblS):
        ymin = close.loc[lblS] * 0.995; ymax = close.loc[lblS] * 1.005
        plt.vlines(lblS, ymin=ymin, ymax=ymax, colors="red", alpha=0.25, label="Label Short")
    if len(prdL): plt.scatter(prdL, close.loc[prdL], marker="^", s=36, label="Model Long")
    if len(prdS): plt.scatter(prdS, close.loc[prdS], marker="v", s=36, label="Model Short")
    plt.title(f"{ticker} - Entries ({'Pred' if use_pred else 'Labels'}) [{side.upper()}]")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / (f"trades_pred_{side}.png" if use_pred else f"trades_label_{side}.png"), dpi=140)
    plt.close()

def save_confusion_and_roc(run_dir: Path, y_true, y_pred, y_prob, side: str):
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        plt.figure(figsize=(3.6,3.2))
        plt.imshow(cm, cmap="Blues"); plt.title(f"CM ({side})")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i,j]), ha="center", va="center", color="black")
        plt.xticks([0,1], ["0","1"]); plt.yticks([0,1], ["0","1"])
        plt.tight_layout()
        plt.savefig(run_dir / f"cm_{side}.png", dpi=140); plt.close()
    except Exception:
        pass
    try:
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(3.6,3.2))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1], 'k--', alpha=0.3)
        plt.legend(); plt.title(f"ROC ({side})"); plt.tight_layout()
        plt.savefig(run_dir / f"roc_{side}.png", dpi=140); plt.close()
        return float(auc)
    except Exception:
        return 0.0

def plot_equity_to(run_dir: Path, eq_df: pd.DataFrame, title="Equity Curve"):
    if eq_df.empty: return
    plt.figure(figsize=(12,4))
    plt.plot(eq_df.index, eq_df["equity"], linewidth=1.2, label="Equity")
    plt.title(title); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / "equity.png", dpi=140); plt.close()

def save_price_preview(df: pd.DataFrame, ticker: str, run_dir: Path):
    try:
        close = df["Close"]
        plt.figure(figsize=(8, 3))
        plt.plot(close.index, close.values, linewidth=1.2)
        plt.title(f"{ticker} - Close")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(run_dir / "yfinance_price.png", dpi=130)
        plt.close()
    except Exception:
        pass

# -----------------------------
# Backtest & PnL (+ SSE emits)
# -----------------------------
def simulate_trades(
    df_feat: pd.DataFrame,
    *,
    side: str = "long",
    start_capital: float = 70000.0,
    qty_percent: float = 10.0,
    use_pred: bool = False,
    fees_bps: float = 0.0,
    mode: str = "hft",
    be_gain_pct: float = 0.20,
    pt_gain_pct: float = 0.60,
    sse_emit=None
):
    if mode.lower() == "lft":
        atrSLmult, atrTPmult = 0.70, 0.90
        atrLen = 12
        useBreakeven = True
        useTrailATR = False
        trailATRmult = 0.80
    else:
        atrSLmult, atrTPmult = 0.80, 0.95
        atrLen = 14
        useBreakeven = True
        useTrailATR = False
        trailATRmult = 0.80

    o = df_feat["Open"]; h = df_feat["High"]; l = df_feat["Low"]; c = df_feat["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1.0/atrLen, adjust=False, min_periods=1).mean().ffill()

    sigL = df_feat["PredLong"  if use_pred and "PredLong"  in df_feat else "LabelLong" ].astype(int)
    sigS = df_feat["PredShort" if use_pred and "PredShort" in df_feat else "LabelShort"].astype(int)

    equity = start_capital
    equity_curve = []
    trades = []

    in_pos = False
    pos_side = None
    qty = 0.0
    entry_px = 0.0
    entry_time = None

    rt_fee = fees_bps / 10000.0
    idx = df_feat.index

    for i in range(len(df_feat)):
        ts = idx[i]
        price = float(c.iloc[i])
        hi = float(h.iloc[i]); lo = float(l.iloc[i])
        atr_now = float(atr_series.iloc[i]) if pd.notna(atr_series.iloc[i]) else 1e-8

        if (not in_pos):
            want_long  = (side in ("long","both"))  and (sigL.iloc[i] == 1)
            want_short = (side in ("short","both")) and (sigS.iloc[i] == 1)

            if want_long or want_short:
                chosen = "LONG" if (want_long and (not want_short or df_feat.get("scoreL", pd.Series(0.0, index=df_feat.index)).iloc[i] >=
                                                   df_feat.get("scoreS", pd.Series(0.0, index=df_feat.index)).iloc[i])) else "SHORT"
                qty = max(1.0, math.floor((equity * (qty_percent/100.0)) / price)) if price > 0 else 0.0
                if qty > 0:
                    in_pos = True
                    pos_side = chosen
                    entry_px = price
                    entry_time = ts
                    if sse_emit:
                        sse_emit({"type":"trade_open","t": str(ts), "side": chosen, "price": float(price),
                                  "entry_col": ("PredShort" if (use_pred and chosen=="SHORT") else ("PredLong" if use_pred else ("LabelShort" if chosen=="SHORT" else "LabelLong"))) })

        if in_pos:
            if pos_side == "LONG":
                base_stop = entry_px - atr_now * atrSLmult
                tp_quick  = entry_px + atr_now * atrTPmult
                one_r     = entry_px + atr_now * atrSLmult

                stop_now = base_stop
                if useBreakeven and entry_px > 0 and hi >= entry_px * (1.0 + be_gain_pct):
                    stop_now = max(stop_now, entry_px)
                if useTrailATR and price >= one_r:
                    stop_now = max(stop_now, price - atr_now * trailATRmult)

                early_tp_hit = (entry_px > 0) and (hi >= entry_px * (1.0 + pt_gain_pct))

                if lo <= stop_now:
                    exit_px = stop_now; exit_reason = "STOP_L"
                elif early_tp_hit:
                    exit_px = entry_px * (1.0 + pt_gain_pct); exit_reason = f"EARLY_TP_L_{int(pt_gain_pct*100)}pct"
                elif hi >= tp_quick:
                    exit_px = tp_quick; exit_reason = "TP_L"
                else:
                    exit_px = None

            else:
                base_stop = entry_px + atr_now * atrSLmult
                tp_quick  = entry_px - atr_now * atrTPmult
                one_r     = entry_px - atr_now * atrSLmult

                stop_now = base_stop
                if useBreakeven and entry_px > 0 and lo <= entry_px * (1.0 - be_gain_pct):
                    stop_now = min(stop_now, entry_px)
                if useTrailATR and price <= one_r:
                    stop_now = min(stop_now, price + atr_now * trailATRmult)

                early_tp_hit = (entry_px > 0) and (lo <= entry_px * (1.0 - pt_gain_pct))

                if hi >= stop_now:
                    exit_px = stop_now; exit_reason = "STOP_S"
                elif early_tp_hit:
                    exit_px = entry_px * (1.0 - pt_gain_pct); exit_reason = f"EARLY_TP_S_{int(pt_gain_pct*100)}pct"
                elif lo <= tp_quick:
                    exit_px = tp_quick; exit_reason = "TP_S"
                else:
                    exit_px = None

            if exit_px is not None:
                sell_qty = qty
                gross = (exit_px - entry_px) * sell_qty if pos_side=="LONG" else (entry_px - exit_px) * sell_qty
                fee_cost = (entry_px + exit_px) * sell_qty * rt_fee
                pnl = gross - fee_cost
                equity += pnl
                trades.append({
                    "entry_time": entry_time, "entry_price": entry_px,
                    "exit_time": ts, "exit_price": exit_px, "qty": sell_qty,
                    "gross": gross, "fees": -fee_cost, "pnl": pnl,
                    "reason": exit_reason, "side": pos_side
                })
                if sse_emit:
                    sse_emit({"type":"trade_close","t": str(ts),"side": pos_side,"price": float(exit_px),
                              "entry_col": ("PredShort" if (use_pred and pos_side=="SHORT") else ("PredLong" if use_pred else ("LabelShort" if pos_side=="SHORT" else "LabelLong"))),
                              "pnl": float(pnl), "reason": exit_reason})
                in_pos = False
                qty = 0.0
                entry_px = 0.0
                entry_time = None

        mtm = equity if not in_pos else (equity + ((price - entry_px) * qty) if pos_side=="LONG" else equity + ((entry_px - price) * qty))
        equity_curve.append({"time": ts, "equity": mtm})
        if sse_emit and (i % max(1, len(df_feat)//250) == 0):
            sse_emit({"type":"equity","t": str(ts),"equity": float(mtm),
                      "entry_col": ("PredBoth" if (side=='both' and use_pred) else ("PredShort" if (use_pred and side=='short') else ("PredLong" if use_pred else ("LabelShort" if side=='short' else "LabelLong"))))})

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve).set_index("time")
    win_rate = ((trades_df["pnl"] > 0).sum() / max(1, len(trades_df))) * 100.0 if len(trades_df) else 0.0
    ret_pct = ((eq_df["equity"].iloc[-1] / start_capital) - 1.0) * 100.0 if len(eq_df) else 0.0
    dd_pct  = ((eq_df["equity"].cummax() - eq_df["equity"]).max() /
               max(1e-9, eq_df["equity"].cummax().max()) * 100.0) if len(eq_df) else 0.0
    stats = {"final_equity": float(eq_df["equity"].iloc[-1]) if len(eq_df) else start_capital,
             "win_rate_pct": float(win_rate), "return_pct": float(ret_pct),
             "max_dd_pct": float(dd_pct), "trades": int(len(trades_df))}
    return trades_df, eq_df, stats

# -----------------------------
# One-ticker training routine
# -----------------------------
def train_for_ticker(
    ticker: str,
    *,
    interval: str,
    period: Optional[str],
    start: Optional[str],
    end: Optional[str],
    mode: str,
    model_name: str,
    epochs: int,
    use_tv: bool,
    use_fa: bool,
    use_ext: bool,
    ext_sources: List[str],
    parent_run_dir: Optional[Path] = None
) -> Dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = (parent_run_dir or RUNS_ROOT) / f"run-{ts}-{ticker.upper()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_emit = _make_train_logger(run_dir)
    nn_emit = _make_nn_emitter(run_dir)
    _write_nn_graph(run_dir, [len(BASE_X_COLUMNS), 1], "bootstrap")
    nn_emit(epoch=0, loss=None, note=f"init {ticker}")

    df = fetch_data(ticker, start=start, end=end, period=period, interval=interval)
    save_price_preview(df, ticker, run_dir)

    feat = build_features_and_labels(df, mode=mode)

    if use_tv:
        feat = augment_with_tradingview(feat, ticker)
    if use_ext and ext_sources:
        feat = augment_with_ext_sources(feat, ticker, ext_sources)
    if use_fa:
        feat = attach_fundamentals(feat, ticker)

    xcols = _select_xcols(feat, use_tv=use_tv, use_fa=use_fa, use_ext=use_ext)

    # Final sanity: ensure we have some rows
    if len(feat) == 0 or len(xcols) == 0:
        raise SystemExit("Not enough engineered samples; increase your period or choose a lower interval.")

    yL = feat["LabelLong"].astype(int).values
    yS = feat["LabelShort"].astype(int).values

    if model_name == "logreg":
        outL, infoL = train_logreg(feat, "long", epochs=epochs, run_dir=run_dir, xcols=xcols)
        outS, infoS = train_logreg(feat, "short", epochs=epochs, run_dir=run_dir, xcols=xcols)
    elif model_name == "torch":
        outL, infoL = train_torch(feat, "long", epochs=max(50, epochs//2), run_dir=run_dir, xcols=xcols)
        outS, infoS = train_torch(feat, "short", epochs=max(50, epochs//2), run_dir=run_dir, xcols=xcols)
    else:
        outL, infoL = train_lgbm(feat, "long", run_dir=run_dir, xcols=xcols)
        outS, infoS = train_lgbm(feat, "short", run_dir=run_dir, xcols=xcols)

    y_prob_L = np.asarray(infoL.get("probs", np.zeros_like(yL)))
    y_pred_L = (y_prob_L >= 0.5).astype(int)
    aucL = save_confusion_and_roc(run_dir, yL, y_pred_L, y_prob_L, "long")

    y_prob_S = np.asarray(infoS.get("probs", np.zeros_like(yS)))
    y_pred_S = (y_prob_S >= 0.5).astype(int)
    aucS = save_confusion_and_roc(run_dir, yS, y_pred_S, y_prob_S, "short")

    plot_results_to(run_dir, outL, ticker, use_pred=True, side="long")
    plot_results_to(run_dir, outS, ticker, use_pred=True, side="short")
    plot_results_to(run_dir, feat,  ticker, use_pred=False, side="long")
    plot_results_to(run_dir, feat,  ticker, use_pred=False, side="short")

    def sse_emit(row): log_emit(row)
    t_PL, eq_PL, s_PL = simulate_trades(outL, side="long",  use_pred=True,  mode=mode, sse_emit=sse_emit)
    t_PS, eq_PS, s_PS = simulate_trades(outS, side="short", use_pred=True,  mode=mode, sse_emit=sse_emit)
    eq_PL.to_csv(run_dir / "equity_pred_long.csv")
    eq_PS.to_csv(run_dir / "equity_pred_short.csv")
    t_PL.to_csv(run_dir / "trades_pred_long.csv", index=False)
    t_PS.to_csv(run_dir / "trades_pred_short.csv", index=False)

    eq_both = pd.concat([eq_PL.rename(columns={"equity":"eq_long"}),
                         eq_PS.rename(columns={"equity":"eq_short"})], axis=1).sort_index().ffill()
    eq_both["equity"] = (eq_both["eq_long"] + eq_both["eq_short"]) / 2.0
    eq_both.to_csv(run_dir / "equity_pred_both.csv")
    for ts_i, row in eq_both.iterrows():
        log_emit({"type":"equity","entry_col":"PredBoth","t":str(ts_i),"equity": float(row["equity"])})

    metrics = {
        "ok": True,
        "ticker": ticker.upper(),
        "run_dir": run_dir.as_posix(),
        "model_long": model_name, "model_short": model_name,
        "n_samples": int(len(feat)),
        "accuracy_long": float((y_pred_L == yL).mean()) if len(yL) else 0.0,
        "roc_auc_long": float(aucL),
        "accuracy_short": float((y_pred_S == yS).mean()) if len(yS) else 0.0,
        "roc_auc_short": float(aucS),
        "class_balance_long": {"NO(0)": int((yL==0).sum()), "YES(1)": int((yL==1).sum())},
        "class_balance_short": {"NO(0)": int((yS==0).sum()), "YES(1)": int((yS==1).sum())},
        "features": xcols,
        "decision_threshold_long": 0.5,
        "decision_threshold_short": 0.5,
        "backtest_pred_long": s_PL,
        "backtest_pred_short": s_PS,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "mode": os.getenv("PINE_MODE", "STD"),
        "timeframe": os.getenv("PINE_TF", "HTF")
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[{ticker}] Saved artifacts -> {run_dir}\n")
    return metrics

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--tickers_json", default=None)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="6mo")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--mode", default="hft", choices=["hft","lft"])
    ap.add_argument("--model", default="logreg", choices=["logreg","torch","lgbm"])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--use_tv", action="store_true")
    ap.add_argument("--use_fundamentals", action="store_true")
    ap.add_argument("--use_ext", action="store_true")
    ap.add_argument("--ext_sources", default="robinhood,webull")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    ap.add_argument("--pool", choices=["thread","process"], default="thread")
    ap.add_argument("--idle", action="store_true")
    ap.add_argument("--idle_interval", type=int, default=900)
    args = ap.parse_args()

    if args.tickers_json:
        p = Path(args.tickers_json).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"--tickers_json not found: {p}")
        try:
            tickers = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(tickers, dict):
                tickers = tickers.get("tickers", [])
            if not isinstance(tickers, list):
                raise ValueError("tickers_json must be a list or an object with 'tickers'")
            tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        except Exception as e:
            raise SystemExit(f"Failed to parse tickers JSON: {e}")
        if not tickers:
            raise SystemExit("tickers_json parsed but empty.")
    else:
        tickers = [str(args.ticker).strip().upper()]

    ext_sources = [s.strip() for s in (args.ext_sources or "").split(",") if s.strip()]
    parent_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    parent_run_dir = RUNS_ROOT / f"run-{parent_ts}"
    parent_run_dir.mkdir(parents=True, exist_ok=True)

    def _onepass():
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        def _submit(executor, tkr: str):
            return executor.submit(
                train_for_ticker,
                tkr,
                interval=args.interval,
                period=args.period,
                start=args.start,
                end=args.end,
                mode=args.mode,
                model_name=args.model,
                epochs=args.epochs,
                use_tv=args.use_tv,
                use_fa=args.use_fundamentals,
                use_ext=args.use_ext,
                ext_sources=ext_sources,
                parent_run_dir=parent_run_dir
            )

        if len(tickers) == 1 or args.workers <= 1:
            for t in tickers:
                try:
                    results.append(train_for_ticker(
                        t, interval=args.interval, period=args.period, start=args.start, end=args.end,
                        mode=args.mode, model_name=args.model, epochs=args.epochs,
                        use_tv=args.use_tv, use_fa=args.use_fundamentals,
                        use_ext=args.use_ext, ext_sources=ext_sources, parent_run_dir=parent_run_dir
                    ))
                except Exception as e:
                    err = {"ticker": t, "error": str(e)}
                    print(f"[{t}] ERROR: {e}")
                    errors.append(err)
        else:
            Exec = ThreadPoolExecutor if args.pool == "thread" else ProcessPoolExecutor
            with Exec(max_workers=args.workers) as ex:
                futs = { _submit(ex, t): t for t in tickers }
                for fut in as_completed(futs):
                    t = futs[fut]
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        err = {"ticker": t, "error": str(e)}
                        print(f"[{t}] ERROR: {e}")
                        errors.append(err)

        summary = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "tickers": tickers,
            "count": len(results),
            "results": results,
            "errors": errors
        }
        (parent_run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSummary saved -> {parent_run_dir/'summary.json'}\n")

    if args.idle:
        while True:
            _onepass()
            print(f"[idle] sleeping {args.idle_interval}s...")
            time.sleep(max(1, int(args.idle_interval)))
    else:
        _onepass()

if __name__ == "__main__":
    main()
