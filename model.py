# -*- coding: utf-8 -*-
# model.py — GPU-aware trainer with live logs/NN, exogenous inputs & “pine” signals

from __future__ import annotations
import os, json, math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Torch / GPU ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
    CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA else "cpu")
except Exception as e:
    # When PyTorch cannot be imported (e.g. missing DLLs on Windows), fail fast with a
    # clear error message. Without a valid torch import the `nn` symbol used below
    # will be undefined, leading to confusing NameError exceptions. Raising here
    # ensures users install the correct CPU-only or CUDA-enabled torch build before
    # running the application.
    raise ImportError(
        "Unable to import PyTorch (torch). Please install a compatible CPU-only "
        "version of torch for your system. Original error: "
        f"{e}"
    ) from e

RUNS_ROOT = Path(os.getenv("RUNS_ROOT", "artifacts")).resolve()
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

def latest_run_path() -> str | None:
    if not RUNS_ROOT.exists(): return None
    runs = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
    if not runs: return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].as_posix()

# ----------------- feature helpers -----------------
def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = -d.clip(upper=0).rolling(n).mean().replace(0, np.nan)
    rs = up / dn
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def _zscore(s: pd.Series, n: int = 50) -> pd.Series:
    m = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0).replace(0, np.nan)
    return ((s - m) / sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)

def build_features(
    ohlcv: pd.DataFrame,
    hftMode: bool = True,
    zThrIn: float | None = None,
    volKIn: float | None = None,
    entryThrIn: float | None = None,
    exog: pd.DataFrame | None = None
) -> pd.DataFrame:
    df = ohlcv.copy().sort_index()
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"OHLCV required; missing columns: {missing_list}")

    fast = df["Close"].pct_change().fillna(0.0)
    slow = df["Close"].rolling(10).mean().pct_change().fillna(0.0)
    macd = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    macline = macd.ewm(span=9).mean()
    rsi = _rsi(df["Close"], 14)
    deltaz = _zscore(fast, 50)
    volspike = (df["Volume"] / (df["Volume"].rolling(30).mean().replace(0, np.nan))).fillna(1.0)

    out = pd.DataFrame({
        "fast": fast, "slow": slow, "macd": macd, "macline": macline,
        "rsi": rsi, "deltaz": deltaz, "volspike": volspike
    }, index=df.index)

    # include exogenous (TradingView/robinhood/webull signals, fundamentals, etc.)
    if exog is not None and len(exog):
        ex = exog.copy()
        for c in ex.columns:
            out[c] = pd.to_numeric(ex[c], errors="coerce").fillna(method="ffill").fillna(0.0)

    # Map “pine modes” to micro-tweaks in thresholds (optional)
    pine_mode = (os.getenv("PINE_MODE") or "").upper()
    tweak = {"ULTRA": 0.50, "SCALPER": 0.65, "SWING": 1.05}.get(pine_mode, 0.95)
    thr = (entryThrIn if entryThrIn is not None else tweak) - 1.0  # map UI ~0.5..1.2 → drift around 0

    fwd = df["Close"].pct_change().shift(-1).fillna(0.0)
    out["LabelLong"]  = (fwd >  thr).astype(int)
    out["LabelShort"] = (fwd < -thr).astype(int)

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

# ----------------- model -----------------
clas
    modelL = MLP(X_t.shape[1]).to(DEVICE)
    modelS = MLP(X_t.shape[1]).to(DEVICE)
    optL, optS = optim.Adam(modelL.parameters(), 1e-3), optim.Adam(modelS.parameters(), 1e-3)
    crit = nn.CrossEntropyLoss()

    B = 256; steps = math.ceil(n/B) if n else 1
    for epoch in range(1, int(max_iter)+1):
        modelL.train(); modelS.train()
        perm = torch.randperm(n, device=DEVICE) if n else torch.arange(0,0,device=DEVICE)
        Xb, yLb, ySb = X_t[perm], yL_t[perm], yS_t[perm]
        el, es = 0.0, 0.0
        for i in range(steps):
            a,b = i*B, min((i+1)*B, n)
            xb

    print(f"[trainer] saved -> {run_dir}  acc={((accL+accS)/2.0):.3f}  n={n}  device={'cuda' if CUDA else 'cpu'}", flush=True)
    return (accL+accS)/2.0, n
