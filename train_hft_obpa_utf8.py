# -*- coding: utf-8 -*-
"""
Learn-from-Pine: Long-Only • Compact HFT+OB/PA (Entries Only)

What it does:
- Download OHLCV with yfinance
- Recreate Pine indicators & entry score/threshold (HFT toggles)
- Produce entry labels (1 = enter long; 0 = no entry)
- Train a simple classifier to imitate entries
- Plot price + learned predictions + original entry points

Run examples:
  python train_hft_obpa.py --ticker SPY --period 1y --interval 1h --plot
  python train_hft_obpa.py --ticker AAPL --start 2023-01-01 --end 2024-01-01 --plot --epochs 200
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pathlib 

src = pathlib.Path(r"C:\users\death\source\repos\tensorflowtest\tensorflowtest\tensorflowtest.py")
data = src.read_bytes()                           # raw bytes, even with weird chars
text = data.decode("latin-1")                     # decode without error
(src.parent / "train_utf8.py").write_text(text, encoding="utf-8")
print("Converted file saved as tensorflowtest_utf8.py")


# -----------------------------
# Utilities: EMA, RMA (Wilder), ATR, RSI, MACD
# -----------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average (Pine: ta.ema)."""
    return series.ewm(span=length, adjust=False).mean()

def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA (Pine: ta.rma)."""
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Average True Range using Wilder's RMA (Pine: ta.atr)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, length)

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing (Pine: ta.rsi)."""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ru = rma(up, length)
    rd = rma(down, length)
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram (Pine: ta.macd)."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

# -----------------------------
# Fetch OHLCV with yfinance
# -----------------------------
def fetch_data(ticker: str, start=None, end=None, period=None, interval="1d") -> pd.DataFrame:
    """Download OHLCV; return DataFrame with columns Open,High,Low,Close,Volume."""
    if period:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    else:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise SystemExit("No data from yfinance. Check ticker / time range / interval.")
    # ensure standard columns and monotonic index
    df = df[['Open','High','Low','Close','Volume']].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df

# -----------------------------
# Recreate Pine's features & signals
# -----------------------------
def build_features_and_labels(
    df: pd.DataFrame,
    *,
    # Inputs (defaults mirror your Pine)
    hftMode: bool = True,
    qtyPercent: float = 10.0,            # unused for learning; Pine sizing
    useDelta: bool = True,
    useOB: bool = True,
    useMAcross: bool = True,
    useMACD: bool = True,
    useRSI: bool = True,
    obLen: int = 10,
    zThrIn: float = 0.5,
    volKIn: float = 1.0,
    atrLenIn: int = 14,
    slMultIn: float = 1.35,              # unused (exits disabled)
    tpMultIn: float = 2.10,              # unused (exits disabled)
    entryThrIn: float = 1.2
):
    """
    Reproduces your Pine logic to generate the entry score and doEnter flag.
    Returns a DataFrame with features + columns:
      fast, slow, macdLine, sigLine, rsi, delta, deltaZ, volSpike, bullEngulf, score, doEnter, Label
    """

    # ===== Effective params under HFT mode =====
    atrLen   = max(7, atrLenIn) if hftMode else atrLenIn
    slMult   = 0.80  if hftMode else slMultIn   # not used for exits here
    tpMult   = 0.95  if hftMode else tpMultIn   # not used for exits here
    zThresh  = 0.50  if hftMode else zThrIn
    volK     = 1.00  if hftMode else volKIn
    entryThr = 1.20  if hftMode else entryThrIn

    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # ===== ATR & safe ATR (for reference)
    atr_val = atr(h, l, c, atrLen)
    atr_safe = np.maximum(atr_val, 1e-12)  # syminfo.mintick proxy

    # ===== Delta and engulf
    upVol   = np.where(c > o, v, 0.0)
    downVol = np.where(c < o, v, 0.0)
    delta   = upVol - downVol

    prev_c, prev_o = c.shift(1), o.shift(1)
    bullEngulf = (c > o) & (prev_c < prev_o) & (c > prev_o) & (o < prev_c)

    deltaMA = pd.Series(delta, index=df.index).rolling(obLen).mean()
    deltaSD = pd.Series(delta, index=df.index).rolling(obLen).std(ddof=0)
    deltaZ  = np.where(deltaSD > 0, (delta - deltaMA) / deltaSD, 0.0)

    volAvg  = v.rolling(obLen).mean()
    volSpike = v > (volAvg * volK)

    deltaLong = (useDelta) & bullEngulf & (delta > 0)
    obLong    = (useOB) & (pd.Series(deltaZ, index=df.index) > zThresh) & volSpike

    # ===== EMA cross
    fast = ema(c, 8)
    slow = ema(c, 21)
    macross = (useMAcross) & (fast > slow) & (fast.shift(1) <= slow.shift(1))

    # ===== MACD cross
    macdLine, sigLine, _ = macd(c, 12, 26, 9)
    macdLong = (useMACD) & (macdLine > sigLine) & (macdLine.shift(1) <= sigLine.shift(1))

    # ===== RSI band
    rsi = rsi_wilder(c, 14)
    rsi_low  = 40 if hftMode else 45
    rsi_high = 75 if hftMode else 65
    rsiLong  = (useRSI) & (rsi > rsi_low) & (rsi < rsi_high)

    # ===== Score
    # score = 0.7*deltaLong + 0.7*obLong + 0.6*maLong + 0.6*macdLong + 0.4*rsiLong
    score = (
        (0.7 * deltaLong.astype(float)) +
        (0.7 * obLong.astype(float)) +
        (0.6 * macross.astype(float)) +
        (0.6 * macdLong.astype(float)) +
        (0.4 * rsiLong.astype(float))
    )
    doEnter = score >= entryThr

    # ===== Build feature frame and labels
    out = df.copy()
    out["fast"]      = fast
    out["slow"]      = slow
    out["macdLine"]  = macdLine
    out["sigLine"]   = sigLine
    out["rsi"]       = rsi
    out["delta"]     = delta
    out["deltaZ"]    = deltaZ
    out["volSpike"]  = volSpike.astype(int)
    out["bullEngulf"]= bullEngulf.astype(int)
    out["score"]     = score
    out["doEnter"]   = doEnter.astype(int)
    out = out.dropna()

    # Training label: 1 if Pine would enter here, else 0
    out["Label"] = out["doEnter"].astype(int)
    return out

# -----------------------------
# Train a simple model to imitate entries
# -----------------------------
def train_model(df_feat: pd.DataFrame, max_iter=500):
    """
    Train logistic regression to classify entry/no-entry based on features.
    """
    # Choose features reflective of your score components
    X_cols = ["fast","slow","macdLine","sigLine","rsi","deltaZ","volSpike","bullEngulf"]
    X = df_feat[X_cols].values
    y = df_feat["Label"].values.astype(int)

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X, y)
    df_feat["Pred"] = model.predict(X)
    acc = (df_feat["Pred"] == y).mean()
    print(f"Train accuracy vs Pine entries: {acc:.3f}  (n={len(df_feat)})")
    return df_feat, model, X_cols

# -----------------------------
# Plot price + Pine entries + model predictions
# -----------------------------
def plot_results(df_feat: pd.DataFrame, ticker: str):
    """
    Price + original Pine entries (green |) + model predictions (triangles).
    """
    close = df_feat["Close"]
    pine_buy_idx = df_feat.index[df_feat["Label"]==1]
    pred_buy_idx = df_feat.index[df_feat["Pred"]==1]

    plt.figure(figsize=(13,6))
    plt.plot(df_feat.index, close, label="Close", linewidth=1.0)

    # Pine entries as vertical markers
    plt.vlines(pine_buy_idx, ymin=close.loc[pine_buy_idx]*0.995, ymax=close.loc[pine_buy_idx]*1.005,
               colors="green", alpha=0.4, label="Pine Entry")

    # Model predicted entries
    plt.scatter(pred_buy_idx, close.loc[pred_buy_idx], marker="^", s=40, color="blue", label="Model Entry")

    plt.title(f"{ticker} — Pine vs Model Entries")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser("Learn Pine entries (HFT+OB/PA)")
    p.add_argument("--ticker", required=True, help="e.g. SPY, AAPL, BTC-USD")
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--period", help="e.g. 1y, 2y, 6mo (use this OR start/end)")
    p.add_argument("--interval", default="1h", help="e.g. 5m, 15m, 1h, 1d")
    p.add_argument("--epochs", type=int, default=500, help="max_iter for LogisticRegression")
    p.add_argument("--plot", action="store_true")
    # optional toggles mirroring Pine inputs
    p.add_argument("--no-hft", action="store_true", help="disable HFT tweaks")
    p.add_argument("--entry-thr", type=float, default=1.2)
    p.add_argument("--ob-len", type=int, default=10)
    p.add_argument("--z-thr", type=float, default=0.5)
    p.add_argument("--vol-k", type=float, default=1.0)
    args = p.parse_args()

    df = fetch_data(args.ticker, start=args.start, end=args.end, period=args.period, interval=args.interval)

    feat = build_features_and_labels(
        df,
        hftMode=(not args.no_hft),
        entryThrIn=args.entry_thr,
        obLen=args.ob_len,
        zThrIn=args.z_thr,
        volKIn=args.vol_k
    )

    feat, model, X_cols = train_model(feat, max_iter=args.epochs)

    if args.plot:
        plot_results(feat, args.ticker)

if __name__ == "__main__":
    main()
