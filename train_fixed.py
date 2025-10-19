import os, time, json, argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import cupy as cp
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import Literal

# =========================
# Config
# =========================
MODEL_PATH = "cupy_model.npz"
SEQ = 64
FEATURES = ["ret1","ret5","sma_gap","rsi14","vol_z","bb_pos"]
THRESH = 0.001         # future-return threshold for label=UP
LR = 1e-2
EPOCHS = 5
BATCH = 512
USE_MLP = False
DEVICE = 0
CONF_BUY = 0.55
CONF_SELL = 0.45

# =========================
# Pydantic payloads
# =========================
class Bar(BaseModel):
    time: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: float

class BarsPayload(BaseModel):
    symbol: str
    tf: str = Field(default="1")
    bars: List[Bar]

class SignalPayload(BaseModel):
    symbol: str
    tf: str = Field(default="1")
    t: Optional[str] = None
    side: Literal["BUY", "SELL", "FLAT"]
    strength: Optional[float] = None
    comment: Optional[str] = None
    source: Optional[str] = None

class TrainRequest(BaseModel):
    epochs: int = EPOCHS
    lr: float = LR
    use_mlp: Optional[bool] = None

# =========================
# GPU helpers / features
# =========================
def with_device(dev=DEVICE):
    return cp.cuda.Device(dev)

def cp_nan_to_num(a, val=0.0):
    a = a.copy()
    a[cp.isnan(a)] = val
    a[cp.isinf(a)] = val
    return a

def cp_rsi(x: cp.ndarray, period: int = 14) -> cp.ndarray:
    dx = cp.diff(x)
    up  = cp.maximum(dx, 0.0)
    down= cp.maximum(-dx, 0.0)
    alpha = 1.0 / period
    def ema(v):
        out = cp.empty_like(v)
        s = 0.0
        for i in range(v.size):
            s = alpha*v[i] + (1-alpha)*s
            out[i] = s
        return out
    ru = ema(up)
    rd = ema(down)
    rs = ru / (rd + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return cp.concatenate([cp.array([cp.nan]), rsi])

def cp_bbands(x: cp.ndarray, period:int=20, k:float=2.0):
    n = x.size
    ma = cp.empty(n); ma[:] = cp.nan
    sd = cp.empty(n); sd[:] = cp.nan
    for i in range(period-1, n):
        w = x[i-period+1:i+1]
        m = cp.mean(w)
        s = cp.std(w)
        ma[i] = m; sd[i] = s
    upper = ma + k*sd
    lower = ma - k*sd
    return ma, upper, lower

def make_features_from_np(df_np: np.ndarray) -> np.ndarray:
    with with_device():
        o = cp.asarray(df_np[:,0])
        h = cp.asarray(df_np[:,1])
        l = cp.asarray(df_np[:,2])
        c = cp.asarray(df_np[:,3])
        v = cp.asarray(df_np[:,4])

        ret1 = cp.nan_to_num(c/cp.concatenate([c[:1], c[:-1]]) - 1.0)
        ret5 = cp.nan_to_num(c/cp.concatenate([c[:5], c[:-5]]) - 1.0)

        def sma(x, n):
            out = cp.empty_like(x); out[:] = cp.nan
            for i in range(n-1, x.size):
                out[i] = cp.mean(x[i-n+1:i+1])
            return out
        sma5  = sma(c,5)
        sma20 = sma(c,20)
        sma_gap = (sma5 - sma20) / (sma20 + 1e-9)

        rsi14 = cp_rsi(c, 14)

        mv = sma(v, 50)
        sv = cp.empty_like(v); sv[:] = cp.nan
        for i in range(50-1, v.size):
            w = v[i-50+1:i+1]
            sv[i] = cp.std(w)
        vol_z = (v - mv) / (sv + 1e-9)

        ma, up, lo = cp_bbands(c, 20, 2.0)
        bb_pos = (c - lo) / (up - lo + 1e-9)

        feats = cp.stack([ret1, ret5, sma_gap, rsi14, vol_z, bb_pos], axis=1).astype(cp.float32)
        feats = cp_nan_to_num(feats)
        return cp.asnumpy(feats)

# =========================
# Model
# =========================
@dataclass
class ModelState:
    W: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    mlp_W1: Optional[np.ndarray] = None
    mlp_b1: Optional[np.ndarray] = None
    mlp_W2: Optional[np.ndarray] = None
    mlp_b2: Optional[np.ndarray] = None
    seq: int = SEQ
    features: List[str] = None

    def save(self, path=MODEL_PATH):
        np.savez(path, **{k: v for k,v in asdict(self).items() if v is not None})

    @staticmethod
    def load(path=MODEL_PATH):
        if not os.path.exists(path):
            return None
        d = np.load(path, allow_pickle=True)
        ms = ModelState(seq=int(d["seq"]), features=list(d["features"]))
        for k in d.files:
            if k in ("seq","features"): continue
            setattr(ms, k, d[k])
        return ms

MODEL = ModelState(seq=SEQ, features=FEATURES)

def softmax(x: cp.ndarray) -> cp.ndarray:
    x = x - cp.max(x, axis=1, keepdims=True)
    e = cp.exp(x)
    return e / (cp.sum(e, axis=1, keepdims=True) + 1e-9)

def train_logreg(X: np.ndarray, y: np.ndarray, lr=LR, epochs=EPOCHS, batch=BATCH):
    n, d = X.shape
    with with_device():
        Xc = cp.asarray(X, dtype=cp.float32)
        yc = cp.asarray(y, dtype=cp.int32)
        W = cp.zeros((d,2), dtype=cp.float32) if MODEL.W is None else cp.asarray(MODEL.W)
        b = cp.zeros((2,), dtype=cp.float32)  if MODEL.b is None else cp.asarray(MODEL.b)
        idx = cp.arange(n)
        for ep in range(1, epochs+1):
            cp.random.shuffle(idx)
            for i in range(0, n, batch):
                bi = idx[i:i+batch]
                xb = Xc[bi]; yb = yc[bi]
                logits = xb @ W + b
                probs = softmax(logits)
                grad = probs
                grad[cp.arange(grad.shape[0]), yb] -= 1.0
                grad /= grad.shape[0]
                gW = xb.T @ grad
                gb = cp.sum(grad, axis=0)
                W -= lr * gW; b -= lr * gb
            pred = cp.argmax(Xc @ W + b, axis=1)
            acc = float(cp.mean((pred == yc).astype(cp.float32)))
            print(f"[logreg] epoch {ep}/{epochs} acc={acc:.3f}")
        MODEL.W, MODEL.b = cp.asnumpy(W), cp.asnumpy(b)

def train_mlp(X: np.ndarray, y: np.ndarray, lr=LR, epochs=EPOCHS, batch=BATCH, hid=64):
    n, d = X.shape
    with with_device():
        Xc = cp.asarray(X, dtype=cp.float32)
        yc = cp.asarray(y, dtype=cp.int32)
        W1 = cp.random.randn(d, hid).astype(cp.float32)*0.05 if MODEL.mlp_W1 is None else cp.asarray(MODEL.mlp_W1)
        b1 = cp.zeros((hid,), dtype=cp.float32)        if MODEL.mlp_b1 is None else cp.asarray(MODEL.mlp_b1)
        W2 = cp.random.randn(hid, 2).astype(cp.float32)*0.05 if MODEL.mlp_W2 is None else cp.asarray(MODEL.mlp_W2)
        b2 = cp.zeros((2,), dtype=cp.float32)          if MODEL.mlp_b2 is None else cp.asarray(MODEL.mlp_b2)
        idx = cp.arange(n)
        for ep in range(1, epochs+1):
            cp.random.shuffle(idx)
            for i in range(0, n, batch):
                bi = idx[i:i+batch]
                xb = Xc[bi]; yb = yc[bi]
                h1 = cp.maximum(0, xb @ W1 + b1)
                logits = h1 @ W2 + b2
                probs = softmax(logits)
                grad = probs
                grad[cp.arange(grad.shape[0]), yb] -= 1.0
                grad /= grad.shape[0]
                gW2 = h1.T @ grad; gb2 = cp.sum(grad, axis=0)
                dh1 = grad @ W2.T; dh1[h1<=0]=0.0
                gW1 = xb.T @ dh1;  gb1 = cp.sum(dh1, axis=0)
                W2 -= lr*gW2; b2 -= lr*gb2
                W1 -= lr*gW1; b1 -= lr*gb1
            pred = cp.argmax(cp.maximum(0, Xc @ W1 + b1) @ W2 + b2, axis=1)
            acc = float(cp.mean((pred == yc).astype(cp.float32)))
            print(f"[mlp] epoch {ep}/{epochs} acc={acc:.3f}")
        MODEL.mlp_W1, MODEL.mlp_b1 = cp.asnumpy(W1), cp.asnumpy(b1)
        MODEL.mlp_W2, MODEL.mlp_b2 = cp.asnumpy(W2), cp.asnumpy(b2)

def infer_proba(x_feat_last_seq: np.ndarray) -> float:
    x = x_feat_last_seq.astype(np.float32).mean(axis=0, keepdims=True)
    with with_device():
        xc = cp.asarray(x)
        if USE_MLP and MODEL.mlp_W1 is not None:
            h1 = cp.maximum(0, xc @ cp.asarray(MODEL.mlp_W1) + cp.asarray(MODEL.mlp_b1))
            logits = h1 @ cp.asarray(MODEL.mlp_W2) + cp.asarray(MODEL.mlp_b2)
        else:
            logits = xc @ cp.asarray(MODEL.W) + cp.asarray(MODEL.b)
        p = softmax(logits)[0,1]
        return float(p.get())

# =========================
# Buffers + dataset (for server mode)
# =========================
class SymBuf:
    def __init__(self):
        self.df: Optional[np.ndarray] = None
        self.X: List[np.ndarray] = []
        self.y: List[int] = []

BUFFERS: Dict[str, SymBuf] = {}
def key(sym: str, tf: str) -> str: return f"{sym}|{tf}"

def push_bars(sym: str, tf: str, bars: List[Bar]):
    k = key(sym, tf)
    if k not in BUFFERS: BUFFERS[k] = SymBuf()
    df = np.array([[b.open,b.high,b.low,b.close,b.volume] for b in bars], dtype=np.float64)
    BUFFERS[k].df = df if BUFFERS[k].df is None else np.vstack([BUFFERS[k].df, df])
    BUFFERS[k].df = BUFFERS[k].df[-max(SEQ+200, 2000):]

def label_from_side(side: str) -> int:
    return 1 if side=="BUY" else (0 if side=="SELL" else -1)

def add_pine_label(sym: str, tf: str, side: str):
    k = key(sym, tf)
    if k not in BUFFERS or BUFFERS[k].df is None or BUFFERS[k].df.shape[0] < SEQ: return
    lbl = label_from_side(side)
    if lbl < 0: return
    feats = make_features_from_np(BUFFERS[k].df)
    Xseq = feats[-SEQ:, :]
    BUFFERS[k].X.append(Xseq.astype(np.float32))
    BUFFERS[k].y.append(int(lbl))

def build_dataset_from_buffer():
    Xs, ys = [], []
    for buf in BUFFERS.values():
        if buf.X:
            Xs.extend(buf.X); ys.extend(buf.y)
    if not Xs: return None, None
    X = np.stack([x.mean(axis=0) for x in Xs], axis=0)
    y = np.array(ys, dtype=np.int32)
    return X, y

# =========================
# Server
# =========================
app = FastAPI(title="CuPy DayTrader", version="1.1")

@app.get("/")
def root():
    return {
        "ok": True,
        "gpu_count": int(cp.cuda.runtime.getDeviceCount()),
        "device": int(DEVICE),
        "seq": int(SEQ),
        "features": FEATURES,
        "have_model": bool((MODEL.W is not None and MODEL.b is not None) or MODEL.mlp_W1 is not None),
    }

@app.post("/bars")
def ingest_bars(p: BarsPayload):
    push_bars(p.symbol, p.tf, p.bars)
    return {"ok": True, "rows": int(BUFFERS[key(p.symbol,p.tf)].df.shape[0])}

@app.post("/signal")
def ingest_signal(p: SignalPayload):
    add_pine_label(p.symbol, p.tf, p.side)
    return {"ok": True, "labeled": True, "side": p.side}

@app.post("/train")
def train(req: TrainRequest):
    global USE_MLP
    if req.use_mlp is not None: USE_MLP = bool(req.use_mlp)
    X, y = build_dataset_from_buffer()
    if X is None: return {"ok": False, "msg": "no data yet"}
    nfeat = X.shape[1]
    if MODEL.W is None and not USE_MLP:
        MODEL.W = np.zeros((nfeat,2), dtype=np.float32)
        MODEL.b = np.zeros((2,), dtype=np.float32)
    if USE_MLP and MODEL.mlp_W1 is None:
        pass
    if USE_MLP: train_mlp(X,y, lr=req.lr, epochs=req.epochs)
    else:       train_logreg(X,y, lr=req.lr, epochs=req.epochs)
    MODEL.save(MODEL_PATH)
    return {"ok": True, "saved": MODEL_PATH, "n_samples": int(X.shape[0]), "use_mlp": USE_MLP}

@app.post("/infer")
def infer(p: BarsPayload):
    push_bars(p.symbol, p.tf, p.bars)
    k = key(p.symbol, p.tf); df = BUFFERS[k].df
    if df is None or df.shape[0] < SEQ:
        return {"side":"FLAT","reason":"not_enough_bars","have": int(0 if df is None else df.shape[0]),"need": int(SEQ)}
    feats = make_features_from_np(df)
    xseq = feats[-SEQ:, :]
    if (MODEL.W is None and MODEL.mlp_W1 is None):
        return {"side":"FLAT","reason":"no_model"}
    prob_up = infer_proba(xseq)
    side = "BUY" if prob_up > CONF_BUY else ("SELL" if prob_up < CONF_SELL else "FLAT")
    c = float(df[-1,3])
    tp = c*(1.004 if side=="BUY" else 0.996) if side!="FLAT" else None
    sl = c*(0.996 if side=="BUY" else 1.004) if side!="FLAT" else None
    return {"symbol":p.symbol,"tf":p.tf,"price":c,"side":side,"prob_up":round(prob_up,4),"tp":tp,"sl":sl}

# =========================
# Offline: yfinance + training + plotting
# =========================
def yf_fetch(symbol: str, period="1y", interval="1h") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df = df.dropna()
    if not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
        raise RuntimeError("Missing OHLCV columns from yfinance")
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    return df[["open","high","low","close","volume"]]

def label_by_future_return(df: pd.DataFrame, horizon=3, thresh=THRESH) -> np.ndarray:
    c = df["close"].values
    fut = np.roll(c, -horizon) / c - 1.0
    fut[-horizon:] = np.nan
    y = (fut > thresh).astype(np.float32)
    y = np.where(np.isnan(fut), np.nan, y)
    return y

def build_Xy_from_df(df: pd.DataFrame, horizon=3, thresh=THRESH):
    arr = df.values.astype(np.float64)
    feats = make_features_from_np(arr)
    y = label_by_future_return(df, horizon=horizon, thresh=thresh)
    # build sequences
    Xs, ys = [], []
    for i in range(SEQ-1, len(df)-horizon):
        Xs.append(feats[i-SEQ+1:i+1,:].astype(np.float32))
        ys.append(int(y[i]))
    X = np.stack([x.mean(axis=0) for x in Xs], axis=0)  # pooled; change to seq model later
    y = np.array(ys, dtype=np.int32)
    return X, y, feats

def plot_quick(df: pd.DataFrame, feats: np.ndarray, symbol: str):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.plot(df.index, df["close"].values)
    ax.set_title(f"{symbol} Close")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# =========================
# Main / CLI
# =========================
def main():
    ap = argparse.ArgumentParser("CuPy Trainer")
    ap.add_argument("--fetch", type=str, help="Ticker (e.g., SPY) to download with yfinance")
    ap.add_argument("--period", type=str, default="1y", help="yfinance period (e.g., 2y, 6mo)")
    ap.add_argument("--interval", type=str, default="1h", help="yfinance interval (e.g., 5m, 15m, 1h)")
    ap.add_argument("--train-offline", action="store_true", help="Train on downloaded data using future-return labels")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--mlp", action="store_true", help="Use tiny MLP instead of logistic regression")
    ap.add_argument("--plot", action="store_true", help="Show a quick price plot")
    ap.add_argument("--serve", action="store_true", help="Run FastAPI server on :8001")
    args = ap.parse_args()

    global USE_MLP
    USE_MLP = bool(args.mlp)

    if args.fetch:
        print(f"[yfinance] downloading {args.fetch} period={args.period} interval={args.interval} ...")
        df = yf_fetch(args.fetch, args.period, args.interval)
        df.to_csv(f"{args.fetch}_{args.period}_{args.interval}.csv")
        print(f"Saved {args.fetch}_{args.period}_{args.interval}.csv  rows={len(df)}")

        X = y = feats = None
        if args.train_offline:
            print("[offline] building dataset…")
            X, y, feats = build_Xy_from_df(df)
            print(f"[offline] samples={X.shape[0]} nfeat={X.shape[1]}")
            if MODEL.W is None and not USE_MLP:
                MODEL.W = np.zeros((X.shape[1],2), dtype=np.float32)
                MODEL.b = np.zeros((2,), dtype=np.float32)
            if USE_MLP:
                train_mlp(X,y, lr=args.lr, epochs=args.epochs)
            else:
                train_logreg(X,y, lr=args.lr, epochs=args.epochs)
            MODEL.save(MODEL_PATH)
            print(f"[offline] saved model to {MODEL_PATH}")

        if args.plot:
            if feats is None:
                feats = make_features_from_np(df.values.astype(np.float64))
            plot_quick(df, feats, args.fetch)

    if args.serve:
        import uvicorn
        with with_device(DEVICE):
            pass
        uvicorn.run(app, host="0.0.0.0", port=8001)

    if not (args.fetch or args.serve):
        print(__doc__)

if __name__ == "__main__":
    main()
