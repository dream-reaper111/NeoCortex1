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
class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

def _ensure_png(path: Path, text="N/A"):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(4,3)); plt.text(0.5,0.5,text,ha="center",va="center"); plt.axis("off")
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    except Exception: pass

def _cm_plot(cm, path: Path, title: str):
    try:
        import seaborn as sns
        plt.figure(figsize=(4,3)); sns.heatmap(cm, annot=True, fmt="d", cbar=False)
        plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
    except Exception:
        plt.figure(figsize=(4,3)); plt.imshow(cm, cmap="Blues")
        plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()

def _roc_plot(y_true, scores, path: Path, title: str):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, scores); A = auc(fpr, tpr)
    plt.figure(figsize=(4,3)); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'k--')
    plt.title(f"{title}\nAUC={A:.3f}"); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
    return float(A)

def _trades_plot(close: pd.Series, sig: pd.Series, path: Path, title: str):
    plt.figure(figsize=(6,3)); plt.plot(close.index, close.values, lw=1.0, label="Close")
    idx = sig[sig==1].index
    if len(idx): plt.scatter(idx, close.loc[idx], s=8)
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()

def train_and_save(feat_df: pd.DataFrame, max_iter: int = 300) -> tuple[float,int]:
    ticker = os.getenv("PINE_TICKER","UNK").upper()
    run_dir = RUNS_ROOT / f"run-{ticker}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Precreate artifacts so UI never 404s
    (run_dir/"nn_graph.json").write_text(json.dumps({
        "ok": True, "model": "MLP", "title": ticker,
        "layers": [{"size": max(2, feat_df.shape[1]-2)}, {"size":64}, {"size":64}, {"size":2}]
    }, indent=2))
    (run_dir/"train.log.jsonl").write_text("")
    (run_dir/"nn_state.jsonl").write_text("")
    _ensure_png(run_dir/"feature_importance_long.png")
    _ensure_png(run_dir/"feature_importance_short.png")

    feat_cols = [c for c in feat_df.columns if c not in ("LabelLong","LabelShort")]
    X = feat_df[feat_cols].values.astype(np.float32)
    yL = feat_df["LabelLong"].astype(int).values
    yS = feat_df["LabelShort"].astype(int).values
    n = len(X)

    if not TORCH_OK:
        # Minimal CPU fallback
        from sklearn.metrics import confusion_matrix, accuracy_score
        predL = np.zeros_like(yL); predS = np.zeros_like(yS)
        cmL = confusion_matrix(yL, predL, labels=[0,1]); cmS = confusion_matrix(yS, predS, labels=[0,1])
        _cm_plot(cmL, run_dir/"cm_long.png", "CM (long)")
        _cm_plot(cmS, run_dir/"cm_short.png", "CM (short)")
        _roc_plot(yL, np.zeros(n), run_dir/"roc_long.png", "ROC (long)")
        _roc_plot(yS, np.zeros(n), run_dir/"roc_short.png", "ROC (short)")
        close = pd.Series(index=feat_df.index, data=100+np.cumsum(feat_df["fast"].fillna(0.0).values))
        _trades_plot(close, pd.Series(yL,index=feat_df.index), run_dir/"trades_label_long.png","Labels (long)")
        _trades_plot(close, pd.Series(predL,index=feat_df.index), run_dir/"trades_pred_long.png","Pred (long)")
        acc = float((yL==0).mean()+ (yS==0).mean())/2.0
        (run_dir/"metrics.json").write_text(json.dumps({
            "ok":True,"model":"MLP-fallback","timeframe":os.getenv("PINE_TF","HFT"),
            "n_samples":n,"accuracy":acc,"accuracy_long":acc,"accuracy_short":acc,
            "roc_auc":0.5,"roc_auc_long":0.5,"roc_auc_short":0.5,
            "class_balance_long":{"NO(0)":int((yL==0).sum()),"YES(1)":int((yL==1).sum())},
            "class_balance_short":{"NO(0)":int((yS==0).sum()),"YES(1)":int((yS==1).sum())},
            "features":feat_cols
        }, indent=2))
        print(f"[trainer] saved -> {run_dir}  acc={acc:.3f}  n={n}  device=cpu", flush=True)
        return acc, n

    X_t = torch.tensor(X, device=DEVICE)
    yL_t = torch.tensor(yL, dtype=torch.long, device=DEVICE)
    yS_t = torch.tensor(yS, dtype=torch.long, device=DEVICE)

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
            xb = Xb[a:b]
            # long
            optL.zero_grad(); outL = modelL(xb); lossL = crit(outL, yLb[a:b]); lossL.backward(); optL.step(); el += float(lossL.item())*(b-a)
            # short
            optS.zero_grad(); outS = modelS(xb); lossS = crit(outS, ySb[a:b]); lossS.backward(); optS.step(); es += float(lossS.item())*(b-a)
        # live NN stats
        with (run_dir/"nn_state.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type":"nn_stats","epoch":epoch,"loss_long":(el/max(1,n)),"loss_short":(es/max(1,n))})+"\n")
        # live equity pulse (for P&L sparkline)
        with (run_dir/"train.log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"phase":"epoch","type":"equity","t":datetime.now(timezone.utc).isoformat(),"equity":epoch,"entry_col":"PredLong"})+"\n")

    modelL.eval(); modelS.eval()
    with torch.no_grad():
        pL = modelL(X_t).softmax(dim=1)[:,1].detach().cpu().numpy()
        pS = modelS(X_t).softmax(dim=1)[:,1].detach().cpu().numpy()
        yhatL = (pL>0.5).astype(int); yhatS = (pS>0.5).astype(int)

    from sklearn.metrics import confusion_matrix, accuracy_score
    cmL = confusion_matrix(yL, yhatL, labels=[0,1]); cmS = confusion_matrix(yS, yhatS, labels=[0,1])
    accL = float(accuracy_score(yL, yhatL)); accS = float(accuracy_score(yS, yhatS))
    aucL = _roc_plot(yL, pL, run_dir/"roc_long.png", "ROC (long)")
    aucS = _roc_plot(yS, pS, run_dir/"roc_short.png", "ROC (short)")
    _cm_plot(cmL, run_dir/"cm_long.png", "CM (long)")
    _cm_plot(cmS, run_dir/"cm_short.png", "CM (short)")

    # trades preview
    close = pd.Series(index=feat_df.index, data=100 + np.cumsum(feat_df["fast"].fillna(0.0).values))
    _trades_plot(close, pd.Series(yL, index=feat_df.index), run_dir/"trades_label_long.png", "Labels (long)")
    _trades_plot(close, pd.Series(yhatL, index=feat_df.index), run_dir/"trades_pred_long.png", "Pred (long)")

    (run_dir/"metrics.json").write_text(json.dumps({
        "ok":True,"model":"MLP","timeframe":os.getenv("PINE_TF","HFT"),
        "n_samples":int(n),"accuracy":round((accL+accS)/2.0,3),
        "accuracy_long":round(accL,3),"accuracy_short":round(accS,3),
        "roc_auc":round((aucL+aucS)/2.0,3),"roc_auc_long":round(aucL,3),"roc_auc_short":round(aucS,3),
        "class_balance_long":{"NO(0)":int((yL==0).sum()),"YES(1)":int((yL==1).sum())},
        "class_balance_short":{"NO(0)":int((yS==0).sum()),"YES(1)":int((yS==1).sum())},
        "features":feat_cols
    }, indent=2))

    print(f"[trainer] saved -> {run_dir}  acc={((accL+accS)/2.0):.3f}  n={n}  device={'cuda' if CUDA else 'cpu'}", flush=True)
    return (accL+accS)/2.0, n
