"""Advanced strategy analytics for the Neo Cortex API.

This module exposes helper utilities that the FastAPI server can call to
identify potential liquidity sweeps during the morning session, infer a
basic order-flow footprint and render a heatmap of intraday volume. The
logic is heuristic and intentionally avoids broker-specific market depth
feeds so that it can run with commonly available OHLCV data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")  # pragma: no cover
import matplotlib.pyplot as plt


NY_TZ = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
AM_SESSION_START = time(9, 30)
AM_SESSION_END = time(12, 0)


class StrategyError(RuntimeError):
    """Raised when strategy computations cannot be produced."""


@dataclass
class SessionWindow:
    ticker: str
    date: date
    start: datetime
    end: datetime


def _standardize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV columns returned by yfinance into a canonical frame."""
    if raw is None or raw.empty:
        raise StrategyError("No data returned for requested window")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(-1, axis=1)
    mapping = {c.lower().replace("_", ""): c for c in df.columns}

    def pick(*names: str) -> str:
        for name in names:
            if name in mapping:
                return mapping[name]
        raise StrategyError(f"Missing OHLCV column: {names[0]}")

    cleaned = pd.DataFrame(
        {
            "Open": pd.to_numeric(df[pick("open")], errors="coerce"),
            "High": pd.to_numeric(df[pick("high")], errors="coerce"),
            "Low": pd.to_numeric(df[pick("low")], errors="coerce"),
            "Close": pd.to_numeric(df[pick("close", "adjclose", "adjustedclose")], errors="coerce"),
            "Volume": pd.to_numeric(df[pick("volume")], errors="coerce"),
        },
        index=pd.to_datetime(df.index, utc=True, errors="coerce"),
    )
    cleaned = cleaned.dropna(how="any").sort_index()
    if cleaned.empty:
        raise StrategyError("No valid OHLCV rows after cleaning")
    return cleaned


def _load_session(
    ticker: str,
    session_date: Optional[date],
    interval: str,
) -> Tuple[pd.DataFrame, SessionWindow]:
    import yfinance as yf

    if session_date is None:
        session_date = datetime.now(tz=NY_TZ).date()
    start_dt = datetime.combine(session_date, time(9, 0), tzinfo=NY_TZ)
    end_dt = datetime.combine(session_date, time(16, 0), tzinfo=NY_TZ)

    try:
        raw = yf.download(
            ticker,
            start=start_dt.astimezone(UTC),
            end=end_dt.astimezone(UTC) + timedelta(minutes=1),
            interval=interval,
            progress=False,
            auto_adjust=True,
            actions=False,
        )
    except Exception as exc:  # pragma: no cover - network/HTTP path
        raise StrategyError(f"Failed to download data: {exc}") from exc

    df = _standardize_ohlcv(raw)
    df.index = df.index.tz_convert(NY_TZ)
    mask = (df.index.time >= AM_SESSION_START) & (df.index.time <= AM_SESSION_END)
    session_df = df.loc[mask]
    if session_df.empty:
        raise StrategyError("No bars within morning session window")

    window = SessionWindow(
        ticker=ticker.upper(),
        date=session_date,
        start=datetime.combine(session_date, AM_SESSION_START, tzinfo=NY_TZ),
        end=datetime.combine(session_date, AM_SESSION_END, tzinfo=NY_TZ),
    )
    return session_df, window


def _compute_footprint(df: pd.DataFrame) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Estimate buy and sell volume imbalance using candle bodies."""
    delta = (df["Close"] - df["Open"]).fillna(0.0)
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    imbalance = np.tanh((delta / rng).fillna(0.0) * 3.0)
    buy_share = (0.5 + imbalance / 2.0).clip(0.0, 1.0)
    buy_vol = (df["Volume"] * buy_share).clip(lower=0.0)
    sell_vol = (df["Volume"] - buy_vol).clip(lower=0.0)
    footprint_rows = []
    for idx in df.index:
        footprint_rows.append(
            {
                "time": idx.isoformat(),
                "buy_volume": float(buy_vol.loc[idx]),
                "sell_volume": float(sell_vol.loc[idx]),
                "delta": float(delta.loc[idx]),
                "imbalance": float(imbalance.loc[idx]),
            }
        )
    totals = {
        "buy_volume": float(buy_vol.sum()),
        "sell_volume": float(sell_vol.sum()),
        "delta": float(delta.sum()),
        "max_delta": float(delta.max()),
        "min_delta": float(delta.min()),
    }
    return footprint_rows, totals


def _detect_sweeps(df: pd.DataFrame) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Flag potential liquidity sweeps based on wick size and abnormal volume."""
    vol_avg = df["Volume"].rolling(30, min_periods=5).mean()
    rng = df["High"] - df["Low"]
    rng_avg = rng.rolling(30, min_periods=5).mean()
    bodies = (df["Close"] - df["Open"]).abs()
    wick_up = df["High"] - df[["Open", "Close"]].max(axis=1)
    wick_down = df[["Open", "Close"]].min(axis=1) - df["Low"]

    sweeps: List[Dict[str, object]] = []
    last_sweep: Optional[Dict[str, object]] = None
    clusters: List[List[Dict[str, object]]] = []

    for ts in df.index:
        vol_ratio = df.loc[ts, "Volume"] / max(vol_avg.loc[ts], 1.0)
        rng_ratio = rng.loc[ts] / max(rng_avg.loc[ts], 1e-6)
        if vol_ratio < 1.6 or rng_ratio < 1.3:
            continue
        wick_up_v = wick_up.loc[ts]
        wick_down_v = wick_down.loc[ts]
        direction = "bearish" if wick_up_v > wick_down_v else "bullish"
        wick = max(wick_up_v, wick_down_v)
        if wick < rng.loc[ts] * 0.45:
            continue
        sweep = {
            "time": ts.isoformat(),
            "volume_ratio": float(vol_ratio),
            "range_ratio": float(rng_ratio),
            "direction": direction,
            "wick": float(wick),
            "body": float(bodies.loc[ts]),
        }
        sweeps.append(sweep)
        if last_sweep:
            last_ts = datetime.fromisoformat(last_sweep["time"])
            if (
                sweep["direction"] != last_sweep["direction"]
                and ts - last_ts <= timedelta(minutes=20)
            ):
                if clusters and clusters[-1][-1] is last_sweep:
                    clusters[-1].append(sweep)
                else:
                    clusters.append([last_sweep, sweep])
        last_sweep = sweep

    summaries: List[Dict[str, object]] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        summaries.append(
            {
                "start": cluster[0]["time"],
                "end": cluster[-1]["time"],
                "count": len(cluster),
                "directions": [c["direction"] for c in cluster],
            }
        )
    return sweeps, summaries


def _render_heatmap(df: pd.DataFrame, out_path: Path) -> Optional[str]:
    if df.empty:
        return None
    prices = (df["High"] + df["Low"]) / 2.0
    bins = np.linspace(prices.min(), prices.max(), min(40, max(10, len(df) // 2)))
    df = df.copy()
    df["price_bin"] = pd.cut(prices, bins=bins, labels=False, include_lowest=True)
    df["time_bin"] = df.index.strftime("%H:%M")
    pivot = df.pivot_table(
        index="price_bin",
        columns="time_bin",
        values="Volume",
        aggfunc="sum",
        fill_value=0.0,
    )
    if pivot.empty:
        return None
    plt.figure(figsize=(12, 5))
    plt.imshow(pivot, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="Volume")
    plt.title("Morning Liquidity Heatmap (volume per price bin)")
    plt.ylabel("Price bin")
    plt.xlabel("Time (ET)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path.name


def analyze_liquidity_session(
    ticker: str,
    session_date: Optional[str] = None,
    interval: str = "1m",
    asset_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Return liquidity sweep analytics for a ticker and session."""
    if not ticker:
        raise StrategyError("Ticker is required")
    parsed_date: Optional[date] = None
    if session_date:
        try:
            parsed_date = datetime.fromisoformat(session_date).date()
        except ValueError as exc:
            raise StrategyError("session_date must be ISO formatted YYYY-MM-DD") from exc

    df, window = _load_session(ticker, parsed_date, interval)
    sweeps, clusters = _detect_sweeps(df)
    footprint_rows, footprint_totals = _compute_footprint(df)
    heatmap_name = None
    heatmap_url = None
    if asset_dir is not None:
        asset_dir.mkdir(parents=True, exist_ok=True)
        file_stub = f"{window.ticker}-{window.date.isoformat()}-{interval}-heatmap.png"
        heatmap_path = asset_dir / file_stub
        rendered = _render_heatmap(df, heatmap_path)
        if rendered:
            heatmap_name = rendered
            heatmap_url = heatmap_path.name

    summary: Dict[str, object] = {
        "ticker": window.ticker,
        "session": {
            "date": window.date.isoformat(),
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "interval": interval,
            "bars": len(df),
        },
        "sweeps": sweeps,
        "manipulation_clusters": clusters,
        "footprint": {
            "rows": footprint_rows,
            "totals": footprint_totals,
        },
        "orderflow": {
            "volume": float(df["Volume"].sum()),
            "average_volume": float(df["Volume"].mean()),
            "average_range": float((df["High"] - df["Low"]).mean()),
        },
    }
    if heatmap_url:
        summary["heatmap"] = {
            "filename": heatmap_name,
            "relative_url": heatmap_url,
        }
    return summary
