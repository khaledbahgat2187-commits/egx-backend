"""
EGX Signals API
================
A FastAPI backend that pulls EGX stock data from Yahoo Finance,
computes technical signals, and serves them to the dashboard.

Endpoints:
  GET /signals        -> list of signal objects for the dashboard
  GET /signals/{tkr}  -> detail for a single ticker (e.g. COMI)
  GET /health         -> status check

Schema returned to the dashboard (array of these):
  { "stock": "COMI", "score": 82, "entry": 85.3, "stop": 82.1,
    "target1": 88.5, "target2": 91.7, "rsi": 58.2, "change_pct": 1.4 }

Signal scoring is a simple technical composite (0-100) based on:
  - Trend (above SMA20 / SMA50)
  - RSI in bullish zone (45-70)
  - Volume surge vs 20-day average
  - 5-day momentum
  - Breakout near 20-day high

Customize EGX_TICKERS and compute_signal() to fit your strategy.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("egx")

app = FastAPI(title="EGX Signals API", version="1.0.0")

# Allow the dashboard (from anywhere) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------- Configuration ----------
# Add or remove tickers here. Yahoo Finance uses the .CA suffix for EGX stocks.
EGX_TICKERS: List[str] = [
    "COMI.CA",  # Commercial International Bank
    "HRHO.CA",  # EFG Hermes
    "TMGH.CA",  # TMG Holding
    "SWDY.CA",  # Elsewedy Electric
    "ORAS.CA",  # Orascom Construction
    "ABUK.CA",  # Abou Kir Fertilizers
    "ETEL.CA",  # Telecom Egypt
    "ESRS.CA",  # Ezz Steel
    "CIEB.CA",  # Credit Agricole Egypt
    "JUFO.CA",  # Juhayna
    "AMOC.CA",  # Alex Mineral Oils
    "SKPC.CA",  # Sidi Kerir Petrochemicals
    "EFID.CA",  # Edita Food
    "PHDC.CA",  # Palm Hills
    "EKHO.CA",  # Egypt Kuwait Holding
    "MFPC.CA",  # Misr Fertilizers (MOPCO)
    "CCAP.CA",  # Qalaa Holdings
    "PIOH.CA",  # Pioneers Holding
    "ISPH.CA",  # Ibnsina Pharma
    "RMDA.CA",  # Rameda Pharma
    "OIH.CA",   # Orascom Investment
    "EAST.CA",  # Eastern Company
]

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL", "300"))  # 5 min default
_cache: Dict[str, object] = {"ts": 0.0, "data": []}


# ---------- Indicator helpers ----------
def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    prev_close = close.shift()
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def compute_signal(ticker: str, hist: pd.DataFrame) -> Optional[Dict]:
    """Compute a 0-100 score and risk levels from a daily OHLCV DataFrame."""
    if hist is None or len(hist) < 50:
        return None

    # yfinance sometimes returns MultiIndex columns — flatten them
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)
    volume = hist["Volume"].astype(float)

    current = float(close.iloc[-1])
    if current <= 0 or np.isnan(current):
        return None

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    rsi = _rsi(close)
    atr = _atr(high, low, close)
    high_20 = float(high.rolling(20).max().iloc[-1])
    vol_avg20 = float(volume.rolling(20).mean().iloc[-1]) or 1.0
    vol_today = float(volume.iloc[-1])
    momentum_5d = (current / float(close.iloc[-5]) - 1.0) * 100.0
    change_pct = (current / float(close.iloc[-2]) - 1.0) * 100.0

    # ----- Score (0-100) -----
    score = 0
    # Trend
    if current > sma20:
        score += 20
    if current > sma50:
        score += 20
    # RSI zone
    if 45 <= rsi <= 70:
        score += 15
    elif 70 < rsi <= 80:
        score += 8
    elif 35 <= rsi < 45:
        score += 5
    # Volume
    if vol_today > 1.5 * vol_avg20:
        score += 15
    elif vol_today > vol_avg20:
        score += 8
    # Momentum
    if momentum_5d > 5:
        score += 15
    elif momentum_5d > 2:
        score += 10
    elif momentum_5d > 0:
        score += 5
    # Breakout
    if current >= high_20 * 0.99:
        score += 15
    score = max(0, min(100, score))

    # ----- Risk levels from ATR -----
    # Stop ~ 2x ATR below, T1 ~ 2x ATR above (1R), T2 ~ 4x ATR above (2R)
    if atr <= 0 or np.isnan(atr):
        atr = current * 0.02  # fallback: 2% of price
    entry = round(current, 2)
    stop = round(current - 2 * atr, 2)
    target1 = round(current + 2 * atr, 2)
    target2 = round(current + 4 * atr, 2)

    return {
        "stock": ticker.replace(".CA", ""),
        "score": int(score),
        "entry": entry,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "rsi": round(rsi, 1),
        "change_pct": round(change_pct, 2),
        "volume_ratio": round(vol_today / vol_avg20, 2) if vol_avg20 else None,
    }


# ---------- Data fetching ----------
def _fetch_one(ticker: str) -> Optional[Dict]:
    try:
        # 4 months of daily data gives us room for SMA50 + breakout checks
        df = yf.download(
            ticker,
            period="4mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df is None or df.empty:
            log.warning("No data returned for %s", ticker)
            return None
        return compute_signal(ticker, df)
    except Exception as e:
        log.warning("Failed %s: %s", ticker, e)
        return None


def _load_all() -> List[Dict]:
    results = []
    for t in EGX_TICKERS:
        sig = _fetch_one(t)
        if sig:
            results.append(sig)
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---------- Routes ----------
@app.get("/")
def root():
    return {
        "name": "EGX Signals API",
        "endpoints": ["/signals", "/signals/{ticker}", "/health"],
        "tickers": [t.replace(".CA", "") for t in EGX_TICKERS],
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/signals")
def get_signals(force: bool = False):
    """Return scored signals for all configured tickers. Cached for CACHE_TTL seconds."""
    now = time.time()
    if not force and _cache["data"] and (now - float(_cache["ts"])) < CACHE_TTL_SECONDS:
        return _cache["data"]
    data = _load_all()
    _cache["ts"] = now
    _cache["data"] = data
    return data


@app.get("/signals/{ticker}")
def get_signal_detail(ticker: str):
    ticker = ticker.upper().strip()
    if not ticker.endswith(".CA"):
        ticker = ticker + ".CA"
    sig = _fetch_one(ticker)
    if not sig:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")
    return sig


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
