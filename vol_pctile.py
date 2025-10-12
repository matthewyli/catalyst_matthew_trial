import argparse
import json
import math
import os
import time
from datetime import datetime, timezone, timedelta
from statistics import median
from typing import Optional, Sequence, Iterable, Any, List

import numpy as np
import pandas as pd
import requests


MOBULA_URL = "https://api.mobula.io/api/1/market/history"
MOBULA_ASSET_MAP: dict[str, str] = {
    "ETH": "ethereum",
    "BTC": "bitcoin",
    "SOL": "solana",
    "BNB": "binancecoin",
    "MATIC": "polygon",
    "AVAX": "avalanche",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "XRP": "ripple",
}
# Mobula likes to reshuffle response wrappers; i just put the ones i consistently found.
MOBULA_PRICE_KEYS: Sequence[str] = (
    "price_history",
    "data",
    "history",
    "prices",
    "items",
    "results",
    "values",
    "close",
    "candles",
)

def _calc_fetch_window(days: int) -> tuple[int, int]:
    """Return (start_ts, end_ts) in unix seconds with a little cushion on both ends"""
    end_ts = int(time.time())
    padding = int(timedelta(days=5).total_seconds())
    start_ts = end_ts - int(timedelta(days=days).total_seconds()) - padding
    return start_ts, end_ts

def _unwrap_price_rows(payload: Any) -> Iterable[Any]:
    """Unpack the various Mobula payload envelopes until we hit raw rows"""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in MOBULA_PRICE_KEYS:
            inner = payload.get(key)
            rows = _unwrap_price_rows(inner)
            if rows:
                return rows
        # fall back to a single dict row if nothing matched
        return [payload]
    return []

def _normalize_price_point(row: Any) -> Optional[tuple[datetime, float]]:
    """Coerce a Mobula row/tuple into (timestamp, price) if possible"""
    ts = None
    px = None

    getter = getattr(row, "get", None)
    if callable(getter): # in case of different labels (not sure how stable it is?)
        ts = getter("timestamp") or getter("timestamp_ms") or getter("time") or getter("date")
        px = getter("close") or getter("price") or getter("close_price") or getter("closePrice") or getter("value") or getter("valueUSD") or getter("priceUSD")
    elif isinstance(row, (list, tuple)) and len(row) >= 2:
        ts, px = row[0], row[1]

    if isinstance(px, dict):
        px = px.get("usd") or px.get("USD")
    if ts is None or px is None:
        return None

    try:
        if isinstance(ts, (int, float)):
            if ts > 10**12:
                dt = pd.to_datetime(ts, unit="ms", utc=True)
            elif ts > 10**10:
                dt = pd.to_datetime(ts, unit="s", utc=True)
            else:
                dt = pd.to_datetime(ts, utc=True)
        else:
            dt = pd.to_datetime(ts, utc=True)

        val = float(px)
        if val > 10_000_000:
            val /= 1_000_000
        elif val > 10_000:
            val /= 1_000
        return dt, val
    except Exception:
        return None

def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def annualize_sigma(sig_samp: pd.Series, freq: str) -> pd.Series:
    if freq == "d":
        return sig_samp * math.sqrt(252.0)
    if freq == "h":
        return sig_samp * math.sqrt(24.0 * 252.0)
    raise ValueError("freq must be 'd' or 'h'")

def fetch_prices_mobula(symbol: str, freq: str, days: int, api_key: str, blockchain: Optional[str] = None) -> pd.DataFrame:
    assert freq in ("d", "h")
    interval = "1d" if freq == "d" else "1h"
    start_ts, end_ts = _calc_fetch_window(days)
    sym_key = symbol.upper()
    asset_slug = MOBULA_ASSET_MAP.get(sym_key, symbol.lower())

    params: dict[str, object] = {
        "asset": asset_slug,
        "symbol": sym_key,
        "interval": interval,
        "from": start_ts * 1000,
        "to": end_ts * 1000,
        "currency": "usd",
    }
    if blockchain:
        params["blockchain"] = blockchain
    if api_key:
        params["api_key"] = api_key
        params["token"] = api_key

    headers: dict[str, str] = {}
    if api_key:
        # Mobula is happy with either header, so just send both lol
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key
    r = requests.get(MOBULA_URL, headers=headers, params=params, timeout=30)

    if r.status_code == 401:
        raise ValueError("mobula_auth_failed: supply --mobula-api-key or MOBULA_API_KEY env var")
    if r.status_code >= 400:
        try:
            err_body = r.json()
        except ValueError:
            err_body = {}
        if isinstance(err_body, dict):
            msg = err_body.get("message") or err_body.get("error") or err_body
        else:
            msg = err_body
        raise ValueError(f"mobula_error:{r.status_code}:{msg}")
    data = r.json()

    rows = []
    for row in _unwrap_price_rows(data):
        normalized = _normalize_price_point(row)
        if normalized:
            rows.append(normalized)

    if not rows:
        preview = str(data)[:200]
        raise ValueError(f"no_price_data:raw={preview}")

    df = pd.DataFrame(rows, columns=["ts", "close"]).dropna().sort_values("ts").set_index("ts")
    # Resample to a tidy series so the rolling math works properly
    if freq == "d":
        df = df[~df.index.duplicated(keep="last")].resample("1D").last().ffill()
    else:
        df = df[~df.index.duplicated(keep="last")].resample("1H").last().ffill()
    return df

def realized_vol_percentile(close: pd.Series, w: int, lb_days: int, freq: str) -> tuple[float, float]:
    # log returns keep things nice and additive
    r = np.log(close).diff().dropna()
    # Guard against impossible windows before we roll
    if w < 2:
        raise ValueError("window must be >=2")
    sig = r.rolling(w).std(ddof=1).dropna()
    sig_ann = annualize_sigma(sig, freq)
    if len(sig_ann) < lb_days + 2:
        # use whatever we have and exclude the live point if not enough for lookback
        hist = sig_ann.iloc[:-1]
    else:
        hist = sig_ann.iloc[-(lb_days+1):-1]
    if hist.empty:
        raise ValueError("insufficient_history")
    current = float(sig_ann.iloc[-1])
    pctile = 100.0 * float((hist <= current).mean())
    return current, pctile

def blend_percentiles(results: List[dict]) -> Optional[float]:
    """Blend percentiles with recency-heavy weights so fresh moves matter more."""
    scored = [(r["w"], r["pctile"]) for r in results if "pctile" in r and not math.isnan(r["pctile"])]
    if not scored:
        return None
    scored = sorted(scored, key=lambda x: x[0])  # oldest window first
    raw_weights = [math.exp(-w / 45.0) for w, _ in scored]
    total = sum(raw_weights)
    if total == 0:
        return float(median(p for _, p in scored))
    weighted = sum(weight * pct for (_, pct), weight in zip(scored, raw_weights))
    return float(weighted / total)

def main():
    p = argparse.ArgumentParser(description="Volatility regime percentile tool")
    p.add_argument("--asset", required=True, help="symbol, e.g., ETH or BTC")
    default_windows = "1,3,7,14,21,30,45,60,90"
    p.add_argument("--windows", default=default_windows, help=f"comma-separated windows in days (default: {default_windows})")
    p.add_argument("--blockchain", default=None, help="Mobula blockchain identifier, e.g., ethereum")
    p.add_argument("--lookback-days", type=int, default=365, help="percentile lookback horizon in days")
    p.add_argument("--freq", choices=["d","h"], default="d", help="price interval")
    p.add_argument("--mobula-api-key", default=os.getenv("MOBULA_API_KEY", ""), help="Mobula API key")
    p.add_argument("--pretty", action="store_true", help="pretty-print JSON when using --json-output")
    p.add_argument("--json-output", action="store_true", help="emit full JSON payload instead of a single score")
    args = p.parse_args()

    try:
        windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
        if not windows:
            raise ValueError("windows cannot be empty")
        max_w = max(windows)
        # grab enough history for the widest window plus a little cushion.
        fetch_days = args.lookback_days + max(5, max_w + 5)
        df = fetch_prices_mobula(args.asset, args.freq, fetch_days, args.mobula_api_key, args.blockchain)
        res = []
        for w in windows:
            try:
                rv, pctl = realized_vol_percentile(df["close"], w=w, lb_days=args.lookback_days, freq=args.freq)
                res.append({"w": w, "realized_vol": round(rv, 8), "pctile": round(pctl, 2)})
            except Exception as e:
                res.append({"w": w, "error": str(e)})
        meta = {"n_obs": int(df.shape[0]), "provider": "mobula"}
        if args.blockchain:
            meta["blockchain"] = args.blockchain
        score = blend_percentiles(res)
        if score is None:
            raise ValueError("no_percentile: all requested windows failed")
        out = {
            "ok": all("error" not in r for r in res),
            "asof": iso_now(),
            "asset": args.asset,
            "freq": args.freq,
            "windows": windows,
            "lookback_days": args.lookback_days,
            "results": res,
            "score": round(score, 2),
            "meta": meta,
        }
        if args.json_output:
            print(json.dumps(out, indent=2 if args.pretty else None))
        else:
            print(f"{out['score']:.2f}")
    except requests.HTTPError as e:
        print(json.dumps({"ok": False, "error": f"http_error:{e.response.status_code}", "asof": iso_now()}))
        raise SystemExit(1)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e), "asof": iso_now()}))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
