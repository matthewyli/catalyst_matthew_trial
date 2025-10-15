import argparse
import json
import math
import os
import time
from datetime import datetime, timezone, timedelta
from statistics import median
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


MOBULA_HISTORY_URL = "https://api.mobula.io/api/1/market/history"

def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _calc_fetch_window(days: int) -> Tuple[int, int]:
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - int(timedelta(days=days).total_seconds() * 1000)
    return start_ts, end_ts

def annualize_sigma(sig_samp: pd.Series, freq: str) -> pd.Series:
    if freq == "d":
        return sig_samp * math.sqrt(365.0)
    if freq == "h":
        return sig_samp * math.sqrt(24.0 * 365.0)
    raise ValueError("d or h please")

def _expected_points(freq: str, days: int) -> int:
    return int(days * (365 if freq == "d" else 24 * 365))

def _period_seconds(freq: str) -> int:
    return 24 * 3600 if freq == "d" else 3600

def _coverage_conf(n_obs: int, freq: str, need_days: int) -> float:
    exp_pts = _expected_points(freq, need_days)
    return max(0.0, min(1.0, n_obs / max(1, exp_pts)))

def _window_success_conf(results: List[dict]) -> float:
    total = len(results)
    ok = sum(1 for r in results if "pctile" in r and not math.isnan(r["pctile"]))
    return 0.0 if total == 0 else ok / total

def _agreement_conf(results: List[dict]) -> float:
    pctiles = [r["pctile"] for r in results if "pctile" in r and not math.isnan(r["pctile"])]
    if len(pctiles) < 2:
        return 0.5
    med = median(pctiles)
    mad = median(abs(p - med) for p in pctiles)
    return max(0.0, min(1.0, 1.0 - (mad / 20.0)))

def _freshness_conf(last_ts: pd.Timestamp, now_ts: pd.Timestamp, freq: str) -> float:
    age_s = max(0.0, (now_ts - last_ts).total_seconds())
    period = _period_seconds(freq)
    if age_s <= period:
        return 1.0
    return math.exp(- (age_s - period) / (6 * period))

def compute_confidence(df: pd.DataFrame, results: List[dict], freq: str, lookback_days: int, windows: List[int]) -> Tuple[float, dict]:
    now_ts = pd.Timestamp.utcnow()
    last_ts = df.index.max()
    need_days = max(lookback_days, max(windows, default=0)) + 1

    c1 = _coverage_conf(len(df), freq, need_days)
    c2 = _window_success_conf(results)
    c3 = _agreement_conf(results)
    c4 = _freshness_conf(last_ts, now_ts, freq)

    weights = dict(c1=0.30, c2=0.25, c3=0.25, c4=0.20)
    parts = dict(c1=c1, c2=c2, c3=c3, c4=c4)
    eps = 1e-12
    log_sum = sum(weights[k] * math.log(max(parts[k], eps)) for k in parts)
    confidence = math.exp(log_sum)
    return confidence, parts


def fetch_prices_mobula(symbol: str, freq: str, days: int, api_key: str, blockchain: Optional[str] = None) -> pd.DataFrame:
    assert freq in ("d", "h")
    period = "1d" if freq == "d" else "1h"
    start_ts, end_ts = _calc_fetch_window(days)

    params: dict[str, object] = {
        "asset": symbol,
        "period": period,
        "from": start_ts,
        "to": end_ts,
    }
    if isinstance(symbol, str) and symbol.lower().startswith("0x") and blockchain:
        params["blockchain"] = blockchain

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.get(MOBULA_HISTORY_URL, headers=headers, params=params, timeout=30)

    if r.status_code == 401:
        raise ValueError("need key")
    if r.status_code >= 400:
        raise ValueError(f"mobula_error:{r.status_code}:{r.text}")

    data = r.json()
    history = data.get("data", {}).get("price_history")
    if not isinstance(history, list):
        raise ValueError(f"no_price_data:raw={data}")

    rows: List[Tuple[datetime, float]] = []
    for entry in history:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        ts, price = entry[0], entry[1]
        if ts is None or price is None:
            continue
        dt = pd.to_datetime(ts, unit="ms", utc=True)
        rows.append((dt, float(price)))

    if not rows:
        raise ValueError("no_price_data:empty")

    df = pd.DataFrame(rows, columns=["ts", "close"]).set_index("ts").sort_index()
    return df


def realized_vol_percentile(close: pd.Series, w: int, lb_days: int, freq: str) -> tuple[float, float, float]:
    r = np.log(close).diff().dropna()
    if w < 2:
        raise ValueError("window must be >=2")
    sig = r.rolling(w).std(ddof=1).dropna()
    sig_ann = annualize_sigma(sig, freq)
    available = sig_ann.iloc[:-1]
    if available.empty:
        raise ValueError("insufficient_history")
    lb_obs = lb_days if freq == "d" else lb_days * 24
    lb_obs = max(lb_obs, 1)
    hist = available if len(available) <= lb_obs else available.iloc[-lb_obs:]
    min_required = max(10, w)
    if len(hist) < min_required:
        raise ValueError("insufficient_history")
    coverage = min(1.0, len(hist) / max(lb_obs, min_required))
    current = float(sig_ann.iloc[-1])
    pctile = 100.0 * float((hist <= current).mean())
    return current, pctile, coverage


def blend_percentiles(results: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    scored = []
    for r in results:
        pct = r.get("pctile")
        cov = r.get("confidence")
        if pct is None or cov is None or math.isnan(pct):
            continue
        cov = max(0.0, min(1.0, float(cov)))
        scored.append((r["w"], pct, cov))
    if not scored:
        return None, None
    scored.sort(key=lambda x: x[0])
    base_weights = [math.exp(-w / 45.0) for w, _, _ in scored]
    base_total = sum(base_weights)
    weighted_cov = [bw * cov for bw, (_, _, cov) in zip(base_weights, scored)]
    cov_total = sum(weighted_cov)
    if cov_total == 0 or base_total == 0:
        return float(median(p for _, p, _ in scored)), 0.0
    score = sum(wc * pct for wc, (_, pct, _) in zip(weighted_cov, scored)) / cov_total
    overall_conf = cov_total / base_total
    return float(score), float(overall_conf)


def main():
    p = argparse.ArgumentParser(description="Volatility regime percentile tool")
    p.add_argument("--asset", required=True, help="Mobula asset identifier")
    default_windows = "1,3,7,14,21,30,45,60,90"
    p.add_argument("--windows", default=default_windows, help=f"windows in days (default: {default_windows})")
    p.add_argument("--blockchain", default=None, help="Mobula blockchain identifier eg ethereum")
    p.add_argument("--lookback-days", type=int, default=365, help="percentile lookback horizon in days")
    p.add_argument("--freq", choices=["d", "h"], default="d", help="price interval")
    p.add_argument("--mobula-api-key", default=os.getenv("MOBULA_API_KEY", ""), help="Mobula API key")
    p.add_argument("--pretty", action="store_true", help="print better looking json")
    p.add_argument("--json-output", action="store_true", help="full json")
    args = p.parse_args()

    try:
        windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
        if not windows:
            raise ValueError("windows cannot be empty")
        max_w = max(windows)
        fetch_days = args.lookback_days + max_w
        df = fetch_prices_mobula(args.asset, args.freq, fetch_days, args.mobula_api_key, args.blockchain)
        results: List[dict] = []
        for w in windows:
            try:
                rv, pctl, cov = realized_vol_percentile(df["close"], w=w, lb_days=args.lookback_days, freq=args.freq)
                results.append(
                    {
                        "w": w,
                        "realized_vol": round(rv, 8),
                        "pctile": round(pctl, 2),
                        "confidence": round(cov, 4),
                        "confidence_pct": round(cov * 100.0, 1), #same as confidence btw
                    }
                )
            except Exception as e:
                results.append({"w": w, "error": str(e)})
        meta = {"n_obs": int(df.shape[0]), "provider": "mobula"}
        if args.blockchain:
            meta["blockchain"] = args.blockchain

        score, score_conf = blend_percentiles(results)
        if score is None:
            raise ValueError("no_percentile: all requested windows failed")

        conf_scalar, conf_parts = compute_confidence(df, results, args.freq, args.lookback_days, windows)

        out = {
            "ok": all("error" not in r for r in results),
            "asof": iso_now(),
            "asset": args.asset,
            "freq": args.freq,
            "windows": windows,
            "lookback_days": args.lookback_days,
            "results": results,
            "score": round(score, 2),
            "confidence": round(conf_scalar, 3),
            "confidence_breakdown": {k: round(v, 3) for k, v in conf_parts.items()},
            "meta": meta,
        }
        if score_conf is not None:
            out["score_confidence"] = round(score_conf, 4)
            out["score_confidence_pct"] = round(score_conf * 100.0, 1)

        if args.json_output:
            print(json.dumps(out, indent=2 if args.pretty else None))
        else:
            if score_conf is not None:
                print(f"{out['score']:.2f} ({score_conf * 100.0:.1f}%)")
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
