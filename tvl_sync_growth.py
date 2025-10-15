#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import hashlib
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:
    SentenceTransformer = None
    st_util = None


LLAMA_TOTAL_URL = "https://api.llama.fi/charts"
LLAMA_CHAIN_URL = "https://api.llama.fi/charts/{chain}"
ASKNEWS_DEFAULT_URL = "https://api.asknews.app/v1/news/search"
ASKNEWS_USER_AGENT = "catalyst-tvl-sync/1.0"

def _dt_utc(ms: int) -> datetime:
    if ms > 10_000_000_000:
        s = ms / 1000.0
    else:
        s = ms
    return datetime.fromtimestamp(s, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

def fetch_llama_series(metric: str) -> pd.Series:
    if metric == "defi_total":
        url = LLAMA_TOTAL_URL
    elif metric.startswith("chain:"):
        chain = metric.split(":", 1)[1]
        url = LLAMA_CHAIN_URL.format(chain=chain)
    else:
        raise ValueError("metric must be 'defi_total' or 'chain:<ChainName>'")

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and "tvl" in data:
        entries = data["tvl"]
    else:
        entries = data

    if not isinstance(entries, list) or len(entries) == 0:
        raise RuntimeError(f"Empty TVL series for {metric}")

    dates = []
    vals = []
    for e in entries:
        ts = e.get("date")
        tvl = e.get("totalLiquidityUSD")
        if ts is None or tvl is None:
            continue
        dt = _dt_utc(int(ts))
        try:
            v = float(tvl)
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        dates.append(dt)
        vals.append(v)

    s = pd.Series(vals, index=pd.to_datetime(dates, utc=True), dtype="float64").sort_index()
    s = s.resample("1D").last().ffill(limit=7)
    s = s[s > 0].dropna()
    if s.empty:
        raise RuntimeError(f"No valid points after cleaning for {metric}")
    return s.rename(metric)


def _normalize_lag_days(lag_days: Optional[Sequence[int]]) -> List[int]:
    if lag_days is None or len(lag_days) == 0:
        return [0]
    seen = set()
    ordered: List[int] = []
    for item in lag_days:
        lag = int(item)
        if lag not in seen:
            seen.add(lag)
            ordered.append(lag)
    if 0 not in seen:
        ordered.insert(0, 0)
    else:
        zero_idx = ordered.index(0)
        if zero_idx != 0:
            ordered.insert(0, ordered.pop(zero_idx))
    return ordered


def _rolling_corr_numpy(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        raise ValueError("lookback/rolling window must be >= 2 for correlation")
    if x.shape != y.shape:
        raise ValueError("arrays must be the same shape")

    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    counts = valid.astype(np.int64)

    x_clean = np.where(valid, x, 0.0)
    y_clean = np.where(valid, y, 0.0)

    xy = x_clean * y_clean
    x2 = x_clean * x_clean
    y2 = y_clean * y_clean

    csum_counts = np.concatenate(([0], np.cumsum(counts)))
    csum_x = np.concatenate(([0.0], np.cumsum(x_clean)))
    csum_y = np.concatenate(([0.0], np.cumsum(y_clean)))
    csum_xy = np.concatenate(([0.0], np.cumsum(xy)))
    csum_x2 = np.concatenate(([0.0], np.cumsum(x2)))
    csum_y2 = np.concatenate(([0.0], np.cumsum(y2)))

    inv_window = 1.0 / window

    for end_idx in range(window, n + 1):
        start_idx = end_idx - window
        cnt = csum_counts[end_idx] - csum_counts[start_idx]
        if cnt != window:
            continue

        sum_x = csum_x[end_idx] - csum_x[start_idx]
        sum_y = csum_y[end_idx] - csum_y[start_idx]
        sum_xy = csum_xy[end_idx] - csum_xy[start_idx]
        sum_x2 = csum_x2[end_idx] - csum_x2[start_idx]
        sum_y2 = csum_y2[end_idx] - csum_y2[start_idx]

        cov = sum_xy - (sum_x * sum_y) * inv_window
        var_x = sum_x2 - (sum_x * sum_x) * inv_window
        var_y = sum_y2 - (sum_y * sum_y) * inv_window
        denom = math.sqrt(var_x * var_y)
        if denom <= 0.0:
            continue
        out[end_idx - 1] = cov / denom

    return out


def _rolling_corr_series(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    if len(x) != len(y):
        raise ValueError("Series must be same length for rolling correlation")
    arr_x = x.to_numpy(dtype=np.float64, copy=False)
    arr_y = y.to_numpy(dtype=np.float64, copy=False)
    values = _rolling_corr_numpy(arr_x, arr_y, window)
    return pd.Series(values, index=x.index, dtype="float64")


def _sleep(seconds: float) -> None:
    if seconds > 0:
        import time

        time.sleep(seconds)


def _respect_rate_limit(resp: requests.Response, fallback: float) -> None:
    retry_after = resp.headers.get("Retry-After")
    wait: Optional[float] = None
    if retry_after:
        try:
            wait = float(retry_after)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(retry_after)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                delta = (parsed - datetime.now(timezone.utc)).total_seconds()
                wait = max(0.0, delta)
            except Exception:
                wait = None
    if wait is None:
        wait = max(fallback, 0.0)
    _sleep(wait)


def _asknews_cache_key(query: str, start: datetime, end: datetime, limit: int) -> str:
    payload = json.dumps(
        {
            "query": query,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": int(limit),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_load(cache_dir: Optional[Path], key: str, ttl_days: Optional[float]) -> Optional[List[Dict[str, Any]]]:
    if not cache_dir:
        return None
    cache_path = cache_dir / f"{key}.json"
    if not cache_path.exists():
        return None
    if ttl_days is not None:
        ttl_seconds = ttl_days * 86400.0
        age = datetime.now(timezone.utc).timestamp() - cache_path.stat().st_mtime
        if age > ttl_seconds:
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return None


def _cache_store(cache_dir: Optional[Path], key: str, payload: List[Dict[str, Any]]) -> None:
    if not cache_dir or not payload:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    cache_path = cache_dir / f"{key}.json"
    try:
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception:
        pass


def _event_context(row: pd.Series, metric1: str, metric2: str) -> str:
    ts = row.name
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts_text = ts.tz_convert("UTC").strftime("%Y-%m-%d")
    else:
        ts_text = str(ts)
    def _pct(v: Any) -> str:
        if v is None:
            return "NA"
        try:
            if not math.isfinite(float(v)):
                return "NA"
        except Exception:
            return "NA"
        return f"{float(v):.2%}"
    def _flt(v: Any) -> str:
        if v is None:
            return "NA"
        try:
            if not math.isfinite(float(v)):
                return "NA"
        except Exception:
            return "NA"
        return f"{float(v):.2f}"
    growth1 = _pct(row.get("growth1"))
    growth2 = _pct(row.get("growth2"))
    corr = _flt(row.get("corr"))
    return (
        f"DeFi synchronized growth event on {ts_text}. "
        f"Metrics: {metric1} and {metric2}. "
        f"Growth {growth1} and {growth2}, correlation {corr}. "
        f"Keywords: DeFi TVL, Ethereum, on-chain liquidity, protocols, staking, layer-2."
    )


def _article_text(article: Dict[str, Any]) -> Tuple[str, Optional[datetime]]:
    title = article.get("title") or article.get("headline") or ""
    summary = article.get("summary") or article.get("description") or ""
    url = article.get("url") or article.get("link") or ""
    text = f"{title}. {summary} [{url}]"
    ts = (
        article.get("published_at")
        or article.get("published")
        or article.get("date")
        or article.get("publishedAt")
        or ""
    )
    dt = None
    if isinstance(ts, str) and ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            dt = None
    return text, dt


class SemanticReranker:
    def __init__(self, model_name: str):
        if SentenceTransformer is None or st_util is None:
            raise ImportError("sentence-transformers is required for --asknews-semantic. Install with `pip install sentence-transformers`.")
        self.model = SentenceTransformer(model_name)

    def rank(
        self,
        event_text: str,
        articles: List[Dict[str, Any]],
        event_center: Optional[datetime],
        date_halflife_days: float,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if not articles:
            return []
        texts, dates = zip(*[_article_text(a) for a in articles])
        event_emb = self.model.encode([event_text], normalize_embeddings=True)
        art_emb = self.model.encode(list(texts), normalize_embeddings=True)
        sims = st_util.cos_sim(event_emb, art_emb).cpu().numpy()[0]
        priors = np.ones_like(sims)
        if event_center is not None:
            center = event_center.replace(tzinfo=timezone.utc)
            denom = max(1e-6, float(date_halflife_days))
            for i, dt in enumerate(dates):
                if isinstance(dt, datetime):
                    ddays = abs((dt.astimezone(timezone.utc) - center).total_seconds()) / 86400.0
                    priors[i] = math.exp(-ddays / denom)
        scores = 0.85 * sims + 0.15 * priors
        order = np.argsort(-scores)
        return [(float(scores[i]), articles[i]) for i in order]


def enrich_events_semantic(
    events: pd.DataFrame,
    *,
    metric1: str,
    metric2: str,
    api_key: str,
    api_id: Optional[str],
    base_url: str,
    window_days: int,
    limit: int,
    fetch_limit: int,
    top_k: int,
    sim_threshold: float,
    model_name: str,
    date_halflife_days: float,
    method: str,
    timeout: float,
    auth_header: Optional[str],
    auth_scheme: Optional[str],
    param_query: str,
    param_start: str,
    param_end: str,
    param_limit: str,
    rate_limit_sleep: float,
    cache_dir: Optional[Path],
    cache_ttl_days: Optional[float],
    group_days: int,
    max_retries: int,
    max_groups: Optional[int],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    reranker = SemanticReranker(model_name)
    window_days = max(0, int(window_days))
    limit = max(1, int(limit))
    fetch_limit = max(int(fetch_limit), limit)
    top_k = max(1, min(int(top_k), fetch_limit, limit))
    sim_threshold = float(sim_threshold)
    date_halflife_days = float(date_halflife_days)
    group_days = max(1, int(group_days))
    enriched = events.copy()
    normalized_index = enriched.index.normalize()
    if normalized_index.empty:
        enriched["asknews_sem_count"] = pd.Series([0] * len(enriched), index=enriched.index, dtype="Int64")
        enriched["asknews_sem_query"] = pd.Series([None] * len(enriched), index=enriched.index, dtype="object")
        return enriched, []
    min_day = normalized_index.min()
    if pd.isna(min_day):
        enriched["asknews_sem_count"] = pd.Series([0] * len(enriched), index=enriched.index, dtype="Int64")
        enriched["asknews_sem_query"] = pd.Series([None] * len(enriched), index=enriched.index, dtype="object")
        return enriched, []
    delta = pd.Timedelta(days=window_days)
    day_offsets = ((normalized_index - min_day) / np.timedelta64(1, "D")).astype(int)
    bucket_ids = day_offsets // group_days
    unique_bucket_ids = np.unique(bucket_ids)
    counts_series = pd.Series([0] * len(enriched), index=enriched.index, dtype="Int64")
    queries_series = pd.Series([None] * len(enriched), index=enriched.index, dtype="object")
    news_details: List[Dict[str, Any]] = []
    processed_groups = 0
    for bucket_id in unique_bucket_ids:
        if max_groups is not None and processed_groups >= max_groups:
            break
        mask = bucket_ids == bucket_id
        bucket_events = enriched.index[mask]
        bucket_days = normalized_index[mask]
        bucket_start_day = bucket_days.min()
        bucket_end_day = bucket_days.max()
        start_dt = (bucket_start_day - delta).to_pydatetime()
        end_dt = (bucket_end_day + delta).to_pydatetime()
        query = f"{metric1} {metric2}"
        cache_key = _asknews_cache_key(query, start_dt, end_dt, fetch_limit) if cache_dir else None
        articles: Optional[List[Dict[str, Any]]] = None
        source = "api"
        if cache_dir and cache_key:
            cached = _cache_load(cache_dir, cache_key, cache_ttl_days)
            if cached is not None:
                articles = cached
                source = "cache"
        try:
            if articles is None:
                articles = fetch_asknews_articles(
                    query=query,
                    start=start_dt,
                    end=end_dt,
                    limit=fetch_limit,
                    api_key=api_key,
                    api_id=api_id,
                    base_url=base_url,
                    method=method,
                    timeout=timeout,
                    auth_header=auth_header,
                    auth_scheme=auth_scheme,
                    param_query=param_query,
                    param_start=param_start,
                    param_end=param_end,
                    param_limit=param_limit,
                    rate_limit_sleep=rate_limit_sleep,
                    max_retries=max_retries,
                )
                if cache_dir and cache_key and articles:
                    _cache_store(cache_dir, cache_key, articles)
        except requests.HTTPError as exc:
            print(f"warn:asknews_http:{bucket_start_day.date()}:{exc}", file=sys.stderr)
            if any(code in str(exc) for code in ("401", "403", "429", "500", "502", "503", "504")):
                break
            continue
        except requests.RequestException as exc:
            print(f"warn:asknews_fetch:{bucket_start_day.date()}:{exc}", file=sys.stderr)
            continue
        if not articles:
            for event_ts in bucket_events:
                row = enriched.loc[event_ts]
                queries_series.loc[event_ts] = _event_context(row, metric1, metric2)
            processed_groups += 1
            continue
        for event_ts in bucket_events:
            row = enriched.loc[event_ts]
            context = _event_context(row, metric1, metric2)
            ranked = reranker.rank(context, articles, event_ts.to_pydatetime() if isinstance(event_ts, pd.Timestamp) else None, date_halflife_days)
            filtered = [(score, art) for score, art in ranked if score >= sim_threshold][:top_k]
            fallback_used = False
            if not filtered and ranked:
                filtered = ranked[:top_k]
                fallback_used = True
            counts_series.loc[event_ts] = len(filtered)
            queries_series.loc[event_ts] = context
            top_articles: List[Dict[str, Any]] = []
            for score, art in filtered:
                text, dt = _article_text(art)
                published = dt.isoformat() if isinstance(dt, datetime) else None
                top_articles.append(
                    {
                        "score": round(float(score), 4),
                        "title": art.get("title") or art.get("headline") or "",
                        "url": art.get("url") or art.get("link") or "",
                        "published_at": published,
                        "raw": art,
                        "fallback": fallback_used,
                    }
                )
            news_details.append(
                {
                    "event_date": event_ts.isoformat() if isinstance(event_ts, pd.Timestamp) else str(event_ts),
                    "group_id": int(bucket_id),
                    "source": source,
                    "query": context,
                    "window_start": start_dt.isoformat(),
                    "window_end": end_dt.isoformat(),
                    "top_articles": top_articles,
                }
            )
        processed_groups += 1
    enriched["asknews_sem_count"] = counts_series
    enriched["asknews_sem_query"] = queries_series.tolist()
    return enriched, news_details
def _extract_asknews_articles(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("articles", "data", "results", "items", "stories"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def fetch_asknews_articles(
    *,
    query: str,
    start: datetime,
    end: datetime,
    limit: int,
    api_key: str,
    api_id: Optional[str] = None,
    base_url: str = ASKNEWS_DEFAULT_URL,
    method: str = "get",
    timeout: float = 30.0,
    auth_header: Optional[str] = "Authorization",
    auth_scheme: Optional[str] = "Bearer",
    param_query: str = "q",
    param_start: str = "from",
    param_end: str = "to",
    param_limit: str = "size",
    rate_limit_sleep: float = 0.0,
    max_retries: int = 0,
) -> List[Dict[str, Any]]:
    if method.lower() != "get":
        print("warn:asknews_method_only_get", file=sys.stderr)
    limit = max(0, int(limit))
    params: Dict[str, Any] = {
        param_query: query,
        param_start: start.isoformat(),
        param_end: end.isoformat(),
        param_limit: int(limit),
    }
    headers: Dict[str, str] = {"User-Agent": ASKNEWS_USER_AGENT}
    if auth_header and api_key:
        token = api_key if not auth_scheme else f"{auth_scheme.strip()} {api_key}".strip()
        headers[auth_header] = token
    max_retries = max(0, int(max_retries))
    attempt = 0
    while True:
        attempt += 1
        resp = requests.get(base_url, headers=headers, params=params, timeout=timeout)
        if resp.status_code == 429:
            msg = resp.text[:500]
            print(f"warn:asknews_rate:429:{msg}", file=sys.stderr)
            fallback = max(rate_limit_sleep, 1.0)
            _respect_rate_limit(resp, fallback)
            raise requests.HTTPError(f"429 {msg}", response=resp)
        if resp.status_code in (401, 403):
            msg = resp.text[:500]
            raise requests.HTTPError(f"{resp.status_code} {msg}", response=resp)
        if 500 <= resp.status_code < 600:
            msg = resp.text[:500]
            print(f"warn:asknews_http:{resp.status_code}:{msg}", file=sys.stderr)
            if attempt > max_retries:
                raise requests.HTTPError(f"{resp.status_code} {msg}", response=resp)
            _respect_rate_limit(resp, max(rate_limit_sleep, 1.0))
            continue
        resp.raise_for_status()
        data = resp.json()
        articles = _extract_asknews_articles(data)
        if not articles or limit == 0:
            return []
        _sleep(rate_limit_sleep)
        return articles[:limit]


def enrich_events_with_asknews(
    events: pd.DataFrame,
    *,
    metric1: str,
    metric2: str,
    api_key: str,
    api_id: Optional[str] = None,
    query_template: str = "{metric1} {metric2}",
    window_days: int = 2,
    limit: int = 3,
    base_url: str = ASKNEWS_DEFAULT_URL,
    method: str = "get",
    timeout: float = 30.0,
    auth_header: Optional[str] = "Authorization",
    auth_scheme: Optional[str] = "Bearer",
    param_query: str = "q",
    param_start: str = "from",
    param_end: str = "to",
    param_limit: str = "size",
    rate_limit_sleep: float = 0.0,
    cache_dir: Optional[Path] = None,
    cache_ttl_days: Optional[float] = None,
    group_days: int = 1,
    max_retries: int = 0,
    max_groups: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if events.empty:
        return events, []
    window_days = max(0, int(window_days))
    group_days = max(1, int(group_days))
    delta = pd.Timedelta(days=window_days)
    enriched = events.copy()
    normalized_index = enriched.index.normalize()
    min_day = normalized_index.min()
    if pd.isna(min_day):
        return enriched, []
    day_offsets = ((normalized_index - min_day) / np.timedelta64(1, "D")).astype(int)
    bucket_ids = day_offsets // group_days
    unique_bucket_ids = np.unique(bucket_ids)
    counts_series = pd.Series([0] * len(enriched), index=enriched.index, dtype="Int64")
    queries_series = pd.Series([None] * len(enriched), index=enriched.index, dtype="object")
    news_details: List[Dict[str, Any]] = []
    processed_groups = 0
    for bucket_id in unique_bucket_ids:
        if max_groups is not None and processed_groups >= max_groups:
            break
        mask = bucket_ids == bucket_id
        bucket_days = normalized_index[mask]
        bucket_events = enriched.index[mask]
        bucket_start_day = bucket_days.min()
        bucket_end_day = bucket_days.max()
        start_dt = (bucket_start_day - delta).to_pydatetime()
        end_dt = (bucket_end_day + delta).to_pydatetime()
        query = query_template.format(
            metric1=metric1,
            metric2=metric2,
            event_date=bucket_start_day.strftime("%Y-%m-%d"),
        )
        cache_key = _asknews_cache_key(query, start_dt, end_dt, limit) if cache_dir else None
        articles: Optional[List[Dict[str, Any]]] = None
        source = "api"
        if cache_dir and cache_key:
            cached = _cache_load(cache_dir, cache_key, cache_ttl_days)
            if cached is not None:
                articles = cached
                source = "cache"
        try:
            if articles is None:
                articles = fetch_asknews_articles(
                    query=query,
                    start=start_dt,
                    end=end_dt,
                    limit=limit,
                    api_key=api_key,
                    api_id=api_id,
                    base_url=base_url,
                    method=method,
                    timeout=timeout,
                    auth_header=auth_header,
                    auth_scheme=auth_scheme,
                    param_query=param_query,
                    param_start=param_start,
                    param_end=param_end,
                    param_limit=param_limit,
                    rate_limit_sleep=rate_limit_sleep,
                    max_retries=max_retries,
                )
                if cache_dir and cache_key and articles is not None:
                    _cache_store(cache_dir, cache_key, articles)
        except requests.HTTPError as exc:
            print(f"warn:asknews_http:{bucket_start_day.date()}:{exc}", file=sys.stderr)
            if any(code in str(exc) for code in ("401", "403", "429", "500", "502", "503", "504")):
                break
            continue
        except requests.RequestException as exc:
            print(f"warn:asknews_fetch:{bucket_start_day.date()}:{exc}", file=sys.stderr)
            continue
        if articles is None:
            continue
        count = len(articles)
        counts_series.loc[bucket_events] = count
        queries_series.loc[bucket_events] = query
        news_details.append(
            {
                "group_id": int(bucket_id),
                "query": query,
                "source": source,
                "window_start": start_dt.isoformat(),
                "window_end": end_dt.isoformat(),
                "event_dates": [ts.isoformat() for ts in bucket_events.to_pydatetime()],
                "articles": articles,
            }
        )
        processed_groups += 1
    enriched["asknews_count"] = counts_series
    enriched["asknews_query"] = queries_series.tolist()
    return enriched, news_details


def sync_growth_cagr(
    s1: pd.Series,
    s2: pd.Series,
    *,
    lookback: int = 14,
    min_corr: float = 0.6,
    min_growth1: float = 0.05,
    min_growth2: float = 0.05,
    n_forward: int = 30,
    require_consecutive: int = 3,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lag_days: Optional[List[int]] = None,
) -> pd.DataFrame:
    df = pd.concat([s1, s2], axis=1).dropna()
    if start:
        df = df[df.index >= pd.to_datetime(start).tz_localize("UTC")]
    if end:
        df = df[df.index <= pd.to_datetime(end).tz_localize("UTC")]
    df = df[(df.iloc[:,0] > 0) & (df.iloc[:,1] > 0)].copy()
    df.columns = ["s1", "s2"]

    if len(df) < max(lookback + 2, n_forward + 2):
        raise ValueError("Insufficient data for requested windows")

    lr1 = np.log(df["s1"]).diff()
    lr2 = np.log(df["s2"]).diff()

    normalized_lags = _normalize_lag_days(lag_days)

    lag_corr_map: Dict[int, pd.Series] = {}
    for lag in normalized_lags:
        if lag == 0:
            target_lr2 = lr2
        else:
            target_lr2 = lr2.shift(lag)
        lag_corr_map[lag] = _rolling_corr_series(lr1, target_lr2, lookback)

    roll_corr = lag_corr_map[0]

    g1 = df["s1"] / df["s1"].shift(lookback) - 1.0
    g2 = df["s2"] / df["s2"].shift(lookback) - 1.0

    cond = (roll_corr >= min_corr) & (g1 >= min_growth1) & (g2 >= min_growth2)

    starts = (cond & ~cond.shift(fill_value=False))
    run_id = starts.cumsum()
    run_len = cond.groupby(run_id).transform("sum")
    is_event = cond & (run_len >= require_consecutive) & ~cond.shift(fill_value=False)

    fwd1 = (df["s1"].shift(-n_forward) / df["s1"]) ** (1.0 / n_forward) - 1.0
    fwd2 = (df["s2"].shift(-n_forward) / df["s2"]) ** (1.0 / n_forward) - 1.0

    out_dict = {
        "s1": df["s1"],
        "s2": df["s2"],
        "corr": roll_corr,
        "growth1": g1,
        "growth2": g2,
        "event": is_event,
        "cagr1_n": fwd1,
        "cagr2_n": fwd2,
    }
    for lag, series in lag_corr_map.items():
        if lag == 0:
            continue
        col = f"corr_lag{lag:+d}".replace("+", "p").replace("-", "m")
        out_dict[col] = series

    if len(lag_corr_map) > 1:
        lag_order = list(lag_corr_map.keys())
        lag_corr_df = pd.concat([lag_corr_map[lag] for lag in lag_order], axis=1)
        lag_corr_df.columns = lag_order
        lag_values = lag_corr_df.to_numpy(dtype=np.float64, copy=False)
        has_valid = np.isfinite(lag_values).any(axis=1)
        best_corr = np.full(len(lag_corr_df), np.nan, dtype=np.float64)
        best_lag = np.full(len(lag_corr_df), np.nan, dtype=np.float64)
        if has_valid.any():
            corr_abs = np.where(np.isfinite(lag_values), np.abs(lag_values), -np.inf)
            best_positions = corr_abs.argmax(axis=1)
            idx_valid = np.where(has_valid)[0]
            col_array = lag_corr_df.columns.to_numpy()
            best_lag[idx_valid] = col_array[best_positions[idx_valid]]
            best_corr[idx_valid] = lag_values[idx_valid, best_positions[idx_valid]]
        out_dict["corr_best"] = pd.Series(best_corr, index=df.index, dtype="float64")
        best_lag_series = pd.Series(best_lag, index=df.index, dtype="float64")
        best_lag_series = best_lag_series.where(best_lag_series.notna())
        out_dict["corr_best_lag"] = best_lag_series.astype("Int64")

    out = pd.DataFrame(out_dict).loc[is_event].copy()

    out["n_forward_days"] = n_forward
    out["lookback_days"] = lookback
    out["min_corr"] = min_corr
    out["min_growth1"] = min_growth1
    out["min_growth2"] = min_growth2
    out["require_consecutive"] = require_consecutive
    out["lag_days"] = ",".join(str(x) for x in normalized_lags)
    return out

def main():
    p = argparse.ArgumentParser(description="Detect synchronized TVL growth events and compute forward n-day CAGR.")
    p.add_argument("--metric1", required=True, help="defi_total or chain:<ChainName> (e.g., chain:Ethereum)")
    p.add_argument("--metric2", required=True, help="defi_total or chain:<ChainName>")
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--min-corr", type=float, default=0.6)
    p.add_argument("--min-growth1", type=float, default=0.05)
    p.add_argument("--min-growth2", type=float, default=0.05)
    p.add_argument("--n-forward", type=int, default=30)
    p.add_argument("--require-consecutive", type=int, default=3)
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (UTC)")
    p.add_argument("--out", type=str, default="events.csv")
    p.add_argument("--dump-series", type=str, default=None, help="Optional CSV to dump aligned s1,s2 daily series")
    p.add_argument(
        "--lag-days",
        type=str,
        default=None,
        help="Comma-separated integer lags (e.g., '1,-1,3') for additional cross-correlation metrics; positive lag shifts metric2 later.",
    )
    p.add_argument("--asknews-key", type=str, default=None, help="AskNews API key; falls back to ASKNEWS_API_KEY env var if omitted.")
    p.add_argument("--asknews-url", type=str, default=ASKNEWS_DEFAULT_URL, help="AskNews search endpoint.")
    p.add_argument(
        "--asknews-http-method",
        type=str,
        default="get",
        choices=["get"],
        help="HTTP method to use for AskNews requests (GET required).",
    )
    p.add_argument("--asknews-window", type=int, default=2, help="Days before/after event date for AskNews query window.")
    p.add_argument("--asknews-limit", type=int, default=3, help="Maximum AskNews stories to retain per event.")
    p.add_argument("--asknews-timeout", type=float, default=30.0, help="Timeout in seconds for AskNews HTTP calls.")
    p.add_argument("--asknews-out", type=str, default=None, help="Optional JSON file to store AskNews results.")
    p.add_argument(
        "--asknews-query-template",
        type=str,
        default="{metric1} {metric2}",
        help="Format string for AskNews queries; has access to {metric1}, {metric2}, {event_date}.",
    )
    p.add_argument("--asknews-auth-header", type=str, default="Authorization", help="Header to carry the AskNews API key.")
    p.add_argument("--asknews-auth-scheme", type=str, default="Bearer", help="Auth scheme prefix for the AskNews header.")
    p.add_argument("--asknews-param-query", type=str, default="q", help="Query field name for AskNews payload.")
    p.add_argument("--asknews-param-start", type=str, default="from", help="Start datetime field name for AskNews payload.")
    p.add_argument("--asknews-param-end", type=str, default="to", help="End datetime field name for AskNews payload.")
    p.add_argument("--asknews-param-limit", type=str, default="size", help="Limit field name for AskNews payload.")
    p.add_argument("--asknews-api-id", type=str, default=None, help="Optional AskNews API ID header value.")
    p.add_argument("--asknews-rate-sleep", type=float, default=0.0, help="Seconds to sleep between successful AskNews calls.")
    p.add_argument("--asknews-group-days", type=int, default=1, help="Group events into N-day buckets for AskNews queries (default 1).")
    p.add_argument("--asknews-cache-dir", type=str, default=None, help="Directory to cache AskNews responses for reuse.")
    p.add_argument("--asknews-cache-ttl-days", type=float, default=None, help="Cache expiration in days (omit to keep indefinitely).")
    p.add_argument("--asknews-max-retries", type=int, default=2, help="Retries for AskNews 5xx responses (default 2).")
    p.add_argument("--asknews-max-groups", type=int, default=None, help="Maximum AskNews groups to process per run (default processes all).")
    p.add_argument("--asknews-semantic", action="store_true", help="Enable semantic reranking of AskNews results.")
    p.add_argument("--asknews-sem-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model for semantic reranking.")
    p.add_argument("--asknews-sem-size", type=int, default=25, help="Number of AskNews articles to retrieve before reranking.")
    p.add_argument("--asknews-sem-top-k", type=int, default=5, help="Top articles to keep after semantic reranking.")
    p.add_argument("--asknews-sem-threshold", type=float, default=0.2, help="Similarity threshold for semantic reranking.")
    p.add_argument("--asknews-sem-halflife", type=float, default=3.0, help="Date proximity halflife (days) for semantic prior.")
    args = p.parse_args()

    try:
        s1 = fetch_llama_series(args.metric1)
        s2 = fetch_llama_series(args.metric2)
    except Exception as e:
        print(f"error_fetch: {e}", file=sys.stderr)
        sys.exit(2)

    aligned = pd.concat([s1.rename("s1"), s2.rename("s2")], axis=1).dropna()
    if args.start:
        aligned = aligned[aligned.index >= pd.to_datetime(args.start).tz_localize("UTC")]
    if args.end:
        aligned = aligned[aligned.index <= pd.to_datetime(args.end).tz_localize("UTC")]
    if args.dump_series:
        aligned.to_csv(args.dump_series, index_label="date")

    lag_list: Optional[List[int]]
    if args.lag_days:
        try:
            lag_list = [int(x.strip()) for x in args.lag_days.split(",") if x.strip()]
        except ValueError:
            print("error_args: lag-days must be comma-separated integers", file=sys.stderr)
            sys.exit(1)
    else:
        lag_list = None

    try:
        events = sync_growth_cagr(
            aligned["s1"],
            aligned["s2"],
            lookback=args.lookback,
            min_corr=args.min_corr,
            min_growth1=args.min_growth1,
            min_growth2=args.min_growth2,
            n_forward=args.n_forward,
            require_consecutive=args.require_consecutive,
            start=args.start,
            end=args.end,
            lag_days=lag_list,
        )
    except Exception as e:
        print(f"error_compute: {e}", file=sys.stderr)
        sys.exit(3)

    asknews_cache_dir: Optional[Path] = None
    if args.asknews_cache_dir:
        asknews_cache_dir = Path(args.asknews_cache_dir).expanduser()
    asknews_key = args.asknews_key or os.environ.get("ASKNEWS_API_KEY")
    asknews_api_id = args.asknews_api_id or os.environ.get("ASKNEWS_API_ID")
    asknews_records: List[Dict[str, Any]] = []
    asknews_mode = None
    if asknews_key:
        try:
            if args.asknews_semantic:
                events, asknews_records = enrich_events_semantic(
                    events,
                    metric1=args.metric1,
                    metric2=args.metric2,
                    api_key=asknews_key,
                    api_id=asknews_api_id,
                    base_url=args.asknews_url,
                    window_days=args.asknews_window,
                    limit=args.asknews_limit,
                    fetch_limit=max(args.asknews_sem_size, args.asknews_limit),
                    top_k=args.asknews_sem_top_k,
                    sim_threshold=args.asknews_sem_threshold,
                    model_name=args.asknews_sem_model,
                    date_halflife_days=args.asknews_sem_halflife,
                    method=args.asknews_http_method,
                    timeout=args.asknews_timeout,
                    auth_header=args.asknews_auth_header if args.asknews_auth_header else None,
                    auth_scheme=args.asknews_auth_scheme,
                    param_query=args.asknews_param_query,
                    param_start=args.asknews_param_start,
                    param_end=args.asknews_param_end,
                    param_limit=args.asknews_param_limit,
                    rate_limit_sleep=args.asknews_rate_sleep,
                    cache_dir=asknews_cache_dir,
                    cache_ttl_days=args.asknews_cache_ttl_days,
                    group_days=args.asknews_group_days,
                    max_retries=args.asknews_max_retries,
                    max_groups=args.asknews_max_groups,
                )
                asknews_mode = "semantic"
            else:
                events, asknews_records = enrich_events_with_asknews(
                    events,
                    metric1=args.metric1,
                    metric2=args.metric2,
                    api_key=asknews_key,
                    api_id=asknews_api_id,
                    query_template=args.asknews_query_template,
                    window_days=args.asknews_window,
                    limit=args.asknews_limit,
                    base_url=args.asknews_url,
                    method=args.asknews_http_method,
                    timeout=args.asknews_timeout,
                    auth_header=args.asknews_auth_header if args.asknews_auth_header else None,
                    auth_scheme=args.asknews_auth_scheme,
                    param_query=args.asknews_param_query,
                    param_start=args.asknews_param_start,
                    param_end=args.asknews_param_end,
                    param_limit=args.asknews_param_limit,
                    rate_limit_sleep=args.asknews_rate_sleep,
                    cache_dir=asknews_cache_dir,
                    cache_ttl_days=args.asknews_cache_ttl_days,
                    group_days=args.asknews_group_days,
                    max_retries=args.asknews_max_retries,
                    max_groups=args.asknews_max_groups,
                )
                asknews_mode = "basic"
        except Exception as exc:
            print(f"warn:asknews:{exc}", file=sys.stderr)
    elif args.asknews_out:
        print("warn:asknews:missing_api_key", file=sys.stderr)

    if args.asknews_out:
        try:
            with open(args.asknews_out, "w", encoding="utf-8") as fp:
                json.dump(asknews_records, fp, indent=2)
        except Exception as exc:
            print(f"warn:asknews_write:{exc}", file=sys.stderr)

    events.to_csv(args.out, index_label="event_date")
    n = len(events)
    if n == 0:
        print("no_events")
        return
    win1 = (events["cagr1_n"] > 0).mean()
    win2 = (events["cagr2_n"] > 0).mean()
    med1 = events["cagr1_n"].median(skipna=True)
    med2 = events["cagr2_n"].median(skipna=True)
    print(f"events={n} win1={win1:.3f} win2={win2:.3f} med1={med1:.5f} med2={med2:.5f}")
    print(f"saved:{args.out}")
    if asknews_mode == "semantic" and "asknews_sem_count" in events.columns and asknews_key:
        asknews_hits = int(events["asknews_sem_count"].fillna(0).sum())
        print(f"asknews_sem_hits={asknews_hits}")
        if args.asknews_out:
            print(f"asknews_saved:{args.asknews_out}")
    elif asknews_mode == "basic" and "asknews_count" in events.columns and asknews_key:
        asknews_hits = int(events["asknews_count"].fillna(0).sum())
        print(f"asknews_hits={asknews_hits}")
        if args.asknews_out:
            print(f"asknews_saved:{args.asknews_out}")
    if args.dump_series:
        print(f"dumped_series:{args.dump_series}")

if __name__ == "__main__":
    main()
