
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import requests
except ImportError as exc:
    raise ImportError(
        "The 'requests' package is required to call AskNews and OpenAI. Install it via 'pip install requests'."
    ) from exc


__TOOL_META__ = {
    "name": "legacy.asknews_framework",
    "module": "tools.legacy.asknews_framework",
    "object": "main",
    "description": "AEI pipeline that scores near-term crypto news using AskNews + OpenAI sentiment.",
    "phases": ["data_gather", "signal_generation", "research"],
    "entrypoint": "python -m tools.legacy.asknews_framework",
    "outputs": [
        "impact_score",
        "direction",
        "confidence",
        "articles",
    ],
}


ASKNEWS_DEFAULT_URL = "https://api.asknews.app/v1/news/search"
ASKNEWS_USER_AGENT = "asknews-aei/1.0"
OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


_ENV_LOADED: Dict[Path, bool] = {}


def _parse_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return env
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        env[key] = value
    return env


def load_env(paths: Optional[Sequence[Path]] = None) -> Dict[str, str]:
    """Load .env files into os.environ (non-destructive)."""

    defaults: List[Path] = []
    if paths is None:
        here = Path(__file__).resolve()
        defaults.extend(
            [
                Path.cwd() / ".env",
                here.parent / ".env",
                here.parent.parent / ".env",
                here.parent.parent.parent / ".env",
            ]
        )
    else:
        defaults.extend(paths)

    merged: Dict[str, str] = {}
    for raw_path in defaults:
        try:
            path = raw_path.resolve()
        except Exception:
            continue
        if not path.exists() or not path.is_file():
            continue
        if _ENV_LOADED.get(path):
            continue
        parsed = _parse_env_file(path)
        if not parsed:
            _ENV_LOADED[path] = True
            continue
        for key, value in parsed.items():
            if key and value and key not in os.environ:
                os.environ[key] = value
            merged[key] = value
        _ENV_LOADED[path] = True
    return merged


def _warn(msg: str) -> None:
    print(f"warn:aei:{msg}", file=sys.stderr)


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_ts(ts: Any) -> dt.datetime:
    if isinstance(ts, dt.datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=dt.timezone.utc)
        return ts.astimezone(dt.timezone.utc)
    if isinstance(ts, (int, float)):
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    if isinstance(ts, str):
        try:
            return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        except Exception:
            return utcnow()
    return utcnow()


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def tanh_clip(x: float) -> float:
    return math.tanh(x)


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    denom = b if abs(b) > eps else (eps if b >= 0 else -eps)
    return a / denom


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1.0 - p))


def inv_logit(z: float) -> float:
    return sigmoid(z)


def minutes_ago(ts: dt.datetime) -> int:
    return max(0, int((utcnow() - ts).total_seconds() // 60))


def cosine_from_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, v in a.items():
        if k in b:
            dot += v * b[k]
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9_+]+", " ", text)
    return [tok for tok in text.split() if len(tok) > 1]


def bag(text: str) -> Counter:
    return Counter(tokenize(text))


from .aei_models import Article, ResearchDecision



DEFAULT_EVENT_PRIORS = {
    "exploit": 0.9,
    "outage": 0.7,
    "delist": 0.6,
    "regulatory": 0.5,
    "tokenomics": 0.45,
    "fundraise": 0.4,
    "partnership": 0.35,
    "listing": 0.3,
    "upgrade": 0.3,
    "metrics": 0.25,
    "macro": 0.2,
    "governance": 0.2,
}

DEFAULT_SOURCE_QUALITY = {
    "reuters.com": 0.95,
    "bloomberg.com": 0.95,
    "coindesk.com": 0.9,
    "cointelegraph.com": 0.85,
    "theblock.co": 0.85,
    "fortune.com": 0.8,
    "wsj.com": 0.9,
    "decrypt.co": 0.75,
    "techcrunch.com": 0.75,
    "twitter.com": 0.6,
    "x.com": 0.6,
}

DEFAULT_ASSET_ALIASES = {
    "BTC": {"tickers": ["btc", "bitcoin"], "chain": ["bitcoin network", "btc chain"], "ecosystem": ["btc ecosystem"]},
    "ETH": {"tickers": ["eth", "ethereum"], "chain": ["eth mainnet", "ethereum mainnet", "l1 ethereum"], "ecosystem": ["ethereum ecosystem", "evm", "rollup", "l2"]},
    "SOL": {"tickers": ["sol", "solana"], "chain": ["solana mainnet", "solana chain"], "ecosystem": ["solana ecosystem"]},
    "USDC": {"tickers": ["usdc", "usd coin"], "chain": ["circle usdc", "stablecoin usdc"], "ecosystem": ["stablecoin", "circle"]},
    "ARBITRUM": {"tickers": ["arb", "arbitrum"], "chain": ["arbitrum one", "arb l2"], "ecosystem": ["arbitrum ecosystem", "rollup"]},
}

DEFAULT_CRYPTO_CONTEXT = [
    "blockchain", "chain", "token", "protocol", "crypto", "defi", "dex", "rollup",
    "bridge", "wallet", "validator", "staking", "mainnet", "l2", "layer2", "airdrop", "mint",
]

DEFAULT_NEWS_EXPANSIONS = {
    "BTC": ["bitcoin", "btc", "lightning network", "ordinals", "etf", "hashrate", "miner", "halving", "macro"],
    "ETH": ["ethereum", "eth", "layer 2", "rollup", "staking", "beacon chain", "lido", "eigenlayer", "restaking", "bridge", "defi", "erc-20", "erc-4337", "arbitrum", "optimism", "base"],
    "SOL": ["solana", "sol", "validator", "helium", "saga phone", "jupiter", "defi", "downtime", "rpc"],
    "USDC": ["usdc", "stablecoin", "circle", "reserves", "treasury", "usd coin", "payment"],
    "ARBITRUM": ["arbitrum", "arb", "rollup", "nitro", "orbit", "dao", "layer 2", "bridge", "sequencer"],
}

DEFAULT_OPS_GUIDANCE = {
    "high_threshold": 0.5,
    "medium_threshold": 0.25,
    "high": {"widen_quotes_bps": 6, "max_kelly_mult": 0.4, "tighten_stops": True},
    "medium": {"widen_quotes_bps": 3, "max_kelly_mult": 0.7, "tighten_stops": True},
    "low": {"widen_quotes_bps": 0, "max_kelly_mult": 1.0, "tighten_stops": False},
}

DEFAULT_GLOBAL_QUERY = '("crypto" OR "blockchain" OR "defi" OR "stablecoin" OR "layer 2" OR "airdrop" OR "governance" OR "exchange" OR "exploit" OR "hack" OR "token" OR "protocol")'

DEFAULT_EVENT_RULES = [
    {"label": "exploit", "keywords": ["exploit", "hack", "breach", "bridge exploit", "compromise"]},
    {"label": "outage", "keywords": ["outage", "downtime", "halt", "stalled", "incident"]},
    {"label": "listing", "keywords": ["lists", "listing", "lists on", "added to"]},
    {"label": "delist", "keywords": ["delist", "removes trading", "delisted"]},
    {"label": "partnership", "keywords": ["partners", "partnership", "collaborates", "integrates with"]},
    {"label": "fundraise", "keywords": ["raises", "fundraise", "series a", "funding round"]},
    {"label": "tokenomics", "keywords": ["burn", "supply", "emissions", "unlock", "staking rewards"]},
    {"label": "governance", "keywords": ["vote", "proposal", "governance", "snapshot", "dao"]},
    {"label": "regulatory", "keywords": ["sec", "regulation", "lawsuit", "complaint", "fine", "approval"]},
    {"label": "macro", "keywords": ["cpi", "macro", "inflation", "unemployment", "rate hike"]},
    {"label": "upgrade", "keywords": ["upgrade", "hardfork", "fork", "merge", "release"]},
    {"label": "metrics", "keywords": ["tvl", "volume", "market cap", "users", "addresses", "growth"]},
]

DEFAULT_POSITIVE_WORDS = [
    "upgrade", "partnership", "listing", "growth", "bullish", "gain", "improve", "fundraise", "integrate", "adoption", "surge"
]
DEFAULT_NEGATIVE_WORDS = [
    "exploit", "hack", "outage", "downtime", "breach", "loss", "bearish", "drop", "cut", "regulatory", "fine", "lawsuit", "delist"
]

DEFAULT_SENTIMENT_BUMPS = {
    "exploit": -0.2,
    "outage": -0.1,
    "listing": 0.2,
    "regulatory": -0.1,
    "upgrade": 0.05,
    "fundraise": 0.05,
}

@dataclass
class AEIConfig:
    tau_min: int = 120
    min_score: float = 0.25
    max_items: int = 100
    dedup_sim_threshold: float = 0.9
    nov_lookback_min: int = 24 * 60
    asknews_cache_ttl_min: int = 5
    max_api_calls_per_run: int = 5
    asknews_min_interval_sec: float = 1.0
    event_priors: Dict[str, float] = dataclasses.field(default_factory=dict)
    source_quality: Dict[str, float] = dataclasses.field(default_factory=dict)
    asset_aliases: Dict[str, Dict[str, List[str]]] = dataclasses.field(default_factory=dict)
    crypto_context: List[str] = dataclasses.field(default_factory=list)
    news_expansions: Dict[str, List[str]] = dataclasses.field(default_factory=dict)
    ops_guidance: Dict[str, Any] = dataclasses.field(default_factory=dict)
    sentiment_bumps: Dict[str, float] = dataclasses.field(default_factory=dict)
    global_query: Optional[str] = None


class SimpleSentiment:
    '''Very small lexicon-based sentiment fallback in [-1, 1], populated dynamically.'''

    def __init__(self, positive: Optional[Sequence[str]] = None, negative: Optional[Sequence[str]] = None) -> None:
        self.POS = set(positive or DEFAULT_POSITIVE_WORDS)
        self.NEG = set(negative or DEFAULT_NEGATIVE_WORDS)

    def polarity(self, text: str) -> float:
        toks = tokenize(text)
        pos = sum(1 for t in toks if t in self.POS)
        neg = sum(1 for t in toks if t in self.NEG)
        if pos == 0 and neg == 0:
            return 0.0
        score = (pos - neg) / max(1, pos + neg)
        return max(-1.0, min(1.0, score))


class ChatGPTSentiment:
    
    def __init__(
        self,
        *,
        api_key: str,
        model: str = OPENAI_DEFAULT_MODEL,
        api_base: str = OPENAI_DEFAULT_BASE,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for ChatGPTSentiment")
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def polarity(self, text: str, *, asset: str = "", event_types: Optional[Sequence[str]] = None) -> float:
        if not text.strip():
            return 0.0
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        labels = ", ".join(event_types or []) or "unknown"
        user_content = (
            "You will rate how bullish or bearish this crypto news item is for the short-term "
            "(1-6h) price move of the specified asset. Return strict JSON with keys sentiment "
            "(float in [-1,1]) and confidence (float in [0,1]).\n"
            f"Asset: {asset or 'unknown'}\n"
            f"Event Types: {labels}\n"
            "News Snippet:\n"
            f"""{text.strip()}"""
        )
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a disciplined quantitative crypto analyst. Respond only with valid JSON. "
                        "sentiment should be -1 for extremely bearish, 0 neutral, +1 extremely bullish."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        }
        url = f"{self.api_base}/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("missing choices in ChatGPT response")
            content = choices[0].get("message", {}).get("content")
            if not content:
                raise ValueError("missing content in ChatGPT response")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"{.*}", content, re.DOTALL)
                if not match:
                    raise
                parsed = json.loads(match.group(0))
            sentiment = float(parsed.get("sentiment", 0.0))
        except Exception as exc:
            raise RuntimeError(f"ChatGPT sentiment failed: {exc}") from exc
        return max(-1.0, min(1.0, sentiment))


class RuleEventClassifier:
    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None) -> None:
        if rules:
            parsed: List[Tuple[str, List[str]]] = []
            for item in rules:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "metrics").strip()
                keywords = [str(k).lower() for k in item.get("keywords", []) if isinstance(k, str) and k]
                if label and keywords:
                    parsed.append((label, keywords))
            if parsed:
                self._rules = parsed
            else:
                self._rules = [(r["label"], [kw.lower() for kw in r["keywords"]]) for r in DEFAULT_EVENT_RULES]
        else:
            self._rules = [(r["label"], [kw.lower() for kw in r["keywords"]]) for r in DEFAULT_EVENT_RULES]

    def predict(self, text: str) -> List[str]:
        t = (text or "").lower()
        labels = [label for label, keys in self._rules if any(k in t for k in keys)]
        if not labels:
            labels.append("metrics")
        return labels


class SimpleEmbedder:
    def embed(self, text: str) -> Counter:
        return bag(text)

    def cosine(self, a: Counter, b: Counter) -> float:
        return cosine_from_counts(a, b)

class ChatGPTResearcher:
    
    def __init__(
        self,
        *,
        api_key: str,
        model: str = OPENAI_DEFAULT_MODEL,
        api_base: str = OPENAI_DEFAULT_BASE,
        timeout: float = 45.0,
        max_items: int = 20,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for ChatGPTResearcher")
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_items = max(1, int(max_items))

    def analyze(self, asset: str, articles: Sequence[Article]) -> ResearchDecision:
        if not articles:
            return ResearchDecision(articles=[], summary=None, notes={})
        payload_articles = []
        for idx, article in enumerate(articles[: self.max_items]):
            payload_articles.append(
                {
                    "index": idx,
                    "title": article.title,
                    "summary": article.summary,
                    "source": article.source_domain,
                    "published_at": article.published_at.isoformat().replace("+00:00", "Z"),
                    "url": article.url,
                }
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        user_content = (
            "You are assisting a crypto trading desk. Review the articles below and decide which ones are relevant "
            "for near-term price impact on the specified asset. Provide a concise research-style brief.\n"
            f"Asset: {asset}\n"
            "Return JSON with keys:\n"
            "  summary: string (<= 120 words) synthesizing the situation.\n"
            "  articles: list of objects with keys index (int), relevant (bool), note (string <= 120 chars).\n"
            "Focus on specific, actionable developments. If an article is irrelevant, mark relevant=false with reason."
        )
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior crypto researcher. Respond ONLY with valid JSON."
                        " Be decisive about relevance."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instructions": user_content,
                            "articles": payload_articles,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        url = f"{self.api_base}/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = (
                (data.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            parsed = json.loads(content)
        except Exception as exc:
            raise RuntimeError(f"ChatGPT research failed: {exc}") from exc

        summary = parsed.get("summary")
        decisions = parsed.get("articles") or []
        has_decisions = bool(decisions)
        evaluated_indices: set[int] = set()
        keep_indices: set[int] = set()
        raw_notes: Dict[int, str] = {}
        for item in decisions:
            try:
                idx = int(item.get("index"))
            except Exception:
                continue
            evaluated_indices.add(idx)
            relevant = bool(item.get("relevant", True))
            note = item.get("note")
            if relevant:
                keep_indices.add(idx)
            if note:
                raw_notes[idx] = str(note)

        selected: List[Article] = []
        notes: Dict[str, str] = {}
        for idx, article in enumerate(articles):
            within_payload = idx < len(payload_articles)
            if has_decisions and within_payload:
                if evaluated_indices and idx not in evaluated_indices:
                    pass
                elif keep_indices and idx not in keep_indices:
                    continue
                elif not keep_indices and evaluated_indices:
                    continue
            selected.append(article)
            note_val = raw_notes.get(idx)
            if note_val:
                key = article.url or f"idx:{idx}"
                notes[key] = note_val

        return ResearchDecision(articles=selected, summary=summary, notes=notes)


class OpenAISyntheticNews:
    
    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        seed_model: str,
        sifter_model: str,
        analyst_model: str,
        max_items: int = 12,
        timeout: float = 45.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for OpenAISyntheticNews")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.seed_model = seed_model
        self.sifter_model = sifter_model
        self.analyst_model = analyst_model
        self.max_items = max(1, int(max_items))
        self.timeout = timeout

    def _post(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if response_format:
            payload["response_format"] = response_format
        url = f"{self.api_base}/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI response missing choices")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("OpenAI response missing content")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"{.*}", content, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def generate(self, *, asset: str, window_min: int, max_items: int, now: Optional[dt.datetime] = None) -> ResearchDecision:
        now = now or utcnow()
        tier0 = self._tier_seed(asset=asset, window_min=window_min)
        tier1 = self._tier_headlines(asset=asset, tier0=tier0, window_min=window_min, max_items=max_items)
        tier2 = self._tier_analyze(asset=asset, tier0=tier0, tier1=tier1, window_min=window_min, now=now)
        articles: List[Article] = tier2.pop("articles", [])  
        summary = tier2.get("research_summary")
        raw_notes = tier2.get("notes") or {}
        notes = {str(k): str(v) for k, v in raw_notes.items() if v is not None}
        return ResearchDecision(articles=articles, summary=summary, notes=notes)

    def _tier_seed(self, *, asset: str, window_min: int) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Tier0 of a crypto news scout. Output compact JSON with fields keywords (array of strings), "
                    "hot_topics (array of strings), macro_watch (array of strings), risk_flags (array of strings)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "asset": asset,
                        "window_min": window_min,
                        "instructions": "Focus on short-horizon (<=6h) catalysts, exploits, regulatory moves, exchange actions, major partners, L2 events."
                    }
                ),
            },
        ]
        data = self._post(
            model=self.seed_model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        data["tier"] = 0
        return data

    def _tier_headlines(
        self,
        *,
        asset: str,
        tier0: Dict[str, Any],
        window_min: int,
        max_items: int,
    ) -> Dict[str, Any]:
        keywords = tier0.get("keywords") or []
        hot_topics = tier0.get("hot_topics") or []
        prompt = {
            "asset": asset,
            "keywords": keywords,
            "topics": hot_topics,
            "window_min": window_min,
            "max_items": min(self.max_items, max_items * 2),
            "instructions": "Return JSON with candidates list; each item must include title, summary, source, estimated_minutes_ago, confidence (0-1), sentiment (-1..1), direction (up/down), event_types (array), novelty_hint (string), tags (array). Mark 'uncertain' if unsure."
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Tier1 of a crypto news scout. You synthesize likely fresh headlines across crypto feeds "
                    "(AskNews, Twitter, exchanges, regulators). Use your knowledge up to the current moment."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        data = self._post(
            model=self.sifter_model,
            messages=messages,
            temperature=0.35,
            response_format={"type": "json_object"},
        )
        data["tier"] = 1
        return data

    def _tier_analyze(
        self,
        *,
        asset: str,
        tier0: Dict[str, Any],
        tier1: Dict[str, Any],
        window_min: int,
        now: dt.datetime,
    ) -> Dict[str, Any]:
        candidates = tier1.get("candidates") or []
        payload = {
            "asset": asset,
            "window_min": window_min,
            "seed": tier0,
            "candidates": candidates,
            "instructions": (
                "Evaluate each candidate. Discard if implausible. For each kept item return title, summary, source, minutes_ago, "
                "sentiment (-1..1), direction, event_types, novelty (0-1), source_confidence (0-1), impact_hint (0-1), confidence (0-1), notes (<=120 chars). "
                "Also provide research_summary (<=120 words) and optional notes map keyed by canonical_id (use url if available)."
            ),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Tier2 - the crypto desk analyst. Be conservative. If a candidate seems speculative or outdated, drop it. "
                    "Only output verified-looking items. Respond with JSON containing research_summary, articles (list), notes (object)."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        data = self._post(
            model=self.analyst_model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        articles_out: List[Dict[str, Any]] = []
        notes_map: Dict[str, str] = {}
        for item in data.get("articles", []) or []:
            title = item.get("title") or ""
            summary = item.get("summary") or item.get("notes") or ""
            if not title:
                continue
            minutes_ago = float(item.get("minutes_ago", window_min / 2))
            minutes_ago = max(0.0, minutes_ago)
            published = now - dt.timedelta(minutes=minutes_ago)
            source_domain = (item.get("source") or "synthetic.ask").lower()
            url_id = item.get("canonical_id") or hashlib.sha1(title.encode("utf-8", errors="ignore")).hexdigest()[:16]
            url = item.get("url") or f"synthetic://{asset.lower()}/{url_id}"
            raw = {
                "tier": 2,
                "seed": tier0,
                "candidate": item,
            }
            article = Article(
                url=url,
                title=title,
                summary=summary,
                published_at=published,
                source_domain=source_domain,
                raw=raw,
            )
            articles_out.append(article)
            noted = item.get("notes")
            if noted:
                notes_map[url] = str(noted)
        merged_notes = data.get("notes") or {}
        for key, value in notes_map.items():
            merged_notes[str(key)] = value
        data["notes"] = merged_notes
        data["articles"] = articles_out
        return data


class LLMConfigProvider:
    '''Fetches configuration scaffolding from OpenAI once per session.'''

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        assets_hint: Sequence[str],
        timeout: float = 45.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for LLMConfigProvider")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.assets_hint = [a.upper() for a in assets_hint if a]
        self._cache: Optional[Dict[str, Any]] = None

    def fetch(self) -> Optional[Dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        instructions = {
            "assets": self.assets_hint,
            "required_keys": [
                "asset_aliases",
                "crypto_context",
                "news_expansions",
                "event_priors",
                "source_quality",
                "event_rules",
                "sentiment_lexicon",
                "sentiment_bumps",
                "ops_guidance",
                "global_query"
            ],
            "notes": (
                "Return concise JSON. asset_aliases should map symbols to tickers/chain/ecosystem lists. "
                "crypto_context should be high-signal context words. news_expansions should map symbols to supporting keywords. "
                "event_priors and source_quality values must be between 0 and 1. event_rules should be list of objects with label and keywords. "
                "sentiment_lexicon should have positive and negative arrays; sentiment_bumps should map event labels to additive adjustments. "
                "ops_guidance should include high_threshold, medium_threshold, and profiles for high/medium/low. global_query should be a boolean expression suitable for broad AskNews search.")
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": ("You are configuring a crypto news impact engine. Respond ONLY with valid JSON covering the required keys."),
                },
                {"role": "user", "content": json.dumps(instructions, ensure_ascii=False)},
            ],
        }
        url = f"{self.api_base}/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if not content:
                raise ValueError("missing content")
            parsed = json.loads(content)
            self._cache = parsed
            return parsed
        except Exception as exc:
            _warn(f"config_oracle_failed:{exc}")
            self._cache = None
            return None


class KeywordOracle:
    """Generates structured keyword packs for assets via OpenAI."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        timeout: float = 30.0,
        cache_size: int = 64,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for KeywordOracle")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._cache: Dict[str, Dict[str, List[str]]] = {}
        self._cache_order: List[str] = []
        self._cache_size = max(1, int(cache_size))

    def get(self, asset: str, *, context: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
        key = asset.upper()
        if key in self._cache:
            return self._cache[key]
        generated = self._generate(asset=asset, context=context)
        if not generated:
            return None
        self._cache[key] = generated
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_size:
            victim = self._cache_order.pop(0)
            self._cache.pop(victim, None)
        return generated

    def _generate(self, *, asset: str, context: Optional[str]) -> Optional[Dict[str, List[str]]]:
        instructions = {
            "asset": asset,
            "context": context,
            "output_spec": [
                "tickers",
                "symbols",
                "chain_terms",
                "ecosystem_terms",
                "sector_keywords",
                "risk_terms",
            ],
            "requirements": (
                "Return JSON with the keys exactly as above. Each value must be an array of unique lowercase strings. "
                "Include protocol nicknames, key partners, L2/L3 names, governance bodies, and event themes relevant to the asset."
            ),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a crypto ontology assistant. Respond ONLY with valid JSON using the required keys."
                    ),
                },
                {"role": "user", "content": json.dumps(instructions, ensure_ascii=False)},
            ],
        }
        url = f"{self.api_base}/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if not content:
                raise ValueError("missing content")
            parsed = json.loads(content)
        except Exception as exc:
            _warn(f"keyword_oracle_failed:{asset}:{exc}")
            return None

        result: Dict[str, List[str]] = {}
        for key, values in parsed.items():
            if not isinstance(values, list):
                continue
            cleaned: List[str] = []
            seen: set[str] = set()
            for val in values:
                if not isinstance(val, str):
                    continue
                v = val.strip().lower()
                if not v or v in seen:
                    continue
                seen.add(v)
                cleaned.append(v)
            if cleaned:
                result[key] = cleaned
        if not result:
            return None
        return result


class AskNewsClient:
    """Light AskNews REST client compatible with AEIEngine."""

    def __init__(
        self,
        *,
        api_key: str,
        api_id: Optional[str] = None,
        base_url: str = ASKNEWS_DEFAULT_URL,
        timeout: float = 20.0,
        rate_limit_sleep: float = 0.0,
        max_retries: int = 1,
        min_interval_sec: float = 0.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for AskNewsClient")
        self.api_key = api_key
        self.api_id = api_id
        self.base_url = base_url
        self.timeout = timeout
        self.rate_limit_sleep = max(0.0, rate_limit_sleep)
        self.max_retries = max(0, int(max_retries))
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._last_call_ts: Optional[float] = None

    def search(self, *, query: str, since_minutes: int, limit: int) -> List[Dict[str, Any]]:
        now = utcnow()
        start = now - dt.timedelta(minutes=max(0, int(since_minutes)))
        params = {
            "q": query,
            "from": start.isoformat(),
            "to": now.isoformat(),
            "size": max(1, int(limit)),
        }
        headers = {
            "User-Agent": ASKNEWS_USER_AGENT,
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
        }
        if self.api_id:
            headers["x-api-id"] = self.api_id

        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            resp = requests.get(self.base_url, headers=headers, params=params, timeout=self.timeout)
            self._last_call_ts = time.time()
            if resp.status_code == 429:
                msg = resp.text[:200]
                _warn(f"asknews_rate_limited:{msg}")
                if attempt > self.max_retries:
                    break
                _respect_rate_limit(resp, self.rate_limit_sleep or 1.0)
                continue
            if resp.status_code in (401, 403):
                raise requests.HTTPError(f"AskNews auth error {resp.status_code}: {resp.text[:200]}", response=resp)
            if 500 <= resp.status_code < 600:
                if attempt > self.max_retries:
                    resp.raise_for_status()
                _warn(f"asknews_retry:{resp.status_code}")
                _respect_rate_limit(resp, self.rate_limit_sleep or 1.0)
                continue
            resp.raise_for_status()
            payload = resp.json()
            articles = _extract_articles(payload)
            return articles[: max(0, int(limit))]
        return []

    def _throttle(self) -> None:
        if self.min_interval_sec <= 0.0:
            return
        now = time.time()
        if self._last_call_ts is None:
            return
        elapsed = now - self._last_call_ts
        wait = self.min_interval_sec - elapsed
        if wait > 0:
            try:
                time.sleep(wait)
            except Exception:
                pass


def _extract_articles(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("articles", "data", "results", "items", "stories"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _respect_rate_limit(resp: requests.Response, fallback: float) -> None:
    retry_after = resp.headers.get("Retry-After")
    wait: Optional[float] = None
    if retry_after:
        try:
            wait = float(retry_after)
        except ValueError:
            try:
                parsed = dt.datetime.fromisoformat(retry_after.replace("Z", "+00:00"))
                wait = max(0.0, (parsed - utcnow()).total_seconds())
            except Exception:
                wait = None
    if wait is None:
        wait = max(fallback, 0.0)
    if wait > 0.0:
        try:
            import time

            time.sleep(wait)
        except Exception:
            pass


class AEIEngine:
    def __init__(
        self,
        *,
        asknews_client: Optional[Any] = None,
        config: Optional[AEIConfig] = None,
        embedder: Optional[SimpleEmbedder] = None,
        event_classifier: Optional[RuleEventClassifier] = None,
        sentiment_model: Optional[SimpleSentiment] = None,
        chat_sentiment: Optional[ChatGPTSentiment] = None,
        research_assistant: Optional[ChatGPTResearcher] = None,
        synthetic_news: Optional[OpenAISyntheticNews] = None,
        keyword_oracle: Optional[KeywordOracle] = None,
        provider_mode: str = "asknews",
        debug: bool = False,
    ) -> None:
        self.client = asknews_client
        self.cfg = config or AEIConfig()
        self.embedder = embedder or SimpleEmbedder()
        self.event_classifier = event_classifier or RuleEventClassifier()
        self.sentiment_fallback = sentiment_model or SimpleSentiment()
        self.chat_sentiment = chat_sentiment
        self.research_assistant = research_assistant
        self.synthetic_news = synthetic_news
        self.keyword_oracle = keyword_oracle

        self.provider_mode = provider_mode.lower()
        self.enable_sentiment_rule_bumps = True
        self.debug = debug

        def _lower_list(values: Sequence[str]) -> List[str]:
            return [str(v).lower() for v in values if isinstance(v, str) and v]

        base_aliases = self.cfg.asset_aliases or DEFAULT_ASSET_ALIASES
        prepared_aliases: Dict[str, Dict[str, List[str]]] = {}
        for sym, groups in base_aliases.items():
            if not isinstance(groups, dict):
                continue
            prepared_aliases[sym.upper()] = {
                "tickers": _lower_list(groups.get("tickers", [])),
                "chain": _lower_list(groups.get("chain", [])),
                "ecosystem": _lower_list(groups.get("ecosystem", [])),
            }
        self.cfg.asset_aliases = prepared_aliases

        base_expansions = self.cfg.news_expansions or DEFAULT_NEWS_EXPANSIONS
        self.cfg.news_expansions = {sym.upper(): _lower_list(words) for sym, words in base_expansions.items()}
        self.cfg.crypto_context = _lower_list(self.cfg.crypto_context or DEFAULT_CRYPTO_CONTEXT)

        base_priors = self.cfg.event_priors or DEFAULT_EVENT_PRIORS
        priors: Dict[str, float] = {}
        for label, value in base_priors.items():
            try:
                priors[str(label).lower()] = float(value)
            except Exception:
                continue
        self.event_priors = priors or dict(DEFAULT_EVENT_PRIORS)

        base_sources = self.cfg.source_quality or DEFAULT_SOURCE_QUALITY
        sources: Dict[str, float] = {}
        for domain, value in base_sources.items():
            try:
                sources[str(domain).lower()] = float(value)
            except Exception:
                continue
        self.source_quality = sources or dict(DEFAULT_SOURCE_QUALITY)

        self.ops_cfg = self._normalise_ops_cfg(self.cfg.ops_guidance or DEFAULT_OPS_GUIDANCE)
        bumps_source = self.cfg.sentiment_bumps or DEFAULT_SENTIMENT_BUMPS
        self.sentiment_bumps = {str(label).lower(): float(value) for label, value in bumps_source.items()}
        self.global_query_template = self.cfg.global_query or DEFAULT_GLOBAL_QUERY

        self._recent_vectors: Dict[str, List[Tuple[dt.datetime, Counter]]] = defaultdict(list)
        self._asknews_cache: Dict[Tuple[str, str, int], Tuple[dt.datetime, List[Article]]] = {}
        self._api_calls_this_run: int = 0
        self._global_news_cache: Dict[int, List[Article]] = {}
        self._dynamic_aliases: Dict[str, Dict[str, List[str]]] = {}
        self._dynamic_expansions: Dict[str, List[str]] = {}

    def _normalise_ops_cfg(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        cfg = copy.deepcopy(DEFAULT_OPS_GUIDANCE)
        if not isinstance(raw, dict):
            return cfg
        high = raw.get("high_threshold")
        medium = raw.get("medium_threshold")
        try:
            if high is not None:
                cfg["high_threshold"] = float(high)
        except Exception:
            pass
        try:
            if medium is not None:
                cfg["medium_threshold"] = float(medium)
        except Exception:
            pass
        for level in ("high", "medium", "low"):
            profile = raw.get(level)
            if isinstance(profile, dict):
                cfg[level].update(profile)
        return cfg

    @staticmethod
    def _apply_ops_profile(flags: Dict[str, Any], profile: Dict[str, Any]) -> None:
        if not isinstance(profile, dict):
            return
        if "tighten_stops" in profile:
            flags["tighten_stops"] = bool(profile["tighten_stops"])
        if "max_kelly_mult" in profile:
            try:
                flags["max_kelly_mult"] = float(profile["max_kelly_mult"])
            except Exception:
                pass
        if "widen_quotes_bps" in profile:
            try:
                flags["widen_quotes_bps"] = int(profile["widen_quotes_bps"])
            except Exception:
                pass
        if "suspend_new_longs" in profile:
            flags["suspend_new_longs"] = bool(profile["suspend_new_longs"])


    def run(
        self,
        *,
        assets: Sequence[str],
        window_min: int,
        min_score: Optional[float] = None,
        max_items: Optional[int] = None,
        market_ref: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        if self.provider_mode in ("asknews", "hybrid") and not hasattr(self.client, "search"):
            raise RuntimeError("AskNews client must provide a search() method when provider_mode requires it")
        if self.provider_mode in ("synthetic", "hybrid") and self.synthetic_news is None:
            raise RuntimeError("Synthetic provider requested but OpenAI synthetic pipeline not configured")

        min_score = self.cfg.min_score if min_score is None else float(min_score)
        max_items = self.cfg.max_items if max_items is None else int(max_items)
        market_ref = market_ref or {}

        self._api_calls_this_run = 0
        self._global_news_cache.clear()
        self._log("run:start", assets=list(assets), window_min=window_min)
        results: List[Dict[str, Any]] = []
        aggregate_components: List[Tuple[float, str]] = []
        for asset in assets:
            self._log("asset:start", asset=asset)
            decision = self._gather_articles(asset=asset, window_min=window_min, max_items=max_items)
            articles_for_scoring = decision.articles
            per_article, agg = self._score_asset(
                asset=asset,
                articles=articles_for_scoring,
                market_ref=market_ref.get(asset, {}),
                research_notes=decision.notes,
                research_summary=decision.summary,
            )
            if agg["impact_score"] >= min_score or per_article:
                results.append(self._format_asset_output(asset, per_article, agg))
                aggregate_components.append((agg["impact_score"], agg["dir"]))
            self._log(
                "asset:end",
                asset=asset,
                n_articles=len(per_article),
                impact_score=agg["impact_score"],
                api_calls=self._api_calls_this_run,
            )
        summary = self._summarize_run(results, aggregate_components)
        if summary:
            results.append(summary)
        return results

    def _summarize_run(self, results: List[Dict[str, Any]], components: List[Tuple[float, str]]) -> Optional[Dict[str, Any]]:
        if not results or not components:
            return None
        total_articles = sum(r.get("n_articles", 0) for r in results if isinstance(r, dict) and "n_articles" in r)
        if len(results) == 1:
            first = results[0]
            return {"summary": {"impact_score": first.get("impact_score", 0.0), "dir": first.get("dir", "flat"), "confidence": first.get("confidence", 0.0), "n_articles": first.get("n_articles", 0)}}
        total_score = sum(score for score, _ in components)
        if total_score <= 0:
            return {"summary": {"impact_score": 0.0, "dir": "flat", "confidence": 0.0, "n_articles": total_articles}}
        dir_score = sum((1 if direction == "up" else -1) * score for score, direction in components)
        aggregate_score = total_score / len(components)
        direction = "up" if dir_score >= 0 else "down"
        normalized_direction = abs(dir_score) / total_score
        confidence = min(0.95, normalized_direction)
        return {"summary": {"impact_score": round(aggregate_score, 4), "dir": direction, "confidence": round(confidence, 4), "n_articles": total_articles}}

    def _log(self, event: str, **fields: Any) -> None:
        if not self.debug:
            return
        payload = {"event": event, **fields}
        try:
            print(json.dumps(payload, default=str), file=sys.stderr)
        except Exception:
            print(f"DEBUG {event} {fields}", file=sys.stderr)

    def _gather_articles(self, *, asset: str, window_min: int, max_items: int) -> ResearchDecision:
        now = utcnow()
        mode = self.provider_mode
        decision: ResearchDecision
        provider = "asknews"

        if mode in ("synthetic", "hybrid") and self.synthetic_news is not None:
            if mode == "synthetic":
                try:
                    decision = self.synthetic_news.generate(
                        asset=asset, window_min=window_min, max_items=max_items, now=now
                    )
                    decision.articles = self._dedup_articles(decision.articles)
                    self._log("provider:synthetic", asset=asset, n_articles=len(decision.articles))
                    return decision
                except Exception as exc:
                    _warn(f"synthetic_generate_failed:{asset}:{exc}")
                    self._log("provider:synthetic_error", asset=asset, error=str(exc))
        articles = self._fetch_articles_for_asset(asset, window_min, max_items)
        articles = self._dedup_articles(articles)
        self._log("provider:asknews", asset=asset, n_articles=len(articles))
        decision = self._apply_research(asset, articles)
        if decision.articles:
            return decision

        if mode in ("asknews", "hybrid"):
            global_articles = self._fetch_global_news(window_min=window_min, max_items=max_items)
            if global_articles:
                self._log("provider:asknews_global", asset=asset, n_articles=len(global_articles))
                decision = self._apply_research(asset, global_articles)
                if decision.articles:
                    return decision

        if mode == "hybrid" and self.synthetic_news is not None:
            provider = "synthetic"
            try:
                decision = self.synthetic_news.generate(
                    asset=asset, window_min=window_min, max_items=max_items, now=now
                )
                decision.articles = self._dedup_articles(decision.articles)
                self._log("provider:fallback_synthetic", asset=asset, n_articles=len(decision.articles))
                return decision
            except Exception as exc:
                _warn(f"synthetic_fallback_failed:{asset}:{exc}")
                self._log("provider:fallback_error", asset=asset, error=str(exc))

        self._log("provider:none", asset=asset, n_articles=0, mode=mode, provider=provider)
        return ResearchDecision(articles=[], summary=None, notes={})

    def _build_query(self, asset: str) -> str:
        asset_key = asset.upper()
        self._ensure_aliases(asset)
        aliases = self.cfg.asset_aliases.get(asset_key, {})
        dynamic_aliases = self._dynamic_aliases.get(asset_key, {})
        if aliases and dynamic_aliases:
            aliases = {
                "tickers": list(dict.fromkeys([*aliases.get("tickers", []), *dynamic_aliases.get("tickers", [])])),
                "chain": list(dict.fromkeys([*aliases.get("chain", []), *dynamic_aliases.get("chain", [])])),
                "ecosystem": list(dict.fromkeys([*aliases.get("ecosystem", []), *dynamic_aliases.get("ecosystem", [])])),
            }
        elif not aliases:
            aliases = dynamic_aliases
        tokens: List[str] = []
        for bucket in ("tickers", "chain", "ecosystem"):
            tokens.extend(aliases.get(bucket, []))
        tokens.extend(self.cfg.news_expansions.get(asset_key, []))
        tokens.extend(self._dynamic_expansions.get(asset_key, []))
        tokens.append(asset)
        primary_terms: List[str] = []
        seen: set[str] = set()
        for term in tokens:
            clean = term.strip()
            if not clean:
                continue
            if clean.lower() in seen:
                continue
            seen.add(clean.lower())
            primary_terms.append(f'"{clean}"')

        if not primary_terms:
            return ""

        primary_clause = " OR ".join(primary_terms)
        context_terms = [f'"{ctx}"' for ctx in self.cfg.crypto_context[:8]]
        event_terms: List[str] = []
        if hasattr(self.event_classifier, "RULES"):
            for _, keys in getattr(self.event_classifier, "RULES", []):
                for key in keys[:2]:
                    key_clean = key.strip()
                    if key_clean and key_clean.lower() not in seen:
                        event_terms.append(f'"{key_clean}"')
                        seen.add(key_clean.lower())
                if len(event_terms) >= 12:
                    break

        clauses: List[str] = [f"({primary_clause})"]
        if context_terms:
            clauses.append(f"(({primary_clause}) AND ({' OR '.join(context_terms)}))")
        if event_terms:
            clauses.append(f"(({primary_clause}) AND ({' OR '.join(event_terms[:12])}))")
        return " OR ".join(clauses)

    def _fetch_articles_for_asset(self, asset: str, window_min: int, max_items: int) -> List[Article]:
        query = self._build_query(asset)
        if not query:
            return []
        self._log("asset:query", asset=asset, query=query)
        articles = self._query_asknews(asset=asset, query=query, window_min=window_min, max_items=max_items)
        cutoff = utcnow() - dt.timedelta(minutes=window_min)
        filtered = [a for a in articles if a.published_at >= cutoff]
        filtered.sort(key=lambda a: a.published_at)
        self._log("asset:fetched", asset=asset, raw=len(articles), filtered=len(filtered))
        return filtered

    def _fetch_global_news(self, *, window_min: int, max_items: int) -> List[Article]:
        cached = self._global_news_cache.get(int(window_min))
        if cached is not None:
            return [dataclasses.replace(article) for article in cached]
        if not hasattr(self.client, "search"):
            return []
        global_query = self.global_query_template
        articles = self._query_asknews(
            asset="__GLOBAL__", query=global_query, window_min=window_min, max_items=max_items
        )
        cutoff = utcnow() - dt.timedelta(minutes=window_min)
        filtered = [a for a in articles if a.published_at >= cutoff]
        filtered.sort(key=lambda a: a.published_at)
        deduped = self._dedup_articles(filtered)
        self._global_news_cache[int(window_min)] = [dataclasses.replace(article) for article in deduped]
        return deduped

    def _ensure_aliases(self, asset: str) -> Dict[str, List[str]]:
        key = asset.upper()
        if key in self.cfg.asset_aliases or key in self._dynamic_aliases:
            aliases = self.cfg.asset_aliases.get(key)
            if aliases is None:
                aliases = self._dynamic_aliases.get(key, {})
            else:
                dynamic = self._dynamic_aliases.get(key, {})
                if dynamic:
                    merged = {k: list(set([*aliases.get(k, []), *dynamic.get(k, [])])) for k in ("tickers", "chain", "ecosystem")}
                    aliases = merged
            return aliases or {}
        if self.keyword_oracle:
            context_hint = " ".join(self.cfg.crypto_context[:12]) if self.cfg.crypto_context else None
            generated = self.keyword_oracle.get(asset, context=context_hint)
            if generated:
                aliases = {
                    "tickers": [str(x).lower() for x in generated.get("tickers", []) + generated.get("symbols", []) if isinstance(x, str)],
                    "chain": [str(x).lower() for x in generated.get("chain_terms", []) if isinstance(x, str)],
                    "ecosystem": [str(x).lower() for x in generated.get("ecosystem_terms", []) if isinstance(x, str)],
                }
                self._dynamic_aliases[key] = aliases
                self.cfg.asset_aliases[key] = aliases
                extras = [str(x).lower() for x in (generated.get("sector_keywords", []) + generated.get("risk_terms", [])) if isinstance(x, str)]
                if extras:
                    self._dynamic_expansions[key] = extras
                self._log("alias:generated", asset=asset, aliases=aliases, extras=extras)
                return aliases
        self._dynamic_aliases[key] = {"tickers": [asset.lower()], "chain": [], "ecosystem": []}
        return self._dynamic_aliases[key]

    def _query_asknews(self, *, asset: str, query: str, window_min: int, max_items: int) -> List[Article]:
        key = (asset.upper(), query, int(window_min))
        now = utcnow()
        ttl = dt.timedelta(minutes=max(1, self.cfg.asknews_cache_ttl_min))
        cached = self._asknews_cache.get(key)
        if cached and now - cached[0] <= ttl:
            cached_articles = cached[1]
            self._log("asknews:cache_hit", asset=asset, age_sec=int((now - cached[0]).total_seconds()), items=len(cached_articles))
            return [dataclasses.replace(article) for article in cached_articles]

        if self._api_calls_this_run >= self.cfg.max_api_calls_per_run:
            _warn(f"asknews_cap_hit:{asset}:{self.cfg.max_api_calls_per_run}")
            self._log("asknews:cap_hit", asset=asset, cap=self.cfg.max_api_calls_per_run)
            if cached:
                return [dataclasses.replace(article) for article in cached[1]]
            return []

        try:
            raw_items = self.client.search(query=query, since_minutes=window_min, limit=max_items)
        except TypeError:
            raw_items = self.client.search(query, window_min, max_items)  
        except Exception as exc:
            _warn(f"asknews_search_failed:{asset}:{exc}")
            self._log("asknews:error", asset=asset, error=str(exc))
            if cached:
                return [dataclasses.replace(article) for article in cached[1]]
            return []
        else:
            self._api_calls_this_run += 1

        articles: List[Article] = []
        for r in raw_items or []:
            url = r.get("url") or r.get("link") or ""
            title = r.get("title") or r.get("headline") or ""
            summary = r.get("summary") or r.get("description") or ""
            ts = parse_ts(r.get("published_at") or r.get("published") or r.get("date"))
            domain = (r.get("source_domain") or r.get("source") or r.get("domain") or "").lower()
            articles.append(
                Article(
                    url=url,
                    title=title,
                    summary=summary,
                    published_at=ts,
                    source_domain=domain,
                    raw=r,
                )
            )

        self._asknews_cache[key] = (now, [dataclasses.replace(article) for article in articles])
        self._log("asknews:fetched", asset=asset, items=len(articles), api_calls=self._api_calls_this_run)
        return articles

    def _apply_research(self, asset: str, articles: List[Article]) -> ResearchDecision:
        if not articles:
            return ResearchDecision(articles=[], summary=None, notes={})
        if not self.research_assistant:
            return ResearchDecision(articles=list(articles), summary=None, notes={})
        try:
            decision = self.research_assistant.analyze(asset, articles)
            self._log(
                "research:result",
                asset=asset,
                kept=len(decision.articles),
                dropped=len(articles) - len(decision.articles),
                summary=(decision.summary[:120] + "…") if decision.summary and len(decision.summary) > 120 else decision.summary,
            )
            return decision
        except Exception as exc:
            _warn(f"research_failed:{asset}:{exc}")
            self._log("research:error", asset=asset, error=str(exc))
            return ResearchDecision(articles=list(articles), summary=None, notes={})

    def _dedup_articles(self, articles: List[Article]) -> List[Article]:
        seen: Dict[str, Article] = {}
        vectors: Dict[str, Counter] = {}
        keep: List[Article] = []

        for article in articles:
            url_key = hashlib.sha1((article.url or article.title).encode("utf-8", errors="ignore")).hexdigest()[:16]
            if url_key in seen:
                continue
            vec = self.embedder.embed(article.text)
            duplicate = False
            for prev in vectors.values():
                if self.embedder.cosine(vec, prev) >= self.cfg.dedup_sim_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen[url_key] = article
            vectors[url_key] = vec
            keep.append(article)
        self._log("asset:dedup", kept=len(keep), dropped=len(articles) - len(keep))
        return keep

    def _entity_match_score(self, asset: str, text: str) -> float:
        aliases = self.cfg.asset_aliases.get(asset.upper(), {})
        t = (text or "").lower()
        context_ok = any(c in t for c in self.cfg.crypto_context) or any(
            kw in t for kw in ["eth", "bitcoin", "blockchain", "token", "defi", "nft", "dex"]
        )
        for tok in aliases.get("tickers", []):
            if re.search(rf"\b{re.escape(tok)}\b", t):
                return 1.0 if context_ok else 0.4
        for tok in aliases.get("chain", []):
            if tok in t:
                return 0.7
        for tok in aliases.get("ecosystem", []):
            if tok in t:
                return 0.4
        if asset.upper() in ("BTC", "ETH") and any(m in t for m in ["cpi", "fed", "macro", "inflation", "unemployment", "rate hike"]):
            return 0.25
        return 0.0

    def _event_types(self, text: str) -> List[str]:
        return self.event_classifier.predict(text)

    def _source_quality(self, domain: str) -> float:
        d = (domain or "").lower().replace("www.", "")
        return float(self.source_quality.get(d, 0.6))

    def _novelty(self, asset: str, vec: Counter, ts: dt.datetime) -> float:
        recent = self._recent_vectors[asset]
        horizon = ts - dt.timedelta(minutes=self.cfg.nov_lookback_min)
        self._recent_vectors[asset] = [(t0, v0) for (t0, v0) in recent if t0 >= horizon]
        max_sim = 0.0
        for (t0, v0) in self._recent_vectors[asset]:
            if t0 <= ts:
                max_sim = max(max_sim, self.embedder.cosine(vec, v0))
        self._recent_vectors[asset].append((ts, vec))
        return max(0.0, min(1.0, 1.0 - max_sim))

    def _sentiment(self, asset: str, text: str, labels: List[str]) -> float:
        sentiment_value: Optional[float] = None
        used_chat = False
        if self.chat_sentiment:
            try:
                sentiment_value = self.chat_sentiment.polarity(text, asset=asset, event_types=labels)
                used_chat = True
            except Exception as exc:
                _warn(f"chatgpt_sentiment_failed:{asset}:{exc}")
                sentiment_value = None
        if sentiment_value is None:
            sentiment_value = self.sentiment_fallback.polarity(text)
        if not used_chat and self.enable_sentiment_rule_bumps:
            bumps = sum(self.sentiment_bumps.get(label, 0.0) for label in labels)
            sentiment_value += bumps
        return max(-1.0, min(1.0, sentiment_value))

    def _time_decay(self, ts: dt.datetime) -> float:
        dt_min = max(0.0, (utcnow() - ts).total_seconds() / 60.0)
        return math.exp(-dt_min / float(self.cfg.tau_min))

    def _p_event(self, labels: List[str]) -> float:
        priors = [self.event_priors.get(label, 0.25) for label in labels]
        return max(priors) if priors else 0.25

    def _per_article_impact(self, *, D: float, P_event: float, Q: float, N: float, E: float, S: float) -> Tuple[float, int]:
        impact = D * P_event * Q * N * E * abs(S)
        if abs(S) < 0.05:
            direction = -1 if P_event >= 0.5 else 1
        else:
            direction = 1 if S > 0 else -1
        return impact, direction

    def _score_asset(
        self,
        *,
        asset: str,
        articles: List[Article],
        market_ref: Dict[str, float],
        research_notes: Optional[Dict[str, str]] = None,
        research_summary: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        per_article: List[Dict[str, Any]] = []

        for article in articles:
            vec = self.embedder.embed(article.text)
            entity_score = self._entity_match_score(asset, article.text)
            if entity_score <= 0.0:
                continue
            labels = self._event_types(article.text)
            q = self._source_quality(article.source_domain)
            novelty = self._novelty(asset, vec, article.published_at)
            decay = self._time_decay(article.published_at)
            sentiment = self._sentiment(asset, article.text, labels)
            p_event = self._p_event(labels)
            impact, direction = self._per_article_impact(
                D=decay, P_event=p_event, Q=q, N=novelty, E=entity_score, S=sentiment
            )
            note_key = article.url or f"idx:{len(per_article)}"
            research_note = None
            if research_notes:
                research_note = research_notes.get(note_key)

            per_article.append(
                {
                    "url": article.url,
                    "title": article.title,
                    "published_at": article.published_at.isoformat().replace("+00:00", "Z"),
                    "source_domain": article.source_domain,
                    "labels": labels,
                    "components": {
                        "event_weight": round(p_event, 4),
                        "sentiment": round(sentiment, 4),
                        "novelty": round(novelty, 4),
                        "source_quality": round(q, 4),
                        "entity_match": round(entity_score, 4),
                        "time_decay": round(decay, 4),
                    },
                    "impact": round(impact, 6),
                    "dir": "up" if direction > 0 else "down",
                    "research_note": research_note,
                }
            )

        total_impact = sum(item["impact"] for item in per_article)
        norm_vol = float(market_ref.get("norm_vol_4h", 1.0))
        impact_score = tanh_clip(safe_div(total_impact, max(1e-6, norm_vol)))

        signed_sum = sum((1 if item["dir"] == "up" else -1) * item["impact"] for item in per_article)
        dir_prob_up = sigmoid(2.0 * signed_sum)
        direction = "up" if dir_prob_up >= 0.5 else "down"

        conf_prod = 1.0
        for item in per_article:
            conf_prod *= (1.0 - min(1.0, float(item["impact"])) )
        confidence = min(0.95, 1.0 - conf_prod)

        event_types = sorted({label for item in per_article for label in item["labels"]})

        def wavg(key: str, default: float = 0.0) -> float:
            if not per_article:
                return default
            weights = [item["impact"] for item in per_article]
            weight_sum = sum(weights)
            if weight_sum <= 0:
                return default
            num = sum(item["components"][key] * item["impact"] for item in per_article)
            return num / weight_sum

        components = {
            "event_weight": round(wavg("event_weight", 0.0), 4),
            "sentiment": round(wavg("sentiment", 0.0), 4),
            "novelty": round(wavg("novelty", 0.0), 4),
            "source_quality": round(wavg("source_quality", 0.0), 4),
            "entity_match": round(wavg("entity_match", 0.0), 4),
            "time_decay": round(wavg("time_decay", 0.0), 4),
        }

        reasons: List[str] = []
        if per_article:
            top = max(per_article, key=lambda item: item["impact"])
            labels_str = ", ".join(top["labels"]) or "unknown"
            reasons.append(f"Highest-impact article tagged {labels_str}")
            reasons.append(f"Average source quality {components['source_quality']:.2f}")
            freshness = minutes_ago(parse_ts(top["published_at"]))
            reasons.append(f"Fresh within {freshness} minutes")

        ops_cfg = self.ops_cfg
        low_profile = ops_cfg.get("low", {})
        ops_flags: Dict[str, Any] = {
            "suspend_new_longs": False,
            "tighten_stops": bool(low_profile.get("tighten_stops", False)),
            "max_kelly_mult": float(low_profile.get("max_kelly_mult", 1.0)),
            "widen_quotes_bps": int(low_profile.get("widen_quotes_bps", 0)),
        }
        high_thr = float(ops_cfg.get("high_threshold", DEFAULT_OPS_GUIDANCE["high_threshold"]))
        med_thr = float(ops_cfg.get("medium_threshold", DEFAULT_OPS_GUIDANCE["medium_threshold"]))
        if impact_score >= high_thr:
            self._apply_ops_profile(ops_flags, ops_cfg.get("high", {}))
        elif impact_score >= med_thr:
            self._apply_ops_profile(ops_flags, ops_cfg.get("medium", {}))
        else:
            self._apply_ops_profile(ops_flags, low_profile)

        if direction == "down" and any(t in event_types for t in ["exploit", "regulatory"]):
            if any(item["impact"] > 0.4 for item in per_article):
                ops_flags["suspend_new_longs"] = True

        agg = {
            "asof": utcnow().isoformat().replace("+00:00", "Z"),
            "n_articles": len(per_article),
            "impact_score": round(impact_score, 4),
            "dir": direction,
            "confidence": round(confidence, 4),
            "components": components,
            "event_types": event_types,
            "top_reasons": reasons,
            "ops_flags": ops_flags,
            "research_summary": research_summary,
        }

        self._log(
            "asset:score",
            asset=asset,
            n_articles=len(per_article),
            impact_score=agg["impact_score"],
            direction=direction,
            confidence=agg["confidence"],
        )

        return per_article, agg

    def _format_asset_output(self, asset: str, per_article: List[Dict[str, Any]], agg: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "asset": asset,
            "asof": agg["asof"],
            "n_articles": agg["n_articles"],
            "impact_score": agg["impact_score"],
            "dir": agg["dir"],
            "confidence": agg["confidence"],
            "components": agg["components"],
            "event_types": agg["event_types"],
            "top_reasons": agg["top_reasons"],
            "ops_flags": agg["ops_flags"],
            "research_summary": agg.get("research_summary"),
            "_articles": per_article,
        }
        return out

    def update_with_realized(
        self,
        *,
        article_records: Sequence[Dict[str, Any]],
        alpha: float = 100.0,
        eta: float = 0.05,
    ) -> None:
        per_label_values: Dict[str, List[float]] = defaultdict(list)
        for record in article_records:
            labels = record.get("labels", []) or []
            value = float(record.get("realized_abs_ret", 0.0))
            for label in labels:
                per_label_values[label].append(value)
        for label, values in per_label_values.items():
            old = float(self.event_priors.get(label, 0.25))
            count = len(values)
            updated = (alpha * old + sum(values)) / (alpha + count)
            self.event_priors[label] = float(min(1.5, max(0.05, updated)))

        for record in article_records:
            domain = (record.get("source_domain") or "").lower().replace("www.", "")
            sentiment = float(record.get("sentiment", 0.0))
            realized = float(record.get("realized_ret", 0.0))
            agree = 0.0
            if abs(sentiment) < 1e-6 and abs(realized) < 1e-6:
                agree = 0.0
            elif sentiment * realized > 0:
                agree = 1.0
            elif sentiment * realized < 0:
                agree = -1.0
            q_old = float(self.source_quality.get(domain, 0.6))
            z = logit(q_old) + eta * agree
            q_new = inv_logit(z)
            self.source_quality[domain] = float(max(0.2, min(1.0, q_new)))


try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    _FASTAPI_AVAILABLE = True
except Exception:
    FastAPI = None
    JSONResponse = None
    _FASTAPI_AVAILABLE = False


app = None
_FASTAPI_ENGINE: Optional[AEIEngine] = None


def _get_fastapi_engine(debug: bool = False) -> AEIEngine:
    global _FASTAPI_ENGINE
    if _FASTAPI_ENGINE is None:
        _FASTAPI_ENGINE = build_engine_from_env(debug=debug)
    return _FASTAPI_ENGINE


if _FASTAPI_AVAILABLE:
    app = FastAPI()

    @app.post("/tools/aei/run")
    def run_aei(payload: Dict[str, Any]):
        engine = _get_fastapi_engine()
        assets = payload.get("assets") or []
        if not assets:
            return JSONResponse({"error": "assets list required"}, status_code=400)
        result = engine.run(
            assets=assets,
            window_min=int(payload.get("window_min", 90)),
            min_score=float(payload.get("min_score", engine.cfg.min_score)),
            max_items=int(payload.get("max_items", engine.cfg.max_items)),
            market_ref=payload.get("market_ref"),
        )
        return JSONResponse(result)


class _DummyAskNewsClient:
    def search(self, *, query: str, since_minutes: int, limit: int) -> List[Dict[str, Any]]:
        now = utcnow()
        return [
            {
                "url": "https://example.com/exploit",
                "title": "Bridge exploit drains $35M on ETH L2",
                "summary": "Attack on a rollup bridge triggers security response.",
                "published_at": (now - dt.timedelta(minutes=45)).isoformat().replace("+00:00", "Z"),
                "source_domain": "coindesk.com",
            },
            {
                "url": "https://example.com/outage",
                "title": "Validator outage on L2",
                "summary": "Partial downtime reported; validators investigating.",
                "published_at": (now - dt.timedelta(minutes=60)).isoformat().replace("+00:00", "Z"),
                "source_domain": "theblock.co",
            },
            {
                "url": "https://example.com/partnership",
                "title": "Partnership ETH x Cloud",
                "summary": "Collaboration aims to scale infrastructure.",
                "published_at": (now - dt.timedelta(minutes=70)).isoformat().replace("+00:00", "Z"),
                "source_domain": "techcrunch.com",
            },
        ][: limit]



def build_engine_from_env(*, debug: bool = False, provider_override: Optional[str] = None) -> AEIEngine:
    load_env()
    asknews_key = os.environ.get("ASKNEWS_API_KEY")
    asknews_id = os.environ.get("ASKNEWS_API_ID")
    asknews_base = os.environ.get("ASKNEWS_BASE_URL", ASKNEWS_DEFAULT_URL)
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHATGPT_API_KEY")
    openai_base = os.environ.get("OPENAI_API_BASE", OPENAI_DEFAULT_BASE)
    openai_model = os.environ.get("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)
    research_model = os.environ.get("OPENAI_RESEARCH_MODEL")
    research_max_items_env = os.environ.get("OPENAI_RESEARCH_MAX_ITEMS")
    synth_seed_model = os.environ.get("OPENAI_SYNTH_SEED_MODEL")
    synth_sifter_model = os.environ.get("OPENAI_SYNTH_SIFTER_MODEL")
    synth_analyst_model = os.environ.get("OPENAI_SYNTH_ANALYST_MODEL")
    synth_max_items_env = os.environ.get("OPENAI_SYNTH_MAX_ITEMS")
    keyword_model = os.environ.get("OPENAI_KEYWORD_MODEL")
    keyword_cache_env = os.environ.get("OPENAI_KEYWORD_CACHE_SIZE")
    config_model = os.environ.get("OPENAI_CONFIG_MODEL", openai_model)
    assets_hint_env = os.environ.get("AEI_CONFIG_ASSETS")
    provider_mode = (provider_override or os.environ.get("AEI_NEWS_PROVIDER", "asknews")).lower()

    assets_hint = [a.strip().upper() for a in assets_hint_env.split(",") if a.strip()] if assets_hint_env else list(DEFAULT_ASSET_ALIASES.keys())

    cfg = AEIConfig()
    max_calls_env = os.environ.get("AEI_MAX_API_CALLS_PER_RUN")
    cache_ttl_env = os.environ.get("AEI_ASKNEWS_CACHE_TTL_MIN")
    min_interval_env = os.environ.get("AEI_ASKNEWS_MIN_INTERVAL_SEC")
    if max_calls_env:
        try:
            cfg.max_api_calls_per_run = max(1, int(max_calls_env))
        except ValueError:
            _warn(f"invalid_env:AEI_MAX_API_CALLS_PER_RUN:{max_calls_env}")
    if cache_ttl_env:
        try:
            cfg.asknews_cache_ttl_min = max(1, int(cache_ttl_env))
        except ValueError:
            _warn(f"invalid_env:AEI_ASKNEWS_CACHE_TTL_MIN:{cache_ttl_env}")
    if min_interval_env:
        try:
            cfg.asknews_min_interval_sec = max(0.0, float(min_interval_env))
        except ValueError:
            _warn(f"invalid_env:AEI_ASKNEWS_MIN_INTERVAL_SEC:{min_interval_env}")

    asknews_client: Optional[Any] = None
    if provider_mode in ("asknews", "hybrid"):
        if asknews_key:
            asknews_client = AskNewsClient(
                api_key=asknews_key,
                api_id=asknews_id,
                base_url=asknews_base,
                min_interval_sec=cfg.asknews_min_interval_sec,
            )
        else:
            _warn("asknews_key_missing; switching to synthetic if available")
            if provider_mode == "asknews":
                provider_mode = "synthetic" if openai_key else "asknews"

    chat_sentiment: Optional[ChatGPTSentiment] = None
    if openai_key:
        try:
            chat_sentiment = ChatGPTSentiment(api_key=openai_key, api_base=openai_base, model=openai_model)
        except Exception as exc:
            _warn(f"chatgpt_init_failed:{exc}")

    research_assistant: Optional[ChatGPTResearcher] = None
    if openai_key:
        model = research_model or openai_model
        max_items = 20
        if research_max_items_env:
            try:
                max_items = max(1, int(research_max_items_env))
            except ValueError:
                _warn(f"invalid_env:OPENAI_RESEARCH_MAX_ITEMS:{research_max_items_env}")
        try:
            research_assistant = ChatGPTResearcher(
                api_key=openai_key,
                api_base=openai_base,
                model=model,
                max_items=max_items,
            )
        except Exception as exc:
            _warn(f"chatgpt_research_init_failed:{exc}")

    config_payload: Optional[Dict[str, Any]] = None
    event_rules: Optional[List[Dict[str, Any]]] = None
    if openai_key:
        try:
            config_provider = LLMConfigProvider(
                api_key=openai_key,
                api_base=openai_base,
                model=config_model,
                assets_hint=assets_hint,
            )
            config_payload = config_provider.fetch()
        except Exception as exc:
            _warn(f"config_oracle_init_failed:{exc}")
            config_payload = None

    sentiment_positive = None
    sentiment_negative = None
    if config_payload:
        aliases = config_payload.get("asset_aliases")
        if isinstance(aliases, dict):
            cfg.asset_aliases = {
                sym.upper(): {
                    "tickers": [str(x).lower() for x in data.get("tickers", []) if isinstance(x, str)],
                    "chain": [str(x).lower() for x in data.get("chain", []) if isinstance(x, str)],
                    "ecosystem": [str(x).lower() for x in data.get("ecosystem", []) if isinstance(x, str)],
                }
                for sym, data in aliases.items()
                if isinstance(data, dict)
            }
        context_words = config_payload.get("crypto_context")
        if isinstance(context_words, list) and context_words:
            cfg.crypto_context = [str(x).lower() for x in context_words if isinstance(x, str)]
        expansions = config_payload.get("news_expansions")
        if isinstance(expansions, dict):
            cfg.news_expansions = {
                sym.upper(): [str(x).lower() for x in words if isinstance(x, str)]
                for sym, words in expansions.items()
            }
        priors = config_payload.get("event_priors")
        if isinstance(priors, dict):
            cleaned_priors: Dict[str, float] = {}
            for label, value in priors.items():
                try:
                    cleaned_priors[str(label).lower()] = float(value)
                except Exception:
                    continue
            cfg.event_priors = cleaned_priors
        sources = config_payload.get("source_quality")
        if isinstance(sources, dict):
            cleaned_sources: Dict[str, float] = {}
            for domain, value in sources.items():
                try:
                    cleaned_sources[str(domain).lower()] = float(value)
                except Exception:
                    continue
            cfg.source_quality = cleaned_sources
        ops_guidance = config_payload.get("ops_guidance")
        if isinstance(ops_guidance, dict):
            cfg.ops_guidance = ops_guidance
        sentiment_bumps_payload = config_payload.get("sentiment_bumps")
        if isinstance(sentiment_bumps_payload, dict):
            cfg.sentiment_bumps = {
                str(label).lower(): float(value)
                for label, value in sentiment_bumps_payload.items()
                if isinstance(value, (int, float, str))
            }
        event_rules = config_payload.get("event_rules")
        sentiment_lexicon = config_payload.get("sentiment_lexicon")
        if isinstance(sentiment_lexicon, dict):
            positive = sentiment_lexicon.get("positive")
            negative = sentiment_lexicon.get("negative")
            if isinstance(positive, list) and positive:
                sentiment_positive = [str(x).lower() for x in positive if isinstance(x, str)]
            if isinstance(negative, list) and negative:
                sentiment_negative = [str(x).lower() for x in negative if isinstance(x, str)]
        global_query = config_payload.get("global_query")
        if isinstance(global_query, str) and global_query.strip():
            cfg.global_query = global_query.strip()

    if not cfg.event_priors:
        cfg.event_priors = dict(DEFAULT_EVENT_PRIORS)
    else:
        cfg.event_priors = {str(k).lower(): float(v) for k, v in cfg.event_priors.items()}
    if not cfg.source_quality:
        cfg.source_quality = dict(DEFAULT_SOURCE_QUALITY)
    else:
        cfg.source_quality = {str(k).lower(): float(v) for k, v in cfg.source_quality.items()}
    if not cfg.asset_aliases:
        cfg.asset_aliases = copy.deepcopy(DEFAULT_ASSET_ALIASES)
    else:
        cfg.asset_aliases = {
            sym.upper(): {
                "tickers": [str(x).lower() for x in groups.get("tickers", [])],
                "chain": [str(x).lower() for x in groups.get("chain", [])],
                "ecosystem": [str(x).lower() for x in groups.get("ecosystem", [])],
            }
            for sym, groups in cfg.asset_aliases.items()
        }
    if not cfg.crypto_context:
        cfg.crypto_context = list(DEFAULT_CRYPTO_CONTEXT)
    else:
        cfg.crypto_context = [str(x).lower() for x in cfg.crypto_context]
    if not cfg.news_expansions:
        cfg.news_expansions = copy.deepcopy(DEFAULT_NEWS_EXPANSIONS)
    else:
        cfg.news_expansions = {sym.upper(): [str(x).lower() for x in words] for sym, words in cfg.news_expansions.items()}
    cfg.ops_guidance = cfg.ops_guidance or copy.deepcopy(DEFAULT_OPS_GUIDANCE)
    cfg.sentiment_bumps = cfg.sentiment_bumps or dict(DEFAULT_SENTIMENT_BUMPS)
    cfg.global_query = cfg.global_query or DEFAULT_GLOBAL_QUERY

    keyword_oracle: Optional[KeywordOracle] = None
    if openai_key:
        km = keyword_model or openai_model
        cache_size = 64
        if keyword_cache_env:
            try:
                cache_size = max(4, int(keyword_cache_env))
            except ValueError:
                _warn(f"invalid_env:OPENAI_KEYWORD_CACHE_SIZE:{keyword_cache_env}")
        try:
            keyword_oracle = KeywordOracle(
                api_key=openai_key,
                api_base=openai_base,
                model=km,
                cache_size=cache_size,
            )
        except Exception as exc:
            _warn(f"keyword_oracle_init_failed:{exc}")

    synthetic_news: Optional[OpenAISyntheticNews] = None
    if provider_mode in ("synthetic", "hybrid") and openai_key:
        synth_seed = synth_seed_model or research_model or openai_model
        synth_sifter = synth_sifter_model or openai_model
        synth_analyst = synth_analyst_model or openai_model
        synth_max_items = 12
        if synth_max_items_env:
            try:
                synth_max_items = max(1, int(synth_max_items_env))
            except ValueError:
                _warn(f"invalid_env:OPENAI_SYNTH_MAX_ITEMS:{synth_max_items_env}")
        try:
            synthetic_news = OpenAISyntheticNews(
                api_key=openai_key,
                api_base=openai_base,
                seed_model=synth_seed,
                sifter_model=synth_sifter,
                analyst_model=synth_analyst,
                max_items=synth_max_items,
            )
        except Exception as exc:
            _warn(f"openai_synthetic_init_failed:{exc}")
            if provider_mode == "synthetic":
                provider_mode = "asknews"

    if provider_mode == "synthetic" and synthetic_news is None:
        _warn("synthetic provider requested but not available; reverting to asknews")
        provider_mode = "asknews"
    if provider_mode == "asknews" and asknews_client is None:
        asknews_client = _DummyAskNewsClient()
        _warn("using_dummy_client_no_real_asknews_results")

    event_classifier_instance = RuleEventClassifier(event_rules)
    sentiment_model_instance = SimpleSentiment(positive=sentiment_positive, negative=sentiment_negative)

    engine = AEIEngine(
        asknews_client=asknews_client,
        config=cfg,
        event_classifier=event_classifier_instance,
        sentiment_model=sentiment_model_instance,
        chat_sentiment=chat_sentiment,
        research_assistant=research_assistant,
        synthetic_news=synthetic_news,
        keyword_oracle=keyword_oracle,
        provider_mode=provider_mode,
        debug=debug,
    )
    return engine


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description="Run AskNews Event Impact (AEI) scoring")
    parser.add_argument("--asset", required=False, help="Single asset symbol or slug")
    parser.add_argument("--assets", nargs="+", default=None, help="(Deprecated) multiple assets")
    parser.add_argument("--window-min", type=int, default=90, help="News lookback window in minutes")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum impact score to surface")
    parser.add_argument("--max-items", type=int, default=None, help="Max AskNews items per query")
    parser.add_argument("--market-ref", type=str, default=None, help="JSON dict with per-asset normalization vols")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--debug", action="store_true", help="Emit debug logs to stderr")
    parser.add_argument(
        "--provider",
        choices=["asknews", "synthetic", "hybrid"],
        default=None,
        help="Override news provider selection (default comes from AEI_NEWS_PROVIDER)",
    )

    args = parser.parse_args(argv)

    if args.market_ref:
        try:
            market_ref = json.loads(args.market_ref)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --market-ref JSON: {exc}")
    else:
        market_ref = None

    if args.asset and args.assets:
        raise SystemExit("Specify either --asset or --assets, not both")
    assets = []
    if args.asset:
        assets = [args.asset]
    elif args.assets:
        assets = args.assets
    else:
        raise SystemExit("Provide --asset to score a single symbol")
    if len(assets) != 1:
        raise SystemExit("This tool accepts exactly one asset per run")
    engine = build_engine_from_env(debug=args.debug, provider_override=args.provider)
    if engine.provider_mode == "asknews" and isinstance(engine.client, _DummyAskNewsClient):
        _warn("using_dummy_client_no_real_asknews_results")

    results = engine.run(
        assets=assets,
        window_min=args.window_min,
        min_score=args.min_score,
        max_items=args.max_items,
        market_ref=market_ref,
    )

    dump = json.dumps(results, indent=2 if args.pretty else None)
    print(dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
