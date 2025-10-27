from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set


ASSET_PATTERN = re.compile(r"\b(?:[A-Z]{2,6}|[A-Z]+USDT)\b")
TIMEFRAME_PATTERN = re.compile(r"\b(\d+)\s*(min(?:ute)?s?|hour[s]?|day[s]?|week[s]?|month[s]?)\b", re.IGNORECASE)
INDICATOR_TERMS = [
    "ema",
    "sma",
    "rsi",
    "macd",
    "bollinger",
    "volatility",
    "momentum",
    "trend",
    "tvl",
    "defi",
    "sentiment",
    "backtest",
]
GOAL_TERMS = [
    "trend-follow",
    "mean reversion",
    "arbitrage",
    "hedge",
    "scalp",
    "swing",
    "capture breakout",
    "reduce risk",
    "maximize sharpe",
    "alpha",
    "carry",
    "liquidity mining",
]
FORCE_REFRESH_TERMS = [
    "force refresh",
    "refresh data",
    "clear cache",
    "no cache",
]


@dataclass
class ParsedPrompt:
    assets: List[str] = field(default_factory=list)
    primary_asset: Optional[str] = None
    timeframe_minutes: Optional[int] = None
    indicators: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    force_refresh: bool = False
    raw: str = ""


def parse_prompt(prompt: str, *, default_asset: str = "BTC", default_timeframe: int = 240) -> ParsedPrompt:
    raw = prompt or ""
    lowered = raw.lower()

    assets = _extract_assets(raw)
    indicators = _extract_terms(lowered, INDICATOR_TERMS)
    goals = _extract_terms(lowered, GOAL_TERMS)
    timeframe = _extract_timeframe(lowered) or default_timeframe

    indicator_set = {term.lower() for term in indicators}
    goal_set = {term.lower() for term in goals}
    assets = [a for a in assets if a.lower() not in indicator_set and a.lower() not in goal_set]

    if not assets:
        assets = [default_asset.upper()]

    primary_asset = assets[0] if assets else None
    force_refresh = any(term in lowered for term in FORCE_REFRESH_TERMS)

    return ParsedPrompt(
        assets=assets,
        primary_asset=primary_asset,
        timeframe_minutes=timeframe,
        indicators=indicators,
        goals=goals,
        force_refresh=force_refresh,
        raw=prompt,
    )


def _extract_assets(text: str) -> List[str]:
    matches = [m.group(0) for m in ASSET_PATTERN.finditer(text)]
    deduped: List[str] = []
    seen: Set[str] = set()
    for token in matches:
        token = token.upper()
        if token not in seen:
            seen.add(token)
            if token.endswith("USDT"):
                token = token[:-4]
            deduped.append(token)
    return deduped


def _extract_timeframe(text: str) -> Optional[int]:
    match = TIMEFRAME_PATTERN.search(text)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("min"):
        return value
    if unit.startswith("hour"):
        return value * 60
    if unit.startswith("day"):
        return value * 60 * 24
    if unit.startswith("week"):
        return value * 60 * 24 * 7
    if unit.startswith("month"):
        return value * 60 * 24 * 30
    return value


def _extract_terms(text: str, terms: Sequence[str]) -> List[str]:
    found: List[str] = []
    for term in terms:
        if term in text:
            found.append(term)
    return found
