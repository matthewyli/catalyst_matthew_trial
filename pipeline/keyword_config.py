from __future__ import annotations

"""Keyword configuration for tool routing."""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class ToolKeywordConfig:
    terms: Sequence[str]
    patterns: Sequence[str]
    literal_weight: float = 1.0
    pattern_weight: float = 1.5
    description: str = ""


TOOL_KEYWORDS: Mapping[str, ToolKeywordConfig] = {
    "textql_primer": ToolKeywordConfig(
        terms=[
            "textql",
            "research plan",
            "search plan",
            "web context",
            "exa search",
            "macro context",
            "market brief",
            "scout",
            "discovery",
        ],
        patterns=[
            r"\bdeep\s+research\b",
            r"\b(search|exa)\s+plan\b",
            r"\b(world|macro)\s+context\b",
            r"\bweb\s+recon\b",
        ],
        literal_weight=1.2,
        pattern_weight=1.7,
        description="Prime the pipeline with a TextQL-powered research scaffold (web + on-chain hypotheses).",
    ),
    "textql_runtime": ToolKeywordConfig(
        terms=[
            "follow up",
            "deep dive",
            "runtime search",
            "extra context",
            "check fact",
            "validate",
            "refresh data",
        ],
        patterns=[
            r"\bruntime\s+(?:search|textql)\b",
            r"\bfollow[-\s]?up\b",
            r"\bdouble\s+check\b",
        ],
        literal_weight=1.1,
        pattern_weight=1.5,
        description="Launches TextQL follow-up research during later phases to validate or refresh signals.",
    ),
    "asknews_impact": ToolKeywordConfig(
        terms=[
            "news",
            "headline",
            "breaking",
            "press release",
            "sentiment",
            "narrative",
            "asknews",
            "event risk",
            "regulatory update",
            "research summary",
        ],
        patterns=[
            r"\bnews(?:flow|wire)?\b",
            r"\b(?:headline|press)\s+(?:scan|run)\b",
            r"\bsentiment\s+(?:pull|score)\b",
            r"\bopenai\s+research\b",
            r"\bevent[-\s]?impact\b",
        ],
        literal_weight=1.0,
        pattern_weight=1.7,
        description="Analyze recent crypto news with AskNews and OpenAI sentiment models to score event impact.",
    ),
    "tvl_growth": ToolKeywordConfig(
        terms=[
            "tvl",
            "defi",
            "liquidity",
            "inflow",
            "outflow",
            "staking",
            "defillama",
            "tvl filter",
            "tvl momentum",
            "synchronized growth",
        ],
        patterns=[
            r"\b(defi|chain)\s+tvl\b",
            r"\bliquidity\s+(?:spike|surge|drain)\b",
            r"\binflow[s]?\s+filter\b",
            r"\bsync(?:hronized)?\s+growth\b",
        ],
        literal_weight=1.1,
        pattern_weight=1.6,
        description="Evaluate DeFiLlama TVL trends to detect synchronized liquidity growth and momentum shifts.",
    ),
    "volatility_percentile": ToolKeywordConfig(
        terms=[
            "vol",
            "volatility",
            "sigma",
            "percentile",
            "risk regime",
            "variance",
            "vol bucket",
            "vol filter",
            "vol targeting",
        ],
        patterns=[
            r"\bvol(?:atility)?\s+(?:percentile|rank)\b",
            r"\bsigma\s*\d*\b",
            r"\b(regime|risk)\s+shift\b",
            r"\bvol\s+(?:z\s*score|z-score)\b",
        ],
        literal_weight=1.0,
        pattern_weight=1.7,
        description="Pull Mobula price history to compute realized volatility percentiles and regime confidence.",
    ),
    "execution_adapter": ToolKeywordConfig(
        terms=[
            "execute",
            "execution",
            "order",
            "trade",
            "fill",
            "live trade",
            "paper trade",
            "dispatch",
        ],
        patterns=[
            r"\bsend\s+(?:the\s+)?orders?\b",
            r"\b(route|dispatch)\s+trades?\b",
            r"\border\s+(?:routing|execution)\b",
        ],
        literal_weight=1.0,
        pattern_weight=1.5,
        description="Route generated orders to paper or live execution backends with risk and policy safeguards.",
    ),
}


ASSET_ALIASES: Dict[str, str] = {
    "btc": "BTC",
    "bitcoin": "BTC",
    "eth": "ETH",
    "ethereum": "ETH",
    "sol": "SOL",
    "solana": "SOL",
    "arb": "ARB",
    "arbitrum": "ARB",
    "op": "OP",
    "optimism": "OP",
    "matic": "MATIC",
    "polygon": "MATIC",
    "avax": "AVAX",
    "avalanche": "AVAX",
    "bnb": "BNB",
    "binance": "BNB",
    "base": "BASE",
}


BLOCKCHAIN_ALIASES: Dict[str, str] = {
    "ethereum": "ethereum",
    "solana": "solana",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "polygon": "polygon",
    "avalanche": "avalanche",
    "binance": "bsc",
    "bsc": "bsc",
    "base": "base",
}


TIMEFRAME_KEYWORDS: Dict[str, int] = {
    "scalp": 15,
    "intraday": 60,
    "short term": 120,
    "swing": 240,
    "medium term": 360,
    "long term": 1440,
}
