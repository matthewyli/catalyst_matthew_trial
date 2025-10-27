from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Pattern, Sequence, Tuple

from .keyword_config import ToolKeywordConfig
from .keyword_llm import LLMRouter


@dataclass(frozen=True)
class DetectionResult:
    tool_names: List[str]
    assets: List[str]
    keywords: List[str]
    metadata: Dict[str, Any]
    scores: Dict[str, float]
    matched_terms: Dict[str, List[str]]
    matched_patterns: Dict[str, List[str]]
    llm_recommendations: List[Tuple[str, float]]


@dataclass(frozen=True)
class _PreparedSpec:
    terms_raw: Tuple[str, ...]
    terms_lower: Tuple[str, ...]
    patterns: Tuple[Pattern[str], ...]
    literal_weight: float
    pattern_weight: float
    bias: float


class KeywordDetector:
    """Keyword-driven tool selector with scoring."""

    UPPER_TICKER_RE = re.compile(r"\b[A-Z]{2,6}\b")
    MOBULA_KEY_PATTERN = re.compile(r"mobula_key\s*=\s*([a-zA-Z0-9_\-]+)", re.IGNORECASE)
    MINUTES_PATTERN = re.compile(r"(\d+)\s*(?:minute|min)\b", re.IGNORECASE)

    def __init__(
        self,
        *,
        tool_keywords: Mapping[str, ToolKeywordConfig],
        asset_aliases: Mapping[str, str],
        blockchain_aliases: Optional[Mapping[str, str]] = None,
        timeframe_keywords: Optional[Mapping[str, int]] = None,
        llm_weight: float = 1.0,
        learned_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
        weights_path: Optional[Path] = None,
    ) -> None:
        self.learned_weights = self._resolve_learned_weights(learned_weights, weights_path)
        self.tool_specs: Dict[str, _PreparedSpec] = {
            tool: _PreparedSpec(
                terms_raw=tuple(cfg.terms),
                terms_lower=tuple(term.lower() for term in cfg.terms),
                patterns=tuple(re.compile(pat, re.IGNORECASE) for pat in cfg.patterns),
                literal_weight=self._override_weight(tool, "literal_weight", cfg.literal_weight),
                pattern_weight=self._override_weight(tool, "pattern_weight", cfg.pattern_weight),
                bias=self._override_weight(tool, "bias", 0.0),
            )
            for tool, cfg in tool_keywords.items()
        }
        self.asset_aliases = {k.lower(): v for k, v in asset_aliases.items()}
        self.blockchain_aliases = {k.lower(): v for k, v in (blockchain_aliases or {}).items()}
        self.timeframe_keywords = {k.lower(): v for k, v in (timeframe_keywords or {}).items()}
        self.tool_descriptions = {tool: (tool_keywords[tool].description or tool) for tool in tool_keywords}
        self.llm_router = LLMRouter()
        self.llm_weight = llm_weight

    def detect(self, prompt: str, *, explicit_asset: Optional[str] = None) -> DetectionResult:
        lowered = prompt.lower()
        detected_keywords: Dict[str, None] = {}
        scores: Dict[str, float] = {}
        matched_terms: Dict[str, List[str]] = {}
        matched_patterns: Dict[str, List[str]] = {}

        for tool, spec in self.tool_specs.items():
            score, terms, patterns = self._score_tool(prompt, lowered, spec)
            scores[tool] = round(score, 6)
            if terms:
                matched_terms[tool] = terms
                detected_keywords.update({t: None for t in terms})
            if patterns:
                matched_patterns[tool] = patterns
                detected_keywords.update({p: None for p in patterns})

        llm_recommendations: List[Tuple[str, float]] = []
        if self.llm_router and getattr(self.llm_router, "enabled", False):
            try:
                llm_recommendations = self.llm_router.recommend(prompt, self.tool_descriptions)
            except Exception:  # pragma: no cover - defensive
                llm_recommendations = []
            for name, confidence in llm_recommendations:
                scores[name] = round(scores.get(name, 0.0) + self.llm_weight * confidence, 6)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ordered_tools = [tool for tool, score in ranked if score > 0.0]

        assets = self._detect_assets(prompt, lowered, explicit_asset)
        metadata: Dict[str, Any] = {
            "keyword_scores": scores,
            "llm_recommendations": llm_recommendations,
            "llm_weight": self.llm_weight,
        }

        timeframe = self._infer_timeframe(lowered)
        if timeframe is not None:
            metadata["window_min"] = timeframe

        blockchain = self._infer_blockchain(lowered, assets)
        if blockchain:
            metadata["blockchain"] = blockchain

        mobula_key = self._infer_mobula_key(prompt)
        if mobula_key:
            metadata["mobula_api_key"] = mobula_key

        number_window = self._extract_minutes(lowered)
        if number_window is not None:
            metadata["window_min"] = number_window

        keyword_list = list(detected_keywords.keys())

        return DetectionResult(
            tool_names=ordered_tools,
            assets=assets,
            keywords=keyword_list,
            metadata=metadata,
            scores=scores,
            matched_terms=matched_terms,
            matched_patterns=matched_patterns,
            llm_recommendations=llm_recommendations,
        )

    def _score_tool(
        self,
        prompt: str,
        lowered: str,
        spec: _PreparedSpec,
    ) -> Tuple[float, List[str], List[str]]:
        score = 0.0
        literal_hits: List[str] = []
        pattern_hits: List[str] = []

        for raw_term, term in zip(spec.terms_raw, spec.terms_lower):
            if term and term in lowered:
                literal_hits.append(raw_term)
                score += spec.literal_weight * self._term_bonus(raw_term)

        for pattern in spec.patterns:
            matches = list(pattern.finditer(prompt))
            if matches:
                pattern_hits.append(pattern.pattern)
                score += spec.pattern_weight * self._pattern_bonus(matches)

        return score + spec.bias, literal_hits, pattern_hits

    @staticmethod
    def _term_bonus(term: str) -> float:
        words = len(term.split())
        length_bonus = min(len(term), 40) / 40.0
        return 1.0 + 0.15 * words + 0.2 * length_bonus

    @staticmethod
    def _pattern_bonus(matches: Sequence[re.Match[str]]) -> float:
        match_count = len(matches)
        longest = max(len(m.group(0)) for m in matches)
        length_bonus = min(longest, 60) / 60.0
        return match_count * (1.0 + 0.3 * length_bonus)

    def _detect_assets(self, prompt: str, lowered: str, explicit_asset: Optional[str]) -> List[str]:
        assets: List[str] = []
        seen: Dict[str, None] = {}

        if explicit_asset:
            sym = explicit_asset.upper()
            assets.append(sym)
            seen[sym] = None

        for alias, symbol in self.asset_aliases.items():
            if alias in lowered:
                sym = symbol.upper()
                if sym not in seen:
                    assets.append(sym)
                    seen[sym] = None

        for match in self.UPPER_TICKER_RE.findall(prompt):
            if match not in seen:
                assets.append(match)
                seen[match] = None

        return assets

    def _infer_timeframe(self, lowered: str) -> Optional[int]:
        for keyword, minutes in self.timeframe_keywords.items():
            if keyword in lowered:
                return minutes
        return None

    def _infer_blockchain(self, lowered: str, assets: Iterable[str]) -> Optional[str]:
        for alias, chain in self.blockchain_aliases.items():
            if alias in lowered:
                return chain

        chain_map = {
            "ETH": "ethereum",
            "ARB": "arbitrum",
            "OP": "optimism",
            "SOL": "solana",
            "MATIC": "polygon",
            "AVAX": "avalanche",
            "BNB": "bsc",
        }
        for asset in assets:
            if asset in chain_map:
                return chain_map[asset]
        return None

    def _infer_mobula_key(self, prompt: str) -> Optional[str]:
        match = self.MOBULA_KEY_PATTERN.search(prompt)
        if match:
            return match.group(1)
        return None

    def _extract_minutes(self, lowered: str) -> Optional[int]:
        match = self.MINUTES_PATTERN.search(lowered)
        if match:
            return int(match.group(1))
        return None

    # ------------------------------------------------------------------ learned weights helpers
    @staticmethod
    def _default_weights_path() -> Path:
        return Path(__file__).with_name("keyword_weights.json")

    def _resolve_learned_weights(
        self,
        learned: Optional[Mapping[str, Mapping[str, float]]],
        weights_path: Optional[Path],
    ) -> Dict[str, Dict[str, float]]:
        if learned is not None:
            return {name: dict(values) for name, values in learned.items()}
        path = weights_path or Path(os.getenv("PIPELINE_KEYWORD_WEIGHTS_PATH") or self._default_weights_path())
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        weights = payload.get("weights") if isinstance(payload, dict) else payload
        if not isinstance(weights, dict):
            return {}
        normalized: Dict[str, Dict[str, float]] = {}
        for tool, values in weights.items():
            if not isinstance(values, dict):
                continue
            normalized[str(tool)] = {
                key: float(values.get(key, 0.0))
                for key in ("literal_weight", "pattern_weight", "bias")
                if key in values
            }
        return normalized

    def _override_weight(self, tool: str, key: str, default: float) -> float:
        tool_weights = self.learned_weights.get(tool)
        if not tool_weights:
            return default
        value = tool_weights.get(key)
        if value is None:
            return default
        if key in {"literal_weight", "pattern_weight"} and value <= 0:
            return default
        return float(value)
