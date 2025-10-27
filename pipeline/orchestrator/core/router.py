from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set


@dataclass
class RouteDecision:
    tools: List[str]
    keywords: Set[str]
    assets: List[str]


class KeywordRouter:
    """Keyword-driven router leveraging tool metadata declarations."""

    UPPER_TICKER = re.compile(r"\b[A-Z]{2,6}\b")

    def __init__(self, registry: Dict[str, object]) -> None:
        self.registry = registry
        self.keyword_map: Dict[str, Set[str]] = {}
        for name, obj in registry.items():
            meta = getattr(obj, "__TOOL_META__", {})
            keywords = set(k.lower() for k in meta.get("keywords", []))
            for kw in keywords:
                self.keyword_map.setdefault(kw, set()).add(name)

    def route(self, prompt: str) -> RouteDecision:
        lowered = prompt.lower()
        hit_keywords: Set[str] = set()
        tool_hits: Set[str] = set()
        for keyword, tools in self.keyword_map.items():
            if keyword in lowered:
                hit_keywords.add(keyword)
                tool_hits.update(tools)
        assets = self._detect_assets(prompt)
        return RouteDecision(tools=sorted(tool_hits), keywords=hit_keywords, assets=assets)

    def _detect_assets(self, prompt: str) -> List[str]:
        seen: Set[str] = set()
        assets: List[str] = []
        for match in self.UPPER_TICKER.findall(prompt):
            if match not in seen:
                assets.append(match)
                seen.add(match)
        return assets
