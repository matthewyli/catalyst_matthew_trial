from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Article:
    url: str
    title: str
    summary: str
    published_at: dt.datetime
    source_domain: str
    raw: Dict[str, Any]

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.summary}".strip()


@dataclass
class ResearchDecision:
    articles: List[Article]
    summary: Optional[str]
    notes: Dict[str, str] = field(default_factory=dict)
