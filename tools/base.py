from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot complete its task."""


@dataclass
class ToolContext:
    """Shared context the pipeline passes to each tool invocation."""

    asset: Optional[str]
    assets: tuple[str, ...]
    detected_keywords: tuple[str, ...]
    metadata: Dict[str, Any]
    usage_counts: Mapping[str, int]
    weights: Mapping[str, float]


@dataclass
class ToolResult:
    """Normalized result emitted by tools in the pipeline."""

    name: str
    weight: float
    summary: str
    payload: Dict[str, Any]


class BaseTool(abc.ABC):
    """Abstract base class for all pipeline tools."""

    name: str
    description: str
    keywords: frozenset[str]

    def __init__(self) -> None:
        if not getattr(self, "name", None):
            raise TypeError(f"{self.__class__.__name__} must define a 'name'")
        if not getattr(self, "description", None):
            raise TypeError(f"{self.__class__.__name__} must define a 'description'")
        if not getattr(self, "keywords", None):
            raise TypeError(f"{self.__class__.__name__} must define a 'keywords' frozenset")

    @abc.abstractmethod
    def execute(self, prompt: str, context: ToolContext) -> ToolResult:
        """Run the tool and return a normalized result."""

