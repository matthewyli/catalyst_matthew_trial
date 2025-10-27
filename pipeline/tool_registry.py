from __future__ import annotations

from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

from tools.base import BaseTool


class ToolRegistry:
    """Maintains the set of tools available to the pipeline."""

    def __init__(self) -> None:
        self._registry: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._registry[tool.name] = tool

    def bulk_register(self, tools: Iterable[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._registry.get(name)

    def __contains__(self, name: object) -> bool:
        return bool(name in self._registry)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry.keys())

    def names(self) -> set[str]:
        return set(self._registry.keys())
